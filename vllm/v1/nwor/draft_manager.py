"""Draft Commit Manager for NWOR (greenfield implementation)."""

from bisect import bisect_left
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class DraftEntry:
    """Draft KV entry for copy-on-write restoration."""
    # Validation
    num_tokens: int

    # Cache layout metadata (for restore_rejected_drafts kernel)
    block_size: int
    block_stride: int
    page_stride: int
    head_stride: int
    layout_id: int  # 0=flash, 1=paged

    # Quantization
    scale_is_per_token: bool
    kv_cache_dtype: str

    # Keep tensors alive
    _key_ref: torch.Tensor
    _value_ref: torch.Tensor
    _slot_ref: torch.Tensor  # Live buffer reference (convert to int32 on-demand)
    _key_cache_ref: torch.Tensor
    _value_cache_ref: torch.Tensor
    _k_scale_ref: Optional[torch.Tensor]
    _v_scale_ref: Optional[torch.Tensor]

    # Copy-on-write logging (for rejected draft restoration)
    log_key_buffer: Optional[torch.Tensor] = None
    log_value_buffer: Optional[torch.Tensor] = None
    log_k_scale_buffer: Optional[torch.Tensor] = None  # For per-token FP8 scales
    log_v_scale_buffer: Optional[torch.Tensor] = None
    chunk_global_indices: Optional[torch.Tensor] = None  # All draft indices for this chunk
    chunk_local_positions: Optional[torch.Tensor] = None  # Chunk-local offsets for those indices
    chunk_logged_indices: Optional[torch.Tensor] = None  # Positions within chunk_global_indices that were logged
    chunk_logged_local: Optional[torch.Tensor] = None  # Chunk-local offsets for logged entries


CacheKey = Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]


@dataclass
class DraftCacheEntry:
    """Cached metadata for CUDA graph replays."""
    positions: List[int]
    drafts: List[DraftEntry]
    num_layers: int


@dataclass(frozen=True)
class ChunkSlice:
    """Chunk-local mapping for draft indices."""
    global_indices: torch.Tensor  # Indices into _draft_positions for drafts in this chunk
    local_positions: torch.Tensor  # Chunk-local offsets corresponding to global_indices


class DraftCommitManager:
    """Manages draft buffers for NWOR with minimal overhead."""

    def __init__(self):
        import os
        nwor_mode = os.getenv("VLLM_NWOR_MODE", "").lower()
        self._nwor_enabled = (nwor_mode != "off")  # Persistent config flag
        self.enabled = False  # Per-window active flag
        self._drafts: List[DraftEntry] = []
        self._logged_failure = False
        # Position tracking for mapping draft-only mask to full staged tensor
        self._logits_indices = None  # All token positions (targets + drafts)
        self._target_logits_indices = None  # Indices into logits_indices pointing to targets
        self._draft_positions = []  # Absolute positions of draft tokens in staged tensor
        # Metrics tracking (per-commit)
        self._emit_metrics = os.getenv("VLLM_NWOR_EMIT_METRICS", "0") == "1"
        self._num_draft_tokens = 0
        self._num_draft_accepted = 0
        self._num_draft_rejected = 0
        # CUDA graph caching
        self._cache: Dict[CacheKey, DraftCacheEntry] = {}
        self._cache_key: Optional[CacheKey] = None
        self._capturing = False
        self._fallback_cached_drafts: List[DraftEntry] = []  # Most recent capture's DraftEntries
        # Copy-on-write log buffers (allocated lazily per layer, persistent across replays)
        self._log_key_buffers: Dict[int, torch.Tensor] = {}
        self._log_value_buffers: Dict[int, torch.Tensor] = {}
        self._log_k_scale_buffers: Dict[int, torch.Tensor] = {}
        self._log_v_scale_buffers: Dict[int, torch.Tensor] = {}
        self._slot_indices_buffers: Dict[int, torch.Tensor] = {}  # Persistent slot indices for CUDA graph
        self._chunk_slices: Dict[Tuple[int, int, int], ChunkSlice] = {}
        # Reusable empty tensor to avoid repeated allocations
        self._empty_tensor: Optional[torch.Tensor] = None
        if self._nwor_enabled:
            logger.info(f"NWOR enabled (VLLM_NWOR_MODE={nwor_mode})")
            if self._emit_metrics:
                logger.info("NWOR metrics emission enabled (VLLM_NWOR_EMIT_METRICS=1)")

    def begin(self, spec_decode_metadata) -> bool:
        """Begin new spec decode window.

        Args:
            spec_decode_metadata: SpecDecodeMetadata containing token positions

        Returns:
            True if NWOR is enabled for this window
        """
        self._drafts.clear()
        self._draft_positions.clear()
        self._cache_key = None
        self._capturing = False

        if spec_decode_metadata is None:
            self.enabled = False
            return False

        total_draft_tokens = sum(spec_decode_metadata.num_draft_tokens)

        # Only activate if NWOR enabled AND we have draft tokens
        if not self._nwor_enabled or total_draft_tokens <= 0:
            self.enabled = False
            return False

        # Extract position metadata for mapping mask to staged tokens
        self._logits_indices = spec_decode_metadata.logits_indices
        self._target_logits_indices = spec_decode_metadata.target_logits_indices
        cache_key = spec_decode_metadata.cache_key
        if cache_key is None:
            logits_indices_cpu = self._logits_indices.cpu()
            target_indices_cpu = self._target_logits_indices.cpu()
            num_draft_tokens_tuple = tuple(spec_decode_metadata.num_draft_tokens)
            cache_key = (
                tuple(int(x) for x in logits_indices_cpu.tolist()),
                tuple(int(x) for x in target_indices_cpu.tolist()),
                num_draft_tokens_tuple,
            )
            logits_list = logits_indices_cpu.tolist()
        else:
            logits_list = list(cache_key[0])
        self._cache_key = cache_key

        cached = self._cache.get(cache_key)

        # Note: We used to copy cached DraftEntries here, but stage_layer() immediately
        # clears self._drafts on first call, making the copies pointless. Removed for performance.

        # ALWAYS recompute positions from live metadata (layout-specific)
        positions: List[int] = []
        token_cursor = 0
        for num_drafts in spec_decode_metadata.num_draft_tokens:
            assert token_cursor < len(logits_list), (
                f"token_cursor {token_cursor} exceeds logits length {len(logits_list)}"
            )
            for offset in range(num_drafts):
                idx = token_cursor + 1 + offset
                assert idx < len(logits_list), (
                    f"Draft position out of bounds: idx={idx}, logits_len={len(logits_list)}"
                )
                positions.append(int(logits_list[idx]))
            token_cursor += num_drafts + 1
        self._draft_positions.extend(positions)
        if __debug__:
            assert self._draft_positions == sorted(self._draft_positions), \
                "NWOR draft positions must be sorted"

        # Update or create cache entry with new positions
        if cached is None:
            self._cache[cache_key] = DraftCacheEntry(
                positions=list(positions),
                drafts=[],
                num_layers=0,
            )
        else:
            # Update positions for this key
            cached.positions = list(positions)

        self.enabled = True
        return True

    def _get_empty_tensor(self, device: torch.device) -> torch.Tensor:
        """Get reusable empty tensor to avoid repeated allocations."""
        if self._empty_tensor is None or self._empty_tensor.device != device:
            self._empty_tensor = torch.empty(0, device=device)
        return self._empty_tensor

    def cancel(self):
        """Cancel without committing."""
        self._drafts.clear()
        self.enabled = False
        self._cache_key = None
        self._capturing = False
        self._chunk_slices.clear()

    def _get_chunk_slice(self, slot_mapping: torch.Tensor) -> ChunkSlice:
        """Return cached chunk metadata for the given slot mapping view."""
        assert slot_mapping.dim() == 1 and slot_mapping.stride(0) == 1, \
            "NWOR requires contiguous 1D slot_mapping views"
        device = slot_mapping.device
        device_index = device.index if device.type == "cuda" else -1
        chunk_start = int(slot_mapping.storage_offset())
        chunk_len = int(slot_mapping.shape[0])
        cache_key = (device_index, chunk_start, chunk_len)
        chunk_slice = self._chunk_slices.get(cache_key)
        if chunk_slice is not None:
            return chunk_slice

        positions = self._draft_positions

        if chunk_len == 0:
            empty = self._get_empty_tensor(device)
            global_indices = empty
            local_positions = empty
        else:
            start_idx = bisect_left(positions, chunk_start)
            end_idx = bisect_left(positions, chunk_start + chunk_len)
            if end_idx > start_idx:
                global_indices = torch.arange(start_idx, end_idx, dtype=torch.long, device=device)
                local_offsets = torch.tensor(
                    positions[start_idx:end_idx], dtype=torch.long, device=device
                ) - chunk_start
                local_positions = local_offsets
            else:
                empty = self._get_empty_tensor(device)
                global_indices = empty
                local_positions = empty

        chunk_slice = ChunkSlice(
            global_indices=global_indices,
            local_positions=local_positions,
        )
        self._chunk_slices[cache_key] = chunk_slice
        return chunk_slice

    def stage_layer(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: Optional[torch.Tensor],
        v_scale: Optional[torch.Tensor],
        kv_cache_dtype: str,
    ):
        """Record draft KV for one layer."""
        if not self.enabled:
            return

        if not self._capturing:
            # First layer observed for this window → fresh capture
            if self._drafts:
                self._drafts.clear()
            self._capturing = True

        # Check if we should skip NWOR logging (but still write to cache)
        skip_nwor_logging = False

        # Validate storage
        for t, n in [(key, "key"), (value, "value"), (key_cache, "key_cache"),
                     (value_cache, "value_cache"), (slot_mapping, "slot_mapping")]:
            try:
                _ = t.data_ptr()
            except RuntimeError:
                logger.error(f"NWOR: {n} has no storage")
                self.enabled = False
                return

        # No pre-conversion - store original and convert on-demand in kernel calls

        # Detect layout using num_heads from key tensor
        # Flash: [num_blocks, block_size, num_heads, head_size] - dim 2 is num_heads
        # Paged: [num_blocks, num_heads, block_size, head_size] - dim 1 is num_heads
        num_heads_from_key = key.shape[1]
        if key_cache.shape[2] == num_heads_from_key:
            # Flash layout
            layout_id = 0
            block_size = key_cache.shape[1]
        else:
            # Paged layout
            layout_id = 1
            block_size = key_cache.shape[2]

        # Detect per-token scale
        scale_is_per_token = k_scale is not None and k_scale.numel() > 1

        # Copy-on-write: Log existing cache data at draft slots before we overwrite
        layer_idx = len(self._drafts)  # Current layer index
        log_key_buffer = None
        log_value_buffer = None
        log_k_scale_buffer = None
        log_v_scale_buffer = None
        chunk_global_indices = None
        chunk_local_positions = None
        chunk_logged_indices = None
        chunk_logged_local = None

        if self._draft_positions:
            chunk_slice = self._get_chunk_slice(slot_mapping)
            chunk_global_indices = chunk_slice.global_indices
            chunk_local_positions = chunk_slice.local_positions

            if chunk_global_indices.numel() > 0 and not skip_nwor_logging:
                current_slots = slot_mapping[chunk_local_positions]
                valid_mask = current_slots >= 0
                logged_indices = valid_mask.nonzero(as_tuple=False).squeeze(1)
                num_logged = int(logged_indices.numel())

                max_size = 512  # Match CUDA graph allocation
                if num_logged > max_size:
                    logger.error(
                        f"NWOR: Too many draft tokens ({num_logged} > {max_size}), "
                        "disabling NWOR for this window to prevent buffer overflow"
                    )
                    skip_nwor_logging = True
                else:
                    if layer_idx not in self._log_key_buffers:
                        self._log_key_buffers[layer_idx] = torch.empty(
                            (max_size, key.shape[1], key.shape[2]),
                            dtype=key.dtype,
                            device=key.device,
                        )
                        self._log_value_buffers[layer_idx] = torch.empty(
                            (max_size, value.shape[1], value.shape[2]),
                            dtype=value.dtype,
                            device=value.device,
                        )
                        self._slot_indices_buffers[layer_idx] = torch.empty(
                            max_size, dtype=torch.int64, device=key.device
                        )
                        if scale_is_per_token:
                            assert k_scale is not None and v_scale is not None
                            self._log_k_scale_buffers[layer_idx] = torch.empty(
                                max_size, dtype=k_scale.dtype, device=k_scale.device
                            )
                            self._log_v_scale_buffers[layer_idx] = torch.empty(
                                max_size, dtype=v_scale.dtype, device=v_scale.device
                            )

                    if not skip_nwor_logging:
                        chunk_logged_local = chunk_local_positions[logged_indices]
                        chunk_logged_indices = logged_indices
                        logged_slots = current_slots[logged_indices].to(torch.int64)

                        slot_indices_buffer = self._slot_indices_buffers[layer_idx]
                        slot_indices_buffer[:num_logged] = logged_slots
                        logged_slots_buffer = slot_indices_buffer[:num_logged]

                        log_key_buffer = self._log_key_buffers[layer_idx][:num_logged]
                        log_value_buffer = self._log_value_buffers[layer_idx][:num_logged]

                        torch.ops._C_cache_ops.log_cache_slots(
                            key_cache,
                            value_cache,
                            logged_slots_buffer,
                            log_key_buffer,
                            log_value_buffer,
                            block_size,
                            key_cache.stride(0),
                            key_cache.stride(2) if layout_id == 1 else key_cache.stride(1),
                            key_cache.stride(1) if layout_id == 1 else key_cache.stride(2),
                            kv_cache_dtype,
                            k_scale if k_scale is not None else self._get_empty_tensor(key.device),
                            v_scale if v_scale is not None else self._get_empty_tensor(key.device),
                        )

                        if scale_is_per_token and layer_idx in self._log_k_scale_buffers:
                            assert k_scale is not None and v_scale is not None
                            log_k_scale_buffer = self._log_k_scale_buffers[layer_idx][:num_logged]
                            log_v_scale_buffer = self._log_v_scale_buffers[layer_idx][:num_logged]
                            log_k_scale_buffer[:num_logged] = k_scale[chunk_logged_local]
                            log_v_scale_buffer[:num_logged] = v_scale[chunk_logged_local]

        # ALWAYS write all tokens to real cache (attention will read this)
        # This must run even if NWOR logging is skipped (FP8, buffer overflow, etc.)
        # Note: kernel handles both int32 and int64 slot_mapping internally
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale if k_scale is not None else self._get_empty_tensor(key.device),
            v_scale if v_scale is not None else self._get_empty_tensor(key.device)
        )

        # If NWOR logging was skipped, disable NWOR and return
        # Cache has been written, so this is safe
        if skip_nwor_logging:
            self.enabled = False
            return

        if chunk_global_indices is None or chunk_global_indices.numel() == 0:
            # No drafts in this chunk – nothing to track for commit.
            return

        # Ensure chunk-local metadata defaults
        if chunk_logged_indices is None:
            chunk_logged_indices = self._get_empty_tensor(key.device)
        if chunk_logged_local is None:
            chunk_logged_local = self._get_empty_tensor(key.device)

        # Create DraftEntry for NWOR restore (only if logging succeeded)
        self._drafts.append(DraftEntry(
            num_tokens=key.shape[0],
            block_size=block_size,
            block_stride=key_cache.stride(0),
            page_stride=key_cache.stride(2) if layout_id == 1 else key_cache.stride(1),
            head_stride=key_cache.stride(1) if layout_id == 1 else key_cache.stride(2),
            layout_id=layout_id,
            scale_is_per_token=scale_is_per_token,
            kv_cache_dtype=kv_cache_dtype,
            _key_ref=key,
            _value_ref=value,
            _slot_ref=slot_mapping,  # Live buffer, convert to int32 on-demand
            _key_cache_ref=key_cache,
            _value_cache_ref=value_cache,
            _k_scale_ref=k_scale,
            _v_scale_ref=v_scale,
            # Copy-on-write log
            log_key_buffer=log_key_buffer,
            log_value_buffer=log_value_buffer,
            log_k_scale_buffer=log_k_scale_buffer,
            log_v_scale_buffer=log_v_scale_buffer,
            chunk_global_indices=chunk_global_indices,
            chunk_local_positions=chunk_local_positions,
            chunk_logged_indices=chunk_logged_indices,
            chunk_logged_local=chunk_logged_local,
        ))

    def commit(self, mask: torch.Tensor) -> int:
        """Commit accepted tokens using CUDA kernel.

        Args:
            mask: Boolean tensor [num_draft_tokens] indicating which drafts were accepted

        Returns:
            Number of tokens committed (including targets and accepted drafts)
        """
        if not self.enabled or not self._drafts:
            self.cancel()  # Clean up state
            return 0

        # No need for replay refresh - _slot_ref points to live buffer
        # CUDA graph replay automatically refills the buffer at the same address
        # Our tensor reference sees the new data automatically

        try:
            device = self._drafts[0]._key_ref.device
            num_tokens = self._drafts[0].num_tokens

            # Validate input mask covers all drafts
            assert mask.shape[0] == len(self._draft_positions), \
                f"mask size {mask.shape[0]} != num drafts {len(self._draft_positions)}"
            assert all(e.num_tokens == num_tokens for e in self._drafts)

            # Prepare draft mask (keep on GPU for zero-copy overhead)
            # Combine dtype and device conversion in single operation
            draft_mask = mask.to(device=device, dtype=torch.bool)

            if self._emit_metrics:
                mask_true = int(draft_mask.sum().item())
                sample_positions = list(self._draft_positions)
                sample_slots = []
                for entry in self._drafts:
                    sample_slots = entry._slot_ref.tolist()
                    break
                logger.debug(
                    "NWOR commit window: drafts=%d mask_true=%d sample_pos=%s sample_slots=%s",
                    len(self._draft_positions),
                    mask_true,
                    sample_positions,
                    sample_slots,
                )

            # Track draft-only metrics (only if metrics enabled)
            if self._emit_metrics:
                self._num_draft_tokens = len(self._draft_positions)
                self._num_draft_accepted = int(draft_mask.sum().item())
                self._num_draft_rejected = self._num_draft_tokens - self._num_draft_accepted

            # Copy-on-write: Accepted tokens are already in cache from reshape_and_cache_flash.
            # Only restore rejected drafts from log, using chunk-local metadata.
            for entry in self._drafts:
                if (
                    entry.chunk_global_indices is None
                    or entry.chunk_global_indices.numel() == 0
                    or entry.chunk_logged_indices is None
                    or entry.chunk_logged_indices.numel() == 0
                    or entry.log_key_buffer is None
                ):
                    continue

                chunk_mask = draft_mask[entry.chunk_global_indices]
                logged_mask = chunk_mask[entry.chunk_logged_indices]

                rejected_rows = (~logged_mask).nonzero(as_tuple=False).squeeze(1)
                if rejected_rows.numel() == 0:
                    continue

                current_slots = entry._slot_ref[entry.chunk_logged_local]
                rejected_slots_int32 = current_slots[rejected_rows].to(dtype=torch.int32)

                rejected_log_key = entry.log_key_buffer[rejected_rows]
                rejected_log_value = entry.log_value_buffer[rejected_rows]

                if entry.log_k_scale_buffer is not None:
                    rejected_k_scale = entry.log_k_scale_buffer[rejected_rows]
                    rejected_v_scale = entry.log_v_scale_buffer[rejected_rows]
                else:
                    empty = self._get_empty_tensor(device)
                    rejected_k_scale = empty
                    rejected_v_scale = empty

                torch.ops._C_cache_ops.restore_rejected_drafts(
                    rejected_log_key,
                    rejected_log_value,
                    entry._key_cache_ref,
                    entry._value_cache_ref,
                    rejected_slots_int32,
                    entry.block_size,
                    entry.block_stride,
                    entry.page_stride,
                    entry.head_stride,
                    entry.kv_cache_dtype,
                    rejected_k_scale,
                    rejected_v_scale,
                )

            # Optional sync (ISSUE #6: Make conditional for debugging)
            import os
            if os.getenv("VLLM_NWOR_DEBUG_SYNC", "0") == "1":
                torch.cuda.synchronize()

            # Log success once for verification
            if not hasattr(self, '_logged_success'):
                if self._emit_metrics:
                    logger.info(f"NWOR kernel succeeded: accepted {self._num_draft_accepted}/{self._num_draft_tokens} draft tokens across {len(self._drafts)} layers (per-pass snapshot)")
                else:
                    logger.info(f"NWOR kernel succeeded across {len(self._drafts)} layers")
                self._logged_success = True

            return self._num_draft_accepted if self._emit_metrics else 0

        except Exception as e:
            if not self._logged_failure:
                import traceback
                logger.warning(f"Draft commit kernel failed: {type(e).__name__}: {str(e)}, using fallback")
                logger.warning(f"Exception details: {repr(e)}")
                logger.warning(f"Traceback:\n{traceback.format_exc()}")
                self._logged_failure = True
            return self._fallback_commit(mask)

        finally:
            if self._cache_key is not None and self._capturing:
                cached_entry = self._cache.get(self._cache_key)
                if cached_entry is None:
                    cached_entry = DraftCacheEntry([], [], 0)
                    self._cache[self._cache_key] = cached_entry
                cached_entry.positions = list(self._draft_positions)
                cached_entry.drafts = [replace(entry) for entry in self._drafts]
                cached_entry.num_layers = len(self._drafts)
                # Update fallback to most recent capture
                self._fallback_cached_drafts = cached_entry.drafts
            self.cancel()

    def get_metrics(self) -> dict:
        """Return draft-only metrics from the last commit.

        Returns:
            Dictionary with keys:
            - num_draft_tokens: Total draft tokens staged
            - num_draft_accepted: Draft tokens accepted
            - num_draft_rejected: Draft tokens rejected
        """
        return {
            "num_draft_tokens": self._num_draft_tokens,
            "num_draft_accepted": self._num_draft_accepted,
            "num_draft_rejected": self._num_draft_rejected,
        }

    def _fallback_commit(self, mask: torch.Tensor) -> int:
        """Fallback to vanilla writer with sliced tensors."""
        if self._drafts:
            device = self._drafts[0]._key_ref.device
            if mask.dtype != torch.bool:
                mask = mask.to(dtype=torch.bool)
            mask = mask.to(device)
        else:
            return 0

        total_accepted = 0
        for entry in self._drafts:
            if entry.chunk_global_indices is None or entry.chunk_global_indices.numel() == 0:
                continue

            chunk_mask = mask[entry.chunk_global_indices]
            accepted_rows = chunk_mask.nonzero(as_tuple=False).squeeze(1)
            if accepted_rows.numel() == 0:
                continue
            local_positions = entry.chunk_local_positions.index_select(0, accepted_rows)

            key_accepted = entry._key_ref.index_select(0, local_positions)
            value_accepted = entry._value_ref.index_select(0, local_positions)
            slot_accepted = entry._slot_ref.index_select(0, local_positions)
            if slot_accepted.dtype != torch.int64:
                slot_accepted = slot_accepted.to(dtype=torch.int64)

            if entry._k_scale_ref is not None:
                if entry.scale_is_per_token:
                    k_scale = entry._k_scale_ref.index_select(0, local_positions)
                else:
                    k_scale = entry._k_scale_ref
            else:
                k_scale = None

            if entry._v_scale_ref is not None:
                if entry.scale_is_per_token:
                    v_scale = entry._v_scale_ref.index_select(0, local_positions)
                else:
                    v_scale = entry._v_scale_ref
            else:
                v_scale = None

            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key_accepted,
                value_accepted,
                entry._key_cache_ref,
                entry._value_cache_ref,
                slot_accepted,
                entry.kv_cache_dtype,
                k_scale,
                v_scale,
            )

            total_accepted += int(accepted_rows.numel())

        return total_accepted


# Global singleton
_draft_manager: Optional[DraftCommitManager] = None


def get_draft_manager() -> DraftCommitManager:
    """Get or create global draft manager singleton."""
    global _draft_manager
    if _draft_manager is None:
        _draft_manager = DraftCommitManager()
    return _draft_manager
