"""Draft Commit Manager for NWOR (greenfield implementation)."""

from dataclasses import dataclass
from typing import Optional, List
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class DraftEntry:
    """Draft KV entry with pointers for kernel dispatch."""
    # Draft tensors (sources)
    key_ptr: int
    value_ptr: int
    num_tokens: int
    num_heads: int
    head_size: int

    # Cache tensors (destinations)
    key_cache_ptr: int
    value_cache_ptr: int
    block_size: int
    block_stride: int
    page_stride: int
    head_stride: int
    layout_id: int  # 0=flash, 1=paged

    # Slot mapping
    slot_ptr: int

    # Quantization
    k_scale_ptr: int  # 0 if None
    v_scale_ptr: int
    scale_is_per_token: bool
    kv_cache_dtype: str
    key_value_dtype: str

    # Keep tensors alive
    _key_ref: torch.Tensor
    _value_ref: torch.Tensor
    _slot_ref: torch.Tensor
    _key_cache_ref: torch.Tensor
    _value_cache_ref: torch.Tensor
    _k_scale_ref: Optional[torch.Tensor]
    _v_scale_ref: Optional[torch.Tensor]


class DraftCommitManager:
    """Manages draft buffers for NWOR with minimal overhead."""

    def __init__(self):
        import os
        nwor_mode = os.getenv("VLLM_NWOR_MODE", "off")
        self._nwor_enabled = (nwor_mode == "stage")  # Persistent config flag
        self.enabled = False  # Per-window active flag
        self._drafts: List[DraftEntry] = []
        self._logged_failure = False
        # Position tracking for mapping draft-only mask to full staged tensor
        self._logits_indices = None  # All token positions (targets + drafts)
        self._target_logits_indices = None  # Indices into logits_indices pointing to targets
        self._draft_positions = []  # Absolute positions of draft tokens in staged tensor
        if self._nwor_enabled:
            logger.info(f"NWOR enabled (VLLM_NWOR_MODE={nwor_mode})")

    def begin(self, spec_decode_metadata) -> bool:
        """Begin new spec decode window.

        Args:
            spec_decode_metadata: SpecDecodeMetadata containing token positions

        Returns:
            True if NWOR is enabled for this window
        """
        self._drafts.clear()
        self._draft_positions.clear()

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

        # Compute draft positions: for each request with drafts,
        # drafts follow target at target_idx+1, target_idx+2, ...
        for req_idx, num_draft in enumerate(spec_decode_metadata.num_draft_tokens):
            if num_draft > 0:
                target_idx = self._target_logits_indices[req_idx]
                for i in range(1, num_draft + 1):
                    pos = self._logits_indices[target_idx + i].item()
                    self._draft_positions.append(pos)

        self.enabled = True
        return True

    def cancel(self):
        """Cancel without committing."""
        self._drafts.clear()
        self.enabled = False

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

        # Validate storage
        for t, n in [(key, "key"), (value, "value"), (key_cache, "key_cache"),
                     (value_cache, "value_cache"), (slot_mapping, "slot_mapping")]:
            try:
                _ = t.data_ptr()
            except RuntimeError:
                logger.error(f"NWOR: {n} has no storage")
                self.enabled = False
                return

        # Ensure int32 contiguous slots
        if slot_mapping.dtype != torch.int32:
            slot_mapping = slot_mapping.to(dtype=torch.int32)
        if not slot_mapping.is_contiguous():
            slot_mapping = slot_mapping.contiguous()

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

        # Map dtype
        dtype_map = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}
        key_value_dtype = dtype_map.get(key.dtype, "fp16")

        self._drafts.append(DraftEntry(
            key_ptr=key.data_ptr(),
            value_ptr=value.data_ptr(),
            num_tokens=key.shape[0],
            num_heads=key.shape[1],
            head_size=key.shape[2],
            key_cache_ptr=key_cache.data_ptr(),
            value_cache_ptr=value_cache.data_ptr(),
            block_size=block_size,
            block_stride=key_cache.stride(0),
            page_stride=key_cache.stride(1),
            head_stride=key_cache.stride(2),
            layout_id=layout_id,
            slot_ptr=slot_mapping.data_ptr(),
            k_scale_ptr=k_scale.data_ptr() if k_scale is not None else 0,
            v_scale_ptr=v_scale.data_ptr() if v_scale is not None else 0,
            scale_is_per_token=scale_is_per_token,
            kv_cache_dtype=kv_cache_dtype,
            key_value_dtype=key_value_dtype,
            _key_ref=key,
            _value_ref=value,
            _slot_ref=slot_mapping,
            _key_cache_ref=key_cache,
            _value_cache_ref=value_cache,
            _k_scale_ref=k_scale,
            _v_scale_ref=v_scale,
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

        try:
            device = self._drafts[0]._key_ref.device
            num_tokens = self._drafts[0].num_tokens

            # Validate input mask covers all drafts
            assert mask.shape[0] == len(self._draft_positions), \
                f"mask size {mask.shape[0]} != num drafts {len(self._draft_positions)}"
            assert all(e.num_tokens == num_tokens for e in self._drafts)

            # Prepare draft mask
            if mask.dtype != torch.bool:
                mask = mask.to(dtype=torch.bool)
            draft_mask = mask.to('cpu') if mask.device.type != 'cpu' else mask

            # Build full mask for all staged tokens
            full_mask = torch.zeros(num_tokens, dtype=torch.bool, device='cpu')

            # Category 1: Prefill tokens (not in logits_indices) - always write
            logits_positions = set(self._logits_indices.cpu().tolist())
            all_positions = set(range(num_tokens))
            prefill_positions = all_positions - logits_positions
            for pos in prefill_positions:
                full_mask[pos] = True

            # Category 2: Target tokens - always write
            for target_idx in self._target_logits_indices.cpu():
                pos = self._logits_indices[target_idx].item()
                full_mask[pos] = True

            # Category 3: Draft tokens - conditional write based on input mask
            for i, pos in enumerate(self._draft_positions):
                full_mask[pos] = draft_mask[i]

            # Move full mask to device and ensure contiguous
            full_mask = full_mask.to(device=device, non_blocking=True)
            if not full_mask.is_contiguous():
                full_mask = full_mask.contiguous()

            # Count accepted (single GPU→CPU sync)
            num_accepted = int(full_mask.sum().item())
            if num_accepted == 0:
                return 0

            # Launch kernel for each layer (pass Tensors, not pointers)
            for entry in self._drafts:
                # Use empty tensor if scale is None
                k_scale = entry._k_scale_ref if entry._k_scale_ref is not None else torch.empty(0, device=device)
                v_scale = entry._v_scale_ref if entry._v_scale_ref is not None else torch.empty(0, device=device)

                torch.ops._C_cache_ops.commit_draft_layer(
                    entry._key_ref,
                    entry._value_ref,
                    entry._key_cache_ref,
                    entry._value_cache_ref,
                    full_mask,
                    entry._slot_ref,
                    k_scale,
                    v_scale,
                    entry.kv_cache_dtype,
                )

            torch.cuda.synchronize()

            # Log success once for verification
            if not hasattr(self, '_logged_success'):
                logger.info(f"NWOR kernel succeeded: committed {num_accepted}/{num_tokens} tokens across {len(self._drafts)} layers")
                self._logged_success = True

            return num_accepted

        except Exception as e:
            if not self._logged_failure:
                import traceback
                logger.warning(f"Draft commit kernel failed: {type(e).__name__}: {str(e)}, using fallback")
                logger.warning(f"Exception details: {repr(e)}")
                logger.warning(f"Traceback:\n{traceback.format_exc()}")
                self._logged_failure = True
            return self._fallback_commit(mask)

        finally:
            self.cancel()

    def _fallback_commit(self, mask: torch.Tensor) -> int:
        """Fallback to vanilla writer with sliced tensors."""
        accepted_indices = mask.nonzero(as_tuple=False).squeeze(1)
        num_accepted = accepted_indices.numel()
        if num_accepted == 0:
            return 0

        for entry in self._drafts:
            key_accepted = entry._key_ref[accepted_indices]
            value_accepted = entry._value_ref[accepted_indices]
            slot_accepted = entry._slot_ref[accepted_indices]

            # reshape_and_cache_flash expects int64
            if slot_accepted.dtype != torch.int64:
                slot_accepted = slot_accepted.to(dtype=torch.int64)

            # Scale slicing ONLY in fallback
            k_scale = None
            v_scale = None
            if entry._k_scale_ref is not None:
                k_scale = (entry._k_scale_ref[accepted_indices] if entry.scale_is_per_token
                          else entry._k_scale_ref)
            if entry._v_scale_ref is not None:
                v_scale = (entry._v_scale_ref[accepted_indices] if entry.scale_is_per_token
                          else entry._v_scale_ref)

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

        return num_accepted


# Global singleton
_draft_manager: Optional[DraftCommitManager] = None


def get_draft_manager() -> DraftCommitManager:
    """Get or create global draft manager singleton."""
    global _draft_manager
    if _draft_manager is None:
        _draft_manager = DraftCommitManager()
    return _draft_manager
