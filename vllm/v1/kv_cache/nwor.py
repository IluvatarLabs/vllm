"""Lightweight controller for NWOR deferred KV writes."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "NWORController",
    "set_global_nwor_controller",
    "get_global_nwor_controller",
    "record_or_write_kv_cache",
    "build_token_request_indices",
    "extract_query_start_loc_cpu",
]


class _StagingBuffers:
    """Device-side staging slabs shared across layers for a window."""

    def __init__(self) -> None:
        self.capacity: int = 0
        self.dtype: Optional[torch.dtype] = None
        self.device: Optional[torch.device] = None
        self.num_heads: Optional[int] = None
        self.head_dim: Optional[int] = None
        self.keys: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.slots: Optional[torch.Tensor] = None

    def ensure(self,
               *,
               num_layers: int,
               num_tokens: int,
               num_heads: int,
               head_dim: int,
               dtype: torch.dtype,
               device: torch.device) -> None:
        need_resize = (
            self.capacity != num_tokens or self.dtype != dtype
            or self.device != device or self.num_heads != num_heads
            or self.head_dim != head_dim)
        need_layer_list_resize = len(self.keys) != num_layers
        if not (need_resize or need_layer_list_resize):
            return

        old_keys = self.keys
        old_values = self.values
        old_slots = self.slots
        old_capacity = self.capacity
        old_num_heads = self.num_heads
        old_head_dim = self.head_dim
        old_dtype = self.dtype
        old_device = self.device

        self.capacity = num_tokens
        self.dtype = dtype
        self.device = device
        self.num_heads = num_heads
        self.head_dim = head_dim
        new_keys = [
            torch.empty((num_tokens, num_heads, head_dim),
                        dtype=dtype,
                        device=device)
            for _ in range(num_layers)
        ]
        new_values = [
            torch.empty((num_tokens, num_heads, head_dim),
                        dtype=dtype,
                        device=device)
            for _ in range(num_layers)
        ]
        new_slots = torch.empty((num_tokens,), dtype=torch.int32, device=device)

        # Preserve any already-staged data when the buffer grows mid-window.
        can_copy = (
            old_keys and old_values and old_capacity > 0
            and old_dtype == dtype and old_device == device
            and old_num_heads == num_heads and old_head_dim == head_dim)
        if can_copy:
            copy_tokens = min(old_capacity, num_tokens)
            copy_layers = min(len(old_keys), num_layers)
            if copy_tokens > 0 and copy_layers > 0:
                slice_spec = slice(0, copy_tokens)
                for layer_idx in range(copy_layers):
                    new_keys[layer_idx][slice_spec].copy_(
                        old_keys[layer_idx][slice_spec])
                    new_values[layer_idx][slice_spec].copy_(
                        old_values[layer_idx][slice_spec])
                if old_slots is not None and new_slots is not None:
                    new_slots[slice_spec].copy_(old_slots[slice_spec])

        self.keys = new_keys
        self.values = new_values
        self.slots = new_slots

    def zero_slots(self) -> None:
        if self.slots is not None:
            self.slots.zero_()


@dataclass
class _PendingLayer:
    layer_name: str
    layer_index: int
    slot_mapping: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    kv_cache_dtype: str
    k_scale: Optional[torch.Tensor]
    v_scale: Optional[torch.Tensor]
    request_indices: torch.Tensor
    staged_tokens: int = 0
    backup_keys: Optional[torch.Tensor] = None
    backup_values: Optional[torch.Tensor] = None
    backup_index_map: Optional[torch.Tensor] = None


@dataclass
class _LayerAccumulator:
    layer_name: str
    layer_index: int
    slot_chunks: list[torch.Tensor]
    draft_staged: int
    request_chunks: list[torch.Tensor]
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    kv_cache_dtype: str
    base_k_scale: Optional[torch.Tensor]
    base_v_scale: Optional[torch.Tensor]
    k_scale_chunks: list[torch.Tensor]
    v_scale_chunks: list[torch.Tensor]
    backup_key_chunks: list[torch.Tensor]
    backup_value_chunks: list[torch.Tensor]


@dataclass
class _VerifierTail:
    layer_name: str
    key: torch.Tensor
    value: torch.Tensor
    slots: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    kv_cache_dtype: str
    k_scale: torch.Tensor
    v_scale: torch.Tensor


class NWORController:
    """Defers KV writes until accepted prefixes are known."""

    def __init__(self, enabled: bool) -> None:
        self._feature_enabled = enabled
        self._window_active = False
        self._pending_layers: list[_PendingLayer] = []
        self._num_draft_tokens: list[int] = []
        self._draft_total: int = 0
        self._total_tokens: int = 0
        self._fallback_reason: Optional[str] = None
        self._shared_slot_mapping: Optional[torch.Tensor] = None
        self._current_accumulator: Optional[_LayerAccumulator] = None
        self._fallback_count: int = 0
        self._total_windows: int = 0
        self._max_fallbacks: int = 10
        self._metrics: dict[str, int] = {
            "windows_attempted": 0,
            "windows_committed": 0,
            "tokens_deferred": 0,
            "tokens_committed": 0,
            "tokens_rejected": 0,
            "fallbacks": 0,
        }
        self._layer_staged_tokens: dict[str, int] = {}
        self._layer_verifier_written: dict[str, bool] = {}
        self._pending_verifier_tails: list[_VerifierTail] = []
        self._window_request_layout: Optional[torch.Tensor] = None
        self._window_request_indices: Optional[torch.Tensor] = None
        self._layer_name_to_index: dict[str, int] = {}
        self._layer_names: list[str] = []
        self._staging_buffers = _StagingBuffers()
        self._staging_reset_pending = False

    # ------------------------------------------------------------------ flags
    @property
    def enabled(self) -> bool:
        return self._feature_enabled and self._fallback_reason is None

    @property
    def can_stage(self) -> bool:
        return self.enabled and self._window_active

    # ------------------------------------------------------------------ window
    def begin_window(self, num_draft_tokens: Sequence[int]) -> bool:
        if not self.enabled:
            self.abort_window()
            return False

        counts = [max(0, int(x)) for x in num_draft_tokens]
        total = sum(counts)
        if total <= 0:
            self.abort_window()
            return False

        self._window_active = True
        self._pending_layers = []
        self._num_draft_tokens = counts
        self._draft_total = total
        self._total_tokens = total

        request_ids = torch.arange(len(counts), dtype=torch.int32)
        counts_tensor = torch.tensor(counts, dtype=torch.int32)
        self._window_request_layout = torch.repeat_interleave(request_ids, counts_tensor)
        self._staging_reset_pending = True

        logger.debug(
            "NWOR begin_window: num_draft=%s draft_total=%d total_tokens=%d",
            counts,
            self._draft_total,
            self._total_tokens,
        )
        self._shared_slot_mapping = None
        self._current_accumulator = None
        self._layer_staged_tokens = {}
        self._layer_verifier_written = {}
        self._pending_verifier_tails = []
        self._window_request_indices = None
        self._total_windows += 1
        self._metrics["windows_attempted"] += 1
        self._metrics["tokens_deferred"] += total
        return True

    def abort_window(self) -> None:
        self._window_active = False
        self._pending_layers = []
        self._num_draft_tokens = []
        self._draft_total = 0
        self._total_tokens = 0
        self._shared_slot_mapping = None
        self._current_accumulator = None
        self._layer_staged_tokens = {}
        self._layer_verifier_written = {}
        self._pending_verifier_tails = []
        self._window_request_layout = None
        self._window_request_indices = None
        self._staging_reset_pending = False

    # ------------------------------------------------------------------ staging
    def record_layer(
        self,
        *,
        layer_name: str,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        token_request_indices: Optional[torch.Tensor],
    ) -> bool:
        if not self.can_stage:
            return False

        canonical_layout = self._window_request_layout
        if canonical_layout is None or canonical_layout.numel() != self._total_tokens:
            self._fallback("canonical request layout missing or inconsistent")
            self._write_pending_fallback()
            self.abort_window()
            return False

        chunk_len = int(slot_mapping.numel())
        if chunk_len == 0:
            return True

        request_indices: Optional[torch.Tensor] = None
        if token_request_indices is not None:
            request_indices = token_request_indices
            if request_indices.device.type != "cpu":
                request_indices = request_indices.cpu()
            request_indices = request_indices.to(torch.int32)

        if key.size(0) < chunk_len or value.size(0) < chunk_len:
            self._fallback("key/value tensors shorter than chunk")
            self._write_pending_fallback()
            self.abort_window()
            return False

        layer_idx = self._layer_name_to_index.get(layer_name)
        if layer_idx is None:
            layer_idx = len(self._layer_names)
            self._layer_names.append(layer_name)
            self._layer_name_to_index[layer_name] = layer_idx

        acc = self._current_accumulator
        if acc is not None and acc.layer_name != layer_name:
            if not self._finalize_current_layer():
                return False
            acc = self._current_accumulator

        staged_total = self._layer_staged_tokens.get(layer_name, 0)
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "NWOR stage_input: layer=%s idx=%d chunk=%d total=%d staged_total=%d req_none=%s",
                layer_name,
                layer_idx,
                chunk_len,
                self._total_tokens,
                staged_total,
                request_indices is None,
            )

        remaining = self._total_tokens - staged_total
        target_stage = min(chunk_len, remaining)
        verifier_len = chunk_len - target_stage

        stage_len = 0
        stage_positions: Optional[list[int]] = None
        tail_positions_for_verifier: Optional[list[int]] = None

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "NWOR record_layer: layer=%s chunk_len=%d stage_len=%d verifier_len=%d staged_total=%d/%d",
                layer_name,
                chunk_len,
                target_stage,
                verifier_len,
                staged_total,
                self._total_tokens,
            )

        if target_stage > 0:
            if acc is None:
                self._current_accumulator = _LayerAccumulator(
                    layer_name=layer_name,
                    layer_index=layer_idx,
                    slot_chunks=[],
                    draft_staged=0,
                    request_chunks=[],
                    key_cache=key_cache,
                    value_cache=value_cache,
                    kv_cache_dtype=kv_cache_dtype,
                    base_k_scale=k_scale,
                    base_v_scale=v_scale,
                    k_scale_chunks=[],
                    v_scale_chunks=[],
                    backup_key_chunks=[],
                    backup_value_chunks=[],
                )
                acc = self._current_accumulator

            canonical_slice = canonical_layout[staged_total:staged_total + target_stage]
            default_stage_positions = list(range(target_stage))
            default_tail_positions = list(range(target_stage, chunk_len))
            stage_positions = default_stage_positions
            tail_positions = default_tail_positions

            if request_indices is not None:
                align_result = self._align_request_indices(request_indices,
                                                           canonical_slice)
                if align_result is None:
                    provided_full = request_indices[:target_stage].tolist()
                    canonical_full = canonical_slice.tolist()
                    max_preview = 32
                    provided_preview = (provided_full if len(provided_full) <= max_preview
                                        else provided_full[:max_preview] +
                                        [f"...(+{len(provided_full) - max_preview} more)"])
                    canonical_preview = (canonical_full if len(canonical_full) <= max_preview
                                         else canonical_full[:max_preview] +
                                         [f"...(+{len(canonical_full) - max_preview} more)"])
                    if logger.isEnabledFor(logging.INFO):
                        logger.info(
                            "NWOR layout_mismatch: layer=%s idx=%d start=%d provided=%s canonical=%s",
                            layer_name,
                            layer_idx,
                            staged_total,
                            provided_preview,
                            canonical_preview,
                        )
                    self._fallback("request index layout mismatch for staged chunk")
                    self._write_pending_fallback()
                    self.abort_window()
                    return False

                stage_positions, tail_positions = align_result

                if stage_positions != default_stage_positions or tail_positions != default_tail_positions:
                    provided_full = request_indices[:target_stage].tolist()
                    canonical_full = canonical_slice.tolist()
                    max_preview = 32
                    provided_preview = (provided_full if len(provided_full) <= max_preview
                                        else provided_full[:max_preview] +
                                        [f"...(+{len(provided_full) - max_preview} more)"])
                    canonical_preview = (canonical_full if len(canonical_full) <= max_preview
                                         else canonical_full[:max_preview] +
                                         [f"...(+{len(canonical_full) - max_preview} more)"])
                    stage_preview = (stage_positions if len(stage_positions) <= max_preview
                                     else stage_positions[:max_preview] +
                                     [f"...(+{len(stage_positions) - max_preview} more)"])
                    tail_preview = (tail_positions if len(tail_positions) <= max_preview
                                    else tail_positions[:max_preview] +
                                    [f"...(+{len(tail_positions) - max_preview} more)"])
                    if logger.isEnabledFor(logging.INFO):
                        logger.info(
                            "NWOR layout_remap: layer=%s idx=%d start=%d provided=%s canonical=%s stage_positions=%s tail_positions=%s",
                            layer_name,
                            layer_idx,
                            staged_total,
                            provided_preview,
                            canonical_preview,
                            stage_preview,
                            tail_preview,
                        )

            idx_cpu = torch.tensor(stage_positions, dtype=torch.long)
            idx_device = idx_cpu.to(key.device)
            key_stage = key.index_select(0, idx_device)
            value_stage = value.index_select(0, idx_device)
            slot_stage = slot_mapping.index_select(0,
                                                   idx_cpu.to(slot_mapping.device))

            slot_indices = slot_stage.to(device=key_cache.device, dtype=torch.long)
            valid_mask = (slot_indices >= 0)
            if not bool(valid_mask.all()):
                mask_list = valid_mask.tolist()
                stage_positions = [pos for pos, keep in zip(stage_positions, mask_list) if keep]
                idx_device = idx_device[valid_mask]
                key_stage = key_stage[valid_mask]
                value_stage = value_stage[valid_mask]
                slot_stage = slot_stage[valid_mask]
                slot_indices = slot_indices[valid_mask]

            stage_len = slot_indices.numel()
            if stage_len == 0:
                stage_positions = []
                if logger.isEnabledFor(logging.INFO):
                    logger.info("NWOR stage_chunk: layer=%s idx=%d len=0 (no valid slots)",
                                layer_name, layer_idx)
                acc.slot_chunks.append(torch.empty((0,), dtype=slot_mapping.dtype,
                                                   device=slot_mapping.device))
                acc.request_chunks.append(torch.empty((0,), dtype=torch.int32))
                tail_positions_for_verifier = tail_positions
                verifier_len = len(tail_positions)
            else:
                stage_k_scale = self._select_token_scale(k_scale, stage_positions,
                                                         chunk_len)
                stage_v_scale = self._select_token_scale(v_scale, stage_positions,
                                                         chunk_len)

                original_keys = acc.key_cache.index_select(0, slot_indices)
                original_values = acc.value_cache.index_select(0, slot_indices)
                acc.backup_key_chunks.append(original_keys)
                acc.backup_value_chunks.append(original_values)
                stage_k_scale_arg = (stage_k_scale if stage_k_scale is not None
                                     else k_scale)
                stage_v_scale_arg = (stage_v_scale if stage_v_scale is not None
                                     else v_scale)
                if isinstance(stage_k_scale_arg, torch.Tensor) and stage_k_scale_arg.device != key_stage.device:
                    stage_k_scale_arg = stage_k_scale_arg.to(device=key_stage.device,
                                                            non_blocking=True)
                if isinstance(stage_v_scale_arg, torch.Tensor) and stage_v_scale_arg.device != value_stage.device:
                    stage_v_scale_arg = stage_v_scale_arg.to(device=value_stage.device,
                                                            non_blocking=True)
                torch.ops._C_cache_ops.reshape_and_cache_flash(
                    key_stage,
                    value_stage,
                    acc.key_cache,
                    acc.value_cache,
                    slot_stage.to(device=acc.key_cache.device,
                                  dtype=torch.long),
                    acc.kv_cache_dtype,
                    stage_k_scale_arg,
                    stage_v_scale_arg,
                )

                if not slot_stage.is_contiguous():
                    slot_stage = slot_stage.contiguous()

                if stage_k_scale is not None:
                    acc.k_scale_chunks.append(stage_k_scale.clone())
                if stage_v_scale is not None:
                    acc.v_scale_chunks.append(stage_v_scale.clone())

                req_stage_tensor = canonical_slice.to(dtype=torch.int32).clone()

                if logger.isEnabledFor(logging.INFO):
                    logger.info(
                        "NWOR stage_chunk: layer=%s idx=%d start=%d len=%d canonical_len=%d",
                        layer_name,
                        layer_idx,
                        staged_total,
                        stage_len,
                        canonical_slice.numel(),
                    )

                self._stage_chunk(layer_idx, key_stage, value_stage, slot_stage,
                                  staged_total)

                req_stage = req_stage_tensor.to(dtype=torch.int32).clone()
                acc.slot_chunks.append(slot_stage)
                acc.request_chunks.append(req_stage)
                acc.draft_staged += stage_len
                staged_total = acc.draft_staged
                self._layer_staged_tokens[layer_name] = staged_total
                tail_positions_for_verifier = tail_positions
                verifier_len = len(tail_positions)

            tail_positions_for_verifier = tail_positions
            verifier_len = len(tail_positions)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[NWOR] layer=%s staged=%d/%d", acc.layer_name,
                              staged_total, self._total_tokens)

            if staged_total > self._total_tokens:
                self._fallback(
                    f"Layer {layer_name} over-accumulated: {staged_total} > {self._total_tokens}")
                self._write_pending_fallback()
                self.abort_window()
                return False
            if staged_total == self._total_tokens:
                if not self._finalize_current_layer():
                    return False
                acc = self._current_accumulator

        if verifier_len > 0:
            staged_now = self._layer_staged_tokens.get(layer_name, 0)
            if staged_now < self._total_tokens:
                self._fallback(
                    f"Layer {layer_name} received verifier tokens before staging completed")
                self._write_pending_fallback()
                self.abort_window()
                return False
            verifier_done = self._layer_verifier_written.get(layer_name, False)
            if verifier_done:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("[NWOR] layer=%s ignoring duplicate verifier tail=%d",
                                  layer_name, verifier_len)
            else:
                if tail_positions_for_verifier is None:
                    tail_positions_for_verifier = list(range(verifier_len))

                tail_idx_cpu = torch.tensor(tail_positions_for_verifier,
                                            dtype=torch.long)
                tail_idx = tail_idx_cpu.to(device=key.device)
                verifier_key = key.index_select(0, tail_idx)
                verifier_value = value.index_select(0, tail_idx)
                verifier_slots = slot_mapping.index_select(
                    0, tail_idx_cpu.to(device=slot_mapping.device))

                verifier_slots_device = verifier_slots.to(device=key_cache.device,
                                                          dtype=torch.long)
                valid_tail_mask = (verifier_slots_device >= 0)
                if not bool(valid_tail_mask.all()):
                    tail_positions_for_verifier = [pos for pos, keep in zip(
                        tail_positions_for_verifier,
                        valid_tail_mask.tolist()) if keep]
                    verifier_key = verifier_key[valid_tail_mask]
                    verifier_value = verifier_value[valid_tail_mask]
                    verifier_slots_device = verifier_slots_device[valid_tail_mask]
                    if verifier_key.numel() == 0:
                        self._layer_verifier_written[layer_name] = True
                        logger.debug("NWOR verifier tail empty after filtering: layer=%s",
                                     layer_name)
                        return True
                verifier_len = len(tail_positions_for_verifier)

                verifier_key = (verifier_key if verifier_key.is_contiguous()
                                else verifier_key.contiguous())
                verifier_value = (verifier_value if verifier_value.is_contiguous()
                                  else verifier_value.contiguous())

                k_scale_tail = self._select_token_scale(k_scale,
                                                        tail_positions_for_verifier,
                                                        chunk_len)
                v_scale_tail = self._select_token_scale(v_scale,
                                                        tail_positions_for_verifier,
                                                        chunk_len)
                tail_k_scale = (k_scale_tail if k_scale_tail is not None else k_scale)
                tail_v_scale = (v_scale_tail if v_scale_tail is not None else v_scale)
                if isinstance(tail_k_scale, torch.Tensor):
                    if tail_k_scale.device != verifier_key.device:
                        tail_k_scale = tail_k_scale.to(device=verifier_key.device,
                                                       non_blocking=True)
                if isinstance(tail_v_scale, torch.Tensor):
                    if tail_v_scale.device != verifier_value.device:
                        tail_v_scale = tail_v_scale.to(device=verifier_value.device,
                                                       non_blocking=True)
                torch.ops._C_cache_ops.reshape_and_cache_flash(
                    verifier_key,
                    verifier_value,
                    key_cache,
                    value_cache,
                    verifier_slots_device,
                    kv_cache_dtype,
                    tail_k_scale,
                    tail_v_scale,
                )
                self._layer_verifier_written[layer_name] = True
                logger.debug("NWOR flushed verifier tail: layer=%s tail=%d",
                             layer_name, verifier_len)

        return True

    # ------------------------------------------------------------------ commit

    def _align_request_indices(
        self,
        request_indices: torch.Tensor,
        canonical_slice: torch.Tensor,
    ) -> Optional[tuple[list[int], list[int]]]:
        """Map chunk token positions to canonical request ordering."""

        if request_indices.numel() < canonical_slice.numel():
            return None

        req_list = request_indices.tolist()
        canonical_list = canonical_slice.tolist()
        num_requests = len(self._num_draft_tokens)

        buckets: dict[int, deque[int]] = defaultdict(deque)
        for pos, req in enumerate(req_list):
            req_int = int(req)
            if req_int < 0 or req_int >= num_requests:
                continue
            buckets[req_int].append(pos)

        stage_positions: list[int] = []
        for value in canonical_list:
            req_int = int(value)
            bucket = buckets.get(req_int)
            if bucket is None or not bucket:
                return None
            stage_positions.append(bucket.popleft())

        used = set(stage_positions)
        tail_positions = [pos for pos in range(len(req_list)) if pos not in used]
        return stage_positions, tail_positions

    @staticmethod
    def _select_token_scale(
        scale: Optional[torch.Tensor],
        positions: Sequence[int],
        chunk_len: int,
    ) -> Optional[torch.Tensor]:
        if scale is None:
            return None
        if scale.dim() == 0:
            return None
        if scale.size(0) != chunk_len:
            return None
        if not positions:
            return scale.new_empty((0, *scale.shape[1:]))
        idx = torch.tensor(positions, dtype=torch.long, device=scale.device)
        return scale.index_select(0, idx)

    @staticmethod
    def _merge_scales(
        chunks: Sequence[torch.Tensor],
        base_scale: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if chunks:
            if len(chunks) == 1:
                return chunks[0]
            return torch.cat(list(chunks), dim=0)
        return base_scale

    def _stage_chunk(self, layer_idx: int, key_chunk: torch.Tensor,
                     value_chunk: torch.Tensor, slot_chunk: torch.Tensor,
                     start: int) -> None:
        """Copy staged tokens into shared device buffers."""
        num_layers = len(self._layer_names)
        num_tokens = self._total_tokens
        num_heads = key_chunk.shape[1]
        head_dim = key_chunk.shape[2]
        dtype = key_chunk.dtype
        device = key_chunk.device

        self._staging_buffers.ensure(num_layers=num_layers,
                                     num_tokens=num_tokens,
                                     num_heads=num_heads,
                                     head_dim=head_dim,
                                     dtype=dtype,
                                     device=device)
        if self._staging_reset_pending:
            self._staging_buffers.zero_slots()
            self._staging_reset_pending = False

        end = start + key_chunk.shape[0]
        staging_keys = self._staging_buffers.keys[layer_idx]
        staging_values = self._staging_buffers.values[layer_idx]
        staging_keys[start:end].copy_(key_chunk.contiguous())
        staging_values[start:end].copy_(value_chunk.contiguous())
        slots = self._staging_buffers.slots
        if slots is not None:
            slots[start:end].copy_(slot_chunk.to(dtype=torch.int32))

    def commit_window(self, accepted_prefix: Sequence[int]) -> None:
        if not self._window_active:
            return

        if len(accepted_prefix) != len(self._num_draft_tokens):
            self._fallback("accepted_prefix length mismatch")
            self.abort_window()
            return

        if self._current_accumulator is not None:
            if not self._finalize_current_layer():
                return

        accepted_total = sum(max(0, int(x)) for x in accepted_prefix)

        if not self._pending_layers:
            self.abort_window()
            return

        try:
            mask = self._build_accept_mask(accepted_prefix)
        except Exception as exc:  # pylint: disable=broad-except
            self._fallback(str(exc))
            self._write_pending_fallback()
            self.abort_window()
            return
        if mask is None:
            self._fallback("failed to build acceptance mask")
            self._write_pending_fallback()
            self.abort_window()
            return

        success = True
        try:
            for pending in self._pending_layers:
                try:
                    self._commit_layer(pending, mask)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning("NWOR commit failed for %s: %s",
                                   pending.layer_name, exc)
                    success = False
                    break

            if not success:
                self._fallback("commit failure")
                self._write_pending_fallback()
            else:
                rejected_total = max(self._draft_total - accepted_total, 0)
                self._metrics["windows_committed"] += 1
                self._metrics["tokens_committed"] += accepted_total
                self._metrics["tokens_rejected"] += rejected_total
                self._flush_verifier_tails()
        finally:
            self.abort_window()

    # ------------------------------------------------------------------ helpers
    def _finalize_current_layer(self) -> bool:
        acc = self._current_accumulator
        if acc is None:
            return True

        logger.debug(
            "NWOR finalize: layer=%s accumulated=%d expected=%d num_chunks=%d",
            acc.layer_name,
            acc.draft_staged,
            self._total_tokens,
            len(acc.slot_chunks),
        )
        if acc.draft_staged != self._total_tokens:
            logger.warning("[NWOR] abort layer=%s accumulated=%d expected=%d",
                           acc.layer_name, acc.draft_staged,
                           self._total_tokens)
            self._current_accumulator = None
            self._fallback(
                f"Layer {acc.layer_name} incomplete: {acc.draft_staged} != {self._total_tokens}")
            self._write_pending_fallback()
            self.abort_window()
            return False

        full_slot = torch.cat(acc.slot_chunks, dim=0)
        if not full_slot.is_contiguous():
            full_slot = full_slot.contiguous()

        full_req = torch.cat(acc.request_chunks, dim=0)
        if full_req.device.type != "cpu":
            full_req = full_req.cpu()
        full_req = full_req.to(dtype=torch.int32)

        backup_keys = (torch.cat(acc.backup_key_chunks, dim=0)
                       if acc.backup_key_chunks else None)
        backup_values = (torch.cat(acc.backup_value_chunks, dim=0)
                         if acc.backup_value_chunks else None)
        backup_index_map = None
        if backup_keys is not None:
            valid_mask = (full_slot >= 0)
            backup_index_map = torch.full((full_slot.size(0),),
                                          -1,
                                          dtype=torch.int32,
                                          device=full_slot.device)
            valid_positions = torch.nonzero(valid_mask, as_tuple=False).view(-1)
            backup_index_map[valid_positions] = torch.arange(
                valid_positions.numel(),
                dtype=torch.int32,
                device=full_slot.device)
            # Ensure backups reside on same device as caches
            if backup_keys.device != acc.key_cache.device:
                backup_keys = backup_keys.to(device=acc.key_cache.device)
            if backup_values.device != acc.value_cache.device:
                backup_values = backup_values.to(device=acc.value_cache.device)

        if self._shared_slot_mapping is None:
            self._shared_slot_mapping = full_slot
        else:
            expected = self._shared_slot_mapping
            if (full_slot.device != expected.device
                    or full_slot.shape != expected.shape
                    or not torch.equal(full_slot, expected)):
                self._current_accumulator = None
                self._fallback("slot mapping values differ across layers")
                self._write_pending_fallback()
                self.abort_window()
                return False
            full_slot = expected

        canonical_layout = self._window_request_layout
        if canonical_layout is None or canonical_layout.numel() != full_req.numel():
            self._current_accumulator = None
            self._fallback("canonical request layout missing during finalize")
            self._write_pending_fallback()
            self.abort_window()
            return False

        canonical_cpu = canonical_layout.to(dtype=torch.int32)
        if not torch.equal(full_req, canonical_cpu):
            self._current_accumulator = None
            self._fallback("Accumulated request indices mismatch canonical layout")
            self._write_pending_fallback()
            self.abort_window()
            return False

        k_scale_full = self._merge_scales(acc.k_scale_chunks, acc.base_k_scale)
        v_scale_full = self._merge_scales(acc.v_scale_chunks, acc.base_v_scale)

        self._pending_layers.append(
            _PendingLayer(
                layer_name=acc.layer_name,
                layer_index=acc.layer_index,
                slot_mapping=full_slot,
                key_cache=acc.key_cache,
                value_cache=acc.value_cache,
                kv_cache_dtype=acc.kv_cache_dtype,
                k_scale=k_scale_full,
                v_scale=v_scale_full,
                request_indices=full_req,
                staged_tokens=acc.draft_staged,
                backup_keys=backup_keys,
                backup_values=backup_values,
                backup_index_map=backup_index_map,
            ))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[NWOR] finalize layer=%s total=%d expected=%d",
                         acc.layer_name, acc.draft_staged, self._total_tokens)
        self._layer_staged_tokens[acc.layer_name] = acc.draft_staged

        if self._window_request_indices is None:
            self._window_request_indices = full_req
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("NWOR request layout: %s",
                             full_req.tolist())
        else:
            if (self._window_request_indices.shape != full_req.shape or
                    not torch.equal(self._window_request_indices, full_req)):
                self._fallback(
                    "inconsistent request index layout across layers")
                self._write_pending_fallback()
                self.abort_window()
                return False

        verifier_logged = self._layer_verifier_written.get(acc.layer_name, False)
        if not verifier_logged:
            logger.debug("NWOR window summary: layer=%s staged=%d verifier=%d",
                         acc.layer_name, acc.draft_staged, 0)
        self._layer_verifier_written[acc.layer_name] = False
        self._current_accumulator = None
        return True

    def _fallback(self, reason: str) -> None:
        raise RuntimeError(f"NWOR fallback: {reason}")

    def _write_pending_fallback(self) -> None:
        if self._current_accumulator is not None:
            acc = self._current_accumulator
            for slot_chunk, key_backup, value_backup in zip(
                    acc.slot_chunks, acc.backup_key_chunks, acc.backup_value_chunks):
                slot_indices = slot_chunk.to(device=acc.key_cache.device, dtype=torch.long)
                valid_mask = slot_indices >= 0
                if not bool(valid_mask.all()):
                    slot_indices = slot_indices[valid_mask]
                    key_backup = key_backup[valid_mask]
                    value_backup = value_backup[valid_mask]
                if slot_indices.numel() == 0:
                    continue
                acc.key_cache.index_copy_(0, slot_indices, key_backup)
                acc.value_cache.index_copy_(0, slot_indices, value_backup)
            self._current_accumulator = None

        if self._pending_layers:
            try:
                for pending in self._pending_layers:
                    if (pending.backup_keys is None or pending.backup_values is None
                            or pending.backup_index_map is None):
                        continue
                    index_map = pending.backup_index_map.to(device=pending.key_cache.device,
                                                            dtype=torch.long)
                    valid_mask = index_map >= 0
                    slot_indices = pending.slot_mapping.to(device=pending.key_cache.device,
                                                            dtype=torch.long)[valid_mask]
                    if slot_indices.numel() == 0:
                        continue
                    key_indices = index_map[valid_mask]
                    key_backup = pending.backup_keys.index_select(0, key_indices)
                    value_backup = pending.backup_values.index_select(0, key_indices)
                    pending.key_cache.index_copy_(0, slot_indices, key_backup)
                    pending.value_cache.index_copy_(0, slot_indices, value_backup)
            finally:
                self._pending_layers.clear()

        self._flush_verifier_tails()

    def _build_accept_mask(self, accepted_prefix: Sequence[int]) -> Optional[torch.Tensor]:
        if not self._pending_layers:
            return None
        layout = self._window_request_indices
        if layout is None:
            layout = self._window_request_layout
            if layout is None:
                return None

        layout_cpu = layout.cpu()
        total_tokens = layout_cpu.numel()
        if total_tokens != self._total_tokens:
            raise ValueError("request index layout length mismatch")

        device = self._pending_layers[0].slot_mapping.device
        mask_cpu = torch.zeros(total_tokens, dtype=torch.bool)

        accepted_counts = [max(0, int(x)) for x in accepted_prefix]
        for idx, count in enumerate(accepted_counts):
            drafted = self._num_draft_tokens[idx]
            if count > drafted:
                raise ValueError(
                    f"Accepted tokens ({count}) exceed drafted tokens ({drafted})")

        consumed = [0] * len(accepted_counts)
        layout_list = layout_cpu.tolist()
        for token_idx, req_idx in enumerate(layout_list):
            if req_idx < 0 or req_idx >= len(accepted_counts):
                continue
            if consumed[req_idx] < accepted_counts[req_idx]:
                mask_cpu[token_idx] = True
                consumed[req_idx] += 1

        return mask_cpu.to(device=device, non_blocking=True)

    def _commit_layer(self, pending: _PendingLayer,
                      accepted_mask: torch.Tensor) -> None:
        slot_mapping = pending.slot_mapping
        if slot_mapping.numel() < self._total_tokens:
            raise ValueError("slot mapping shorter than expected window")

        layer_idx = pending.layer_index
        staging_keys = self._staging_buffers.keys[layer_idx]
        staging_values = self._staging_buffers.values[layer_idx]

        staged_mask = (pending.backup_index_map >= 0
                       if pending.backup_index_map is not None else
                       (slot_mapping >= 0))
        local_mask = accepted_mask[:slot_mapping.numel()] & staged_mask
        if bool(local_mask.any()):
            key_slice = staging_keys[local_mask].contiguous()
            value_slice = staging_values[local_mask].contiguous()
            slots_slice = slot_mapping[local_mask].contiguous()
            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key_slice,
                value_slice,
                pending.key_cache,
                pending.value_cache,
                slots_slice,
                pending.kv_cache_dtype,
                pending.k_scale,
                pending.v_scale,
            )

        if (pending.backup_keys is not None and pending.backup_values is not None
                and pending.backup_index_map is not None):
            rejected_mask = staged_mask & (~accepted_mask[:slot_mapping.numel()])
            if bool(rejected_mask.any()):
                restore_index = pending.backup_index_map[rejected_mask]
                valid_restore = restore_index >= 0
                if bool(valid_restore.any()):
                    restore_index = restore_index[valid_restore].to(
                        device=pending.key_cache.device, dtype=torch.long)
                    slot_indices = slot_mapping[rejected_mask][valid_restore].to(
                        device=pending.key_cache.device, dtype=torch.long)
                    restore_keys = pending.backup_keys.index_select(0, restore_index)
                    restore_values = pending.backup_values.index_select(0, restore_index)
                    pending.key_cache.index_copy_(0, slot_indices, restore_keys)
                    pending.value_cache.index_copy_(0, slot_indices, restore_values)
        pending.backup_keys = None
        pending.backup_values = None
        pending.backup_index_map = None

    def _flush_verifier_tails(self) -> None:
        if not self._pending_verifier_tails:
            return
        try:
            for tail in self._pending_verifier_tails:
                torch.ops._C_cache_ops.reshape_and_cache_flash(
                    tail.key,
                    tail.value,
                    tail.key_cache,
                    tail.value_cache,
                    tail.slots,
                    tail.kv_cache_dtype,
                    tail.k_scale,
                    tail.v_scale,
                )
        finally:
            self._pending_verifier_tails.clear()

    def commit_all_pending(self) -> None:
        if not self._window_active:
            self._pending_layers.clear()
            return
        self._write_pending_fallback()
        self.abort_window()

    def get_metrics(self) -> dict[str, float]:
        stats = dict(self._metrics)
        deferred = stats.get("tokens_deferred", 0)
        committed = stats.get("tokens_committed", 0)
        if deferred > 0:
            committed_clamped = min(committed, deferred)
            stats["acceptance_rate_pct"] = 100.0 * committed_clamped / deferred
            stats["live_cache_write_reduction_pct"] = 100.0 * (
                1.0 - committed_clamped / deferred)
        else:
            stats["acceptance_rate_pct"] = 0.0
            stats["live_cache_write_reduction_pct"] = 0.0
        attempts = stats.get("windows_attempted", 0)
        stats["fallback_rate_pct"] = (100.0 * stats["fallbacks"] /
                                       attempts if attempts > 0 else 0.0)
        return stats


_global_controller: Optional[NWORController] = None


def set_global_nwor_controller(controller: Optional[NWORController]) -> None:
    global _global_controller
    _global_controller = controller


def get_global_nwor_controller() -> Optional[NWORController]:
    return _global_controller


def record_or_write_kv_cache(
    *,
    layer_name: str,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    token_request_indices: Optional[torch.Tensor] = None,
) -> None:
    controller = get_global_nwor_controller()
    if controller is None or not controller.can_stage:
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
        return

    staged = controller.record_layer(
        layer_name=layer_name,
        key=key,
        value=value,
        slot_mapping=slot_mapping,
        key_cache=key_cache,
        value_cache=value_cache,
        kv_cache_dtype=kv_cache_dtype,
        k_scale=k_scale,
        v_scale=v_scale,
        token_request_indices=token_request_indices,
    )
    if not staged:
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )


def build_token_request_indices(
    query_start_loc_cpu: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Builds per-token request indices from query start offsets."""

    if query_start_loc_cpu is None:
        return None
    if query_start_loc_cpu.numel() < 2:
        return None

    if query_start_loc_cpu.device.type != "cpu":
        qsl = query_start_loc_cpu.cpu()
    else:
        qsl = query_start_loc_cpu

    qsl = qsl.to(dtype=torch.int32)
    lengths = qsl[1:] - qsl[:-1]
    if lengths.numel() == 0:
        return torch.empty(0, dtype=torch.int32)

    total = int(lengths.sum().item())
    if total == 0:
        return torch.empty(0, dtype=torch.int32)

    request_ids = torch.arange(lengths.numel(), dtype=torch.int32)
    return torch.repeat_interleave(request_ids, lengths)


def extract_query_start_loc_cpu(attn_metadata: object) -> Optional[torch.Tensor]:
    """Best-effort extraction of query-start offsets from metadata graphs."""

    if attn_metadata is None:
        return None

    def _coerce(candidate: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if candidate is None or not isinstance(candidate, torch.Tensor):
            return None
        if candidate.numel() < 2:
            return None
        if candidate.device.type != "cpu":
            candidate = candidate.cpu()
        return candidate

    def _from(obj: object) -> Optional[torch.Tensor]:
        if obj is None:
            return None
        for name in ("query_start_loc_cpu", "query_start_loc"):
            value = _coerce(getattr(obj, name, None))
            if value is not None:
                return value
        return None

    direct = _from(attn_metadata)
    if direct is not None:
        return direct

    for attr in ("prefill_metadata", "decode_metadata", "prefill", "decode"):
        nested = _from(getattr(attn_metadata, attr, None))
        if nested is not None:
            return nested

    return None
