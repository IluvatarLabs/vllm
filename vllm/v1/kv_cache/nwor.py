"""Lightweight controller for NWOR deferred KV writes."""

from __future__ import annotations

import logging
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
        need_realloc = (
            self.capacity < num_tokens or self.dtype != dtype
            or self.device != device or self.num_heads != num_heads
            or self.head_dim != head_dim or len(self.keys) != num_layers)
        if not need_realloc:
            return

        self.capacity = num_tokens
        self.dtype = dtype
        self.device = device
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.keys = [
            torch.empty((num_tokens, num_heads, head_dim),
                        dtype=dtype,
                        device=device)
            for _ in range(num_layers)
        ]
        self.values = [
            torch.empty((num_tokens, num_heads, head_dim),
                        dtype=dtype,
                        device=device)
            for _ in range(num_layers)
        ]
        self.slots = torch.empty((num_tokens,), dtype=torch.int32, device=device)

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
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    request_indices: torch.Tensor
    staged_tokens: int = 0


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
    k_scale: torch.Tensor
    v_scale: torch.Tensor


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
                k_scale=k_scale,
                v_scale=v_scale,
            )
            acc = self._current_accumulator

        staged_total = self._layer_staged_tokens.get(layer_name, 0)
        if staged_total >= self._total_tokens:
            return False

        remaining = self._total_tokens - staged_total
        stage_len = min(chunk_len, remaining)
        verifier_len = chunk_len - stage_len

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "NWOR record_layer: layer=%s chunk_len=%d stage_len=%d verifier_len=%d staged_total=%d/%d",
                layer_name,
                chunk_len,
                stage_len,
                verifier_len,
                staged_total,
                self._total_tokens,
            )

        if stage_len > 0:
            canonical_slice = canonical_layout[staged_total:staged_total + stage_len]
            if request_indices is not None:
                provided_slice = request_indices[:stage_len]
                if provided_slice.numel() != canonical_slice.numel() or not torch.equal(
                        provided_slice, canonical_slice):
                    self._fallback("request index layout mismatch for staged chunk")
                    self._write_pending_fallback()
                    self.abort_window()
                    return False
                req_stage = provided_slice
            else:
                req_stage = canonical_slice

            slot_stage = slot_mapping[:stage_len]
            if not slot_stage.is_contiguous():
                slot_stage = slot_stage.contiguous()
            self._stage_chunk(layer_idx, key[:stage_len], value[:stage_len],
                              slot_stage, staged_total)

            acc.slot_chunks.append(slot_stage)
            acc.request_chunks.append(req_stage.clone())
            acc.draft_staged += stage_len
            staged_total = acc.draft_staged
            self._layer_staged_tokens[layer_name] = staged_total
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[NWOR] layer=%s staged=%d/%d", acc.layer_name,
                              staged_total, self._total_tokens)

            if staged_total == self._total_tokens:
                if not self._finalize_current_layer():
                    return False
                acc = self._current_accumulator
            elif staged_total > self._total_tokens:
                self._fallback(
                    f"Layer {layer_name} over-accumulated: {staged_total} > {self._total_tokens}")
                self._write_pending_fallback()
                self.abort_window()
                return False

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
                start = stage_len
                verifier_key = key[start:]
                verifier_value = value[start:]
                verifier_slots = slot_mapping[start:]
                verifier_key = (verifier_key if verifier_key.is_contiguous() else
                                verifier_key.contiguous())
                verifier_value = (verifier_value if verifier_value.is_contiguous()
                                  else verifier_value.contiguous())
                verifier_slots = (verifier_slots if verifier_slots.is_contiguous()
                                  else verifier_slots.contiguous())
                tail = _VerifierTail(
                    layer_name=layer_name,
                    key=verifier_key,
                    value=verifier_value,
                    slots=verifier_slots,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    kv_cache_dtype=kv_cache_dtype,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )
                self._pending_verifier_tails.append(tail)
                self._layer_verifier_written[layer_name] = True
                logger.debug("NWOR queued verifier tail: layer=%s tail=%d",
                             layer_name, verifier_len)

        return True

    # ------------------------------------------------------------------ commit

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
            slots[start:end].copy_(slot_chunk.to(torch.int32))

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
        full_req = full_req.to(torch.int32)

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

        canonical_cpu = canonical_layout.to(torch.int32)
        if not torch.equal(full_req, canonical_cpu):
            self._current_accumulator = None
            self._fallback("Accumulated request indices mismatch canonical layout")
            self._write_pending_fallback()
            self.abort_window()
            return False

        self._pending_layers.append(
            _PendingLayer(
                layer_name=acc.layer_name,
                layer_index=acc.layer_index,
                slot_mapping=full_slot,
                key_cache=acc.key_cache,
                value_cache=acc.value_cache,
                kv_cache_dtype=acc.kv_cache_dtype,
                k_scale=acc.k_scale,
                v_scale=acc.v_scale,
                request_indices=full_req,
                staged_tokens=acc.draft_staged,
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
        self._fallback_count += 1
        self._metrics["fallbacks"] += 1
        if self._fallback_count < self._max_fallbacks:
            logger.warning("NWOR fallback #%d/%d: %s",
                           self._fallback_count, self._max_fallbacks, reason)
            return

        if self._fallback_reason is None:
            failure_rate = (100.0 * self._fallback_count /
                            max(1, self._total_windows))
            logger.error(
                "NWOR disabled after %d failures (%.1f%% of %d windows): %s",
                self._fallback_count,
                failure_rate,
                self._total_windows,
                reason,
            )
            self._fallback_reason = reason
            self._feature_enabled = False
        else:
            logger.debug("NWOR fallback after disable: %s", reason)

    def _write_pending_fallback(self) -> None:
        if self._current_accumulator is not None:
            acc = self._current_accumulator
            length = acc.draft_staged
            if length > 0:
                layer_idx = acc.layer_index
                staging_keys = self._staging_buffers.keys[layer_idx]
                staging_values = self._staging_buffers.values[layer_idx]
                full_slot = torch.cat(acc.slot_chunks, dim=0)
                if not full_slot.is_contiguous():
                    full_slot = full_slot.contiguous()
                torch.ops._C_cache_ops.reshape_and_cache_flash(
                    staging_keys[:length],
                    staging_values[:length],
                    acc.key_cache,
                    acc.value_cache,
                    full_slot[:length],
                    acc.kv_cache_dtype,
                    acc.k_scale,
                    acc.v_scale,
                )
            self._current_accumulator = None

        if self._pending_layers:
            try:
                for pending in self._pending_layers:
                    length = pending.staged_tokens
                    if length <= 0:
                        continue
                    layer_idx = pending.layer_index
                    staging_keys = self._staging_buffers.keys[layer_idx]
                    staging_values = self._staging_buffers.values[layer_idx]
                    torch.ops._C_cache_ops.reshape_and_cache_flash(
                        staging_keys[:length],
                        staging_values[:length],
                        pending.key_cache,
                        pending.value_cache,
                        pending.slot_mapping[:length],
                        pending.kv_cache_dtype,
                        pending.k_scale,
                        pending.v_scale,
                    )
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

        return mask_cpu.to(device, non_blocking=True)

    def _commit_layer(self, pending: _PendingLayer,
                      accepted_mask: torch.Tensor) -> None:
        slot_mapping = pending.slot_mapping
        if slot_mapping.numel() < self._total_tokens:
            raise ValueError("slot mapping shorter than expected window")

        local_mask = accepted_mask[:slot_mapping.numel()] & (slot_mapping >= 0)
        if not bool(local_mask.any()):
            return

        layer_idx = pending.layer_index
        staging_keys = self._staging_buffers.keys[layer_idx]
        staging_values = self._staging_buffers.values[layer_idx]

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

    qsl = qsl.to(torch.int32)
    lengths = qsl[1:] - qsl[:-1]
    if lengths.numel() == 0:
        return torch.empty(0, dtype=torch.int32)

    total = int(lengths.sum().item())
    if total == 0:
        return torch.empty(0, dtype=torch.int32)

    request_ids = torch.arange(lengths.numel(), dtype=torch.int32)
    return torch.repeat_interleave(request_ids, lengths)
