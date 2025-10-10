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
]


@dataclass
class _PendingLayer:
    layer_name: str
    key: torch.Tensor
    value: torch.Tensor
    slot_mapping: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    kv_cache_dtype: str
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    staged_tokens: int = 0


@dataclass
class _LayerAccumulator:
    layer_name: str
    key_chunks: list[torch.Tensor]
    value_chunks: list[torch.Tensor]
    slot_chunks: list[torch.Tensor]
    tokens_accumulated: int
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
        self._verifier_tokens: list[int] = []
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
        self._verifier_tokens = [1 if count > 0 else 0 for count in counts]
        self._total_tokens = sum(
            count + verify for count, verify in zip(counts, self._verifier_tokens))
        self._shared_slot_mapping = None
        self._current_accumulator = None
        self._total_windows += 1
        self._metrics["windows_attempted"] += 1
        self._metrics["tokens_deferred"] += total
        return True

    def abort_window(self) -> None:
        self._window_active = False
        self._pending_layers = []
        self._num_draft_tokens = []
        self._draft_total = 0
        self._verifier_tokens = []
        self._total_tokens = 0
        self._shared_slot_mapping = None
        self._current_accumulator = None

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
    ) -> bool:
        if not self.can_stage:
            return False

        chunk_len = int(slot_mapping.numel())
        if chunk_len == 0:
            return True

        if key.size(0) < chunk_len or value.size(0) < chunk_len:
            self._fallback("key/value tensors shorter than chunk")
            self._write_pending_fallback()
            self.abort_window()
            return False

        # Fast path: single chunk covers entire window and no accumulator active.
        if self._current_accumulator is None and chunk_len == self._total_tokens:
            slot_view = slot_mapping[:chunk_len]
            if self._shared_slot_mapping is None:
                slot_full = slot_view if slot_view.is_contiguous() else slot_view.contiguous()
                self._shared_slot_mapping = slot_full
            else:
                expected = self._shared_slot_mapping
                if (slot_view.device != expected.device
                        or slot_view.shape != expected.shape
                        or not torch.equal(slot_view, expected)):
                    self._fallback("slot mapping values differ across layers")
                    self._write_pending_fallback()
                    self.abort_window()
                    return False
                slot_full = expected

            key_full_view = key[:chunk_len]
            value_full_view = value[:chunk_len]
            key_full = key_full_view if key_full_view.is_contiguous() else key_full_view.contiguous()
            value_full = value_full_view if value_full_view.is_contiguous() else value_full_view.contiguous()

            self._pending_layers.append(
                _PendingLayer(
                    layer_name=layer_name,
                    key=key_full,
                    value=value_full,
                    slot_mapping=slot_full,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    kv_cache_dtype=kv_cache_dtype,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    staged_tokens=chunk_len,
                ))
            return True

        key_chunk = key[:chunk_len]
        value_chunk = value[:chunk_len]
        slot_chunk = slot_mapping[:chunk_len]

        acc = self._current_accumulator
        if acc is None or acc.layer_name != layer_name:
            if acc is not None:
                if not self._finalize_current_layer():
                    return False
            self._current_accumulator = _LayerAccumulator(
                layer_name=layer_name,
                key_chunks=[key_chunk],
                value_chunks=[value_chunk],
                slot_chunks=[slot_chunk],
                tokens_accumulated=chunk_len,
                key_cache=key_cache,
                value_cache=value_cache,
                kv_cache_dtype=kv_cache_dtype,
                k_scale=k_scale,
                v_scale=v_scale,
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("NWOR chunk layer=%s chunk_len=%d acc=%d/%d",
                             layer_name, chunk_len, chunk_len,
                             self._total_tokens)
        else:
            acc.key_chunks.append(key_chunk)
            acc.value_chunks.append(value_chunk)
            acc.slot_chunks.append(slot_chunk)
            acc.tokens_accumulated += chunk_len
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("NWOR chunk layer=%s chunk_len=%d acc=%d/%d",
                             layer_name, chunk_len, acc.tokens_accumulated,
                             self._total_tokens)

        acc = self._current_accumulator
        assert acc is not None
        if acc.tokens_accumulated == self._total_tokens:
            return self._finalize_current_layer()
        if acc.tokens_accumulated > self._total_tokens:
            self._fallback(
                f"Layer {acc.layer_name} over-accumulated: {acc.tokens_accumulated} > {self._total_tokens}")
            self._write_pending_fallback()
            self.abort_window()
            return False

        return True

    # ------------------------------------------------------------------ commit
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
        finally:
            self.abort_window()

    # ------------------------------------------------------------------ helpers
    def _finalize_current_layer(self) -> bool:
        acc = self._current_accumulator
        if acc is None:
            return True

        if acc.tokens_accumulated != self._total_tokens:
            self._current_accumulator = None
            self._fallback(
                f"Layer {acc.layer_name} incomplete: {acc.tokens_accumulated} != {self._total_tokens}")
            self._write_pending_fallback()
            self.abort_window()
            return False

        full_key = torch.cat(acc.key_chunks, dim=0)
        full_value = torch.cat(acc.value_chunks, dim=0)
        full_slot = torch.cat(acc.slot_chunks, dim=0)

        if not full_key.is_contiguous():
            full_key = full_key.contiguous()
        if not full_value.is_contiguous():
            full_value = full_value.contiguous()
        if not full_slot.is_contiguous():
            full_slot = full_slot.contiguous()

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

        self._pending_layers.append(
            _PendingLayer(
                layer_name=acc.layer_name,
                key=full_key,
                value=full_value,
                slot_mapping=full_slot,
                key_cache=acc.key_cache,
                value_cache=acc.value_cache,
                kv_cache_dtype=acc.kv_cache_dtype,
                k_scale=acc.k_scale,
                v_scale=acc.v_scale,
                staged_tokens=self._total_tokens,
            ))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("NWOR finalize layer=%s tokens=%d expected=%d",
                         acc.layer_name, self._total_tokens,
                         self._total_tokens)
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
            for key_chunk, value_chunk, slot_chunk in zip(
                    acc.key_chunks, acc.value_chunks, acc.slot_chunks):
                key_local = key_chunk if key_chunk.is_contiguous() else key_chunk.contiguous()
                value_local = value_chunk if value_chunk.is_contiguous() else value_chunk.contiguous()
                slot_local = slot_chunk if slot_chunk.is_contiguous() else slot_chunk.contiguous()
                torch.ops._C_cache_ops.reshape_and_cache_flash(
                    key_local,
                    value_local,
                    acc.key_cache,
                    acc.value_cache,
                    slot_local,
                    acc.kv_cache_dtype,
                    acc.k_scale,
                    acc.v_scale,
                )
            self._current_accumulator = None

        if not self._pending_layers:
            return
        try:
            for pending in self._pending_layers:
                staged = pending.staged_tokens or pending.key.shape[0]
                torch.ops._C_cache_ops.reshape_and_cache_flash(
                    pending.key[:staged],
                    pending.value[:staged],
                    pending.key_cache,
                    pending.value_cache,
                    pending.slot_mapping[:staged],
                    pending.kv_cache_dtype,
                    pending.k_scale,
                    pending.v_scale,
                )
        finally:
            self._pending_layers.clear()

    def _build_accept_mask(self, accepted_prefix: Sequence[int]) -> Optional[torch.Tensor]:
        if not self._pending_layers:
            return None
        device = self._pending_layers[0].slot_mapping.device
        mask_cpu = torch.zeros(self._total_tokens, dtype=torch.bool)
        cursor = 0
        for count, verifier, accepted in zip(self._num_draft_tokens,
                                             self._verifier_tokens,
                                             accepted_prefix):
            accepted_int = max(0, int(accepted))
            if accepted_int > count:
                raise ValueError(
                    f"Accepted tokens ({accepted_int}) exceed drafted tokens ({count})")
            end = cursor + accepted_int
            if end > mask_cpu.numel():
                return None
            if end > cursor:
                mask_cpu[cursor:end] = True
            cursor += count + verifier
        return mask_cpu.to(device, non_blocking=True)

    def _commit_layer(self, pending: _PendingLayer,
                      accepted_mask: torch.Tensor) -> None:
        slot_mapping = pending.slot_mapping
        if slot_mapping.numel() < self._total_tokens:
            raise ValueError("slot mapping shorter than expected window")

        local_mask = accepted_mask[:slot_mapping.numel()] & (slot_mapping >= 0)
        if not bool(local_mask.any()):
            return

        key_slice = pending.key[local_mask].contiguous()
        value_slice = pending.value[local_mask].contiguous()
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
