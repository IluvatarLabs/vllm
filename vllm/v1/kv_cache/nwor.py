"""Lightweight controller for NWOR deferred KV writes."""

from __future__ import annotations

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


class NWORController:
    """Defers KV writes until accepted prefixes are known."""

    def __init__(self, enabled: bool) -> None:
        self._feature_enabled = enabled
        self._window_active = False
        self._pending_layers: list[_PendingLayer] = []
        self._num_draft_tokens: list[int] = []
        self._total_tokens: int = 0
        self._fallback_reason: Optional[str] = None
        self._shared_slot_mapping: Optional[torch.Tensor] = None
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
        self._total_tokens = total
        self._shared_slot_mapping = None
        self._total_windows += 1
        self._metrics["windows_attempted"] += 1
        self._metrics["tokens_deferred"] += total
        return True

    def abort_window(self) -> None:
        self._window_active = False
        self._pending_layers = []
        self._num_draft_tokens = []
        self._total_tokens = 0
        self._shared_slot_mapping = None

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

        if slot_mapping.numel() < self._total_tokens:
            self._fallback(
                f"slot mapping shorter than window: {slot_mapping.numel()} < {self._total_tokens}")
            self._write_pending_fallback()
            self.abort_window()
            return False

        if key.size(0) < self._total_tokens or value.size(0) < self._total_tokens:
            self._fallback("key/value tensors shorter than expected window")
            self._write_pending_fallback()
            self.abort_window()
            return False

        if self._shared_slot_mapping is None:
            slot_mapping = slot_mapping.contiguous()
            self._shared_slot_mapping = slot_mapping
        else:
            expected = self._shared_slot_mapping
            if (slot_mapping.device != expected.device
                    or slot_mapping.shape != expected.shape
                    or not torch.equal(slot_mapping[:self._total_tokens],
                                       expected[:self._total_tokens])):
                self._fallback("slot mapping values differ across layers")
                self._write_pending_fallback()
                self.abort_window()
                return False
            slot_mapping = expected

        self._pending_layers.append(
            _PendingLayer(
                layer_name=layer_name,
                key=key,
                value=value,
                slot_mapping=slot_mapping,
                key_cache=key_cache,
                value_cache=value_cache,
                kv_cache_dtype=kv_cache_dtype,
                k_scale=k_scale,
                v_scale=v_scale,
            ))
        return True

    # ------------------------------------------------------------------ commit
    def commit_window(self, accepted_prefix: Sequence[int]) -> None:
        if not self._window_active:
            return

        if len(accepted_prefix) != len(self._num_draft_tokens):
            self._fallback("accepted_prefix length mismatch")
            self.abort_window()
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
                rejected_total = max(self._total_tokens - accepted_total, 0)
                self._metrics["windows_committed"] += 1
                self._metrics["tokens_committed"] += accepted_total
                self._metrics["tokens_rejected"] += rejected_total
        finally:
            self.abort_window()

    # ------------------------------------------------------------------ helpers
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
        if not self._pending_layers:
            return
        try:
            for pending in self._pending_layers:
                torch.ops._C_cache_ops.reshape_and_cache_flash(
                    pending.key,
                    pending.value,
                    pending.key_cache,
                    pending.value_cache,
                    pending.slot_mapping,
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
        for count, accepted in zip(self._num_draft_tokens, accepted_prefix):
            accepted_int = max(0, int(accepted))
            if accepted_int > count:
                raise ValueError(
                    f"Accepted tokens ({accepted_int}) exceed drafted tokens ({count})")
            end = cursor + accepted_int
            if end > mask_cpu.numel():
                return None
            if end > cursor:
                mask_cpu[cursor:end] = True
            cursor += count
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
