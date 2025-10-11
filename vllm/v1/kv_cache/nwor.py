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
    request_indices: torch.Tensor
    staged_tokens: int = 0


@dataclass
class _LayerAccumulator:
    layer_name: str
    key_chunks: list[torch.Tensor]
    value_chunks: list[torch.Tensor]
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

        chunk_len = int(slot_mapping.numel())
        if chunk_len == 0:
            return True

        canonical_layout = self._window_request_layout
        if canonical_layout is None or canonical_layout.numel() != self._total_tokens:
            self._fallback("canonical request layout missing or inconsistent")
            self._write_pending_fallback()
            self.abort_window()
            return False

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

        acc = self._current_accumulator
        if acc is not None and acc.layer_name != layer_name:
            if not self._finalize_current_layer():
                return False
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

        if (self._current_accumulator is None and staged_total == 0
                and stage_len == self._total_tokens and verifier_len == 0):
            if request_indices is not None:
                if request_indices.numel() != canonical_layout.numel():
                    logger.warning(
                        "NWOR fast-path length mismatch: layer=%s provided=%d canonical=%d",
                        layer_name,
                        request_indices.numel(),
                        canonical_layout.numel(),
                    )
                    self._fallback(
                        "fast path request indices length mismatch canonical layout")
                    self._write_pending_fallback()
                    self.abort_window()
                    return False
                if not torch.equal(request_indices, canonical_layout):
                    logger.debug(
                        "NWOR fast-path layout diff: layer=%s provided=%s canonical=%s",
                        layer_name,
                        request_indices.tolist(),
                        canonical_layout.tolist(),
                    )
                req_view = request_indices
            else:
                req_view = canonical_layout

            slot_view = slot_mapping[:stage_len]
            if self._shared_slot_mapping is None:
                slot_full = (slot_view if slot_view.is_contiguous() else
                             slot_view.contiguous())
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

            key_view = key[:stage_len]
            value_view = value[:stage_len]
            key_full = (key_view if key_view.is_contiguous() else
                        key_view.contiguous())
            value_full = (value_view if value_view.is_contiguous() else
                          value_view.contiguous())
            req_full = req_view.clone().to(torch.int32)
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
                    request_indices=req_full,
                    staged_tokens=stage_len,
                ))
            self._layer_staged_tokens[layer_name] = self._total_tokens
            self._layer_verifier_written[layer_name] = False
            if self._window_request_indices is None:
                self._window_request_indices = req_full
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("NWOR request layout: %s",
                                 req_full.tolist())
            elif self._window_request_indices.shape != req_full.shape or not torch.equal(
                    self._window_request_indices, req_full):
                self._fallback("inconsistent request index layout across layers")
                self._write_pending_fallback()
                self.abort_window()
                return False
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[NWOR] layer=%s staged=%d/%d", layer_name,
                              self._total_tokens, self._total_tokens)
            logger.debug("NWOR window summary: layer=%s staged=%d verifier=%d",
                         layer_name, self._total_tokens, 0)
            return True

        stage_success = True
        if stage_len > 0:
            key_stage = key[:stage_len]
            value_stage = value[:stage_len]
            slot_stage = slot_mapping[:stage_len]
            if request_indices is not None and request_indices.numel() >= stage_len:
                req_stage = request_indices[:stage_len].to(torch.int32)
                canonical_slice = canonical_layout[staged_total:staged_total + stage_len]
                if canonical_slice.numel() != req_stage.numel():
                    logger.warning(
                        "NWOR chunk length mismatch: layer=%s chunk_offset=%d provided=%d canonical=%d",
                        layer_name,
                        staged_total,
                        req_stage.numel(),
                        canonical_slice.numel(),
                    )
                    self._fallback("chunk layout length mismatch")
                    self._write_pending_fallback()
                    self.abort_window()
                    return False
                if not torch.equal(req_stage, canonical_slice):
                    logger.debug(
                        "NWOR chunk layout diff: layer=%s chunk_offset=%d provided=%s canonical=%s",
                        layer_name,
                        staged_total,
                        req_stage.tolist(),
                        canonical_slice.tolist(),
                    )
                else:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "NWOR chunk layout match: layer=%s offset=%d len=%d",
                            layer_name,
                            staged_total,
                            stage_len,
                        )
            else:
                if staged_total + stage_len > canonical_layout.numel():
                    self._fallback("canonical layout shorter than staged tokens")
                    self._write_pending_fallback()
                    self.abort_window()
                    return False
                req_stage = canonical_layout[staged_total:staged_total + stage_len].clone()

            acc = self._current_accumulator
            if acc is None:
                self._current_accumulator = _LayerAccumulator(
                    layer_name=layer_name,
                    key_chunks=[key_stage],
                    value_chunks=[value_stage],
                    slot_chunks=[slot_stage],
                    draft_staged=stage_len,
                    request_chunks=[req_stage],
                    key_cache=key_cache,
                    value_cache=value_cache,
                    kv_cache_dtype=kv_cache_dtype,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("NWOR chunk layer=%s chunk_len=%d acc=%d/%d",
                                 layer_name, stage_len, stage_len,
                                 self._total_tokens)
                acc = self._current_accumulator
            else:
                acc.key_chunks.append(key_stage)
                acc.value_chunks.append(value_stage)
                acc.slot_chunks.append(slot_stage)
                acc.request_chunks.append(req_stage)
                acc.draft_staged += stage_len
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("NWOR chunk layer=%s chunk_len=%d acc=%d/%d",
                                 layer_name, stage_len, acc.draft_staged,
                                 self._total_tokens)

            assert acc is not None
            staged_total = acc.draft_staged
            self._layer_staged_tokens[layer_name] = staged_total
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[NWOR] layer=%s staged=%d/%d", acc.layer_name,
                              staged_total, self._total_tokens)

            if staged_total == self._total_tokens:
                stage_success = self._finalize_current_layer()
            elif staged_total > self._total_tokens:
                self._fallback(
                    f"Layer {acc.layer_name} over-accumulated: {staged_total} > {self._total_tokens}")
                self._write_pending_fallback()
                self.abort_window()
                return False

            if not stage_success:
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

        return stage_success

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
            len(acc.key_chunks),
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

        full_key = torch.cat(acc.key_chunks, dim=0)
        full_value = torch.cat(acc.value_chunks, dim=0)
        full_slot = torch.cat(acc.slot_chunks, dim=0)
        full_req = torch.cat(acc.request_chunks, dim=0)

        if not full_key.is_contiguous():
            full_key = full_key.contiguous()
        if not full_value.is_contiguous():
            full_value = full_value.contiguous()
        if not full_slot.is_contiguous():
            full_slot = full_slot.contiguous()
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
                key=full_key,
                value=full_value,
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

        if self._pending_layers:
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
