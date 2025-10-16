# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Deferred KV cache staging for No-Write-On-Reject (NWOR)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
from torch import Tensor

from vllm.logger import init_logger

logger = init_logger(__name__)

try:  # pragma: no cover - optional import
    from torch._subclasses.fake_tensor import FakeTensor  # type: ignore
except Exception:  # pragma: no cover - fallback for older PyTorch
    FakeTensor = ()

try:  # pragma: no cover - optional import
    from torch._C import _is_fake_tensor as torch_is_fake_tensor  # type: ignore
except Exception:  # pragma: no cover - fallback when helper missing
    torch_is_fake_tensor = None


class ShouldFallback(RuntimeError):
    """Raised when the deferred writer must abandon staging."""


@dataclass
class _LayerEntry:
    layer_id: str
    start: int
    length: int
    key_source: Tensor
    value_source: Tensor
    slot_mapping: Tensor
    key_cache: Tensor
    value_cache: Tensor
    kv_cache_dtype: str
    k_scale: Optional[Tensor]
    v_scale: Optional[Tensor]
    writer: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor, str, Optional[Tensor], Optional[Tensor]], None]


_global_manager: Optional["DeferredWriteManager"] = None


def get_global_deferred_manager() -> Optional["DeferredWriteManager"]:
    return _global_manager


def set_global_deferred_manager(manager: Optional["DeferredWriteManager"]) -> None:
    global _global_manager
    _global_manager = manager


def _is_fake_tensor(tensor: Tensor) -> bool:
    if isinstance(tensor, FakeTensor) or tensor.__class__.__name__ == "FakeTensor":
        return True
    if torch_is_fake_tensor is not None:
        try:
            return bool(torch_is_fake_tensor(tensor))
        except TypeError:  # pragma: no cover - defensive
            return False
    return False


def _tensor_has_storage(tensor: Tensor) -> bool:
    if not isinstance(tensor, Tensor):
        return False
    if tensor.is_meta:
        return False
    if _is_fake_tensor(tensor):
        return False
    try:
        tensor.data_ptr()
    except (RuntimeError, ValueError):
        return False
    return True


def _in_restricted_context() -> bool:
    try:  # pragma: no cover - torch.compile path
        import torch._dynamo as dynamo  # type: ignore

        if dynamo.is_compiling():
            return True
    except (ImportError, AttributeError):  # pragma: no cover - optional
        pass

    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_current_stream_capturing()
    except (RuntimeError, AttributeError):  # pragma: no cover - defensive
        return False


def _ensure_int32_slots(slot_mapping: Tensor, device: torch.device) -> Tensor:
    if slot_mapping.dtype != torch.int32 or slot_mapping.device != device:
        slot_mapping = slot_mapping.to(device=device, dtype=torch.int32, copy=False)
    if not slot_mapping.is_contiguous():
        slot_mapping = slot_mapping.contiguous()
    return slot_mapping


def _slice_scale(scale: Optional[Tensor], indices: Tensor) -> Optional[Tensor]:
    if scale is None:
        return None
    if scale.ndim == 0:
        return scale
    if scale.shape[0] == 0:
        return scale
    first_dim = scale.shape[0]
    target = int(indices.numel())
    if first_dim == target:
        return torch.index_select(scale, 0, indices)
    # Some implementations append an extra sentinel slot; ignore it.
    if first_dim == target + 1:
        return torch.index_select(scale[:-1], 0, indices)
    # Default: return the original scale (per-layer scale etc.).
    return scale


def _slice_scale_segment(
    scale: Optional[Tensor],
    start: int,
    end: int,
    entry_length: int,
) -> Optional[Tensor]:
    if scale is None:
        return None
    if scale.ndim == 0 or scale.shape[0] == 0:
        return scale
    length = end - start
    if length == 0:
        return scale.new_empty((0,), dtype=scale.dtype, device=scale.device)
    if scale.shape[0] == entry_length:
        return scale.narrow(0, start, length)
    if scale.shape[0] == entry_length + 1:
        return scale.narrow(0, start, length)
    return scale


class DeferredWriteManager:
    """Stages KV writes until acceptance is known."""

    SUPPORTED_MODES = {"stage", "immediate", "off"}

    def __init__(self, *, mode: str = "stage") -> None:
        self._window_active = False
        self._num_draft_tokens: list[int] = []
        self._expected_tokens = 0
        self._staged_tokens = 0
        self._req_start_offsets: list[int] = []
        self._entries: list[_LayerEntry] = []
        self._fallback_reason: Optional[str] = None
        self._metrics = {
            "windows": 0,
            "tokens_staged": 0,
            "tokens_committed": 0,
            "tokens_rejected": 0,
            "tokens_fallback": 0,
            "fallbacks": 0,
        }
        self._mode = self._validate_mode(mode)
        self._last_window_metrics: dict[str, int | str] | None = None

    # ----------------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------------
    @property
    def window_active(self) -> bool:
        return self._window_active

    @property
    def fallback_reason(self) -> Optional[str]:
        return self._fallback_reason

    def begin_window(self, num_draft_tokens: Sequence[int]) -> bool:
        """Arm the manager for a new speculative decode window."""

        if self._mode != "stage":
            return False

        self._clear_window()

        total_tokens = sum(int(n) for n in num_draft_tokens)
        if total_tokens <= 0:
            return False

        self._num_draft_tokens = [int(n) for n in num_draft_tokens]
        self._req_start_offsets.clear()
        running = 0
        for n in self._num_draft_tokens:
            self._req_start_offsets.append(running)
            running += n

        if _in_restricted_context():
            self._record_fallback("cuda_graph_capture")
            return False

        self._window_active = True
        self._expected_tokens = total_tokens
        self._staged_tokens = 0
        self._entries.clear()
        self._fallback_reason = None
        self._last_window_metrics = None
        self._metrics["windows"] += 1
        self._metrics["tokens_staged"] += total_tokens
        return True

    def set_mode(self, mode: str) -> None:
        self._mode = self._validate_mode(mode)

    def get_mode(self) -> str:
        return self._mode

    def finish_step(self) -> None:
        """Flush any pending data if the window did not complete."""

        if self._window_active:
            self.cancel_and_flush("incomplete_window")

    def get_metrics(self) -> dict[str, int | str]:
        metrics = dict(self._metrics)
        metrics["mode"] = self._mode
        return metrics

    # ------------------------------------------------------------------
    # Staging
    # ------------------------------------------------------------------
    def stage_layer(
        self,
        *,
        layer_id: str,
        key: Tensor,
        value: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        slot_mapping: Tensor,
        kv_cache_dtype: str,
        k_scale: Optional[Tensor],
        v_scale: Optional[Tensor],
        writer: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor, str, Optional[Tensor], Optional[Tensor]], None],
    ) -> bool:
        if not self._window_active:
            return False

        if not (_tensor_has_storage(key) and _tensor_has_storage(value)):
            raise ShouldFallback("kv_slice_without_storage")

        if not (_tensor_has_storage(key_cache) and _tensor_has_storage(value_cache)):
            raise ShouldFallback("kv_cache_not_materialized")

        slot_mapping = _ensure_int32_slots(slot_mapping, key.device)

        length = int(slot_mapping.shape[0])
        if length == 0:
            return True

        if self._staged_tokens + length > self._expected_tokens:
            raise ShouldFallback("staged_tokens_exceed_expected")

        entry = _LayerEntry(
            layer_id=layer_id,
            start=self._staged_tokens,
            length=length,
            key_source=key,
            value_source=value,
            slot_mapping=slot_mapping,
            key_cache=key_cache,
            value_cache=value_cache,
            kv_cache_dtype=kv_cache_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
            writer=writer,
        )
        self._entries.append(entry)
        self._staged_tokens += length
        return True

    # ------------------------------------------------------------------
    # Commit / Fallback
    # ------------------------------------------------------------------
    def commit(self, accepted_counts: Sequence[int]) -> None:
        if not self._window_active:
            return

        if len(accepted_counts) != len(self._num_draft_tokens):
            raise ShouldFallback("accepted_counts_mismatch")

        committed_total = 0
        total_requests = len(self._num_draft_tokens)
        expected_tokens = self._expected_tokens

        for entry in self._entries:
            entry_start = entry.start
            entry_end = entry_start + entry.length

            accepted_segments: list[tuple[int, int]] = []
            total_segment_tokens = 0
            for req_idx in range(total_requests):
                req_tokens = self._num_draft_tokens[req_idx]
                if req_tokens == 0:
                    continue
                req_start = self._req_start_offsets[req_idx]
                req_end = req_start + req_tokens
                if req_end <= entry_start:
                    continue
                if req_start >= entry_end:
                    break

                accepted = min(int(accepted_counts[req_idx]), req_tokens)
                if accepted <= 0:
                    continue

                accepted_end = req_start + accepted
                seg_start = max(entry_start, req_start)
                seg_end = min(entry_end, accepted_end)
                if seg_end <= seg_start:
                    continue

                local_start = seg_start - entry_start
                local_end = seg_end - entry_start
                accepted_segments.append((local_start, local_end))
                total_segment_tokens += seg_end - seg_start

            if total_segment_tokens == 0:
                continue

            if total_segment_tokens == entry.length and len(accepted_segments) == 1:
                segment_start, segment_end = accepted_segments[0]
                if segment_start == 0 and segment_end == entry.length:
                    try:
                        entry.writer(
                            entry.key_source,
                            entry.value_source,
                            entry.key_cache,
                            entry.value_cache,
                            _ensure_int32_slots(entry.slot_mapping, entry.slot_mapping.device),
                            entry.kv_cache_dtype,
                            entry.k_scale,
                            entry.v_scale,
                        )
                    except Exception as exc:  # pragma: no cover
                        reason = f"commit_failed:{entry.layer_id}"
                        self._record_fallback(reason)
                        self._flush_entries()
                        self._last_window_metrics = {
                            "mode": self._mode,
                            "committed": 0,
                            "rejected": expected_tokens,
                            "fallback": 1,
                            "reason": reason,
                        }
                        self._clear_window()
                        raise ShouldFallback(reason) from exc
                    committed_total += entry.length
                    continue

            for segment_start, segment_end in accepted_segments:
                length = segment_end - segment_start
                if length <= 0:
                    continue
                key_slice = entry.key_source.narrow(0, segment_start, length)
                value_slice = entry.value_source.narrow(0, segment_start, length)
                slot_slice = entry.slot_mapping.narrow(0, segment_start, length)
                slot_slice = _ensure_int32_slots(slot_slice, entry.slot_mapping.device)

                k_scale_slice = _slice_scale_segment(
                    entry.k_scale, segment_start, segment_end, entry.length
                )
                v_scale_slice = _slice_scale_segment(
                    entry.v_scale, segment_start, segment_end, entry.length
                )

                try:
                    entry.writer(
                        key_slice,
                        value_slice,
                        entry.key_cache,
                        entry.value_cache,
                        slot_slice,
                        entry.kv_cache_dtype,
                        k_scale_slice,
                        v_scale_slice,
                    )
                except Exception as exc:  # pragma: no cover
                    reason = f"commit_failed:{entry.layer_id}"
                    self._record_fallback(reason)
                    self._flush_entries()
                    self._last_window_metrics = {
                        "mode": self._mode,
                        "committed": 0,
                        "rejected": expected_tokens,
                        "fallback": 1,
                        "reason": reason,
                    }
                    self._clear_window()
                    raise ShouldFallback(reason) from exc

                committed_total += length

        rejected = max(self._expected_tokens - committed_total, 0)
        self._metrics["tokens_committed"] += committed_total
        self._metrics["tokens_rejected"] += rejected
        self._last_window_metrics = {
            "mode": self._mode,
            "committed": committed_total,
            "rejected": rejected,
            "fallback": 0,
        }
        self._clear_window()

    def cancel_and_flush(self, reason: str) -> None:
        if not self._window_active:
            return
        self._record_fallback(reason)
        self._flush_entries()
        self._last_window_metrics = {
            "mode": self._mode,
            "committed": 0,
            "rejected": 0,
            "fallback": 1,
            "reason": reason,
        }
        self._clear_window()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _flush_entries(self) -> None:
        for entry in self._entries:
            try:
                entry.writer(
                    entry.key_source,
                    entry.value_source,
                    entry.key_cache,
                    entry.value_cache,
                    entry.slot_mapping,
                    entry.kv_cache_dtype,
                    entry.k_scale,
                    entry.v_scale,
                )
            except Exception:  # pragma: no cover - log and continue
                logger.exception("NWOR fallback failed for layer %s", entry.layer_id)
        if self._entries:
            flushed_tokens = sum(e.length for e in self._entries)
            self._metrics["tokens_fallback"] += flushed_tokens

    def _record_fallback(self, reason: str) -> None:
        self._fallback_reason = reason
        self._metrics["fallbacks"] += 1

    def _clear_window(self) -> None:
        self._window_active = False
        self._num_draft_tokens.clear()
        self._expected_tokens = 0
        self._staged_tokens = 0
        self._entries.clear()
        self._req_start_offsets.clear()

    def _validate_mode(self, mode: str) -> str:
        normalized = mode.lower()
        if normalized in self.SUPPORTED_MODES:
            return normalized
        logger.warning("NWOR: unsupported mode '%s', defaulting to 'stage'", mode)
        return "stage"

    def pop_last_window_metrics(self) -> dict[str, int | str] | None:
        metrics = self._last_window_metrics
        self._last_window_metrics = None
        return metrics


def record_or_write_kv_cache(
    *,
    writer: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor, str, Optional[Tensor], Optional[Tensor]], None],
    layer_id: str,
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    slot_mapping: Tensor,
    kv_cache_dtype: str,
    k_scale: Optional[Tensor],
    v_scale: Optional[Tensor],
) -> None:
    manager = get_global_deferred_manager()
    if manager is None:
        writer(
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

    try:
        staged = manager.stage_layer(
            layer_id=layer_id,
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping,
            kv_cache_dtype=kv_cache_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
            writer=writer,
        )
    except ShouldFallback as exc:
        manager.cancel_and_flush(str(exc))
        set_global_deferred_manager(None)
        writer(
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

    if not staged:
        writer(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
