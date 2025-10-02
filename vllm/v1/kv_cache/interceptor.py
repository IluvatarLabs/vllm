# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NWOR (No-Write-On-Reject) interceptor with centralized staging queue."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

_global_interceptor: Optional["KVCacheInterceptor"] = None

def get_global_interceptor() -> Optional["KVCacheInterceptor"]:
    return _global_interceptor

def set_global_interceptor(interceptor: Optional["KVCacheInterceptor"]) -> None:
    global _global_interceptor
    _global_interceptor = interceptor

def has_real_storage(tensor: Tensor) -> bool:
    if not isinstance(tensor, Tensor):
        return False
    if tensor.is_meta:
        return False
    if hasattr(torch, "_is_fake_tensor"):
        try:
            if torch._is_fake_tensor(tensor):
                return False
        except Exception:  # pragma: no cover - defensive guard
            pass
    if tensor.__class__.__name__ == "FakeTensor":
        return False
    try:
        tensor.data_ptr()
        return True
    except (RuntimeError, NotImplementedError):
        return False

@dataclass
class StagedWrite:
    key: Tensor
    value: Tensor
    key_cache: Tensor
    value_cache: Tensor
    slot_mapping: Tensor
    kv_dtype: str
    k_scale: Optional[Tensor]
    v_scale: Optional[Tensor]

class KVCacheInterceptor:
    """Queues verifier KV writes and replays accepted prefix atomically."""

    def __init__(self, vllm_config) -> None:
        self.config = vllm_config
        self.ready = False

        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.n_layers = model_config.get_num_layers(parallel_config)

        spec_config = vllm_config.speculative_config
        if spec_config and spec_config.num_speculative_tokens and vllm_config.use_shadow_kv:
            self.max_spec_tokens = spec_config.num_speculative_tokens
            self.nwor_enabled = True
            logger.info("NWOR enabled (max speculative tokens: %d)", self.max_spec_tokens)
        else:
            self.max_spec_tokens = 0
            self.nwor_enabled = False
            logger.info("NWOR disabled (missing speculative config or use_shadow_kv=False)")

        self.mode = "direct"
        self.pending_writes: list[StagedWrite] = []
        self.num_draft_tokens = 0

        self.total_committed = 0
        self.total_rejected = 0
        self.total_staged_tokens = 0
        self.fallback_count = 0

    # ------------------------------------------------------------------ lifecycle
    def ensure_ready(self, key_cache: Tensor, value_cache: Tensor) -> None:
        if self.ready or not self.nwor_enabled:
            return
        if has_real_storage(key_cache) and has_real_storage(value_cache):
            self.ready = True
            logger.info("NWOR: KV cache ready, staging available")

    def should_queue(self) -> bool:
        return self.mode == "staging" and self.ready and self.nwor_enabled

    def enable_staging(self, num_tokens: Union[int, Sequence[int]]) -> bool:
        if not self.ready or not self.nwor_enabled:
            return False

        if isinstance(num_tokens, Sequence) and not isinstance(num_tokens, (str, bytes)):
            draft_total = sum(int(d) for d in num_tokens)
        else:
            draft_total = int(num_tokens)

        if draft_total <= 0:
            logger.debug("NWOR: enable_staging skipped (no speculative tokens)")
            return False

        if torch.cuda.is_current_stream_capturing():
            logger.warning("NWOR: Cannot enable staging during CUDA graph capture")
            return False

        self.mode = "staging"
        self.pending_writes.clear()
        self.num_draft_tokens = draft_total
        logger.debug("NWOR: Staging enabled for %d draft tokens", draft_total)
        return True

    def disable_staging(self, reason: str = "normal") -> None:
        logger.debug(
            "NWOR: Disabling staging (reason=%s, pending_writes=%d)",
            reason,
            len(self.pending_writes),
        )
        self.mode = "direct"
        self.pending_writes.clear()
        self.num_draft_tokens = 0

    def _fatal_error(self, reason: str, exc: Optional[Exception] = None) -> None:
        """Hard fail for true NWOR violations."""
        if exc is not None:
            logger.error("NWOR fatal error: %s (%s)", reason, exc)
        else:
            logger.error("NWOR fatal error: %s", reason)

        # Track once for diagnostics before aborting.
        self.fallback_count += 1

        # Ensure we leave staging mode to avoid leaking state while we crash.
        self.disable_staging(reason)

        if exc is not None:
            raise RuntimeError(f"NWOR fatal error: {reason}") from exc
        raise RuntimeError(f"NWOR fatal error: {reason}")

    # ------------------------------------------------------------------ staging queue
    def enqueue_write(
        self,
        key: Tensor,
        value: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        slot_mapping: Tensor,
        kv_cache_dtype: str,
        k_scale: Optional[float],
        v_scale: Optional[float],
    ) -> bool:
        if not self.should_queue():
            return False

        try:
            key_buf = key.detach().contiguous().clone()
            value_buf = value.detach().contiguous().clone()
            slot_buf = slot_mapping.detach().clone()
        except RuntimeError as exc:  # pragma: no cover - defensive
            self._fatal_error("failed to clone staging tensors", exc)

        write = StagedWrite(
            key=key_buf,
            value=value_buf,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_buf,
            kv_dtype=kv_cache_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        self.pending_writes.append(write)
        return True

    # ------------------------------------------------------------------ commit
    def commit_window(self, accepted_len: int, proposed_len: int) -> None:
        if self.mode != "staging":
            return

        accepted = max(0, min(int(accepted_len), self.num_draft_tokens))
        if accepted_len > self.num_draft_tokens:
            logger.warning(
                "NWOR: accepted_len (%d) exceeds draft tokens (%d), clamping",
                accepted_len,
                self.num_draft_tokens,
            )

        if not self.pending_writes:
            if accepted > 0:
                self._fatal_error(
                    f"missing staged writes for {accepted} accepted tokens"
                )
            self.total_rejected += max(proposed_len, 0)
            self.disable_staging("missing staged writes")
            return

        from vllm.attention.utils import fa_utils  # local import to avoid cycles

        for write in self.pending_writes:
            total = write.key.shape[0]
            if total < self.num_draft_tokens:
                self._fatal_error(
                    f"staged tensor smaller than draft window (total={total}, drafts={self.num_draft_tokens})"
                )

            draft_start = total - self.num_draft_tokens
            tokens_to_write = draft_start + accepted
            tokens_to_write = max(0, min(tokens_to_write, total))

            if tokens_to_write == 0:
                continue

            try:
                fa_utils._reshape_and_cache_flash_impl(  # type: ignore[attr-defined]
                    write.key[:tokens_to_write],
                    write.value[:tokens_to_write],
                    write.key_cache,
                    write.value_cache,
                    write.slot_mapping[:tokens_to_write],
                    write.kv_dtype,
                    write.k_scale,
                    write.v_scale,
                )
            except Exception as exc:  # pragma: no cover - defensive
                self._fatal_error("commit failure", exc)

        self.total_staged_tokens += accepted
        self.total_committed += accepted
        self.total_rejected += max(proposed_len - accepted, 0)
        self.disable_staging()

    # ------------------------------------------------------------------ metrics
    def get_window_tokens(self) -> int:
        return self.num_draft_tokens

    def get_metrics(self) -> dict:
        total = self.total_committed + self.total_rejected
        acceptance_rate = self.total_committed / total if total > 0 else 0.0
        return {
            "nwor_total_committed": self.total_committed,
            "nwor_total_rejected": self.total_rejected,
            "nwor_acceptance_rate": acceptance_rate,
            "nwor_total_staged_tokens": self.total_staged_tokens,
            "nwor_fallback_count": self.fallback_count,
            "nwor_current_queue_depth": len(self.pending_writes),
        }

    def __del__(self):  # pragma: no cover - diagnostic only
        try:
            logger.info(
                "KVCacheInterceptor.__del__(): mode=%s, enabled=%s, pending=%d",
                getattr(self, "mode", "unknown"),
                getattr(self, "nwor_enabled", "unknown"),
                len(getattr(self, "pending_writes", [])),
            )
        except Exception:
            pass
