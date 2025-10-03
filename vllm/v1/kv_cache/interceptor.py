# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NWOR (No-Write-On-Reject) interceptor with device-side staging."""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Union

import torch
from torch import Tensor

from .staging import StagingBuffers

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
        self.num_draft_tokens = 0

        self.total_committed = 0
        self.total_rejected = 0
        self.total_staged_tokens = 0
        self.fallback_count = 0

        # Device-side staging buffers & layer cache mapping (populated by worker)
        self.staging_buffers: Optional[StagingBuffers] = None
        self.layer_key_caches: list[Tensor] = []
        self.layer_value_caches: list[Tensor] = []
        self.cache_ptr_to_layer: dict[int, int] = {}
        self.requires_eager_warmup: bool = False

    # ------------------------------------------------------------------ lifecycle
    def ensure_ready(self, key_cache: Tensor, value_cache: Tensor) -> None:
        if self.ready or not self.nwor_enabled:
            return
        if has_real_storage(key_cache) and has_real_storage(value_cache):
            self.ready = True
            logger.info("NWOR: KV cache ready, staging available")

    # Device staging buffers -------------------------------------------------

    def set_staging_buffers(self, buffers: Optional[StagingBuffers]) -> None:
        """Register device staging buffers allocated by the worker."""
        self.staging_buffers = buffers
        if buffers is not None:
            logger.info(
                "NWOR: Device staging buffers registered (capacity=%d)",
                buffers.capacity,
            )
            self.requires_eager_warmup = True
        else:
            self.requires_eager_warmup = False
            self.layer_key_caches = []
            self.layer_value_caches = []
            self.cache_ptr_to_layer = {}

    def needs_eager_warmup(self) -> bool:
        return self.requires_eager_warmup

    def register_layer_caches(
        self, key_caches: Sequence[Tensor], value_caches: Sequence[Tensor]
    ) -> None:
        if self.staging_buffers is not None and \
                len(key_caches) != len(self.staging_buffers.keys):
            self._fatal_error(
                "Layer cache count does not match staging buffers")
        self.layer_key_caches = list(key_caches)
        self.layer_value_caches = list(value_caches)
        self.cache_ptr_to_layer = {
            key.data_ptr(): idx for idx, key in enumerate(self.layer_key_caches)
        }

    def _layer_index_for_cache(self, key_cache: Tensor) -> int:
        ptr = key_cache.data_ptr()
        layer_idx = self.cache_ptr_to_layer.get(ptr)
        if layer_idx is None:
            self._fatal_error("Unknown key cache pointer in staging path")
        return layer_idx

    def enable_staging(self, num_tokens: Union[int, Sequence[int]]) -> bool:
        if not self.ready or not self.nwor_enabled:
            return False

        if self.staging_buffers is None:
            logger.warning("NWOR: Staging buffers unavailable; cannot enable staging")
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

        capacity = self.staging_buffers.capacity
        if draft_total > capacity:
            self._fatal_error(
                f"staging capacity {capacity} insufficient for draft window {draft_total}"
            )

        self.mode = "staging"
        self.num_draft_tokens = draft_total
        current_stream = torch.cuda.current_stream(self.staging_buffers.device)
        self.staging_buffers.reset(stream=current_stream)
        logger.debug(
            "NWOR: Staging enabled for %d draft tokens (capacity=%d)",
            draft_total,
            capacity,
        )
        return True

    def disable_staging(self, reason: str = "normal") -> None:
        logger.debug(
            "NWOR: Disabling staging (reason=%s)",
            reason,
        )
        self.mode = "direct"
        self.num_draft_tokens = 0
        if self.staging_buffers is not None:
            current_stream = torch.cuda.current_stream(self.staging_buffers.device)
            self.staging_buffers.reset(stream=current_stream)
        if reason != "normal":
            self.requires_eager_warmup = True

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

    def stage_layer_writes(
        self,
        key: Tensor,
        value: Tensor,
        key_cache: Tensor,
        slot_mapping: Tensor,
        kv_cache_dtype: str,
        k_scale: Optional[Tensor],
        v_scale: Optional[Tensor],
    ) -> bool:
        if self.mode != "staging" or self.staging_buffers is None:
            return False

        layer_idx = self._layer_index_for_cache(key_cache)
        layer_bufs = self.staging_buffers.layer_buffers(layer_idx)
        metadata = self.staging_buffers.metadata[layer_idx]

        torch.ops._C_cache_ops.stage_kv_cache(
            key,
            value,
            slot_mapping,
            layer_bufs.key,
            layer_bufs.value,
            layer_bufs.slots,
            metadata,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
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

        if self.staging_buffers is None:
            self._fatal_error("staging buffers not initialized")

        for layer_idx, (key_cache, value_cache) in enumerate(
                zip(self.layer_key_caches, self.layer_value_caches)):
            metadata = self.staging_buffers.metadata[layer_idx]
            staged_count = int(metadata[0].item())
            error_flag = int(metadata[1].item())
            if error_flag:
                self._fatal_error(
                    f"staging overflow detected for layer {layer_idx}")
            if accepted > staged_count:
                self._fatal_error(
                    f"staged tokens {staged_count} < accepted {accepted} for layer {layer_idx}")

        if accepted == 0:
            self.total_rejected += max(proposed_len, 0)
            self.disable_staging()
            return

        for layer_idx, (key_cache, value_cache) in enumerate(
                zip(self.layer_key_caches, self.layer_value_caches)):
            torch.ops._C_cache_ops.commit_staged_kv_cache(
                self.staging_buffers.keys[layer_idx],
                self.staging_buffers.values[layer_idx],
                self.staging_buffers.slots[layer_idx],
                self.staging_buffers.metadata[layer_idx],
                key_cache,
                value_cache,
                accepted,
            )

        self.total_staged_tokens += accepted
        self.total_committed += accepted
        self.total_rejected += max(proposed_len - accepted, 0)
        self.requires_eager_warmup = False
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
            "nwor_current_staged": int(
                self.staging_buffers.metadata[:, 0].sum().item()
            ) if self.staging_buffers is not None else 0,
        }

    def __del__(self):  # pragma: no cover - diagnostic only
        try:
            logger.info(
                "KVCacheInterceptor.__del__(): mode=%s, enabled=%s",
                getattr(self, "mode", "unknown"),
                getattr(self, "nwor_enabled", "unknown"),
            )
        except Exception:
            pass
