# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NWOR (No-Write-On-Reject) KV cache interceptor for speculative decoding.
Stages KV writes during verification and commits only accepted tokens.
"""

import logging
from typing import Any, NamedTuple, Optional, Sequence, Union

import torch
from torch import Tensor

# Layer-specific cache references for commit
LayerRef = NamedTuple("LayerRef", [
    ("kv_ops", Any),
    ("key_cache", Tensor),
    ("value_cache", Tensor),
    ("kv_dtype", str),
    ("k_scale", Optional[float]),
    ("v_scale", Optional[float]),
])

logger = logging.getLogger(__name__)

# Global NWOR interceptor instance (singleton pattern for easy access)
_global_interceptor: Optional["KVCacheInterceptor"] = None

def get_global_interceptor() -> Optional["KVCacheInterceptor"]:
    """Get the global NWOR interceptor instance."""
    return _global_interceptor

def set_global_interceptor(interceptor: Optional["KVCacheInterceptor"]) -> None:
    """Set the global NWOR interceptor instance."""
    global _global_interceptor
    _global_interceptor = interceptor

# Multi-method FakeTensor detection for robustness across PyTorch versions
def has_real_storage(tensor: Tensor) -> bool:
    """Check if tensor has real device memory (not fake/meta)."""
    if not isinstance(tensor, Tensor):
        return False

    # Check meta tensor
    if tensor.is_meta:
        return False

    # Check FakeTensor via internal function (PyTorch >= 2.1)
    if hasattr(torch, '_is_fake_tensor'):
        try:
            if torch._is_fake_tensor(tensor):
                return False
        except:
            pass

    # Check via class name (fallback)
    if tensor.__class__.__name__ == "FakeTensor":
        return False

    # Try to get data pointer
    try:
        tensor.data_ptr()
        return True
    except (RuntimeError, NotImplementedError):
        return False


class StagingBuffer:
    """
    Transient buffer for staged KV pairs during speculation.

    Key Design Decisions:
    1. Single slot buffer shared across ALL layers (critical fix)
    2. Explicit token indexing (no sequential assumption)
    3. Token mask for validation
    4. All-or-nothing commit semantics
    """

    def __init__(self,
                 n_layers: int,
                 max_tokens: int,
                 n_heads: int,
                 head_dim: int,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        """
        Initialize staging buffer with TP-aware sizing.

        Args:
            n_layers: Number of transformer layers
            max_tokens: Maximum speculative tokens (e.g., 48)
            n_heads: Number of KV heads (local to this TP rank)
            head_dim: Dimension per head
            device: Device for buffers
            dtype: Data type for KV tensors
        """
        self.n_layers = n_layers
        self.max_tokens = max_tokens
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Allocate KV buffers
        shape = (n_layers, max_tokens, n_heads, head_dim)
        self.k_buffer = torch.empty(shape, device=device, dtype=dtype)
        self.v_buffer = torch.empty(shape, device=device, dtype=dtype)

        # CRITICAL: Single slot buffer for ALL layers (not per-layer!)
        self.slot_buffer = torch.empty(max_tokens, device=device, dtype=torch.long)

        # Track which positions have been staged
        self.token_mask = torch.zeros(max_tokens, device=device, dtype=torch.bool)

        # Per-layer cache references (stored during staging, used during commit)
        self.layer_refs: dict[int, LayerRef] = {}

        # Metrics
        self.stage_count = 0  # Total stage operations

        logger.info(f"StagingBuffer initialized: {n_layers} layers, {max_tokens} max tokens, "
                   f"{n_heads} heads, {head_dim} head_dim")

    def reset(self):
        """Clear buffer for new speculation round."""
        self.token_mask.zero_()
        self.layer_refs.clear()
        self.stage_count = 0

    def is_busy(self) -> bool:
        """Check if buffer has uncommitted data."""
        return self.token_mask.any().item()

    def unique_tokens(self) -> int:
        """Count unique token positions staged."""
        return int(self.token_mask.count_nonzero().item())

    def stage(self,
              layer_idx: int,
              token_idx: int,
              k_slice: Tensor,
              v_slice: Tensor,
              slot_tensor: Tensor,
              kv_cache_ops,
              key_cache: Tensor,
              value_cache: Tensor,
              kv_cache_dtype: str,
              k_scale: Optional[float] = None,
              v_scale: Optional[float] = None):
        """
        Stage KV slice at specific position.

        Args:
            layer_idx: Transformer layer index
            token_idx: Token position in speculation window
            k_slice: Key tensor [n_heads, head_dim] or [1, n_heads, head_dim]
            v_slice: Value tensor [n_heads, head_dim] or [1, n_heads, head_dim]
            slot_tensor: Slot mapping (scalar or [1])
            kv_cache_ops: Module with reshape_and_cache_flash function
            key_cache: Real key cache for this layer
            value_cache: Real value cache for this layer
            kv_cache_dtype: Cache data type
            k_scale: Optional key quantization scale
            v_scale: Optional value quantization scale
        """
        # Bounds check
        if not (0 <= layer_idx < self.n_layers):
            logger.error(f"Layer index {layer_idx} out of bounds [0, {self.n_layers})")
            raise ValueError(f"Invalid layer index: {layer_idx}")

        if not (0 <= token_idx < self.max_tokens):
            logger.error(f"Token index {token_idx} out of bounds [0, {self.max_tokens})")
            raise ValueError(f"Invalid token index: {token_idx}")

        # Store cache reference on first call for this layer
        if layer_idx not in self.layer_refs:
            self.layer_refs[layer_idx] = LayerRef(
                kv_ops=kv_cache_ops,
                key_cache=key_cache,
                value_cache=value_cache,
                kv_dtype=kv_cache_dtype,
                k_scale=k_scale,
                v_scale=v_scale,
            )

        # Store KV (squeeze batch dim if present)
        if k_slice.dim() == 3:
            k_slice = k_slice.squeeze(0)
        if v_slice.dim() == 3:
            v_slice = v_slice.squeeze(0)

        self.k_buffer[layer_idx, token_idx] = k_slice
        self.v_buffer[layer_idx, token_idx] = v_slice

        # CRITICAL: Store slot only on first layer (shared across all layers!)
        if layer_idx == 0:
            if slot_tensor.dim() > 0:
                slot_tensor = slot_tensor.squeeze()
            self.slot_buffer[token_idx] = slot_tensor

        # Mark position as staged
        self.token_mask[token_idx] = True
        self.stage_count += 1

    def commit(self, accepted_len: int) -> int:
        """
        Commit accepted prefix to real KV cache (all-or-nothing).

        Uses stored LayerRef objects captured during staging.

        Args:
            accepted_len: Number of accepted tokens

        Returns:
            Number of tokens committed (0 if failed, accepted_len if success)
        """
        # Handle edge case: zero accepted tokens
        if accepted_len <= 0:
            logger.debug("No tokens accepted, skipping commit")
            return 0

        # Validate complete staging for accepted range
        if not self.token_mask[:accepted_len].all():
            logger.warning(f"Incomplete staging for tokens [0:{accepted_len}], rejecting all")
            return 0

        # Validate we have all layer references
        if not self.layer_refs:
            logger.error("No layer references stored, cannot commit")
            return 0

        # Get shared slot mapping for all layers
        slots = self.slot_buffer[:accepted_len].contiguous()

        # All-or-nothing commit: try all layers, fail if any fails
        for layer_idx in sorted(self.layer_refs.keys()):
            ref = self.layer_refs[layer_idx]

            try:
                k_accepted = self.k_buffer[layer_idx, :accepted_len].contiguous()
                v_accepted = self.v_buffer[layer_idx, :accepted_len].contiguous()

                # Use the same slots for ALL layers (critical!)
                ref.kv_ops.reshape_and_cache_flash(
                    k_accepted,
                    v_accepted,
                    ref.key_cache,
                    ref.value_cache,
                    slots,
                    ref.kv_dtype,
                    ref.k_scale,
                    ref.v_scale,
                )
            except Exception as e:
                logger.error(f"Commit failed at layer {layer_idx}: {e}")
                return 0  # Reject entire window on any failure

        logger.debug(f"Successfully committed {accepted_len} tokens across {len(self.layer_refs)} layers")
        return accepted_len  # All layers succeeded

    def __del__(self):
        """DIAGNOSTIC: Track buffer destruction."""
        try:
            logger.info(f"DIAGNOSTIC: StagingBuffer.__del__() called - stage_count={getattr(self, 'stage_count', 'unknown')}, "
                       f"token_mask_sum={getattr(self, 'token_mask', torch.tensor([])).sum().item() if hasattr(self, 'token_mask') else 'unknown'}")
        except Exception as e:
            # Don't let __del__ errors propagate
            logger.error(f"DIAGNOSTIC: Error in StagingBuffer.__del__(): {e}")


class KVCacheInterceptor:
    """
    Intercepts KV cache writes for NWOR optimization.

    Minimal state machine:
    - direct mode: Normal KV cache writes
    - staging mode: Buffer writes for speculative tokens
    """

    def __init__(self, vllm_config):
        """
        Initialize interceptor.

        Args:
            vllm_config: VllmConfig object with model and speculation settings
        """
        self.config = vllm_config
        self.buffer: Optional[StagingBuffer] = None
        self.mode = "direct"
        self.ready = False
        self.last_token_idx = -1  # Track previous token index for layer boundary detection

        # Get model dimensions (use vLLM helpers for compatibility)
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.n_layers = model_config.get_num_layers(parallel_config)  # PP-aware!
        self.n_heads = model_config.get_num_kv_heads(parallel_config)  # TP-aware!
        self.head_dim = model_config.get_head_size()

        # Get speculation settings
        spec_config = vllm_config.speculative_config
        if spec_config and spec_config.num_speculative_tokens and vllm_config.use_shadow_kv:
            self.max_spec_tokens = spec_config.num_speculative_tokens
            self.nwor_enabled = True
            logger.info(f"NWOR enabled with max {self.max_spec_tokens} speculative tokens")
        else:
            self.max_spec_tokens = 0
            self.nwor_enabled = False
            logger.info("NWOR disabled (no speculative config or use_shadow_kv=False)")

        # Dynamic window sizing (updated at enable_staging time)
        self.window_capacity = max(self.max_spec_tokens + 1, 0)
        self.active_window_tokens = 0

        # Metrics
        self.total_staged = 0
        self.total_committed = 0
        self.total_rejected = 0
        self.fallback_count = 0
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3  # Disable NWOR after 3 consecutive commit failures
        self.last_token_idx = -1  # Track previous token index for layer boundary detection
        self.min_token_idx = float('inf')  # Track smallest token index seen in current window

        # Layer tracking for staging
        # Incremented when token_idx==0 (new layer starting)
        # Reset when starting new speculation window
        self.current_layer_idx = -1

    def ensure_ready(self, key_cache: Tensor, value_cache: Tensor) -> None:
        """Check if KV cache has real storage (post-warmup)."""
        if self.ready or not self.nwor_enabled:
            return

        # Check multiple layers to be sure
        if has_real_storage(key_cache) and has_real_storage(value_cache):
            self.ready = True
            logger.info("NWOR: KV cache ready, staging enabled")

    def enable_staging(self, num_tokens: Union[int, Sequence[int]]) -> bool:
        """
        Switch to staging mode for speculation with dynamic window sizing.

        Buffer is created lazily on first write() with real KV tensor dtype/device.

        Args:
            num_tokens: Either int (legacy sum) or list of per-request draft counts

        Returns:
            True if staging enabled, False if fallback to direct
        """
        if not self.ready or not self.nwor_enabled:
            return False

        # Circuit breaker: Disable NWOR after consecutive failures
        if self.consecutive_failures >= self.max_consecutive_failures:
            logger.error(f"NWOR: {self.consecutive_failures} consecutive failures, permanently disabling NWOR")
            self.nwor_enabled = False
            return False

        # Compute dynamic window size FIRST (before any early returns)
        # This ensures window size is always updated for new batch
        if isinstance(num_tokens, (list, tuple)):
            total_drafts = sum(int(d) for d in num_tokens)
            active_reqs = sum(1 for d in num_tokens if int(d) > 0)
            # Each active request has: 1 verified + N drafts
            # Total = sum(drafts) + count(active_reqs with >0 drafts)
            self.active_window_tokens = total_drafts + active_reqs
            logger.info(
                "NWOR: Dynamic window size=%d (total_drafts=%d, active_reqs=%d)",
                self.active_window_tokens, total_drafts, active_reqs,
            )
        else:
            total = int(num_tokens)
            self.active_window_tokens = total + 1
            logger.info(
                "NWOR: Legacy window size=%d (num_draft_tokens=%d)",
                self.active_window_tokens, total,
            )

        if self.active_window_tokens <= 0:
            logger.debug("NWOR: enable_staging skipped (no speculative tokens)")
            self.active_window_tokens = 0
            return False

        # If already in staging mode, window size is now updated - just return
        if self.mode == "staging":
            logger.debug(f"NWOR: Already in staging mode, updated window to {self.active_window_tokens}")
            # Reset layer tracking even on early return (for deferred commit scenarios)
            self.current_layer_idx = -1
            self.last_token_idx = -1
            return True

        # Check if existing buffer is busy (shouldn't happen with proper lifecycle)
        if self.buffer is not None and self.buffer.is_busy():
            logger.warning("NWOR: Buffer busy, falling back to direct mode")
            self.fallback_count += 1
            return False

        # Check if window exceeds current capacity; grow buffer if needed
        if self.active_window_tokens > self.window_capacity:
            logger.warning(f"NWOR: Window size {self.active_window_tokens} exceeds capacity "
                          f"{self.window_capacity}, growing buffer")
            self.window_capacity = self.active_window_tokens
            # Force buffer recreation with new capacity
            if self.buffer is not None:
                self.buffer = None

        # Mark staging mode active; buffer will be created lazily on first write()
        self.mode = "staging"
        if self.buffer is not None:
            self.buffer.reset()
        self.current_layer_idx = -1  # Reset layer counter for new window
        self.last_token_idx = -1  # Reset token tracking for new window
        self.min_token_idx = float('inf')
        logger.info(f"NWOR: Staging mode ENABLED (buffer will be created on first write)")
        return True

    def get_window_tokens(self) -> int:
        """Return the currently active staging window size."""
        return self.active_window_tokens

    def write(self,
              layer_idx: int,  # NOTE: Ignored, auto-determined from token_idx pattern
              token_idx: int,
              key: Tensor,
              value: Tensor,
              slot: Tensor,
              kv_cache_ops,
              key_cache: Tensor,
              value_cache: Tensor,
              kv_cache_dtype: str,
              k_scale: Optional[float] = None,
              v_scale: Optional[float] = None):
        """
        Route KV write to staging buffer or direct to cache.

        Args:
            layer_idx: Transformer layer index (IGNORED - auto-determined)
            token_idx: Token position in speculation window
            key: Key tensor
            value: Value tensor
            slot: Slot mapping
            kv_cache_ops: Module with reshape_and_cache_flash
            key_cache: Real key cache
            value_cache: Real value cache
            kv_cache_dtype: Cache data type
            k_scale: Optional key quantization scale
            v_scale: Optional value quantization scale
        """
        # Always check for fake tensors
        if not (has_real_storage(key) and has_real_storage(value) and has_real_storage(slot)):
            # Fake tensor - always direct write (during warmup)
            if self.mode == "staging":
                logger.debug(f"NWOR: Fake tensor detected, falling back to direct write")
                self.fallback_count += 1
            # Direct write (skip completely for fake tensors during warmup)
            return

        if self.mode == "staging":
            logger.debug(f"NWOR: write() in staging mode - token_idx={token_idx}, buffer_exists={self.buffer is not None}")
            # Lazy buffer creation: allocate on first real KV write
            if self.buffer is None:
                logger.info(f"NWOR: Creating staging buffer with dtype={key.dtype}, device={key.device}")
                self.buffer = StagingBuffer(
                    n_layers=self.n_layers,
                    max_tokens=max(self.window_capacity, 1),
                    n_heads=self.n_heads,
                    head_dim=self.head_dim,
                    device=str(key.device),
                    dtype=key.dtype,  # Use actual KV dtype (float16/bfloat16)
                )
            # Detect layer boundary via token index reset (handles true wrap-around only)
            if token_idx < self.min_token_idx:
                self.min_token_idx = token_idx
            if token_idx <= self.last_token_idx and token_idx <= self.min_token_idx:
                self.current_layer_idx += 1
                logger.debug(f"NWOR: New layer {self.current_layer_idx} (token reset {self.last_token_idx}→{token_idx}, min={self.min_token_idx})")

            # Handle first real write (token_idx > 0, current_layer_idx still -1)
            if self.current_layer_idx < 0:
                self.current_layer_idx = 0
                logger.debug("NWOR: First real write, initializing layer index to 0")

            # Use auto-detected layer index (ignore passed parameter)
            actual_layer_idx = self.current_layer_idx

            try:
                # Pass all cache parameters - stage() will store LayerRef on first call
                self.buffer.stage(
                    actual_layer_idx, token_idx, key, value, slot,
                    kv_cache_ops, key_cache, value_cache, kv_cache_dtype,
                    k_scale, v_scale
                )
                self.total_staged += 1
                self.last_token_idx = token_idx  # Update for next iteration
            except Exception as e:
                logger.warning(f"NWOR: Stage failed: {e}, falling back to direct write")
                self.disable_staging()
                self.fallback_count += 1
                # Fall through to direct write
                kv_cache_ops.reshape_and_cache_flash(
                    key, value, key_cache, value_cache, slot,
                    kv_cache_dtype, k_scale, v_scale
                )
        else:
            # Direct write
            kv_cache_ops.reshape_and_cache_flash(
                key, value, key_cache, value_cache, slot,
                kv_cache_dtype, k_scale, v_scale
            )

    def commit_window(self, accepted_len: int, proposed_len: int) -> None:
        """
        Commit accepted tokens after rejection sampling.

        Args:
            accepted_len: Number of accepted tokens
            proposed_len: Number of proposed/draft tokens
        """
        if self.mode != "staging" or self.buffer is None:
            return

        try:
            # Buffer.commit() now handles everything internally using stored LayerRefs
            committed = self.buffer.commit(accepted_len)

            self.total_committed += committed
            self.total_rejected += max(proposed_len - committed, 0)

            # Reset consecutive failures on ANY successful commit
            if committed > 0:
                self.consecutive_failures = 0
            else:
                # Commit returned 0 (validation failed, incomplete staging, etc.)
                self.consecutive_failures += 1

            if proposed_len > 0:
                acceptance_rate = 100 * committed / proposed_len
                logger.info(f"NWOR: Committed {committed}/{proposed_len} tokens "
                           f"(acceptance={acceptance_rate:.1f}%)")

        except Exception as e:
            logger.error(f"NWOR: Commit failed: {e}")
            self.fallback_count += 1
            self.consecutive_failures += 1
        finally:
            self.disable_staging()

    def disable_staging(self):
        """Return to direct write mode."""
        logger.info(f"NWOR: Disabling staging mode (was in mode={self.mode}, buffer={self.buffer is not None})")
        self.mode = "direct"
        if self.buffer:
            self.buffer.reset()
        self.last_token_idx = -1  # Reset token tracking
        self.min_token_idx = float('inf')
        self.active_window_tokens = 0  # Clear window size for next window

    def get_metrics(self) -> dict:
        """Get current metrics."""
        total = self.total_committed + self.total_rejected
        acceptance_rate = self.total_committed / total if total > 0 else 0.0

        return {
            "nwor_total_staged": self.total_staged,
            "nwor_total_committed": self.total_committed,
            "nwor_total_rejected": self.total_rejected,
            "nwor_acceptance_rate": acceptance_rate,
            "nwor_fallback_count": self.fallback_count,
            "nwor_unique_tokens": self.buffer.unique_tokens() if self.buffer else 0,
            "nwor_stage_operations": self.buffer.stage_count if self.buffer else 0
        }

    def __del__(self):
        """DIAGNOSTIC: Track interceptor destruction."""
        try:
            logger.info(f"DIAGNOSTIC: KVCacheInterceptor.__del__() called - mode={getattr(self, 'mode', 'unknown')}, "
                       f"buffer={getattr(self, 'buffer', None) is not None}, "
                       f"ready={getattr(self, 'ready', 'unknown')}")
        except Exception as e:
            # Don't let __del__ errors propagate
            logger.error(f"DIAGNOSTIC: Error in KVCacheInterceptor.__del__(): {e}")
