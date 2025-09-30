"""
ShadowKV - Transient KV buffer for NWOR (No-Write-On-Reject) optimization
Stages speculative KV writes during verification, commits only accepted tokens
"""

from typing import List, Optional
import torch
import logging
import sys

logger = logging.getLogger(__name__)

# Import FakeTensor to detect fake tensors explicitly
try:
    from torch._subclasses.fake_tensor import FakeTensor
except Exception:
    FakeTensor = ()

# Import PyTorch's internal FakeTensor detection (PyTorch >= 2.1)
try:
    from torch._C import _is_fake_tensor as _torch_is_fake_tensor
except Exception:
    _torch_is_fake_tensor = None


def _is_fake_tensor(t: torch.Tensor) -> bool:
    """Detect FakeTensors using multiple methods for robustness."""
    # Direct isinstance check
    if isinstance(t, FakeTensor) or t.__class__.__name__ == "FakeTensor":
        return True
    # Use PyTorch's internal function if available
    if _torch_is_fake_tensor is not None:
        try:
            if _torch_is_fake_tensor(t):
                return True
        except TypeError:
            pass
    return False


def _tensor_has_storage(tensor: torch.Tensor) -> bool:
    """Return False for fake/meta tensors that can't expose a data pointer."""
    if not isinstance(tensor, torch.Tensor):
        return False
    # Check for FakeTensor using robust detection (KEY FIX: catches PyTorch 2.3 FakeTensors)
    if _is_fake_tensor(tensor):
        return False
    # Check for meta tensors
    if tensor.is_meta:
        return False
    try:
        tensor.data_ptr()
        return True
    except (RuntimeError, NotImplementedError, AssertionError) as exc:
        msg = str(exc)
        if "doesn't have storage" in msg or "meta tensor" in msg or "FakeTensor" in msg:
            return False
        raise


# Global registry for per-process shadow_kv instance
_local_shadow_kv: Optional["ShadowKV"] = None


def set_local_shadow_kv(shadow_kv: Optional["ShadowKV"]) -> None:
    """Set the per-process local shadow_kv instance."""
    global _local_shadow_kv
    _local_shadow_kv = shadow_kv


def get_local_shadow_kv() -> Optional["ShadowKV"]:
    """Get the per-process local shadow_kv instance."""
    return _local_shadow_kv


class ShadowKV:
    """
    Transient per-layer staging buffers for the current verify window.
    Stages up to max_chunk tokens, then coalesced-commit the accepted prefix.
    """

    def __init__(self,
                 n_layers: int,
                 n_heads: int,
                 head_dim: int,
                 max_chunk: int,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        """
        Initialize ShadowKV staging buffers.

        Args:
            n_layers: Number of transformer layers
            n_heads: Number of KV heads
            head_dim: Dimension per head
            max_chunk: Maximum tokens to stage (should be >= num_speculative_tokens)
            device: Device to allocate buffers on
            dtype: Data type for KV tensors
        """
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_chunk = max_chunk
        self.device = device
        self.dtype = dtype

        # Pre-allocate staging buffers for each layer. During CUDA graph capture
        # these may be lazy tensors; we will materialize them after warmup.
        self._K: List[torch.Tensor] = [
            torch.empty((max_chunk, n_heads, head_dim), device=device, dtype=dtype)
            for _ in range(n_layers)
        ]
        self._V: List[torch.Tensor] = [
            torch.empty((max_chunk, n_heads, head_dim), device=device, dtype=dtype)
            for _ in range(n_layers)
        ]

        # Store slot mappings for each staged token
        # These will be concatenated for the commit operation
        self._slot_mappings: List[List[torch.Tensor]] = [
            [] for _ in range(n_layers)
        ]

        # Track whether buffers have been materialized post-warmup.
        self._materialized = False

        # Current staged length
        self._len = 0

        # Metrics for monitoring
        self._total_staged = 0
        self._total_committed = 0
        self._total_rejected = 0
        self._debug_stage_calls = 0  # Track if staging is actually happening

        logger.info(
            "Initialized ShadowKV: %d layers, %d heads, %d head_dim, max_chunk=%d",
            n_layers, n_heads, head_dim, max_chunk
        )

    @torch.no_grad()
    def begin(self, length_hint: int):
        """
        Begin staging for a new verification chunk.

        Args:
            length_hint: Expected number of tokens to stage
        """
        if length_hint > self.max_chunk:
            raise ValueError(
                f"ShadowKV max_chunk={self.max_chunk}, got length_hint={length_hint}"
            )
        self._len = 0
        # Clear slot mappings from previous round
        for layer_slots in self._slot_mappings:
            layer_slots.clear()

        logger.debug("ShadowKV: Beginning staging for %d tokens", length_hint)

    def materialize(self) -> None:
        """Force real device buffers after CUDA graph warmup."""
        if self._materialized:
            return

        shape = (self.max_chunk, self.n_heads, self.head_dim)
        logger.info("ShadowKV: materializing %d layers (%s) with real tensors",
                    self.n_layers, self.device)
        for idx in range(self.n_layers):
            self._K[idx] = torch.zeros(shape, device=self.device, dtype=self.dtype)
            self._V[idx] = torch.zeros(shape, device=self.device, dtype=self.dtype)

        self._materialized = True

    @torch.no_grad()
    def stage(self,
              layer_idx: int,
              t: int,
              k_slice: torch.Tensor,
              v_slice: torch.Tensor,
              slot_mapping_1t: torch.Tensor):
        """
        Stage a single token's KV and slot mapping to transient buffer.

        Args:
            layer_idx: Which transformer layer
            t: Position in the staging buffer
            k_slice: Key tensor [1, H, D]
            v_slice: Value tensor [1, H, D]
            slot_mapping_1t: Slot mapping for this single timestep
        """
        # Handle both [1, H, D] and [H, D] shapes
        if k_slice.dim() == 2:
            k_slice = k_slice.unsqueeze(0)
        if v_slice.dim() == 2:
            v_slice = v_slice.unsqueeze(0)

        # Emit STAGING marker on first write per step
        if not getattr(self, "_staging_marked", False):
            import sys
            print(f"🔴 SHADOW: STAGING layer={layer_idx} t={t}", file=sys.stderr, flush=True)
            self._staging_marked = True

        # Skip staging entirely if KV slices are still fake (e.g. during torch.compile capture).
        if not (_tensor_has_storage(k_slice) and _tensor_has_storage(v_slice)):
            logger.info("ShadowKV: skipping staging for fake KV slice (layer=%d, t=%d)",
                        layer_idx, t)
            return

        # Copy KV to staging buffer now that the slices are confirmed real
        self._K[layer_idx][t:t+1].copy_(k_slice)
        self._V[layer_idx][t:t+1].copy_(v_slice)

        # Normalize slot mapping to int64 on the staging device (matches cache op).
        # Force a real owning tensor (copy=True) so we never hold on to fake/meta views.
        try:
            slot_gpu = slot_mapping_1t.to(dtype=torch.int32,
                                          device=self.device,
                                          copy=True).contiguous()
        except (RuntimeError, NotImplementedError) as exc:
            msg = str(exc)
            if "doesn't have storage" in msg or "meta tensor" in msg:
                logger.debug("ShadowKV: skipping staging for fake slot mapping on layer %d", layer_idx)
                return
            raise

        # Append to this layer's slot mappings
        if len(self._slot_mappings[layer_idx]) <= t:
            # Extend list if needed
            self._slot_mappings[layer_idx].extend([None] * (t + 1 - len(self._slot_mappings[layer_idx])))
        self._slot_mappings[layer_idx][t] = slot_gpu.contiguous()

        if t + 1 > self._len:
            self._len = t + 1
            self._total_staged += 1
            self._debug_stage_calls += 1
            if self._debug_stage_calls == 1:
                print(f"🔴 SHADOW: STAGING TOKENS (first call, layer={layer_idx}, t={t})",
                      file=sys.stderr, flush=True)
                logger.info("[NWOR DEBUG] First stage() call - NWOR is active!")

    @torch.no_grad()
    def commit_to(self, persistent_writer, accepted_len: int):
        """
        Commit accepted prefix to persistent storage.

        Args:
            persistent_writer: PersistentKVWriter instance
            accepted_len: Number of tokens accepted (rest are rejected)
        """
        # Reset staging marker for next step
        self._staging_marked = False

        staged_tokens = self._len

        accepted_len = min(accepted_len, staged_tokens)

        # Short-circuit if nothing was staged
        if staged_tokens == 0:
            if accepted_len:
                logger.debug("ShadowKV: commit requested but staging buffer empty")
            return

        if persistent_writer is None:
            rejected = staged_tokens
            self._total_rejected += rejected
            self._len = 0
            print(f"🔴 SHADOW: REJECTING ALL {rejected} STAGED TOKENS", file=sys.stderr, flush=True)
            logger.info("ShadowKV: Writer missing; rejected all %d staged tokens",
                        rejected)
            return

        if accepted_len <= 0:
            rejected = staged_tokens
            self._total_rejected += rejected
            self._len = 0
            print(f"🔴 SHADOW: REJECTING ALL {rejected} STAGED TOKENS", file=sys.stderr, flush=True)
            logger.info("ShadowKV: Rejected all %d staged tokens", rejected)
            return

        print(f"🔴 SHADOW: COMMITTING {accepted_len} OF {self._len} STAGED TOKENS",
              file=sys.stderr, flush=True)

        # Commit accepted tokens to persistent storage layer by layer
        for layer_idx in range(self.n_layers):
            # Get accepted KV slices - make them contiguous
            K_accepted = self._K[layer_idx][:accepted_len].contiguous()
            V_accepted = self._V[layer_idx][:accepted_len].contiguous()

            layer_slots = self._slot_mappings[layer_idx][:accepted_len]

            if len(layer_slots) != accepted_len or not layer_slots or not all(s is not None for s in layer_slots):
                logger.warning(
                    "ShadowKV: Missing slot mappings for layer %d (have %d, expected %d); skipping",
                    layer_idx, len(layer_slots), accepted_len)
                continue

            if layer_slots[0].dim() == 0:
                slot_mapping_run = torch.stack(layer_slots)
            elif layer_slots[0].dim() == 1:
                slot_mapping_run = torch.cat(layer_slots)
            else:
                slot_mapping_run = torch.cat([s.flatten() for s in layer_slots])

            slot_mapping_run = slot_mapping_run.to(
                device=self.device,
                dtype=torch.int32,
                non_blocking=False,
                copy=True).contiguous()

            try:
                persistent_writer.append_run(
                    layer_idx,
                    K_accepted,
                    V_accepted,
                    slot_mapping_run
                )
            except RuntimeError as err:
                logger.warning("ShadowKV: append_run failed on layer %d: %s", layer_idx, err)
                continue

        # Update metrics
        rejected = max(0, staged_tokens - accepted_len)
        self._total_committed += accepted_len
        if rejected:
            self._total_rejected += rejected
        self._len = 0

        logger.info(
            "ShadowKV: Committed %d tokens, rejected %d tokens",
            accepted_len, rejected
        )

    # REMOVED: memory_saved_mb property
    # This was a fake metric - actual memory bandwidth must be measured
    # with Nsight Compute using dram__bytes_write.sum in NVTX ranges

    @property
    def acceptance_rate(self) -> float:
        """Calculate acceptance rate from staged tokens."""
        total = self._total_committed + self._total_rejected
        if total == 0:
            return 0.0
        return self._total_committed / total

    def get_metrics(self) -> dict:
        """Get current metrics for monitoring."""
        return {
            "total_staged": self._total_staged,
            "total_committed": self._total_committed,
            "total_rejected": self._total_rejected,
            "acceptance_rate": self.acceptance_rate,
            # NOTE: memory_saved_mb removed - use Nsight Compute for real measurements
        }

    def reset_metrics(self):
        """Reset metrics counters."""
        self._total_staged = 0
        self._total_committed = 0
        self._total_rejected = 0

    def __repr__(self):
        return (
            f"ShadowKV(layers={self.n_layers}, heads={self.n_heads}, "
            f"dim={self.head_dim}, max_chunk={self.max_chunk}, "
            f"acceptance={self.acceptance_rate:.1%})"
        )
