"""
ShadowKV - Transient KV buffer for NWOR (No-Write-On-Reject) optimization
Stages speculative KV writes during verification, commits only accepted tokens
"""

from typing import List, Optional
import torch
import logging
import sys

logger = logging.getLogger(__name__)

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

        # Pre-allocate staging buffers for each layer
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

        # Copy KV to staging buffer
        self._K[layer_idx][t:t+1].copy_(k_slice)
        self._V[layer_idx][t:t+1].copy_(v_slice)

        # Store slot mapping for this token
        # Ensure it's on the right device
        if slot_mapping_1t.device != self.device:
            slot_mapping_1t = slot_mapping_1t.to(self.device)

        # Append to this layer's slot mappings
        if len(self._slot_mappings[layer_idx]) <= t:
            # Extend list if needed
            self._slot_mappings[layer_idx].extend([None] * (t + 1 - len(self._slot_mappings[layer_idx])))
        self._slot_mappings[layer_idx][t] = slot_mapping_1t

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

        if accepted_len <= 0:
            # All rejected - just reset
            rejected = self._len
            self._total_rejected += rejected
            self._len = 0
            print(f"🔴 SHADOW: REJECTING ALL {rejected} STAGED TOKENS", file=sys.stderr, flush=True)
            logger.info("ShadowKV: Rejected all %d staged tokens", rejected)
            return

        print(f"🔴 SHADOW: COMMITTING {accepted_len} OF {self._len} STAGED TOKENS",
              file=sys.stderr, flush=True)
        # Commit accepted tokens to persistent storage
        for layer_idx in range(self.n_layers):
            # Get accepted KV slices
            K_accepted = self._K[layer_idx][:accepted_len]
            V_accepted = self._V[layer_idx][:accepted_len]

            # Build concatenated slot mapping for accepted tokens
            layer_slots = self._slot_mappings[layer_idx][:accepted_len]

            # Concatenate slot mappings for this layer's accepted tokens
            if all(s is not None for s in layer_slots):
                # Different builds may have different slot mapping shapes
                # Handle both 1D and 2D cases
                if layer_slots[0].dim() == 0:
                    # Scalar slot indices - stack them
                    slot_mapping_run = torch.stack(layer_slots)
                elif layer_slots[0].dim() == 1:
                    # Already 1D arrays - concatenate
                    slot_mapping_run = torch.cat(layer_slots)
                else:
                    # 2D or higher - flatten and concatenate
                    slot_mapping_run = torch.cat([s.flatten() for s in layer_slots])

                # Commit to persistent cache
                persistent_writer.append_run(
                    layer_idx,
                    K_accepted,
                    V_accepted,
                    slot_mapping_run
                )
            else:
                logger.warning(
                    "Missing slot mappings for layer %d, skipping commit",
                    layer_idx
                )

        # Update metrics
        rejected = self._len - accepted_len
        self._total_committed += accepted_len
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