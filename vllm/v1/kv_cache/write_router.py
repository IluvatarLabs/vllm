"""
KV Write Router for vLLM v0.11+ - NWOR Implementation
Routes KV cache writes through ShadowKV during speculative verification
"""

from typing import Optional
import torch
import vllm._custom_ops as ops  # reshape_and_cache_flash is here


class PersistentKVWriter:
    """
    Adapter for vLLM v0.10.2 KV cache operations.
    Uses the same reshape_and_cache_flash op that the backend uses.
    """

    def __init__(self, kv_cache_manager, kv_cache_dtype: torch.dtype):
        """
        Args:
            kv_cache_manager: v1 KVCacheManager instance from engine
            kv_cache_dtype: dtype used by cache ops (fp16/bf16/fp8)
        """
        self.mgr = kv_cache_manager
        self.kv_cache_dtype = kv_cache_dtype

    def get_kv_cache_tensors(self, layer_idx: int):
        """Get key and value cache tensors for a specific layer."""
        # Access the cache tensors from the manager
        # The exact attribute depends on your v0.10.2 build
        if hasattr(self.mgr, 'key_caches') and hasattr(self.mgr, 'value_caches'):
            return self.mgr.key_caches[layer_idx], self.mgr.value_caches[layer_idx]
        elif hasattr(self.mgr, 'kv_cache'):
            # Some builds use a combined structure
            kv_cache = self.mgr.kv_cache[layer_idx]
            return kv_cache[0], kv_cache[1]  # [key_cache, value_cache]
        else:
            raise AttributeError(f"Cannot find KV cache tensors in manager: {dir(self.mgr)}")

    @torch.no_grad()
    def append_slice(self,
                     layer_idx: int,
                     k_slice: torch.Tensor,     # [1, H, D]
                     v_slice: torch.Tensor,     # [1, H, D]
                     slot_mapping_1t: torch.Tensor):
        """
        Single-timestep append using the same op the backend uses.
        Uses reshape_and_cache_flash from v0.10.2.
        """
        # Get layer KV cache tensors
        key_cache, value_cache = self.get_kv_cache_tensors(layer_idx)

        # Call the fused cache writer used by flash-attn backends
        ops.reshape_and_cache_flash(
            k_slice,
            v_slice,
            key_cache,
            value_cache,
            slot_mapping_1t.flatten(),  # Flatten for the op
            self.kv_cache_dtype,
            None,  # k_scale for quantized KV
            None   # v_scale for quantized KV
        )

    @torch.no_grad()
    def append_run(self,
                   layer_idx: int,
                   K_run: torch.Tensor,       # [T, H, D]
                   V_run: torch.Tensor,       # [T, H, D]
                   slot_mapping_run: torch.Tensor):
        """
        Coalesced commit for accepted prefix [T, H, D].
        Uses the same reshape_and_cache_flash for bulk write.
        """
        key_cache, value_cache = self.get_kv_cache_tensors(layer_idx)

        # Bulk write using the same op
        ops.reshape_and_cache_flash(
            K_run,
            V_run,
            key_cache,
            value_cache,
            slot_mapping_run.flatten(),  # Flatten for the op
            self.kv_cache_dtype,
            None,
            None
        )


class KVWriteRouter:
    """
    Routes KV writes either directly to persistent cache (immediate mode),
    or to a ShadowKV staging buffer (defer mode) during verify windows.
    """

    def __init__(self, persistent_writer: PersistentKVWriter):
        """
        Args:
            persistent_writer: Adapter to the actual vLLM cache
        """
        self._persistent = persistent_writer
        self._shadow = None
        self._mode = "immediate"  # or "defer"

    def immediate(self):
        """Switch to immediate write mode (normal operation)."""
        self._mode = "immediate"
        self._shadow = None

    def defer(self, shadow):
        """
        Switch to deferred write mode (NWOR during verification).

        Args:
            shadow: ShadowKV instance to stage writes
        """
        self._mode = "defer"
        self._shadow = shadow

    @torch.no_grad()
    def write(self,
              layer_idx: int,
              t: int,
              k_slice: torch.Tensor,
              v_slice: torch.Tensor,
              slot_mapping_1t: torch.Tensor):
        """
        Route a single timestep KV write.

        Args:
            layer_idx: Transformer layer index
            t: Position in the staging buffer
            k_slice: Key tensor [1, H, D]
            v_slice: Value tensor [1, H, D]
            slot_mapping_1t: Slot mapping for this timestep
        """
        if self._mode == "immediate" or self._shadow is None:
            # Direct write to persistent cache
            self._persistent.append_slice(layer_idx, k_slice, v_slice, slot_mapping_1t)
        else:
            # Stage in shadow buffer for later commit
            self._shadow.stage(layer_idx, t, k_slice, v_slice, slot_mapping_1t)

    @torch.no_grad()
    def commit(self, accepted_len: int):
        """
        Commit accepted tokens from shadow buffer to persistent cache.
        Only called in defer mode after verification.

        Args:
            accepted_len: Number of accepted tokens to commit
        """
        if self._mode == "defer" and self._shadow is not None:
            self._shadow.commit_to(self._persistent, accepted_len)

    def get_persistent_writer(self):
        """Get the underlying persistent writer for direct access if needed."""
        return self._persistent