"""
KV Write Router for vLLM v0.11+ - NWOR Implementation
Routes KV cache writes through ShadowKV during speculative verification
"""

from typing import Optional
import logging
import torch
import vllm._custom_ops as ops  # reshape_and_cache_flash is here

logger = logging.getLogger(__name__)

# Import FakeTensor to detect fake tensors explicitly
try:
    from torch._subclasses.fake_tensor import FakeTensor
except Exception:  # pragma: no cover
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
    """Return False for Fake/Meta tensors that can't expose a data pointer."""
    if not isinstance(tensor, torch.Tensor):
        return False
    # Check for FakeTensor using robust detection
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
        mgr = self.mgr

        # Modern path: engine hands us the runner's kv_caches list directly.
        if isinstance(mgr, (list, tuple)):
            kv_cache = mgr[layer_idx]

            # Torch tensor packing keys/values along the first dimension.
            if isinstance(kv_cache, torch.Tensor):
                if kv_cache.dim() == 0:
                    raise ValueError("KV cache tensor is scalar; expected stacked K/V tensors")

                if kv_cache.shape[0] == 2:
                    key_cache, value_cache = kv_cache.unbind(0)
                    return key_cache, value_cache

                # Fallback: split the leading dimension in half.
                if kv_cache.shape[0] % 2 == 0:
                    mid = kv_cache.shape[0] // 2
                    return kv_cache[:mid], kv_cache[mid:]

                raise ValueError(
                    f"Unrecognized KV cache tensor layout for layer {layer_idx}: "
                    f"shape={tuple(kv_cache.shape)}")

            # Some builds may already expose (key, value) tuples/lists per layer.
            if isinstance(kv_cache, (list, tuple)) and len(kv_cache) >= 2:
                return kv_cache[0], kv_cache[1]

            raise AttributeError(
                f"Cannot interpret KV cache entry for layer {layer_idx}: "
                f"type={type(kv_cache)}")

        # Legacy paths: manager objects with explicit attributes.
        if hasattr(mgr, 'key_caches') and hasattr(mgr, 'value_caches'):
            return mgr.key_caches[layer_idx], mgr.value_caches[layer_idx]
        if hasattr(mgr, 'kv_cache'):
            # Some builds use a combined structure
            kv_cache = mgr.kv_cache[layer_idx]
            return kv_cache[0], kv_cache[1]  # [key_cache, value_cache]

        raise AttributeError(f"Cannot find KV cache tensors in manager: {dir(mgr)}")

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

        # Flatten slot mapping before checking (flattened view might be fake even if original isn't)
        slot_mapping_flat = slot_mapping_1t.flatten()

        # Skip if ANY tensor is fake (during warmup/compilation)
        tensors_ok = (
            _tensor_has_storage(key_cache)
            and _tensor_has_storage(value_cache)
            and _tensor_has_storage(k_slice)
            and _tensor_has_storage(v_slice)
            and _tensor_has_storage(slot_mapping_flat)
        )
        if not tensors_ok:
            logger.debug("PersistentKVWriter.append_slice: skipping write with fake tensors on layer %d", layer_idx)
            return

        # Call the fused cache writer used by flash-attn backends
        ops.reshape_and_cache_flash(
            k_slice,
            v_slice,
            key_cache,
            value_cache,
            slot_mapping_flat,
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

        # Flatten slot mapping before checking (flattened view might be fake even if original isn't)
        slot_mapping_flat = slot_mapping_run.flatten()

        # Skip if ANY tensor is fake (during warmup/compilation)
        tensors_ok = (
            _tensor_has_storage(key_cache)
            and _tensor_has_storage(value_cache)
            and _tensor_has_storage(K_run)
            and _tensor_has_storage(V_run)
            and _tensor_has_storage(slot_mapping_flat)
        )
        if not tensors_ok:
            logger.debug("PersistentKVWriter.append_run: skipping write with fake tensors on layer %d", layer_idx)
            return

        # Bulk write using the same op
        ops.reshape_and_cache_flash(
            K_run,
            V_run,
            key_cache,
            value_cache,
            slot_mapping_flat,
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
        self._slot_mapping = None  # Stored during begin() for use in stage()

    def immediate(self):
        """Switch to immediate write mode (normal operation)."""
        self._mode = "immediate"
        self._shadow = None

    def defer(self, shadow):
        """
        Switch to deferred write mode (NWOR during verification).

        Args:
            shadow: ShadowKV instance to stage writes

        Raises:
            RuntimeError: If shadow is None
        """
        if shadow is None:
            raise RuntimeError("KVWriteRouter.defer() requires a ShadowKV instance")
        self._mode = "defer"
        self._shadow = shadow

    def is_deferred(self) -> bool:
        """
        Check if router is in deferred mode with valid shadow buffer.

        Returns:
            True if router is armed for NWOR staging
        """
        return self._mode == "defer" and self._shadow is not None

    @torch.no_grad()
    def begin(self, length_hint: int, slot_mapping: torch.Tensor, seg_lens: Optional[torch.Tensor] = None):
        """
        Begin staging for a verification window.
        Called by flash_attn backend before staging tokens.

        Args:
            length_hint: Expected number of tokens to stage
            slot_mapping: Slot mapping tensor for all tokens in this window
            seg_lens: Segment lengths (optional, for context)
        """
        if self._mode == "defer" and self._shadow is not None:
            # Store slot_mapping for use in stage() calls
            self._slot_mapping = slot_mapping
            # Initialize shadow buffer for this verification window
            self._shadow.begin(length_hint)

    @torch.no_grad()
    def stage(self, layer_idx: int, t: int, k_slice: torch.Tensor, v_slice: torch.Tensor):
        """
        Stage a single timestep's KV during verification.
        Called by flash_attn backend for each token being verified.

        Args:
            layer_idx: Transformer layer index
            t: Position in the staging buffer (0-indexed)
            k_slice: Key tensor [1, H, D]
            v_slice: Value tensor [1, H, D]
        """
        if self._mode == "defer" and self._shadow is not None:
            # Extract slot mapping for this specific timestep
            if self._slot_mapping is not None:
                slot_t = self._slot_mapping[t:t+1]
            else:
                # Fallback: create a dummy slot mapping
                slot_t = torch.tensor([t], dtype=torch.int64, device=k_slice.device)

            # Stage in shadow buffer
            self._shadow.stage(layer_idx, t, k_slice, v_slice, slot_t)
        elif self._mode == "immediate":
            # In immediate mode, write directly to persistent cache
            if self._slot_mapping is not None:
                slot_t = self._slot_mapping[t:t+1]
            else:
                slot_t = torch.tensor([t], dtype=torch.int64, device=k_slice.device)
            self._persistent.append_slice(layer_idx, k_slice, v_slice, slot_t)

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

    def current_writer(self):
        """Get the current active writer (persistent or shadow)."""
        if self._mode == "defer" and self._shadow is not None:
            return self._shadow
        return self._persistent
