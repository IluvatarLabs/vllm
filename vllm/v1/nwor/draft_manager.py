"""Draft Commit Manager for NWOR.

This module implements the draft commit kernel approach to NWOR, which achieves
minimal overhead by:
1. Staging: Store pointers to KV tensors during forward pass (no copies)
2. Commit: Launch single CUDA kernel per layer to scatter accepted tokens

Correctness first: All 10 issues from initial design review are addressed.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class CacheLayout(IntEnum):
    """KV cache memory layout types."""
    FLASH = 0  # [num_blocks, block_size, num_heads, head_size]
    PAGED = 1  # [num_blocks, num_heads, block_size, head_size]


@dataclass
class DraftEntry:
    """Complete draft entry with all fields for kernel and fallback.

    Addresses Issue #2: Added all missing fields including cache targets,
    scale references, and layout information.
    """
    # Draft tensors (sources)
    key_ptr: int
    value_ptr: int
    num_tokens: int
    num_heads: int
    head_size: int

    # Cache tensors (destinations) - Issue #1 fix
    key_cache_ptr: int
    value_cache_ptr: int
    block_size: int
    block_stride: int
    page_stride: int
    head_stride: int

    # Slot mapping
    slot_ptr: int

    # Quantization
    k_scale_ptr: int  # 0 if None
    v_scale_ptr: int
    scale_is_per_token: bool

    # Layout (enum, not string) - Issue #2 fix
    layout_enum: int  # CacheLayout enum value
    kv_cache_dtype: str

    # Keep alive - ALL tensors needed for fallback - Issue #2 fix
    _key_ref: torch.Tensor
    _value_ref: torch.Tensor
    _slot_ref: torch.Tensor
    _key_cache_ref: torch.Tensor
    _value_cache_ref: torch.Tensor
    _k_scale_ref: Optional[torch.Tensor]  # NEW: for fallback
    _v_scale_ref: Optional[torch.Tensor]  # NEW: for fallback


class DraftCommitManager:
    """Manages draft buffers for NWOR with correctness-first design.

    Lifecycle (Issue #9):
    - begin(): Prepare for new spec decode window
    - stage_layer(): Record KV pointers for each layer
    - commit(): Launch kernels to scatter accepted tokens
    - cancel(): Clean up without committing (for early exits)
    """

    def __init__(self):
        self.enabled = False
        self.drafts: List[DraftEntry] = []
        self._logged_failure = False  # One-time warning for fallback

    def begin(self, num_draft_tokens: int) -> bool:
        """Begin new window - clear any stale state first (Issue #9)."""
        self.cancel()  # Clean up any previous state

        if num_draft_tokens <= 0:
            return False

        self.enabled = True
        return True

    def cancel(self):
        """Cancel current window without committing (Issue #9)."""
        self.drafts.clear()
        self.enabled = False

    def _validate_tensor(self, tensor: torch.Tensor, name: str) -> bool:
        """Validate tensor has storage (Issue #7: safety checks)."""
        if not isinstance(tensor, torch.Tensor):
            logger.error(f"NWOR: {name} is not a tensor")
            return False
        if tensor.numel() == 0:
            logger.error(f"NWOR: {name} is empty")
            return False
        # Check for storage (mirrors _tensor_has_storage)
        try:
            _ = tensor.data_ptr()
            return True
        except RuntimeError:
            logger.error(f"NWOR: {name} has no storage")
            return False

    def _ensure_int32_slots(
        self,
        slot_mapping: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """Ensure slot_mapping is int32 and contiguous (Issue #6)."""
        if slot_mapping.dtype != torch.int32:
            slot_mapping = slot_mapping.to(dtype=torch.int32, copy=False)
        if not slot_mapping.is_contiguous():
            slot_mapping = slot_mapping.contiguous()
        if slot_mapping.device != device:
            slot_mapping = slot_mapping.to(device=device, copy=False)
        return slot_mapping

    def _detect_layout(
        self,
        key_cache: torch.Tensor,
        block_size: int
    ) -> int:
        """Detect cache layout from tensor shape (Issue #7)."""
        # key_cache shape options:
        # Flash: [num_blocks, block_size, num_heads, head_size]
        # Paged: [num_blocks, num_heads, block_size, head_size]

        if key_cache.shape[1] == block_size:
            return CacheLayout.FLASH
        else:
            return CacheLayout.PAGED

    def stage_layer(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: Optional[torch.Tensor],
        v_scale: Optional[torch.Tensor],
        kv_cache_dtype: str,
    ) -> None:
        """Record draft KV for one layer with full validation (Issue #7)."""
        if not self.enabled:
            return

        # Issue #7: Validate all tensors before staging
        if not all([
            self._validate_tensor(key, "key"),
            self._validate_tensor(value, "value"),
            self._validate_tensor(key_cache, "key_cache"),
            self._validate_tensor(value_cache, "value_cache"),
            self._validate_tensor(slot_mapping, "slot_mapping"),
        ]):
            logger.warning("Invalid tensors, disabling NWOR for this window")
            self.enabled = False
            return

        # Issue #6: Ensure slot_mapping is int32 contiguous
        slot_mapping = self._ensure_int32_slots(slot_mapping, key.device)

        # Compute cache strides
        block_stride = key_cache.stride(0)
        page_stride = key_cache.stride(1)
        head_stride = key_cache.stride(2)
        block_size = key_cache.size(1) if key_cache.shape[1] <= 256 else key_cache.size(2)

        # Issue #7: Detect layout
        layout_enum = self._detect_layout(key_cache, block_size)

        # Determine if scales are per-token or scalar
        scale_is_per_token = False
        if k_scale is not None and k_scale.numel() > 1:
            scale_is_per_token = True

        entry = DraftEntry(
            # Draft tensors
            key_ptr=key.data_ptr(),
            value_ptr=value.data_ptr(),
            num_tokens=key.shape[0],
            num_heads=key.shape[1],
            head_size=key.shape[2],

            # Cache tensors (Issue #1 fix)
            key_cache_ptr=key_cache.data_ptr(),
            value_cache_ptr=value_cache.data_ptr(),
            block_size=block_size,
            block_stride=block_stride,
            page_stride=page_stride,
            head_stride=head_stride,

            # Slot mapping
            slot_ptr=slot_mapping.data_ptr(),

            # Quantization
            k_scale_ptr=k_scale.data_ptr() if k_scale is not None else 0,
            v_scale_ptr=v_scale.data_ptr() if v_scale is not None else 0,
            scale_is_per_token=scale_is_per_token,

            # Layout
            layout_enum=layout_enum,
            kv_cache_dtype=kv_cache_dtype,

            # Keep alive - Issue #2: added scale refs for fallback
            _key_ref=key,
            _value_ref=value,
            _slot_ref=slot_mapping,  # Hold the converted version
            _key_cache_ref=key_cache,
            _value_cache_ref=value_cache,
            _k_scale_ref=k_scale,  # NEW: for fallback
            _v_scale_ref=v_scale,  # NEW: for fallback
        )

        self.drafts.append(entry)

    def _check_contiguous_prefix(
        self,
        mask: torch.Tensor,
        num_accepted: int
    ) -> bool:
        """Check if mask is [True]*N + [False]*M (Issue #1: fast path)."""
        if num_accepted == mask.numel():
            return True  # All accepted
        if num_accepted == 0:
            return True  # All rejected (already handled)

        # Check if first num_accepted are True, rest are False
        # Avoid Python loop - use tensor ops
        prefix = mask[:num_accepted]
        suffix = mask[num_accepted:]
        return prefix.all().item() and not suffix.any().item()

    def _commit_contiguous(
        self,
        entry: DraftEntry,
        num_accepted: int
    ):
        """Fast path: use existing writer for contiguous prefix (Issue #1)."""
        # Source slices
        key_src = entry._key_ref[:num_accepted]
        value_src = entry._value_ref[:num_accepted]
        slot_src = entry._slot_ref[:num_accepted]

        # Slice scales if needed (mirror _slice_scale behavior)
        k_scale_src = None
        v_scale_src = None
        if entry._k_scale_ref is not None:
            k_scale_src = entry._k_scale_ref[:num_accepted] if entry.scale_is_per_token else entry._k_scale_ref
        if entry._v_scale_ref is not None:
            v_scale_src = entry._v_scale_ref[:num_accepted] if entry.scale_is_per_token else entry._v_scale_ref

        # Use existing reshape_and_cache for contiguous slice
        # This is bandwidth-optimal for dense prefix
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key_src,
            value_src,
            entry._key_cache_ref,
            entry._value_cache_ref,
            slot_src,
            entry.kv_cache_dtype,
            k_scale_src,
            v_scale_src,
        )

    def _commit_sparse(
        self,
        entry: DraftEntry,
        mask: torch.Tensor,
        num_accepted: int
    ):
        """Sparse path: kernel scatter for non-contiguous acceptance (Issue #3)."""
        torch.ops._C_cache_ops.commit_draft_layer(
            entry.key_ptr,
            entry.value_ptr,
            entry.key_cache_ptr,
            entry.value_cache_ptr,
            mask.data_ptr(),
            entry.slot_ptr,
            entry.k_scale_ptr,
            entry.v_scale_ptr,
            entry.scale_is_per_token,
            entry.num_tokens,
            entry.num_heads,
            entry.head_size,
            entry.block_size,
            entry.block_stride,
            entry.page_stride,
            entry.head_stride,
            entry.layout_enum,
            entry.kv_cache_dtype,
        )

    def _fallback_commit(self, mask: torch.Tensor):
        """Fallback to vanilla writer for all layers (Issue #6, #9)."""
        accepted_indices = mask.nonzero(as_tuple=False).squeeze(1)

        if accepted_indices.numel() == 0:
            return

        # EVERY entry gets fallback (even if kernel succeeded for some)
        for entry in self.drafts:
            # Extract accepted tokens
            key_accepted = entry._key_ref[accepted_indices]
            value_accepted = entry._value_ref[accepted_indices]
            slot_accepted = entry._slot_ref[accepted_indices]

            # Issue #6: Ensure int32 for fallback too
            slot_accepted = self._ensure_int32_slots(slot_accepted, key_accepted.device)

            # Slice scales if needed (mirror _slice_scale behavior)
            k_scale_fallback = None
            v_scale_fallback = None
            if entry._k_scale_ref is not None:
                k_scale_fallback = entry._k_scale_ref[accepted_indices] if entry.scale_is_per_token else entry._k_scale_ref
            if entry._v_scale_ref is not None:
                v_scale_fallback = entry._v_scale_ref[accepted_indices] if entry.scale_is_per_token else entry._v_scale_ref

            # Use vanilla writer
            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key_accepted,
                value_accepted,
                entry._key_cache_ref,
                entry._value_cache_ref,
                slot_accepted,
                entry.kv_cache_dtype,
                k_scale_fallback,
                v_scale_fallback,
            )

    def commit(self, mask: torch.Tensor) -> int:
        """Commit with mask validation, fast path, and fallback (Issues #1, #7, #9, #10).

        Returns:
            Number of committed tokens.
        """
        if not self.enabled or not self.drafts:
            return 0

        # Issue #7: Validate mask
        if mask.numel() == 0:
            self.drafts.clear()
            self.enabled = False
            return 0

        # Issue #5: Assert all layers have identical token counts
        expected_tokens = self.drafts[0].num_tokens
        for i, entry in enumerate(self.drafts):
            assert entry.num_tokens == expected_tokens, \
                f"Layer {i} has {entry.num_tokens} tokens, expected {expected_tokens}"

        assert mask.shape[0] == expected_tokens, \
            f"Mask has {mask.shape[0]} tokens, expected {expected_tokens}"

        # Issue #1: Ensure correct dtype, device, layout
        device = self.drafts[0]._key_ref.device
        if mask.dtype != torch.bool:
            mask = mask.to(dtype=torch.bool)
        if mask.device != device:
            mask = mask.to(device=device)
        if not mask.is_contiguous():
            mask = mask.contiguous()

        # Issue #1: Cache mask count ONCE (single GPU→CPU sync)
        num_accepted = int(mask.sum().item())

        if num_accepted == 0:
            self.drafts.clear()
            self.enabled = False
            return 0

        # Issue #1: FAST PATH - Contiguous prefix detection
        is_contiguous_prefix = self._check_contiguous_prefix(mask, num_accepted)

        try:
            if is_contiguous_prefix:
                # Fast path: use existing writer for contiguous slice
                for entry in self.drafts:
                    self._commit_contiguous(entry, num_accepted)
            else:
                # Sparse path: kernel scatter
                for entry in self.drafts:
                    self._commit_sparse(entry, mask, num_accepted)

            # Issue #10: Device sync before reuse
            torch.cuda.synchronize()

        except Exception as e:
            # Issue #9: Fallback on any error
            if not self._logged_failure:
                logger.warning(f"Draft commit failed: {e}, falling back to dense write")
                self._logged_failure = True
            self._fallback_commit(mask)

        finally:
            # Issue #9: Always clean up, even on exception
            self.drafts.clear()
            self.enabled = False

        return num_accepted


# Global singleton instance
_draft_manager: Optional[DraftCommitManager] = None


def get_draft_manager() -> DraftCommitManager:
    """Get or create the global draft commit manager singleton."""
    global _draft_manager
    if _draft_manager is None:
        _draft_manager = DraftCommitManager()
    return _draft_manager
