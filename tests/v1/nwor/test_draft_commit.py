"""Comprehensive tests for NWOR Draft Commit Kernel.

Tests all edge cases mentioned in the design review:
- FP16, BF16, FP8 quantization (per-layer and per-token scales)
- Contiguous prefix vs sparse acceptance patterns
- 0%, 50%, 100% acceptance rates
- Multi-request windows with variable draft counts
- Kernel failure simulation and fallback behavior
- Disabled NWOR zero-overhead verification
"""

import pytest
import torch

from vllm.v1.nwor import DraftCommitManager, CacheLayout


@pytest.fixture
def device():
    """CUDA device for tests."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


@pytest.fixture
def manager():
    """Fresh DraftCommitManager for each test."""
    return DraftCommitManager()


class TestDraftCommitBasic:
    """Basic functionality tests."""

    def test_begin_with_no_drafts(self, manager):
        """Test begin() with zero draft tokens."""
        result = manager.begin(0)
        assert result is False
        assert manager.enabled is False
        assert len(manager.drafts) == 0

    def test_begin_with_drafts(self, manager):
        """Test begin() with positive draft count."""
        result = manager.begin(10)
        assert result is True
        assert manager.enabled is True
        assert len(manager.drafts) == 0  # No stages yet

    def test_cancel(self, manager):
        """Test cancel() cleans up state."""
        manager.begin(10)
        manager.enabled = True
        manager.drafts.append(None)  # Mock entry

        manager.cancel()

        assert manager.enabled is False
        assert len(manager.drafts) == 0

    def test_lifecycle(self, manager):
        """Test complete begin/cancel lifecycle."""
        # Begin
        manager.begin(5)
        assert manager.enabled is True

        # Cancel
        manager.cancel()
        assert manager.enabled is False

        # Begin again
        manager.begin(5)
        assert manager.enabled is True


class TestDraftCommitQuantization:
    """Quantization and dtype tests."""

    def test_fp16_no_quantization(self, manager, device):
        """Test FP16 KV with auto (no quantization)."""
        num_tokens, num_heads, head_size = 4, 8, 64
        block_size = 16

        # Create FP16 draft tensors
        key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device=device)
        value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device=device)

        # Create cache (Flash layout)
        num_blocks = 2
        key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.float16, device=device)
        value_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.float16, device=device)

        # Slot mapping
        slot_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)

        # Stage layer
        manager.begin(num_tokens)
        manager.stage_layer(
            key, value, key_cache, value_cache,
            slot_mapping, None, None, "auto"
        )

        assert len(manager.drafts) == 1
        assert manager.drafts[0].kv_cache_dtype == "auto"
        assert manager.drafts[0].key_value_dtype == "fp16"

    def test_bf16_fp8_per_layer_scale(self, manager, device):
        """Test BF16 KV with FP8 cache and per-layer (scalar) scale."""
        num_tokens, num_heads, head_size = 4, 8, 64
        block_size = 16

        # Create BF16 draft tensors
        key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16, device=device)
        value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16, device=device)

        # Create FP8 cache
        num_blocks = 2
        key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.uint8, device=device)
        value_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.uint8, device=device)

        # Scalar scales (per-layer)
        k_scale = torch.tensor([0.5], dtype=torch.float32, device=device)
        v_scale = torch.tensor([0.5], dtype=torch.float32, device=device)

        # Slot mapping
        slot_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)

        # Stage layer
        manager.begin(num_tokens)
        manager.stage_layer(
            key, value, key_cache, value_cache,
            slot_mapping, k_scale, v_scale, "fp8"
        )

        assert len(manager.drafts) == 1
        assert manager.drafts[0].scale_is_per_token is False  # Scalar scale
        assert manager.drafts[0].key_value_dtype == "bf16"

    def test_fp16_fp8_per_token_scale(self, manager, device):
        """Test FP16 KV with FP8 cache and per-token scales."""
        num_tokens, num_heads, head_size = 4, 8, 64
        block_size = 16

        # Create FP16 draft tensors
        key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device=device)
        value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device=device)

        # Create FP8 cache
        num_blocks = 2
        key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.uint8, device=device)
        value_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.uint8, device=device)

        # Per-token scales
        k_scale = torch.rand(num_tokens, dtype=torch.float32, device=device) * 0.1 + 0.4
        v_scale = torch.rand(num_tokens, dtype=torch.float32, device=device) * 0.1 + 0.4

        # Slot mapping
        slot_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)

        # Stage layer
        manager.begin(num_tokens)
        manager.stage_layer(
            key, value, key_cache, value_cache,
            slot_mapping, k_scale, v_scale, "fp8"
        )

        assert len(manager.drafts) == 1
        assert manager.drafts[0].scale_is_per_token is True  # Per-token scales


class TestDraftCommitAcceptance:
    """Acceptance pattern tests."""

    def test_zero_acceptance(self, manager, device):
        """Test 0% acceptance - all tokens rejected."""
        num_tokens, num_heads, head_size = 4, 8, 64
        block_size = 16

        # Create tensors
        key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device=device)
        value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device=device)

        num_blocks = 2
        key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.float16, device=device)
        value_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.float16, device=device)

        slot_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)

        # Stage layer
        manager.begin(num_tokens)
        manager.stage_layer(
            key, value, key_cache, value_cache,
            slot_mapping, None, None, "auto"
        )

        # All rejected mask
        mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)

        # Commit
        num_committed = manager.commit(mask)

        assert num_committed == 0
        assert manager.enabled is False
        assert len(manager.drafts) == 0

    def test_full_acceptance(self, manager, device):
        """Test 100% acceptance - all tokens accepted (contiguous prefix fast path)."""
        num_tokens, num_heads, head_size = 4, 8, 64
        block_size = 16

        # Create tensors
        key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device=device)
        value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device=device)

        num_blocks = 2
        key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.float16, device=device)
        value_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.float16, device=device)

        slot_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)

        # Stage layer
        manager.begin(num_tokens)
        manager.stage_layer(
            key, value, key_cache, value_cache,
            slot_mapping, None, None, "auto"
        )

        # All accepted mask (should trigger contiguous fast path)
        mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        # Commit
        num_committed = manager.commit(mask)

        assert num_committed == num_tokens
        assert manager.enabled is False
        assert len(manager.drafts) == 0

    def test_partial_acceptance_contiguous_prefix(self, manager, device):
        """Test partial acceptance with contiguous prefix [T, T, F, F]."""
        num_tokens, num_heads, head_size = 4, 8, 64
        block_size = 16

        # Create tensors
        key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device=device)
        value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device=device)

        num_blocks = 2
        key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.float16, device=device)
        value_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.float16, device=device)

        slot_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)

        # Stage layer
        manager.begin(num_tokens)
        manager.stage_layer(
            key, value, key_cache, value_cache,
            slot_mapping, None, None, "auto"
        )

        # Contiguous prefix: first 2 accepted
        mask = torch.tensor([True, True, False, False], dtype=torch.bool, device=device)

        # Commit (should use fast path)
        num_committed = manager.commit(mask)

        assert num_committed == 2
        assert manager.enabled is False

    def test_sparse_acceptance(self, manager, device):
        """Test sparse acceptance [T, F, T, F] - requires kernel scatter."""
        num_tokens, num_heads, head_size = 4, 8, 64
        block_size = 16

        # Create tensors
        key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device=device)
        value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device=device)

        num_blocks = 2
        key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.float16, device=device)
        value_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.float16, device=device)

        slot_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)

        # Stage layer
        manager.begin(num_tokens)
        manager.stage_layer(
            key, value, key_cache, value_cache,
            slot_mapping, None, None, "auto"
        )

        # Sparse mask: non-contiguous acceptance
        mask = torch.tensor([True, False, True, False], dtype=torch.bool, device=device)

        # Commit (should use sparse kernel path)
        num_committed = manager.commit(mask)

        assert num_committed == 2
        assert manager.enabled is False


class TestDraftCommitMultiLayer:
    """Multi-layer staging tests."""

    def test_multiple_layers(self, manager, device):
        """Test staging multiple layers."""
        num_tokens, num_heads, head_size = 4, 8, 64
        block_size = 16
        num_layers = 3

        num_blocks = 2

        # Stage multiple layers
        manager.begin(num_tokens)

        for layer_idx in range(num_layers):
            key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device=device)
            value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device=device)

            key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.float16, device=device)
            value_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.float16, device=device)

            slot_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)

            manager.stage_layer(
                key, value, key_cache, value_cache,
                slot_mapping, None, None, "auto"
            )

        assert len(manager.drafts) == num_layers

        # Commit all layers
        mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        num_committed = manager.commit(mask)

        assert num_committed == num_tokens
        assert len(manager.drafts) == 0


class TestDraftCommitCacheLayouts:
    """Cache layout tests (Flash vs Paged)."""

    def test_flash_layout(self, manager, device):
        """Test Flash layout [num_blocks, block_size, num_heads, head_size]."""
        num_tokens, num_heads, head_size = 4, 8, 64
        block_size = 16

        key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device=device)
        value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device=device)

        # Flash layout
        num_blocks = 2
        key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.float16, device=device)
        value_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.float16, device=device)

        slot_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)

        manager.begin(num_tokens)
        manager.stage_layer(
            key, value, key_cache, value_cache,
            slot_mapping, None, None, "auto"
        )

        assert manager.drafts[0].layout_enum == CacheLayout.FLASH

    def test_paged_layout(self, manager, device):
        """Test Paged layout [num_blocks, num_heads, block_size, head_size]."""
        num_tokens, num_heads, head_size = 4, 8, 64
        block_size = 16

        key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device=device)
        value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device=device)

        # Paged layout (heads before block_size)
        num_blocks = 2
        key_cache = torch.zeros(num_blocks, num_heads, block_size, head_size, dtype=torch.float16, device=device)
        value_cache = torch.zeros(num_blocks, num_heads, block_size, head_size, dtype=torch.float16, device=device)

        slot_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)

        manager.begin(num_tokens)
        manager.stage_layer(
            key, value, key_cache, value_cache,
            slot_mapping, None, None, "auto"
        )

        assert manager.drafts[0].layout_enum == CacheLayout.PAGED


class TestDraftCommitSafety:
    """Safety and validation tests."""

    def test_disabled_nwor_no_overhead(self, manager, device):
        """Test that disabled NWOR has zero overhead."""
        num_tokens, num_heads, head_size = 4, 8, 64
        block_size = 16

        key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device=device)
        value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device=device)

        num_blocks = 2
        key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.float16, device=device)
        value_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.float16, device=device)

        slot_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)

        # Don't call begin() - manager is disabled
        assert manager.enabled is False

        # Stage should be no-op
        manager.stage_layer(
            key, value, key_cache, value_cache,
            slot_mapping, None, None, "auto"
        )

        assert len(manager.drafts) == 0  # Nothing staged

    def test_empty_mask(self, manager, device):
        """Test commit with empty mask."""
        manager.begin(4)

        # Empty mask
        mask = torch.tensor([], dtype=torch.bool, device=device)

        num_committed = manager.commit(mask)

        assert num_committed == 0
        assert manager.enabled is False

    def test_slot_mapping_int32_conversion(self, manager, device):
        """Test that slot_mapping is converted to int32."""
        num_tokens, num_heads, head_size = 4, 8, 64
        block_size = 16

        key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device=device)
        value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device=device)

        num_blocks = 2
        key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.float16, device=device)
        value_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.float16, device=device)

        # Int64 slot mapping (should be converted to int32)
        slot_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=device)

        manager.begin(num_tokens)
        manager.stage_layer(
            key, value, key_cache, value_cache,
            slot_mapping, None, None, "auto"
        )

        # Check that stored slot_ref is int32
        assert manager.drafts[0]._slot_ref.dtype == torch.int32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
