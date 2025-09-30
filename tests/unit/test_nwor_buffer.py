"""Unit tests for NWOR StagingBuffer - validates shared slot design."""

import pytest
import torch

from vllm.v1.kv_cache.interceptor import StagingBuffer, has_real_storage


class TestFakeTensorDetection:
    """Test fake tensor detection across PyTorch versions."""

    def test_real_tensor(self):
        """Real tensors should return True."""
        tensor = torch.randn(10, 10, device="cpu")
        assert has_real_storage(tensor)

        if torch.cuda.is_available():
            cuda_tensor = torch.randn(10, 10, device="cuda")
            assert has_real_storage(cuda_tensor)

    def test_meta_tensor(self):
        """Meta tensors should return False."""
        meta_tensor = torch.randn(10, 10, device="meta")
        assert not has_real_storage(meta_tensor)

    def test_non_tensor(self):
        """Non-tensors should return False."""
        assert not has_real_storage(None)
        assert not has_real_storage(42)
        assert not has_real_storage("not a tensor")


class TestStagingBuffer:
    """Test StagingBuffer with emphasis on shared slot semantics."""

    @pytest.fixture
    def buffer(self):
        """Create a test buffer."""
        return StagingBuffer(
            n_layers=4,
            max_tokens=10,
            n_heads=8,
            head_dim=64,
            device="cpu",  # Use CPU for tests
            dtype=torch.float16
        )

    def test_initialization(self, buffer):
        """Test buffer initialization."""
        assert buffer.n_layers == 4
        assert buffer.max_tokens == 10
        assert buffer.k_buffer.shape == (4, 10, 8, 64)
        assert buffer.v_buffer.shape == (4, 10, 8, 64)
        assert buffer.slot_buffer.shape == (10,)
        assert buffer.unique_tokens() == 0
        assert not buffer.is_busy()

    def test_shared_slots_critical(self, buffer):
        """
        CRITICAL TEST: Verify all layers use the same slot mapping.
        This was THE bug in the original implementation.
        """
        # Create different slots for each layer (simulating the bug)
        k = torch.randn(8, 64, dtype=torch.float16)
        v = torch.randn(8, 64, dtype=torch.float16)

        # Stage token 0 with different "attempted" slots per layer
        for layer_idx in range(4):
            # Each layer tries to set a different slot (this was the bug!)
            slot = torch.tensor([layer_idx * 100])  # 0, 100, 200, 300
            buffer.stage(layer_idx, token_idx=0, k_slice=k, v_slice=v,
                        slot_tensor=slot)

        # CRITICAL: Only layer 0's slot should be stored
        assert buffer.slot_buffer[0] == 0, "Slot should be from layer 0 only!"
        # It should NOT be 300 (from layer 3)

    def test_out_of_order_staging(self, buffer):
        """Test non-sequential token staging."""
        k = torch.randn(8, 64, dtype=torch.float16)
        v = torch.randn(8, 64, dtype=torch.float16)

        # Stage tokens in random order: 3, 1, 0, 2
        buffer.stage(0, 3, k, v, torch.tensor(103))
        buffer.stage(0, 1, k, v, torch.tensor(101))
        buffer.stage(0, 0, k, v, torch.tensor(100))
        buffer.stage(0, 2, k, v, torch.tensor(102))

        # All positions should be marked
        assert buffer.token_mask[:4].all()
        assert buffer.unique_tokens() == 4

        # Slots should be in the right positions
        assert buffer.slot_buffer[0] == 100
        assert buffer.slot_buffer[1] == 101
        assert buffer.slot_buffer[2] == 102
        assert buffer.slot_buffer[3] == 103

    def test_stage_count_vs_unique_tokens(self, buffer):
        """Test the distinction between operations and unique tokens."""
        k = torch.randn(8, 64, dtype=torch.float16)
        v = torch.randn(8, 64, dtype=torch.float16)

        # Stage 2 tokens across 4 layers = 8 operations
        for layer_idx in range(4):
            for token_idx in range(2):
                buffer.stage(layer_idx, token_idx, k, v,
                           torch.tensor(token_idx + 100))

        # Should have 8 stage operations but only 2 unique tokens
        assert buffer.stage_count == 8  # 4 layers × 2 tokens
        assert buffer.unique_tokens() == 2  # Only 2 unique positions

    def test_bounds_checking(self, buffer):
        """Test bounds validation."""
        k = torch.randn(8, 64, dtype=torch.float16)
        v = torch.randn(8, 64, dtype=torch.float16)

        # Invalid layer index
        with pytest.raises(ValueError, match="Invalid layer index"):
            buffer.stage(4, 0, k, v, torch.tensor(100))  # layer 4 doesn't exist

        # Invalid token index
        with pytest.raises(ValueError, match="Invalid token index"):
            buffer.stage(0, 10, k, v, torch.tensor(100))  # token 10 out of bounds

    def test_reset(self, buffer):
        """Test buffer reset."""
        k = torch.randn(8, 64, dtype=torch.float16)
        v = torch.randn(8, 64, dtype=torch.float16)

        # Stage some data
        buffer.stage(0, 0, k, v, torch.tensor(100))
        assert buffer.is_busy()
        assert buffer.unique_tokens() == 1

        # Reset
        buffer.reset()
        assert not buffer.is_busy()
        assert buffer.unique_tokens() == 0
        assert buffer.stage_count == 0

    def test_incomplete_staging_rejection(self, buffer):
        """Test that incomplete staging causes rejection."""
        k = torch.randn(8, 64, dtype=torch.float16)
        v = torch.randn(8, 64, dtype=torch.float16)

        # Stage tokens 0, 2, 3 (missing token 1)
        buffer.stage(0, 0, k, v, torch.tensor(100))
        buffer.stage(0, 2, k, v, torch.tensor(102))
        buffer.stage(0, 3, k, v, torch.tensor(103))

        # Mock kv_cache_ops
        class MockOps:
            def reshape_and_cache_flash(self, *args):
                pass

        # Try to commit 4 tokens (but token 1 is missing)
        result = buffer.commit(4, MockOps(),
                             torch.empty(1), torch.empty(1),
                             "float16", None, None)

        # Should reject all due to incomplete staging
        assert result == 0

    def test_successful_commit(self, buffer):
        """Test successful commit with complete staging."""
        k = torch.randn(8, 64, dtype=torch.float16)
        v = torch.randn(8, 64, dtype=torch.float16)

        # Stage 3 tokens completely across all layers
        for layer_idx in range(4):
            for token_idx in range(3):
                buffer.stage(layer_idx, token_idx, k, v,
                           torch.tensor(token_idx + 100))

        # Mock successful kv_cache_ops
        class MockOps:
            def __init__(self):
                self.call_count = 0
                self.received_slots = None

            def reshape_and_cache_flash(self, k, v, k_cache, v_cache,
                                      slots, dtype, k_scale, v_scale):
                self.call_count += 1
                if self.received_slots is None:
                    self.received_slots = slots.clone()
                else:
                    # Verify same slots used for all layers
                    assert torch.equal(slots, self.received_slots), \
                           "Different slots used for different layers!"

        mock_ops = MockOps()
        result = buffer.commit(3, mock_ops,
                             torch.empty(1), torch.empty(1),
                             "float16", None, None)

        # Should succeed and use same slots for all layers
        assert result == 3
        assert mock_ops.call_count == 4  # Called for each layer
        assert mock_ops.received_slots is not None

    def test_all_or_nothing_commit(self, buffer):
        """Test all-or-nothing commit semantics."""
        k = torch.randn(8, 64, dtype=torch.float16)
        v = torch.randn(8, 64, dtype=torch.float16)

        # Stage 2 tokens across all layers
        for layer_idx in range(4):
            for token_idx in range(2):
                buffer.stage(layer_idx, token_idx, k, v,
                           torch.tensor(token_idx + 100))

        # Mock ops that fails on layer 2
        class FailingOps:
            def __init__(self):
                self.call_count = 0

            def reshape_and_cache_flash(self, *args):
                self.call_count += 1
                if self.call_count == 3:  # Fail on 3rd call (layer 2)
                    raise RuntimeError("Simulated failure")

        failing_ops = FailingOps()
        result = buffer.commit(2, failing_ops,
                             torch.empty(1), torch.empty(1),
                             "float16", None, None)

        # Should return 0 (all rejected) due to layer 2 failure
        assert result == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])