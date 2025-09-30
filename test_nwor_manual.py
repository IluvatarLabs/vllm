#!/usr/bin/env python3
"""Manual test of NWOR StagingBuffer - focus on critical shared slot design."""

import sys
import torch
sys.path.append('.')

from vllm.v1.kv_cache.interceptor import StagingBuffer, has_real_storage

def test_shared_slots():
    """CRITICAL TEST: Verify all layers use the same slot mapping."""
    print("Testing shared slots (THE critical fix)...")

    buffer = StagingBuffer(
        n_layers=4,
        max_tokens=10,
        n_heads=8,
        head_dim=64,
        device="cpu",
        dtype=torch.float16
    )

    k = torch.randn(8, 64, dtype=torch.float16)
    v = torch.randn(8, 64, dtype=torch.float16)

    # Each layer tries to set a different slot (simulating the old bug)
    for layer_idx in range(4):
        slot = torch.tensor([layer_idx * 100])  # 0, 100, 200, 300
        buffer.stage(layer_idx, token_idx=0, k_slice=k, v_slice=v, slot_tensor=slot)

    # CRITICAL: Only layer 0's slot should be stored
    actual_slot = int(buffer.slot_buffer[0].item())
    expected_slot = 0

    if actual_slot == expected_slot:
        print(f"✓ PASS: Slot is {actual_slot} (from layer 0 only)")
    else:
        print(f"✗ FAIL: Slot is {actual_slot}, expected {expected_slot}")
        print("  This would cause the 0% acceptance bug!")
        return False

    return True

def test_stage_operations_vs_unique_tokens():
    """Test we track both operations and unique tokens correctly."""
    print("\nTesting stage operations vs unique tokens (solving 4x inflation)...")

    buffer = StagingBuffer(
        n_layers=40,  # Like real model
        max_tokens=48,  # Typical speculation window
        n_heads=8,
        head_dim=64,
        device="cpu",
        dtype=torch.float16
    )

    k = torch.randn(8, 64, dtype=torch.float16)
    v = torch.randn(8, 64, dtype=torch.float16)

    # Stage 48 tokens across 40 layers
    for layer_idx in range(40):
        for token_idx in range(48):
            buffer.stage(layer_idx, token_idx, k, v, torch.tensor(token_idx + 1000))

    operations = buffer.stage_count
    unique = buffer.unique_tokens()
    expected_ops = 40 * 48  # 1920
    expected_unique = 48

    print(f"  Stage operations: {operations} (expected {expected_ops})")
    print(f"  Unique tokens: {unique} (expected {expected_unique})")

    if operations == expected_ops and unique == expected_unique:
        print("✓ PASS: Correctly tracks operations vs unique tokens")
        return True
    else:
        print("✗ FAIL: Incorrect tracking")
        return False

def test_all_or_nothing():
    """Test all-or-nothing commit semantics."""
    print("\nTesting all-or-nothing commit...")

    buffer = StagingBuffer(n_layers=4, max_tokens=10, n_heads=8, head_dim=64,
                          device="cpu", dtype=torch.float16)

    k = torch.randn(8, 64, dtype=torch.float16)
    v = torch.randn(8, 64, dtype=torch.float16)

    # Stage completely
    for layer_idx in range(4):
        for token_idx in range(3):
            buffer.stage(layer_idx, token_idx, k, v, torch.tensor(token_idx + 100))

    # Mock ops that fails on layer 2
    class FailingOps:
        def __init__(self):
            self.call_count = 0

        def reshape_and_cache_flash(self, *args):
            self.call_count += 1
            if self.call_count == 3:  # Fail on layer 2
                raise RuntimeError("Simulated failure")

    failing_ops = FailingOps()
    result = buffer.commit(3, failing_ops, torch.empty(1), torch.empty(1),
                          "float16", None, None)

    if result == 0:
        print("✓ PASS: Rejected all tokens on layer failure (all-or-nothing)")
        return True
    else:
        print(f"✗ FAIL: Returned {result} instead of 0")
        return False

def test_fake_tensor_detection():
    """Test fake tensor detection."""
    print("\nTesting fake tensor detection...")

    # Real tensor
    real = torch.randn(10, 10)
    if has_real_storage(real):
        print("✓ PASS: Real tensor detected")
    else:
        print("✗ FAIL: Real tensor not detected")
        return False

    # Meta tensor
    meta = torch.randn(10, 10, device="meta")
    if not has_real_storage(meta):
        print("✓ PASS: Meta tensor rejected")
    else:
        print("✗ FAIL: Meta tensor not rejected")
        return False

    return True

def main():
    print("=" * 60)
    print("NWOR StagingBuffer Manual Tests")
    print("=" * 60)

    tests = [
        test_shared_slots,
        test_stage_operations_vs_unique_tokens,
        test_all_or_nothing,
        test_fake_tensor_detection
    ]

    passed = sum(test() for test in tests)
    total = len(tests)

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✓ ALL TESTS PASSED - Ready for Phase 2!")
    else:
        print("✗ Some tests failed - Fix before proceeding")
        sys.exit(1)

if __name__ == "__main__":
    main()