#!/usr/bin/env python3
"""Test if PyTorch CUDA graphs use persistent buffers for operation outputs.

This is critical for NWOR's caching strategy: if CUDA graphs allocate new
output buffers on each replay, cached tensor references become invalid.
"""

import torch

print("="*80)
print("CUDA Graph Persistent Buffer Test")
print("="*80)

# Test 1: Basic matmul (QKV projection pattern)
print("\n[Test 1] Basic matmul (simulating QKV projection)")
print("-" * 80)

x = torch.randn(32, 512, device='cuda')  # 32 tokens, 512 hidden
weight = torch.randn(512, 1536, device='cuda')  # project to 3*512 for QKV

# Warmup
print("Warming up...")
for _ in range(3):
    _ = torch.matmul(x, weight)
torch.cuda.synchronize()

# Capture
g1 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g1):
    qkv = torch.matmul(x, weight)
    ptr_capture = qkv.data_ptr()

print(f"  Capture:   {hex(ptr_capture)}")

# Multiple replays
addrs = []
for i in range(5):
    g1.replay()
    addrs.append(qkv.data_ptr())
    print(f"  Replay {i+1}:  {hex(addrs[-1])} {'✓' if addrs[-1] == ptr_capture else '✗ CHANGED!'}")

test1_pass = all(addr == ptr_capture for addr in addrs)
print(f"  Result: {'PASS ✓' if test1_pass else 'FAIL ✗'}")

# Test 2: Multiple outputs (Q, K, V split)
print("\n[Test 2] Multiple tensor outputs (Q, K, V split)")
print("-" * 80)

g2 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g2):
    qkv_split = torch.matmul(x, weight)
    q, k, v = torch.chunk(qkv_split, 3, dim=-1)
    q_ptr_cap = q.data_ptr()
    k_ptr_cap = k.data_ptr()
    v_ptr_cap = v.data_ptr()

print(f"  Capture Q: {hex(q_ptr_cap)}")
print(f"  Capture K: {hex(k_ptr_cap)}")
print(f"  Capture V: {hex(v_ptr_cap)}")

q_addrs, k_addrs, v_addrs = [], [], []
for i in range(5):
    g2.replay()
    q_addrs.append(q.data_ptr())
    k_addrs.append(k.data_ptr())
    v_addrs.append(v.data_ptr())
    q_stable = '✓' if q_addrs[-1] == q_ptr_cap else '✗'
    k_stable = '✓' if k_addrs[-1] == k_ptr_cap else '✗'
    v_stable = '✓' if v_addrs[-1] == v_ptr_cap else '✗'
    print(f"  Replay {i+1}: Q={q_stable} K={k_stable} V={v_stable}")

test2_pass = (all(a == q_ptr_cap for a in q_addrs) and
              all(a == k_ptr_cap for a in k_addrs) and
              all(a == v_ptr_cap for a in v_addrs))
print(f"  Result: {'PASS ✓' if test2_pass else 'FAIL ✗'}")

# Test 3: Input data changes (addresses should stay same, values change)
print("\n[Test 3] Input data changes (verify addresses stable but values update)")
print("-" * 80)

x_input = torch.randn(10, 20, device='cuda')
w_input = torch.randn(20, 30, device='cuda')

# Warmup
_ = torch.matmul(x_input, w_input)
torch.cuda.synchronize()

g3 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g3):
    y_out = torch.matmul(x_input, w_input)

# Capture initial values
g3.replay()
val_before = y_out[0, 0].item()
addr_before = y_out.data_ptr()

# Change input data
x_input.fill_(999.0)
g3.replay()
val_after = y_out[0, 0].item()
addr_after = y_out.data_ptr()

print(f"  Before: addr={hex(addr_before)}, val={val_before:.4f}")
print(f"  After:  addr={hex(addr_after)}, val={val_after:.4f}")
print(f"  Address stable: {addr_before == addr_after} ✓")
print(f"  Value changed:  {abs(val_after - val_before) > 100} ✓")

test3_pass = (addr_before == addr_after) and (abs(val_after - val_before) > 100)
print(f"  Result: {'PASS ✓' if test3_pass else 'FAIL ✗'}")

# Test 4: Different graphs have different addresses
print("\n[Test 4] Different graphs use different memory")
print("-" * 80)

x4 = torch.randn(10, 20, device='cuda')
w4 = torch.randn(20, 30, device='cuda')

# Warmup
_ = torch.matmul(x4, w4)
torch.cuda.synchronize()

# Graph A
ga = torch.cuda.CUDAGraph()
with torch.cuda.graph(ga):
    ya = torch.matmul(x4, w4)
ga.replay()
ptr_a = ya.data_ptr()

# Graph B (same operation, different graph)
gb = torch.cuda.CUDAGraph()
with torch.cuda.graph(gb):
    yb = torch.matmul(x4, w4)
gb.replay()
ptr_b = yb.data_ptr()

print(f"  Graph A: {hex(ptr_a)}")
print(f"  Graph B: {hex(ptr_b)}")
print(f"  Different: {ptr_a != ptr_b} ✓")

test4_pass = (ptr_a != ptr_b)
print(f"  Result: {'PASS ✓' if test4_pass else 'FAIL ✗'}")

# Test 5: Nested operations (attention-like pattern)
print("\n[Test 5] Nested operations (attention pattern: matmul + softmax)")
print("-" * 80)

q_test = torch.randn(8, 16, 64, device='cuda')  # 8 heads, 16 tokens, 64 dim
k_test = torch.randn(8, 16, 64, device='cuda')

# Warmup
_ = torch.matmul(q_test, k_test.transpose(-2, -1))
_ = torch.softmax(_, dim=-1)
torch.cuda.synchronize()

g5 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g5):
    scores = torch.matmul(q_test, k_test.transpose(-2, -1))
    attn = torch.softmax(scores, dim=-1)
    scores_ptr_cap = scores.data_ptr()
    attn_ptr_cap = attn.data_ptr()

print(f"  Capture scores: {hex(scores_ptr_cap)}")
print(f"  Capture attn:   {hex(attn_ptr_cap)}")

scores_stable = []
attn_stable = []
for i in range(5):
    g5.replay()
    scores_stable.append(scores.data_ptr() == scores_ptr_cap)
    attn_stable.append(attn.data_ptr() == attn_ptr_cap)
    s_mark = '✓' if scores_stable[-1] else '✗'
    a_mark = '✓' if attn_stable[-1] else '✗'
    print(f"  Replay {i+1}: scores={s_mark} attn={a_mark}")

test5_pass = all(scores_stable) and all(attn_stable)
print(f"  Result: {'PASS ✓' if test5_pass else 'FAIL ✗'}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

all_tests = [
    ("Basic matmul (QKV projection)", test1_pass),
    ("Multiple outputs (Q/K/V split)", test2_pass),
    ("Input data changes", test3_pass),
    ("Different graphs use different memory", test4_pass),
    ("Nested operations (attention)", test5_pass),
]

for name, passed in all_tests:
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")

all_pass = all(p for _, p in all_tests)
print(f"\n{'='*80}")
if all_pass:
    print("ALL TESTS PASSED ✓")
    print("\nConclusion: CUDA graphs DO use persistent output buffers.")
    print("NWOR can safely cache tensor references across graph replays.")
else:
    print("SOME TESTS FAILED ✗")
    print("\nConclusion: CUDA graph behavior is NOT as expected.")
    print("NWOR caching strategy needs to be reconsidered.")
print(f"{'='*80}\n")
