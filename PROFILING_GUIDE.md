# NWOR + SCV Profiling Guide

## Overview

This guide explains what NWOR and SCV optimize, what metrics to measure, and which tools to use.

---

## NWOR (Non-blocking Write-Or-Read) Stage Mode

### What NWOR Optimizes
**Problem**: Speculative decoding writes draft tokens to KV cache, then overwrites them when rejected (wasted DRAM bandwidth).

**Solution**: Stage draft tokens in temporary buffers, only write accepted tokens to KV cache.

### What NWOR Does NOT Optimize
- ❌ Latency (adds 2-3% overhead from staging logic)
- ❌ Computation (same model forward passes)
- ❌ CPU time (minimal impact)

### What NWOR DOES Optimize
- ✅ **DRAM write bandwidth** (primary benefit)
- ✅ **Memory write pressure** (reduces cache contention)
- ✅ **KV cache write traffic** (only accepted tokens)

### Metrics to Measure

| Metric | Tool | Purpose | Expected Result |
|--------|------|---------|-----------------|
| **`dram__bytes_write.sum`** | NCU | Total DRAM writes | ↓ 10-15% (matches rejection rate) |
| **`dram__bytes_read.sum`** | NCU | Total DRAM reads | No change (same reads) |
| **`lts__t_sectors_op_write.sum`** | NCU | L2 cache write traffic | ↓ 10-15% (tracks DRAM writes) |
| **`dram__throughput.avg.pct_of_peak`** | NCU | Memory bandwidth utilization | ↓ if memory-bound |
| **Latency (E2E)** | Benchmark | Total request latency | ↑ 2-3% (staging overhead) |
| **Tokens Staged** | vLLM metrics | Draft tokens staged | Should equal draft tokens |
| **Tokens Committed** | vLLM metrics | Staged tokens written | Should equal accepted tokens |
| **Writes Saved %** | vLLM metrics | (staged - committed) / staged | Should be ~100% |

### When NWOR Shows Benefits

✅ **Large batches** (32-128 requests) → more rejected writes
✅ **High memory pressure** → bandwidth bottleneck visible
✅ **Long sequences** → larger KV cache footprint
✅ **Multi-GPU** → inter-GPU bandwidth constrained
✅ **Sustained workload** → cumulative bandwidth savings

❌ **Small batches** (8 requests) → low memory pressure, overhead dominates
❌ **Short runs** → overhead visible, benefits don't accumulate

### How to Profile NWOR

```bash
# 1. Run NCU bandwidth test
./run_ncu_bandwidth_test.sh

# 2. Check key metrics
python3 << EOF
import json
with open('sweeps/ncu_analysis/small_baseline_t0.7.json') as f:
    baseline = json.load(f)
with open('sweeps/ncu_analysis/small_nwor_t0.7.json') as f:
    nwor = json.load(f)

base_writes = baseline['summary']['per_mode'][0]['ncu_metrics']['dram__bytes_write.sum']
nwor_writes = nwor['summary']['per_mode'][0]['ncu_metrics']['dram__bytes_write.sum']

reduction_pct = ((base_writes - nwor_writes) / base_writes) * 100
print(f"DRAM Write Reduction: {reduction_pct:.2f}%")
print(f"Baseline: {base_writes/1e9:.4f} GB")
print(f"NWOR:     {nwor_writes/1e9:.4f} GB")
print(f"Saved:    {(base_writes - nwor_writes)/1e9:.4f} GB")
EOF
```

### Expected NCU Output

```
Baseline (NWOR off):
  DRAM Writes:  1,250,000,000 bytes (1.25 GB)
  DRAM Reads:   5,000,000,000 bytes (5.00 GB)
  L2 Writes:    45,200,000 sectors
  BW Util:      12.50%

NWOR Stage:
  DRAM Writes:  1,125,000,000 bytes (1.13 GB)  ← 10% reduction!
  DRAM Reads:   5,000,000,000 bytes (5.00 GB)  ← Same
  L2 Writes:    40,700,000 sectors              ← 10% reduction
  BW Util:      11.80%                           ← Lower

Delta: -125 MB (-10%) in DRAM writes
```

---

## SCV (Speculative Comparison Vectorized) Graph Mode

### What SCV Optimizes
**Problem**: Mask computation for speculative verification uses Python host-side loop (slow, sequential).

**Solution**: Vectorized GPU kernel + CUDA graph capture (fast, parallel, near-zero dispatch).

### What SCV Does NOT Optimize
- ❌ DRAM bandwidth (same memory operations)
- ❌ KV cache writes (NWOR's job)
- ❌ Model computation (same forward passes)

### What SCV DOES Optimize
- ✅ **Host CPU overhead** (Python loop → GPU kernel)
- ✅ **Kernel launch overhead** (N launches → 1 launch, or graph = 0)
- ✅ **CPU-GPU sync points** (loop syncs → single sync)
- ✅ **Parallelism** (sequential requests → parallel)
- ✅ **Dispatch overhead** (kernel launch ~5µs → graph replay <1µs)

### Metrics to Measure

| Metric | Tool | Purpose | Expected Result |
|--------|------|---------|-----------------|
| **Host CPU time** | Nsight Systems | Python loop overhead | ↓ 10-100µs (baseline has loop) |
| **Kernel launch count** | Nsight Systems / NCU | Number of CUDA kernel launches | N launches → 1 (or 0 with graph) |
| **CUDA API overhead** | Nsight Systems | cudaLaunchKernel time | ↓ 90% with graph capture |
| **GPU kernel time** | Nsight Systems / NCU | Actual computation time | Similar (same work, better parallelism) |
| **NVTX range** | Nsight Systems | "scv_compute_mask" marker | Visible in timeline |
| **Latency (E2E)** | Benchmark | Total request latency | ↓ 0-5µs or neutral |
| **`gpu__time_duration.sum`** | NCU | Total GPU time in kernel | Similar baseline vs SCV |
| **`sm__warps_launched.sum`** | NCU | Parallelism (warps) | Higher with SCV (parallel) |

### How to Profile SCV

```bash
# 1. Run Nsight Systems analysis
./run_scv_benefit_analysis.sh

# 2. Open reports in GUI
nsight-sys sweeps/scv_benefit_analysis/baseline_off_small_nsys.nsys-rep
nsight-sys sweeps/scv_benefit_analysis/scv_graph_small_nsys.nsys-rep

# 3. Compare timelines:
#    - CPU timeline: Look for Python function calls (baseline) vs kernel launch (SCV)
#    - GPU timeline: Count kernel launches
#    - CUDA API: Count cudaLaunchKernel calls
#    - NVTX: Find "scv_compute_mask" markers
```

### Expected Nsight Systems Output

**Baseline (SCV off)**:
```
CPU Timeline:
  ├─ Python: _compute_acceptance_mask (50µs)
  │   └─ for loop over requests
  │       ├─ cudaLaunchKernel (5µs) ← Multiple launches
  │       ├─ cudaLaunchKernel (5µs)
  │       └─ cudaLaunchKernel (5µs)
  └─ cudaDeviceSynchronize (10µs)

GPU Timeline:
  ├─ Kernel: compare_tokens (2µs)
  ├─ Kernel: compare_tokens (2µs)
  └─ Kernel: compare_tokens (2µs)

Total: ~80µs (50µs host + 30µs GPU/sync)
```

**SCV Graph Mode**:
```
CPU Timeline:
  ├─ Python: _scv_vectorized_mask (5µs) ← Single call
  │   └─ cudaGraphLaunch (<1µs) ← Graph replay!
  └─ cudaDeviceSynchronize (10µs)

GPU Timeline:
  └─ Kernel: _scv_compute_mask_inplace (6µs) ← Single kernel

NVTX:
  └─ [scv_compute_mask] (20µs total)

Total: ~20µs (5µs host + 6µs kernel + 10µs sync)
```

**Savings**: 80µs → 20µs = **60µs reduction (~75%)**

### SCV Graph Capture Benefit

**Without graph** (SCV vectorized mode):
- Kernel launch overhead: ~5µs per call
- Host dispatch: ~2µs
- Total overhead: ~7µs

**With graph** (SCV graph mode):
- Graph replay: <1µs
- Host dispatch: ~0.5µs
- Total overhead: ~1.5µs

**Graph benefit**: ~5.5µs saved per mask computation

At 100 iterations:
- Without graph: 7µs × 100 = 700µs
- With graph: 1.5µs × 100 = 150µs
- **Savings: 550µs (0.55ms)**

---

## Combined Analysis

### Trade-offs Summary

| Mode | Latency Impact | Bandwidth Impact | When to Use |
|------|----------------|------------------|-------------|
| **NWOR off, SCV off** | Baseline | Baseline | Never (baseline only) |
| **NWOR stage, SCV off** | +2-3% | -10-15% writes | High memory pressure |
| **NWOR off, SCV graph** | -0.5% or neutral | None | Always (no downside) |
| **NWOR stage, SCV graph** | +2-3% | -10-15% writes | High memory pressure |

### Recommendations

1. **SCV Graph Mode**: ✅ **Always enable**
   - Negligible overhead (<2%)
   - Some scenarios show improvement
   - No downside, pure benefit

2. **NWOR Stage Mode**: ⚠️ **Enable for high-throughput workloads**
   - Costs 2-3% latency
   - Saves 10-15% DRAM writes
   - Net positive under memory pressure (large batches, multi-GPU)
   - Make configurable, document trade-off

3. **Combined Mode**: ⚠️ **Use case dependent**
   - SCV overhead negligible, NWOR overhead dominates
   - Best for sustained high-throughput workloads
   - Profile your specific workload first

---

## Quick Reference Commands

### Measure NWOR Bandwidth Savings
```bash
./run_ncu_bandwidth_test.sh
# Check: sweeps/ncu_analysis/*_stats.txt
# Look for: dram__bytes_write.sum reduction
```

### Measure SCV Host Overhead Reduction
```bash
./run_scv_benefit_analysis.sh
# Open: nsight-sys sweeps/scv_benefit_analysis/*_nsys.nsys-rep
# Compare: CPU timeline, kernel launch counts
```

### Quick Latency-Only Test
```bash
./run_benchmark_sweep.sh
# Check: sweeps/*.json for latency_avg_s
```

---

## Interpretation

### NWOR is Working If:
- ✅ `nwor_writes_saved_pct` = 100%
- ✅ `dram__bytes_write.sum` reduced by ~10-15%
- ✅ `lts__t_sectors_op_write.sum` reduced proportionally
- ⚠️ Latency increased by 2-3% (expected overhead)

### SCV is Working If:
- ✅ Latency neutral or slightly improved
- ✅ Nsight Systems shows fewer kernel launches
- ✅ Nsight Systems shows reduced host CPU time
- ✅ NVTX markers visible for "scv_compute_mask"
- ✅ Graph replay <1µs (vs ~5µs kernel launch)

### Both are Working If:
- ✅ NWOR metrics correct (above)
- ✅ SCV metrics correct (above)
- ⚠️ Combined overhead ~= NWOR overhead (SCV adds minimal)
