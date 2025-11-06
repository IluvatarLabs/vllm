# NWOR Experiment Scripts

Comprehensive experiment suite for profiling adaptive draft length and confidence-based early exit features.

## Quick Start

### Run Everything (4.5 hours)
```bash
./run_all_experiments.sh
```

### Run Individual Grids

**Adaptive Draft Length (1 hour, 16 runs)**
```bash
./run_adaptive_grid.sh
```

**Confidence Early Exit (2.5 hours, 7 runs with NCU)**
```bash
./run_early_exit_grid.sh
```

**One-Off Scenarios (1 hour, 4 runs)**
```bash
./run_oneoff_scenarios.sh
```

## Model Configuration

Default models:
- Target: `meta-llama/Llama-3.1-8B-Instruct`
- Draft: `yuhuili/EAGLE-LLaMA3.1-Instruct-8B`
- Tensor Parallel: `2`

### Override Models

```bash
# Set environment variables before running
export TARGET_MODEL="your-target-model"
export DRAFT_MODEL="your-draft-model"
export TENSOR_PARALLEL_SIZE=4

./run_adaptive_grid.sh
```

Or inline:
```bash
TARGET_MODEL="llama-70b" DRAFT_MODEL="eagle-70b" ./run_adaptive_grid.sh
```

## Output Structure

```
sweeps/
├── adaptive_grid/
│   ├── run1_r12_t64_temp0.0_d10_adaptive0.json
│   ├── run1_r12_t64_temp0.0_d10_adaptive1.json
│   ├── ...
│   └── experiment.log
├── early_exit_grid/
│   ├── run1_r36_t128_temp0.0_thresh0.0.json
│   ├── run3_r36_t128_temp0.0_thresh0.5.json
│   ├── ...
│   └── experiment.log
├── oneoff_scenarios/
│   ├── scenario_a_code_completion.json
│   ├── scenario_b_chatbot.json
│   ├── scenario_c_high_throughput.json
│   ├── scenario_d_creative_writing.json
│   └── experiment.log
└── master_experiment.log
```

## Resume Capability

Scripts automatically skip completed runs. If a run fails:
1. Fix the issue
2. Re-run the script
3. It will skip completed runs and continue where it left off

## Key Experiments

### ⭐ Most Important Runs

**Adaptive Grid:**
- Runs 5-6: Medium interactive (36 requests, 128 tokens, temp=0.0)
  - Most representative of production workloads
  - Compare adaptive=0 vs adaptive=1 for latency improvement

**Early Exit Grid:**
- Run 1 vs Run 3: Baseline vs default threshold (0.0 vs 0.5)
  - Shows bandwidth savings via NCU metrics
  - Check `dram__bytes_write.sum` in JSON output

**One-Off Scenarios:**
- Scenario B: Customer service chatbot
  - 80% of production deployments
- Scenario C: High-throughput server
  - Shows combined benefit of both features

## What Each Grid Tests

### Adaptive Grid (Grid 1)
**Purpose:** Measure latency improvement across workload types

**Tests:**
- Batch sizes: 12, 36, 60 requests
- Sequence lengths: 64, 128, 256 tokens
- Temperatures: 0.0 (deterministic), 0.7 (creative)
- Draft sizes: 5, 10, 15 tokens

**Key Metrics:**
- `latency_avg_s`: Average latency per batch
- `spec_num_draft_tokens`: Total draft tokens generated
- `spec_acceptance_ratio`: Acceptance rate

### Early Exit Grid (Grid 2)
**Purpose:** Measure bandwidth savings via NCU profiling

**Tests:**
- Thresholds: 0.0, 0.3, 0.5, 0.7
- With/without sampling (temp 0.0 vs 0.7)
- Short vs long sequences

**Key Metrics:**
- `ncu_metrics.dram__bytes_write.sum`: DRAM write bandwidth
- `ncu_metrics.dram__throughput.avg.pct_of_peak_sustained_elapsed`: Utilization %
- Instrumentation logs: "Confidence early exit stats: N tokens stopped"

### One-Off Scenarios
**Purpose:** Show real-world representative deployments

- **Scenario A (Code Completion):** Short responses, high acceptance
- **Scenario B (Chatbot) ⭐:** Most common production use case
- **Scenario C (High Throughput):** Large batch, bandwidth matters
- **Scenario D (Creative Writing):** Low acceptance, adaptive adapts

## Troubleshooting

### NCU Hangs
Fixed in latest commit. If still hanging:
1. Check terminal for "[INFO] Exporting CSV from NCU report" message
2. Wait up to 5 minutes for export to complete
3. If truly stuck > 10 minutes, kill and report the error

### Out of Memory
Reduce batch size or requests:
```bash
# Edit the script and reduce --requests or --batches parameters
```

### Model Not Found
Ensure models are downloaded or specify correct paths:
```bash
export TARGET_MODEL="/path/to/model"
export DRAFT_MODEL="/path/to/draft"
```

## Analyzing Results

### Extract Latency Improvements
```bash
# Compare adaptive off vs on
for f in sweeps/adaptive_grid/run*_adaptive0.json; do
    base=$(basename "$f" _adaptive0.json)
    lat0=$(jq -r '.summary.per_mode[0].latency_avg_s' "$f")
    lat1=$(jq -r '.summary.per_mode[0].latency_avg_s' "${f/_adaptive0/_adaptive1}")
    improvement=$(echo "scale=2; ($lat0 - $lat1) / $lat0 * 100" | bc)
    echo "$base: $improvement% improvement"
done
```

### Extract Bandwidth Savings
```bash
# Compare threshold 0.0 vs 0.5
baseline=$(jq -r '.summary.per_mode[0].ncu_metrics."dram__bytes_write.sum"' sweeps/early_exit_grid/run1_r36_t128_temp0.0_thresh0.0.json)
threshold=$(jq -r '.summary.per_mode[0].ncu_metrics."dram__bytes_write.sum"' sweeps/early_exit_grid/run3_r36_t128_temp0.0_thresh0.5.json)
echo "Baseline: $baseline bytes"
echo "Threshold 0.5: $threshold bytes"
```

### Check Instrumentation Logs
```bash
# See how many tokens were stopped by confidence threshold
grep "Confidence early exit stats" sweeps/early_exit_grid/experiment.log
```

## Time Estimates

| Script | Runs | NCU | Est. Time |
|--------|------|-----|-----------|
| Adaptive Grid | 16 | No | 1 hour |
| Early Exit Grid | 7 | Yes | 2.5 hours |
| One-Off Scenarios | 4 | 1 run | 1 hour |
| **Total** | **27** | **8 runs** | **~4.5 hours** |

## Blog Post Data Collection

After running all experiments, collect:

### Adaptive Draft Length:
1. Latency reduction % (average across all workloads)
2. Token reduction % (spec_num_draft_tokens)
3. Acceptance rate correlation
4. Throughput improvement (tokens/sec)

### Confidence Early Exit:
1. DRAM write reduction (bytes, from NCU)
2. Tokens stopped % (from instrumentation)
3. Latency overhead %
4. Bandwidth savings vs latency tradeoff

### Both Features:
1. Determinism test (temp=0.0 outputs)
2. Acceptance rate by workload
3. Optimal configuration recommendations
