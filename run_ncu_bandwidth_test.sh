#!/bin/bash
#
# NWOR Bandwidth Analysis - NCU Profiling
# Measures DRAM bandwidth savings from NWOR stage mode
#
# This script runs focused tests with NCU metrics enabled to measure:
# 1. DRAM write bandwidth (primary NWOR benefit)
# 2. L2 cache write traffic
# 3. Memory bandwidth utilization
#
# Usage: ./run_ncu_bandwidth_test.sh
#

set -e
set -u

# Configuration
TARGET_MODEL="meta-llama/Llama-3.2-3B-Instruct"
DRAFT_MODEL="linborui/EAGLE-Llama-3.2-3B-Instruct"
SWEEPS_DIR="sweeps/ncu_analysis"

# NCU metrics to capture
NCU_METRICS="dram__bytes_write.sum,dram__bytes_read.sum,lts__t_sectors_op_write.sum,lts__t_sectors_op_read.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed"

# Create output directory
mkdir -p "$SWEEPS_DIR"

# Log file
LOG_FILE="$SWEEPS_DIR/ncu_bandwidth_test_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "NWOR Bandwidth Analysis - NCU Profiling"
echo "Started: $(date)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Target Model: $TARGET_MODEL"
echo "  Draft Model: $DRAFT_MODEL"
echo "  NCU Metrics: $NCU_METRICS"
echo "  Output Directory: $SWEEPS_DIR"
echo ""

# Function to run NCU-enabled benchmark
run_ncu_test() {
    local test_name=$1
    local scenario=$2
    local nwor_mode=$3
    local scv_mode=$4
    local temperature=$5
    local requests=$6
    local draft_tokens=$7
    local batches=$8

    echo ""
    echo "=========================================="
    echo "Test: $test_name"
    echo "  Scenario: $scenario"
    echo "  NWOR: $nwor_mode, SCV: $scv_mode"
    echo "  Temp: $temperature, Requests: $requests"
    echo "  Draft Tokens: $draft_tokens, Batches: $batches"
    echo "  Started: $(date)"
    echo "=========================================="

    local output_file="$SWEEPS_DIR/${test_name}.json"

    # Set environment variables
    export VLLM_SCV_MODE=$scv_mode
    export VLLM_NWOR_MODE=$nwor_mode
    export VLLM_SCV_PROFILE=0
    export TARGET_MODEL=$TARGET_MODEL
    export DRAFT_MODEL=$DRAFT_MODEL

    # Run with NCU metrics enabled
    if python3 tools/profiling/run_nwor_microbench.py \
        --scenario "$scenario" \
        --requests "$requests" \
        --batches "$batches" \
        --draft-tokens "$draft_tokens" \
        --temperature "$temperature" \
        --nwor-modes "$nwor_mode" \
        --scv-modes "$scv_mode" \
        --max-model-len 8196 \
        --enable-ncu \
        --ncu-metrics "$NCU_METRICS" \
        --output "$output_file"; then
        echo "✓ Completed: $output_file"

        # Extract and display NCU metrics
        if [ -f "$output_file" ]; then
            echo ""
            echo "NCU Metrics Summary:"
            python3 -c "
import json
with open('$output_file') as f:
    data = json.load(f)
    for mode_data in data.get('summary', {}).get('per_mode', []):
        metrics = mode_data.get('ncu_metrics', {})
        if metrics:
            print('  DRAM Writes:  {:>15,} bytes'.format(int(metrics.get('dram__bytes_write.sum', 0))))
            print('  DRAM Reads:   {:>15,} bytes'.format(int(metrics.get('dram__bytes_read.sum', 0))))
            print('  L2 Writes:    {:>15,} sectors'.format(int(metrics.get('lts__t_sectors_op_write.sum', 0))))
            print('  L2 Reads:     {:>15,} sectors'.format(int(metrics.get('lts__t_sectors_op_read.sum', 0))))
            print('  BW Util:      {:>15.2f}%'.format(float(metrics.get('dram__throughput.avg.pct_of_peak_sustained_elapsed', 0))))
        else:
            print('  No NCU metrics captured')
" || echo "  Failed to parse metrics"
        fi
    else
        echo "✗ Output file not found: $output_file"
    fi

    echo "  Finished: $(date)"
}

# Start timer
START_TIME=$(date +%s)

echo ""
echo "=========================================="
echo "Phase 1: Small Batch Tests (Baseline)"
echo "  Requests: 8, Draft Tokens: 4"
echo "=========================================="

# Test 1: Baseline (no NWOR, no SCV) - Small batch, temp 0.7
run_ncu_test "small_baseline_t0.7" "short" "off" "off" "0.7" 8 4 2

# Test 2: NWOR stage mode - Small batch, temp 0.7
run_ncu_test "small_nwor_t0.7" "short" "stage" "off" "0.7" 8 4 2

# Test 3: Baseline - Small batch, temp 0.0 (high acceptance)
run_ncu_test "small_baseline_t0.0" "short" "off" "off" "0.0" 8 4 2

# Test 4: NWOR stage mode - Small batch, temp 0.0
run_ncu_test "small_nwor_t0.0" "short" "stage" "off" "0.0" 8 4 2

echo ""
echo "=========================================="
echo "Phase 2: Medium Batch Tests"
echo "  Requests: 16, Draft Tokens: 6"
echo "=========================================="

# Test 5: Baseline - Medium batch
run_ncu_test "medium_baseline_t0.7" "short" "off" "off" "0.7" 16 6 4

# Test 6: NWOR stage mode - Medium batch
run_ncu_test "medium_nwor_t0.7" "short" "stage" "off" "0.7" 16 6 4

echo ""
echo "=========================================="
echo "Phase 3: Large Batch Tests (High Memory Pressure)"
echo "  Requests: 32, Draft Tokens: 8"
echo "=========================================="

# Test 7: Baseline - Large batch
run_ncu_test "large_baseline_t0.7" "short" "off" "off" "0.7" 32 8 8

# Test 8: NWOR stage mode - Large batch
run_ncu_test "large_nwor_t0.7" "short" "stage" "off" "0.7" 32 8 8

echo ""
echo "=========================================="
echo "Phase 4: Sustained Load Tests"
echo "  Requests: 16, Draft Tokens: 4, Batches: 20"
echo "=========================================="

# Test 9: Baseline - Sustained load
run_ncu_test "sustained_baseline_t0.7" "short" "off" "off" "0.7" 16 4 20

# Test 10: NWOR stage mode - Sustained load
run_ncu_test "sustained_nwor_t0.7" "short" "stage" "off" "0.7" 16 4 20

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=========================================="
echo "NCU Bandwidth Analysis Complete!"
echo "=========================================="
echo ""
echo "Elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Results directory: $SWEEPS_DIR"
echo "Log file: $LOG_FILE"
echo "Finished: $(date)"
echo ""

# Generate comparison report
echo "=========================================="
echo "Generating Bandwidth Savings Report..."
echo "=========================================="

python3 << 'PYTHON_SCRIPT'
import json
import os
from pathlib import Path
from typing import Dict, Any

sweeps_dir = Path("sweeps/ncu_analysis")
results = {}

# Load all NCU test results
for json_file in sorted(sweeps_dir.glob("*.json")):
    try:
        with open(json_file) as f:
            data = json.load(f)

        test_name = json_file.stem

        if "summary" in data and "per_mode" in data["summary"]:
            mode_data = data["summary"]["per_mode"][0]
            results[test_name] = {
                "nwor_mode": mode_data.get("nwor_mode", "N/A"),
                "latency_ms": mode_data.get("latency_avg_s", 0) * 1000,
                "ncu_metrics": mode_data.get("ncu_metrics", {}),
                "spec_acceptance_ratio": mode_data.get("spec_acceptance_ratio", 0),
                "nwor_writes_saved_pct": mode_data.get("nwor_writes_saved_pct", 0),
            }
    except Exception as e:
        print(f"Error loading {json_file}: {e}")

if not results:
    print("No results found. Tests may have failed.")
    exit(1)

# Generate comparison report
print("\n" + "="*160)
print("NWOR BANDWIDTH SAVINGS ANALYSIS")
print("="*160)

test_pairs = [
    ("small_baseline_t0.7", "small_nwor_t0.7", "Small Batch (8 req, 4 draft) - Temp 0.7"),
    ("small_baseline_t0.0", "small_nwor_t0.0", "Small Batch (8 req, 4 draft) - Temp 0.0"),
    ("medium_baseline_t0.7", "medium_nwor_t0.7", "Medium Batch (16 req, 6 draft) - Temp 0.7"),
    ("large_baseline_t0.7", "large_nwor_t0.7", "Large Batch (32 req, 8 draft) - Temp 0.7"),
    ("sustained_baseline_t0.7", "sustained_nwor_t0.7", "Sustained Load (16 req, 4 draft, 20 batches)"),
]

print(f"\n{'Test Configuration':<50} {'Mode':<8} {'Latency (ms)':<14} {'DRAM Writes (GB)':<18} {'DRAM Reads (GB)':<17} {'L2 Write (M)':<13} {'BW Util %':<10}")
print("-"*160)

for baseline_name, nwor_name, description in test_pairs:
    baseline = results.get(baseline_name)
    nwor = results.get(nwor_name)

    if baseline and nwor:
        # Print baseline
        base_metrics = baseline["ncu_metrics"]
        base_dram_write_gb = base_metrics.get("dram__bytes_write.sum", 0) / 1e9
        base_dram_read_gb = base_metrics.get("dram__bytes_read.sum", 0) / 1e9
        base_l2_write_m = base_metrics.get("lts__t_sectors_op_write.sum", 0) / 1e6
        base_bw_util = base_metrics.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", 0)

        print(f"{description:<50} {'baseline':<8} {baseline['latency_ms']:<14.2f} {base_dram_write_gb:<18.4f} {base_dram_read_gb:<17.4f} {base_l2_write_m:<13.2f} {base_bw_util:<10.2f}")

        # Print NWOR
        nwor_metrics = nwor["ncu_metrics"]
        nwor_dram_write_gb = nwor_metrics.get("dram__bytes_write.sum", 0) / 1e9
        nwor_dram_read_gb = nwor_metrics.get("dram__bytes_read.sum", 0) / 1e9
        nwor_l2_write_m = nwor_metrics.get("lts__t_sectors_op_write.sum", 0) / 1e6
        nwor_bw_util = nwor_metrics.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", 0)

        print(f"{'':<50} {'nwor':<8} {nwor['latency_ms']:<14.2f} {nwor_dram_write_gb:<18.4f} {nwor_dram_read_gb:<17.4f} {nwor_l2_write_m:<13.2f} {nwor_bw_util:<10.2f}")

        # Calculate deltas
        latency_delta_ms = nwor["latency_ms"] - baseline["latency_ms"]
        latency_delta_pct = (latency_delta_ms / baseline["latency_ms"]) * 100 if baseline["latency_ms"] > 0 else 0

        if base_dram_write_gb > 0:
            dram_write_delta_gb = nwor_dram_write_gb - base_dram_write_gb
            dram_write_saved_pct = (dram_write_delta_gb / base_dram_write_gb) * 100
        else:
            dram_write_delta_gb = 0
            dram_write_saved_pct = 0

        if base_l2_write_m > 0:
            l2_write_delta_m = nwor_l2_write_m - base_l2_write_m
            l2_write_saved_pct = (l2_write_delta_m / base_l2_write_m) * 100
        else:
            l2_write_delta_m = 0
            l2_write_saved_pct = 0

        bw_util_delta = nwor_bw_util - base_bw_util

        print(f"{'':<50} {'Δ':<8} {latency_delta_ms:<+14.2f} {dram_write_delta_gb:<+18.4f} {'':<17} {l2_write_delta_m:<+13.2f} {bw_util_delta:<+10.2f}")
        print(f"{'':<50} {'Δ%':<8} {latency_delta_pct:<+14.2f} {dram_write_saved_pct:<+18.2f} {'':<17} {l2_write_saved_pct:<+13.2f} {'':<10}")
        print(f"{'':<50} {'Accept':<8} {'':<14} {'Writes Saved':<18} {nwor['nwor_writes_saved_pct']:<17.1f}% {'':<13} {'':<10}")
        print("-"*160)

print("\n" + "="*160)
print("INTERPRETATION GUIDE")
print("="*160)
print("""
Expected Results if NWOR is working correctly:
1. DRAM Writes: Should decrease by ~(rejection_rate)%
   - At 10% acceptance: ~90% of draft tokens rejected → ~10-15% write reduction
   - At 15% acceptance: ~85% of draft tokens rejected → ~8-12% write reduction

2. Latency: May increase by 2-3% due to staging overhead (this is expected)

3. L2 Write Sectors: Should track with DRAM writes reduction

4. Bandwidth Utilization: May decrease if memory-bound (good sign)

Key Question: Does DRAM write reduction exceed latency overhead cost?
- If DRAM writes ↓ 10% but latency ↑ 3% → Net positive under memory pressure
- If DRAM writes ↓ 1% and latency ↑ 3% → Not worth it in this regime

Scaling Prediction:
- Small batches (8 req): Low memory pressure, overhead visible, benefit small
- Large batches (32+ req): High memory pressure, benefit should exceed overhead
- Sustained load: Cumulative bandwidth savings should translate to throughput gain
""")

print("\n" + "="*160)

PYTHON_SCRIPT

echo ""
echo "Analysis complete! Check $SWEEPS_DIR for detailed results."
echo ""
