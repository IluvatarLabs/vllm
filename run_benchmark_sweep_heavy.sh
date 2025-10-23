#!/bin/bash
#
# NWOR + SCV Bandwidth-Heavy Benchmark Sweep
# Tests NWOR performance on bandwidth-bound workloads
#
# This configuration maximizes memory bandwidth to test if NWOR helps:
# - Larger model (8B vs 3B) → more memory traffic
# - Higher batch size (16 vs 8) → more concurrent operations
# - Fewer draft tokens (1 vs 4) → lower compute intensity
# - FP16 precision → 2x bandwidth vs FP8
#
# Usage: ./run_benchmark_sweep_heavy.sh [--with-nsight]
#

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration - BANDWIDTH-HEAVY SETTINGS
TARGET_MODEL="meta-llama/Llama-3.2-3B-Instruct"  # Larger model (8B vs 3B)
DRAFT_MODEL="linborui/EAGLE-Llama-3.2-3B-Instruct"
REQUESTS=16  # Increased from 8 → more batch parallelism
BATCHES=2
DRAFT_TOKENS=1  # Reduced from 4 → less compute, more bandwidth-bound
MAX_MODEL_LEN=8196
SWEEPS_DIR="sweeps_heavy"

# Parse arguments
WITH_NSIGHT=false
if [[ "${1:-}" == "--with-nsight" ]]; then
    WITH_NSIGHT=true
    echo "Nsight profiling enabled for select runs"
fi

# Create sweeps directory
mkdir -p "$SWEEPS_DIR"

# Log file
LOG_FILE="$SWEEPS_DIR/benchmark_sweep_heavy_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "NWOR + SCV Bandwidth-Heavy Benchmark"
echo "Started: $(date)"
echo "=========================================="
echo ""
echo "Configuration (Bandwidth-Optimized):"
echo "  Target Model: $TARGET_MODEL (8B - larger)"
echo "  Draft Model: $DRAFT_MODEL"
echo "  Requests: $REQUESTS (increased batch)"
echo "  Batches: $BATCHES"
echo "  Draft Tokens: $DRAFT_TOKENS (reduced compute)"
echo "  Max Model Len: $MAX_MODEL_LEN"
echo "  Precision: FP16 (high bandwidth)"
echo "  Nsight Profiling: $WITH_NSIGHT"
echo ""
echo "Goal: Test if NWOR helps on bandwidth-bound workloads"
echo ""

# Counter for progress
TOTAL_RUNS=12  # Reduced to just mixed scenario
CURRENT_RUN=0

# Function to run a single benchmark
run_benchmark() {
    local scenario=$1
    local nwor_mode=$2
    local scv_mode=$3
    local temperature=$4
    local output_suffix=$5

    CURRENT_RUN=$((CURRENT_RUN + 1))

    echo ""
    echo "=========================================="
    echo "Run $CURRENT_RUN/$TOTAL_RUNS: $scenario scenario (bandwidth-heavy)"
    echo "  NWOR: $nwor_mode, SCV: $scv_mode, Temp: $temperature"
    echo "  Started: $(date)"
    echo "=========================================="

    local output_file="$SWEEPS_DIR/${scenario}_${output_suffix}.json"

    # Set environment variables
    export VLLM_SCV_MODE=$scv_mode
    export VLLM_NWOR_MODE=$nwor_mode
    export TARGET_MODEL=$TARGET_MODEL
    export DRAFT_MODEL=$DRAFT_MODEL

    # Enable profiling for SCV graph mode
    if [[ "$scv_mode" == "graph" ]] || [[ "$scv_mode" == "adaptive" ]]; then
        export VLLM_SCV_PROFILE=1
    else
        export VLLM_SCV_PROFILE=0
    fi

    # Run benchmark
    if python3 tools/profiling/run_nwor_microbench.py \
        --scenario "$scenario" \
        --requests $REQUESTS \
        --batches $BATCHES \
        --draft-tokens $DRAFT_TOKENS \
        --temperature "$temperature" \
        --nwor-modes "$nwor_mode" \
        --scv-modes "$scv_mode" \
        --max-model-len $MAX_MODEL_LEN \
        --output "$output_file"; then
        echo "✓ Completed successfully: $output_file"
    else
        echo "✗ FAILED: $scenario/$output_suffix (exit code: $?)"
        echo "  Continuing with remaining tests..."
    fi

    echo "  Finished: $(date)"
}

# Function to run benchmark with Nsight profiling
run_benchmark_nsight() {
    local scenario=$1
    local nwor_mode=$2
    local scv_mode=$3
    local temperature=$4
    local output_suffix=$5

    echo ""
    echo "=========================================="
    echo "Nsight Profile: $scenario scenario"
    echo "  NWOR: $nwor_mode, SCV: $scv_mode, Temp: $temperature"
    echo "  Started: $(date)"
    echo "=========================================="

    local output_file="$SWEEPS_DIR/${scenario}_${output_suffix}.json"
    local nsight_output="$SWEEPS_DIR/${scenario}_${output_suffix}_nsight"

    # Set environment variables
    export VLLM_SCV_MODE=$scv_mode
    export VLLM_NWOR_MODE=$nwor_mode
    export VLLM_SCV_PROFILE=1
    export TARGET_MODEL=$TARGET_MODEL
    export DRAFT_MODEL=$DRAFT_MODEL

    # Run with Nsight
    if nsys profile --trace=cuda,nvtx,osrt \
                  --sample=none \
                  --force-overwrite=true \
                  --trace-fork-before-exec=true \
                  --output "$nsight_output" \
                  python3 tools/profiling/run_nwor_microbench.py \
                  --scenario "$scenario" \
                  --requests $REQUESTS \
                  --batches $BATCHES \
                  --draft-tokens $DRAFT_TOKENS \
                  --temperature "$temperature" \
                  --nwor-modes "$nwor_mode" \
                  --scv-modes "$scv_mode" \
                  --max-model-len $MAX_MODEL_LEN \
                  --output "$output_file"; then
        echo "✓ Nsight profiling completed: $nsight_output.nsys-rep"
    else
        echo "✗ Nsight profiling FAILED (exit code: $?)"
        echo "  Continuing with remaining tests..."
    fi

    echo "  Finished: $(date)"
}

# Start timer
START_TIME=$(date +%s)

echo ""
echo "=========================================="
echo "Bandwidth-Heavy Test: Mixed Scenario"
echo "=========================================="
echo ""
echo "Testing with 8B model, 16 requests, 1 draft token"
echo "This maximizes bandwidth:compute ratio"
echo ""

# Focus on mixed scenario with both temperatures
# Temperature 0.7 (realistic acceptance)
run_benchmark "mixed" "off" "off" "0.7" "baseline_t0.7"
run_benchmark "mixed" "stage" "off" "0.7" "nwor_t0.7"
run_benchmark "mixed" "off" "graph" "0.7" "scv_t0.7"
run_benchmark "mixed" "stage" "graph" "0.7" "both_t0.7"

# Temperature 0.0 (higher acceptance - more KV cache traffic)
run_benchmark "mixed" "off" "off" "0.0" "baseline_t0.0"
run_benchmark "mixed" "stage" "off" "0.0" "nwor_t0.0"
run_benchmark "mixed" "off" "graph" "0.0" "scv_t0.0"
run_benchmark "mixed" "stage" "graph" "0.0" "both_t0.0"

# Also test medium scenario (longer sequences = more KV cache)
echo ""
echo "=========================================="
echo "Bandwidth-Heavy Test: Medium Scenario"
echo "=========================================="
echo ""

run_benchmark "medium" "off" "off" "0.7" "baseline_t0.7"
run_benchmark "medium" "stage" "off" "0.7" "nwor_t0.7"
run_benchmark "medium" "off" "graph" "0.7" "scv_t0.7"
run_benchmark "medium" "stage" "graph" "0.7" "both_t0.7"

# Optional: Nsight profiling runs
if [[ "$WITH_NSIGHT" == true ]]; then
    echo ""
    echo "=========================================="
    echo "Phase 2: Nsight Profiling (Optional)"
    echo "=========================================="

    # Profile NWOR to see memory traffic patterns
    run_benchmark_nsight "mixed" "stage" "graph" "0.7" "both_t0.7_profile"
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=========================================="
echo "Bandwidth-Heavy Benchmark Complete!"
echo "=========================================="
echo ""
echo "Total runs completed: $CURRENT_RUN/$TOTAL_RUNS"
echo "Elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Results directory: $SWEEPS_DIR"
echo "Log file: $LOG_FILE"
echo "Finished: $(date)"
echo ""

# List all output files
echo "Generated files:"
ls -lh "$SWEEPS_DIR"/*.json 2>/dev/null || echo "  No JSON files found"
if [[ "$WITH_NSIGHT" == true ]]; then
    ls -lh "$SWEEPS_DIR"/*.nsys-rep 2>/dev/null || echo "  No Nsight files found"
fi

echo ""
echo "=========================================="
echo "Analysis Instructions:"
echo "=========================================="
echo ""
echo "Compare NWOR overhead between this run and standard run:"
echo ""
echo "Standard (3B, 8 req, 4 draft):  sweeps/mixed_*_t0.7.json"
echo "Heavy (8B, 16 req, 1 draft):    $SWEEPS_DIR/mixed_*_t0.7.json"
echo ""
echo "If NWOR overhead is LOWER in heavy run → bandwidth helps!"
echo "If NWOR overhead is SAME/HIGHER → need even larger models"
echo ""
echo "Key metrics to compare:"
echo "  - NWOR overhead (baseline → nwor)"
echo "  - NWOR+SCV overhead (baseline → both)"
echo "  - Absolute latencies"
echo ""
