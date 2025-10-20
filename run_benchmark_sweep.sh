#!/bin/bash
#
# NWOR + SCV Benchmark Sweep
# Runs comprehensive testing grid across 3 scenarios × 4 mode pairs × 2 temperatures
#
# Usage: ./run_benchmark_sweep.sh [--with-nsight]
#

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
TARGET_MODEL="meta-llama/Llama-3.2-3B-Instruct"
DRAFT_MODEL="linborui/EAGLE-Llama-3.2-3B-Instruct"
REQUESTS=8
BATCHES=2
DRAFT_TOKENS=4
MAX_MODEL_LEN=8196
SWEEPS_DIR="sweeps"

# Parse arguments
WITH_NSIGHT=false
if [[ "${1:-}" == "--with-nsight" ]]; then
    WITH_NSIGHT=true
    echo "Nsight profiling enabled for select runs"
fi

# Create sweeps directory
mkdir -p "$SWEEPS_DIR"

# Log file
LOG_FILE="$SWEEPS_DIR/benchmark_sweep_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "NWOR + SCV Benchmark Sweep"
echo "Started: $(date)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Target Model: $TARGET_MODEL"
echo "  Draft Model: $DRAFT_MODEL"
echo "  Requests: $REQUESTS"
echo "  Batches: $BATCHES"
echo "  Draft Tokens: $DRAFT_TOKENS"
echo "  Max Model Len: $MAX_MODEL_LEN"
echo "  Nsight Profiling: $WITH_NSIGHT"
echo ""

# Counter for progress
TOTAL_RUNS=24
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
    echo "Run $CURRENT_RUN/$TOTAL_RUNS: $scenario scenario"
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
echo "Phase 1: Short Scenario (OpenAssistant)"
echo "=========================================="

# Short scenario - Temperature 0.7 (low acceptance)
run_benchmark "short" "off" "off" "0.7" "baseline_t0.7"
run_benchmark "short" "stage" "off" "0.7" "nwor_t0.7"
run_benchmark "short" "off" "graph" "0.7" "scv_t0.7"
run_benchmark "short" "stage" "graph" "0.7" "both_t0.7"

# Short scenario - Temperature 0.0 (high acceptance)
run_benchmark "short" "off" "off" "0.0" "baseline_t0.0"
run_benchmark "short" "stage" "off" "0.0" "nwor_t0.0"
run_benchmark "short" "off" "graph" "0.0" "scv_t0.0"
run_benchmark "short" "stage" "graph" "0.0" "both_t0.0"

echo ""
echo "=========================================="
echo "Phase 2: Medium Scenario (CNN/DailyMail)"
echo "=========================================="

# Medium scenario - Temperature 0.7
run_benchmark "medium" "off" "off" "0.7" "baseline_t0.7"
run_benchmark "medium" "stage" "off" "0.7" "nwor_t0.7"
run_benchmark "medium" "off" "graph" "0.7" "scv_t0.7"
run_benchmark "medium" "stage" "graph" "0.7" "both_t0.7"

# Medium scenario - Temperature 0.0
run_benchmark "medium" "off" "off" "0.0" "baseline_t0.0"
run_benchmark "medium" "stage" "off" "0.0" "nwor_t0.0"
run_benchmark "medium" "off" "graph" "0.0" "scv_t0.0"
run_benchmark "medium" "stage" "graph" "0.0" "both_t0.0"

echo ""
echo "=========================================="
echo "Phase 3: Mixed Scenario (OpenOrca)"
echo "=========================================="

# Mixed scenario - Temperature 0.7
run_benchmark "mixed" "off" "off" "0.7" "baseline_t0.7"
run_benchmark "mixed" "stage" "off" "0.7" "nwor_t0.7"
run_benchmark "mixed" "off" "graph" "0.7" "scv_t0.7"
run_benchmark "mixed" "stage" "graph" "0.7" "both_t0.7"

# Mixed scenario - Temperature 0.0
run_benchmark "mixed" "off" "off" "0.0" "baseline_t0.0"
run_benchmark "mixed" "stage" "off" "0.0" "nwor_t0.0"
run_benchmark "mixed" "off" "graph" "0.0" "scv_t0.0"
run_benchmark "mixed" "stage" "graph" "0.0" "both_t0.0"

# Optional: Nsight profiling runs
if [[ "$WITH_NSIGHT" == true ]]; then
    echo ""
    echo "=========================================="
    echo "Phase 4: Nsight Profiling (Optional)"
    echo "=========================================="

    # Nsight profile for SCV graph mode (low acceptance)
    run_benchmark_nsight "short" "stage" "graph" "0.7" "both_t0.7_profile"

    # Optional: SCV adaptive mode
    echo ""
    echo "Running SCV adaptive mode test..."
    run_benchmark "short" "stage" "adaptive" "0.7" "adaptive_t0.7"
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=========================================="
echo "Benchmark Sweep Complete!"
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
echo "To analyze results, check the JSON files in $SWEEPS_DIR/"
echo ""
