#!/bin/bash
#
# SCV Benefit Analysis - Comprehensive Profiling
# Measures what SCV actually optimizes: host overhead and kernel efficiency
#
# SCV optimizes:
# 1. Host CPU time (Python loop → GPU kernel)
# 2. Number of kernel launches (N loops → 1 kernel)
# 3. CPU-GPU synchronization overhead
# 4. Mask computation parallelism
#
# This script uses BOTH Nsight Systems (for host/device timeline)
# AND NCU (for GPU kernel metrics)
#
# Usage: ./run_scv_benefit_analysis.sh
#

set -e
set -u

# Configuration
TARGET_MODEL="meta-llama/Llama-3.2-3B-Instruct"
DRAFT_MODEL="linborui/EAGLE-Llama-3.2-3B-Instruct"
SWEEPS_DIR="sweeps/scv_benefit_analysis"

# Create output directory
mkdir -p "$SWEEPS_DIR"

# Log file
LOG_FILE="$SWEEPS_DIR/scv_benefit_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "SCV Benefit Analysis - What SCV Actually Optimizes"
echo "Started: $(date)"
echo "=========================================="
echo ""
echo "SCV optimizes mask computation by:"
echo "  1. Replacing Python host loop with vectorized GPU kernel"
echo "  2. Reducing kernel launch overhead (N loops → 1 kernel)"
echo "  3. Eliminating CPU-GPU sync points in the loop"
echo "  4. Enabling CUDA graph capture for near-zero dispatch"
echo ""
echo "We measure:"
echo "  - Host CPU time (Nsight Systems)"
echo "  - GPU kernel time (Nsight Systems + NCU)"
echo "  - Kernel launch counts (NCU)"
echo "  - CUDA API overhead (Nsight Systems)"
echo ""

# Function to run with Nsight Systems profiling
run_nsys_profile() {
    local test_name=$1
    local scv_mode=$2
    local scenario=$3
    local temperature=$4
    local requests=$5
    local draft_tokens=$6

    echo ""
    echo "=========================================="
    echo "Nsight Systems Profile: $test_name"
    echo "  SCV Mode: $scv_mode"
    echo "  Scenario: $scenario, Temp: $temperature"
    echo "  Requests: $requests, Draft Tokens: $draft_tokens"
    echo "=========================================="

    local output_file="$SWEEPS_DIR/${test_name}.json"
    local nsys_output="$SWEEPS_DIR/${test_name}_nsys"

    export VLLM_SCV_MODE=$scv_mode
    export VLLM_NWOR_MODE=off
    export VLLM_SCV_PROFILE=1  # Enable NVTX markers
    export TARGET_MODEL=$TARGET_MODEL
    export DRAFT_MODEL=$DRAFT_MODEL

    echo "Running Nsight Systems profiling..."
    if nsys profile \
        --trace=cuda,nvtx,osrt,python \
        --sample=cpu \
        --cpuctxsw=none \
        --python-sampling=true \
        --force-overwrite=true \
        --output="$nsys_output" \
        python3 tools/profiling/run_nwor_microbench.py \
        --scenario "$scenario" \
        --requests "$requests" \
        --batches 2 \
        --draft-tokens "$draft_tokens" \
        --temperature "$temperature" \
        --nwor-modes off \
        --scv-modes "$scv_mode" \
        --max-model-len 8196 \
        --output "$output_file"; then
        echo "✓ Nsight Systems profiling complete: ${nsys_output}.nsys-rep"

        # Generate stats report
        echo ""
        echo "Generating stats summary..."
        nsys stats --report cuda_api_sum,cuda_gpu_kern_sum "$nsys_output.nsys-rep" > "$SWEEPS_DIR/${test_name}_stats.txt" 2>&1 || true

        # Show key metrics
        echo ""
        echo "Key Metrics from Nsight Systems:"
        echo "--------------------------------"
        grep -A 20 "CUDA API Statistics" "$SWEEPS_DIR/${test_name}_stats.txt" 2>/dev/null | head -25 || echo "  (CUDA API stats not available)"
        echo ""
        grep -A 20 "CUDA Kernel Statistics" "$SWEEPS_DIR/${test_name}_stats.txt" 2>/dev/null | head -25 || echo "  (Kernel stats not available)"
    else
        echo "✗ Nsight Systems profiling failed"
    fi
}

# Function to run with NCU profiling (GPU kernel details)
run_ncu_kernel_profile() {
    local test_name=$1
    local scv_mode=$2
    local scenario=$3
    local temperature=$4
    local requests=$5
    local draft_tokens=$6

    echo ""
    echo "=========================================="
    echo "NCU Kernel Profile: $test_name"
    echo "  SCV Mode: $scv_mode"
    echo "=========================================="

    local output_file="$SWEEPS_DIR/${test_name}_ncu.json"

    export VLLM_SCV_MODE=$scv_mode
    export VLLM_NWOR_MODE=off
    export VLLM_SCV_PROFILE=1
    export TARGET_MODEL=$TARGET_MODEL
    export DRAFT_MODEL=$DRAFT_MODEL

    # Try to find the right NCU command
    NCU_CMD=""
    if command -v ncu &> /dev/null; then
        NCU_CMD="ncu"
    elif command -v nv-nsight-cu-cli &> /dev/null; then
        NCU_CMD="nv-nsight-cu-cli"
    else
        echo "⚠ NCU command not found (tried 'ncu' and 'nv-nsight-cu-cli')"
        echo "  Skipping NCU profiling for this test"
        return 1
    fi

    echo "Using NCU command: $NCU_CMD"
    echo "Running NCU kernel profiling (this may take a while)..."

    # NCU metrics specifically for kernel efficiency
    NCU_METRICS="gpu__time_duration.sum,sm__warps_launched.sum,sm__cycles_elapsed.avg,dram__bytes.sum,l1tex__t_bytes.sum"

    if $NCU_CMD \
        --metrics "$NCU_METRICS" \
        --target-processes all \
        --export "$SWEEPS_DIR/${test_name}_ncu" \
        --force-overwrite \
        python3 tools/profiling/run_nwor_microbench.py \
        --scenario "$scenario" \
        --requests "$requests" \
        --batches 1 \
        --draft-tokens "$draft_tokens" \
        --temperature "$temperature" \
        --nwor-modes off \
        --scv-modes "$scv_mode" \
        --max-model-len 8196 \
        --output "$output_file" 2>&1 | tee "$SWEEPS_DIR/${test_name}_ncu.log"; then
        echo "✓ NCU profiling complete"
    else
        echo "⚠ NCU profiling failed (this is expected if ncu command isn't available)"
    fi
}

# Start timer
START_TIME=$(date +%s)

echo ""
echo "=========================================="
echo "Phase 1: Baseline (SCV Off) - Nsight Systems"
echo "=========================================="

run_nsys_profile "baseline_off_small" "off" "short" "0.7" 8 4
run_nsys_profile "baseline_off_medium" "off" "short" "0.7" 16 6
run_nsys_profile "baseline_off_large" "off" "short" "0.7" 32 8

echo ""
echo "=========================================="
echo "Phase 2: SCV Graph Mode - Nsight Systems"
echo "=========================================="

run_nsys_profile "scv_graph_small" "graph" "short" "0.7" 8 4
run_nsys_profile "scv_graph_medium" "graph" "short" "0.7" 16 6
run_nsys_profile "scv_graph_large" "graph" "short" "0.7" 32 8

echo ""
echo "=========================================="
echo "Phase 3: NCU Kernel Analysis (Optional)"
echo "=========================================="

# Only run NCU if command is available
if command -v ncu &> /dev/null || command -v nv-nsight-cu-cli &> /dev/null; then
    echo "NCU command found - running kernel profiling..."
    run_ncu_kernel_profile "ncu_baseline_off" "off" "short" "0.7" 8 4
    run_ncu_kernel_profile "ncu_scv_graph" "graph" "short" "0.7" 8 4
else
    echo "⚠ NCU command not found - skipping kernel profiling"
    echo "  (This is OK - Nsight Systems data is sufficient for SCV analysis)"
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=========================================="
echo "SCV Benefit Analysis Complete!"
echo "=========================================="
echo ""
echo "Elapsed time: ${MINUTES}m ${SECONDS}s"
echo "Results directory: $SWEEPS_DIR"
echo ""
echo "To analyze results:"
echo "  1. Open Nsight Systems reports in GUI:"
echo "     nsight-sys $SWEEPS_DIR/*_nsys.nsys-rep"
echo ""
echo "  2. Compare timeline views:"
echo "     - Baseline (off): Look for Python loops in CPU timeline"
echo "     - SCV Graph: Look for single kernel launch with NVTX marker"
echo ""
echo "  3. Key metrics to compare:"
echo "     - CPU timeline: Python overhead (baseline) vs kernel launch (SCV)"
echo "     - GPU timeline: Kernel time and count"
echo "     - CUDA API: cudaLaunchKernel count and overhead"
echo ""
echo "  4. Check stats files:"
echo "     cat $SWEEPS_DIR/*_stats.txt"
echo ""

echo "=========================================="
echo "INTERPRETATION GUIDE"
echo "=========================================="
cat << 'EOF'

What SCV Should Show:

1. REDUCED HOST CPU TIME
   Baseline: Python loop iterating over requests
   SCV: Single kernel launch, rest is GPU-side

   Expected: 10-100µs reduction in host overhead

2. REDUCED KERNEL LAUNCH COUNT
   Baseline: N kernel launches (one per loop iteration)
   SCV Graph: 1 kernel launch (or even graph replay = 0 launches)

   Expected: N launches → 1 launch (or 0 with graph)

3. IMPROVED PARALLELISM
   Baseline: Sequential processing of requests
   SCV: Parallel processing across all requests

   Expected: Better GPU utilization

4. REDUCED SYNC POINTS
   Baseline: CPU-GPU sync in each loop iteration
   SCV: Single sync after kernel completion

   Expected: Fewer cudaDeviceSynchronize calls

5. GRAPH CAPTURE BENEFIT (SCV Graph mode)
   Baseline: Kernel launch overhead every time
   SCV Graph: Near-zero graph replay overhead

   Expected: <1µs dispatch vs ~5-10µs kernel launch

Look For in Nsight Systems:
- NVTX markers: "scv_compute_mask"
- Python timeline: Function call overhead
- CUDA API timeline: cudaLaunchKernel frequency
- GPU timeline: Kernel duration and occupancy

The benefit scales with:
- Number of requests (more parallel work)
- Number of draft tokens (larger mask computation)
- Batch frequency (graph capture amortization)

EOF

echo ""
echo "Done! Review Nsight Systems reports to see SCV's actual benefits."
echo ""
