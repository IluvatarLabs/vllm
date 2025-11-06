#!/bin/bash
# Experiment Grid 2: Confidence Early Exit (Bandwidth Focus)
# Tests confidence-based early exit with NCU bandwidth profiling
# Total: 7 runs (different thresholds and workloads)

set -e

# Default models (can be overridden via environment variables)
TARGET_MODEL="${TARGET_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
DRAFT_MODEL="${DRAFT_MODEL:-yuhuili/EAGLE-LLaMA3.1-Instruct-8B}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"

# Output directory
OUTPUT_DIR="sweeps/early_exit_grid"
LOG_FILE="${OUTPUT_DIR}/experiment.log"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

# Run benchmark function with NCU
run_benchmark() {
    local run_id=$1
    local desc=$2
    local requests=$3
    local tokens=$4
    local temp=$5
    local threshold=$6

    local output_file="${OUTPUT_DIR}/run${run_id}_r${requests}_t${tokens}_temp${temp}_thresh${threshold}.json"

    # Skip if already completed
    if [ -f "${output_file}" ]; then
        log "SKIP: Run ${run_id} (${desc}, threshold=${threshold}) - already completed"
        return 0
    fi

    log "START: Run ${run_id}/7 - ${desc}, threshold=${threshold}"
    log "  [NCU PROFILING ENABLED - This will take ~20-30 minutes]"

    if python3 tools/profiling/run_nwor_microbench.py \
        --target-model "${TARGET_MODEL}" \
        --draft-model "${DRAFT_MODEL}" \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
        --scenario short \
        --requests "${requests}" \
        --draft-tokens 10 \
        --batches 5 \
        --warmup-steps 2 \
        --temperature "${temp}" \
        --top-p 0.9 \
        --max-new-tokens "${tokens}" \
        --max-model-len 4096 \
        --prompt-count 3200 \
        --measure-steps 6 \
        --prompt-shuffle-seed 42 \
        --adaptive-draft-length 0 \
        --confidence-threshold "${threshold}" \
        --enable-ncu \
        --ncu-metrics "dram__bytes_write.sum,dram__bytes_read.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_bytes.sum" \
        --output "${output_file}" 2>&1 | tee -a "${LOG_FILE}"; then
        log "COMPLETE: Run ${run_id} (${desc}, threshold=${threshold})"

        # Extract and log key metrics
        if [ -f "${output_file}" ]; then
            local dram_writes=$(grep -o '"dram__bytes_write.sum": [0-9.e+]*' "${output_file}" | head -1 | grep -o '[0-9.e+]*$' || echo "N/A")
            log "  -> DRAM writes: ${dram_writes} bytes"
        fi
    else
        log "ERROR: Run ${run_id} (${desc}, threshold=${threshold}) failed"
        return 1
    fi
}

# Print configuration
log "========================================"
log "Confidence Early Exit Experiment Grid"
log "========================================"
log "Target Model: ${TARGET_MODEL}"
log "Draft Model: ${DRAFT_MODEL}"
log "Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}"
log "Output Directory: ${OUTPUT_DIR}"
log "NCU Profiling: ENABLED"
log "========================================"
log ""
log "WARNING: Each run takes ~20-30 minutes with NCU"
log "Total estimated time: 2.5-3.5 hours"
log ""

# Run 1: Baseline (no early exit)
run_benchmark 1 "Baseline (no early exit)" 36 128 0.0 0.0 || true

# Run 2: Conservative threshold
run_benchmark 2 "Conservative threshold" 36 128 0.0 0.3 || true

# Run 3: Moderate threshold (⭐ DEFAULT)
run_benchmark 3 "Moderate threshold ⭐" 36 128 0.0 0.5 || true

# Run 4: Aggressive threshold
run_benchmark 4 "Aggressive threshold" 36 128 0.0 0.7 || true

# Run 5: Creative baseline
run_benchmark 5 "Creative baseline" 36 128 0.7 0.0 || true

# Run 6: Creative + early exit
run_benchmark 6 "Creative + early exit" 36 128 0.7 0.5 || true

# Run 7: Long sequence
run_benchmark 7 "Long sequence" 36 256 0.0 0.5 || true

log ""
log "========================================"
log "Early Exit Grid Complete!"
log "Results: ${OUTPUT_DIR}"
log "========================================"
log ""
log "Key results to check:"
log "  - Run 1 vs 3 (⭐): Baseline vs default threshold"
log "  - Compare dram__bytes_write.sum (bandwidth savings)"
log "  - Compare latency_avg_s (overhead cost)"
log "  - Check instrumentation logs for tokens_stopped count"
log ""
log "Next steps:"
log "  1. Compare DRAM writes across thresholds"
log "  2. Calculate bandwidth savings percentage"
log "  3. Evaluate if savings justify latency cost"
