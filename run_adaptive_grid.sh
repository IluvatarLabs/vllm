#!/bin/bash
# Experiment Grid 1: Adaptive Draft Length (Latency Focus)
# Tests adaptive draft length across different workload characteristics
# Total: 16 runs (8 workloads × 2 adaptive settings)

set -e

# Default models (can be overridden via environment variables)
TARGET_MODEL="${TARGET_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
DRAFT_MODEL="${DRAFT_MODEL:-yuhuili/EAGLE-LLaMA3.1-Instruct-8B}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"

# Output directory
OUTPUT_DIR="sweeps/adaptive_grid"
LOG_FILE="${OUTPUT_DIR}/experiment.log"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

# Run benchmark function
run_benchmark() {
    local run_id=$1
    local desc=$2
    local requests=$3
    local tokens=$4
    local temp=$5
    local draft=$6
    local adaptive=$7

    local output_file="${OUTPUT_DIR}/run${run_id}_r${requests}_t${tokens}_temp${temp}_d${draft}_adaptive${adaptive}.json"

    # Skip if already completed
    if [ -f "${output_file}" ]; then
        log "SKIP: Run ${run_id} (${desc}, adaptive=${adaptive}) - already completed"
        return 0
    fi

    log "START: Run ${run_id}/16 - ${desc}, adaptive=${adaptive}"

    if python3 tools/profiling/run_nwor_microbench.py \
        --target-model "${TARGET_MODEL}" \
        --draft-model "${DRAFT_MODEL}" \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
        --scenario short \
        --requests "${requests}" \
        --draft-tokens "${draft}" \
        --batches 10 \
        --warmup-steps 2 \
        --temperature "${temp}" \
        --top-p 0.9 \
        --max-new-tokens "${tokens}" \
        --max-model-len 4096 \
        --prompt-count 3200 \
        --measure-steps 6 \
        --prompt-shuffle-seed 42 \
        --adaptive-draft-length "${adaptive}" \
        --confidence-threshold 0.0 \
        --output "${output_file}" 2>&1 | tee -a "${LOG_FILE}"; then
        log "COMPLETE: Run ${run_id} (${desc}, adaptive=${adaptive})"
    else
        log "ERROR: Run ${run_id} (${desc}, adaptive=${adaptive}) failed"
        return 1
    fi
}

# Print configuration
log "========================================"
log "Adaptive Draft Length Experiment Grid"
log "========================================"
log "Target Model: ${TARGET_MODEL}"
log "Draft Model: ${DRAFT_MODEL}"
log "Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}"
log "Output Directory: ${OUTPUT_DIR}"
log "========================================"
log ""

# Track run counter
run_counter=1

# Run 1: Short burst (baseline chat turn)
for adaptive in 0 1; do
    run_benchmark "${run_counter}" "Short burst" 12 64 0.0 10 "${adaptive}" || true
    ((run_counter++))
done

# Run 2: Short burst with sampling
for adaptive in 0 1; do
    run_benchmark "${run_counter}" "Short burst creative" 12 64 0.7 10 "${adaptive}" || true
    ((run_counter++))
done

# Run 3: Medium interactive (⭐ MOST REPRESENTATIVE)
for adaptive in 0 1; do
    run_benchmark "${run_counter}" "Medium interactive ⭐" 36 128 0.0 10 "${adaptive}" || true
    ((run_counter++))
done

# Run 4: Medium interactive creative
for adaptive in 0 1; do
    run_benchmark "${run_counter}" "Medium creative" 36 128 0.7 10 "${adaptive}" || true
    ((run_counter++))
done

# Run 5: Large batch (high throughput)
for adaptive in 0 1; do
    run_benchmark "${run_counter}" "Large batch throughput" 60 64 0.0 10 "${adaptive}" || true
    ((run_counter++))
done

# Run 6: Long form generation
for adaptive in 0 1; do
    run_benchmark "${run_counter}" "Long form generation" 36 256 0.0 10 "${adaptive}" || true
    ((run_counter++))
done

# Run 7: Small draft baseline
for adaptive in 0 1; do
    run_benchmark "${run_counter}" "Small draft baseline" 36 128 0.0 5 "${adaptive}" || true
    ((run_counter++))
done

# Run 8: Large draft aggressive
for adaptive in 0 1; do
    run_benchmark "${run_counter}" "Large draft aggressive" 36 128 0.0 15 "${adaptive}" || true
    ((run_counter++))
done

log ""
log "========================================"
log "Adaptive Grid Complete!"
log "Results: ${OUTPUT_DIR}"
log "========================================"
log ""
log "Key results to check:"
log "  - Run 5-6 (⭐): Medium interactive baseline"
log "  - Compare latency_avg_s between adaptive=0 vs adaptive=1"
log "  - Check spec_num_draft_tokens reduction"
