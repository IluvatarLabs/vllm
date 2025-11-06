#!/bin/bash
# One-Off Representative Scenarios
# Real-world deployment scenarios that showcase specific benefits
# Total: 4 scenarios

set -e

# Default models (can be overridden via environment variables)
TARGET_MODEL="${TARGET_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
DRAFT_MODEL="${DRAFT_MODEL:-yuhuili/EAGLE-LLaMA3.1-Instruct-8B}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"

# Output directory
OUTPUT_DIR="sweeps/oneoff_scenarios"
LOG_FILE="${OUTPUT_DIR}/experiment.log"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

# Print configuration
log "========================================"
log "One-Off Representative Scenarios"
log "========================================"
log "Target Model: ${TARGET_MODEL}"
log "Draft Model: ${DRAFT_MODEL}"
log "Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}"
log "Output Directory: ${OUTPUT_DIR}"
log "========================================"
log ""

# Scenario A: Code Completion (Adaptive Shines)
log "Running Scenario A: Code Completion"
log "  Why: Short responses, high acceptance → adaptive picks longer drafts"

output_file="${OUTPUT_DIR}/scenario_a_code_completion.json"
if [ -f "${output_file}" ]; then
    log "SKIP: Scenario A - already completed"
else
    if python3 tools/profiling/run_nwor_microbench.py \
        --target-model "${TARGET_MODEL}" \
        --draft-model "${DRAFT_MODEL}" \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
        --scenario short \
        --requests 24 \
        --draft-tokens 10 \
        --batches 10 \
        --warmup-steps 2 \
        --temperature 0.0 \
        --top-p 0.9 \
        --max-new-tokens 64 \
        --max-model-len 4096 \
        --prompt-count 3200 \
        --measure-steps 6 \
        --prompt-shuffle-seed 42 \
        --adaptive-draft-length 1 \
        --confidence-threshold 0.0 \
        --output "${output_file}" 2>&1 | tee -a "${LOG_FILE}"; then
        log "COMPLETE: Scenario A"
    else
        log "ERROR: Scenario A failed"
    fi
fi

log ""

# Scenario B: Customer Service Chatbot (Most Common Deployment)
log "Running Scenario B: Customer Service Chatbot ⭐"
log "  Why: Moderate responses, controlled creativity → 80% of production use"

output_file="${OUTPUT_DIR}/scenario_b_chatbot.json"
if [ -f "${output_file}" ]; then
    log "SKIP: Scenario B - already completed"
else
    if python3 tools/profiling/run_nwor_microbench.py \
        --target-model "${TARGET_MODEL}" \
        --draft-model "${DRAFT_MODEL}" \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
        --scenario short \
        --requests 36 \
        --draft-tokens 10 \
        --batches 10 \
        --warmup-steps 2 \
        --temperature 0.3 \
        --top-p 0.9 \
        --max-new-tokens 128 \
        --max-model-len 4096 \
        --prompt-count 3200 \
        --measure-steps 6 \
        --prompt-shuffle-seed 42 \
        --adaptive-draft-length 1 \
        --confidence-threshold 0.0 \
        --output "${output_file}" 2>&1 | tee -a "${LOG_FILE}"; then
        log "COMPLETE: Scenario B"
    else
        log "ERROR: Scenario B failed"
    fi
fi

log ""

# Scenario C: High-Throughput Server (Bandwidth Matters)
log "Running Scenario C: High-Throughput Server [NCU ENABLED]"
log "  Why: Large batch, bandwidth bottleneck → early exit + adaptive helps"
log "  [This will take ~20-30 minutes]"

output_file="${OUTPUT_DIR}/scenario_c_high_throughput.json"
if [ -f "${output_file}" ]; then
    log "SKIP: Scenario C - already completed"
else
    if python3 tools/profiling/run_nwor_microbench.py \
        --target-model "${TARGET_MODEL}" \
        --draft-model "${DRAFT_MODEL}" \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
        --scenario short \
        --requests 60 \
        --draft-tokens 10 \
        --batches 5 \
        --warmup-steps 2 \
        --temperature 0.0 \
        --top-p 0.9 \
        --max-new-tokens 128 \
        --max-model-len 4096 \
        --prompt-count 3200 \
        --measure-steps 6 \
        --prompt-shuffle-seed 42 \
        --adaptive-draft-length 1 \
        --confidence-threshold 0.5 \
        --enable-ncu \
        --ncu-metrics "dram__bytes_write.sum,dram__bytes_read.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_bytes.sum" \
        --output "${output_file}" 2>&1 | tee -a "${LOG_FILE}"; then
        log "COMPLETE: Scenario C"
    else
        log "ERROR: Scenario C failed"
    fi
fi

log ""

# Scenario D: Creative Writing (Low Acceptance)
log "Running Scenario D: Creative Writing"
log "  Why: Long sequences, creative sampling → adaptive reduces to 2-5 tokens"

output_file="${OUTPUT_DIR}/scenario_d_creative_writing.json"
if [ -f "${output_file}" ]; then
    log "SKIP: Scenario D - already completed"
else
    if python3 tools/profiling/run_nwor_microbench.py \
        --target-model "${TARGET_MODEL}" \
        --draft-model "${DRAFT_MODEL}" \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
        --scenario short \
        --requests 24 \
        --draft-tokens 10 \
        --batches 10 \
        --warmup-steps 2 \
        --temperature 0.7 \
        --top-p 0.9 \
        --max-new-tokens 256 \
        --max-model-len 4096 \
        --prompt-count 3200 \
        --measure-steps 6 \
        --prompt-shuffle-seed 42 \
        --adaptive-draft-length 1 \
        --confidence-threshold 0.0 \
        --output "${output_file}" 2>&1 | tee -a "${LOG_FILE}"; then
        log "COMPLETE: Scenario D"
    else
        log "ERROR: Scenario D failed"
    fi
fi

log ""
log "========================================"
log "One-Off Scenarios Complete!"
log "Results: ${OUTPUT_DIR}"
log "========================================"
log ""
log "Scenario Highlights:"
log "  A - Code Completion: Best adaptive benefit (short, high acceptance)"
log "  B - Chatbot ⭐: Most representative of production deployments"
log "  C - High Throughput: Shows combined benefit + bandwidth savings"
log "  D - Creative Writing: Adaptive handles low acceptance gracefully"
