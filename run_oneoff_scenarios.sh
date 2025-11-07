#!/bin/bash
# One-Off Representative Scenarios
# Real-world deployment scenarios that showcase specific benefits
# Total: 4 scenarios × NUM_SEEDS

set -e

# Default models (can be overridden via environment variables)
TARGET_MODEL="${TARGET_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
DRAFT_MODEL="${DRAFT_MODEL:-yuhuili/EAGLE-LLaMA3.1-Instruct-8B}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
NUM_SEEDS="${NUM_SEEDS:-3}"
SCENARIO="${SCENARIO:-coding}"

# Output directory for logs
OUTPUT_DIR="sweeps/logs"
LOG_FILE="${OUTPUT_DIR}/oneoff_scenarios.log"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

# Scenario A: Code Completion (Adaptive Shines)
run_scenario_a() {
    local seed=$1

    log "Running Scenario A: Code Completion"
    log "  Why: Short responses, high acceptance → adaptive picks longer drafts"

    local output_file="scenario_a_r24_t64_temp0.0_d10_adaptive1.json"
    local target_name=$(basename "${TARGET_MODEL}")
    local model_pair="${target_name}"
    local expected_output="sweeps/${model_pair}/${SCENARIO}/seed_${seed}/${output_file}"

    if [ -f "${expected_output}" ]; then
        log "SKIP: Seed ${seed}, Scenario A - already completed"
    else
        if python3 tools/profiling/run_nwor_microbench.py \
            --target-model "${TARGET_MODEL}" \
            --draft-model "${DRAFT_MODEL}" \
            --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
            --scenario "${SCENARIO}" \
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
            --seed "${seed}" \
            --output "${output_file}" 2>&1 | tee -a "${LOG_FILE}"; then
            log "COMPLETE: Seed ${seed}, Scenario A"
        else
            log "ERROR: Seed ${seed}, Scenario A failed"
        fi
    fi

    log ""
}

# Scenario B: Customer Service Chatbot (Most Common Deployment)
run_scenario_b() {
    local seed=$1

    log "Running Scenario B: Customer Service Chatbot ⭐"
    log "  Why: Moderate responses, controlled creativity → 80% of production use"

    local output_file="scenario_b_r36_t128_temp0.3_d10_adaptive1.json"
    local target_name=$(basename "${TARGET_MODEL}")
    local model_pair="${target_name}"
    local expected_output="sweeps/${model_pair}/${SCENARIO}/seed_${seed}/${output_file}"

    if [ -f "${expected_output}" ]; then
        log "SKIP: Seed ${seed}, Scenario B - already completed"
    else
        if python3 tools/profiling/run_nwor_microbench.py \
            --target-model "${TARGET_MODEL}" \
            --draft-model "${DRAFT_MODEL}" \
            --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
            --scenario "${SCENARIO}" \
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
            --seed "${seed}" \
            --output "${output_file}" 2>&1 | tee -a "${LOG_FILE}"; then
            log "COMPLETE: Seed ${seed}, Scenario B"
        else
            log "ERROR: Seed ${seed}, Scenario B failed"
        fi
    fi

    log ""
}

# Scenario C: High-Throughput Server (Bandwidth Matters)
run_scenario_c() {
    local seed=$1

    log "Running Scenario C: High-Throughput Server [NCU ENABLED]"
    log "  Why: Large batch, bandwidth bottleneck → early exit + adaptive helps"
    log "  [This will take ~20-30 minutes]"

    local output_file="scenario_c_r60_t128_temp0.0_d10_adaptive1_thresh0.5.json"
    local target_name=$(basename "${TARGET_MODEL}")
    local model_pair="${target_name}"
    local expected_output="sweeps/${model_pair}/${SCENARIO}/seed_${seed}/${output_file}"

    if [ -f "${expected_output}" ]; then
        log "SKIP: Seed ${seed}, Scenario C - already completed"
    else
        if python3 tools/profiling/run_nwor_microbench.py \
            --target-model "${TARGET_MODEL}" \
            --draft-model "${DRAFT_MODEL}" \
            --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
            --scenario "${SCENARIO}" \
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
            --seed "${seed}" \
            --enable-ncu \
            --ncu-metrics "dram__bytes_write.sum,dram__bytes_read.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_bytes.sum" \
            --output "${output_file}" 2>&1 | tee -a "${LOG_FILE}"; then
            log "COMPLETE: Seed ${seed}, Scenario C"
        else
            log "ERROR: Seed ${seed}, Scenario C failed"
        fi
    fi

    log ""
}

# Scenario D: Creative Writing (Low Acceptance)
run_scenario_d() {
    local seed=$1

    log "Running Scenario D: Creative Writing"
    log "  Why: Long sequences, creative sampling → adaptive reduces to 2-5 tokens"

    local output_file="scenario_d_r24_t256_temp0.7_d10_adaptive1.json"
    local target_name=$(basename "${TARGET_MODEL}")
    local model_pair="${target_name}"
    local expected_output="sweeps/${model_pair}/${SCENARIO}/seed_${seed}/${output_file}"

    if [ -f "${expected_output}" ]; then
        log "SKIP: Seed ${seed}, Scenario D - already completed"
    else
        if python3 tools/profiling/run_nwor_microbench.py \
            --target-model "${TARGET_MODEL}" \
            --draft-model "${DRAFT_MODEL}" \
            --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
            --scenario "${SCENARIO}" \
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
            --seed "${seed}" \
            --output "${output_file}" 2>&1 | tee -a "${LOG_FILE}"; then
            log "COMPLETE: Seed ${seed}, Scenario D"
        else
            log "ERROR: Seed ${seed}, Scenario D failed"
        fi
    fi

    log ""
}

# Print configuration
log "========================================"
log "One-Off Representative Scenarios"
log "========================================"
log "Target Model: ${TARGET_MODEL}"
log "Draft Model: ${DRAFT_MODEL}"
log "Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}"
log "Scenario: ${SCENARIO}"
log "Number of Seeds: ${NUM_SEEDS}"
log "Total Runs: $((4 * NUM_SEEDS))"
log "Output Directory: sweeps/"
log "========================================"
log ""

# Run experiments with multiple seeds
for seed in $(seq 0 $((NUM_SEEDS - 1))); do
    log "========================================"
    log "Starting seed ${seed}/${NUM_SEEDS}"
    log "========================================"

    run_scenario_a "${seed}" || true
    run_scenario_b "${seed}" || true
    run_scenario_c "${seed}" || true
    run_scenario_d "${seed}" || true

    log "Completed seed ${seed}/${NUM_SEEDS}"
    log ""
done

log ""
log "========================================"
log "One-Off Scenarios Complete!"
log "Results: sweeps/"
log "========================================"
log ""
log "Scenario Highlights:"
log "  A - Code Completion: Best adaptive benefit (short, high acceptance)"
log "  B - Chatbot ⭐: Most representative of production deployments"
log "  C - High Throughput: Shows combined benefit + bandwidth savings"
log "  D - Creative Writing: Adaptive handles low acceptance gracefully"
log ""
