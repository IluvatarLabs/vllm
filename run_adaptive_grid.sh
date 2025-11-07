#!/bin/bash
# Experiment Grid 1: Adaptive Draft Length (Latency Focus)
# Tests adaptive draft length across different workload characteristics
# Total: 16 runs × NUM_SEEDS (8 workloads × 2 adaptive settings × seeds)

set -e

# Default models (can be overridden via environment variables)
TARGET_MODEL="${TARGET_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
DRAFT_MODEL="${DRAFT_MODEL:-yuhuili/EAGLE-LLaMA3.1-Instruct-8B}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
NUM_SEEDS="${NUM_SEEDS:-3}"
SCENARIO="${SCENARIO:-short}"

# Output directory for logs
OUTPUT_DIR="sweeps/logs"
LOG_FILE="${OUTPUT_DIR}/experiment.log"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

# Run benchmark function
run_benchmark() {
    local seed=$1
    local run_id=$2
    local desc=$3
    local requests=$4
    local tokens=$5
    local temp=$6
    local draft=$7
    local adaptive=$8

    # Build descriptive output filename
    local target_name=$(basename "${TARGET_MODEL}")
    local model_pair="${target_name}"

    local output_file="run${run_id}_r${requests}_t${tokens}_temp${temp}_d${draft}_adaptive${adaptive}.json"
    local expected_output="sweeps/${model_pair}/${SCENARIO}/seed_${seed}/${output_file}"

    # Skip if already completed
    if [ -f "${expected_output}" ]; then
        log "SKIP: Seed ${seed}, Run ${run_id} (${desc}, adaptive=${adaptive}) - already completed"
        return 0
    fi

    log "START: Seed ${seed}, Run ${run_id} - ${desc}, adaptive=${adaptive}"

    if python3 tools/profiling/run_nwor_microbench.py \
        --target-model "${TARGET_MODEL}" \
        --draft-model "${DRAFT_MODEL}" \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
        --scenario "${SCENARIO}" \
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
        --seed "${seed}" \
        --output "${output_file}" 2>&1 | tee -a "${LOG_FILE}"; then
        log "COMPLETE: Seed ${seed}, Run ${run_id} (${desc}, adaptive=${adaptive})"
    else
        log "ERROR: Seed ${seed}, Run ${run_id} (${desc}, adaptive=${adaptive}) failed"
        return 1
    fi
}

# Run vanilla (no speculation) benchmark
run_vanilla_benchmark() {
    local seed=$1
    local run_id=$2
    local desc=$3
    local requests=$4
    local tokens=$5
    local temp=$6

    # Build descriptive output filename for vanilla
    local target_name=$(basename "${TARGET_MODEL}")
    local model_pair="${target_name}"

    local output_file="run${run_id}_r${requests}_t${tokens}_temp${temp}_vanilla.json"
    local expected_output="sweeps/${model_pair}/${SCENARIO}/seed_${seed}/${output_file}"

    # Skip if already completed
    if [ -f "${expected_output}" ]; then
        log "SKIP: Seed ${seed}, Run ${run_id} (${desc}, vanilla) - already completed"
        return 0
    fi

    log "START: Seed ${seed}, Run ${run_id} - ${desc}, vanilla (no speculation)"

    if python3 tools/profiling/run_nwor_microbench.py \
        --target-model "${TARGET_MODEL}" \
        --draft-model "${DRAFT_MODEL}" \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
        --scenario "${SCENARIO}" \
        --requests "${requests}" \
        --batches 10 \
        --warmup-steps 2 \
        --temperature "${temp}" \
        --top-p 0.9 \
        --max-new-tokens "${tokens}" \
        --max-model-len 4096 \
        --prompt-count 3200 \
        --measure-steps 6 \
        --prompt-shuffle-seed 42 \
        --no-speculation \
        --seed "${seed}" \
        --output "${output_file}" 2>&1 | tee -a "${LOG_FILE}"; then
        log "COMPLETE: Seed ${seed}, Run ${run_id} (${desc}, vanilla)"
    else
        log "ERROR: Seed ${seed}, Run ${run_id} (${desc}, vanilla) failed"
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
log "Scenario: ${SCENARIO}"
log "Number of Seeds: ${NUM_SEEDS}"
log "Speculation Runs: $((16 * NUM_SEEDS)) (8 workloads × 2 adaptive × ${NUM_SEEDS} seeds)"
log "Vanilla Runs: $((6 * NUM_SEEDS)) (6 workloads × ${NUM_SEEDS} seeds)"
log "Total Runs: $(((16 + 6) * NUM_SEEDS))"
log "Output Directory: sweeps/"
log "========================================"
log ""

# Run experiments with multiple seeds
for seed in $(seq 0 $((NUM_SEEDS - 1))); do
    log "========================================"
    log "Starting seed ${seed}/${NUM_SEEDS}"
    log "========================================"

    # Track run counter
    run_counter=1

    # Run 1: Short burst (baseline chat turn)
    for adaptive in 0 1; do
        run_benchmark "${seed}" "${run_counter}" "Short burst" 12 64 0.0 10 "${adaptive}" || true
        ((run_counter++))
    done
    run_vanilla_benchmark "${seed}" "${run_counter}" "Short burst" 12 64 0.0 || true
    ((run_counter++))

    # Run 2: Short burst with sampling
    for adaptive in 0 1; do
        run_benchmark "${seed}" "${run_counter}" "Short burst creative" 12 64 0.7 10 "${adaptive}" || true
        ((run_counter++))
    done
    run_vanilla_benchmark "${seed}" "${run_counter}" "Short burst creative" 12 64 0.7 || true
    ((run_counter++))

    # Run 3: Medium interactive (⭐ MOST REPRESENTATIVE)
    for adaptive in 0 1; do
        run_benchmark "${seed}" "${run_counter}" "Medium interactive ⭐" 36 128 0.0 10 "${adaptive}" || true
        ((run_counter++))
    done
    run_vanilla_benchmark "${seed}" "${run_counter}" "Medium interactive ⭐" 36 128 0.0 || true
    ((run_counter++))

    # Run 4: Medium interactive creative
    for adaptive in 0 1; do
        run_benchmark "${seed}" "${run_counter}" "Medium creative" 36 128 0.7 10 "${adaptive}" || true
        ((run_counter++))
    done
    run_vanilla_benchmark "${seed}" "${run_counter}" "Medium creative" 36 128 0.7 || true
    ((run_counter++))

    # Run 5: Large batch (high throughput)
    for adaptive in 0 1; do
        run_benchmark "${seed}" "${run_counter}" "Large batch throughput" 60 64 0.0 10 "${adaptive}" || true
        ((run_counter++))
    done
    run_vanilla_benchmark "${seed}" "${run_counter}" "Large batch throughput" 60 64 0.0 || true
    ((run_counter++))

    # Run 6: Long form generation
    for adaptive in 0 1; do
        run_benchmark "${seed}" "${run_counter}" "Long form generation" 36 256 0.0 10 "${adaptive}" || true
        ((run_counter++))
    done
    run_vanilla_benchmark "${seed}" "${run_counter}" "Long form generation" 36 256 0.0 || true
    ((run_counter++))

    # Run 7: Small draft baseline (no vanilla - draft length variation)
    for adaptive in 0 1; do
        run_benchmark "${seed}" "${run_counter}" "Small draft baseline" 36 128 0.0 5 "${adaptive}" || true
        ((run_counter++))
    done

    # Run 8: Large draft aggressive (no vanilla - draft length variation)
    for adaptive in 0 1; do
        run_benchmark "${seed}" "${run_counter}" "Large draft aggressive" 36 128 0.0 15 "${adaptive}" || true
        ((run_counter++))
    done

    log "Completed seed ${seed}/${NUM_SEEDS}"
    log ""
done

log ""
log "========================================"
log "Adaptive Grid Complete!"
log "Results: sweeps/"
log "========================================"
log ""
log "Key results to check:"
log "  - Run 5-6 (⭐): Medium interactive baseline"
log "  - Compare latency_avg_s between adaptive=0 vs adaptive=1"
log "  - Check spec_num_draft_tokens reduction"
