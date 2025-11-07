#!/bin/bash
# Experiment Grid 2: Confidence Early Exit (Bandwidth Focus)
# Tests confidence-based early exit with NCU bandwidth profiling
# Total: 7 runs × NUM_SEEDS (different thresholds and workloads × seeds)

set -e

# Default models (can be overridden via environment variables)
TARGET_MODEL="${TARGET_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
DRAFT_MODEL="${DRAFT_MODEL:-yuhuili/EAGLE-LLaMA3.1-Instruct-8B}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
NUM_SEEDS="${NUM_SEEDS:-3}"
SCENARIO="${SCENARIO:-coding}"

# Output directory for logs
OUTPUT_DIR="sweeps/logs"
LOG_FILE="${OUTPUT_DIR}/early_exit_experiment.log"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

# Run benchmark function with NCU
run_benchmark() {
    local seed=$1
    local run_id=$2
    local desc=$3
    local requests=$4
    local tokens=$5
    local temp=$6
    local threshold=$7

    # Build descriptive output filename
    local target_name=$(basename "${TARGET_MODEL}")
    local model_pair="${target_name}"

    # For early exit, we always use adaptive=0, threshold varies
    local output_file="run${run_id}_r${requests}_t${tokens}_temp${temp}_thresh${threshold}.json"
    local expected_output="sweeps/${model_pair}/${SCENARIO}/seed_${seed}/${output_file}"

    # Skip if already completed
    if [ -f "${expected_output}" ]; then
        log "SKIP: Seed ${seed}, Run ${run_id} (${desc}, threshold=${threshold}) - already completed"
        return 0
    fi

    log "START: Seed ${seed}, Run ${run_id}/7 - ${desc}, threshold=${threshold}"
    log "  [NCU PROFILING ENABLED - This will take ~20-30 minutes]"

    if python3 tools/profiling/run_nwor_microbench.py \
        --target-model "${TARGET_MODEL}" \
        --draft-model "${DRAFT_MODEL}" \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
        --scenario "${SCENARIO}" \
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
        --seed "${seed}" \
        --enable-ncu \
        --ncu-metrics "dram__bytes_write.sum,dram__bytes_read.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_bytes.sum" \
        --output "${output_file}" 2>&1 | tee -a "${LOG_FILE}"; then
        log "COMPLETE: Seed ${seed}, Run ${run_id} (${desc}, threshold=${threshold})"

        # Extract and log key metrics
        if [ -f "${expected_output}" ]; then
            local dram_writes=$(grep -o '"dram__bytes_write.sum": [0-9.e+]*' "${expected_output}" | head -1 | grep -o '[0-9.e+]*$' || echo "N/A")
            log "  -> DRAM writes: ${dram_writes} bytes"
        fi
    else
        log "ERROR: Seed ${seed}, Run ${run_id} (${desc}, threshold=${threshold}) failed"
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
log "Scenario: ${SCENARIO}"
log "Number of Seeds: ${NUM_SEEDS}"
log "Total Runs: $((7 * NUM_SEEDS))"
log "Output Directory: sweeps/"
log "NCU Profiling: ENABLED"
log "========================================"
log ""
log "WARNING: Each run takes ~20-30 minutes with NCU"
log "Total estimated time per seed: 2.5-3.5 hours"
log "Total estimated time: $((3 * NUM_SEEDS)) hours"
log ""

# Run experiments with multiple seeds
for seed in $(seq 0 $((NUM_SEEDS - 1))); do
    log "========================================"
    log "Starting seed ${seed}/${NUM_SEEDS}"
    log "========================================"

    # Run 1: Baseline (no early exit)
    run_benchmark "${seed}" 1 "Baseline (no early exit)" 36 128 0.0 0.0 || true

    # Run 2: Conservative threshold
    run_benchmark "${seed}" 2 "Conservative threshold" 36 128 0.0 0.3 || true

    # Run 3: Moderate threshold (⭐ DEFAULT)
    run_benchmark "${seed}" 3 "Moderate threshold ⭐" 36 128 0.0 0.5 || true

    # Run 4: Aggressive threshold
    run_benchmark "${seed}" 4 "Aggressive threshold" 36 128 0.0 0.7 || true

    # Run 5: Creative baseline
    run_benchmark "${seed}" 5 "Creative baseline" 36 128 0.7 0.0 || true

    # Run 6: Creative + early exit
    run_benchmark "${seed}" 6 "Creative + early exit" 36 128 0.7 0.5 || true

    # Run 7: Long sequence
    run_benchmark "${seed}" 7 "Long sequence" 36 256 0.0 0.5 || true

    log "Completed seed ${seed}/${NUM_SEEDS}"
    log ""
done

log ""
log "========================================"
log "Early Exit Grid Complete!"
log "Results: sweeps/"
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
