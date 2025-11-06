#!/bin/bash
# Master Script: Run All NWOR Experiments
# Orchestrates all experiment grids and one-off scenarios

set -e

# Default models (can be overridden via environment variables)
TARGET_MODEL="${TARGET_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
DRAFT_MODEL="${DRAFT_MODEL:-yuhuili/EAGLE-LLaMA3.1-Instruct-8B}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"

# Export for child scripts
export TARGET_MODEL
export DRAFT_MODEL
export TENSOR_PARALLEL_SIZE

# Master log
MASTER_LOG="sweeps/master_experiment.log"
mkdir -p sweeps

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${MASTER_LOG}"
}

log "========================================"
log "NWOR Complete Experiment Suite"
log "========================================"
log "Configuration:"
log "  Target Model: ${TARGET_MODEL}"
log "  Draft Model: ${DRAFT_MODEL}"
log "  Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}"
log "========================================"
log ""

# Print time estimates
log "Estimated Time:"
log "  - Adaptive Grid (16 runs):      ~1 hour"
log "  - Early Exit Grid (7 runs):     ~2.5 hours (NCU enabled)"
log "  - One-Off Scenarios (4 runs):   ~1 hour"
log "  - Total:                         ~4.5 hours"
log ""

read -p "Press Enter to start, or Ctrl+C to cancel..."
log ""

# Phase 1: Adaptive Draft Length Grid
log "========================================"
log "Phase 1: Adaptive Draft Length Grid"
log "========================================"
if bash run_adaptive_grid.sh; then
    log "✓ Phase 1 Complete"
else
    log "✗ Phase 1 Failed (continuing anyway)"
fi
log ""

# Phase 2: Confidence Early Exit Grid
log "========================================"
log "Phase 2: Confidence Early Exit Grid"
log "========================================"
log "WARNING: This phase includes NCU profiling (~2.5 hours)"
if bash run_early_exit_grid.sh; then
    log "✓ Phase 2 Complete"
else
    log "✗ Phase 2 Failed (continuing anyway)"
fi
log ""

# Phase 3: One-Off Representative Scenarios
log "========================================"
log "Phase 3: One-Off Representative Scenarios"
log "========================================"
if bash run_oneoff_scenarios.sh; then
    log "✓ Phase 3 Complete"
else
    log "✗ Phase 3 Failed"
fi
log ""

# Final summary
log "========================================"
log "All Experiments Complete!"
log "========================================"
log ""
log "Results Location:"
log "  - Adaptive Grid:     sweeps/adaptive_grid/"
log "  - Early Exit Grid:   sweeps/early_exit_grid/"
log "  - One-Off Scenarios: sweeps/oneoff_scenarios/"
log ""
log "Next Steps for Analysis:"
log ""
log "1. Adaptive Draft Length Analysis:"
log "   - Compare latency_avg_s: adaptive=0 vs adaptive=1"
log "   - Check spec_num_draft_tokens reduction"
log "   - Calculate % improvement across workloads"
log ""
log "2. Confidence Early Exit Analysis:"
log "   - Extract dram__bytes_write.sum from NCU metrics"
log "   - Compare bandwidth: threshold=0.0 vs 0.5"
log "   - Check instrumentation logs for tokens stopped"
log "   - Evaluate latency overhead vs bandwidth savings"
log ""
log "3. Blog Post Data:"
log "   - Latency improvement %: From adaptive grid"
log "   - Bandwidth savings %: From early exit grid"
log "   - Representative scenarios: From one-off runs"
log "   - Acceptance rate correlation: Check all runs"
log ""
log "4. Generate Graphs:"
log "   - Latency by workload type (adaptive on/off)"
log "   - Bandwidth vs latency tradeoff (early exit)"
log "   - Acceptance rate distribution"
log ""

log "Experiment suite finished at $(date)"
