# Semantic Similarity Evaluation

This script evaluates whether baseline (adaptive=0) and adaptive (adaptive=1) speculation produce semantically equivalent outputs at temperature=0.0.

## Quick Start

### 1. Install Dependencies

```bash
./install_eval_deps.sh
```

This will install required packages and detect AMX support.

**For AMX-capable CPUs (Sapphire Rapids/Emerald Rapids):**
```bash
pip install intel_extension_for_pytorch
```

Or manually:
```bash
pip install bert-score sentence-transformers scipy transformers torch numpy
pip install intel_extension_for_pytorch  # Optional, for AMX acceleration
```

### 2. Run Evaluation

```bash
python3 evaluate_semantic_similarity.py
```

**Note:** This will take 10-30 minutes depending on GPU availability and number of outputs.

## What It Measures

### Primary Metrics

1. **BERTScore F1** (most reliable)
   - Uses contextual embeddings to compare semantic meaning
   - **Scientific basis:** Zhang et al., ICLR 2020 (0.93 correlation with human judgment)
   - **Interpretation:**
     - F1 ≥ 0.90 = Semantically equivalent
     - F1 0.85-0.90 = Paraphrases (high similarity)
     - F1 < 0.80 = Semantically different (potential bug)

2. **Cosine Similarity** (faster sanity check)
   - Sentence-BERT embeddings (all-mpnet-base-v2)
   - **Scientific basis:** Reimers & Gurevych, 2019 (0.88 correlation with human judgment)
   - **Interpretation:**
     - ≥ 0.85 = Semantic equivalence
     - 0.70-0.85 = Paraphrases
     - < 0.60 = Different meaning

3. **Reward Model Quality**
   - Scores output quality using OpenAssistant reward model
   - Tests if adaptive speculation degrades output quality

### Reference Metrics

- **Exact Match Rate**: Byte-for-byte matching (for reference only)
  - Low match rate (e.g., 47%) is NOT necessarily a bug
  - Different valid generation paths can be textually different but semantically equivalent
  - **Important:** Temperature 0.0 does NOT guarantee bit-perfect determinism
    - Modern LLMs have inherent non-determinism from:
      - Floating-point arithmetic non-associativity
      - GPU thread scheduling variations
      - Infrastructure optimizations (batching, prefix caching)
    - OpenAI, Anthropic, and Google all confirm temp=0 is "mostly" deterministic, not fully
    - Expect semantic equivalence (BERTScore ≥0.85), not exact string matching

## Output Interpretation

The script will output a verdict for each commit based on scientifically-validated thresholds:

### ✓ SEMANTICALLY EQUIVALENT
- **Criteria:** BERTScore F1 ≥0.90 AND Cosine ≥0.85
- Outputs convey the same meaning despite textual differences
- Low exact match rate (even <50%) is EXPECTED and NOT a bug
- Adaptive speculation is working correctly
- **Normal at temp=0.0:** Minor wording variations, phrase reordering

### ✓ PARAPHRASE-LEVEL SIMILARITY
- **Criteria:** BERTScore F1 0.85-0.90 AND Cosine 0.70-0.85
- Outputs are highly similar paraphrases
- Acceptable wording variations within normal temp=0.0 behavior
- No bug - this is expected for modern LLM implementations
- **Cause:** Floating-point arithmetic, GPU scheduling, infrastructure optimizations

### ~ MODERATE SIMILARITY
- **Criteria:** BERTScore F1 0.80-0.85 OR Cosine 0.60-0.70
- Some semantic differences exist
- Manual inspection recommended to assess if acceptable
- May warrant investigation depending on use case

### ✗ SEMANTICALLY DIFFERENT
- **Criteria:** BERTScore F1 <0.80 OR Cosine <0.60
- Significant semantic divergence detected
- This IS a bug that needs investigation
- Not just rewording - meaningful semantic differences

## What Gets Compared

### Commits Evaluated

1. **67e38dfff**: Single CUDA graph capture (old code)
2. **70c8daf53**: Multi-graph capture per draft_length (new code)

### Configurations (temp=0.0 only)

- r12_t64_d10
- r36_t128_d10
- r60_t64_d10
- r36_t256_d10
- r36_t128_d5
- r36_t128_d15 (only in 67e38dfff)

Each configuration compares:
- Baseline: `adaptive=0` (fixed draft_length)
- Adaptive: `adaptive=1` (variable draft_length based on EWMA)

## Cross-Commit Comparison

The script also compares whether multi-graph capture (70c8daf53) affects semantic similarity compared to single-graph capture (67e38dfff).

**Key question:** Does the CUDA graph implementation change semantic equivalence?

## Expected Runtime

- **With GPU:** 10-15 minutes
- **With AMX-capable CPU:** 15-25 minutes (with IPEX installed)
- **Without GPU/AMX (CPU only):** 30-60 minutes

BERTScore and reward model scoring are the slowest steps.

**AMX Acceleration:**
- The script automatically detects and uses AMX if available
- AMX provides 2-3x speedup for matrix operations on Sapphire Rapids+ CPUs
- For best performance, install Intel Extension for PyTorch (IPEX)

## Troubleshooting

### Out of Memory

If you get CUDA OOM errors:
1. Script will fall back to CPU automatically
2. Or reduce batch size in the code:
   - Line 197: Change `batch_size=32` to `batch_size=8`

### Missing Files

If the script reports missing files:
- Check that sweep results exist in `sweeps/adaptive_grid_67e38dfff/` and `sweeps/adaptive_grid_70c8daf53/`
- Ensure you've run the experiments at temp=0.0

### Model Download Issues

First run will download models (~2GB total):
- microsoft/deberta-large-mnli (~1.4GB)
- all-mpnet-base-v2 (~400MB)
- OpenAssistant/reward-model-deberta-v3-large-v2 (~1.4GB)

Ensure internet connection and sufficient disk space.
