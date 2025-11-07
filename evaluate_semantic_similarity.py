#!/usr/bin/env python3
"""
Semantic Similarity Evaluation for Adaptive Speculation

Compares two vLLM runs using:
1. BERTScore (semantic similarity)
2. Sentence-BERT cosine similarity
3. Reward model quality scoring

Scientific Basis for Thresholds:
- BERTScore: Zhang et al., ICLR 2020 (0.93 Pearson correlation with humans)
  * ≥0.90 = Semantic equivalence
  * 0.85-0.90 = Paraphrases (high similarity)
  * <0.80 = Semantically different

- Cosine Similarity: Reimers & Gurevych, 2019 (0.88 Pearson on STS benchmark)
  * ≥0.85 = Semantic equivalence / near-duplicates
  * 0.70-0.85 = Paraphrases
  * <0.60 = Semantically different

Important: Temperature 0.0 does NOT guarantee bit-perfect determinism due to
floating-point non-associativity and GPU scheduling. Semantic equivalence is
the realistic goal, not exact string matching.

Usage:
    # As CLI tool (prints human-readable output):
    python evaluate_semantic_similarity.py run1.json run2.json

    # As importable module:
    from evaluate_semantic_similarity import compare_two_runs
    result = compare_two_runs("run1.json", "run2.json", verbose=False)
"""

import json
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Check dependencies and provide helpful error messages
try:
    from bert_score import score as bert_score
except ImportError:
    print("ERROR: bert_score not installed. Run: pip install bert-score")
    exit(1)

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("ERROR: sentence-transformers not installed. Run: pip install sentence-transformers")
    exit(1)

try:
    from scipy.stats import ttest_rel
except ImportError:
    print("ERROR: scipy not installed. Run: pip install scipy")
    exit(1)

try:
    from transformers import pipeline
    import torch
except ImportError:
    print("ERROR: transformers not installed. Run: pip install transformers torch")
    exit(1)


def setup_device_acceleration():
    """Configure device acceleration (CUDA multi-GPU or AMX)."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"\n✓ CUDA acceleration available with {num_gpus} GPU(s)")

        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            total_mem_gb = props.total_memory / 1024**3
            print(f"  GPU {i}: {props.name} ({total_mem_gb:.2f} GB)")

        if num_gpus > 1:
            print(f"  → Using multi-GPU setup ({num_gpus} GPUs, ~{total_mem_gb * num_gpus:.0f} GB total)")

        # Enable memory management optimizations
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        return 'cuda', num_gpus
    else:
        # CPU execution - check for AMX
        print("\nChecking for AMX (Advanced Matrix Extensions) support...")

        # Enable oneDNN optimizations (includes AMX)
        os.environ['DNNL_VERBOSE'] = '0'  # Set to 1 for verbose AMX logging
        os.environ['ONEDNN_DEFAULT_FPMATH_MODE'] = 'BF16'  # Use BF16 for AMX

        # Check if Intel Extension for PyTorch is available
        try:
            import intel_extension_for_pytorch as ipex
            print(f"  ✓ Intel Extension for PyTorch {ipex.__version__} detected")

            # Check AMX availability
            if hasattr(ipex, '_C') and hasattr(ipex._C, '_has_amx'):
                has_amx = ipex._C._has_amx()
                if has_amx:
                    print("  ✓ AMX acceleration ENABLED")
                    torch.set_float32_matmul_precision('medium')  # Use TF32/BF16 for matmul
                else:
                    print("  ✗ AMX not available on this CPU")
            else:
                print("  ~ AMX detection not available, using standard oneDNN optimizations")
        except ImportError:
            print("  ℹ Intel Extension for PyTorch not installed")
            print("    For best AMX performance, install: pip install intel_extension_for_pytorch")
            print("    Using standard PyTorch with oneDNN optimizations")

        # Even without IPEX, oneDNN will use AMX if available
        print("  → Using oneDNN backend (includes AMX support if CPU capable)")

        return 'cpu', 1


def load_json_outputs(json_path: str) -> List[str]:
    """
    Load outputs from a single JSON file.

    Returns:
        List of output strings from all batches
    """
    with open(json_path) as f:
        data = json.load(f)

    outputs = []
    for result in data.get('results', []):
        outputs.extend(result.get('outputs', []))

    return outputs


def extract_config_name(json_path: str) -> str:
    """Extract a short config name from the JSON file for display."""
    try:
        with open(json_path) as f:
            data = json.load(f)

        config = data.get('config', {})
        requests = config.get('num_requests', '?')
        tokens = config.get('max_new_tokens', '?')
        draft = config.get('draft_tokens', '?')

        return f"r{requests}_t{tokens}_d{draft}"
    except Exception:
        # Fallback to filename
        return Path(json_path).stem


def compute_exact_match(baseline: List[str], adaptive: List[str]) -> float:
    """Compute exact byte-for-byte match rate."""
    matches = sum(1 for b, a in zip(baseline, adaptive) if b == a)
    return matches / len(baseline) if baseline else 0.0


def compute_semantic_similarity(baseline: List[str], adaptive: List[str],
                               device: str = 'cuda', num_gpus: int = 1) -> Dict:
    """Compute BERTScore and cosine similarity metrics."""
    print("  Computing BERTScore (this may take a while)...", flush=True)

    # BERTScore with deberta-large for high quality
    # BERTScore handles multi-GPU internally when device='cuda'
    P, R, F1 = bert_score(
        adaptive, baseline,
        lang='en',
        model_type='microsoft/deberta-large-mnli',
        verbose=False,
        device=device,
        batch_size=8 if num_gpus > 1 else 16  # Smaller batch for multi-GPU to fit memory
    )

    print("  Computing Sentence-BERT embeddings...", flush=True)

    # Sentence-BERT cosine similarity
    model = SentenceTransformer('all-mpnet-base-v2')

    if device == 'cuda':
        if num_gpus > 1:
            # For multi-GPU, use the model pool which distributes across GPUs
            print(f"    → Using multi-GPU encoding across {num_gpus} GPUs")
            # Start the multi-process pool
            pool = model.start_multi_process_pool(target_devices=[f'cuda:{i}' for i in range(num_gpus)])
            baseline_emb = model.encode_multi_process(baseline, pool, batch_size=32)
            adaptive_emb = model.encode_multi_process(adaptive, pool, batch_size=32)
            model.stop_multi_process_pool(pool)
        else:
            # Single GPU
            model = model.to('cuda')
            baseline_emb = model.encode(baseline, show_progress_bar=False, batch_size=32)
            adaptive_emb = model.encode(adaptive, show_progress_bar=False, batch_size=32)
    else:
        # CPU with AMX optimization
        try:
            import intel_extension_for_pytorch as ipex
            model = ipex.optimize(model, dtype=torch.bfloat16)
            print("    → Model optimized with IPEX for AMX acceleration")
        except (ImportError, AttributeError):
            pass  # Fall back to standard execution

        baseline_emb = model.encode(baseline, show_progress_bar=False, batch_size=32)
        adaptive_emb = model.encode(adaptive, show_progress_bar=False, batch_size=32)

    cosine_scores = util.cos_sim(baseline_emb, adaptive_emb).diagonal().cpu().numpy()

    # Compute distribution
    bins = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    hist, _ = np.histogram(cosine_scores, bins=bins)
    distribution = {
        f"[{bins[i]:.2f}, {bins[i+1]:.2f})": int(hist[i])
        for i in range(len(bins)-1)
    }

    return {
        'bertscore_f1': F1.cpu().numpy(),
        'cosine_scores': cosine_scores,
        'cosine_distribution': distribution,
    }


def compute_quality_scores(baseline: List[str], adaptive: List[str],
                          device: str = 'cuda', num_gpus: int = 1) -> Dict:
    """Compute quality scores using reward model."""
    print("  Loading reward model...", flush=True)

    try:
        # Use GPU 0 for reward model (or CPU if no CUDA)
        device_id = 0 if device == 'cuda' else -1
        print(f"Device set to use cuda:{device_id}" if device_id >= 0 else "Device set to use CPU")

        reward_model = pipeline(
            'text-classification',
            model='OpenAssistant/reward-model-deberta-v3-large-v2',
            device=device_id,
            truncation=True,
            max_length=512,
            batch_size=4 if num_gpus > 1 else 8  # Smaller batch for multi-GPU setup
        )

        if num_gpus > 1:
            print(f"    → Reward model on GPU 0 (other GPUs used for embeddings)")
    except Exception as e:
        print(f"  WARNING: Could not load reward model: {e}")
        print("  Skipping quality scoring.")
        return {
            'baseline_quality': [0.0] * len(baseline),
            'adaptive_quality': [0.0] * len(adaptive),
        }

    print("  Scoring baseline outputs...", flush=True)
    baseline_quality = []
    for output in baseline:
        try:
            # Truncate long outputs
            truncated = output[:1000] if len(output) > 1000 else output
            score = reward_model(truncated)[0]['score']
            baseline_quality.append(score)
        except Exception:
            baseline_quality.append(0.0)

    print("  Scoring adaptive outputs...", flush=True)
    adaptive_quality = []
    for output in adaptive:
        try:
            truncated = output[:1000] if len(output) > 1000 else output
            score = reward_model(truncated)[0]['score']
            adaptive_quality.append(score)
        except Exception:
            adaptive_quality.append(0.0)

    return {
        'baseline_quality': baseline_quality,
        'adaptive_quality': adaptive_quality,
    }


def compare_two_runs(baseline_json: str, adaptive_json: str,
                    device: Optional[str] = None, num_gpus: Optional[int] = None,
                    verbose: bool = True) -> Dict:
    """
    Compare two runs and return all metrics.

    Args:
        baseline_json: Path to baseline JSON file
        adaptive_json: Path to adaptive JSON file
        device: Device to use ('cuda' or 'cpu'), auto-detected if None
        num_gpus: Number of GPUs to use, auto-detected if None
        verbose: Whether to print human-readable output

    Returns:
        Dict with all metrics
    """
    # Setup device if not provided
    if device is None or num_gpus is None:
        if verbose:
            device, num_gpus = setup_device_acceleration()
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            num_gpus = torch.cuda.device_count() if device == 'cuda' else 1
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Load outputs
    baseline_outputs = load_json_outputs(baseline_json)
    adaptive_outputs = load_json_outputs(adaptive_json)

    # Extract memory stats from JSONs
    baseline_memory = 0.0
    adaptive_memory = 0.0
    baseline_acceptance = 0.0
    adaptive_acceptance = 0.0

    try:
        with open(baseline_json) as f:
            baseline_data = json.load(f)
            if 'summary' in baseline_data and 'per_mode' in baseline_data['summary']:
                mode_data = baseline_data['summary']['per_mode'][0]
                baseline_memory = mode_data.get('peak_memory_gb', 0.0)
                baseline_acceptance = mode_data.get('spec_acceptance_ratio', 0.0)
    except Exception:
        pass

    try:
        with open(adaptive_json) as f:
            adaptive_data = json.load(f)
            if 'summary' in adaptive_data and 'per_mode' in adaptive_data['summary']:
                mode_data = adaptive_data['summary']['per_mode'][0]
                adaptive_memory = mode_data.get('peak_memory_gb', 0.0)
                adaptive_acceptance = mode_data.get('spec_acceptance_ratio', 0.0)
    except Exception:
        pass

    # Handle mismatched lengths
    if len(baseline_outputs) != len(adaptive_outputs):
        if verbose:
            print(f"WARNING: Mismatched output counts: {len(baseline_outputs)} vs {len(adaptive_outputs)}")
            print(f"  Truncating to shorter length: {min(len(baseline_outputs), len(adaptive_outputs))}")
        min_len = min(len(baseline_outputs), len(adaptive_outputs))
        baseline_outputs = baseline_outputs[:min_len]
        adaptive_outputs = adaptive_outputs[:min_len]

    # Extract config name for display
    config_name = extract_config_name(baseline_json)

    if verbose:
        print(f"\nComparing A vs B:\n")
        print(f"  Configuration: {config_name} ({len(baseline_outputs)} outputs)")

    # Exact match (for reference)
    exact_match = compute_exact_match(baseline_outputs, adaptive_outputs)
    if verbose:
        print(f"    Exact match rate: {exact_match*100:.1f}%")

    # Semantic similarity
    sem_metrics = compute_semantic_similarity(baseline_outputs, adaptive_outputs, device, num_gpus)

    f1_mean = sem_metrics['bertscore_f1'].mean()
    f1_std = sem_metrics['bertscore_f1'].std()
    f1_min = sem_metrics['bertscore_f1'].min()
    f1_max = sem_metrics['bertscore_f1'].max()

    cosine_mean = sem_metrics['cosine_scores'].mean()
    cosine_median = np.median(sem_metrics['cosine_scores'])
    cosine_std = sem_metrics['cosine_scores'].std()
    cosine_min = sem_metrics['cosine_scores'].min()

    if verbose:
        print(f"    BERTScore F1: {f1_mean:.4f} (±{f1_std:.4f}) [{f1_min:.4f}, {f1_max:.4f}]")
        print(f"    Cosine similarity: {cosine_mean:.4f} (median={cosine_median:.4f}, min={cosine_min:.4f})")

    # Quality scores
    quality_metrics = compute_quality_scores(baseline_outputs, adaptive_outputs, device, num_gpus)

    baseline_quality_mean = np.mean(quality_metrics['baseline_quality'])
    adaptive_quality_mean = np.mean(quality_metrics['adaptive_quality'])

    # Statistical test
    try:
        t_stat, p_value = ttest_rel(
            quality_metrics['baseline_quality'],
            quality_metrics['adaptive_quality']
        )
    except Exception:
        p_value = 1.0

    if verbose:
        print(f"    Quality scores: baseline={baseline_quality_mean:.4f}, "
              f"adaptive={adaptive_quality_mean:.4f} (p={p_value:.4f})")
        print(f"    Acceptance rate: baseline={baseline_acceptance*100:.1f}%, adaptive={adaptive_acceptance*100:.1f}%")
        print(f"    Peak memory: baseline={baseline_memory:.2f} GB, adaptive={adaptive_memory:.2f} GB")

    # Return all metrics as dict
    return {
        'config_name': config_name,
        'num_outputs': len(baseline_outputs),
        'exact_match': exact_match,
        'bertscore_f1_mean': f1_mean,
        'bertscore_f1_std': f1_std,
        'bertscore_f1_min': f1_min,
        'bertscore_f1_max': f1_max,
        'cosine_mean': cosine_mean,
        'cosine_median': cosine_median,
        'cosine_std': cosine_std,
        'cosine_min': cosine_min,
        'baseline_quality_mean': baseline_quality_mean,
        'adaptive_quality_mean': adaptive_quality_mean,
        'quality_pvalue': p_value,
        'baseline_acceptance_ratio': baseline_acceptance,
        'adaptive_acceptance_ratio': adaptive_acceptance,
        'baseline_peak_memory_gb': baseline_memory,
        'adaptive_peak_memory_gb': adaptive_memory,
        'cosine_distribution': sem_metrics['cosine_distribution'],
        # Raw scores for further analysis
        'bertscore_f1_scores': sem_metrics['bertscore_f1'].tolist(),
        'cosine_scores': sem_metrics['cosine_scores'].tolist(),
        'baseline_quality_scores': quality_metrics['baseline_quality'],
        'adaptive_quality_scores': quality_metrics['adaptive_quality'],
    }


def main():
    """CLI interface for comparing two runs."""
    if len(sys.argv) != 3:
        print("Usage: python evaluate_semantic_similarity.py <baseline_json> <adaptive_json>")
        print("\nExample:")
        print("  python evaluate_semantic_similarity.py run1_adaptive0.json run2_adaptive1.json")
        sys.exit(1)

    baseline_json = sys.argv[1]
    adaptive_json = sys.argv[2]

    # Validate files exist
    if not Path(baseline_json).exists():
        print(f"ERROR: Baseline file not found: {baseline_json}")
        sys.exit(1)
    if not Path(adaptive_json).exists():
        print(f"ERROR: Adaptive file not found: {adaptive_json}")
        sys.exit(1)

    # Run comparison with verbose output
    result = compare_two_runs(baseline_json, adaptive_json, verbose=True)

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
