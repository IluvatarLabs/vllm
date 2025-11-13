#!/usr/bin/env python3
"""
Multi-Seed Semantic Similarity Aggregation - Two-Set Analysis

Analyzes two independent experiment sets:
  SET 1: Adaptive Draft Length (adaptive=0 vs adaptive=1, threshold=0.0)
  SET 2: Early Exit (thresh=0.0 vs thresh=X, adaptive=0)

ONE-OFF SCENARIOS: Excluded from semantic similarity analysis

For each set, computes:
1. Direct comparison: variant vs baseline (per seed)
2. Baseline comparison A: variant vs vanilla (per seed)
3. Baseline comparison B: baseline vs vanilla (per seed)
4. Delta analysis: A - B computed per-seed then aggregated (DELTA-FIRST METHOD)
5. Cross-seed aggregation: mean ± std for all metrics

Usage:
    python compare_runs_folder.py sweeps/Llama-3.1-8B-Instruct/coding
"""

import json
import sys
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

# Import Script 1 for pairwise comparisons
from evaluate_semantic_similarity import compare_two_runs


@dataclass
class RunData:
    """Data for a single benchmark run."""
    filepath: Path
    category: str  # "adaptive", "early_exit", "vanilla", "oneoff"
    experiment_type: str  # "vanilla", "adaptive-off", "adaptive-on", "thresh0.0", "thresh0.3", etc.
    requests: int
    tokens: int
    temperature: float
    draft_tokens: int
    threshold: Optional[float]  # Only for early exit runs
    adaptive: int


def classify_experiment(config: Dict, filename: str) -> Tuple[str, str]:
    """
    Classify experiment into (category, experiment_type).

    Categories:
        - "adaptive": Adaptive draft length experiments (enable_ncu=false)
        - "early_exit": Early exit experiments (enable_ncu=true)
        - "vanilla": No speculation baseline
        - "oneoff": One-off scenarios (EXCLUDED from analysis)

    Experiment types:
        - "vanilla": No speculation
        - "adaptive-off": adaptive=0, threshold=0.0, enable_ncu=false
        - "adaptive-on": adaptive=1, threshold=0.0, enable_ncu=false
        - "thresh{X}": adaptive=0, threshold=X, enable_ncu=true (e.g., "thresh0.0", "thresh0.3")

    Returns:
        (category, experiment_type)

    Raises:
        ValueError: If config doesn't match any known pattern
    """
    no_spec = config.get('no_speculation', False)
    adaptive = config.get('adaptive_draft_length', 0)
    threshold = config.get('confidence_threshold', 0.0)
    enable_ncu = config.get('enable_ncu', False)

    # ONE-OFF SCENARIOS: Excluded from semantic similarity analysis
    if Path(filename).name.startswith('scenario_'):
        return "oneoff", "oneoff"

    # VANILLA: no_speculation=True
    if no_spec:
        return "vanilla", "vanilla"

    # EARLY EXIT SET: Detect by filename pattern (thresh in filename)
    # Must check BEFORE adaptive set to avoid misclassification
    if 'thresh' in Path(filename).name:
        if adaptive == 0:
            return "early_exit", f"thresh{threshold}"
        else:
            # Hybrid mode (adaptive=1, with threshold)
            return "oneoff", "hybrid"

    # ORIGINAL LOGIC (COMMENTED OUT - enable_ncu not reliable):
    # # EARLY EXIT SET: enable_ncu=True (includes thresh=0.0 baseline)
    # # Must check BEFORE adaptive set to avoid misclassification
    # if enable_ncu:
    #     if adaptive == 0:
    #         return "early_exit", f"thresh{threshold}"
    #     else:
    #         # Hybrid mode in one-offs (adaptive=1, enable_ncu=true)
    #         return "oneoff", "hybrid"

    # ADAPTIVE SET: No threshold in filename, threshold=0.0, adaptive varies
    if threshold == 0.0 and 'thresh' not in Path(filename).name:
        if adaptive == 1:
            return "adaptive", "adaptive-on"
        elif adaptive == 0:
            return "adaptive", "adaptive-off"
        else:
            raise ValueError(f"Invalid adaptive value: {adaptive}")

    raise ValueError(
        f"Unknown experiment type: adaptive={adaptive}, threshold={threshold}, enable_ncu={enable_ncu}, no_spec={no_spec}, file={filename}"
    )


def find_seed_folders(parent_folder: Path) -> List[Path]:
    """Find all seed_* subfolders and sort them numerically."""
    seed_folders = []

    for item in parent_folder.iterdir():
        if item.is_dir() and item.name.startswith('seed_'):
            seed_folders.append(item)

    # Sort numerically by seed number
    def extract_seed_num(path: Path) -> int:
        match = re.search(r'seed_(\d+)', path.name)
        return int(match.group(1)) if match else 999999

    seed_folders.sort(key=extract_seed_num)

    return seed_folders


def load_runs_from_folder(folder: Path) -> List[RunData]:
    """
    Load all benchmark runs from a folder and classify them.

    Excludes:
        - One-off scenarios (scenario_*.json)
        - NCU-specific files (*.ncu.json)

    Returns:
        List of RunData objects
    """
    runs = []

    for json_file in folder.glob("*.json"):
        # Skip NCU files and one-off scenarios
        if '.ncu.json' in json_file.name:
            continue

        try:
            with open(json_file) as f:
                data = json.load(f)

            config = data.get('config', {})

            # Classify experiment
            category, experiment_type = classify_experiment(config, str(json_file))

            # Skip one-offs
            if category == "oneoff":
                continue

            run = RunData(
                filepath=json_file,
                category=category,
                experiment_type=experiment_type,
                requests=config.get('num_requests', 0),
                tokens=config.get('max_new_tokens', 0),
                temperature=config.get('temperature', 0.0),
                draft_tokens=config.get('draft_tokens', 0),
                threshold=config.get('confidence_threshold'),
                adaptive=config.get('adaptive_draft_length', 0),
            )

            runs.append(run)

        except Exception as e:
            print(f"  Warning: Could not load {json_file.name}: {e}")
            continue

    return runs


def group_by_workload_with_vanilla(runs: List[RunData], include_threshold: bool = False) -> Dict[Tuple, Dict[str, RunData]]:
    """
    Group runs by workload parameters with special handling for vanilla.

    Vanilla runs are grouped by (r, t, temp) and attached to ALL spec decode groups
    with matching (r, t, temp) regardless of draft_tokens (and threshold if applicable).

    Spec decode runs are grouped by (r, t, temp, draft_tokens, [threshold if include_threshold]).

    Args:
        runs: List of RunData objects
        include_threshold: If True, include threshold in grouping key (for early exit)

    Returns:
        Dict: {(requests, tokens, temp, draft, [threshold]): {experiment_type: RunData}}
    """
    # Separate vanilla and spec decode runs
    vanilla_runs = []
    spec_runs = []

    for run in runs:
        if run.experiment_type == "vanilla":
            vanilla_runs.append(run)
        else:
            spec_runs.append(run)

    # Group vanilla by (r, t, temp) only
    vanilla_groups = {}
    for run in vanilla_runs:
        key = (run.requests, run.tokens, run.temperature)
        if key in vanilla_groups:
            print(f"  ⚠ WARNING: Duplicate vanilla run for r={key[0]}, t={key[1]}, temp={key[2]} - using latest")
        vanilla_groups[key] = run  # Keep last one if duplicates

    # Group spec decode by (r, t, temp, draft, [threshold])
    grouped = defaultdict(dict)
    for run in spec_runs:
        if include_threshold:
            # For early exit: include threshold in key
            key = (run.requests, run.tokens, run.temperature, run.draft_tokens, run.threshold)
        else:
            # For adaptive: no threshold in key
            key = (run.requests, run.tokens, run.temperature, run.draft_tokens)

        if run.experiment_type in grouped[key]:
            if include_threshold:
                print(f"  ⚠ WARNING: Duplicate {run.experiment_type} run for r={key[0]}, t={key[1]}, temp={key[2]}, d={key[3]}, thresh={key[4]} - using latest")
            else:
                print(f"  ⚠ WARNING: Duplicate {run.experiment_type} run for r={key[0]}, t={key[1]}, temp={key[2]}, d={key[3]} - using latest")

        grouped[key][run.experiment_type] = run

    # Attach vanilla to all matching spec groups
    for key in list(grouped.keys()):
        if include_threshold:
            r, t, temp, draft, thresh = key
            vanilla_key = (r, t, temp)
        else:
            r, t, temp, draft = key
            vanilla_key = (r, t, temp)

        if vanilla_key in vanilla_groups:
            grouped[key]['vanilla'] = vanilla_groups[vanilla_key]

    return grouped


def compare_two_variants(variant_run: RunData, baseline_run: RunData, vanilla_run: Optional[RunData],
                        device: str, num_gpus: int, seed_name: str) -> Dict:
    """
    Compare two variants (e.g., adaptive-on vs adaptive-off, or thresh=X vs thresh=0.0).

    Computes:
        1. Direct comparison: variant vs baseline
        2. Baseline A: variant vs vanilla (if vanilla exists)
        3. Baseline B: baseline vs vanilla (if vanilla exists)
        4. Delta: A - B (computed per-seed for delta-first method)

    Returns:
        Dict with comparison results
    """
    results = {}

    # 1. Direct comparison: variant vs baseline
    print(f"  [1/3] Comparing {variant_run.experiment_type} vs {baseline_run.experiment_type}...")
    direct_result = compare_two_runs(
        str(baseline_run.filepath),
        str(variant_run.filepath),
        device=device,
        num_gpus=num_gpus,
        verbose=False
    )
    results['direct_comparison'] = direct_result

    # 2. Scores A: variant vs vanilla
    if vanilla_run:
        print(f"  [2/3] Comparing {variant_run.experiment_type} vs vanilla...")
        scores_a = compare_two_runs(
            str(vanilla_run.filepath),
            str(variant_run.filepath),
            device=device,
            num_gpus=num_gpus,
            verbose=False
        )
        results['variant_vs_vanilla'] = scores_a
    else:
        print(f"  ⚠ WARNING: Skipping variant vs vanilla - no vanilla baseline")

    # 3. Scores B: baseline vs vanilla
    if vanilla_run:
        print(f"  [3/3] Comparing {baseline_run.experiment_type} vs vanilla...")
        scores_b = compare_two_runs(
            str(vanilla_run.filepath),
            str(baseline_run.filepath),
            device=device,
            num_gpus=num_gpus,
            verbose=False
        )
        results['baseline_vs_vanilla'] = scores_b
    else:
        print(f"  ⚠ WARNING: Skipping baseline vs vanilla - no vanilla baseline")

    # 4. Compute delta: A - B (delta-first method)
    if 'variant_vs_vanilla' in results and 'baseline_vs_vanilla' in results:
        scores_a = results['variant_vs_vanilla']
        scores_b = results['baseline_vs_vanilla']

        results['delta_metrics'] = {
            'delta_f1': scores_a['bertscore_f1_mean'] - scores_b['bertscore_f1_mean'],
            'delta_cosine': scores_a['cosine_mean'] - scores_b['cosine_mean'],
            'delta_quality': scores_a['adaptive_quality_mean'] - scores_b['adaptive_quality_mean'],
        }

    return results


def process_adaptive_set(all_runs: List[RunData], device: str, num_gpus: int) -> Dict:
    """
    Process adaptive draft length experiment set.

    Comparisons:
        - adaptive-on vs adaptive-off (direct)
        - adaptive-on vs vanilla (baseline A)
        - adaptive-off vs vanilla (baseline B)
        - Delta: (A - B) per-seed, then aggregate

    Returns:
        Dict with per-seed and aggregated results
    """
    print("\n" + "="*80)
    print("SET 1: ADAPTIVE DRAFT LENGTH ANALYSIS")
    print("="*80)

    # Filter to adaptive set only
    adaptive_runs = [r for r in all_runs if r.category in ["adaptive", "vanilla"]]

    if not adaptive_runs:
        print("\n⚠ No adaptive runs found")
        return {}

    # Group by seed
    per_seed_runs = defaultdict(list)
    for run in adaptive_runs:
        # Extract seed from path
        match = re.search(r'seed_(\d+)', str(run.filepath))
        if match:
            seed_name = f"seed_{match.group(1)}"
            per_seed_runs[seed_name].append(run)

    # Process each seed
    per_seed_results = {}

    for seed_name in sorted(per_seed_runs.keys()):
        seed_runs = per_seed_runs[seed_name]

        print(f"\n{'='*80}")
        print(f"PROCESSING {seed_name} (Adaptive Set)")
        print(f"{'='*80}")

        print(f"\nLoaded {len(seed_runs)} runs")

        # Group by workload
        grouped = group_by_workload_with_vanilla(seed_runs, include_threshold=False)
        print(f"Found {len(grouped)} unique workload configurations")

        # Compare each workload group
        seed_results = []
        for workload_key in sorted(grouped.keys()):
            r, t, temp, draft = workload_key
            runs_dict = grouped[workload_key]

            config_name = f"r{r}_t{t}_temp{temp}_d{draft}"
            print(f"\nWorkload: {config_name}")

            adaptive_off = runs_dict.get("adaptive-off")
            adaptive_on = runs_dict.get("adaptive-on")
            vanilla = runs_dict.get("vanilla")

            # Need both adaptive-off and adaptive-on for comparison
            if not adaptive_off or not adaptive_on:
                missing = []
                if not adaptive_off:
                    missing.append("adaptive-off")
                if not adaptive_on:
                    missing.append("adaptive-on")
                print(f"  ⚠ WARNING: Skipping {config_name} - missing {', '.join(missing)}")
                continue

            # Compare adaptive-on vs adaptive-off
            comparison = compare_two_variants(
                variant_run=adaptive_on,
                baseline_run=adaptive_off,
                vanilla_run=vanilla,
                device=device,
                num_gpus=num_gpus,
                seed_name=seed_name
            )

            result = {
                'workload': {'requests': r, 'tokens': t, 'temperature': temp, 'draft_tokens': draft},
                'config_name': config_name,
                **comparison
            }

            seed_results.append(result)

        per_seed_results[seed_name] = seed_results

    # Aggregate across seeds
    if per_seed_results:
        print(f"\n{'='*80}")
        print("AGGREGATING ADAPTIVE SET ACROSS SEEDS")
        print(f"{'='*80}\n")

        aggregated = aggregate_across_seeds(per_seed_results)

        return {
            'per_seed_results': per_seed_results,
            'aggregated_results': aggregated,
            'num_seeds': len(per_seed_results)
        }
    else:
        return {}


def process_early_exit_set(all_runs: List[RunData], device: str, num_gpus: int) -> Dict:
    """
    Process early exit experiment set.

    For each threshold X:
        - thresh=X vs thresh=0.0 (direct)
        - thresh=X vs vanilla (baseline A)
        - thresh=0.0 vs vanilla (baseline B, computed once)
        - Delta: (A - B) per-seed, then aggregate

    Returns:
        Dict with per-seed and aggregated results
    """
    print("\n" + "="*80)
    print("SET 2: EARLY EXIT ANALYSIS")
    print("="*80)

    # Filter to early exit set only
    early_exit_runs = [r for r in all_runs if r.category in ["early_exit", "vanilla"]]

    if not early_exit_runs:
        print("\n⚠ No early exit runs found")
        return {}

    # Group by seed
    per_seed_runs = defaultdict(list)
    for run in early_exit_runs:
        # Extract seed from path
        match = re.search(r'seed_(\d+)', str(run.filepath))
        if match:
            seed_name = f"seed_{match.group(1)}"
            per_seed_runs[seed_name].append(run)

    # Process each seed
    per_seed_results = {}

    for seed_name in sorted(per_seed_runs.keys()):
        seed_runs = per_seed_runs[seed_name]

        print(f"\n{'='*80}")
        print(f"PROCESSING {seed_name} (Early Exit Set)")
        print(f"{'='*80}")

        print(f"\nLoaded {len(seed_runs)} runs")

        # Group by workload (including threshold)
        grouped = group_by_workload_with_vanilla(seed_runs, include_threshold=True)
        print(f"Found {len(grouped)} unique workload+threshold configurations")

        # Group by base workload (r, t, temp, d) to process together
        workload_groups = defaultdict(dict)
        for full_key in grouped.keys():
            r, t, temp, draft, thresh = full_key
            base_key = (r, t, temp, draft)
            workload_groups[base_key][thresh] = grouped[full_key]

        # Compare each workload group
        seed_results = []
        for base_key in sorted(workload_groups.keys()):
            r, t, temp, draft = base_key
            threshold_variants = workload_groups[base_key]

            print(f"\nWorkload: r{r}_t{t}_temp{temp}_d{draft}")

            # Get baseline (thresh=0.0) and vanilla
            baseline_runs = threshold_variants.get(0.0, {})
            baseline = baseline_runs.get("thresh0.0")
            vanilla = baseline_runs.get("vanilla")

            if not baseline:
                print(f"  ⚠ WARNING: No thresh=0.0 baseline - skipping workload")
                continue

            # Compare each non-zero threshold against baseline
            for thresh in sorted(threshold_variants.keys()):
                if thresh == 0.0:
                    continue  # Skip baseline itself

                variant_runs = threshold_variants[thresh]
                variant = variant_runs.get(f"thresh{thresh}")

                if not variant:
                    print(f"  ⚠ WARNING: Missing thresh={thresh} variant")
                    continue

                config_name = f"r{r}_t{t}_temp{temp}_d{draft}_thresh{thresh}"
                print(f"\n  Threshold: {thresh} (config: {config_name})")

                # Compare variant vs baseline
                comparison = compare_two_variants(
                    variant_run=variant,
                    baseline_run=baseline,
                    vanilla_run=vanilla,
                    device=device,
                    num_gpus=num_gpus,
                    seed_name=seed_name
                )

                result = {
                    'workload': {'requests': r, 'tokens': t, 'temperature': temp, 'draft_tokens': draft},
                    'config_name': config_name,
                    'threshold': thresh,
                    **comparison
                }

                seed_results.append(result)

        per_seed_results[seed_name] = seed_results

    # Aggregate across seeds
    if per_seed_results:
        print(f"\n{'='*80}")
        print("AGGREGATING EARLY EXIT SET ACROSS SEEDS")
        print(f"{'='*80}\n")

        aggregated = aggregate_across_seeds(per_seed_results)

        return {
            'per_seed_results': per_seed_results,
            'aggregated_results': aggregated,
            'num_seeds': len(per_seed_results)
        }
    else:
        return {}


def aggregate_across_seeds(per_seed_results: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Aggregate results across seeds: compute mean ± std for each metric.

    Uses delta-first method: deltas are computed per-seed, then aggregated.

    Args:
        per_seed_results: Dict mapping seed_name -> list of workload results

    Returns:
        List of aggregated results with mean, std, n for each metric
    """
    # Group by config_name
    config_groups = defaultdict(lambda: defaultdict(list))

    for seed_name, workload_results in per_seed_results.items():
        for result in workload_results:
            config_name = result['config_name']

            # For each comparison type
            for comp_type in ['direct_comparison', 'variant_vs_vanilla', 'baseline_vs_vanilla', 'delta_metrics']:
                if comp_type in result:
                    config_groups[config_name][comp_type].append({
                        'seed': seed_name,
                        'data': result[comp_type]
                    })

    # Compute aggregates
    aggregated = []

    for config_name in sorted(config_groups.keys()):
        comp_data = config_groups[config_name]

        agg_result = {
            'config_name': config_name,
        }

        # Get workload info and threshold (if applicable) from first available result
        for seed_name, workload_results in per_seed_results.items():
            for res in workload_results:
                if res['config_name'] == config_name:
                    agg_result['workload'] = res['workload']
                    if 'threshold' in res:
                        agg_result['threshold'] = res['threshold']
                    break
            if 'workload' in agg_result:
                break

        # Aggregate each comparison type
        for comp_type in ['direct_comparison', 'variant_vs_vanilla', 'baseline_vs_vanilla']:
            if comp_type in comp_data and comp_data[comp_type]:
                agg_result[comp_type] = aggregate_comparison_metrics(comp_data[comp_type])

        # Aggregate deltas (already computed per-seed)
        if 'delta_metrics' in comp_data and comp_data['delta_metrics']:
            agg_result['delta_metrics'] = aggregate_delta_metrics(comp_data['delta_metrics'])

        aggregated.append(agg_result)

    return aggregated


def aggregate_comparison_metrics(seed_data_list: List[Dict]) -> Dict:
    """
    Aggregate metrics across seeds for a single comparison type.

    Args:
        seed_data_list: List of {'seed': seed_name, 'data': comparison_result}

    Returns:
        Dict with aggregated metrics (each metric has seed values + mean/std/n)
    """
    aggregated = {}

    # Define metrics to aggregate
    metrics = [
        'exact_match', 'bertscore_f1_mean', 'bertscore_f1_std', 'bertscore_f1_min', 'bertscore_f1_max',
        'cosine_mean', 'cosine_median', 'cosine_std', 'cosine_min',
        'baseline_quality_mean', 'adaptive_quality_mean', 'quality_pvalue',
        'baseline_acceptance_ratio', 'adaptive_acceptance_ratio',
        'baseline_peak_memory_gb', 'adaptive_peak_memory_gb'
    ]

    for metric in metrics:
        values_by_seed = {}
        values = []

        for item in seed_data_list:
            seed_name = item['seed']
            data = item['data']

            if metric in data:
                values_by_seed[seed_name] = data[metric]
                values.append(data[metric])

        if values:
            aggregated[metric] = {
                **values_by_seed,  # seed_0: value, seed_1: value, ...
                'mean': float(np.mean(values)),
                'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                'n': len(values)
            }

    return aggregated


def aggregate_delta_metrics(seed_data_list: List[Dict]) -> Dict:
    """
    Aggregate delta metrics across seeds (delta-first method).

    Args:
        seed_data_list: List of {'seed': seed_name, 'data': delta_dict}

    Returns:
        Dict with aggregated deltas
    """
    aggregated = {}

    delta_metrics = ['delta_f1', 'delta_cosine', 'delta_quality']

    for metric in delta_metrics:
        values_by_seed = {}
        values = []

        for item in seed_data_list:
            seed_name = item['seed']
            data = item['data']

            if metric in data:
                values_by_seed[seed_name] = data[metric]
                values.append(data[metric])

        if values:
            aggregated[metric] = {
                **values_by_seed,
                'mean': float(np.mean(values)),
                'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                'n': len(values)
            }

    return aggregated


def get_interpretation(f1_mean: float, cosine_mean: float) -> str:
    """Get semantic similarity interpretation based on thresholds."""
    if f1_mean >= 0.90 and cosine_mean >= 0.85:
        return "SEMANTICALLY EQUIVALENT"
    elif f1_mean >= 0.85 and cosine_mean >= 0.70:
        return "PARAPHRASE-LEVEL SIMILARITY"
    elif f1_mean >= 0.80 and cosine_mean >= 0.60:
        return "MODERATE SIMILARITY"
    else:
        return "SEMANTICALLY DIFFERENT"


def print_set_results(set_name: str, results: Dict, comparison_label: str):
    """
    Print formatted results for one experiment set.

    Args:
        set_name: "ADAPTIVE" or "EARLY EXIT"
        results: Dict with aggregated_results and num_seeds
        comparison_label: Label for the comparison (e.g., "ADAPTIVE-ON vs ADAPTIVE-OFF")
    """
    if not results:
        print(f"\n⚠ No results for {set_name} set")
        return

    aggregated_results = results['aggregated_results']
    num_seeds = results['num_seeds']

    print("\n" + "="*80)
    print(f"{set_name} SET RESULTS")
    print("="*80)
    print(f"\nSeeds analyzed: {num_seeds}")

    # Print direct comparison
    print("\n" + "-"*80)
    print(f"{comparison_label}")
    print("-"*80)

    valid_results = [r for r in aggregated_results if 'direct_comparison' in r]
    if valid_results:
        print_aggregate_section(valid_results, 'direct_comparison', num_seeds)
    else:
        print("  No data available")

    # Print variant vs vanilla
    print("\n" + "-"*80)
    print("VARIANT VS VANILLA")
    print("-"*80)

    valid_results = [r for r in aggregated_results if 'variant_vs_vanilla' in r]
    if valid_results:
        print_aggregate_section(valid_results, 'variant_vs_vanilla', num_seeds)
    else:
        print("  No data available")

    # Print baseline vs vanilla
    print("\n" + "-"*80)
    print("BASELINE VS VANILLA")
    print("-"*80)

    valid_results = [r for r in aggregated_results if 'baseline_vs_vanilla' in r]
    if valid_results:
        print_aggregate_section(valid_results, 'baseline_vs_vanilla', num_seeds)
    else:
        print("  No data available")

    # Print delta comparison
    print("\n" + "-"*80)
    print("DELTA COMPARISON (Variant Impact)")
    print("-"*80)
    print_comparison_table(aggregated_results, num_seeds)


def print_aggregate_section(valid_results: List[Dict], comparison_key: str, num_seeds: int):
    """Print aggregate metrics table for a specific comparison type."""
    # Compute overall aggregates across configs
    all_exact = []
    all_f1 = []
    all_cosine = []
    all_accept = []
    all_memory_baseline = []
    all_memory_adaptive = []

    for r in valid_results:
        comp = r[comparison_key]
        if 'exact_match' in comp:
            all_exact.append(comp['exact_match']['mean'] * 100)
        if 'bertscore_f1_mean' in comp:
            all_f1.append(comp['bertscore_f1_mean']['mean'])
        if 'cosine_mean' in comp:
            all_cosine.append(comp['cosine_mean']['mean'])
        if 'adaptive_acceptance_ratio' in comp:
            all_accept.append(comp['adaptive_acceptance_ratio']['mean'] * 100)
        if 'baseline_peak_memory_gb' in comp:
            all_memory_baseline.append(comp['baseline_peak_memory_gb']['mean'])
        if 'adaptive_peak_memory_gb' in comp:
            all_memory_adaptive.append(comp['adaptive_peak_memory_gb']['mean'])

    print(f"\nAggregate Metrics Across All Configurations (N={num_seeds} seeds):")
    if all_exact:
        print(f"  Exact match rate:        {np.mean(all_exact):.1f}% ± {np.std(all_exact, ddof=1):.1f}%")
    if all_f1:
        print(f"  BERTScore F1 (average):  {np.mean(all_f1):.4f} ± {np.std(all_f1, ddof=1):.4f}")
    if all_cosine:
        print(f"  Cosine sim (average):    {np.mean(all_cosine):.4f} ± {np.std(all_cosine, ddof=1):.4f}")
    if all_accept:
        print(f"  Acceptance rate (avg):   {np.mean(all_accept):.1f}% ± {np.std(all_accept, ddof=1):.1f}%")
    if all_memory_baseline and all_memory_adaptive:
        print(f"  Peak memory (avg):       {np.mean(all_memory_baseline):.1f} GB (baseline), {np.mean(all_memory_adaptive):.1f} GB (variant)")

    # Per-configuration summary table
    print(f"\nPer-Configuration Summary:")
    print(f"  {'Config':<30} {'Exact%':<18} {'F1':<18} {'Cosine':<18} {'Accept%':<12} {'N':<3}")
    print(f"  {'-'*110}")

    for r in valid_results:
        comp = r[comparison_key]
        config = r['config_name']

        # Extract mean ± std for display
        exact = comp.get('exact_match', {})
        f1 = comp.get('bertscore_f1_mean', {})
        cosine = comp.get('cosine_mean', {})
        accept = comp.get('adaptive_acceptance_ratio', {})

        n = exact.get('n', 0)
        flag = "  ⚠" if n < num_seeds else ""

        exact_str = f"{exact.get('mean', 0)*100:.1f} ± {exact.get('std', 0)*100:.1f}" if 'mean' in exact else "N/A"
        f1_str = f"{f1.get('mean', 0):.4f} ± {f1.get('std', 0):.4f}" if 'mean' in f1 else "N/A"
        cosine_str = f"{cosine.get('mean', 0):.4f} ± {cosine.get('std', 0):.4f}" if 'mean' in cosine else "N/A"
        accept_str = f"{accept.get('mean', 0)*100:.1f}" if 'mean' in accept else "N/A"

        print(f"  {config:<30} {exact_str:<18} {f1_str:<18} {cosine_str:<18} {accept_str:<12} {n:<3}{flag}")

    if any(r[comparison_key].get('exact_match', {}).get('n', num_seeds) < num_seeds for r in valid_results):
        print(f"\n  Note: ⚠ indicates fewer than {num_seeds} seeds available for this configuration")

    # Interpretation
    if all_f1 and all_cosine:
        interpretation = get_interpretation(np.mean(all_f1), np.mean(all_cosine))
        print(f"\nInterpretation:")
        print(f"  ~ {interpretation}")


def print_comparison_table(aggregated_results: List[Dict], num_seeds: int):
    """Print delta comparison table with mean ± std."""
    valid_results = [r for r in aggregated_results if 'delta_metrics' in r]

    if not valid_results:
        print("  No delta data available")
        return

    print(f"\n  {'Config':<30} {'ΔF1':<18} {'ΔCosine':<18} {'ΔQuality':<18} {'N':<3}")
    print(f"  {'-'*90}")

    all_delta_f1 = []
    all_delta_cosine = []
    all_delta_quality = []

    for r in valid_results:
        config = r['config_name']
        delta = r['delta_metrics']

        delta_f1 = delta.get('delta_f1', {})
        delta_cosine = delta.get('delta_cosine', {})
        delta_quality = delta.get('delta_quality', {})

        n = delta_f1.get('n', 0)
        flag = "  ⚠" if n < num_seeds else ""

        if 'mean' in delta_f1:
            all_delta_f1.append(delta_f1['mean'])
            f1_str = f"{delta_f1['mean']:+.4f} ± {delta_f1['std']:.4f}"
        else:
            f1_str = "N/A"

        if 'mean' in delta_cosine:
            all_delta_cosine.append(delta_cosine['mean'])
            cosine_str = f"{delta_cosine['mean']:+.4f} ± {delta_cosine['std']:.4f}"
        else:
            cosine_str = "N/A"

        if 'mean' in delta_quality:
            all_delta_quality.append(delta_quality['mean'])
            quality_str = f"{delta_quality['mean']:+.4f} ± {delta_quality['std']:.4f}"
        else:
            quality_str = "N/A"

        print(f"  {config:<30} {f1_str:<18} {cosine_str:<18} {quality_str:<18} {n:<3}{flag}")

    # Overall average
    if all_delta_f1:
        print(f"  {'-'*90}")
        print(f"  {'Average':<30} {np.mean(all_delta_f1):+.4f} ± {np.std(all_delta_f1, ddof=1):.4f}  "
              f"{np.mean(all_delta_cosine):+.4f} ± {np.std(all_delta_cosine, ddof=1):.4f}  "
              f"{np.mean(all_delta_quality):+.4f} ± {np.std(all_delta_quality, ddof=1):.4f}")

        print(f"\nInterpretation:")
        avg_f1_delta = np.mean(all_delta_f1)
        if abs(avg_f1_delta) < 0.01:
            print("  Variant has NEGLIGIBLE impact on semantic divergence vs baseline.")
        elif avg_f1_delta > 0:
            print("  Variant REDUCES semantic divergence compared to baseline.")
            print("  (Positive delta = variant is MORE similar to vanilla)")
        else:
            print("  Variant INCREASES semantic divergence compared to baseline.")
            print("  (Negative delta = variant is LESS similar to vanilla)")


def main():
    """Main entry point for two-set semantic similarity analysis."""
    if len(sys.argv) != 2:
        print("Usage: python compare_runs_folder.py <parent_folder>")
        print("\nExample:")
        print("  python compare_runs_folder.py sweeps/Llama-3.1-8B-Instruct/coding")
        print("\nExpected structure: <parent_folder>/seed_0/, seed_1/, seed_2/, ...")
        print("\nAnalyzes two experiment sets:")
        print("  SET 1: Adaptive draft length (adaptive=0 vs adaptive=1)")
        print("  SET 2: Early exit (thresh=0.0 vs thresh=X)")
        print("\nONE-OFF SCENARIOS: Excluded from semantic similarity analysis")
        sys.exit(1)

    parent_folder = Path(sys.argv[1])

    if not parent_folder.exists():
        print(f"ERROR: Folder not found: {parent_folder}")
        sys.exit(1)

    print("="*80)
    print("SEMANTIC SIMILARITY EVALUATION - TWO-SET ANALYSIS")
    print("="*80)

    # Find seed folders
    print(f"\nScanning for seed folders in: {parent_folder}")
    seed_folders = find_seed_folders(parent_folder)

    if not seed_folders:
        print(f"ERROR: No seed_* folders found in {parent_folder}")
        print("Expected folder structure: <parent>/seed_0/, seed_1/, seed_2/, ...")
        sys.exit(1)

    seed_names = [f.name for f in seed_folders]
    print(f"Seeds found: {', '.join(seed_names)} (N={len(seed_folders)})")

    # Setup device (do once, reuse for all comparisons)
    print("\nSetting up device acceleration...")
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_gpus = torch.cuda.device_count() if device == 'cuda' else 1

    if device == 'cuda':
        print(f"  Using CUDA with {num_gpus} GPU(s)")
    else:
        print(f"  Using CPU")

    # Load all runs from all seeds
    print("\nLoading all runs...")
    all_runs = []
    for seed_folder in seed_folders:
        runs = load_runs_from_folder(seed_folder)
        all_runs.extend(runs)

    print(f"Loaded {len(all_runs)} total runs (excluding one-offs and NCU files)")

    # Count by category
    category_counts = defaultdict(int)
    for run in all_runs:
        category_counts[run.category] += 1

    print("\nRuns by category:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")

    # Process Set 1: Adaptive
    adaptive_results = process_adaptive_set(all_runs, device, num_gpus)

    # Process Set 2: Early Exit
    early_exit_results = process_early_exit_set(all_runs, device, num_gpus)

    # Print results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    if adaptive_results:
        print_set_results("ADAPTIVE", adaptive_results, "ADAPTIVE-ON vs ADAPTIVE-OFF")

    if early_exit_results:
        print_set_results("EARLY EXIT", early_exit_results, "THRESHOLD=X vs THRESHOLD=0.0")

    # Save detailed results (per-seed + aggregated)
    detailed_output = parent_folder / "semantic_similarity_analysis_detailed.json"
    with detailed_output.open('w') as f:
        json.dump({
            'parent_folder': str(parent_folder),
            'seeds': seed_names,
            'num_seeds': len(seed_folders),
            'adaptive_set': adaptive_results,
            'early_exit_set': early_exit_results,
        }, f, indent=2)

    print(f"\n\nDetailed results saved to: {detailed_output}")

    # Save summary (aggregated only)
    summary_output = parent_folder / "semantic_similarity_analysis_summary.json"

    adaptive_summary = None
    if adaptive_results:
        adaptive_summary = {
            'aggregated_results': adaptive_results['aggregated_results'],
            'num_seeds': adaptive_results['num_seeds']
        }

    early_exit_summary = None
    if early_exit_results:
        early_exit_summary = {
            'aggregated_results': early_exit_results['aggregated_results'],
            'num_seeds': early_exit_results['num_seeds']
        }

    with summary_output.open('w') as f:
        json.dump({
            'parent_folder': str(parent_folder),
            'seeds': seed_names,
            'num_seeds': len(seed_folders),
            'adaptive_set': adaptive_summary,
            'early_exit_set': early_exit_summary,
        }, f, indent=2)

    print(f"Summary results saved to: {summary_output}")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
