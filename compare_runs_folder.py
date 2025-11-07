#!/usr/bin/env python3
"""
Multi-Seed Semantic Similarity Aggregation

Processes multiple seed folders of benchmark runs and computes:
1. Direct comparison: adaptive-on vs adaptive-off (per seed)
2. Baseline comparison A: adaptive-on vs vanilla (per seed)
3. Baseline comparison B: adaptive-off vs vanilla (per seed)
4. Delta analysis: A - B computed per-seed then aggregated
5. Cross-seed aggregation: mean ± std for all metrics

Saves both per-seed detailed results and cross-seed aggregated summary.

Usage:
    python compare_runs_folder.py sweeps/Model+Draft/coding
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
    experiment_type: str  # "vanilla", "adaptive-off", "adaptive-on"
    requests: int
    tokens: int
    temperature: float
    draft_tokens: int


def find_seed_folders(parent_folder: Path) -> List[Path]:
    """
    Find all seed_* subfolders and sort them numerically.

    Returns:
        Sorted list of seed folder paths
    """
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

    Returns:
        List of RunData objects
    """
    runs = []

    for json_file in folder.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            config = data.get('config', {})

            # Determine experiment type
            if config.get('no_speculation'):
                experiment_type = "vanilla"
            elif config.get('adaptive_draft_length') == 1:
                experiment_type = "adaptive-on"
            else:
                experiment_type = "adaptive-off"

            run = RunData(
                filepath=json_file,
                experiment_type=experiment_type,
                requests=config.get('num_requests', 0),
                tokens=config.get('max_new_tokens', 0),
                temperature=config.get('temperature', 0.0),
                draft_tokens=config.get('draft_tokens', 0),
            )

            runs.append(run)

        except Exception as e:
            print(f"  Warning: Could not load {json_file.name}: {e}")
            continue

    return runs


def group_by_workload(runs: List[RunData]) -> Dict[Tuple, Dict[str, RunData]]:
    """
    Group runs by workload parameters (requests, tokens, temperature).

    Returns:
        Dict: {(requests, tokens, temp): {experiment_type: RunData}}
    """
    grouped = defaultdict(dict)

    for run in runs:
        key = (run.requests, run.tokens, run.temperature)
        grouped[key][run.experiment_type] = run

    return grouped


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


def compare_workload_group(workload_key: Tuple, runs: Dict[str, RunData],
                           device: str, num_gpus: int, seed_name: str) -> Dict:
    """
    Compare all run types within a single workload group.

    Returns:
        Dict with all comparison results for this workload, or warnings if skipped
    """
    requests, tokens, temp = workload_key

    vanilla = runs.get("vanilla")
    adaptive_off = runs.get("adaptive-off")
    adaptive_on = runs.get("adaptive-on")

    results = {
        'workload': {
            'requests': requests,
            'tokens': tokens,
            'temperature': temp,
        },
        'config_name': f"r{requests}_t{tokens}_d10",
    }

    # 1. Direct comparison: adaptive-on vs adaptive-off
    if adaptive_off and adaptive_on:
        print(f"\n  [1/3] Comparing adaptive=1 vs adaptive=0...")
        direct_result = compare_two_runs(
            str(adaptive_off.filepath),
            str(adaptive_on.filepath),
            device=device,
            num_gpus=num_gpus,
            verbose=False
        )
        results['adaptive_on_vs_off'] = direct_result
    else:
        missing = []
        if not adaptive_off:
            missing.append("adaptive-off")
        if not adaptive_on:
            missing.append("adaptive-on")
        print(f"  ⚠ WARNING: Skipping adaptive_on_vs_off for {results['config_name']} in {seed_name} - missing {', '.join(missing)}")

    # 2. Scores A: adaptive-on vs vanilla
    if vanilla and adaptive_on:
        print(f"  [2/3] Comparing adaptive=1 vs vanilla...")
        scores_a = compare_two_runs(
            str(vanilla.filepath),
            str(adaptive_on.filepath),
            device=device,
            num_gpus=num_gpus,
            verbose=False
        )
        results['adaptive_on_vs_vanilla'] = scores_a
    else:
        missing = []
        if not vanilla:
            missing.append("vanilla")
        if not adaptive_on:
            missing.append("adaptive-on")
        print(f"  ⚠ WARNING: Skipping adaptive_on_vs_vanilla for {results['config_name']} in {seed_name} - missing {', '.join(missing)}")

    # 3. Scores B: adaptive-off vs vanilla
    if vanilla and adaptive_off:
        print(f"  [3/3] Comparing adaptive=0 vs vanilla...")
        scores_b = compare_two_runs(
            str(vanilla.filepath),
            str(adaptive_off.filepath),
            device=device,
            num_gpus=num_gpus,
            verbose=False
        )
        results['adaptive_off_vs_vanilla'] = scores_b
    else:
        missing = []
        if not vanilla:
            missing.append("vanilla")
        if not adaptive_off:
            missing.append("adaptive-off")
        print(f"  ⚠ WARNING: Skipping adaptive_off_vs_vanilla for {results['config_name']} in {seed_name} - missing {', '.join(missing)}")

    # 4. Compute delta: A - B (delta-first method)
    if 'adaptive_on_vs_vanilla' in results and 'adaptive_off_vs_vanilla' in results:
        scores_a = results['adaptive_on_vs_vanilla']
        scores_b = results['adaptive_off_vs_vanilla']

        results['score_comparison'] = {
            'delta_f1': scores_a['bertscore_f1_mean'] - scores_b['bertscore_f1_mean'],
            'delta_cosine': scores_a['cosine_mean'] - scores_b['cosine_mean'],
            'delta_quality': scores_a['adaptive_quality_mean'] - scores_b['adaptive_quality_mean'],
        }

    return results


def aggregate_across_seeds(per_seed_results: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Aggregate results across seeds: compute mean ± std for each metric.

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
            for comp_type in ['adaptive_on_vs_off', 'adaptive_on_vs_vanilla', 'adaptive_off_vs_vanilla', 'score_comparison']:
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

        # Get workload info from first available result
        for comp_type, seed_list in comp_data.items():
            if seed_list:
                # Find the original result to get workload info
                for seed_name, workload_results in per_seed_results.items():
                    for res in workload_results:
                        if res['config_name'] == config_name and 'workload' in res:
                            agg_result['workload'] = res['workload']
                            break
                    if 'workload' in agg_result:
                        break
                break

        # Aggregate each comparison type
        for comp_type in ['adaptive_on_vs_off', 'adaptive_on_vs_vanilla', 'adaptive_off_vs_vanilla']:
            if comp_type in comp_data and comp_data[comp_type]:
                agg_result[comp_type] = aggregate_comparison_metrics(comp_data[comp_type])

        # Aggregate deltas (delta-first method)
        if 'score_comparison' in comp_data and comp_data['score_comparison']:
            agg_result['score_comparison'] = aggregate_delta_metrics(comp_data['score_comparison'])

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
        'baseline_quality_mean', 'adaptive_quality_mean', 'quality_pvalue'
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


def print_aggregate_section(title: str, aggregated_results: List[Dict], comparison_key: str, num_seeds: int):
    """Print aggregate metrics with mean ± std for a specific comparison type."""
    print(f"\n{title}")

    # Filter results that have this comparison
    valid_results = [r for r in aggregated_results if comparison_key in r]
    if not valid_results:
        print("  No data available for this comparison.")
        return

    # Compute overall aggregates across configs
    all_exact = []
    all_f1 = []
    all_cosine = []

    for r in valid_results:
        comp = r[comparison_key]
        if 'exact_match' in comp:
            all_exact.append(comp['exact_match']['mean'] * 100)
        if 'bertscore_f1_mean' in comp:
            all_f1.append(comp['bertscore_f1_mean']['mean'])
        if 'cosine_mean' in comp:
            all_cosine.append(comp['cosine_mean']['mean'])

    print(f"\nAggregate Metrics Across All Configurations (N={num_seeds} seeds):")
    if all_exact:
        print(f"  Exact match rate:        {np.mean(all_exact):.1f}% ± {np.std(all_exact, ddof=1):.1f}%")
    if all_f1:
        print(f"  BERTScore F1 (average):  {np.mean(all_f1):.4f} ± {np.std(all_f1, ddof=1):.4f}")
    if all_cosine:
        print(f"  Cosine sim (average):    {np.mean(all_cosine):.4f} ± {np.std(all_cosine, ddof=1):.4f}")

    # Per-configuration summary table with mean ± std
    print(f"\nPer-Configuration Summary:")
    print(f"  {'Config':<15} {'Exact%':<18} {'F1':<18} {'Cosine':<18} {'Quality_p':<18} {'N':<3}")
    print(f"  {'-'*92}")

    for r in valid_results:
        comp = r[comparison_key]
        config = r['config_name']

        # Extract mean ± std for display
        exact = comp.get('exact_match', {})
        f1 = comp.get('bertscore_f1_mean', {})
        cosine = comp.get('cosine_mean', {})
        quality_p = comp.get('quality_pvalue', {})

        n = exact.get('n', 0)
        flag = "  ⚠" if n < num_seeds else "   "

        exact_str = f"{exact.get('mean', 0)*100:.1f} ± {exact.get('std', 0)*100:.1f}" if 'mean' in exact else "N/A"
        f1_str = f"{f1.get('mean', 0):.4f} ± {f1.get('std', 0):.4f}" if 'mean' in f1 else "N/A"
        cosine_str = f"{cosine.get('mean', 0):.4f} ± {cosine.get('std', 0):.4f}" if 'mean' in cosine else "N/A"
        quality_str = f"{quality_p.get('mean', 0):.3f} ± {quality_p.get('std', 0):.3f}" if 'mean' in quality_p else "N/A"

        print(f"  {config:<15} {exact_str:<18} {f1_str:<18} {cosine_str:<18} {quality_str:<18} {n:<3}{flag}")

    if any(r[comparison_key].get('exact_match', {}).get('n', num_seeds) < num_seeds for r in valid_results):
        print(f"\n  Note: ⚠ indicates fewer than {num_seeds} seeds available for this configuration")

    # Interpretation based on mean F1 and cosine
    if all_f1 and all_cosine:
        interpretation = get_interpretation(np.mean(all_f1), np.mean(all_cosine))
        print(f"\nInterpretation:")
        print(f"  ~ {interpretation}")

        if interpretation == "MODERATE SIMILARITY":
            print("    Notable semantic differences exist.")
            print("    May indicate quality differences worth investigating.")


def print_comparison_table(aggregated_results: List[Dict], num_seeds: int):
    """Print delta comparison table with mean ± std."""
    print(f"\nCOMPARISON TABLE (Delta: A - B, computed per-seed then averaged):")
    print(f"  {'Config':<15} {'ΔF1':<18} {'ΔCosine':<18} {'ΔQuality':<18} {'N':<3}")
    print(f"  {'-'*75}")

    all_delta_f1 = []
    all_delta_cosine = []
    all_delta_quality = []

    for r in aggregated_results:
        if 'score_comparison' not in r:
            continue

        config = r['config_name']
        delta = r['score_comparison']

        delta_f1 = delta.get('delta_f1', {})
        delta_cosine = delta.get('delta_cosine', {})
        delta_quality = delta.get('delta_quality', {})

        n = delta_f1.get('n', 0)
        flag = "  ⚠" if n < num_seeds else "   "

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

        print(f"  {config:<15} {f1_str:<18} {cosine_str:<18} {quality_str:<18} {n:<3}{flag}")

    # Overall average
    if all_delta_f1:
        print(f"  {'-'*75}")
        print(f"  {'Average':<15} {np.mean(all_delta_f1):+.4f} ± {np.std(all_delta_f1, ddof=1):.4f}  "
              f"{np.mean(all_delta_cosine):+.4f} ± {np.std(all_delta_cosine, ddof=1):.4f}  "
              f"{np.mean(all_delta_quality):+.4f} ± {np.std(all_delta_quality, ddof=1):.4f}")

        print(f"\nInterpretation:")
        avg_f1_delta = np.mean(all_delta_f1)
        if abs(avg_f1_delta) < 0.01:
            print("  Adaptive speculation has NEGLIGIBLE impact on semantic divergence vs baseline.")
        elif avg_f1_delta > 0:
            print("  Adaptive speculation REDUCES semantic divergence compared to fixed draft length.")
            print("  (Positive delta = adaptive-on is MORE similar to vanilla)")
        else:
            print("  Adaptive speculation INCREASES semantic divergence compared to fixed draft length.")
            print("  (Negative delta = adaptive-on is LESS similar to vanilla)")


def main():
    """Main entry point for multi-seed folder comparison."""
    if len(sys.argv) != 2:
        print("Usage: python compare_runs_folder.py <parent_folder>")
        print("\nExample:")
        print("  python compare_runs_folder.py sweeps/Llama-3.1-8B+EAGLE/coding")
        print("\nExpected structure: <parent_folder>/seed_0/, seed_1/, seed_2/, ...")
        sys.exit(1)

    parent_folder = Path(sys.argv[1])

    if not parent_folder.exists():
        print(f"ERROR: Folder not found: {parent_folder}")
        sys.exit(1)

    print("="*80)
    print("SEMANTIC SIMILARITY EVALUATION - MULTI-SEED ANALYSIS")
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

    # Process each seed
    per_seed_results = {}

    for seed_folder in seed_folders:
        seed_name = seed_folder.name

        print(f"\n{'='*80}")
        print(f"PROCESSING {seed_name}")
        print(f"{'='*80}")

        # Load runs from this seed
        runs = load_runs_from_folder(seed_folder)

        if not runs:
            print(f"  WARNING: No runs found in {seed_name}")
            continue

        print(f"\nLoaded {len(runs)} runs:")
        for run in runs:
            print(f"  {run.filepath.name}: {run.experiment_type}")

        # Group by workload
        grouped = group_by_workload(runs)
        print(f"\nFound {len(grouped)} unique workload configurations")

        # Compare each workload group
        seed_results = []
        for workload_key in sorted(grouped.keys()):
            requests, tokens, temp = workload_key
            print(f"\nWorkload: r{requests}_t{tokens}_temp{temp}")

            result = compare_workload_group(
                workload_key,
                grouped[workload_key],
                device,
                num_gpus,
                seed_name
            )
            seed_results.append(result)

        per_seed_results[seed_name] = seed_results

    # Aggregate across seeds
    print(f"\n{'='*80}")
    print("AGGREGATING ACROSS SEEDS")
    print(f"{'='*80}\n")

    aggregated_results = aggregate_across_seeds(per_seed_results)

    # Print aggregate results
    print("\n" + "="*80)
    print("ADAPTIVE ON vs OFF (SPEC DECODE)")
    print("="*80)

    print_aggregate_section(
        "",
        aggregated_results,
        'adaptive_on_vs_off',
        len(seed_folders)
    )

    print("\n" + "="*80)
    print("SPEC DECODE VS BASELINE")
    print("="*80)

    print_aggregate_section(
        "\nADAPTIVE ON VS BASELINE:",
        aggregated_results,
        'adaptive_on_vs_vanilla',
        len(seed_folders)
    )

    print_aggregate_section(
        "\nADAPTIVE OFF VS BASELINE:",
        aggregated_results,
        'adaptive_off_vs_vanilla',
        len(seed_folders)
    )

    print_comparison_table(aggregated_results, len(seed_folders))

    # Save detailed results (per-seed + aggregated)
    detailed_output = parent_folder / "comparison_results_detailed.json"
    with detailed_output.open('w') as f:
        json.dump({
            'parent_folder': str(parent_folder),
            'seeds': seed_names,
            'num_seeds': len(seed_folders),
            'per_seed_results': per_seed_results,
            'aggregated_results': aggregated_results,
        }, f, indent=2)

    print(f"\n\nDetailed results saved to: {detailed_output}")

    # Save summary (aggregated only, simplified)
    summary_output = parent_folder / "comparison_results_summary.json"

    summary = {
        'parent_folder': str(parent_folder),
        'seeds': seed_names,
        'num_seeds': len(seed_folders),
        'aggregated_summary': aggregated_results,
    }

    with summary_output.open('w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary results saved to: {summary_output}")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
