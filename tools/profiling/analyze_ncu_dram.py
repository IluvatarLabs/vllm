#!/usr/bin/env python3
"""
NCU DRAM Write Analysis - Multi-Seed Aggregation

Analyzes DRAM write savings from early exit experiments with NCU profiling.
Computes per-seed savings percentages, then aggregates with mean ± std.

Usage:
    python analyze_ncu_dram.py sweeps/Llama-3.1-8B-Instruct/coding
"""

import json
import sys
import re
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class NCURunData:
    """Data for a single NCU run with config and DRAM stats."""
    json_path: Path
    csv_path: Path
    config_key: str  # "r{requests}_t{tokens}_temp{temp}_thresh{thresh}"
    num_requests: int
    max_new_tokens: int
    temperature: float
    threshold: float
    total_mb: float
    num_kernels: int
    mean_mb: float
    std_mb: float


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


def find_csv_for_json(json_path: Path) -> Optional[Path]:
    """
    Find the corresponding CSV file for a JSON metadata file.

    For run1_r36_t128_temp0.0_thresh0.0.json, looks for run1_thresh0.0.csv
    """
    # Extract run number from JSON filename
    match = re.search(r'(run\d+)', json_path.name)
    if not match:
        return None

    run_prefix = match.group(1)

    # Look for CSV files with this run prefix and _thresh
    csv_files = list(json_path.parent.glob(f"{run_prefix}_thresh*.csv"))

    if not csv_files:
        return None

    if len(csv_files) > 1:
        # If multiple CSVs, try to match by threshold value
        # For now, just use the first one and warn
        print(f"  ⚠ WARNING: Multiple CSV files found for {run_prefix}, using {csv_files[0].name}")

    return csv_files[0]


def is_anomalous_run(filename: str, num_kernels: int) -> bool:
    """
    Check if this is an anomalous run to skip.

    Anomalous runs have only ~2000 kernels (run4_thresh0.7, run7_thresh0.5_t256)
    due to different NCU capture window (larger batch sizes).
    """
    # Skip if kernel count is suspiciously low
    if num_kernels < 5000:
        return True
    return False


def analyze_csv_dram_writes(csv_path: Path) -> Dict:
    """
    Analyze a single NCU CSV file for DRAM writes.

    Returns:
        Dict with: total_mb, num_kernels, mean_mb, std_mb, or None if anomalous
    """
    dram_writes = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Read header
        next(reader)  # Skip units row

        # Find column index dynamically (CSV reader handles quoted fields correctly)
        try:
            write_col_idx = header.index('dram__bytes_write.sum')
        except ValueError:
            print(f"  ⚠ ERROR: Column 'dram__bytes_write.sum' not found in {csv_path.name}")
            return None

        for row in reader:
            if len(row) <= write_col_idx:
                continue

            # Get DRAM write value from the correct column
            write_mb = row[write_col_idx].strip().strip('"')
            if write_mb:
                try:
                    dram_writes.append(float(write_mb))
                except ValueError:
                    continue

    if not dram_writes:
        return None

    # Check for anomalous run
    if is_anomalous_run(csv_path.name, len(dram_writes)):
        print(f"  ⚠ SKIPPING anomalous run: {csv_path.name} (only {len(dram_writes)} kernels, likely wrong capture window)")
        return None

    # Use nansum/nanmean/nanstd to skip NaN values from NCU profiling errors
    dram_array = np.array(dram_writes)
    nan_count = np.isnan(dram_array).sum()
    if nan_count > 0:
        print(f"  ⚠ WARNING: {nan_count} NaN values found, skipping them")

    return {
        'total_mb': float(np.nansum(dram_array)),
        'num_kernels': len(dram_writes),
        'mean_mb': float(np.nanmean(dram_array)),
        'std_mb': float(np.nanstd(dram_array, ddof=1)) if len(dram_writes) > 1 else 0.0,
    }


def load_ncu_run(json_path: Path) -> Optional[NCURunData]:
    """
    Load a single NCU run from JSON metadata and corresponding CSV.

    Returns:
        NCURunData object, or None if load fails or run is anomalous
    """
    try:
        # Load JSON metadata
        with open(json_path) as f:
            data = json.load(f)

        config = data.get('config', {})

        # Skip if not an NCU run
        if not config.get('enable_ncu'):
            return None

        # Extract config
        num_requests = config.get('num_requests', 0)
        max_new_tokens = config.get('max_new_tokens', 0)
        temperature = config.get('temperature', 0.0)
        threshold = config.get('confidence_threshold')

        if threshold is None:
            print(f"  ⚠ WARNING: Missing threshold in {json_path.name}")
            return None

        # Find corresponding CSV
        csv_path = find_csv_for_json(json_path)
        if not csv_path:
            print(f"  ⚠ WARNING: No CSV file found for {json_path.name}")
            return None

        # Analyze CSV for DRAM stats
        dram_stats = analyze_csv_dram_writes(csv_path)
        if not dram_stats:
            return None

        # Build config key
        config_key = f"r{num_requests}_t{max_new_tokens}_temp{temperature:.1f}_thresh{threshold:.1f}"

        return NCURunData(
            json_path=json_path,
            csv_path=csv_path,
            config_key=config_key,
            num_requests=num_requests,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            threshold=threshold,
            total_mb=dram_stats['total_mb'],
            num_kernels=dram_stats['num_kernels'],
            mean_mb=dram_stats['mean_mb'],
            std_mb=dram_stats['std_mb'],
        )

    except Exception as e:
        print(f"  ⚠ ERROR: Could not load {json_path.name}: {e}")
        return None


def analyze_seed_folder(seed_folder: Path) -> List[NCURunData]:
    """
    Analyze all NCU runs in a single seed folder by loading JSON metadata.

    Returns:
        List of NCURunData objects
    """
    runs = []

    # Find all JSON files for NCU runs (run*.json, but exclude .ncu.json files)
    json_files = [f for f in seed_folder.glob("run*.json") if '.ncu.json' not in f.name]

    if not json_files:
        print(f"  ⚠ WARNING: No JSON files found in {seed_folder.name}")
        return runs

    print(f"\n  Found {len(json_files)} JSON files")

    for json_path in sorted(json_files):
        run_data = load_ncu_run(json_path)
        if run_data:
            runs.append(run_data)
            print(f"    ✓ {json_path.name}: {run_data.config_key} - {run_data.num_kernels} kernels, {run_data.total_mb:.1f} MB total")

    return runs


def compute_per_seed_savings(seed_runs: List[NCURunData]) -> Dict[str, float]:
    """
    Compute DRAM savings percentages vs baseline (thresh=0.0) for a single seed.

    Args:
        seed_runs: List of NCURunData for one seed

    Returns:
        Dict mapping config_key -> savings_pct
    """
    # Group by config to find baselines (thresh=0.0) for each workload
    baselines = {}  # (requests, tokens, temp) -> baseline_mb
    savings = {}  # config_key -> savings_pct

    for run in seed_runs:
        workload_key = (run.num_requests, run.max_new_tokens, run.temperature)
        if run.threshold == 0.0:
            baselines[workload_key] = run.total_mb

    # Compute savings for each run
    for run in seed_runs:
        workload_key = (run.num_requests, run.max_new_tokens, run.temperature)

        if workload_key not in baselines:
            continue  # No baseline for this workload

        baseline_mb = baselines[workload_key]

        if baseline_mb == 0:
            continue  # Can't compute savings

        savings_pct = ((baseline_mb - run.total_mb) / baseline_mb) * 100
        savings[run.config_key] = savings_pct

    return savings


def aggregate_across_seeds(per_seed_runs: Dict[str, List[NCURunData]]) -> Dict[str, Dict]:
    """
    Aggregate DRAM statistics across seeds.

    Args:
        per_seed_runs: Dict mapping seed_name -> List[NCURunData]

    Returns:
        Dict mapping config_key -> aggregated stats with mean ± std
    """
    # Group by config_key
    config_groups = defaultdict(lambda: {
        'total_mb_values': [],
        'savings_pct_values': [],
        'num_kernels_values': [],
        'seed_data': {
            'total_mb': {},
            'savings_pct': {}
        },
        'config_info': None,
    })

    for seed_name, seed_runs in per_seed_runs.items():
        savings = compute_per_seed_savings(seed_runs)

        for run in seed_runs:
            group = config_groups[run.config_key]

            group['total_mb_values'].append(run.total_mb)
            group['num_kernels_values'].append(run.num_kernels)

            # Store per-seed data
            group['seed_data']['total_mb'][seed_name] = run.total_mb

            if run.config_key in savings:
                group['savings_pct_values'].append(savings[run.config_key])
                group['seed_data']['savings_pct'][seed_name] = savings[run.config_key]

            # Store config info (same for all seeds with this config_key)
            if group['config_info'] is None:
                group['config_info'] = {
                    'num_requests': run.num_requests,
                    'max_new_tokens': run.max_new_tokens,
                    'temperature': run.temperature,
                    'threshold': run.threshold,
                }

    # Compute aggregates
    aggregated = {}

    for config_key in sorted(config_groups.keys()):
        group = config_groups[config_key]

        total_mb_vals = group['total_mb_values']
        savings_vals = group['savings_pct_values']
        kernels_vals = group['num_kernels_values']

        aggregated[config_key] = {
            'config_key': config_key,
            'config_info': group['config_info'],
            'total_mb': {
                **group['seed_data']['total_mb'],
                'mean': float(np.mean(total_mb_vals)),
                'std': float(np.std(total_mb_vals, ddof=1)) if len(total_mb_vals) > 1 else 0.0,
                'n': len(total_mb_vals)
            },
            'savings_pct': {
                **group['seed_data']['savings_pct'],
                'mean': float(np.mean(savings_vals)) if savings_vals else 0.0,
                'std': float(np.std(savings_vals, ddof=1)) if len(savings_vals) > 1 else 0.0,
                'n': len(savings_vals)
            } if savings_vals else None,
            'num_kernels': {
                'mean': float(np.mean(kernels_vals)),
                'std': float(np.std(kernels_vals, ddof=1)) if len(kernels_vals) > 1 else 0.0,
                'n': len(kernels_vals)
            }
        }

    return aggregated


def print_results(aggregated: Dict[str, Dict], num_seeds: int):
    """Print formatted table of results."""
    print("\n" + "="*110)
    print("RAW METRICS (All Configurations)")
    print("="*110)

    print(f"\nAggregate Metrics Across {num_seeds} Seeds:\n")
    print(f"  {'Config':<35} {'Total DRAM (MB)':<25} {'Kernels':<15} {'N':<3}")
    print(f"  {'-'*80}")

    for config_key in sorted(aggregated.keys()):
        data = aggregated[config_key]

        total = data['total_mb']
        kernels = data['num_kernels']

        n = total['n']
        flag = "  ⚠" if n < num_seeds else ""

        total_str = f"{total['mean']:.1f} ± {total['std']:.1f}"
        kernels_str = f"{int(kernels['mean'])} ± {int(kernels['std'])}"

        print(f"  {config_key:<35} {total_str:<25} {kernels_str:<15} {n:<3}{flag}")

    if any(data['total_mb']['n'] < num_seeds for data in aggregated.values()):
        print(f"\n  Note: ⚠ indicates fewer than {num_seeds} seeds available for this configuration")

    print("\n" + "="*110)


def print_comparison_results(aggregated: Dict[str, Dict], num_seeds: int):
    """Print formatted table of DRAM savings comparisons (thresh=X vs thresh=0.0)."""
    print("\n" + "="*110)
    print("ON VS OFF COMPARISONS (DRAM Bandwidth Savings)")
    print("="*110)

    # Only show non-baseline configs (those with savings data)
    comparisons = {k: v for k, v in aggregated.items() if v['savings_pct'] is not None and v['savings_pct']['n'] > 0}

    if comparisons:
        print(f"\nEarly Exit Comparisons (thresh=X vs thresh=0.0):\n")
        print(f"  {'Config':<35} {'DRAM Savings%':<20} {'N':<3}")
        print(f"  {'-'*60}")

        for config_key in sorted(comparisons.keys()):
            data = comparisons[config_key]
            savings = data['savings_pct']

            n = savings['n']
            flag = "  ⚠" if n < num_seeds else ""

            savings_str = f"{savings['mean']:.1f}% ± {savings['std']:.1f}%"

            print(f"  {config_key:<35} {savings_str:<20} {n:<3}{flag}")

        if any(data['savings_pct']['n'] < num_seeds for data in comparisons.values()):
            print(f"\n  Note: ⚠ indicates fewer than {num_seeds} seeds available for this comparison")
            print(f"  Note: DRAM Savings% = bandwidth reduction compared to thresh=0.0 baseline")
    else:
        print("\n  ⚠ No comparisons found (need matching baseline thresh=0.0)")

    print("\n" + "="*110)


def main():
    """Main entry point for NCU DRAM analysis."""
    if len(sys.argv) != 2:
        print("Usage: python analyze_ncu_dram.py <parent_folder>")
        print("\nExample:")
        print("  python analyze_ncu_dram.py sweeps/Llama-3.1-8B-Instruct/coding")
        print("\nExpected structure: <parent_folder>/seed_0/, seed_1/, seed_2/, ...")
        print("Each seed folder should contain run*_thresh*.csv files from NCU profiling.")
        sys.exit(1)

    parent_folder = Path(sys.argv[1])

    if not parent_folder.exists():
        print(f"ERROR: Folder not found: {parent_folder}")
        sys.exit(1)

    print("="*90)
    print("NCU DRAM WRITE ANALYSIS - MULTI-SEED AGGREGATION")
    print("="*90)

    # Find seed folders
    print(f"\nScanning for seed folders in: {parent_folder}")
    seed_folders = find_seed_folders(parent_folder)

    if not seed_folders:
        print(f"ERROR: No seed_* folders found in {parent_folder}")
        print("Expected folder structure: <parent>/seed_0/, seed_1/, seed_2/, ...")
        sys.exit(1)

    seed_names = [f.name for f in seed_folders]
    print(f"Seeds found: {', '.join(seed_names)} (N={len(seed_folders)})")

    # Process each seed
    per_seed_data = {}

    for seed_folder in seed_folders:
        seed_name = seed_folder.name

        print(f"\n{'='*90}")
        print(f"PROCESSING {seed_name}")
        print(f"{'='*90}")

        seed_runs = analyze_seed_folder(seed_folder)

        if seed_runs:
            per_seed_data[seed_name] = seed_runs
            print(f"\n  ✓ Loaded {len(seed_runs)} NCU runs for {seed_name}")
        else:
            print(f"\n  ⚠ No valid data for {seed_name}")

    if not per_seed_data:
        print("\nERROR: No valid data found in any seed folder")
        sys.exit(1)

    # Aggregate across seeds
    print(f"\n{'='*90}")
    print("AGGREGATING ACROSS SEEDS")
    print(f"{'='*90}\n")

    aggregated = aggregate_across_seeds(per_seed_data)

    # Print results
    print_results(aggregated, len(seed_folders))
    print_comparison_results(aggregated, len(seed_folders))

    # Save detailed results
    detailed_output = parent_folder / "ncu_dram_analysis_detailed.json"
    with detailed_output.open('w') as f:
        json.dump({
            'parent_folder': str(parent_folder),
            'seeds': seed_names,
            'num_seeds': len(seed_folders),
            'per_seed_data': {
                seed: [
                    {
                        'json_path': str(run.json_path),
                        'csv_path': str(run.csv_path),
                        'config_key': run.config_key,
                        'total_mb': run.total_mb,
                        'num_kernels': run.num_kernels,
                    }
                    for run in runs
                ]
                for seed, runs in per_seed_data.items()
            },
            'aggregated_results': aggregated,
        }, f, indent=2)

    print(f"\nDetailed results saved to: {detailed_output}")

    # Save summary
    summary_output = parent_folder / "ncu_dram_analysis_summary.json"
    with summary_output.open('w') as f:
        json.dump({
            'parent_folder': str(parent_folder),
            'seeds': seed_names,
            'num_seeds': len(seed_folders),
            'aggregated_summary': aggregated,
        }, f, indent=2)

    print(f"Summary results saved to: {summary_output}")

    print("\n" + "="*90)
    print("ANALYSIS COMPLETE")
    print("="*90 + "\n")


if __name__ == '__main__':
    main()
