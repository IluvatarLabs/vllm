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
from typing import Dict, List
from collections import defaultdict


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


def extract_threshold_from_filename(filename: str) -> float:
    """Extract threshold value from CSV filename (e.g., run1_thresh0.5.csv -> 0.5)"""
    match = re.search(r'thresh([0-9]+(?:\.[0-9]+)?)', filename)
    if match:
        return float(match.group(1))
    return None


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

    return {
        'total_mb': sum(dram_writes),
        'num_kernels': len(dram_writes),
        'mean_mb': np.mean(dram_writes),
        'std_mb': np.std(dram_writes, ddof=1) if len(dram_writes) > 1 else 0.0,
    }


def analyze_seed_folder(seed_folder: Path) -> Dict[float, Dict]:
    """
    Analyze all NCU CSV files in a single seed folder.

    Returns:
        Dict mapping threshold -> DRAM stats
    """
    results = {}

    csv_files = list(seed_folder.glob("run*_thresh*.csv"))

    if not csv_files:
        print(f"  ⚠ WARNING: No CSV files found in {seed_folder.name}")
        return results

    print(f"\n  Found {len(csv_files)} CSV files:")

    for csv_path in sorted(csv_files):
        threshold = extract_threshold_from_filename(csv_path.name)
        if threshold is None:
            print(f"  ⚠ Skipping {csv_path.name}: couldn't extract threshold")
            continue

        print(f"    Processing {csv_path.name}...", end=" ")

        stats = analyze_csv_dram_writes(csv_path)
        if stats:
            results[threshold] = stats
            print(f"✓ {stats['num_kernels']} kernels, {stats['total_mb']:.1f} MB total")
        else:
            print("✗ (skipped)")

    return results


def compute_per_seed_savings(seed_results: Dict[float, Dict]) -> Dict[float, float]:
    """
    Compute DRAM savings percentages vs baseline (thresh=0.0) for a single seed.

    Returns:
        Dict mapping threshold -> savings_pct
    """
    if 0.0 not in seed_results:
        print("  ⚠ WARNING: No baseline (thresh=0.0) found, cannot compute savings")
        return {}

    baseline_mb = seed_results[0.0]['total_mb']

    if baseline_mb == 0:
        print("  ⚠ WARNING: Baseline DRAM write is 0 MB, cannot compute savings percentage")
        return {}

    savings = {}

    for threshold, stats in seed_results.items():
        total_mb = stats['total_mb']
        savings_pct = ((baseline_mb - total_mb) / baseline_mb) * 100
        savings[threshold] = savings_pct

    return savings


def aggregate_across_seeds(per_seed_data: Dict[str, Dict]) -> Dict[float, Dict]:
    """
    Aggregate DRAM statistics across seeds.

    Args:
        per_seed_data: Dict mapping seed_name -> {threshold -> stats}

    Returns:
        Dict mapping threshold -> aggregated stats with mean ± std
    """
    # Group by threshold
    threshold_groups = defaultdict(lambda: {
        'total_mb_values': [],
        'savings_pct_values': [],
        'num_kernels_values': [],
        'seed_data': {}
    })

    for seed_name, seed_results in per_seed_data.items():
        savings = compute_per_seed_savings(seed_results)

        for threshold, stats in seed_results.items():
            group = threshold_groups[threshold]

            group['total_mb_values'].append(stats['total_mb'])
            group['num_kernels_values'].append(stats['num_kernels'])

            if threshold in savings:
                group['savings_pct_values'].append(savings[threshold])

            # Store per-seed data
            if 'total_mb' not in group['seed_data']:
                group['seed_data']['total_mb'] = {}
            if 'savings_pct' not in group['seed_data']:
                group['seed_data']['savings_pct'] = {}

            group['seed_data']['total_mb'][seed_name] = stats['total_mb']
            if threshold in savings:
                group['seed_data']['savings_pct'][seed_name] = savings[threshold]

    # Compute aggregates
    aggregated = {}

    for threshold in sorted(threshold_groups.keys()):
        group = threshold_groups[threshold]

        total_mb_vals = group['total_mb_values']
        savings_vals = group['savings_pct_values']
        kernels_vals = group['num_kernels_values']

        aggregated[threshold] = {
            'threshold': threshold,
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


def print_results(aggregated: Dict[float, Dict], num_seeds: int):
    """Print formatted table of results."""
    print("\n" + "="*90)
    print("NCU DRAM WRITE SAVINGS (Early Exit Grid)")
    print("="*90)

    print(f"\nAggregate Metrics Across {num_seeds} Seeds:\n")
    print(f"  {'Threshold':<12} {'Savings%':<20} {'Total DRAM (MB)':<25} {'Kernels':<15} {'N':<3}")
    print(f"  {'-'*85}")

    for threshold in sorted(aggregated.keys()):
        data = aggregated[threshold]

        total = data['total_mb']
        savings = data['savings_pct']
        kernels = data['num_kernels']

        n = total['n']
        flag = "  ⚠" if n < num_seeds else ""

        if savings and savings['n'] > 0:
            savings_str = f"{savings['mean']:.1f}% ± {savings['std']:.1f}%"
        else:
            savings_str = "0.0% (baseline)"

        total_str = f"{total['mean']:.1f} ± {total['std']:.1f}"
        kernels_str = f"{int(kernels['mean'])} ± {int(kernels['std'])}"

        print(f"  {threshold:<12.1f} {savings_str:<20} {total_str:<25} {kernels_str:<15} {n:<3}{flag}")

    if any(data['total_mb']['n'] < num_seeds for data in aggregated.values()):
        print(f"\n  Note: ⚠ indicates fewer than {num_seeds} seeds available for this threshold")

    print("\n" + "="*90)


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

        seed_results = analyze_seed_folder(seed_folder)

        if seed_results:
            per_seed_data[seed_name] = seed_results
            print(f"\n  ✓ Processed {len(seed_results)} thresholds for {seed_name}")
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

    # Save detailed results
    detailed_output = parent_folder / "ncu_dram_analysis_detailed.json"
    with detailed_output.open('w') as f:
        json.dump({
            'parent_folder': str(parent_folder),
            'seeds': seed_names,
            'num_seeds': len(seed_folders),
            'per_seed_data': {
                seed: {str(k): v for k, v in data.items()}
                for seed, data in per_seed_data.items()
            },
            'aggregated_results': {str(k): v for k, v in aggregated.items()},
        }, f, indent=2)

    print(f"\nDetailed results saved to: {detailed_output}")

    # Save summary
    summary_output = parent_folder / "ncu_dram_analysis_summary.json"
    with summary_output.open('w') as f:
        json.dump({
            'parent_folder': str(parent_folder),
            'seeds': seed_names,
            'num_seeds': len(seed_folders),
            'aggregated_summary': {str(k): v for k, v in aggregated.items()},
        }, f, indent=2)

    print(f"Summary results saved to: {summary_output}")

    print("\n" + "="*90)
    print("ANALYSIS COMPLETE")
    print("="*90 + "\n")


if __name__ == '__main__':
    main()
