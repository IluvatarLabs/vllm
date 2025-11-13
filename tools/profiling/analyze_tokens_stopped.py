#!/usr/bin/env python3
"""
Tokens Stopped Analysis - Multi-Seed Aggregation

Analyzes token stopping from confidence-based early exit experiments.
Computes per-seed stop rates, then aggregates with mean ± std.

Usage:
    python analyze_tokens_stopped.py sweeps/Llama-3.1-8B-Instruct/coding
"""

import json
import sys
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TokenStoppedRunData:
    """Data for a single early exit run with token stopping stats."""
    json_path: Path
    config_key: str  # "r{requests}_t{tokens}_temp{temp}_thresh{thresh}"
    num_requests: int
    max_new_tokens: int
    temperature: float
    threshold: float
    tokens_stopped: int
    total_checks: int
    stop_rate_pct: float


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


def load_token_stopped_run(json_path: Path) -> Optional[TokenStoppedRunData]:
    """
    Load a single early exit run from JSON with token stopping stats.

    Returns:
        TokenStoppedRunData object, or None if load fails or data missing
    """
    try:
        # Load JSON metadata
        with open(json_path) as f:
            data = json.load(f)

        config = data.get('config', {})

        # Extract config
        num_requests = config.get('num_requests', 0)
        max_new_tokens = config.get('max_new_tokens', 0)
        temperature = config.get('temperature', 0.0)
        threshold = config.get('confidence_threshold')

        # Skip baseline runs (thresh=0.0) or runs without threshold
        if threshold is None or threshold == 0.0:
            return None

        # Extract tokens stopped from per_mode data
        summary = data.get('summary', {})
        per_mode = summary.get('per_mode', [])

        if not per_mode:
            print(f"  ⚠ WARNING: No per_mode data in {json_path.name}")
            return None

        mode_data = per_mode[0]

        tokens_stopped = mode_data.get('tokens_stopped_by_confidence')
        total_checks = mode_data.get('total_confidence_checks')

        # Skip if early exit data not present
        if tokens_stopped is None or total_checks is None:
            # Silently skip - likely an old run without token tracking
            return None

        # Compute stop rate
        stop_rate_pct = (tokens_stopped / total_checks * 100) if total_checks > 0 else 0.0

        # Build config key - use filename for scenarios, standard format for grid runs
        is_scenario = 'scenario' in json_path.name
        if is_scenario:
            config_key = json_path.stem  # Use full filename for scenarios
        else:
            config_key = f"r{num_requests}_t{max_new_tokens}_temp{temperature:.1f}_thresh{threshold:.1f}"

        return TokenStoppedRunData(
            json_path=json_path,
            config_key=config_key,
            num_requests=num_requests,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            threshold=threshold,
            tokens_stopped=tokens_stopped,
            total_checks=total_checks,
            stop_rate_pct=stop_rate_pct,
        )

    except Exception as e:
        print(f"  ⚠ ERROR: Could not load {json_path.name}: {e}")
        return None


def analyze_seed_folder(seed_folder: Path) -> List[TokenStoppedRunData]:
    """
    Analyze all early exit runs in a single seed folder by loading JSON metadata.

    Returns:
        List of TokenStoppedRunData objects (deduplicated by config_key)
    """
    # Find all JSON files (exclude .ncu.json metadata files)
    json_files = [f for f in seed_folder.glob("*.json") if '.ncu.json' not in f.name]

    if not json_files:
        print(f"  ⚠ WARNING: No JSON files found in {seed_folder.name}")
        return []

    print(f"\n  Found {len(json_files)} JSON files")

    # Use dict to deduplicate by config_key (keeps last occurrence)
    runs_dict = {}

    for json_path in sorted(json_files):
        run_data = load_token_stopped_run(json_path)
        if run_data:
            # Check for duplicates
            if run_data.config_key in runs_dict:
                prev_file = runs_dict[run_data.config_key].json_path.name
                print(f"  ⚠ WARNING: Duplicate config {run_data.config_key}")
                print(f"    Previous: {prev_file}")
                print(f"    Current:  {json_path.name}")
                print(f"    → Keeping current (last occurrence)")

            runs_dict[run_data.config_key] = run_data
            print(f"    ✓ {json_path.name}: {run_data.config_key} - {run_data.tokens_stopped}/{run_data.total_checks} ({run_data.stop_rate_pct:.1f}%)")

    runs = list(runs_dict.values())

    if len(runs) < len([f for f in json_files if load_token_stopped_run(f) is not None]):
        print(f"  → Deduplicated to {len(runs)} unique configs")

    return runs


def aggregate_across_seeds(per_seed_runs: Dict[str, List[TokenStoppedRunData]]) -> Dict[str, Dict]:
    """
    Aggregate token stopping statistics across seeds.

    Args:
        per_seed_runs: Dict mapping seed_name -> List[TokenStoppedRunData]

    Returns:
        Dict mapping config_key -> aggregated stats with mean ± std
    """
    # Group by config_key
    config_groups = defaultdict(lambda: {
        'tokens_stopped_values': [],
        'total_checks_values': [],
        'stop_rate_pct_values': [],
        'seed_data': {
            'tokens_stopped': {},
            'stop_rate_pct': {}
        },
        'config_info': None,
    })

    for seed_name, seed_runs in per_seed_runs.items():
        for run in seed_runs:
            group = config_groups[run.config_key]

            group['tokens_stopped_values'].append(run.tokens_stopped)
            group['total_checks_values'].append(run.total_checks)
            group['stop_rate_pct_values'].append(run.stop_rate_pct)

            # Store per-seed data
            group['seed_data']['tokens_stopped'][seed_name] = run.tokens_stopped
            group['seed_data']['stop_rate_pct'][seed_name] = run.stop_rate_pct

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

        tokens_stopped_vals = group['tokens_stopped_values']
        total_checks_vals = group['total_checks_values']
        stop_rate_vals = group['stop_rate_pct_values']

        aggregated[config_key] = {
            'config_key': config_key,
            'config_info': group['config_info'],
            'tokens_stopped': {
                **group['seed_data']['tokens_stopped'],
                'mean': float(np.mean(tokens_stopped_vals)),
                'std': float(np.std(tokens_stopped_vals, ddof=1)) if len(tokens_stopped_vals) > 1 else 0.0,
                'n': len(tokens_stopped_vals)
            },
            'total_checks': {
                'mean': float(np.mean(total_checks_vals)),
                'std': float(np.std(total_checks_vals, ddof=1)) if len(total_checks_vals) > 1 else 0.0,
                'n': len(total_checks_vals)
            },
            'stop_rate_pct': {
                **group['seed_data']['stop_rate_pct'],
                'mean': float(np.mean(stop_rate_vals)),
                'std': float(np.std(stop_rate_vals, ddof=1)) if len(stop_rate_vals) > 1 else 0.0,
                'n': len(stop_rate_vals)
            }
        }

    return aggregated


def print_results(aggregated: Dict[str, Dict], num_seeds: int):
    """Print formatted table of results."""
    print("\n" + "="*110)
    print("TOKEN STOPPING METRICS (All Configurations)")
    print("="*110)

    print(f"\nAggregate Metrics Across {num_seeds} Seeds:\n")
    print(f"  {'Config':<35} {'Tokens Stopped':<20} {'Total Checks':<20} {'Stop Rate%':<20} {'N':<3}")
    print(f"  {'-'*100}")

    for config_key in sorted(aggregated.keys()):
        data = aggregated[config_key]

        tokens_stopped = data['tokens_stopped']
        total_checks = data['total_checks']
        stop_rate = data['stop_rate_pct']

        n = tokens_stopped['n']
        flag = "  ⚠" if n < num_seeds else ""

        tokens_stopped_str = f"{int(tokens_stopped['mean'])} ± {int(tokens_stopped['std'])}"
        total_checks_str = f"{int(total_checks['mean'])} ± {int(total_checks['std'])}"
        stop_rate_str = f"{stop_rate['mean']:.1f}% ± {stop_rate['std']:.1f}%"

        print(f"  {config_key:<35} {tokens_stopped_str:<20} {total_checks_str:<20} {stop_rate_str:<20} {n:<3}{flag}")

    if any(data['tokens_stopped']['n'] < num_seeds for data in aggregated.values()):
        print(f"\n  Note: ⚠ indicates fewer than {num_seeds} seeds available for this configuration")

    print("\n" + "="*110)
    print(f"\nKey Insights:")
    print(f"  - Stop Rate% = percentage of draft tokens stopped by early exit")
    print(f"  - Higher stop rate → more KV cache writes saved → more DRAM bandwidth saved")
    print(f"  - Total Checks ≈ spec_num_draft_tokens (validated during data collection)")
    print("\n" + "="*110)


def main():
    """Main entry point for token stopping analysis."""
    if len(sys.argv) != 2:
        print("Usage: python analyze_tokens_stopped.py <parent_folder>")
        print("\nExample:")
        print("  python analyze_tokens_stopped.py sweeps/Llama-3.1-8B-Instruct/coding")
        print("\nExpected structure: <parent_folder>/seed_0/, seed_1/, seed_2/, ...")
        print("Each seed folder should contain run*_thresh*.json files from early exit experiments.")
        sys.exit(1)

    parent_folder = Path(sys.argv[1])

    if not parent_folder.exists():
        print(f"ERROR: Folder not found: {parent_folder}")
        sys.exit(1)

    print("="*90)
    print("TOKEN STOPPING ANALYSIS - MULTI-SEED AGGREGATION")
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
            print(f"\n  ✓ Loaded {len(seed_runs)} early exit runs for {seed_name}")
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
    detailed_output = parent_folder / "tokens_stopped_analysis_detailed.json"
    with detailed_output.open('w') as f:
        json.dump({
            'parent_folder': str(parent_folder),
            'seeds': seed_names,
            'num_seeds': len(seed_folders),
            'per_seed_data': {
                seed: [
                    {
                        'json_path': str(run.json_path),
                        'config_key': run.config_key,
                        'tokens_stopped': run.tokens_stopped,
                        'total_checks': run.total_checks,
                        'stop_rate_pct': run.stop_rate_pct,
                    }
                    for run in runs
                ]
                for seed, runs in per_seed_data.items()
            },
            'aggregated_results': aggregated,
        }, f, indent=2)

    print(f"\nDetailed results saved to: {detailed_output}")

    # Save summary
    summary_output = parent_folder / "tokens_stopped_analysis_summary.json"
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
