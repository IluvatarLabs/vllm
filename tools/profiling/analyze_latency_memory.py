#!/usr/bin/env python3
"""
Latency & Memory Analysis - Multi-Seed Aggregation

Analyzes latency, memory, and acceptance metrics from experiment runs.
Computes per-seed values, then aggregates with mean ± std.

Usage:
    python analyze_latency_memory.py sweeps/Llama-3.1-8B-Instruct/coding
"""

import json
import sys
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class RunMetrics:
    """Metrics extracted from a single run."""
    filepath: Path
    config_key: str  # Unique identifier: "r{requests}_t{tokens}_temp{temp}_thresh{thresh}" or "r{requests}_t{tokens}_temp{temp}_adaptive{adaptive}_d{draft}"
    num_requests: int
    max_new_tokens: int
    threshold: Optional[float]
    temperature: float
    adaptive: int
    draft_tokens: int
    latency_avg_s: float
    latency_p50_s: float
    latency_p95_s: float
    peak_memory_gb: float
    spec_acceptance_ratio: float
    spec_num_drafts: int
    spec_num_draft_tokens: int
    spec_num_accepted_tokens: int


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


def load_run_metrics(json_path: Path) -> Optional[RunMetrics]:
    """
    Load metrics from a single run JSON file.

    Returns:
        RunMetrics object, or None if file can't be loaded
    """
    try:
        with open(json_path) as f:
            data = json.load(f)

        config = data.get('config', {})
        summary = data.get('summary', {})

        # Get per-mode metrics (should only be one mode in early exit runs)
        per_mode = summary.get('per_mode', [])
        if not per_mode:
            print(f"  ⚠ WARNING: No per_mode data in {json_path.name}")
            return None

        if len(per_mode) == 0:
            print(f"  ⚠ WARNING: Empty per_mode array in {json_path.name}")
            return None

        mode_data = per_mode[0]  # Take first mode

        # Extract full config
        num_requests = config.get('num_requests', 0)
        max_new_tokens = config.get('max_new_tokens', 0)
        threshold = config.get('confidence_threshold')
        temperature = config.get('temperature', 0.0)
        adaptive = config.get('adaptive_draft_length', 0)
        draft_tokens = config.get('draft_tokens', 10)

        # Build config key with ALL parameters to avoid grouping different workloads
        # Distinguish between early exit grid and adaptive grid
        if config.get('enable_ncu'):  # Early exit grid
            if threshold is None:
                print(f"  ⚠ WARNING: Missing threshold in NCU run {json_path.name}")
                return None
            config_key = f"r{num_requests}_t{max_new_tokens}_temp{temperature:.1f}_thresh{threshold:.1f}"
        else:  # Adaptive grid
            config_key = f"r{num_requests}_t{max_new_tokens}_temp{temperature:.1f}_adaptive{adaptive}_d{draft_tokens}"

        return RunMetrics(
            filepath=json_path,
            config_key=config_key,
            num_requests=num_requests,
            max_new_tokens=max_new_tokens,
            threshold=threshold,
            temperature=temperature,
            adaptive=adaptive,
            draft_tokens=draft_tokens,
            latency_avg_s=mode_data.get('latency_avg_s', 0.0),
            latency_p50_s=mode_data.get('latency_p50_s', 0.0),
            latency_p95_s=mode_data.get('latency_p95_s', 0.0),
            peak_memory_gb=mode_data.get('peak_memory_gb', 0.0),
            spec_acceptance_ratio=mode_data.get('spec_acceptance_ratio', 0.0),
            spec_num_drafts=mode_data.get('spec_num_drafts', 0),
            spec_num_draft_tokens=mode_data.get('spec_num_draft_tokens', 0),
            spec_num_accepted_tokens=mode_data.get('spec_num_accepted_tokens', 0),
        )

    except Exception as e:
        print(f"  ⚠ WARNING: Could not load {json_path.name}: {e}")
        return None


def load_seed_metrics(seed_folder: Path) -> List[RunMetrics]:
    """Load all run metrics from a single seed folder."""
    metrics = []

    json_files = list(seed_folder.glob("run*.json"))

    # Filter out NCU-specific JSONs (e.g., *.ncu.json)
    json_files = [f for f in json_files if '.ncu.json' not in f.name]

    if not json_files:
        print(f"  ⚠ WARNING: No JSON files found in {seed_folder.name}")
        return metrics

    print(f"\n  Found {len(json_files)} JSON files")

    for json_path in sorted(json_files):
        run_metrics = load_run_metrics(json_path)
        if run_metrics:
            metrics.append(run_metrics)

    return metrics


def aggregate_metrics(per_seed_metrics: Dict[str, List[RunMetrics]]) -> Dict[str, Dict]:
    """
    Aggregate metrics across seeds.

    Args:
        per_seed_metrics: Dict mapping seed_name -> list of RunMetrics

    Returns:
        Dict mapping config_key -> aggregated metrics with mean ± std
    """
    # Group by config_key
    config_groups = defaultdict(lambda: {
        'latency_avg_s': [],
        'latency_p50_s': [],
        'latency_p95_s': [],
        'peak_memory_gb': [],
        'spec_acceptance_ratio': [],
        'spec_num_drafts': [],
        'spec_num_draft_tokens': [],
        'spec_num_accepted_tokens': [],
        'seed_data': defaultdict(dict),
        'config_info': None,
    })

    for seed_name, metrics_list in per_seed_metrics.items():
        for metrics in metrics_list:
            group = config_groups[metrics.config_key]

            # Aggregate values
            group['latency_avg_s'].append(metrics.latency_avg_s)
            group['latency_p50_s'].append(metrics.latency_p50_s)
            group['latency_p95_s'].append(metrics.latency_p95_s)
            group['peak_memory_gb'].append(metrics.peak_memory_gb)
            group['spec_acceptance_ratio'].append(metrics.spec_acceptance_ratio)
            group['spec_num_drafts'].append(metrics.spec_num_drafts)
            group['spec_num_draft_tokens'].append(metrics.spec_num_draft_tokens)
            group['spec_num_accepted_tokens'].append(metrics.spec_num_accepted_tokens)

            # Store per-seed data
            group['seed_data']['latency_avg_s'][seed_name] = metrics.latency_avg_s
            group['seed_data']['peak_memory_gb'][seed_name] = metrics.peak_memory_gb
            group['seed_data']['spec_acceptance_ratio'][seed_name] = metrics.spec_acceptance_ratio

            # Store config info (same for all seeds)
            if group['config_info'] is None:
                group['config_info'] = {
                    'num_requests': metrics.num_requests,
                    'max_new_tokens': metrics.max_new_tokens,
                    'threshold': metrics.threshold,
                    'temperature': metrics.temperature,
                    'adaptive': metrics.adaptive,
                    'draft_tokens': metrics.draft_tokens,
                }

    # Compute aggregates
    aggregated = {}

    for config_key in sorted(config_groups.keys()):
        group = config_groups[config_key]

        # Helper to compute mean/std
        def agg_metric(values):
            return {
                'mean': float(np.mean(values)),
                'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                'n': len(values)
            }

        aggregated[config_key] = {
            'config_key': config_key,
            'config_info': group['config_info'],
            'latency_avg_s': {
                **group['seed_data']['latency_avg_s'],
                **agg_metric(group['latency_avg_s'])
            },
            'latency_p50_s': agg_metric(group['latency_p50_s']),
            'latency_p95_s': agg_metric(group['latency_p95_s']),
            'peak_memory_gb': {
                **group['seed_data']['peak_memory_gb'],
                **agg_metric(group['peak_memory_gb'])
            },
            'spec_acceptance_ratio': {
                **group['seed_data']['spec_acceptance_ratio'],
                **agg_metric(group['spec_acceptance_ratio'])
            },
            'spec_num_drafts': agg_metric(group['spec_num_drafts']),
            'spec_num_draft_tokens': agg_metric(group['spec_num_draft_tokens']),
            'spec_num_accepted_tokens': agg_metric(group['spec_num_accepted_tokens']),
        }

    return aggregated


def print_results(aggregated: Dict[str, Dict], num_seeds: int):
    """Print formatted table of results."""
    print("\n" + "="*110)
    print("LATENCY & MEMORY ANALYSIS")
    print("="*110)

    print(f"\nAggregate Metrics Across {num_seeds} Seeds:\n")
    print(f"  {'Config':<25} {'Latency (s)':<18} {'Memory (GB)':<18} {'Accept%':<18} {'N':<3}")
    print(f"  {'-'*100}")

    for config_key in sorted(aggregated.keys()):
        data = aggregated[config_key]

        latency = data['latency_avg_s']
        memory = data['peak_memory_gb']
        accept = data['spec_acceptance_ratio']

        n = latency['n']
        flag = "  ⚠" if n < num_seeds else ""

        latency_str = f"{latency['mean']:.2f} ± {latency['std']:.2f}"
        memory_str = f"{memory['mean']:.2f} ± {memory['std']:.2f}"
        accept_str = f"{accept['mean']*100:.1f} ± {accept['std']*100:.1f}"

        print(f"  {config_key:<25} {latency_str:<18} {memory_str:<18} {accept_str:<18} {n:<3}{flag}")

    if any(data['latency_avg_s']['n'] < num_seeds for data in aggregated.values()):
        print(f"\n  Note: ⚠ indicates fewer than {num_seeds} seeds available for this configuration")

    print("\n" + "="*110)


def main():
    """Main entry point for latency & memory analysis."""
    if len(sys.argv) != 2:
        print("Usage: python analyze_latency_memory.py <parent_folder>")
        print("\nExample:")
        print("  python analyze_latency_memory.py sweeps/Llama-3.1-8B-Instruct/coding")
        print("\nExpected structure: <parent_folder>/seed_0/, seed_1/, seed_2/, ...")
        print("Each seed folder should contain run*.json files from experiments.")
        sys.exit(1)

    parent_folder = Path(sys.argv[1])

    if not parent_folder.exists():
        print(f"ERROR: Folder not found: {parent_folder}")
        sys.exit(1)

    print("="*110)
    print("LATENCY & MEMORY ANALYSIS - MULTI-SEED AGGREGATION")
    print("="*110)

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
    per_seed_metrics = {}

    for seed_folder in seed_folders:
        seed_name = seed_folder.name

        print(f"\n{'='*110}")
        print(f"PROCESSING {seed_name}")
        print(f"{'='*110}")

        metrics = load_seed_metrics(seed_folder)

        if metrics:
            per_seed_metrics[seed_name] = metrics
            print(f"\n  ✓ Loaded {len(metrics)} runs for {seed_name}")
        else:
            print(f"\n  ⚠ No valid data for {seed_name}")

    if not per_seed_metrics:
        print("\nERROR: No valid data found in any seed folder")
        sys.exit(1)

    # Aggregate across seeds
    print(f"\n{'='*110}")
    print("AGGREGATING ACROSS SEEDS")
    print(f"{'='*110}\n")

    aggregated = aggregate_metrics(per_seed_metrics)

    # Print results
    print_results(aggregated, len(seed_folders))

    # Save detailed results
    detailed_output = parent_folder / "latency_memory_analysis_detailed.json"
    with detailed_output.open('w') as f:
        json.dump({
            'parent_folder': str(parent_folder),
            'seeds': seed_names,
            'num_seeds': len(seed_folders),
            'per_seed_data': {
                seed: [
                    {
                        'filepath': str(m.filepath),
                        'config_key': m.config_key,
                        'latency_avg_s': m.latency_avg_s,
                        'peak_memory_gb': m.peak_memory_gb,
                        'spec_acceptance_ratio': m.spec_acceptance_ratio,
                    }
                    for m in metrics
                ]
                for seed, metrics in per_seed_metrics.items()
            },
            'aggregated_results': aggregated,
        }, f, indent=2)

    print(f"\nDetailed results saved to: {detailed_output}")

    # Save summary
    summary_output = parent_folder / "latency_memory_analysis_summary.json"
    with summary_output.open('w') as f:
        json.dump({
            'parent_folder': str(parent_folder),
            'seeds': seed_names,
            'num_seeds': len(seed_folders),
            'aggregated_summary': aggregated,
        }, f, indent=2)

    print(f"Summary results saved to: {summary_output}")

    print("\n" + "="*110)
    print("ANALYSIS COMPLETE")
    print("="*110 + "\n")


if __name__ == '__main__':
    main()
