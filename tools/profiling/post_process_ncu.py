#!/usr/bin/env python3
"""
Post-process NCU report files to extract bandwidth metrics.

Usage:
    python tools/profiling/post_process_ncu.py sweeps/ncu_analysis

This script:
1. Finds all .ncu-rep files in the directory
2. Exports them to CSV using ncu --import
3. Parses and sums the bandwidth metrics
4. Generates a comparison report
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any


def export_ncu_to_csv(ncu_rep_path: Path, output_csv_path: Path) -> bool:
    """Export NCU report to CSV using ncu --import."""
    print(f"  Exporting {ncu_rep_path.name}...", flush=True)

    try:
        cmd = [
            "ncu",
            "--import", str(ncu_rep_path),
            "--csv",
            "--page", "raw",
        ]

        with open(output_csv_path, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.PIPE,
                check=True,
                timeout=300  # 5 minute timeout per file
            )

        print(f"  ✓ Exported to {output_csv_path.name}", flush=True)
        return True

    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout exporting {ncu_rep_path.name}", flush=True)
        return False
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to export {ncu_rep_path.name}: {e.stderr.decode()}", flush=True)
        return False
    except FileNotFoundError:
        print(f"  ✗ ncu command not found. Make sure CUDA toolkit is installed.", flush=True)
        return False


def parse_ncu_csv(csv_path: Path) -> Dict[str, float]:
    """Parse NCU CSV and sum all metrics."""
    metrics = {
        'dram__bytes_read.sum': 0.0,
        'dram__bytes_write.sum': 0.0,
        'lts__t_sectors_op_read.sum': 0.0,
        'lts__t_sectors_op_write.sum': 0.0,
        'dram__throughput.avg.pct_of_peak_sustained_elapsed': 0.0,
        'kernel_count': 0,
        'bw_util_count': 0,
    }

    if not csv_path.exists():
        return metrics

    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Sum DRAM metrics (already in MB from NCU)
                    metrics['dram__bytes_read.sum'] += float(row.get('dram__bytes_read.sum', 0) or 0)
                    metrics['dram__bytes_write.sum'] += float(row.get('dram__bytes_write.sum', 0) or 0)

                    # Sum L2 metrics (in sectors)
                    metrics['lts__t_sectors_op_read.sum'] += float(row.get('lts__t_sectors_op_read.sum', 0) or 0)
                    metrics['lts__t_sectors_op_write.sum'] += float(row.get('lts__t_sectors_op_write.sum', 0) or 0)

                    # Sum BW utilization (for averaging later)
                    bw_util = float(row.get('dram__throughput.avg.pct_of_peak_sustained_elapsed', 0) or 0)
                    if bw_util > 0:
                        metrics['dram__throughput.avg.pct_of_peak_sustained_elapsed'] += bw_util
                        metrics['bw_util_count'] += 1

                    metrics['kernel_count'] += 1

                except (ValueError, KeyError):
                    continue

    except Exception as e:
        print(f"  Warning: Error parsing {csv_path}: {e}", flush=True)

    return metrics


def update_json_with_metrics(json_path: Path, metrics: Dict[str, float]) -> None:
    """Update the benchmark JSON file with NCU metrics."""
    if not json_path.exists():
        print(f"  Warning: JSON file not found: {json_path}", flush=True)
        return

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Update the ncu_metrics field in summary
        if 'summary' in data and 'per_mode' in data['summary']:
            for mode_data in data['summary']['per_mode']:
                # Calculate average BW utilization
                avg_bw_util = 0.0
                if metrics['bw_util_count'] > 0:
                    avg_bw_util = metrics['dram__throughput.avg.pct_of_peak_sustained_elapsed'] / metrics['bw_util_count']

                mode_data['ncu_metrics'] = {
                    'dram__bytes_read.sum': metrics['dram__bytes_read.sum'],
                    'dram__bytes_write.sum': metrics['dram__bytes_write.sum'],
                    'lts__t_sectors_op_read.sum': metrics['lts__t_sectors_op_read.sum'],
                    'lts__t_sectors_op_write.sum': metrics['lts__t_sectors_op_write.sum'],
                    'dram__throughput.avg.pct_of_peak_sustained_elapsed': avg_bw_util,
                    'kernel_count': metrics['kernel_count'],
                }

        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"  ✓ Updated {json_path.name} with NCU metrics", flush=True)

    except Exception as e:
        print(f"  ✗ Error updating JSON {json_path}: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Post-process NCU report files")
    parser.add_argument("directory", help="Directory containing .ncu-rep files")
    parser.add_argument("--export-only", action="store_true", help="Only export to CSV, don't update JSON")
    args = parser.parse_args()

    sweep_dir = Path(args.directory)
    if not sweep_dir.exists():
        print(f"Error: Directory not found: {sweep_dir}")
        sys.exit(1)

    # Find all NCU report files
    ncu_reports = sorted(sweep_dir.glob("*.ncu-rep"))

    if not ncu_reports:
        print(f"No .ncu-rep files found in {sweep_dir}")
        sys.exit(1)

    print(f"Found {len(ncu_reports)} NCU report files")
    print("=" * 80)

    results = {}

    for ncu_rep_path in ncu_reports:
        # Determine test name from filename
        # e.g., "small_baseline_t0.7.off-off.ncu.ncu-rep" -> "small_baseline_t0.7"
        stem = ncu_rep_path.stem.replace('.ncu', '')
        test_name = stem.rsplit('.', 2)[0]  # Remove ".off-off" or ".off-stage"

        print(f"\n{test_name}:")

        # Export to CSV
        csv_path = ncu_rep_path.with_suffix('.csv')
        if not export_ncu_to_csv(ncu_rep_path, csv_path):
            continue

        # Parse metrics
        metrics = parse_ncu_csv(csv_path)
        results[test_name] = metrics

        # Display summary
        dram_read_gb = metrics['dram__bytes_read.sum'] / 1024  # MB to GB
        dram_write_gb = metrics['dram__bytes_write.sum'] / 1024  # MB to GB
        l2_write_m = metrics['lts__t_sectors_op_write.sum'] / 1e6  # sectors to M
        avg_bw = metrics['dram__throughput.avg.pct_of_peak_sustained_elapsed'] / metrics['bw_util_count'] if metrics['bw_util_count'] > 0 else 0

        print(f"  Kernels: {metrics['kernel_count']}")
        print(f"  DRAM Read:  {dram_read_gb:.2f} GB")
        print(f"  DRAM Write: {dram_write_gb:.2f} GB")
        print(f"  L2 Write:   {l2_write_m:.1f} M sectors")
        print(f"  Avg BW Util: {avg_bw:.2f}%")

        # Update JSON file if not export-only
        if not args.export_only:
            json_path = sweep_dir / f"{test_name}.json"
            update_json_with_metrics(json_path, metrics)

    # Generate comparison report
    print("\n" + "=" * 80)
    print("COMPARISON REPORT")
    print("=" * 80)

    test_pairs = [
        ("small_baseline_t0.7", "small_nwor_t0.7", "Small Batch (temp 0.7)"),
        ("small_baseline_t0.0", "small_nwor_t0.0", "Small Batch (temp 0.0)"),
        ("medium_baseline_t0.7", "medium_nwor_t0.7", "Medium Batch"),
        ("large_baseline_t0.7", "large_nwor_t0.7", "Large Batch"),
        ("sustained_baseline_t0.7", "sustained_nwor_t0.7", "Sustained Load"),
    ]

    for baseline_name, nwor_name, description in test_pairs:
        baseline = results.get(baseline_name)
        nwor = results.get(nwor_name)

        if not baseline or not nwor:
            continue

        print(f"\n{description}:")

        baseline_write_gb = baseline['dram__bytes_write.sum'] / 1024
        nwor_write_gb = nwor['dram__bytes_write.sum'] / 1024

        baseline_l2_write_m = baseline['lts__t_sectors_op_write.sum'] / 1e6
        nwor_l2_write_m = nwor['lts__t_sectors_op_write.sum'] / 1e6

        if baseline_write_gb > 0:
            dram_write_delta_pct = ((nwor_write_gb - baseline_write_gb) / baseline_write_gb) * 100
            print(f"  Baseline DRAM Write: {baseline_write_gb:.2f} GB")
            print(f"  NWOR DRAM Write:     {nwor_write_gb:.2f} GB")
            print(f"  DRAM Write Δ:        {dram_write_delta_pct:+.2f}%")

        if baseline_l2_write_m > 0:
            l2_write_delta_pct = ((nwor_l2_write_m - baseline_l2_write_m) / baseline_l2_write_m) * 100
            print(f"  L2 Write Δ:          {l2_write_delta_pct:+.2f}%")

        # Verdict
        if baseline_write_gb > 0:
            if dram_write_delta_pct < -5:
                print(f"  ✓ NWOR is helping! ({abs(dram_write_delta_pct):.1f}% write reduction)")
            elif abs(dram_write_delta_pct) < 5:
                print(f"  ~ NWOR has minimal impact")
            else:
                print(f"  ✗ NWOR is increasing writes!")

    print("\n" + "=" * 80)
    print("Post-processing complete!")


if __name__ == "__main__":
    main()
