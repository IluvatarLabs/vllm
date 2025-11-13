#!/usr/bin/env python3
"""
Temporary script to generate CSV files from existing NCU .ncu-rep files.

This script is needed because the original profiling run had a bug where
NCU reports were created but CSVs were not exported due to a file naming mismatch.

Usage:
    python tools/profiling/generate_ncu_csvs.py sweeps/Llama-3.1-8B-Instruct/alpaca
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

# NCU metrics to export (same as run_nwor_microbench.py)
METRICS = [
    "dram__bytes_write.sum",
    "dram__bytes_read.sum",
    "lts__t_sectors_op_write.sum",
    "lts__t_sectors_op_read.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
]


def find_ncu_reports(folder: Path) -> List[Path]:
    """Find all .ncu-rep files in seed folders."""
    ncu_reports = []

    # Look for seed_* folders
    seed_folders = sorted([d for d in folder.iterdir() if d.is_dir() and d.name.startswith('seed_')])

    if not seed_folders:
        print(f"⚠ WARNING: No seed folders found in {folder}")
        return ncu_reports

    print(f"Found {len(seed_folders)} seed folders")

    for seed_folder in seed_folders:
        # Find all .ncu-rep files (should end with .ncu.ncu-rep)
        reports = list(seed_folder.glob("*.ncu.ncu-rep"))
        ncu_reports.extend(reports)
        print(f"  {seed_folder.name}: {len(reports)} NCU reports")

    return ncu_reports


def export_ncu_csv(ncu_report: Path, ncu_cmd: str = "ncu") -> bool:
    """
    Export NCU report to CSV.

    Args:
        ncu_report: Path to .ncu-rep file
        ncu_cmd: NCU command (default: "ncu")

    Returns:
        True if successful, False otherwise
    """
    # Generate CSV filename: file.ncu.ncu-rep -> file.csv
    csv_path = ncu_report.with_suffix('').with_suffix('.csv')

    # Build NCU export command
    cmd = [
        ncu_cmd,
        "--import", str(ncu_report),
        "--csv",
        "--page", "raw",
        "--metrics", ",".join(METRICS),
    ]

    try:
        print(f"  Exporting: {ncu_report.name} -> {csv_path.name}")

        # Run NCU and redirect output to CSV
        with open(csv_path, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )

        print(f"  ✓ Created: {csv_path.name}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"  ✗ ERROR exporting {ncu_report.name}: {e}")
        print(f"    stderr: {e.stderr}")

        # Clean up partial CSV file
        if csv_path.exists():
            csv_path.unlink()

        return False
    except FileNotFoundError:
        print(f"  ✗ ERROR: NCU command not found. Is Nsight Compute installed?")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate CSV files from existing NCU .ncu-rep files"
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing seed_* subfolders with NCU reports"
    )
    parser.add_argument(
        "--ncu-cmd",
        type=str,
        default="ncu",
        help="Path to NCU command (default: ncu)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually doing it"
    )

    args = parser.parse_args()

    folder = args.folder.resolve()

    if not folder.exists():
        print(f"ERROR: Folder not found: {folder}")
        sys.exit(1)

    if not folder.is_dir():
        print(f"ERROR: Not a directory: {folder}")
        sys.exit(1)

    print(f"Analyzing folder: {folder}")
    print()

    # Find all NCU reports
    ncu_reports = find_ncu_reports(folder)

    if not ncu_reports:
        print("\n⚠ No NCU reports found!")
        sys.exit(0)

    print(f"\nFound {len(ncu_reports)} total NCU reports")
    print()

    if args.dry_run:
        print("DRY RUN - would export the following:")
        for report in ncu_reports:
            csv_path = report.with_suffix('').with_suffix('.csv')
            status = "exists" if csv_path.exists() else "would create"
            print(f"  {report.relative_to(folder)} -> {csv_path.name} ({status})")
        sys.exit(0)

    # Export each report to CSV
    print("Exporting CSVs...")
    success_count = 0
    fail_count = 0

    for report in ncu_reports:
        if export_ncu_csv(report, args.ncu_cmd):
            success_count += 1
        else:
            fail_count += 1

    print()
    print(f"Results:")
    print(f"  ✓ Created: {success_count}")
    print(f"  ✗ Failed: {fail_count}")
    print(f"  Total: {len(ncu_reports)}")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
