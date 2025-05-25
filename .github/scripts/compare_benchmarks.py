#!/usr/bin/env python3
"""
Script to compare benchmark results between runs.
This can be integrated into the CI workflow to track performance changes over time.
"""

import json
import sys
from pathlib import Path
import argparse
from typing import Dict, Any, List, Tuple


def load_benchmark(file_path: Path) -> Dict[str, Any]:
    """Load benchmark results from a Criterion JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def compare_benchmarks(old_file: Path, new_file: Path) -> List[Tuple[str, float]]:
    """Compare benchmark results between two runs."""
    old_data = load_benchmark(old_file)
    new_data = load_benchmark(new_file)
    
    results = []
    
    # Extract benchmark name and mean execution time
    old_mean = old_data.get('mean', {}).get('point_estimate', 0)
    new_mean = new_data.get('mean', {}).get('point_estimate', 0)
    
    # Calculate percentage change
    if old_mean > 0:
        percent_change = ((new_mean - old_mean) / old_mean) * 100
    else:
        percent_change = 0
    
    benchmark_name = str(old_file).split('/')[-2]
    results.append((benchmark_name, percent_change))
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare Criterion benchmark results')
    parser.add_argument('--old-dir', type=str, required=True, help='Directory with old benchmark results')
    parser.add_argument('--new-dir', type=str, required=True, help='Directory with new benchmark results')
    
    args = parser.parse_args()
    
    old_dir = Path(args.old_dir)
    new_dir = Path(args.new_dir)
    
    if not old_dir.exists() or not new_dir.exists():
        print("Error: One or both of the specified directories don't exist")
        sys.exit(1)
    
    # Find all benchmark result files
    old_files = list(old_dir.glob('**/new/estimates.json'))
    
    all_results = []
    for old_file in old_files:
        # Construct path to corresponding new benchmark file
        rel_path = old_file.relative_to(old_dir)
        new_file = new_dir / rel_path
        
        if new_file.exists():
            results = compare_benchmarks(old_file, new_file)
            all_results.extend(results)
    
    # Print results in a table format
    print("| Benchmark | Change (%) |")
    print("|-----------|------------|")
    for name, change in all_results:
        print(f"| {name} | {change:.2f}% |")
    
    # Return non-zero exit code if any benchmark has regressed by more than 5%
    if any(change > 5.0 for _, change in all_results):
        print("\n⚠️ Performance regression detected! Some benchmarks have slowed down by more than 5%.")
        sys.exit(1)
    
    print("\n✅ No significant performance regressions detected.")


if __name__ == "__main__":
    main()
