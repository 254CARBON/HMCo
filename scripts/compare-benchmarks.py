#!/usr/bin/env python3
"""
Compare benchmark results with baseline to detect performance regressions.
"""
import json
import sys
import os
from pathlib import Path


def load_benchmark_results(filepath: str) -> dict:
    """Load benchmark results from JSON file."""
    if not os.path.exists(filepath):
        print(f"Benchmark file not found: {filepath}")
        return None
    
    with open(filepath, 'r') as f:
        return json.load(f)


def compare_benchmarks(current_results: dict, baseline_results: dict, threshold: float = 0.10):
    """
    Compare current benchmark results with baseline.
    
    Args:
        current_results: Current benchmark results
        baseline_results: Baseline benchmark results
        threshold: Regression threshold (default 10%)
    
    Returns:
        bool: True if no regressions detected, False otherwise
    """
    if not current_results or not baseline_results:
        print("⚠️  Skipping benchmark comparison - missing results")
        return True
    
    regressions = []
    improvements = []
    
    # Compare benchmark statistics
    for benchmark in current_results.get('benchmarks', []):
        name = benchmark.get('name', 'unknown')
        current_mean = benchmark.get('stats', {}).get('mean', 0)
        
        # Find matching baseline benchmark
        baseline_benchmark = next(
            (b for b in baseline_results.get('benchmarks', []) if b.get('name') == name),
            None
        )
        
        if not baseline_benchmark:
            print(f"ℹ️  New benchmark: {name} (mean: {current_mean:.4f}s)")
            continue
        
        baseline_mean = baseline_benchmark.get('stats', {}).get('mean', 0)
        
        if baseline_mean == 0:
            continue
        
        # Calculate percentage change
        change = ((current_mean - baseline_mean) / baseline_mean) * 100
        
        if change > (threshold * 100):
            regressions.append({
                'name': name,
                'baseline': baseline_mean,
                'current': current_mean,
                'change': change
            })
        elif change < -(threshold * 100):
            improvements.append({
                'name': name,
                'baseline': baseline_mean,
                'current': current_mean,
                'change': change
            })
    
    # Report results
    print("=" * 60)
    print("Benchmark Comparison Results")
    print("=" * 60)
    
    if improvements:
        print(f"\n✅ Improvements detected ({len(improvements)}):")
        for imp in improvements:
            print(f"  • {imp['name']}: {imp['baseline']:.4f}s → {imp['current']:.4f}s ({imp['change']:.1f}%)")
    
    if regressions:
        print(f"\n❌ Regressions detected ({len(regressions)}):")
        for reg in regressions:
            print(f"  • {reg['name']}: {reg['baseline']:.4f}s → {reg['current']:.4f}s (+{reg['change']:.1f}%)")
        print(f"\n⚠️  Performance regression threshold exceeded: {threshold * 100}%")
        return False
    
    if not improvements and not regressions:
        print("\n✅ No significant performance changes detected")
    
    return True


def main():
    """Main entry point."""
    current_file = "benchmark-results.json"
    baseline_file = "benchmark-baseline.json"
    
    # Load results
    current_results = load_benchmark_results(current_file)
    baseline_results = load_benchmark_results(baseline_file)
    
    # If no baseline exists, save current as baseline for future comparisons
    if not baseline_results and current_results:
        print(f"ℹ️  No baseline found. Current results will be saved as baseline.")
        with open(baseline_file, 'w') as f:
            json.dump(current_results, f, indent=2)
        return 0
    
    # Compare benchmarks
    success = compare_benchmarks(current_results, baseline_results)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
