#!/usr/bin/env python3
"""
Compare memory usage across all model variants.
Usage: python scripts/compare_memory.py
"""

from pathlib import Path


def parse_memory_file(path: Path) -> dict:
    """Parse memory_usage.txt file."""
    if not path.exists():
        return None
    
    stats = {}
    with open(path, 'r') as f:
        for line in f:
            if "Peak GPU Memory:" in line:
                stats['peak_gpu_mb'] = float(line.split(':')[1].strip().split()[0])
            elif "Peak CPU Memory:" in line:
                stats['peak_cpu_mb'] = float(line.split(':')[1].strip().split()[0])
            elif "GPU Device:" in line:
                stats['gpu_device'] = line.split(':')[1].strip()
    return stats


def main():
    variants = {
        'Markov (ε)': 'experiments/markov_ddim/logs/memory_usage.txt',
        'Non-Markov (ε, Trans)': 'experiments/nonmarkov_ddim/logs/memory_usage.txt',
        'Non-Markov (x₀, Trans)': 'experiments/nonmarkov_x0/logs/memory_usage.txt',
        'DART (Trans, CFG)': 'experiments/dart/logs/memory_usage.txt',
        'DART (Trans, no CFG)': 'experiments/dart_no_cfg/logs/memory_usage.txt',
        'Non-Markov (ε, Sig)': 'experiments/nonmarkov_signature/logs/memory_usage.txt',
        'DART (Sig, CFG)': 'experiments/dart_signature/logs/memory_usage.txt',
    }

    print("Memory Usage Comparison")
    print("=" * 80)
    print(f"{'Variant':<25} {'Peak GPU (MB)':<15} {'Peak CPU (MB)':<15} {'Status'}")
    print("-" * 80)

    results = []
    for name, path in variants.items():
        mem_path = Path(path)
        stats = parse_memory_file(mem_path)
        
        if stats is None:
            print(f"{name:<25} {'N/A':<15} {'N/A':<15} Not trained yet")
        else:
            peak_gpu = stats.get('peak_gpu_mb', 0)
            peak_cpu = stats.get('peak_cpu_mb', 0)
            print(f"{name:<25} {peak_gpu:<15.1f} {peak_cpu:<15.1f} ✓")
            results.append((name, peak_gpu, peak_cpu))

    if results:
        print("-" * 80)
        
        # Find baseline (Markov)
        baseline = next((r for r in results if 'Markov (ε)' in r[0]), None)
        
        if baseline:
            print("\nMemory Overhead vs Markov Baseline:")
            print("-" * 80)
            baseline_gpu = baseline[1]
            baseline_cpu = baseline[2]
            
            for name, gpu, cpu in results:
                if name == baseline[0]:
                    continue
                gpu_overhead = ((gpu - baseline_gpu) / baseline_gpu * 100) if baseline_gpu > 0 else 0
                cpu_overhead = ((cpu - baseline_cpu) / baseline_cpu * 100) if baseline_cpu > 0 else 0
                print(f"{name:<25} GPU: +{gpu_overhead:>6.1f}%  ({gpu - baseline_gpu:>6.1f} MB)")
                print(f"{'':<25} CPU: +{cpu_overhead:>6.1f}%  ({cpu - baseline_cpu:>6.1f} MB)")
        
        # Summary statistics
        print("\n" + "=" * 80)
        print("Summary:")
        max_gpu = max(r[1] for r in results)
        min_gpu = min(r[1] for r in results if r[1] > 0)
        print(f"  GPU Range: {min_gpu:.1f} - {max_gpu:.1f} MB")
        print(f"  GPU Variation: {(max_gpu - min_gpu):.1f} MB ({(max_gpu - min_gpu) / min_gpu * 100:.1f}%)")
        
        # Find most memory-efficient
        sorted_gpu = sorted(results, key=lambda x: x[1] if x[1] > 0 else float('inf'))
        print(f"\n  Most GPU-efficient: {sorted_gpu[0][0]} ({sorted_gpu[0][1]:.1f} MB)")
        print(f"  Most GPU-intensive: {sorted_gpu[-1][0]} ({sorted_gpu[-1][1]:.1f} MB)")
    else:
        print("\nNo memory data found. Train some models first!")
        print("Run: bash scripts/run_all_configs.sh")


if __name__ == "__main__":
    main()

