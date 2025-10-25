"""Benchmark suite for probabilistic data structures.

This script benchmarks Bloom Filter, Count-Min Sketch, and LogLog implementations
to demonstrate their key characteristics, trade-offs, and performance.
"""

import sys
from pathlib import Path

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import math

from src.data_structures.bloom_filter import BloomFilter
from src.data_structures.count_min_sketch import CountMinSketch
from src.data_structures.loglog import LogLog


def memory_usage_bytes(obj: Any) -> int:
    """Calculate approximate memory usage in bytes.

    Args:
        obj: Data structure instance

    Returns:
        Approximate memory usage in bytes
    """
    if isinstance(obj, BloomFilter):
        # Bit array + metadata
        return obj.size // 8 + 100  # 100 bytes for overhead
    elif isinstance(obj, CountMinSketch):
        # 2D array of integers (4 bytes each)
        return obj.width * obj.depth * 4 + 100
    elif isinstance(obj, LogLog):
        # Register array (4 bytes per register)
        return obj.m * 4 + 100
    return 0


def benchmark_bloom_filter(
    dataset_path: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """Benchmark Bloom Filter with varying parameters.

    Tests:
    1. False positive rate vs. load factor
    2. Memory efficiency vs. accuracy
    3. Insertion and query performance
    4. Parameter optimization

    Args:
        dataset_path: Path to URL stream file
        output_dir: Output directory for results

    Returns:
        Dictionary containing benchmark results
    """
    print("\n" + "=" * 70)
    print("BLOOM FILTER BENCHMARKS")
    print("=" * 70)

    results = {
        "name": "Bloom Filter",
        "experiments": []
    }

    # Load dataset
    print(f"\nLoading dataset from {dataset_path}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f]

    n = len(urls)
    unique_urls = list(set(urls))
    print(f"Loaded {n:,} events ({len(unique_urls):,} unique URLs)")

    # Experiment 1: False Positive Rate vs Load Factor
    print("\n[1] Testing false positive rate vs load factor...")

    fpr_results = []
    test_configs = [
        (10000, 3, 0.001),   # Optimal for low FPR
        (10000, 5, 0.01),    # Balanced
        (5000, 3, 0.1),      # High FPR (small filter)
        (20000, 7, 0.0001),  # Very low FPR
    ]

    for size, k, target_fpr in test_configs:
        bf = BloomFilter(size=size, num_hashes=k)

        # Add first half of unique URLs
        train_size = len(unique_urls) // 2
        train_set = unique_urls[:train_size]
        test_set = unique_urls[train_size:]

        start_time = time.time()
        for url in train_set:
            bf.add(url)
        insert_time = time.time() - start_time

        # Test false positives
        start_time = time.time()
        false_positives = sum(1 for url in test_set if bf.contains(url))
        query_time = time.time() - start_time

        actual_fpr = false_positives / len(test_set) if test_set else 0
        expected_fpr = bf.expected_false_positive_rate()
        load_factor = bf.load_factor()
        memory_bytes = memory_usage_bytes(bf)

        config_result = {
            "size": size,
            "num_hashes": k,
            "target_fpr": target_fpr,
            "items_added": train_size,
            "load_factor": load_factor,
            "actual_fpr": actual_fpr,
            "expected_fpr": expected_fpr,
            "memory_bytes": memory_bytes,
            "memory_kb": memory_bytes / 1024,
            "insert_time_ms": insert_time * 1000,
            "query_time_ms": query_time * 1000,
            "insert_throughput": train_size / insert_time if insert_time > 0 else 0,
            "query_throughput": len(test_set) / query_time if query_time > 0 else 0
        }

        fpr_results.append(config_result)

        print(f"  Size={size:,}, k={k}: FPR={actual_fpr:.4f} (expected={expected_fpr:.4f}), "
              f"Load={load_factor:.3f}, Mem={memory_bytes/1024:.1f}KB")

    results["experiments"].append({
        "name": "False Positive Rate vs Configuration",
        "description": "Tests how FPR varies with filter size and hash count",
        "data": fpr_results
    })

    # Experiment 2: Scalability Test
    print("\n[2] Testing scalability with different dataset sizes...")

    scalability_results = []

    # Use optimal parameters for 1% FPR
    target_fpr = 0.01
    # Generate test sizes based on available unique URLs
    max_size = len(unique_urls)
    test_sizes = [max(50, max_size // 10), max(100, max_size // 4), max(200, max_size // 2),
                  max(300, int(max_size * 0.75)), max_size]
    test_sizes = sorted(list(set(test_sizes)))  # Remove duplicates and sort

    for test_size in test_sizes:
        if test_size > len(unique_urls):
            continue

        optimal_m = BloomFilter.optimal_size(test_size, target_fpr)
        optimal_k = BloomFilter.optimal_num_hashes(optimal_m, test_size)

        bf = BloomFilter(size=optimal_m, num_hashes=optimal_k)

        train_set = unique_urls[:test_size]

        start_time = time.time()
        for url in train_set:
            bf.add(url)
        insert_time = time.time() - start_time

        # Query performance
        query_set = train_set[:1000]  # Query first 1000
        start_time = time.time()
        hits = sum(1 for url in query_set if bf.contains(url))
        query_time = time.time() - start_time

        memory_bytes = memory_usage_bytes(bf)

        scalability_results.append({
            "dataset_size": test_size,
            "filter_size": optimal_m,
            "num_hashes": optimal_k,
            "memory_bytes": memory_bytes,
            "memory_kb": memory_bytes / 1024,
            "insert_time_ms": insert_time * 1000,
            "query_time_ms": query_time * 1000,
            "insert_throughput": test_size / insert_time if insert_time > 0 else 0,
            "query_throughput": len(query_set) / query_time if query_time > 0 else 0,
            "bits_per_element": optimal_m / test_size
        })

        throughput_str = f"{test_size/insert_time:.0f}" if insert_time > 0 else ">1M"
        print(f"  n={test_size:,}: m={optimal_m:,}, k={optimal_k}, "
              f"Mem={memory_bytes/1024:.1f}KB, {throughput_str} inserts/sec")

    results["experiments"].append({
        "name": "Scalability Analysis",
        "description": "Performance with increasing dataset sizes",
        "data": scalability_results
    })

    # Save results
    output_file = output_dir / "bloom_filter_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Bloom Filter results saved to {output_file}")

    return results


def benchmark_count_min_sketch(
    dataset_path: Path,
    ground_truth_path: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """Benchmark Count-Min Sketch with varying parameters.

    Tests:
    1. Frequency estimation accuracy
    2. Error bounds (ε-δ guarantees)
    3. Performance vs. sketch dimensions
    4. Heavy hitter detection

    Args:
        dataset_path: Path to URL stream file
        ground_truth_path: Path to ground truth JSON
        output_dir: Output directory for results

    Returns:
        Dictionary containing benchmark results
    """
    print("\n" + "=" * 70)
    print("COUNT-MIN SKETCH BENCHMARKS")
    print("=" * 70)

    results = {
        "name": "Count-Min Sketch",
        "experiments": []
    }

    # Load dataset and ground truth
    print(f"\nLoading dataset from {dataset_path}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f]

    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    print(f"Loaded {len(urls):,} events")

    # Build true frequency map
    true_frequencies = {}
    for url in urls:
        true_frequencies[url] = true_frequencies.get(url, 0) + 1

    # Experiment 1: Error Analysis with Different Dimensions
    print("\n[1] Testing estimation accuracy with different dimensions...")

    error_results = []

    configs = [
        (100, 3, 0.1, 0.05),      # Small sketch
        (500, 5, 0.02, 0.01),     # Medium sketch
        (1000, 7, 0.01, 0.005),   # Large sketch
        (2000, 10, 0.005, 0.001)  # Very large sketch
    ]

    for width, depth, epsilon, delta in configs:
        cms = CountMinSketch(width=width, depth=depth)

        # Insert all URLs
        start_time = time.time()
        for url in urls:
            cms.update(url)
        insert_time = time.time() - start_time

        # Test estimation accuracy on top 100 URLs
        top_urls = sorted(true_frequencies.items(), key=lambda x: x[1], reverse=True)[:100]

        errors = []
        relative_errors = []

        start_time = time.time()
        for url, true_freq in top_urls:
            estimated = cms.estimate(url)
            error = estimated - true_freq
            relative_error = error / true_freq if true_freq > 0 else 0
            errors.append(error)
            relative_errors.append(relative_error)
        query_time = time.time() - start_time

        # Calculate statistics
        avg_error = sum(errors) / len(errors)
        max_error = max(errors)
        avg_relative_error = sum(relative_errors) / len(relative_errors)

        # Check ε guarantee
        expected_error_bound = epsilon * len(urls)
        violations = sum(1 for e in errors if e > expected_error_bound)

        memory_bytes = memory_usage_bytes(cms)

        config_result = {
            "width": width,
            "depth": depth,
            "epsilon": epsilon,
            "delta": delta,
            "memory_bytes": memory_bytes,
            "memory_kb": memory_bytes / 1024,
            "insert_time_ms": insert_time * 1000,
            "query_time_ms": query_time * 1000,
            "insert_throughput": len(urls) / insert_time if insert_time > 0 else 0,
            "query_throughput": len(top_urls) / query_time if query_time > 0 else 0,
            "avg_error": avg_error,
            "max_error": max_error,
            "avg_relative_error": avg_relative_error,
            "expected_error_bound": expected_error_bound,
            "violations": violations,
            "violation_rate": violations / len(top_urls)
        }

        error_results.append(config_result)

        print(f"  w={width}, d={depth}: Avg Error={avg_error:.1f}, "
              f"Max Error={max_error:.0f}, Violations={violations}/100, "
              f"Mem={memory_bytes/1024:.1f}KB")

    results["experiments"].append({
        "name": "Estimation Accuracy vs Dimensions",
        "description": "How sketch dimensions affect frequency estimation accuracy",
        "data": error_results
    })

    # Experiment 2: Heavy Hitters Detection
    print("\n[2] Testing heavy hitter detection...")

    heavy_hitter_results = []

    # Use medium-sized sketch
    cms = CountMinSketch(width=500, depth=5)
    for url in urls:
        cms.update(url)

    # Define heavy hitter thresholds
    thresholds = [0.01, 0.005, 0.001, 0.0005]  # 1%, 0.5%, 0.1%, 0.05%

    for threshold in thresholds:
        threshold_count = int(threshold * len(urls))

        # Find true heavy hitters
        true_heavy = [url for url, freq in true_frequencies.items() if freq >= threshold_count]

        # Estimate candidates (check all unique URLs)
        candidates = list(true_frequencies.keys())
        estimated_heavy = [url for url in candidates if cms.estimate(url) >= threshold_count]

        # Calculate precision and recall
        true_positives = len(set(true_heavy) & set(estimated_heavy))
        precision = true_positives / len(estimated_heavy) if estimated_heavy else 0
        recall = true_positives / len(true_heavy) if true_heavy else 0

        heavy_hitter_results.append({
            "threshold": threshold,
            "threshold_count": threshold_count,
            "true_heavy_count": len(true_heavy),
            "estimated_heavy_count": len(estimated_heavy),
            "true_positives": true_positives,
            "precision": precision,
            "recall": recall
        })

        print(f"  Threshold {threshold*100:.2f}%: Precision={precision:.3f}, "
              f"Recall={recall:.3f}, True={len(true_heavy)}, Est={len(estimated_heavy)}")

    results["experiments"].append({
        "name": "Heavy Hitter Detection",
        "description": "Precision and recall for finding frequent items",
        "data": heavy_hitter_results
    })

    # Experiment 3: Scalability
    print("\n[3] Testing scalability...")

    scalability_results = []

    # Fixed epsilon/delta, vary dataset size
    epsilon = 0.01
    delta = 0.01
    width = CountMinSketch.required_width(epsilon)
    depth = CountMinSketch.required_depth(delta)

    test_sizes = [10000, 50000, 100000, 500000, len(urls)]

    for size in test_sizes:
        cms = CountMinSketch(width=width, depth=depth)

        subset = urls[:size]

        start_time = time.time()
        for url in subset:
            cms.update(url)
        insert_time = time.time() - start_time

        memory_bytes = memory_usage_bytes(cms)

        scalability_results.append({
            "dataset_size": size,
            "width": width,
            "depth": depth,
            "memory_bytes": memory_bytes,
            "memory_kb": memory_bytes / 1024,
            "insert_time_ms": insert_time * 1000,
            "insert_throughput": size / insert_time if insert_time > 0 else 0,
            "bytes_per_element": memory_bytes / size
        })

        print(f"  n={size:,}: Mem={memory_bytes/1024:.1f}KB, "
              f"{size/insert_time:.0f} updates/sec")

    results["experiments"].append({
        "name": "Scalability Analysis",
        "description": "Performance with increasing dataset sizes",
        "data": scalability_results
    })

    # Save results
    output_file = output_dir / "count_min_sketch_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Count-Min Sketch results saved to {output_file}")

    return results


def benchmark_loglog(
    dataset_path: Path,
    ground_truth_path: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """Benchmark LogLog with varying parameters.

    Tests:
    1. Cardinality estimation accuracy
    2. Precision vs. memory trade-off
    3. Performance across different cardinalities
    4. Merge operations

    Args:
        dataset_path: Path to user stream file
        ground_truth_path: Path to ground truth JSON
        output_dir: Output directory for results

    Returns:
        Dictionary containing benchmark results
    """
    print("\n" + "=" * 70)
    print("LOGLOG BENCHMARKS")
    print("=" * 70)

    results = {
        "name": "LogLog",
        "experiments": []
    }

    # Load dataset and ground truth
    print(f"\nLoading dataset from {dataset_path}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        users = [line.strip() for line in f]

    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    true_cardinality = ground_truth['unique_users_actual']

    print(f"Loaded {len(users):,} events (true cardinality: {true_cardinality:,})")

    # Experiment 1: Accuracy vs Precision
    print("\n[1] Testing accuracy with different precision values...")

    accuracy_results = []

    precisions = [8, 10, 12, 14, 16]

    for precision in precisions:
        ll = LogLog(precision=precision)

        start_time = time.time()
        for user in users:
            ll.add(user)
        insert_time = time.time() - start_time

        estimated = ll.cardinality()
        error = estimated - true_cardinality
        relative_error = abs(error) / true_cardinality
        theoretical_std_error = ll.standard_error()

        memory_bytes = memory_usage_bytes(ll)

        accuracy_results.append({
            "precision": precision,
            "num_registers": ll.m,
            "true_cardinality": true_cardinality,
            "estimated_cardinality": estimated,
            "error": error,
            "relative_error": relative_error,
            "theoretical_std_error": theoretical_std_error,
            "within_1_std_error": relative_error <= theoretical_std_error,
            "within_3_std_error": relative_error <= 3 * theoretical_std_error,
            "memory_bytes": memory_bytes,
            "memory_kb": memory_bytes / 1024,
            "insert_time_ms": insert_time * 1000,
            "insert_throughput": len(users) / insert_time if insert_time > 0 else 0,
            "bytes_per_element": memory_bytes / true_cardinality
        })

        print(f"  Precision {precision} (m={ll.m}): Est={estimated:,}, "
              f"Error={relative_error:.3f}, StdErr={theoretical_std_error:.3f}, "
              f"Mem={memory_bytes/1024:.1f}KB")

    results["experiments"].append({
        "name": "Accuracy vs Precision",
        "description": "How precision parameter affects estimation accuracy",
        "data": accuracy_results
    })

    # Experiment 2: Performance Across Different Cardinalities
    print("\n[2] Testing performance across different cardinalities...")

    cardinality_results = []

    # Use medium precision
    precision = 12

    # Test with different dataset sizes
    test_sizes = [10000, 50000, 100000, 500000, len(users)]

    for size in test_sizes:
        ll = LogLog(precision=precision)

        subset = users[:size]
        unique_subset = set(subset)
        true_card = len(unique_subset)

        start_time = time.time()
        for user in subset:
            ll.add(user)
        insert_time = time.time() - start_time

        estimated = ll.cardinality()
        error = estimated - true_card
        relative_error = abs(error) / true_card if true_card > 0 else 0

        cardinality_results.append({
            "dataset_size": size,
            "true_cardinality": true_card,
            "estimated_cardinality": estimated,
            "error": error,
            "relative_error": relative_error,
            "insert_time_ms": insert_time * 1000,
            "insert_throughput": size / insert_time if insert_time > 0 else 0
        })

        print(f"  n={size:,}, card={true_card:,}: Est={estimated:,}, "
              f"Error={relative_error:.3f}")

    results["experiments"].append({
        "name": "Performance Across Cardinalities",
        "description": "Accuracy and speed with different dataset sizes",
        "data": cardinality_results
    })

    # Experiment 3: Merge Performance
    print("\n[3] Testing merge operations...")

    merge_results = []

    precision = 12

    # Split dataset into chunks and merge
    chunk_sizes = [2, 4, 8, 16]

    for num_chunks in chunk_sizes:
        chunk_size = len(users) // num_chunks

        loglogs = []
        start_time = time.time()

        for i in range(num_chunks):
            ll = LogLog(precision=precision)
            chunk = users[i * chunk_size:(i + 1) * chunk_size]
            for user in chunk:
                ll.add(user)
            loglogs.append(ll)

        # Merge all
        merged = loglogs[0]
        for ll in loglogs[1:]:
            merged = merged.merge(ll)

        merge_time = time.time() - start_time

        estimated = merged.cardinality()
        error = estimated - true_cardinality
        relative_error = abs(error) / true_cardinality

        merge_results.append({
            "num_chunks": num_chunks,
            "chunk_size": chunk_size,
            "estimated_cardinality": estimated,
            "true_cardinality": true_cardinality,
            "error": error,
            "relative_error": relative_error,
            "total_time_ms": merge_time * 1000
        })

        print(f"  {num_chunks} chunks: Est={estimated:,}, Error={relative_error:.3f}")

    results["experiments"].append({
        "name": "Merge Operations",
        "description": "Accuracy when merging multiple LogLog sketches",
        "data": merge_results
    })

    # Save results
    output_file = output_dir / "loglog_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] LogLog results saved to {output_file}")

    return results


def compare_structures(
    all_results: List[Dict[str, Any]],
    output_dir: Path
) -> None:
    """Generate comparison metrics across all three structures.

    Args:
        all_results: List of result dictionaries from each benchmark
        output_dir: Output directory for comparison results
    """
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS")
    print("=" * 70)

    comparison = {
        "structures": [],
        "summary": {}
    }

    for result in all_results:
        name = result["name"]

        # Extract key metrics from experiments
        if name == "Bloom Filter":
            # Get scalability data for largest size
            scalability = result["experiments"][1]["data"][-1]

            comparison["structures"].append({
                "name": name,
                "primary_use": "Set membership testing",
                "guarantees": "No false negatives, tunable false positive rate",
                "memory_kb": scalability["memory_kb"],
                "insert_throughput": scalability["insert_throughput"],
                "query_throughput": scalability["query_throughput"],
                "bits_per_element": scalability["bits_per_element"],
                "key_strength": "Space-efficient set membership with no false negatives",
                "key_limitation": "Cannot remove elements, only membership queries"
            })

        elif name == "Count-Min Sketch":
            # Get scalability data for largest size
            scalability = result["experiments"][2]["data"][-1]

            comparison["structures"].append({
                "name": name,
                "primary_use": "Frequency estimation",
                "guarantees": "Never underestimates, ε-δ error bounds",
                "memory_kb": scalability["memory_kb"],
                "insert_throughput": scalability["insert_throughput"],
                "query_throughput": "N/A",
                "bytes_per_element": scalability["bytes_per_element"],
                "key_strength": "Accurate frequency counts with theoretical error bounds",
                "key_limitation": "Can overestimate frequencies due to collisions"
            })

        elif name == "LogLog":
            # Get accuracy data for medium precision
            accuracy = result["experiments"][0]["data"][2]  # precision=12

            comparison["structures"].append({
                "name": name,
                "primary_use": "Cardinality estimation",
                "guarantees": "Probabilistic with standard error bounds",
                "memory_kb": accuracy["memory_kb"],
                "insert_throughput": accuracy["insert_throughput"],
                "query_throughput": "O(1) - constant time",
                "bytes_per_element": accuracy["bytes_per_element"],
                "key_strength": "Estimates cardinality in log-log space",
                "key_limitation": "Higher variance compared to HyperLogLog"
            })

    # Generate summary
    comparison["summary"] = {
        "memory_efficiency_ranking": sorted(
            comparison["structures"],
            key=lambda x: x.get("bytes_per_element", float('inf'))
        ),
        "throughput_ranking": sorted(
            comparison["structures"],
            key=lambda x: x.get("insert_throughput", 0),
            reverse=True
        ),
        "use_cases": {
            "Bloom Filter": "Cache filtering, spell checking, malicious URL detection",
            "Count-Min Sketch": "Network traffic analysis, trending topics, query frequency",
            "LogLog": "Unique visitor counting, database query optimization, large-scale analytics"
        }
    }

    # Save comparison
    output_file = output_dir / "comparison.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)

    print(f"\n[OK] Comparison results saved to {output_file}")

    # Print summary
    print("\nMemory Efficiency (bytes per element):")
    for i, struct in enumerate(comparison["summary"]["memory_efficiency_ranking"], 1):
        bpe = struct.get("bytes_per_element", struct.get("bits_per_element", 0) / 8)
        print(f"  {i}. {struct['name']}: {bpe:.4f} bytes/element")

    print("\nInsertion Throughput:")
    for i, struct in enumerate(comparison["summary"]["throughput_ranking"], 1):
        throughput = struct.get("insert_throughput", 0)
        print(f"  {i}. {struct['name']}: {throughput:,.0f} ops/sec")


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Benchmark probabilistic data structures"
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/medium',
        help='Directory containing dataset files (default: data/medium)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/benchmarks',
        help='Output directory for benchmark results (default: data/benchmarks)'
    )

    parser.add_argument(
        '--structures',
        nargs='+',
        choices=['bloom', 'cms', 'loglog', 'all'],
        default=['all'],
        help='Which structures to benchmark (default: all)'
    )

    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    url_stream = data_dir / "url_stream.txt"
    user_stream = data_dir / "user_stream.txt"
    ground_truth = data_dir / "ground_truth.json"

    # Verify files exist
    for path in [url_stream, user_stream, ground_truth]:
        if not path.exists():
            print(f"ERROR: Required file not found: {path}")
            print(f"Please run generate_dataset.py first to create the dataset.")
            return

    structures = args.structures
    if 'all' in structures:
        structures = ['bloom', 'cms', 'loglog']

    all_results = []

    # Run benchmarks
    if 'bloom' in structures:
        bloom_results = benchmark_bloom_filter(url_stream, output_dir)
        all_results.append(bloom_results)

    if 'cms' in structures:
        cms_results = benchmark_count_min_sketch(url_stream, ground_truth, output_dir)
        all_results.append(cms_results)

    if 'loglog' in structures:
        loglog_results = benchmark_loglog(user_stream, ground_truth, output_dir)
        all_results.append(loglog_results)

    # Generate comparison if all structures were benchmarked
    if len(all_results) == 3:
        compare_structures(all_results, output_dir)

    print("\n" + "=" * 70)
    print("BENCHMARKING COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Review JSON result files for detailed metrics")
    print("  2. Use the Jupyter notebook in notebooks/ to visualize results")
    print("  3. Analyze trade-offs between accuracy, memory, and speed")


if __name__ == "__main__":
    main()
