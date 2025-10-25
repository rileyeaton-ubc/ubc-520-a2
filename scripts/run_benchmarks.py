"""Script to run all benchmarks."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_structures import BloomFilter, CountMinSketch, LogLog
from benchmarks.benchmark import benchmark_operation


def main():
    """Run benchmarks for all data structures."""
    print("Running benchmarks for probabilistic data structures...")

    # TODO: Implement benchmark suite
    print("\nBloom Filter Benchmarks:")
    print("  - Not yet implemented")

    print("\nCount-Min Sketch Benchmarks:")
    print("  - Not yet implemented")

    print("\nLogLog Benchmarks:")
    print("  - Not yet implemented")


if __name__ == "__main__":
    main()
