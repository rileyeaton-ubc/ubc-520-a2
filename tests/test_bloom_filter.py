"""Tests for Bloom filter implementation."""

import pytest
import json
from pathlib import Path
from src.data_structures.bloom_filter import BloomFilter


class TestBloomFilter:
    """Test cases for BloomFilter."""

    def test_initialization(self):
        """Test Bloom filter initialization."""
        bf = BloomFilter(size=1000, num_hashes=3)
        assert bf is not None
        assert bf.size == 1000
        assert bf.num_hashes == 3
        assert len(bf) == 0
        assert bf.bits_set() == 0

    def test_invalid_initialization(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            BloomFilter(size=0, num_hashes=3)

        with pytest.raises(ValueError):
            BloomFilter(size=1000, num_hashes=0)

        with pytest.raises(ValueError):
            BloomFilter(size=-100, num_hashes=3)

    def test_add_and_contains(self):
        """Test adding items and checking membership."""
        bf = BloomFilter(size=1000, num_hashes=3)

        # Add some items
        items = ["apple", "banana", "cherry", "date", "elderberry"]
        for item in items:
            bf.add(item)

        # All added items should be found (no false negatives)
        for item in items:
            assert bf.contains(item), f"{item} should be in the filter"

        assert len(bf) == 5

    def test_does_not_contain(self):
        """Test that items not added are correctly identified (mostly)."""
        bf = BloomFilter(size=10000, num_hashes=5)

        # Add some items
        added_items = [f"item_{i}" for i in range(100)]
        for item in added_items:
            bf.add(item)

        # Check items that were not added
        not_added = [f"different_{i}" for i in range(100)]
        false_positives = sum(1 for item in not_added if bf.contains(item))

        # Most items not added should not be in the filter
        # With good parameters, false positive rate should be low
        assert false_positives < len(not_added), "All non-added items returned True"

    def test_no_false_negatives(self):
        """Test that there are absolutely no false negatives."""
        bf = BloomFilter(size=5000, num_hashes=4)

        # Add many items
        items = [f"test_item_{i}" for i in range(500)]
        for item in items:
            bf.add(item)

        # Check every added item - all must be found
        for item in items:
            assert bf.contains(item), f"False negative for {item}"

    def test_false_positive_rate(self):
        """Test that false positive rate is within expected bounds."""
        # Use parameters that give predictable false positive rate
        n = 1000  # Number of items
        m = 10000  # Bit array size
        k = 5      # Number of hash functions

        bf = BloomFilter(size=m, num_hashes=k)

        # Add n items
        added_items = [f"item_{i}" for i in range(n)]
        for item in added_items:
            bf.add(item)

        # Calculate expected false positive rate
        expected_fp_rate = bf.expected_false_positive_rate()

        # Test with items not added
        test_size = 10000
        test_items = [f"test_{i}" for i in range(test_size)]
        false_positives = sum(1 for item in test_items if bf.contains(item))
        actual_fp_rate = false_positives / test_size

        # Actual rate should be reasonably close to expected (within 50%)
        # This is a statistical test, so we allow some variance
        assert abs(actual_fp_rate - expected_fp_rate) < expected_fp_rate * 0.5, \
            f"FP rate {actual_fp_rate} differs too much from expected {expected_fp_rate}"

        # Verify it's in a reasonable range
        assert actual_fp_rate < 0.1, "False positive rate too high"

    def test_large_dataset(self):
        """Test Bloom filter with a large dataset."""
        # Parameters for ~1% false positive rate with 10000 items
        n = 10000
        p = 0.01
        m = BloomFilter.optimal_size(n, p)
        k = BloomFilter.optimal_num_hashes(m, n)

        bf = BloomFilter(size=m, num_hashes=k)

        # Add many items
        items = [f"large_test_{i}" for i in range(n)]
        for item in items:
            bf.add(item)

        # Verify no false negatives
        for item in items:
            assert bf.contains(item)

        # Check false positive rate
        expected_fp = bf.expected_false_positive_rate()
        assert expected_fp < 0.02, f"Expected FP rate {expected_fp} too high"

    def test_different_parameters(self):
        """Test Bloom filters with different parameter combinations."""
        test_cases = [
            (1000, 3),   # Small, few hashes
            (10000, 5),  # Medium
            (100000, 7), # Large, many hashes
            (500, 1),    # Minimal hashes
        ]

        for size, num_hashes in test_cases:
            bf = BloomFilter(size=size, num_hashes=num_hashes)

            items = [f"param_test_{i}" for i in range(100)]
            for item in items:
                bf.add(item)

            # No false negatives regardless of parameters
            for item in items:
                assert bf.contains(item), \
                    f"False negative with size={size}, k={num_hashes}"

    def test_optimal_parameters(self):
        """Test calculation of optimal parameters."""
        # Test optimal size calculation
        n = 1000
        p = 0.01
        m = BloomFilter.optimal_size(n, p)
        assert m > 0
        assert m > n  # Size should be larger than number of elements

        # Test optimal num_hashes calculation
        k = BloomFilter.optimal_num_hashes(m, n)
        assert k > 0
        assert k < 20  # Should be reasonable number

        # Test with invalid parameters
        with pytest.raises(ValueError):
            BloomFilter.optimal_size(0, 0.01)

        with pytest.raises(ValueError):
            BloomFilter.optimal_size(1000, 0)

        with pytest.raises(ValueError):
            BloomFilter.optimal_size(1000, 1.5)

    def test_load_factor(self):
        """Test load factor calculation."""
        bf = BloomFilter(size=1000, num_hashes=3)

        # Initially load factor should be 0
        assert bf.load_factor() == 0.0

        # Add items and check load factor increases
        for i in range(100):
            bf.add(f"item_{i}")

        load = bf.load_factor()
        assert 0 < load < 1.0, "Load factor should be between 0 and 1"
        assert bf.bits_set() > 0, "Some bits should be set"

    def test_with_real_dataset(self):
        """Test Bloom filter with real dataset (if available)."""
        # Try to load small dataset
        data_path = Path("data/small/url_stream.txt")
        if not data_path.exists():
            pytest.skip("Small dataset not available")

        # Read URLs from dataset
        with open(data_path, 'r') as f:
            urls = [line.strip() for line in f.readlines()]

        # Get unique URLs
        unique_urls = list(set(urls))
        n = len(unique_urls)

        # Create Bloom filter with 1% false positive rate
        m = BloomFilter.optimal_size(n, 0.01)
        k = BloomFilter.optimal_num_hashes(m, n)
        bf = BloomFilter(size=m, num_hashes=k)

        # Add all unique URLs
        for url in unique_urls:
            bf.add(url)

        # Verify no false negatives
        for url in unique_urls:
            assert bf.contains(url), f"False negative for {url}"

        # Load ground truth if available
        ground_truth_path = Path("data/small/ground_truth.json")
        if ground_truth_path.exists():
            with open(ground_truth_path, 'r') as f:
                ground_truth = json.load(f)

            actual_unique = ground_truth['unique_urls_actual']
            assert len(bf) <= len(urls), "Items added should not exceed total items"
