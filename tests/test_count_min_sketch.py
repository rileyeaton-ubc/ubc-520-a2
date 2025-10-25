"""Tests for Count-Min Sketch implementation."""

import pytest
import json
from pathlib import Path
from collections import Counter
from src.data_structures.count_min_sketch import CountMinSketch


class TestCountMinSketch:
    """Test cases for CountMinSketch."""

    def test_initialization(self):
        """Test Count-Min Sketch initialization."""
        cms = CountMinSketch(width=1000, depth=5)
        assert cms is not None
        assert cms.width == 1000
        assert cms.depth == 5
        assert len(cms) == 0
        assert cms.get_total_count() == 0

    def test_invalid_initialization(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            CountMinSketch(width=0, depth=5)

        with pytest.raises(ValueError):
            CountMinSketch(width=1000, depth=0)

        with pytest.raises(ValueError):
            CountMinSketch(width=-100, depth=5)

    def test_update_and_estimate(self):
        """Test updating counts and estimating frequencies."""
        cms = CountMinSketch(width=1000, depth=5)

        # Update some items
        cms.update("apple", 5)
        cms.update("banana", 3)
        cms.update("cherry", 1)

        # Estimates should be at least the true counts (no underestimation)
        assert cms.estimate("apple") >= 5
        assert cms.estimate("banana") >= 3
        assert cms.estimate("cherry") >= 1

        # Total count should match
        assert cms.get_total_count() == 9

    def test_point_query(self):
        """Test single item with multiple updates."""
        cms = CountMinSketch(width=1000, depth=5)

        item = "test_item"
        total = 0

        # Add same item multiple times
        for i in range(1, 101):
            cms.update(item, 1)
            total += 1
            estimate = cms.estimate(item)
            # Should never underestimate
            assert estimate >= total, f"Underestimated at count {total}: got {estimate}"

    def test_no_underestimation(self):
        """Test that frequency estimates are never below true counts."""
        cms = CountMinSketch(width=5000, depth=7)

        # Track true frequencies
        true_counts = {}
        items = [f"item_{i % 100}" for i in range(1000)]

        for item in items:
            cms.update(item)
            true_counts[item] = true_counts.get(item, 0) + 1

        # Check that all estimates are >= true counts
        for item, true_count in true_counts.items():
            estimate = cms.estimate(item)
            assert estimate >= true_count, \
                f"Underestimated {item}: true={true_count}, estimate={estimate}"

    def test_accuracy_with_ground_truth(self):
        """Test accuracy using simple dataset with known frequencies."""
        # Use good parameters for accuracy
        epsilon = 0.01
        delta = 0.01
        w = CountMinSketch.required_width(epsilon)
        d = CountMinSketch.required_depth(delta)

        cms = CountMinSketch(width=w, depth=d)

        # Create dataset with known distribution
        items = []
        for i in range(100):
            # Create power-law distribution: item_i appears (100-i) times
            for j in range(100 - i):
                items.append(f"item_{i}")

        # Update sketch
        for item in items:
            cms.update(item)

        # Track true frequencies
        true_freq = Counter(items)

        # Check accuracy for some items
        total = len(items)
        errors = []

        for item, true_count in list(true_freq.items())[:20]:  # Check top 20
            estimate = cms.estimate(item)
            error = estimate - true_count
            errors.append(error)

            # Should not underestimate
            assert estimate >= true_count

            # Error should be within bounds (εN) for most items
            max_error = epsilon * total
            assert error <= max_error * 2, \
                f"Error too large for {item}: {error} > {max_error * 2}"

    def test_error_bounds(self):
        """Test that error is within ε*N bounds."""
        epsilon = 0.01
        delta = 0.01
        w = CountMinSketch.required_width(epsilon)
        d = CountMinSketch.required_depth(delta)

        cms = CountMinSketch(width=w, depth=d)

        # Add many items
        n = 10000
        items = [f"item_{i % 1000}" for i in range(n)]  # 1000 unique items

        for item in items:
            cms.update(item)

        # Calculate expected error bound
        max_error = cms.expected_error(epsilon)

        # Each item appears 10 times (n/1000)
        true_count = 10
        errors_within_bound = 0
        total_tested = 0

        for i in range(100):  # Test first 100 items
            item = f"item_{i}"
            estimate = cms.estimate(item)
            error = estimate - true_count

            if error <= max_error:
                errors_within_bound += 1
            total_tested += 1

        # With probability 1-δ, error should be within bound
        # We expect at least 95% to be within bounds (allowing some variance)
        success_rate = errors_within_bound / total_tested
        assert success_rate >= 0.90, \
            f"Only {success_rate*100:.1f}% within error bounds (expected >90%)"

    def test_invalid_update(self):
        """Test that negative counts raise errors."""
        cms = CountMinSketch(width=100, depth=3)

        with pytest.raises(ValueError):
            cms.update("item", -5)

    def test_different_parameters(self):
        """Test Count-Min Sketch with different parameter combinations."""
        test_cases = [
            (100, 3),    # Small
            (1000, 5),   # Medium
            (10000, 7),  # Large
            (500, 1),    # Single hash function
        ]

        for width, depth in test_cases:
            cms = CountMinSketch(width=width, depth=depth)

            items = {"a": 10, "b": 20, "c": 5}
            for item, count in items.items():
                cms.update(item, count)

            # No underestimation regardless of parameters
            for item, true_count in items.items():
                estimate = cms.estimate(item)
                assert estimate >= true_count, \
                    f"Underestimated with w={width}, d={depth}"

    def test_required_parameters(self):
        """Test calculation of required width and depth."""
        # Test width calculation
        epsilon = 0.01
        w = CountMinSketch.required_width(epsilon)
        assert w > 0
        assert w == pytest.approx(272, abs=1)  # e/0.01 ≈ 271.8

        # Test depth calculation
        delta = 0.01
        d = CountMinSketch.required_depth(delta)
        assert d > 0
        assert d == pytest.approx(5, abs=1)  # ln(100) ≈ 4.6

        # Test with invalid parameters
        with pytest.raises(ValueError):
            CountMinSketch.required_width(0)

        with pytest.raises(ValueError):
            CountMinSketch.required_width(1.5)

        with pytest.raises(ValueError):
            CountMinSketch.required_depth(0)

    def test_merge(self):
        """Test merging two Count-Min Sketches."""
        cms1 = CountMinSketch(width=1000, depth=5)
        cms2 = CountMinSketch(width=1000, depth=5)

        # Add different items to each sketch
        cms1.update("a", 10)
        cms1.update("b", 5)

        cms2.update("c", 7)
        cms2.update("a", 3)  # Same item in both

        # Merge sketches
        merged = cms1.merge(cms2)

        # Merged sketch should have combined counts
        assert merged.estimate("a") >= 13  # 10 + 3
        assert merged.estimate("b") >= 5
        assert merged.estimate("c") >= 7
        assert merged.get_total_count() == 25  # 10 + 5 + 7 + 3

    def test_merge_incompatible(self):
        """Test that merging incompatible sketches raises error."""
        cms1 = CountMinSketch(width=1000, depth=5)
        cms2 = CountMinSketch(width=500, depth=5)  # Different width

        with pytest.raises(ValueError):
            cms1.merge(cms2)

    def test_with_real_dataset(self):
        """Test Count-Min Sketch with real dataset (if available)."""
        # Try to load small dataset
        data_path = Path("data/small/url_stream.txt")
        if not data_path.exists():
            pytest.skip("Small dataset not available")

        # Read URLs from dataset
        with open(data_path, 'r') as f:
            urls = [line.strip() for line in f.readlines()]

        # Create Count-Min Sketch with good parameters
        epsilon = 0.01
        delta = 0.01
        w = CountMinSketch.required_width(epsilon)
        d = CountMinSketch.required_depth(delta)

        cms = CountMinSketch(width=w, depth=d)

        # Track true frequencies
        true_freq = Counter(urls)

        # Update sketch
        for url in urls:
            cms.update(url)

        # Verify total count
        assert cms.get_total_count() == len(urls)

        # Check accuracy for top URLs
        ground_truth_path = Path("data/small/ground_truth.json")
        if ground_truth_path.exists():
            with open(ground_truth_path, 'r') as f:
                ground_truth = json.load(f)

            # Check top 10 URLs
            for url, true_count in ground_truth['top_10_urls'][:10]:
                estimate = cms.estimate(url)
                # Should not underestimate
                assert estimate >= true_count, \
                    f"Underestimated {url}: true={true_count}, est={estimate}"

                # Error should be reasonable
                error = estimate - true_count
                max_error = epsilon * len(urls)
                assert error <= max_error * 3, \
                    f"Error too large for {url}: {error} > {max_error * 3}"
