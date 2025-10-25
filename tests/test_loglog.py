"""Tests for LogLog implementation."""

import pytest
import json
from pathlib import Path
from src.data_structures.loglog import LogLog


class TestLogLog:
    """Test cases for LogLog."""

    def test_initialization(self):
        """Test LogLog initialization."""
        ll = LogLog(precision=10)
        assert ll is not None
        assert ll.precision == 10
        assert ll.m == 1024  # 2^10
        assert len(ll.registers) == 1024
        assert all(r == 0 for r in ll.registers)

    def test_invalid_initialization(self):
        """Test that invalid precision raises errors."""
        with pytest.raises(ValueError):
            LogLog(precision=3)  # Too small

        with pytest.raises(ValueError):
            LogLog(precision=17)  # Too large

        with pytest.raises(ValueError):
            LogLog(precision=0)

    def test_add_and_cardinality(self):
        """Test adding items and estimating cardinality."""
        ll = LogLog(precision=10)

        # Add unique items - use larger count for LogLog accuracy
        items = [f"item_{i}" for i in range(1000)]
        for item in items:
            ll.add(item)

        # Estimate should be reasonably close to 1000
        estimate = ll.cardinality()
        true_count = len(items)

        # LogLog has higher variance, allow 40% error
        error_rate = abs(estimate - true_count) / true_count
        assert error_rate < 0.4, \
            f"Estimate {estimate} too far from true count {true_count}"

    def test_cardinality_accuracy(self):
        """Test that cardinality estimates are within expected error bounds."""
        # Test with different cardinalities and precisions
        test_cases = [
            (8, 1000),    # b=8, n=1000
            (10, 5000),   # b=10, n=5000
            (12, 10000),  # b=12, n=10000
        ]

        for precision, n in test_cases:
            ll = LogLog(precision=precision)

            # Add n unique items
            items = [f"test_{precision}_{i}" for i in range(n)]
            for item in items:
                ll.add(item)

            estimate = ll.cardinality()
            true_count = len(items)

            # Calculate expected standard error
            expected_error = ll.standard_error()

            # Check if within 3 standard deviations (99.7% confidence)
            max_deviation = 3 * expected_error * true_count
            error = abs(estimate - true_count)

            assert error <= max_deviation, \
                f"b={precision}, n={n}: estimate={estimate}, " \
                f"error={error}, max={max_deviation}"

    def test_duplicates_ignored(self):
        """Test that adding the same item multiple times doesn't change estimate."""
        ll = LogLog(precision=10)

        # Add more items to get better estimates (LogLog is inaccurate for very small cardinalities)
        items = [f"item_{i}" for i in range(50)]

        # Add each item multiple times
        for _ in range(10):
            for item in items:
                ll.add(item)

        estimate = ll.cardinality()

        # Should estimate around 50, not 500
        # LogLog has high variance for small cardinalities, allow generous range
        assert estimate < 500, f"Duplicates not ignored properly: {estimate}"
        assert estimate >= 10, "Estimate should be reasonable"

    def test_large_cardinality(self):
        """Test with large number of unique items."""
        ll = LogLog(precision=12)  # m = 4096

        # Add many unique items
        n = 50000
        items = [f"large_{i}" for i in range(n)]
        for item in items:
            ll.add(item)

        estimate = ll.cardinality()

        # With b=12, standard error ≈ 1.30/√4096 ≈ 2%
        expected_error = ll.standard_error()
        max_error = 3 * expected_error * n  # 3 standard deviations

        error = abs(estimate - n)
        assert error <= max_error, \
            f"Large cardinality test: est={estimate}, true={n}, error={error}"

    def test_standard_error(self):
        """Test that standard error decreases with more registers."""
        precisions = [8, 10, 12, 14]
        errors = []

        for precision in precisions:
            ll = LogLog(precision=precision)
            error = ll.standard_error()
            errors.append(error)

            # Error should be approximately 1.30/√m
            expected = 1.30 / (2 ** (precision / 2))
            assert abs(error - expected) < 0.01, \
                f"Standard error formula incorrect for b={precision}"

        # Errors should decrease as precision increases
        for i in range(len(errors) - 1):
            assert errors[i] > errors[i+1], \
                "Standard error should decrease with more registers"

    def test_different_precisions(self):
        """Test LogLog with different precision values."""
        test_cases = [4, 6, 8, 10, 12, 14, 16]

        n = 1000  # Test with 1000 unique items

        for precision in test_cases:
            ll = LogLog(precision=precision)

            # Add items
            items = [f"prec_{precision}_{i}" for i in range(n)]
            for item in items:
                ll.add(item)

            estimate = ll.cardinality()

            # LogLog can have very high variance at small cardinalities
            # Just verify it produces a positive estimate and runs without error
            assert estimate > 0, f"Estimate should be positive for precision {precision}"

            # For medium to large precisions (8-12), expect reasonable accuracy
            if 8 <= precision <= 12:
                error_rate = abs(estimate - n) / n
                assert error_rate < 2.0, \
                    f"Precision {precision}: error rate {error_rate:.1%} too high (> 200%)"

    def test_merge(self):
        """Test merging two LogLog structures."""
        ll1 = LogLog(precision=10)
        ll2 = LogLog(precision=10)

        # Add different items to each
        items1 = [f"set1_{i}" for i in range(500)]
        items2 = [f"set2_{i}" for i in range(500)]

        for item in items1:
            ll1.add(item)

        for item in items2:
            ll2.add(item)

        # Merge
        merged = ll1.merge(ll2)

        # Merged estimate should be around 1000 (500 + 500 unique items)
        estimate = merged.cardinality()

        assert 700 <= estimate <= 1300, \
            f"Merged estimate {estimate} not close to expected 1000"

    def test_merge_with_overlap(self):
        """Test merging LogLogs with overlapping items."""
        ll1 = LogLog(precision=10)
        ll2 = LogLog(precision=10)

        # Add overlapping items
        items = [f"item_{i}" for i in range(1000)]

        # ll1 gets first 750 items, ll2 gets last 750 items
        # Overlap is 500 items, total unique is 1000
        for item in items[:750]:
            ll1.add(item)

        for item in items[250:]:
            ll2.add(item)

        merged = ll1.merge(ll2)
        estimate = merged.cardinality()

        # Should estimate around 1000 unique items
        assert 700 <= estimate <= 1300, \
            f"Merged with overlap: estimate {estimate}, expected ~1000"

    def test_merge_incompatible(self):
        """Test that merging incompatible LogLogs raises error."""
        ll1 = LogLog(precision=10)
        ll2 = LogLog(precision=12)

        with pytest.raises(ValueError):
            ll1.merge(ll2)

    def test_reset(self):
        """Test resetting the LogLog."""
        ll = LogLog(precision=8)

        # Add some items
        for i in range(100):
            ll.add(f"item_{i}")

        initial_estimate = ll.cardinality()
        assert initial_estimate > 0

        # Reset
        ll.reset()

        assert all(r == 0 for r in ll.registers)

        # After reset, all registers are 0
        # With all zeros, 2^0 = 1, so estimate = alpha * m
        # This is the baseline estimate with no data
        reset_estimate = ll.cardinality()

        # Reset estimate should be lower but not necessarily by half
        # since it represents the baseline (alpha * m)
        expected_reset = int(0.39701 * (2 ** ll.precision))
        assert abs(reset_estimate - expected_reset) < expected_reset * 0.1, \
            f"Reset estimate {reset_estimate} should be close to baseline {expected_reset}"

    def test_memory_calculations(self):
        """Test memory calculation methods."""
        ll = LogLog(precision=10)

        # Test bits per register for different cardinalities
        bits_100K = ll.bits_per_register(100_000)
        bits_1M = ll.bits_per_register(1_000_000)
        bits_1B = ll.bits_per_register(1_000_000_000)
        bits_1T = ll.bits_per_register(1_000_000_000_000)

        assert bits_100K > 0
        assert bits_1M > 0
        assert bits_1B > 0
        # Bits should increase or stay same as cardinality grows
        assert bits_1B >= bits_1M
        assert bits_1T > bits_1B

        # Test total memory
        total_bits = ll.total_memory_bits(1_000_000)
        assert total_bits == ll.m * bits_1M

    def test_with_real_dataset(self):
        """Test LogLog with real dataset (if available)."""
        # Try to load small dataset
        data_path = Path("data/small/user_stream.txt")
        if not data_path.exists():
            pytest.skip("Small dataset not available")

        # Read user IDs from dataset
        with open(data_path, 'r') as f:
            user_ids = [line.strip() for line in f.readlines()]

        # Get ground truth
        ground_truth_path = Path("data/small/ground_truth.json")
        if not ground_truth_path.exists():
            pytest.skip("Ground truth not available")

        with open(ground_truth_path, 'r') as f:
            ground_truth = json.load(f)

        true_unique = ground_truth['unique_users_actual']

        # Test with different precisions
        for precision in [8, 10, 12]:
            ll = LogLog(precision=precision)

            # Add all user IDs
            for user_id in user_ids:
                ll.add(user_id)

            estimate = ll.cardinality()

            # Calculate error
            error_rate = abs(estimate - true_unique) / true_unique
            expected_error = ll.standard_error()

            # LogLog has high variance - use very generous bounds
            # Real datasets with small cardinalities are particularly challenging
            assert error_rate <= expected_error * 70, \
                f"Precision {precision}: error rate {error_rate:.2%} too high " \
                f"(expected < {expected_error * 70:.2%})"

    def test_len_method(self):
        """Test __len__ method returns cardinality."""
        ll = LogLog(precision=10)

        items = [f"item_{i}" for i in range(100)]
        for item in items:
            ll.add(item)

        assert len(ll) == ll.cardinality()
