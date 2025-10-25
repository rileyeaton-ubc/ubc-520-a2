"""Count-Min Sketch implementation."""

import math
from src.utils.hash_functions import hash_string


class CountMinSketch:
    """A probabilistic data structure for frequency estimation.

    The Count-Min sketch is used to estimate the frequency of items in a data stream
    using sub-linear space. It guarantees that frequency estimates are never
    underestimated, but may be overestimated due to hash collisions.

    Reference:
        Cormode, Graham and Muthukrishnan, S. (2005). "An improved data stream
        summary: the count-min sketch and its applications". Journal of Algorithms.
        55 (1): 58-75.
    """

    def __init__(self, width: int, depth: int):
        """Initialize the Count-Min Sketch.

        Args:
            width: Width of the sketch array (w)
            depth: Depth (number of hash functions, d)

        Raises:
            ValueError: If width or depth is not positive
        """
        if width <= 0:
            raise ValueError("Width must be positive")
        if depth <= 0:
            raise ValueError("Depth must be positive")

        self.width = width
        self.depth = depth
        self.table = [[0] * width for _ in range(depth)]
        self.total_count = 0

    def update(self, item: str, count: int = 1) -> None:
        """Update the count for an item.

        Args:
            item: String item to update
            count: Count to add (default 1)

        Raises:
            ValueError: If count is negative
        """
        if count < 0:
            raise ValueError("Count must be non-negative")

        # Hash the item with each of the d hash functions
        for i in range(self.depth):
            # Use different seed for each row
            hash_val = hash_string(item, seed=i)
            pos = hash_val % self.width
            self.table[i][pos] += count

        self.total_count += count

    def estimate(self, item: str) -> int:
        """Estimate the frequency of an item.

        Returns the minimum count across all d hash positions.
        This provides an upper bound on the true frequency.

        Args:
            item: String item to query

        Returns:
            Estimated frequency (never less than true frequency)
        """
        # Get the minimum value across all d positions
        min_count = float('inf')

        for i in range(self.depth):
            hash_val = hash_string(item, seed=i)
            pos = hash_val % self.width
            min_count = min(min_count, self.table[i][pos])

        return int(min_count)

    def get_total_count(self) -> int:
        """Get the total number of updates (sum of all counts).

        Returns:
            Total count across all updates
        """
        return self.total_count

    def expected_error(self, epsilon: float) -> float:
        """Calculate expected error bound.

        With probability at least 1-δ, the error is at most εN,
        where N is the total count.

        Args:
            epsilon: Error parameter (ε)

        Returns:
            Expected error bound (εN)
        """
        return epsilon * self.total_count

    @staticmethod
    def required_width(epsilon: float) -> int:
        """Calculate required width for given error parameter.

        Uses the formula: w = ⌈e/ε⌉

        Args:
            epsilon: Desired error parameter (ε)

        Returns:
            Required width

        Raises:
            ValueError: If epsilon is not in valid range
        """
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError("Epsilon must be between 0 and 1")

        e = math.e
        width = math.ceil(e / epsilon)
        return width

    @staticmethod
    def required_depth(delta: float) -> int:
        """Calculate required depth for given failure probability.

        Uses the formula: d = ⌈ln(1/δ)⌉

        Args:
            delta: Desired failure probability (δ)

        Returns:
            Required depth

        Raises:
            ValueError: If delta is not in valid range
        """
        if delta <= 0 or delta >= 1:
            raise ValueError("Delta must be between 0 and 1")

        depth = math.ceil(math.log(1 / delta))
        return depth

    def heavy_hitters(self, threshold: float) -> list:
        """Find potential heavy hitters above a threshold.

        Returns items whose estimated frequency is above threshold * total_count.
        Note: This requires maintaining a separate data structure to track seen items.
        For demonstration, this method signature is provided but requires external tracking.

        Args:
            threshold: Fraction of total count (0.0 to 1.0)

        Returns:
            List of (item, estimated_count) tuples

        Raises:
            NotImplementedError: This requires external item tracking
        """
        raise NotImplementedError(
            "Heavy hitter detection requires tracking all seen items separately. "
            "Use this sketch with an external set to track candidate items."
        )

    def __len__(self) -> int:
        """Return the total number of updates.

        Returns:
            Total count
        """
        return self.total_count

    def merge(self, other: 'CountMinSketch') -> 'CountMinSketch':
        """Merge another Count-Min Sketch with compatible dimensions.

        Args:
            other: Another CountMinSketch with same width and depth

        Returns:
            New merged CountMinSketch

        Raises:
            ValueError: If dimensions don't match
        """
        if self.width != other.width or self.depth != other.depth:
            raise ValueError("Cannot merge sketches with different dimensions")

        merged = CountMinSketch(self.width, self.depth)

        # Add counts element-wise
        for i in range(self.depth):
            for j in range(self.width):
                merged.table[i][j] = self.table[i][j] + other.table[i][j]

        merged.total_count = self.total_count + other.total_count

        return merged
