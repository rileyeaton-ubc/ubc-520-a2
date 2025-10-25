"""Bloom filter implementation."""

import math
from src.utils.hash_functions import multi_hash


class BloomFilter:
    """A probabilistic data structure for set membership testing.

    A Bloom filter is a space-efficient probabilistic data structure used to test
    whether an element is a member of a set. False positive matches are possible,
    but false negatives are not.

    Reference:
        Bloom, Burton H. (1970). "Space/time trade-offs in hash coding with
        allowable errors". Communications of the ACM. 13 (7): 422-426.
    """

    def __init__(self, size: int, num_hashes: int):
        """Initialize the Bloom filter.

        Args:
            size: Size of the bit array (m)
            num_hashes: Number of hash functions to use (k)

        Raises:
            ValueError: If size or num_hashes is not positive
        """
        if size <= 0:
            raise ValueError("Size must be positive")
        if num_hashes <= 0:
            raise ValueError("Number of hash functions must be positive")

        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = [0] * size
        self.items_added = 0

    def add(self, item: str) -> None:
        """Add an item to the filter.

        Args:
            item: String item to add to the filter
        """
        positions = multi_hash(item, self.num_hashes, self.size)
        for pos in positions:
            self.bit_array[pos] = 1
        self.items_added += 1

    def contains(self, item: str) -> bool:
        """Check if an item might be in the set.

        Returns True if the item might be in the set (possible false positive),
        or False if the item is definitely not in the set (no false negatives).

        Args:
            item: String item to check

        Returns:
            True if item might be in set, False if definitely not in set
        """
        positions = multi_hash(item, self.num_hashes, self.size)
        return all(self.bit_array[pos] == 1 for pos in positions)

    def expected_false_positive_rate(self) -> float:
        """Calculate the expected false positive rate.

        Uses the formula: p = (1 - e^(-kn/m))^k
        where:
            k = number of hash functions
            n = number of items added
            m = size of bit array

        Returns:
            Expected false positive probability (0.0 to 1.0)
        """
        if self.items_added == 0:
            return 0.0

        # p = (1 - e^(-k*n/m))^k
        exponent = -self.num_hashes * self.items_added / self.size
        return (1 - math.exp(exponent)) ** self.num_hashes

    def __len__(self) -> int:
        """Return the number of items added to the filter.

        Returns:
            Number of items added
        """
        return self.items_added

    def bits_set(self) -> int:
        """Count the number of bits set to 1 in the bit array.

        Returns:
            Number of bits set to 1
        """
        return sum(self.bit_array)

    def load_factor(self) -> float:
        """Calculate the load factor (fraction of bits set).

        Returns:
            Fraction of bits set to 1 (0.0 to 1.0)
        """
        return self.bits_set() / self.size

    @staticmethod
    def optimal_size(n: int, p: float) -> int:
        """Calculate optimal bit array size for given parameters.

        Uses the formula: m = -n * ln(p) / (ln(2))^2

        Args:
            n: Expected number of elements
            p: Desired false positive rate

        Returns:
            Optimal size of bit array

        Raises:
            ValueError: If n or p are invalid
        """
        if n <= 0:
            raise ValueError("Expected number of elements must be positive")
        if p <= 0 or p >= 1:
            raise ValueError("False positive rate must be between 0 and 1")

        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(math.ceil(m))

    @staticmethod
    def optimal_num_hashes(m: int, n: int) -> int:
        """Calculate optimal number of hash functions.

        Uses the formula: k = (m/n) * ln(2)

        Args:
            m: Size of bit array
            n: Expected number of elements

        Returns:
            Optimal number of hash functions

        Raises:
            ValueError: If m or n are invalid
        """
        if m <= 0 or n <= 0:
            raise ValueError("Size and expected elements must be positive")

        k = (m / n) * math.log(2)
        return max(1, int(math.ceil(k)))
