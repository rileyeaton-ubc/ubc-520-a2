"""LogLog implementation."""

import math
from src.utils.hash_functions import hash_to_bits


class LogLog:
    """A probabilistic data structure for cardinality estimation.

    LogLog uses m = 2^b registers to estimate the number of distinct elements
    in a multiset. Each register stores log(log(n_max)) bits, where n_max is
    the maximum expected cardinality.

    Reference:
        Durand, Marianne and Flajolet, Philippe (2003). "Loglog counting of
        large cardinalities". European Symposium on Algorithms. LNCS 2832: 605-617.
    """

    def __init__(self, precision: int):
        """Initialize the LogLog.

        Args:
            precision: Precision parameter b (typically 4-16).
                      Number of registers will be m = 2^b.

        Raises:
            ValueError: If precision is not in valid range
        """
        if precision < 4 or precision > 16:
            raise ValueError("Precision must be between 4 and 16")

        self.precision = precision
        self.m = 2 ** precision  # Number of registers
        self.registers = [0] * self.m  # Each stores max leading zeros count

    def add(self, item: str) -> None:
        """Add an item to the sketch.

        Args:
            item: String item to add
        """
        # Hash item to 64-bit value
        hash_val = hash_to_bits(item, num_bits=64, seed=0)

        # Use first b bits to select register
        register_idx = hash_val & ((1 << self.precision) - 1)

        # Use remaining bits to count leading zeros
        # Shift right by b bits to remove the index bits
        remaining_bits = hash_val >> self.precision

        # Count leading zeros + 1 (the ρ value)
        rho = self._leading_zeros(remaining_bits, 64 - self.precision) + 1

        # Update register with maximum ρ value seen
        self.registers[register_idx] = max(self.registers[register_idx], rho)

    def cardinality(self) -> int:
        """Estimate the number of unique items.

        Uses the LogLog formula (geometric mean):
            E = α_m * m * (∏ 2^M[j])^(1/m)
            E = α_m * m * 2^((1/m) * Σ M[j])

        where M[j] is the value in register j, and α_m is a correction constant.

        Returns:
            Estimated cardinality
        """
        # Calculate geometric mean: 2^((1/m) * sum of register values)
        sum_registers = sum(self.registers)
        geometric_mean = 2.0 ** (sum_registers / self.m)

        # Get correction factor
        alpha = self._get_alpha(self.m)

        # Calculate raw estimate: α_m * m * geometric_mean
        estimate = alpha * self.m * geometric_mean

        return int(round(estimate))

    def _leading_zeros(self, value: int, max_bits: int) -> int:
        """Count the number of leading zeros in a value.

        Args:
            value: Integer value to examine
            max_bits: Maximum number of bits to consider

        Returns:
            Number of leading zeros (0 to max_bits)
        """
        if value == 0:
            return max_bits

        # Count leading zeros by checking each bit from left
        count = 0
        for i in range(max_bits - 1, -1, -1):
            if value & (1 << i):
                break
            count += 1

        return count

    def _get_alpha(self, m: int) -> float:
        """Get the correction constant α_m based on number of registers.

        For LogLog, the constant is approximately 0.39701 for all m.
        This is based on the Durand & Flajolet (2003) paper.

        Args:
            m: Number of registers

        Returns:
            Correction constant
        """
        # LogLog uses a constant alpha regardless of m
        return 0.39701

    def standard_error(self) -> float:
        """Calculate the standard error of the estimate.

        For LogLog, the standard error is approximately 1.30/√m.

        Returns:
            Standard error as a fraction (e.g., 0.05 for 5%)
        """
        return 1.30 / math.sqrt(self.m)

    def merge(self, other: 'LogLog') -> 'LogLog':
        """Merge another LogLog with the same precision.

        Takes the maximum register value for each position.

        Args:
            other: Another LogLog with same precision

        Returns:
            New merged LogLog

        Raises:
            ValueError: If precisions don't match
        """
        if self.precision != other.precision:
            raise ValueError("Cannot merge LogLogs with different precisions")

        merged = LogLog(self.precision)

        # Take maximum of each register
        for i in range(self.m):
            merged.registers[i] = max(self.registers[i], other.registers[i])

        return merged

    def __len__(self) -> int:
        """Return the estimated cardinality.

        Returns:
            Estimated number of unique items
        """
        return self.cardinality()

    def bits_per_register(self, max_cardinality: int) -> int:
        """Calculate bits needed per register for given maximum cardinality.

        Uses the formula: log2(log2(n_max))

        Args:
            max_cardinality: Maximum expected cardinality

        Returns:
            Bits needed per register
        """
        if max_cardinality <= 1:
            return 1

        # Calculate log2(log2(n_max))
        log_log = math.log2(math.log2(max_cardinality))
        return max(1, int(math.ceil(log_log)))

    def total_memory_bits(self, max_cardinality: int) -> int:
        """Calculate total memory usage in bits.

        Args:
            max_cardinality: Maximum expected cardinality

        Returns:
            Total bits needed (m * bits_per_register)
        """
        bits_per_reg = self.bits_per_register(max_cardinality)
        return self.m * bits_per_reg

    def reset(self) -> None:
        """Reset all registers to zero."""
        self.registers = [0] * self.m
