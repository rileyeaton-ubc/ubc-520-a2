"""Main benchmarking module."""

import time
from typing import Callable, Any


def benchmark_operation(func: Callable, *args, **kwargs) -> tuple[Any, float]:
    """Benchmark a single operation.

    Args:
        func: Function to benchmark
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Tuple of (result, elapsed_time_seconds)
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed
