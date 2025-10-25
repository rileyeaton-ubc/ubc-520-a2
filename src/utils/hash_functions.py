"""Hash function utilities for probabilistic data structures."""

import hashlib


def hash_string(item: str, seed: int = 0) -> int:
    """Generate a hash value for a string.

    Args:
        item: String to hash
        seed: Seed value for the hash function

    Returns:
        Integer hash value
    """
    hash_input = f"{seed}{item}".encode('utf-8')
    return int(hashlib.sha256(hash_input).hexdigest(), 16)


def multi_hash(item: str, num_hashes: int) -> list[int]:
    """Generate multiple hash values for an item.

    Args:
        item: String to hash
        num_hashes: Number of hash values to generate

    Returns:
        List of hash values
    """
    return [hash_string(item, seed=i) for i in range(num_hashes)]
