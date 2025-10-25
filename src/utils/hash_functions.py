"""Hash function utilities for probabilistic data structures.

This module implements MurmurHash3 from scratch without using external hashing libraries.
MurmurHash3 was created by Austin Appleby and is in the public domain.
"""


def _rotl32(x: int, r: int) -> int:
    """Rotate a 32-bit integer left by r bits.

    Args:
        x: The integer to rotate
        r: Number of bits to rotate

    Returns:
        Rotated 32-bit integer
    """
    return ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF


def _fmix32(h: int) -> int:
    """Finalization mix for MurmurHash3 32-bit.

    Force all bits of a hash block to avalanche.

    Args:
        h: Hash value to mix

    Returns:
        Mixed hash value
    """
    h &= 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x85ebca6b) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xc2b2ae35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


def murmurhash3_32(data: bytes, seed: int = 0) -> int:
    """MurmurHash3 32-bit implementation.

    A fast, non-cryptographic hash function suitable for hash tables
    and probabilistic data structures.

    Args:
        data: Bytes to hash
        seed: Seed value for the hash function (allows multiple hash functions)

    Returns:
        32-bit hash value as integer
    """
    c1 = 0xcc9e2d51
    c2 = 0x1b873593

    length = len(data)
    h1 = seed & 0xFFFFFFFF
    roundedEnd = length & 0xFFFFFFFC  # Round down to 4 byte block

    # Process 4-byte blocks
    for i in range(0, roundedEnd, 4):
        # Read 4 bytes as little-endian
        k1 = (data[i] |
              (data[i + 1] << 8) |
              (data[i + 2] << 16) |
              (data[i + 3] << 24))

        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = _rotl32(k1, 15)
        k1 = (k1 * c2) & 0xFFFFFFFF

        h1 ^= k1
        h1 = _rotl32(h1, 13)
        h1 = (h1 * 5 + 0xe6546b64) & 0xFFFFFFFF

    # Process remaining bytes (0-3 bytes)
    k1 = 0
    val = length & 0x03
    if val == 3:
        k1 = (data[roundedEnd + 2] << 16) & 0xFFFFFFFF
    if val >= 2:
        k1 |= (data[roundedEnd + 1] << 8) & 0xFFFFFFFF
    if val >= 1:
        k1 |= data[roundedEnd] & 0xFFFFFFFF

    if val > 0:
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = _rotl32(k1, 15)
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1

    # Finalization
    h1 ^= length
    h1 = _fmix32(h1)

    return h1


def hash_string(item: str, seed: int = 0) -> int:
    """Generate a hash value for a string using MurmurHash3.

    Args:
        item: String to hash
        seed: Seed value for the hash function

    Returns:
        32-bit hash value as integer
    """
    data = item.encode('utf-8')
    return murmurhash3_32(data, seed)


def multi_hash(item: str, num_hashes: int, size: int) -> list:
    """Generate multiple hash positions using double hashing.

    Uses the double hashing technique: h_i(x) = (h1(x) + i * h2(x)) mod size
    This is more efficient than computing k independent hash functions.

    Args:
        item: String to hash
        num_hashes: Number of hash values to generate
        size: Size of the hash table/bit array (for modulo operation)

    Returns:
        List of hash positions in range [0, size)
    """
    # Generate two base hash values with different seeds
    h1 = hash_string(item, seed=0) % size
    h2 = hash_string(item, seed=1) % size

    # Ensure h2 is odd (and thus coprime with power-of-2 sizes)
    if h2 % 2 == 0:
        h2 += 1

    # Generate k hash values using double hashing
    positions = []
    for i in range(num_hashes):
        pos = (h1 + i * h2) % size
        positions.append(pos)

    return positions


def hash_to_bits(item: str, num_bits: int = 64, seed: int = 0) -> int:
    """Generate a hash value with specified number of bits.

    For values larger than 32 bits, combines multiple MurmurHash3 calls.

    Args:
        item: String to hash
        num_bits: Number of bits in the hash (default 64 for LogLog)
        seed: Seed value for the hash function

    Returns:
        Hash value as integer with num_bits bits
    """
    if num_bits <= 32:
        return hash_string(item, seed) & ((1 << num_bits) - 1)

    # For 64-bit hashes, combine two 32-bit hashes
    h1 = hash_string(item, seed)
    h2 = hash_string(item, seed + 1)

    # Combine into 64-bit value
    result = (h1 << 32) | h2

    # Mask to num_bits
    if num_bits < 64:
        result &= (1 << num_bits) - 1

    return result
