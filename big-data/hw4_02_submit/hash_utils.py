# hash_utils.py 

# Simple hashing utilities for the Bloom Filter assignment.
# You MUST implement one functionnmarked as TODO:
#   - get_hashes(key, m, k)

# No external libraries are needed. Use only basic Python libraries.

from typing import List
import hashlib

def _hash_bytes(data: bytes, seed: int) -> int:
    """
    Deterministic 64-bit hash based on SHA-256.
    Using explicit seed so results are stable across runs and processes.
    """
    # 4-byte little-endian seed prefix
    seed_bytes = seed.to_bytes(4, byteorder="little", signed=False)
    h = hashlib.sha256(seed_bytes + data).digest()
    # use first 8 bytes as unsigned 64-bit integer
    return int.from_bytes(h[:8], byteorder="little", signed=False)

def get_hashes(key: str, m: int, k: int, base_seed: int = 0) -> List[int]:
    """
    Compute k hash indices in [0, m) for the given key.

    Args:
        key: string (e.g., URL)
        m: Bloom filter size (#bits)
        k: number of hash functions (k <= len(A))

    We obtain independent-ish hashes by varying the seed.
    """
    if m <= 0:
        raise ValueError("m must be positive")
    if k <= 0:
        raise ValueError("k must be positive")

    key_bytes = key.encode("utf-8")
    indices = []
    ###################### TODO ####################
    for i in range(k):
        hash_value = _hash_bytes(key_bytes, base_seed + i)
        indices.append(hash_value % m)
    ################################################
    return indices