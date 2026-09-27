# bloom_single.py 

# Single-machine Bloom filter implementation.
# You MUST implement the parts marked TODO:
#   - initialize the bit array
#   - insert(self, key)
#   - query(self, key)

import numpy as np
from typing import Iterable
from hash_utils import get_hashes


class BloomFilter:
    """
    Simple single-machine Bloom filter.

    Bit array is stored as a numpy array of length m, with values in {0,1}.
    """

    def __init__(self, m: int, k: int, base_seed: int = 0):
        """
        Args:
            m: number of bits in the filter.
            k: number of hash functions.
        """
        if m <= 0:
            raise ValueError("m must be positive")
        if k <= 0:
            raise ValueError("k must be positive")

        self.m = m
        self.k = k
        self.base_seed = base_seed
        ###################### TODO ####################
        self.bits = np.zeros(m, dtype=np.uint8)  # Hint: use numpy array of dtype uint8 with length m, initially all zeros.
        ################################################

    def _indices(self, key: str):
        return get_hashes(key, self.m, self.k, self.base_seed)

    def insert(self, key: str) -> None:
        ###################### TODO ####################
        indices = self._indices(key)
        self.bits[indices] = 1
        ################################################

    def insert_many(self, keys: Iterable[str]) -> None:
        """Insert many keys. (You do NOT need to modify this method.)"""
        for key in keys:
            self.insert(key)

    def query(self, key: str) -> bool:
        """
        Return True if key is *possibly* in the set (all bits for this key are 1),
        and False if key is definitely not in the set (at least one bit is 0).
        """
        ###################### TODO ####################
        indices = self._indices(key)
        if np.all(self.bits[indices] == 1):
            return True
        return False
        ################################################