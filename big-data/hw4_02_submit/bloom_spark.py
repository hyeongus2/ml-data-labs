import numpy as np
from pyspark import RDD
from hash_utils import get_hashes
from main_single import id_to_url


###############################################################
# (1) Distributed Bloom Filter Construction
###############################################################

def build_bloom_filter(rdd_keys: RDD, m: int, k: int, base_seed: int = 0) -> np.ndarray:
    """
    Distributed Bloom filter construction.

    Steps:
      1) For each key -> compute hash indices
      2) Emit (index,1)
      3) reduceByKey(OR)
      4) Collect & build numpy array
    """
    sc = rdd_keys.context

    # flatMap: key → [ (idx,1), (idx,1), ... ]
    ###################### TODO ####################
    idx_rdd = rdd_keys.flatMap(lambda key: [(idx, 1) for idx in get_hashes(key, m, k, base_seed)])
    ################################################

    # apply OR operation on the bit values at each index position in the BloomFilter
    ###################### TODO ####################
    reduced = idx_rdd.reduceByKey(lambda a, b: a | b)
    ################################################

    # Collect final set bits
    ones = reduced.collect()

    # Make bit array
    bits = np.zeros(m, dtype=np.uint8)
    ###################### TODO ####################
    for idx, _ in ones:
        bits[idx] = 1
    ################################################
    return bits


###############################################################
# (2) Distributed FPR Evaluation
###############################################################

def evaluate_fpr(rdd_neg_ids: RDD, bits: np.ndarray,
                 m: int, k: int, base_seed: int = 0) -> float:
    """
    TODO: Compute false positive rate distributedly.

    Steps:
      1) Broadcast the bit array.
      2) Map over negative keys:
            - Convert ID -> URL
            - Compute k hash positions
            - Return 1 if all bits==1, else 0
      3) Sum and divide by count.
    """
    sc = rdd_neg_ids.context
    bits_bc = sc.broadcast(bits)

    def check_negative(i):
        url = id_to_url(i)
        ###################### TODO ####################
        idxs = get_hashes(url, m, k, base_seed)
        ################################################
        b = bits_bc.value
        return 1 if all(b[idx] == 1 for idx in idxs) else 0

    ###################### TODO ####################
    false_pos = rdd_neg_ids.map(check_negative).sum()
    total = rdd_neg_ids.count()
    ################################################
    return false_pos / total if total > 0 else 0.0


###############################################################
# (3) Two-Stage Cascade Construction (Bonus)
###############################################################

def build_two_stage_cascade(rdd_train_ids: RDD,
                            m1: int, k1: int,
                            m2: int, k2: int,
                            base_seed1: int = 0,
                            base_seed2: int = 1000):
    """
    Build two cascade Bloom filters over exact same training set.
    """
    ###################### TODO ####################
    rdd_urls = rdd_train_ids.map(id_to_url)
    bits1 = build_bloom_filter(rdd_urls, m1, k1, base_seed1)
    bits2 = build_bloom_filter(rdd_urls, m2, k2, base_seed2)
    ################################################
    return bits1, bits2


###############################################################
# (4) Two-Stage Cascade FPR Evaluation (Bonus)
###############################################################

def evaluate_fpr_cascade(rdd_neg_ids: RDD,
                         bits1: np.ndarray, m1: int, k1: int,
                         bits2: np.ndarray, m2: int, k2: int,
                         base_seed1: int = 0,
                         base_seed2: int = 1000) -> float:
    """
    A key is FP only if BOTH Bloom filters accept it.
    """
    sc = rdd_neg_ids.context

    bits1_bc = sc.broadcast(bits1)
    bits2_bc = sc.broadcast(bits2)

    def check(i):
        url = id_to_url(i)
        ###################### TODO ####################
        # BF1 check
        idxs1 = get_hashes(url, m1, k1, base_seed1)
        accept1 = all(bits1_bc.value[idx] == 1 for idx in idxs1)
        if not accept1:
            accept = False
        else:
            # BF2 check
            idxs2 = get_hashes(url, m2, k2, base_seed2)
            accept2 = all(bits2_bc.value[idx] == 1 for idx in idxs2)
            accept = accept2
        ################################################
        return 1 if accept else 0 

    ###################### TODO ####################
    false_pos = rdd_neg_ids.map(check).sum()
    total = rdd_neg_ids.count()
    ###################### TODO ####################
    return false_pos / total if total > 0 else 0.0