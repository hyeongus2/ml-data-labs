import sys
import re
import time
import numpy as np
from itertools import combinations

K = 3   # length of character shingles
B = 6   # number of bands
R = 20  # number of rows per band
M = B * R  # signature length
SIG_THRESHOLD = 0.9 # similarity threshold for candidate pairs
SEED = 0    # random seed for reproducibility


def preprocess_text(s):
    """
    Keep only lowercase alphabets and whitespace.
    Remove everything else.
    """
    s = s.lower()
    # delete all non-[a-z ] characters
    s = re.sub(r'[^a-z ]+', '', s)
    return s

def get_shingles(s):
    """
    Return the set of k-shingles for the given string s.
    """
    if len(s) < K:
        return set()
    return {s[i:i + K] for i in range(len(s) - K + 1)}

def get_smallest_prime_gte(n):
    """
    Return the smallest prime number greater than or equal to n.
    """
    if n <= 2:
        return 2
    
    p = n if n % 2 else n + 1

    def is_prime(x):
        if x < 2:
            return False
        if x % 2 == 0:
            return x == 2
        for i in range(3, int(x**0.5) + 1, 2):
            if x % i == 0:
                return False
        return True

    while not is_prime(p):
        p += 2
    return p

def get_hash_components(n):
    """
    Return two lists of hash function components a and b, and c.
    Each hash function is of the form: h(x) = (a*x + b) mod c
    where c is a prime number greater than n.
    """
    np.random.seed(SEED)
    c = get_smallest_prime_gte(n if n > 0 else 2)
    a_arr = np.random.randint(0, c, size=M)
    b_arr = np.random.randint(0, c, size=M)
    return a_arr, b_arr, c

def compute_minhash_signature(rows, a_arr, b_arr, c):
    """
    Compute the MinHash signature for a given set of shingles.
    signature[i] = min_{shingle in shingle_set} (a_i * r + b_i) % c
    rows: 1D numpy array mapping shingle to its row index
    Return a numpy array of length m representing the signature.
    """
    if len(rows) == 0:
        return np.full(M, c)
    
    # c is a prime number greater than number of unique shingles
    # Thus, it has the same effect as infinity
    signature = np.full(M, c)
    for i in range(M):
        a = a_arr[i]
        b = b_arr[i]
        sigs  = (a * rows + b) % c
        signature[i] = int(np.min(sigs))
    return signature

def get_bands(signature):
    """
    Yield (band_index, band_tuple) of the given signature.
    """
    for i in range(B):
        start = i * R
        end = start + R
        yield i, tuple(signature[start:end])

def signature_similarity(sig1, sig2):
    """
    Compute the similarity between two signatures.
    """
    assert len(sig1) == len(sig2)
    return float(np.mean(sig1 == sig2))


def main():
    file_path = sys.argv[1]
    start_time = time.time()

    article_ids = []
    texts = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split(' ', 1)
            if len(parts) ==1:
                article_id, text = parts[0], ''
            else:
                article_id, text = parts
            article_ids.append(article_id)
            texts.append(preprocess_text(text))

    # Step 1: Shingling per text and build global shingles
    shingles_per_text = []
    global_shingles = set()
    for text in texts:
        shs = get_shingles(text)
        shingles_per_text.append(shs)
        global_shingles.update(shs)
    global_shingles = sorted(global_shingles)  # lexicographical order
    row_indices = {shingle: idx for idx, shingle in enumerate(global_shingles)}
    N = len(global_shingles)  # number of unique shingles

    # Step 2: MinHash Signature Computation
    a_arr, b_arr, c = get_hash_components(N)

    signatures = []
    for shs in shingles_per_text:
        rows = np.array([row_indices[sh] for sh in shs], dtype=np.int64)
        sig = compute_minhash_signature(rows, a_arr, b_arr, c)
        signatures.append(sig)

    # Step 3: Locality Sensitive Hashing (LSH)
    buckets = dict()  # (band_index, band_tuple) -> list of article indices
    for article_idx, sig in enumerate(signatures):
        for band_idx, band_tuple in get_bands(sig):
            key = (band_idx, band_tuple)
            if key not in buckets:
                buckets[key] = [article_idx]
            else:
                buckets[key].append(article_idx)

    # Step 4: Candidates Generation
    candidate_pairs = set()
    for bucket in buckets.values():
        if len(bucket) >= 2:
            for i, j in combinations(sorted(bucket), 2):
                id1, id2 = article_ids[i], article_ids[j]
                if id1 < id2:
                    candidate_pairs.add((i, j))
                else:
                    candidate_pairs.add((j, i))

    # Step 5: Candidate Verification
    verified_pairs = []
    for i, j in candidate_pairs:
        if len(shingles_per_text[i]) == 0 or len(shingles_per_text[j]) == 0:
            continue
        sim = signature_similarity(signatures[i], signatures[j])
        if sim >= SIG_THRESHOLD:
            verified_pairs.append((article_ids[i], article_ids[j], sim))

    verified_pairs.sort()

    # Output results
    for id1, id2, sim in verified_pairs:
        print(f"{id1}\t{id2}\t{sim:.6f}")

    end_time = time.time()
    # print(f"Elapsed time: {end_time - start_time:.3f} sec")

if __name__ == "__main__":
    main()