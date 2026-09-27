# main_single.py 

# Single-machine experiment driver for Bloom filter.
# You MUST complete the run_single_experiment function where marked TODO.

# The script will:
#   1) Generate synthetic "web crawler" URLs using integer IDs.
#   2) Insert n_train URLs into a Bloom filter.
#   3) Use a disjoint set of n_test URLs as negative queries.
#   4) Measure empirical FPR for various k.

import argparse
from bloom_single import BloomFilter


def id_to_url(i: int) -> str:
    """
    Deterministic mapping from integer ID to URL string.
    This simulates URLs discovered by a national-level web crawler.
    """
    return f"https://site{i % 100000}.gov.kr/page/{i}"


def generate_train_ids(n_train: int):
    # IDs [0, n_train)
    return list(range(n_train))


def generate_negative_test_ids(n_train: int, n_test: int):
    # Pure negatives: IDs [n_train, n_train + n_test)
    return list(range(n_train, n_train + n_test))


def run_single_experiment(m: int, k: int, n_train: int, n_test: int) -> float:
    """
    TODO: Complete this function.

    Steps:
      1) Generate train IDs and negative test IDs using helper functions.
      2) Convert train IDs to URLs and insert them into a BloomFilter(m, k).
      3) Convert negative test IDs to URLs and query the Bloom filter.
      4) Count how many of these negative URLs are reported as True.
      5) Return the empirical false positive rate = (#false_positives / n_test).
    """
    # 1) Generate IDs
    train_ids = generate_train_ids(n_train)
    neg_ids = generate_negative_test_ids(n_train, n_test)

    # 2) Build Bloom filter and insert train URLs that are converted from train IDs
    ###################### TODO ####################
    # create BloomFilter(m, k) and insert all train URLs.
    bf = BloomFilter(m, k)

    for tid in train_ids:
        url = id_to_url(tid)
        bf.insert(url)
    ################################################

    # 3) Evaluate on negative URLs
    false_positives = 0
    ###################### TODO ####################
    # query the BloomFilter with all negative test URLs, compute the empirical fpr
    for nid in neg_ids:
        url = id_to_url(nid)
        if bf.query(url):
            false_positives += 1

    fpr = false_positives / n_test if n_test > 0 else 0.0
    ################################################
    return fpr 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1_000_000,
                        help="Bloom filter size (number of bits)")
    parser.add_argument("--n_train", type=int, default=100_000,
                        help="Number of URLs to insert")
    parser.add_argument("--n_test", type=int, default=100_000,
                        help="Number of negative test URLs")
    parser.add_argument("--k_max", type=int, default=12,
                        help="Max number of hash functions to sweep (1..k_max)")
    args = parser.parse_args()

    print(f"Single-machine Bloom filter experiment")
    print(f"m={args.m}, n_train={args.n_train}, n_test={args.n_test}")

    results = []
    for k in range(1, args.k_max + 1):
        fpr = run_single_experiment(
            m=args.m,
            k=k,
            n_train=args.n_train,
            n_test=args.n_test,
        )
        results.append((k, fpr))
        print(f"k={k:2d}  empirical FPR={fpr:.6f}")

    print("\nSummary (k, FPR):")
    for k, fpr in results:
        print(f"{k}\t{fpr:.8f}")


if __name__ == "__main__":
    main()