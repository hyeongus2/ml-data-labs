# main_spark.py (FULL SOLUTION)

import argparse
from pyspark.sql import SparkSession
from main_single import id_to_url
from bloom_spark import (
    build_bloom_filter,
    evaluate_fpr,
    build_two_stage_cascade,
    evaluate_fpr_cascade,
)


###############################################################
# Spark session constructor
###############################################################

def create_spark(app_name="BloomFilterSpark"):
    return (
        SparkSession.builder
        .appName(app_name)
        .getOrCreate()
    )


###############################################################
# Data generation
###############################################################

def generate_train_and_negative_rdds(spark, n_train: int, n_test: int):
    sc = spark.sparkContext
    return sc.range(0, n_train), sc.range(n_train, n_train + n_test)


###############################################################
# (1) Sweep-k experiment
###############################################################

def sweep_k_experiment(spark, m, n_train, n_test, k_values, base_seed=0):
    """
    For each k in k_values:

        1) Build distributed Bloom filter
        2) Compute distributed FPR
        3) Store results as list (k, fpr)
    """
    rdd_train_ids, rdd_neg_ids = generate_train_and_negative_rdds(spark, n_train, n_test)
    rdd_train_ids = rdd_train_ids.cache()
    rdd_neg_ids = rdd_neg_ids.cache()

    results = []

    for k in k_values:
        print(f"Building Bloom filter for k={k}...")
        # Build distributed BF
        ###################### TODO ####################
        rdd_urls = rdd_train_ids.map(id_to_url)
        bits = build_bloom_filter(rdd_urls, m, k, base_seed)
        ################################################

        # Evaluate FPR
        ###################### TODO ####################
        fpr = evaluate_fpr(rdd_neg_ids, bits, m, k, base_seed)
        ################################################
        results.append((k, fpr))
        print(f"  k={k}, FPR={fpr:.6f}")

    return results


###############################################################
# (2) Cascade vs Dense (Bonus)
###############################################################

def cascade_vs_dense_experiment(
        spark, m_total, n_train, n_test,
        m1, k1, m2, k2, k_dense
):
    # Load RDDs
    rdd_train_ids, rdd_neg_ids = generate_train_and_negative_rdds(spark, n_train, n_test)
    rdd_train_ids = rdd_train_ids.cache()
    rdd_neg_ids = rdd_neg_ids.cache()

    # (A) Cascade build
    print("Building cascade Bloom filters...")
    ###################### TODO ####################
    bits1, bits2 = build_two_stage_cascade(
        rdd_train_ids,
        m1, k1,
        m2, k2
    )
    ################################################

    print("Evaluating cascade...")
    ###################### TODO ####################
    fpr_cascade = evaluate_fpr_cascade(
        rdd_neg_ids,
        bits1, m1, k1,
        bits2, m2, k2
    )
    ################################################

    # (B) Dense build
    print("Building dense Bloom filter...")
    ###################### TODO ####################
    rdd_urls = rdd_train_ids.map(id_to_url)
    bits_dense = build_bloom_filter(rdd_urls, m_total, k_dense)
    ################################################    

    print("Evaluating dense Bloom filter...")
    ###################### TODO ####################
    fpr_dense = evaluate_fpr(rdd_neg_ids, bits_dense, m_total, k_dense)
    ###################### TODO ####################

    return fpr_cascade, fpr_dense

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True,
                        choices=["sweep_k", "cascade"],
                        help="Which experiment to run.")

    parser.add_argument("--m", type=int, default=5_000_000,
                        help="Bloom filter size (for sweep_k)")

    parser.add_argument("--n_train", type=int, default=500_000)
    parser.add_argument("--n_test", type=int, default=500_000)

    parser.add_argument("--k_max", type=int, default=12,
                        help="Max hash functions for sweep_k")

    # Cascade settings
    parser.add_argument("--m1", type=int, default=2_500_000)
    parser.add_argument("--k1", type=int, default=5)
    parser.add_argument("--m2", type=int, default=2_500_000)
    parser.add_argument("--k2", type=int, default=5)
    parser.add_argument("--k_dense", type=int, default=12)

    args = parser.parse_args()

    spark = create_spark("BloomFilterSpark")

    if args.mode == "sweep_k":
        k_values = list(range(1, args.k_max + 1))
        results = sweep_k_experiment(
            spark,
            m=args.m,
            n_train=args.n_train,
            n_test=args.n_test,
            k_values=k_values
        )
        print("\n=== Final Sweep-k Results ===")
        for k, fpr in results:
            print(f"k={k}, FPR={fpr:.6f}")

    elif args.mode == "cascade":
        fpr_cascade, fpr_dense = cascade_vs_dense_experiment(
            spark,
            m_total=args.m1 + args.m2,
            n_train=args.n_train,
            n_test=args.n_test,
            m1=args.m1, k1=args.k1,
            m2=args.m2, k2=args.k2,
            k_dense=args.k_dense
        )
        print("\n=== Cascade vs Dense Results ===")
        print(f"Cascade FPR = {fpr_cascade:.6f}")
        print(f"Dense   FPR = {fpr_dense:.6f}")

    spark.stop()