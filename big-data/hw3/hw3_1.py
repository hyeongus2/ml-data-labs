import sys
import re
import math
from pyspark import SparkContext, SparkConf

def main():
    """
    Main function to run the PageRank algorithm.
    """

    # 1. --- Spark Context Setup ---
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    # 2. --- Input File Handling ---
    if len(sys.argv) != 2:
        print("Usage: spark-submit hw3_1.py <input_file_path>", file=sys.stderr)
        sc.stop()
        exit(-1)

    input_path = sys.argv[1]

    # 3. --- Graph RDD Creation ---
    # Load dataset, ignore comments, parse lines, cast to int, and remove duplicates.
    links_rdd = sc.textFile(input_path) \
        .filter(lambda line: not line.startswith('#')) \
        .map(lambda line: re.split(r'\s+', line.strip())) \
        .filter(lambda parts: len(parts) == 2) \
        .map(lambda parts: (int(parts[0]), int(parts[1]))) \
        .distinct()  # (src, dst)

    # Create an adjacency list: (source_node, [list_of_dest_nodes])
    adj_list_rdd = links_rdd.groupByKey().mapValues(list).cache()

    # Extract all unique nodes (from both source and destination columns)
    all_nodes_rdd = links_rdd.flatMap(lambda x: [x[0], x[1]]).distinct()
    all_nodes = all_nodes_rdd.collect()
    N = len(all_nodes)

    # 4. --- Dead End Calculation ---
    # Dead ends: nodes that have no outgoing links.
    dead_ends_rdd = all_nodes_rdd.subtract(adj_list_rdd.map(lambda x: x[0]))
    dead_end_nodes = dead_ends_rdd.collect()
    dead_end_count = len(dead_end_nodes)
    dead_end_set = set(dead_end_nodes)

    # 5. --- Spider Trap Calculation ---
    #  - single node with self-loop only
    #  - pair of nodes pointing only to each other (no external out-links)

    # 5.1. Single-node traps (self-loop only, out-degree == 1)
    isolated_self_loops = adj_list_rdd.filter(
        lambda x: len(x[1]) == 1 and x[1][0] == x[0]
    ).map(lambda x: x[0])

    # 5.2. Pair traps
    def get_exclusive_partner(node_data):
        """
        Checks if a node has exactly one external neighbor (ignoring self-loops).
        Returns (node, partner) if true, otherwise None.
        """
        node, neighbors = node_data
        others = [n for n in neighbors if n != node]
        if len(others) == 1:
            return (node, others[0])
        else:
            return None

    candidate_links = adj_list_rdd.map(get_exclusive_partner) \
                                  .filter(lambda x: x is not None)

    # pair trap if (A,B) and (B,A) both exist
    pair_nodes = candidate_links.map(
        lambda x: (tuple(sorted((x[0], x[1]))), 1)
    ).reduceByKey(lambda a, b: a + b) \
     .filter(lambda x: x[1] == 2) \
     .flatMap(lambda x: x[0])

    all_trap_nodes = isolated_self_loops.union(pair_nodes) \
                                        .distinct() \
                                        .sortBy(lambda x: x) \
                                        .collect()

    # 6. --- PageRank Iteration Setup ---
    beta = 0.9
    teleport_component = (1.0 - beta) / N
    epsilon = 1e-5

    # PageRank vector is stored as a Python dict on the driver
    ranks = {node: 1.0 / N for node in all_nodes}

    # 7. --- Iteration Loop (Max 50 with L2 convergence check) ---
    for _ in range(50):
        # 7.0. Rank mass on dead-end nodes
        dead_ends_rank = 0.0
        for node in dead_end_set:
            dead_ends_rank += ranks[node]

        # Broadcast current ranks to workers
        ranks_bc = sc.broadcast(ranks)

        # 7.1. Contributions from non-dead-end nodes
        # (src, ([dst...])) -> (dst, beta * rank_src / out_degree)
        def contrib_func(item):
            src, dst_list = item
            r_dict = ranks_bc.value
            out_deg = len(dst_list)
            if out_deg == 0:
                return []
            rank_src = r_dict[src]
            if rank_src == 0.0:
                return []
            share = beta * rank_src / out_deg
            return [(dst, share) for dst in dst_list]

        contributions_rdd = adj_list_rdd.flatMap(contrib_func)

        # Aggregate contributions per destination node
        summed_contributions = contributions_rdd.reduceByKey(lambda a, b: a + b)
        contribs_dict = dict(summed_contributions.collect())

        # Unpersist broadcast variable (Not needed anymore)
        ranks_bc.unpersist()

        # 7.2. New ranks with teleportation + dead end mass redistribution
        base_component = teleport_component + (beta * dead_ends_rank / N)

        new_ranks = {}
        l2_sum = 0.0

        for node in all_nodes:
            old_val = ranks[node]
            contrib_val = contribs_dict.get(node, 0.0)
            new_val = base_component + contrib_val
            new_ranks[node] = new_val
            diff = new_val - old_val
            l2_sum += diff * diff

        l2_norm = math.sqrt(l2_sum)
        ranks = new_ranks

        if l2_norm < epsilon:
            break

    # 8. --- Format and Print Output ---

    # 8.1. Print Dead Ends Count
    print(dead_end_count)

    # 8.2. Print Spider Traps (ascending order)
    for node in all_trap_nodes:
        print(node)

    # 8.3. Print Top 10 PageRank Scores
    top_10 = sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:10]

    for (node, rank) in top_10:
        floored_rank = math.floor(rank * 100000) / 100000
        print(f"{node}\t{floored_rank:.5f}")

    # 9. --- Stop Spark ---
    sc.stop()

if __name__ == "__main__":
    main()
