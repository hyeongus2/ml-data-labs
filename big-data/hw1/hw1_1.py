import sys
import time
from itertools import combinations
from pyspark import SparkConf, SparkContext


def parse_line_to_pairs(line):
    """
    Parse a line into [(user, friend1), (user, friend2), ...]
    """
    parts = line.split('\t')
    if len(parts) != 2 or not parts[1]:
        return []
    user = int(parts[0])
    results = []
    for friend in parts[1].split(','):
        friend = int(friend)
        # Remove self-loops (user, user)
        if user != friend:
            results.append((user, friend))
    return results

def build_undirected_edges(directed):
    """
    From distinct directed edges (u,v), keep only mutual relationships as undirected edges.
    1. Map (u,v) -> ((min(u,v), max(u,v)), 1)
    2. Reduce by key and keep those with count==2 (both (u,v) and (v,u) exist)
    3. Return RDD[(u,v)] with u<v
    """
    ordered_directed = directed.map(lambda uv: ((uv[0] if uv[0] < uv[1] else uv[1], uv[1] if uv[0] < uv[1] else uv[0]), 1))
    undirected = (
        ordered_directed
        .reduceByKey(lambda x, y: x + y)
        .filter(lambda kv: kv[1] == 2)
        .keys()
    )
    return undirected

def build_adjacency(undirected):
    """
    Build adjacency list as (u, set_of_neighbors).
    Input: RDD[(u,v)] where u<v
    Output: RDD[(u, set_of_neighbors)]
    """
    # Expand to (u,v) and (v,u), then group
    adj_pairs = undirected.flatMap(lambda uv: [(uv[0], uv[1]), (uv[1], uv[0])])
    adjacency = adj_pairs.groupByKey().mapValues(lambda nbrs: set(nbrs))
    return adjacency

def main():
    input_path = sys.argv[1]
    conf = SparkConf()
    sc = SparkContext(conf=conf)

    start_time = time.time()

    # 1. Parse input -> directed edges, remove self-loops & duplicates
    directed = sc.textFile(input_path).flatMap(parse_line_to_pairs).distinct()

    # 2. Keep only mutual (undirected) edges (u,v) with u<v
    undirected = build_undirected_edges(directed)

    # 3. Build adjacency list (u, set_of_neighbors)
    adjacency = build_adjacency(undirected).persist()

    # Broadcast adjacency dict for fast checks
    adj_dict = dict(adjacency.collect())
    adj_broad = sc.broadcast(adj_dict)

    def edge_exists(u, v):
        """
        Check if edge (u,v) exists in the broadcast adjacency dict.
        """
        return v in adj_broad.value.get(u, ())

    # 4. For each vertex v with neighbors v_nbrs, generate all (u,w) pairs from v_nbrs.
    # Relationship: (u-v-w)
    # For each (u,w), emit ((u,w), v) if (u,w) is not an edge.
    # Result: RDD[((u,w), [v1, v2, ...])]
    uw_v = adjacency.flatMap(
        lambda v_nbrs: [
            (((u, w), v_nbrs[0]))  # ((u,w), v)
            for (u, w) in combinations(sorted(v_nbrs[1]), 2)  # for all pairs (u,w) in neighbors of v
            if not edge_exists(u, w)
        ]
    )

    # Group by (u,w) to get list of common neighbors vs
    uw_vs = uw_v.groupByKey().mapValues(lambda vs: sorted(set(vs)))

    # 5. From each (u,w) with common neighbors vs, create chordless squares (u-v1-w-v2-u)
    # Pick (v,x) in vs with v<x if (v,x) is not an edge.
    # Build a sorted tuple (u<v<w<x)
    squares = uw_vs.flatMap(
        lambda uw_vx: [
            tuple(sorted((uw_vx[0][0], v, uw_vx[0][1], x)))  # (u,v,w,x)
            for (v, x) in combinations(uw_vx[1], 2)
            if not edge_exists(v, x)
        ]
    ).distinct().persist()

    # 6. Print outputs
    # If total_cnt <= 10: print all squares (lexicographically ascending)
    # Else: print first 10 (lex asc) and last 10 squares (lex asc)
    total_cnt = squares.count()

    def tuple_to_line(t):
        return f"{t[0]}\t{t[1]}\t{t[2]}\t{t[3]}"

    if total_cnt <= 20:
        all_sorted = squares.sortBy(lambda t: t).collect()
        for t in all_sorted:
            print(tuple_to_line(t))
    else:
        # First 10
        head10 = squares.sortBy(lambda t: t).take(10)
        for t in head10:
            print(tuple_to_line(t))
        # Last 10
        tail10_desc = squares.sortBy(lambda t: t, ascending=False).take(10)
        tail10 = list(reversed(tail10_desc))
        for t in tail10:
            print(tuple_to_line(t))

    end_time = time.time()
    elapsed = end_time - start_time
    # print(f"Elapsed time : {elapsed:.3f} sec")

    sc.stop()

if __name__ == "__main__":
    main()