import sys
import time
from itertools import combinations

SUPPORT_THRESHOLD = 100
CONFIDENCE_THRESHOLD = 0.5

def read_sessions(file_path):
    """
    Read sessions from the input file.
    Generator that yields sets of items per session.
    """
    with open(file_path, 'r') as f:
        for line in f:
            items = set(item for item in line.strip().split() if item)
            if items:
                yield items

def pass1_count_items(file_path):
    """
    Pass 1: Count support for each item.
    """
    item_supports = {}
    for session in read_sessions(file_path):
        for item in session:
            item_supports[item] = item_supports.get(item, 0) + 1
    return item_supports

def map_frequent_items(item_supports):
    """
    Reindex items that meet the support threshold.
    Return mappings from original items to new indices [0,1,...,m-1].
    """
    frequent_items = [item for item, support in item_supports.items() if support >= SUPPORT_THRESHOLD]
    frequent_items.sort()
    frequent_indices = {item: idx for idx, item in enumerate(frequent_items)}
    frequent_supports = {item: item_supports[item] for item in frequent_items}
    return frequent_items, frequent_indices, frequent_supports

def tri_index(i, j, m):
    """
    0 <= i < j < m
    Implement triangular matrix as a 1D array.
    Return corresponding index.
    """
    if i >= j or i < 0 or j >= m:
        raise ValueError("Invalid indices for triangular matrix.")
    return i * (2 * m - i - 1) // 2 + (j - i - 1)

def pass2_count_pairs(file_path, frequent_indices, m):
    """
    Pass 2: Count support for each item pair using triangular matrix.
    m: number of frequent items
    """
    pair_counts = [0] * (m * (m - 1) // 2)
    if m < 2:
        return pair_counts  # No pairs possible

    for session in read_sessions(file_path):
        # Map session items to frequent indices
        fi = [frequent_indices[item] for item in session if item in frequent_indices]
        if len(fi) < 2:
            continue
        fi.sort()
        # Generate unique pairs and count
        for i, j in combinations(fi, 2):
            idx = tri_index(i, j, m)
            pair_counts[idx] += 1

    return pair_counts

def pair_supports(pair_counts, frequent_items):
    """
    Generate (item1, item2, support) for pairs meeting the support threshold.
    """
    m = len(frequent_items)
    if m < 2:
        return  # No pairs possible

    idx = 0
    for i in range(m-1):
        for j in range(i+1, m):
            support = pair_counts[idx]
            if support >= SUPPORT_THRESHOLD:
                yield (frequent_items[i], frequent_items[j], support)
            idx += 1

def association_rules(frequent_pairs, item_supports):
    """
    Generate association rules from frequent item pairs.
    Return list of (A, B, confAB, supportAB) for rules A -> B.
    """
    rules = []
    for A, B, supportAB in frequent_pairs:
        supportA = item_supports[A]
        supportB = item_supports[B]
        # Rule A -> B
        confAB = supportAB / supportA if supportA > 0 else 0.0
        if confAB >= CONFIDENCE_THRESHOLD:
            rules.append((A, B, confAB, supportAB))
        # Rule B -> A
        confBA = supportAB / supportB if supportB > 0 else 0.0
        if confBA >= CONFIDENCE_THRESHOLD:
            rules.append((B, A, confBA, supportAB))
    return rules

def main():
    file_path = sys.argv[1]
    start_time = time.time()

    # Pass 1: Count individual item supports
    item_supports = pass1_count_items(file_path)
    frequent_items, frequent_indices, frequent_supports = map_frequent_items(item_supports)
    m = len(frequent_items)

    # Pass 2: Count item pairs
    pair_counts = pass2_count_pairs(file_path, frequent_indices, m)

    # Generate frequent item pairs
    frequent_pairs = list(pair_supports(pair_counts, frequent_items))
    num_frequent_pairs = len(frequent_pairs)

    # Association Rules
    rules = association_rules(frequent_pairs, frequent_supports)
    rules.sort(key=lambda x: (-x[2], -x[3], x[0], x[1]))  # Sort by confidence desc, support desc, A asc, B asc
    num_valid_rules = len(rules)

    # Output results
    print(num_frequent_pairs)
    print(num_valid_rules)

    # Top 10 rules
    top_k = 10 if num_valid_rules >= 10 else num_valid_rules
    for i in range(top_k):
        A, B, confAB, supportAB = rules[i]
        print(f"Rule: {A} -> {B}, Confidence: {confAB:.6f}, Support: {supportAB}")

    end_time = time.time()
    # print(f"Elapsed time: {end_time - start_time:.3f} sec")

if __name__ == "__main__":
    main()