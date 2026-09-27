import sys
import re
import time
import numpy as np
from itertools import combinations

SEED = 0
NUM_HYPERPLANES = 10    # Number of hyperplanes to use for LSH
COSINE_DIST_THRESHOLD = 0.1  # Cosine distance threshold for candidate pairs


def preprocess_text(s):
    """
    Keep only lowercase alphabets and whitespace.
    Remove everything else.
    """
    s = s.lower()
    # delete all non-[a-z ] characters
    s = re.sub(r'[^a-z ]+', '', s)
    return s

def tokenize_words(s):
    """
    Tokenize the input string s into words based on whitespace.
    """
    return [word for word in s.split() if word]

def build_tfidf_matrix(docs_tokens):
    """
    Build the TF-IDF matrix for the given documents.
    TF : # of times term t appears in document d
    IDF: log(N / df_t) where df_t is the number of documents containing term t
    """
    N = len(docs_tokens)    # Number of documents

    # Build vocabulary and compute document frequencies
    vocab = set()
    for tokens in docs_tokens:
        vocab.update(tokens)
    vocab = sorted(vocab)
    V = len(vocab)  # Vocabulary size
    if V == 0:
        return vocab, np.zeros((N, 0))
    
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    df = np.zeros(V, dtype=int)
    for tokens in docs_tokens:
        if not tokens:
            continue
        unique_tokens = set(tokens)
        for term in unique_tokens:
            df[word2idx[term]] += 1

    # Compute IDF
    idf = np.log((N + 0.0) / df)

    # Build TF-IDF matrix
    tfidf_matrix = np.zeros((N, V))
    for doc_idx, tokens in enumerate(docs_tokens):
        if not tokens:
            continue

        # Compute TF
        tf = dict()
        for term in tokens:
            tf[term] = tf.get(term, 0) + 1

        for term, tf in tf.items():
            term_idx = word2idx[term]
            tfidf_matrix[doc_idx, term_idx] = tf * idf[term_idx]

        # Normalize the TF-IDF vector per document
        norm = np.linalg.norm(tfidf_matrix[doc_idx])
        if norm > 0:
            tfidf_matrix[doc_idx] /= norm

    return vocab, tfidf_matrix

def make_hyperplanes(dim):
    """
    Generate random hyperplanes for LSH.
    Each hyperplane is represented by a random normal vector.
    """
    np.random.seed(SEED)
    if dim == 0:
        return np.zeros((NUM_HYPERPLANES, 0))
    return np.random.randn(NUM_HYPERPLANES, dim)

def compute_signatures(tfidf, hyperplanes):
    """
    Compute the LSH signatures for all documents.
    Each signature is a binary vector indicating which side of each hyperplane the document lies on.
    """
    N = tfidf.shape[0]  # Number of documents

    signatures = np.zeros((N, NUM_HYPERPLANES), dtype=int)
    if tfidf.shape[1] == 0:
        return signatures
    dots = tfidf @ hyperplanes.T  # Shape: (N, NUM_HYPERPLANES)
    signatures = (dots >= 0).astype(int)
    return signatures

def cosine_similarity(sig1, sig2):
    """
    Compute the cosine similarity between two binary signatures.
    Cosine similarity = (A . B) / (||A|| * ||B||)
    """
    dot_product = float(np.dot(sig1, sig2))
    norm1 = float(np.linalg.norm(sig1))
    norm2 = float(np.linalg.norm(sig2))
    if norm1 == 0 or norm2 == 0:
        return 0.0  # If either vector is zero, define similarity as 0
    sim = dot_product / (norm1 * norm2)
    if sim < -1.0:
        sim = -1.0
    if sim > 1.0:
        sim = 1.0
    return sim

def cosine_distance(sig1, sig2):
    """
    Compute the cosine distance between two binary signatures.
    Cosine distance = 1 - (A . B) / (||A|| * ||B||)
    """
    sim = cosine_similarity(sig1, sig2)
    return 1.0 - sim


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

    # Tokenize texts
    docs_tokens = [tokenize_words(text) for text in texts]

    # Build TF-IDF matrix
    vocab, tfidf_matrix = build_tfidf_matrix(docs_tokens)

    # Generate hyperplanes
    hyperplanes = make_hyperplanes(len(vocab))

    # Compute LSH signatures
    signatures = compute_signatures(tfidf_matrix, hyperplanes)

    # LSH
    buckets = dict()
    for i in range(len(signatures)):
        key = tuple(signatures[i])
        if key not in buckets:
            buckets[key] = [i]
        else:
            buckets[key].append(i)

    # Candidate pairs
    candidate_pairs = set()
    for bucket in buckets.values():
        if len(bucket) >= 2:
            for i, j in combinations(bucket, 2):
                id1, id2 = article_ids[i], article_ids[j]
                if id1 < id2:
                    candidate_pairs.add((i, j))
                else:
                    candidate_pairs.add((j, i))

    # Compute cosine distances for candidate pairs
    verified_pairs = []
    for i, j in candidate_pairs:
        if len(tfidf_matrix[i]) == 0 or len(tfidf_matrix[j]) == 0:
            continue
        dist = cosine_distance(tfidf_matrix[i], tfidf_matrix[j])
        if dist < COSINE_DIST_THRESHOLD:
            verified_pairs.append((article_ids[i], article_ids[j], 1.0 - dist))

    verified_pairs.sort()

    # Output results
    for id1, id2, sim in verified_pairs:
        print(f"{id1}\t{id2}\t{sim:.6f}")

    end_time = time.time()
    # print(f"Elapsed Time: {end_time - start_time:.3f} sec")

if __name__ == "__main__":
    main()