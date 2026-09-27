import numpy as np
import sys
import re
import math


# Graph class
class Graph:
    
    # Store the graph as set of adjacency lists of each nodes
    # Assume undirected graph
    # For example, if we have node 1, node 2 and edge (1, 2), (1, 3),
    # adj_list = {1: [2, 3], 2: [1], 3: [1]} 
    def __init__(self):
        self.adj_list = {}
    
    # Add an edge to the graph (i.e., update adj_list)
    def add_edge(self, u, v):
        """Add an undirected edge (u, v) to the adjacency list."""
        # Ensure both endpoints exist in the adjacency list
        if u not in self.adj_list:
            self.adj_list[u] = []
        if v not in self.adj_list:
            self.adj_list[v] = []
        # Undirected graph: add v to u's list and u to v's list
        self.adj_list[u].append(v)
        self.adj_list[v].append(u)

    # Return the neighbors of a node
    def neighbors(self, node):
        """Return sorted neighbors of a given node (empty list if none)."""
        return sorted(self.adj_list.get(node, []))


# This random_walk funtion is a hint 
# Refer to this function to implement the node2vec_walk function below
def random_walk(graph, start, length=5):
    walk = [start]
    for _ in range(length - 1):
        neighbors = graph.neighbors(walk[-1])
        if not neighbors:
            break
        walk.append(np.random.choice(neighbors))
    return walk



# Implement node2vec algorithm
# The length of the walk is fixed to 5
# When sampling next node, visit the node with highest probability
# If multiple nodes have the same probability, visit the node with the smallest index
def node2vec_walk(graph, start, length=5, p=1.0, q=1.0):
    walk = [start]
    
    # Repeat until the walk length reaches the target length
    for _ in range(length - 1):
        curr_node = walk[-1]
        curr_neighbors = graph.neighbors(curr_node)
        
        if not curr_neighbors:
            break
            
        # Case 1: First step (no previous node)
        if len(walk) == 1:
            # Treat all neighbors as equal weight
            # Tie-breaking: choose the neighbor with the smallest index.
            # Since curr_neighbors is sorted, the first element is the smallest.
            next_node = curr_neighbors[0]
            walk.append(next_node)
            continue
            
        # Case 2: Subsequent steps (t-1 -> t -> x)
        prev_node = walk[-2]
        best_node = -1
        max_weight = -1.0
        
        # Iterate neighbors in increasing order (deterministic tie-breaking)
        for neighbor in curr_neighbors:
            weight = 0.0
            
            if neighbor == prev_node:
                # Case: Back to previous node (d_tx = 0)
                weight = 1.0 / p
            elif neighbor in graph.neighbors(prev_node):
                # Case: Connected to previous node (d_tx = 1)
                weight = 1.0
            else:
                # Case: Not connected to previous node
                weight = 1.0 / q
            
            # Find argmax
            # Since we iterate sorted neighbors, strict > ensures we keep the smallest index on ties
            if weight > max_weight:
                max_weight = weight
                best_node = neighbor
        
        walk.append(best_node)
        
    return walk


# Train W1, W2 matrices using Skip-Gram
# The window size of fixed to 2, which means you should check each 2 nodes before and after the center node.
# Repeat the training process for 3 epochs, with learning rate 0.01
# Use softmax function when computing the loss
def train_skipgram(walks, n_nodes, dim=128, lr=0.01, window=2, epochs=3):
    # Mapping: node i -> index i-1 (0-based indexing)
    W1 = np.random.randn(n_nodes, dim)
    W2 = np.random.randn(dim, n_nodes)
    
    for _ in range(epochs):
        for walk in walks:
            # Iterate through the walk (time-order)
            for i, u in enumerate(walk):
                # u is the center word (target)
                # Convert 1-based node index to 0-based matrix index
                u_idx = u - 1
                
                # Define window range
                start_idx = max(0, i - window)
                end_idx = min(len(walk), i + window + 1)
                
                # Iterate context nodes (v) in increasing time-order
                for j in range(start_idx, end_idx):
                    if i == j:
                        continue
                    
                    v = walk[j]
                    v_idx = v - 1 # 0-based index for context (label)
                    
                    # --- Forward Propagation ---
                    # Hidden layer: z = W1[u_idx] (Input is one-hot, so just lookup)
                    z = W1[u_idx]  # Shape: (dim,)
                    
                    # Output layer (Scores): s = z @ W2
                    # Shape: (dim,) @ (dim, n_nodes) -> (n_nodes,)
                    scores = np.dot(z, W2)
                    
                    # Softmax: exp(s) / sum(exp(s))
                    # Subtract max for numerical stability
                    exp_scores = np.exp(scores - np.max(scores))
                    # Shape: (n_nodes,)
                    probs = exp_scores / np.sum(exp_scores)
                    
                    # --- Backward Propagation ---
                    # Error: y_pred - y_true
                    # y_true is one-hot vector for v_idx
                    error = probs
                    error[v_idx] -= 1.0  # Shape: (n_nodes,)
                    
                    # Gradient for W2: z.T * error
                    # Shape: (dim, 1) * (1, n_nodes) -> (dim, n_nodes)
                    grad_W2 = np.outer(z, error)
                    
                    # Gradient for W1 (for row u_idx): error * W2.T
                    # Shape: (1, n_nodes) * (n_nodes, dim) -> (1, dim)
                    grad_W1_row = np.dot(error, W2.T)
                    
                    # --- Update Weights ---
                    W2 -= lr * grad_W2
                    W1[u_idx] -= lr * grad_W1_row
    
    return W1


# You can freely define your functions/classes if you want
def your_function():
    pass


# Main function
def main():

    # Don't change this code
    # This will guarantee the same output when we test your code
    np.random.seed(1116)


    # Create graph
    graph = Graph()


    # Edges list
    # Note that the edges are undirected, and node idx starts with 1
    # ex. edges = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)]  
    edges = []


    # Parse edges from the command line file path ======================
    # Implement your code here
    file_path = sys.argv[1]
    max_node_id = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            # Skip comments or empty lines if any
            if line.strip().startswith('['): 
                continue 
            parts = re.split(r'\s+', line.strip())
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                edges.append((u, v))
                max_node_id = max(max_node_id, u, v)

    # ====================================================================
    

    # Update graph
    for edge in edges:
        graph.add_edge(*edge)

    # Define the target nodes to print embeddings for
    target_nodes = [228, 102, 500, 73, 991]

    pq_pairs = [(1.0, 1.0), (2.0, 0.5), (0.5, 2.0)]
    for p, q in pq_pairs:
        np.random.seed(1116) # Don't change this line
        

        # You can freely change and implement the code here ==================

        # 1. Generate Walks
        # We iterate through sorted node keys to ensure deterministic order of walk generation
        sorted_nodes = sorted(graph.adj_list.keys())
        walks = [node2vec_walk(graph, node, p=p, q=q) for node in sorted_nodes]
        
        # 2. Train Skip-Gram
        # max_node_id is used as n_nodes because indices are 1 to max_node_id
        # The weight matrix will have size max_node_id (mapping 1..N to 0..N-1)
        final_embeddings = train_skipgram(walks, n_nodes=max_node_id, dim=128, lr=0.01, window=2, epochs=3)

        # 3. Print Results
        print(f"p:{p}, q:{q}")
        for node_id in target_nodes:
            # Map node_id (1-based) to matrix index (0-based)
            idx = node_id - 1
            
            # Get the first element of the embedding
            val = final_embeddings[idx][0]
            
            # Truncate to 5 decimal places
            truncated_val = math.floor(val * 100000) / 100000
            print(f"{truncated_val:.5f}")

    # ====================================================================


if __name__ == "__main__":
    main()
