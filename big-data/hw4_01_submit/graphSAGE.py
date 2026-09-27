# graphSAGE.py

import numpy as np
import scipy.sparse as sp
# DO NOT IMPORT ANY OTHER LIBRARIES


class GraphSAGE:
    """
    Two-layer GraphSAGE with mean aggregation and concatenation.

      H0 = X          
      M0 = A_norm @ H0
      H0_cat = [H0 || M0]
      H1 = ReLU(H0_cat @ W0)

      M1 = A_norm @ H1
      H1_cat = [H1 || M1]
      logits = H1_cat @ W1
      probs  = softmax(logits)
    """

    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_nodes=None, use_embedding=False):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        ###################### TODO ####################
        # (i) Weight Initialization

        limit0 = np.sqrt(1.0 / hidden_dim)
        size0 = (2 * input_dim, hidden_dim)         # tuple of 2 ints

        limit1 = np.sqrt(1.0 / output_dim)
        size1 = (2 * hidden_dim, output_dim)        # tuple of 2 ints
        ################################################

        self.W0 = np.random.uniform(
            -limit0, limit0, size=size0
        )

        self.W1 = np.random.uniform(
            -limit1, limit1, size=size1
        )

        # caches for backprop
        self.H0 = None
        self.M0 = None
        self.H0_cat = None
        self.pre_H1 = None
        self.H1 = None
        self.M1 = None
        self.H1_cat = None
        self.pre_Z = None
        self.probs = None

    @staticmethod
    def softmax(self, logits):
        pass

    @staticmethod
    def softmax(logits):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def forward(self, X, A_norm):
        """
        X      : (N, d_in) csr or dense
        A_norm: (N, N) csr, row-normalized adjacency
        """
        # dense features
        if sp.isspmatrix(X):
            X0 = X.toarray()
        else:
            X0 = np.asarray(X)

        self.H0 = X0

        ###################### TODO ####################
        # (ii) Two-layer GraphSAGE forward pass
        # ----- graphSAGE Layer 0 -----
        self.M0 = A_norm @ self.H0               # Average neighbor embeddings
        self.H0_cat = np.concatenate([self.H0, self.M0], axis=1)           # Concat self and neighbor embeddings
        self.pre_H1 = self.H0_cat @ self.W0           # Apply a linear transformation
        self.H1 = np.maximum(0, self.pre_H1)               # Nonlinearity

        # ----- graphSAGE Layer 1 -----
        self.M1 = A_norm @ self.H1               # Average neighbor embeddings
        self.H1_cat = np.concatenate([self.H1, self.M1], axis=1)           # Concat self and neighbor embeddings
        self.pre_Z = self.H1_cat @ self.W1            # Apply a linear transformation
        ################################################
        self.probs = self.softmax(self.pre_Z)                  
        return self.probs

    def loss_accuracy_macrof1(self, X, Y_onehot, A_norm, mask):
        probs = self.forward(X, A_norm)
        N = mask.sum()
        log_probs = np.log(np.clip(probs, 1e-12, 1.0))
        loss = -np.sum(Y_onehot[mask] * log_probs[mask]) / N

        pred = np.argmax(probs[mask], axis=1)
        true = np.argmax(Y_onehot[mask], axis=1)
        num_classes = Y_onehot.shape[1]
        acc = np.mean(pred == true)

        f1_list = []
        for c in range(num_classes):
            ###################### TODO ####################
            # (iv) Evaluation metric
            pred_c = (pred == c)
            true_c = (true == c)
            
            tp = np.sum(pred_c & true_c)
            fp = np.sum(pred_c & ~true_c)
            fn = np.sum(~pred_c & true_c)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            ################################################
            f1_list.append(f1)
            
        macro_f1 = float(np.mean(f1_list))

        return float(loss), float(acc), macro_f1

    def backprop(self, X, Y_onehot, A_norm, mask):
        """
        Returns:
            dW0, dW1, dEmb  (dEmb is None if use_embedding=False)
        """
        N_lab = mask.sum()

        # ----- gradient wrt logits -----
        dpre_Z = np.zeros_like(self.probs)  # (N, C)
        ###################### TODO ####################
        # (iii) Backpropagation and weight update
        # Use cached:
        #   H0, M0, H0_cat, pre_H1, H1, M1, H1_cat, pre_Z, probs
        dpre_Z[mask] = (self.probs - Y_onehot)[mask] / N_lab # (Hint) Question (a-i)

        # ----- layer 1 -----
        dW1 = self.H1_cat.T @ dpre_Z        # (Hint) Question (a-ii)                         
        dH1_cat = dpre_Z @ self.W1.T        # (Hint) Question (a-iii)                         

        dH1_direct = dH1_cat[:, :self.hidden_dim]   # (Hint) Question (a-iv)                    
        dM1 = dH1_cat[:, self.hidden_dim:]          # (Hint) Question (a-iv)

        dH1_agg = A_norm.T @ dM1            # (Hint) Question (a-iv)                     

        dH1_total = dH1_direct + dH1_agg    # (Hint) Question (a-iv)
        dpre_H1 = dH1_total * (self.pre_H1 > 0)      # (Hint) Question (a-v)
        
        # ----- layer 0 -----
        dW0 = self.H0_cat.T @ dpre_H1          # (Hint) Question (a-vi)
        ################################################

        return dW0, dW1

    def weight_update(self, dW0, dW1, lr):
        """
        SGD update for W0, W1 and (optionally) emb.
        """
        ###################### TODO ####################
        # (iii) Backpropagation and weight update
        self.W0 = self.W0 - lr * dW0
        self.W1 = self.W1 - lr * dW1
        ################################################

        return None

    def grad_descent_step(self, X, Y_onehot, A_norm, mask, lr):
        """
        One full-graph gradient descent step.
        """
        _ = self.forward(X, A_norm)
        dW0, dW1 = self.backprop(X, Y_onehot, A_norm, mask)
        self.weight_update(dW0, dW1, lr)