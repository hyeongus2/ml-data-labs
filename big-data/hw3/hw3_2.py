import numpy as np
import csv
import sys

# Fix random seed for reproducibility
SEED = 42
np.random.seed(SEED)

def sigmoid(x):
    """Sigmoid activation function"""
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    """Derivative of sigmoid with respect to its input"""
    s = sigmoid(x)
    return s * (1.0 - s)

def one_hot(label, num_classes=10):
    """Convert integer label to one-hot vector"""
    v = np.zeros(num_classes)
    v[int(label)] = 1.0
    return v

def load_csv(path):
    """Load CSV file and return (data, one-hot labels)"""
    data = []
    labels = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            row = [float(x) for x in row]
            x = np.array(row[:-1], dtype=float)
            y = int(row[-1])
            data.append(x)
            labels.append(one_hot(y))
    return np.array(data, dtype=float), np.array(labels, dtype=float)


class Fully_Connected_Layer:
    def __init__(self, learning_rate):
        self.InputDim = 784
        self.HiddenDim = 128
        self.OutputDim = 10
        self.learning_rate = learning_rate
        
        '''Weight Initialization'''
        # Xavier initialization
        self.W1 = np.random.randn(self.InputDim, self.HiddenDim) * np.sqrt(1.0 / self.InputDim)
        self.W2 = np.random.randn(self.HiddenDim, self.OutputDim) * np.sqrt(1.0 / self.HiddenDim) 
        
    def Forward(self, Input):
        '''Implement forward propagation'''
        # Ensure Input is 2D: (N, 784)
        X = Input
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self.Input = X                            # (N, 784)
        self.z1 = self.Input @ self.W1           # (N, 128)
        self.h = sigmoid(self.z1)                # (N, 128)
        self.z2 = self.h @ self.W2               # (N, 10)
        Output = sigmoid(self.z2)                # (N, 10)
        self.Output = Output
        return Output

    def Backward(self, Label, Output):
        '''Implement backward propagation (MSE loss + sigmoid)'''
        # Ensure Label is 2D: (N, 10)
        Y = Label
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        X = self.Input
        N = X.shape[0]
        
        # Mean squared error loss:
        # L = (1 / (2N)) * sum_n sum_k (o_k - y_k)^2
        diff = Output - Y                        # (N, 10)
        
        # Output layer delta: (o - y) * sigma'(z^2)
        delta2 = diff * sigmoid_prime(self.z2)   # (N, 10)
        # Gradient for W2: h^T @ delta2 / N
        dW2 = self.h.T @ delta2 / float(N)       # (128, 10)
        
        # Hidden layer delta: (delta2 @ W2^T) * sigma'(z^1)
        delta1 = (delta2 @ self.W2.T) * sigmoid_prime(self.z1)  # (N, 128)
        # Gradient for W1: X^T @ delta1 / N
        dW1 = self.Input.T @ delta1 / float(N)   # (784, 128)
        
        # Gradient descent update
        self.W2 -= self.learning_rate * dW2
        self.W1 -= self.learning_rate * dW1
        
    def Train(self, Input, Label):
        Output = self.Forward(Input)
        self.Backward(Label, Output)


def accuracy(Network, data, label):
    """Compute accuracy of the network on given dataset"""
    Output = Network.Forward(data)              # (N, 10)
    pred = np.argmax(Output, axis=1)
    true = np.argmax(label, axis=1)
    return float(np.mean(pred == true))


# Parse command-line arguments for dataset paths
if len(sys.argv) != 3:
    # Problem statement assumes correct usage; exit silently on wrong usage
    sys.exit(0)

train_path = sys.argv[1]
test_path = sys.argv[2]

# Load training and testing data
train_data, train_label = load_csv(train_path)
test_data, test_label = load_csv(test_path)

# Hyperparameters (full-batch gradient descent)
learning_rate = 0.5  # η
iteration = 2000  # number of full-batch updates (epochs)

'''Construct a fully-connected network'''        
Network = Fully_Connected_Layer(learning_rate)

'''Train the network for the number of iterations'''
'''Implement function to measure the accuracy'''
for i in range(iteration):
    # Full-batch gradient descent: use all training samples in each iteration
    Network.Train(train_data, train_label)

# Measure accuracy on training and test data
train_acc = accuracy(Network, train_data, train_label)
test_acc = accuracy(Network, test_data, test_label)

# Print results in the exact required format:
# <TRAINING ACCURACY>
# <TEST ACCURACY>
# <NUMBER OF ITERATIONS>
# <η>
print(f"{train_acc:.3f}")
print(f"{test_acc:.3f}")
print(iteration)
print(learning_rate)
