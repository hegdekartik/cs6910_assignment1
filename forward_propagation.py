import numpy as np
from activators import *

class fp():
    def __init__(self):
        pass
    def forwardPropagate(self, X_train_batch, weights, biases,FFNN):
        """
        Returns the neural network given input data, weights, biases.
        Arguments:
                    : X_train_batch - input matrix
                    : Weights  - Weights matrix
                    : biases - Bias vectors 
        """
        # Number of layers = length of weight matrix + 1
        num_layers = len(weights) + 1
        # A - Preactivations
        # H - Activations
        X = X_train_batch
        H = {}
        A = {}
        H["0"] = X
        A["0"] = X
        for l in range(0, num_layers - 2):
            if l == 0:
                W = weights[str(l + 1)]
                b = biases[str(l + 1)]
                A[str(l + 1)] = np.add(np.matmul(W, X), b)
                H[str(l + 1)] = FFNN.activation(A[str(l + 1)])
            else:
                W = weights[str(l + 1)]
                b = biases[str(l + 1)]
                A[str(l + 1)] = np.add(np.matmul(W, H[str(l)]), b)
                H[str(l + 1)] = FFNN.activation(A[str(l + 1)])

        # Here the last layer is not activated as it is a regression problem
        W = weights[str(num_layers - 1)]
        b = biases[str(num_layers - 1)]
        A[str(num_layers - 1)] = np.add(np.matmul(W, H[str(num_layers - 2)]), b)
        # Y = softmax(A[-1])
        Y = softmax(A[str(num_layers - 1)])
        H[str(num_layers - 1)] = Y
        return Y, H, A