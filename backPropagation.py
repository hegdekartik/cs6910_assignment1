import numpy as np

class bp():
    def __init__(self):
        pass
    def backPropagate(self, Y, H, A, Y_train_batch, FFNN,weight_decay=0):

        ALPHA = weight_decay
        gradients_weights = []
        gradients_biases = []
        num_layers = len(FFNN.layers)

        # Gradient with respect to the output layer is absolutely fine.
        if FFNN.loss_function == "CROSS":
            globals()["grad_a" + str(num_layers - 1)] = -(Y_train_batch - Y)
        elif FFNN.loss_function == "MSE":
            globals()["grad_a" + str(num_layers - 1)] = np.multiply(
                2 * (Y - Y_train_batch), np.multiply(Y, (1 - Y))
            )

        for l in range(num_layers - 2, -1, -1):

            if ALPHA != 0:
                globals()["grad_W" + str(l + 1)] = (
                    np.outer(globals()["grad_a" + str(l + 1)], H[str(l)])
                    + ALPHA * FFNN.weights[str(l + 1)]
                )
            elif ALPHA == 0:
                globals()["grad_W" + str(l + 1)] = np.outer(
                    globals()["grad_a" + str(l + 1)], H[str(l)]
                )
            globals()["grad_b" + str(l + 1)] = globals()["grad_a" + str(l + 1)]
            gradients_weights.append(globals()["grad_W" + str(l + 1)])
            gradients_biases.append(globals()["grad_b" + str(l + 1)])
            if l != 0:
                globals()["grad_h" + str(l)] = np.matmul(
                    FFNN.weights[str(l + 1)].transpose(),
                    globals()["grad_a" + str(l + 1)],
                )
                globals()["grad_a" + str(l)] = np.multiply(
                    globals()["grad_h" + str(l)], FFNN.der_activation(A[str(l)])
                )
            elif l == 0:

                globals()["grad_h" + str(l)] = np.matmul(
                    FFNN.weights[str(l + 1)].transpose(),
                    globals()["grad_a" + str(l + 1)],
                )
                globals()["grad_a" + str(l)] = np.multiply(
                    globals()["grad_h" + str(l)], (A[str(l)])
                )
        return gradients_weights, gradients_biases