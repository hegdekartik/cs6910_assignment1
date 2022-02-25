import numpy as np

class prediction():
    def __init__(self):
        pass
    def predict(self,X,length_dataset,FFNN):
        Y_pred = []        
        for i in range(length_dataset):

            Y, H, A = FFNN.fp.forwardPropagate(
                X[:, i].reshape(FFNN.img_flattened_size, 1),
                FFNN.weights,
                FFNN.biases,
                FFNN
            )

            Y_pred.append(Y.reshape(FFNN.num_classes,))
        Y_pred = np.array(Y_pred).transpose()
        return Y_pred