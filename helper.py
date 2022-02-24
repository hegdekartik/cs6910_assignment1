import numpy as np

class OHE():
    def __init__(self):
        pass
    
    # One Hot Encoder function
    def oneHotEncode(self,num_classes, Y_train_raw):
        Ydata = np.zeros((num_classes, Y_train_raw.shape[0]))
        for i in range(Y_train_raw.shape[0]):
            value = Y_train_raw[i]
            Ydata[int(value)][i] = 1.0
        return Ydata
