import numpy as np

class Intializers():
    def __init__(self):
        pass

    def Xavier_initializer(self, size):
            in_dim = size[1]
            out_dim = size[0]
            xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
            return np.random.normal(0, xavier_stddev, size=(out_dim, in_dim))

    def random_initializer(self, size):
        in_dim = size[1]
        out_dim = size[0]
        return np.random.normal(0, 1, size=(out_dim, in_dim))


    def He_initializer(self,size):
        in_dim = size[1]
        out_dim = size[0]
        He_stddev = np.sqrt(2 / (in_dim))
        return np.random.normal(0, 1, size=(out_dim, in_dim)) * He_stddev
