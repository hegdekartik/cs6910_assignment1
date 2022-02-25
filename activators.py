import numpy as np


def sigmoid(n):
    value = 1 + np.exp(-(n))
    return 1.0 / value


def tanh(n):
    return np.tanh(n)


def sin(n):
    return np.sin(n)


def relu(n):
    value = max(0.0, n)
    # (n>0)*(n) + ((n<0)*(n)*0.01)
    return value

def softmax(n):
    value = np.exp(n) / np.sum(np.exp(n) , axis = 0)
    return value


def sigmoid_derivative(n):
    x = sigmoid(n)
    value = (x)*(1 - x)
    return value

def tanh_derivative(n):
    return 1 - np.tanh(n) ** 2


def relu_derivative(n):
    return (n>0)*np.ones(n.shape) + (n<0)*(0.01*np.ones(n.shape) )

def softmax_derivative(n):
    x = softmax(n)
    mat = np.tile(x, x.shape[0])
    value = np.diag(x.reshape(-1,)) - (mat*mat.T)
    return value