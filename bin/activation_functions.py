import numpy as np



def sigmoid(x):
    s = 1 / (1 + (np.exp(x) ** - 1))
    return s

def delta_sigmoid(x):
    d_s = sigmoid(x) * (1 - sigmoid(x))
    return d_s

def relu(x):
    r = np.maximum(0, x)
    return r

def softmax(self, x):
    pass




