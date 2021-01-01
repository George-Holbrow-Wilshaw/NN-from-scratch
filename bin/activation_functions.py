import numpy as np



def sigmoid(x, prime = False):
    if not prime:
        s = 1 / (1 + (np.exp(x) ** - 1))
    else:
        s = sigmoid(x) * (1 - sigmoid(x))
    return s


def relu(x, prime = False):
    if not prime:
        r = np.maximum(0, x)
        return r
    else:
        r = np.array([[1 if xi > 0 else 0 for xi in x[0]]])
        return r

def softmax(x, prime = False):
    s = np.exp(x - np.max(x))
    s = s / s.sum()
    if prime:
        s = s.reshape(-1,1)
        s = np.diagflat(s) - np.dot(s, s.T)
    return s
