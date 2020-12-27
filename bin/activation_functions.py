import numpy as np

class ActivationFunction:

    def sigmoid(x):
        s = 1 / (1 + (np.exp(x) ** - 1))
        return(s)

    def relu(x):
        r = max(0, x)

    def softmax(x):
        pass




