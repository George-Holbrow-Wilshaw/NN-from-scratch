import numpy as np

class ActivationFunction:

    def sigmoid(self, x):
        s = 1 / (1 + (np.exp(x) ** - 1))
        return(s)

    def relu(self, x):
        r = max(0, x)
        return(r)

    def softmax(self, x):
        pass




