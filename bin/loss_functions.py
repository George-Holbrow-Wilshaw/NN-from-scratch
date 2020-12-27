import numpy as np


class LossFunction:

    def categorical_loss_entropy(self, actual, target):
        # Takes np.arrays as scalars and targets
        loss = -(target - np.log(actual))
        return(loss)
