import numpy as np

def categorical_loss_entropy(actual, target):
    # Takes np.arrays as scalars and targets
    loss = -(target - np.log(actual))
    return(loss)
