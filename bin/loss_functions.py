import numpy as np


def categorical_loss_entropy(actual, target):
    # Takes np.arrays as scalars and targets
    loss = -(target - np.log(actual))
    return(loss)


'''self.input_weights = np.random.rand(self.n_nodes_input, self.layers[1].n_nodes)
self.input_activation = np.zeros(shape = (self.n_nodes_input))

self.weight_cache = [self.input_weights]
self.bias_cache = []
self.activation_cache = [self.input_activation]

for l in range(1, len(layers)):
    current_layer = self.layers[l]

    try:
        next_layer = self.layers[l + 1]
    except:
        next_layer = current_layer

    layer_weights = np.random.rand(current_layer.n_nodes, next_layer.n_nodes)
    layer_bias = np.random.rand(next_layer.n_nodes)
    layer_activation = np.zeros(shape = (current_layer.n_nodes))

    self.weight_cache.append(layer_weights)
    self.bias_cache.append(layer_bias)
    self.activation_cache.append(layer_activation)

self.activation_cache.append(np.zeros(shape = self.layers[-1].n_nodes))
self.z_cache = self.activation_cache'''