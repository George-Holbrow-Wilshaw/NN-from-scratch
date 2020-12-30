import numpy as np
import activation_functions as af
import loss_functions as lf

class SequentialNetwork:

    def __init__(self, layers):
        self.layers = layers

    def compile(self, optimiser, loss_function, metrics):
        net = NeuralNetwork(layers = self.layers, 
                            optimiser = optimiser, 
                            loss_function = loss_function, 
                            metrics = metrics)
        return net

    def summary(self):
        for l in self.layers:
            print('%s Input shape = %s  Output shape = %s' %(l.name, l.input_shape, l.output_shape))


class NeuralNetwork:

    def __init__(self, layers, optimiser, loss_function, metrics):
        
        self.input_shape = layers[0].input_shape
        self.layers = layers
        self.n_layers = len(layers)
        self.n_layers_hidden = len(layers) - 1
        self.n_nodes_hidden = sum([l.n_nodes for l in layers[1:]])
        self.n_nodes_input = self.input_shape[0]
        self.n_nodes_output = layers[-1].output_shape
        self.input_layer = layers[0]
        self.output_layer = layers[-1]

        # Produce weights and activation caches 
        
        self.activation_cache = []
        self.bias_cache = []
        self.weight_cache = []

        for l in range(len(layers)):
            current_layer = layers[l]
            depth = l

            self.activation_cache.append(np.zeros(shape = (current_layer.n_nodes)))

            if (depth > 0):
                prev_layer = layers[l - 1]
                self.bias_cache.append(np.random.rand(current_layer.n_nodes))
                self.weight_cache.append(np.random.rand(prev_layer.n_nodes, current_layer.n_nodes))

        self.z_cache = self.activation_cache

    def forward_propogate(self, training_data):

        self.activation_cache[0] = training_data

        for l in enumerate(self.layers[:-1]):
            layer_n = l[0]
            layer = l[1]
            layer.call_layer(self.activation_cache, self.weight_cache, self.bias_cache, layer_n)

    def back_propogate(self, labels, activation_cache, z_cache, loss_function):
        # First we figure out the gradient loss function
        # This is defined as dC/dW = dZ/dW * dAdZ * dCdA
        weights_adjustment_cache = [None] * (self.n_layers - 1)
        bias_adjustment_cache = [None] * (self.n_layers - 1)

        deltas = (activation_cache[-1] - labels)* af.delta_sigmoid(z_cache[- 1])
        dZdW = activation_cache[- 2]
        dLdW = np.dot(deltas, dZdW)
        dLdB = deltas

        weights_adjustment_cache[-1] = (dLdW)
        bias_adjustment_cache[-1] = (dLdB)
        
        for i in range(self.n_layers - 1, 1, -1):

            deltas = np.dot(self.weight_cache[i - 1], deltas) * af.delta_sigmoid(z_cache[i - 1]).T
            dZdW = activation_cache[i - 2]
            dLdW = np.dot(deltas, dZdW)
            dLdB = deltas

            # Now we adjust our weights by a fraction of this gradient function (the gradient function tells us the direction and proportions we need to reduce the weights)
            # This is in effect moving down an N dimensional surface 
            # This reduction will be done in another function, for the mean time we will store the amounts by which we need to reduce our weights

            weights_adjustment_cache[i - 2] = dLdW
            bias_adjustment_cache[i - 2] = dLdB

        return weights_adjustment_cache, bias_adjustment_cache
        


    def produce_metrics(self):
        pass

    def fit(self, training_data, labels, validation_data, metrics, n_epochs, learning_rate):
        
        self.n_training_samples = len(training_data)
        
        # Store intermediate results of activator and activation function outputs
        self.a_hidden = np.zeros(shape = [self.n_training_samples, self.n_nodes_hidden, self.n_layers_hidden])
        
        for _ in range(n_epochs):
            for x in enumerate(training_data):
                y = labels[x[0]]
                self.forward_propogate([x[1]])
                weight_adjustments, bias_adjustments = self.back_propogate(y, self.activation_cache, self.z_cache, lf.categorical_loss_entropy)
                self.weight_cache = [w - w_adj.T * learning_rate for w, w_adj in zip(self.weight_cache, weight_adjustments)]
                self.bias_cache = [b - b_adj.T * learning_rate for b, b_adj in zip(self.bias_cache, bias_adjustments)]

        print(self.weight_cache)

        return(self)

    def predict(self, data):
        
        self.forward_propogate(data)
        print(self.activation_cache[-1])
    

class FullyConnectedLayer:

    def __init__(self, n_nodes, input_shape, activation_function):
        self.name = 'FullyConnectedLayer'
        self.input_shape = input_shape
        self.output_shape = 3
        self.n_nodes = n_nodes
        self.activation_function = activation_function

    def call_layer(self, data, weights, bias, layer_number = 0):
        layer_weights = weights[layer_number]
        layer_bias = bias[layer_number]
        layer_data = data[layer_number]

        z = np.dot(layer_data, layer_weights) + layer_bias
        a = self.activation_function(z)

        data[layer_number + 1] = a 
        
class ConvolutionLayer:

    def __init__(self, n_filters, kernel_size, input_shape, activation_function):
        self.name = 'ConvolutionLayer'
        self.input_shape = input_shape
        self.output_shape = 'tbd'
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.activation_function = activation_function

    def call_layer(self):
        pass


net = SequentialNetwork([
    FullyConnectedLayer(2, input_shape = (7, ), activation_function =  af.sigmoid),
    FullyConnectedLayer(3, input_shape = (3, ), activation_function =  af.sigmoid),
    FullyConnectedLayer(1, input_shape = (3, ), activation_function =  af.sigmoid)
])

net = net.compile(optimiser = 'SGD',
           loss_function = '',
           metrics = '')



# Testing section


train_data = np.array([[0, 0], [1, 0]])

model = net.fit(training_data = train_data,  
        labels = [1, 0], 
        validation_data = train_data, 
        metrics='bel', 
        n_epochs = 10000, 
        learning_rate = 5)
        
model.predict(train_data[1])