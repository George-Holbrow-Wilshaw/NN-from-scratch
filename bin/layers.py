import numpy as np
import activation_functions as af


class SequentialNetwork:

    def __init__(self, layers):
        self.layers = layers

    def compile(self, optimiser, loss_function, metrics):
        net = NeuralNetwork(layers = self.layers, 
                            optimiser = optimiser, 
                            loss_function = loss_function, 
                            metrics = metrics)
        return(net)

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

        # Produce weights and activation caches 
        
        self.input_weights = np.random.rand(self.n_nodes_input, self.layers[1].n_nodes)
        self.input_bias = np.random.rand(self.layers[2].n_nodes)
        self.input_activation = np.zeros(shape = (self.n_nodes_input))

        self.weight_cache = [self.input_weights]
        self.bias_cache = [self.input_bias]
        self.activation_cache = [self.input_activation]

        for l in range(1, len(layers)):
            current_layer = self.layers[l]

            try:
                next_layer = self.layers[l + 1]
            except:
                next_layer = current_layer

            layer_weights = np.random.rand(current_layer.n_nodes, next_layer.n_nodes)
            layer_bias = np.random.rand(current_layer.n_nodes)
            layer_activation = np.zeros(shape = (current_layer.n_nodes))

            self.weight_cache.append(layer_weights)
            self.bias_cache.append(layer_bias)
            self.activation_cache.append(layer_activation)
        
        self.activation_cache.append(np.zeros(shape = self.layers[-1].n_nodes))

    def forward_propogate(self, training_data, a_hidden):

        self.activation_cache[0] = training_data

        for l in enumerate(self.layers):
            layer_n = l[0]
            layer = l[1]
            layer.call_layer(self.activation_cache, self.weight_cache, self.bias_cache, layer_n)


    def back_propogate(self):
        pass

    def calc_loss(self):
        pass

    def produce_metrics(self):
        pass

    def fit(self, training_data, validation_data, metrics):
        
        n_training_samples = len(training_data)
        
        # Store intermediate results of activator and activation function outputs
        self.a_hidden = np.zeros(shape = [n_training_samples, self.n_nodes_hidden, self.n_layers_hidden])

        # Propogate through each layer
        self.forward_propogate(training_data, self.a_hidden)

        print(self.activation_cache)

    def predict(self):
        pass
    

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
        a = getattr(af.ActivationFunction, self.activation_function)(None, z)

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
    FullyConnectedLayer(7, input_shape = (7, ), activation_function =  'sigmoid'),
    FullyConnectedLayer(16, input_shape = (3, ), activation_function =  'sigmoid'),
    FullyConnectedLayer(16, input_shape = (3, ), activation_function =  'sigmoid')
])

net = net.compile(optimiser = 'adam',
           loss_function = 'cross_entropy',
           metrics = 'accuracy')


train_data = np.array([0, 1, 1, 0, 0, 1, 0])

net.fit(training_data = train_data, validation_data = train_data, metrics='bel')