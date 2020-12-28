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

        # We need to initialise the weights and biases for our NN
        self.init_weights = np.random.rand(self.n_nodes_input, self.n_nodes_hidden)
        self.init_bias = np.random.rand(self.n_nodes_hidden)

        self.hidden_weights = np.random.rand(self.n_nodes_hidden, self.n_nodes_hidden, self.n_layers_hidden - 1)
        self.hidden_bias = np.random.rand(self.n_nodes_hidden, self.n_layers_hidden - 1)

        self.output_weights = np.random.rand(self.n_nodes_hidden, self.n_nodes_output)
        self.output_bias = np.random.rand(self.n_nodes_output)

    def forward_propogate(self, training_data, a_hidden):

        data = training_data

        for l in enumerate(self.layers):
            if l[0] == 0:
                 l[1].call_layer(data, self.init_weights, self.init_bias)
            elif l[1] < (self.n_layers - 1):
                 l[1].call_layer(a_hidden, self.hidden_weights, self.hidden_bias)
            else:
                 l[1].call_layer(a_hidden, self.output_weights, self.output_bias)


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

        # Print output answer
        print(a[:, :, 1])

    def predict(self):
        pass
    

class FullyConnectedLayer:

    def __init__(self, n_nodes, input_shape, activation_function):
        self.name = 'FullyConnectedLayer'
        self.input_shape = input_shape
        self.output_shape = 3
        self.n_nodes = n_nodes
        self.activation_function = activation_function

    def call_layer(self, data, weights, bias, z_hidden, activation_function, layer_number = 0):

        z = np.dot(data, weights) + bias
        a = getattr(af.ActivationFunction, activation_function)(z)

        weights[:, :, layer_number] = z
        data[:, :, layer_number] = a



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
    FullyConnectedLayer(16, input_shape = (3, ), activation_function =  'relu'),
    FullyConnectedLayer(16, input_shape = (3, ), activation_function =  'relu')
])

net.compile(optimiser = 'adam',
           loss_function = 'cross_entropy',
           metrics = 'accuracy')