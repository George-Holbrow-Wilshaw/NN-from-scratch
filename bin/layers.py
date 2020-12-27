class NeuralNetwork:

    def __init__(self, layers, learning_rate, epochs, batch_size):
        pass

    def compile(self, optimiser, loss_function):
        pass

    def summary(self):
        pass

    def back_propogate(self):
        pass

    def calc_loss(self):
        pass

    def produce_metrics(self):
        pass
    
    def fit(self):
        pass

    def predict(self):
        pass
    

class NetworkLayer(NeuralNetwork):

    def __init__(self):
        pass


class ConvolutionLayer(NetworkLayer):

    def __init__(self, filters, kernel_size, input_shape, activation_function):
        pass

    def build_layer(self):
        pass


class FullyConnected(NetworkLayer):

    def __init__(self, units, input_shape, activation_function):
        pass

    def build_layer(self):
        pass