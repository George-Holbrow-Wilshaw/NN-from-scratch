net = SequentialNetwork([
    FullyConnectedLayer(784, input_shape = (7, ), activation_function =  af.sigmoid),
    FullyConnectedLayer(120, input_shape = (3, ), activation_function =  af.relu),
    FullyConnectedLayer(28, input_shape = (3, ), activation_function =  af.relu),
    FullyConnectedLayer(10, input_shape = (3, ), activation_function =  af.softmax)
])

net = net.compile(optimiser = 'SGD',
           loss_function = '',
           metrics = '')

from keras.datasets import mnist
(train_X, train_y), (test_X, test_y) = mnist.load_data()
training_data = train_X[:10000, :, :].reshape(1000, 784) / 255.0
train_y = train_y[:1000]

test_data = test_X[:10000, :, :].reshape(10000, 784) / 255.0
test_labels = test_y[:10000]


# We need to one-hot encode our training labels

training_labels = np.zeros((train_y.shape[0], train_y.max()+1), dtype=np.float32)
training_labels[np.arange(train_y.shape[0]), train_y] = 1

model = net.fit(training_data = training_data,  
        labels = training_labels, 
        validation_data = None, 
        metrics='bel', 
        n_epochs = 1, 
        learning_rate = 0.001)

test_d = [test_data[5]]
test_l = test_y[5]

prediction_weights = model.predict(test_d)
prediction_classes = [np.argmax(p) for p in prediction_weights]

print(prediction_classes)
print(test_l)