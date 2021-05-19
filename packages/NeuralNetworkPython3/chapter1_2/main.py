import sys
import Network as nw
import numpy as np

sys.path.append("../")

import mnist_loader

# Load the data for the neural network.
training_data, validation_data, verification_data = mnist_loader.load_data_wrapper()

# Parameters for the neural network.
mini_batch_size = 2
layer_sizes = [784, 30, 10]
weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]

net = nw.Network(layer_sizes, weights, biases)  # Declare the network.
net2 = nw.Network(layer_sizes, weights, biases)  # Declare second network.

mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]

x, y = mini_batches[0][0]

print(mini_batches[0])

delta_b, delta_w = net.backprop(x, y)

# print(x, "\n", y)

net.learn(training_data, 3, 10, 3.0, test_data=verification_data)  # Let the network learn.
net2.learn(training_data, 3, 10, 3.0, test_data=verification_data)

print("The two networks have the same weights:", np.array_equal(net.weights[0], net2.weights[0]))
