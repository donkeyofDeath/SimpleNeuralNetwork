import numpy as np
import SimpleNeuralNetwork as snn
import packages.NeuralNetworkPython3.chapter1_2.Network as nn
import loadMnistData as lmd

num_pixels, training_data, verification_data = lmd.load_data()

num_output_neurons = 10  # Number of neurons in the output layer.
layer_sizes = np.array([784, 10])  # Sizes of the layers in the neural network.

weights = [np.random.rand(num_output_neurons, num_pixels)]  # Weights used in the neural network.
biases = [np.random.rand(num_output_neurons)]  # Biases used in the neural networks.

neural_network = snn.SimpleNeuralNetwork(layer_sizes, weights, biases)  # Define the neural network.

mini_batch_size = 10
learning_rate = 3.
epochs = 3

print("Started learning.")
neural_network.learn(training_data, mini_batch_size, epochs, learning_rate)  # Let the neural network learn.
print("Finished learning.")

# print(neural_network.weights, neural_network.biases)

# Test the neural network by going through the test images and counting the number of rightly classified images.
result_counter = 0

for data, desired_result in verification_data:

    last_layer_activation = neural_network.feed_forward(data)
    # print(last_layer_activation)
    if desired_result == np.argmax(last_layer_activation):
        result_counter += 1

sum_counter = sum([result == np.argmax(neural_network.feed_forward(data)) for data, result in verification_data])

print(f"{result_counter}, {sum_counter} of {len(verification_data)} test images were verified correctly by my net work.")

reference_neural_network = nn.Network(layer_sizes, weights, [snn.convert_array(bias_vec) for bias_vec in biases])

# Convert format of the data.
training_data = [(snn.convert_array(x), snn.convert_array(y)) for x, y in training_data]
verification_data = [(snn.convert_array(x), y) for x, y in verification_data]

reference_neural_network.learn(training_data, epochs, mini_batch_size, learning_rate, test_data=verification_data)
