from keras.datasets import mnist
import numpy as np
import SimpleNeuralNetwork as snn

num_pixels = 784  # Number of pixels in a training images.
num_output_neurons = 10  # Number of neurons in the output layer.
layer_sizes = np.array([784, 10])  # Sizes of the layers in the neural network.

weights = [np.random.rand(num_output_neurons, num_pixels)]  # Weights used in the neural network.
biases = [np.random.rand(num_output_neurons)]  # Biases used in the neural networks.

neural_network = snn.SimpleNeuralNetwork(layer_sizes, weights, biases)  # Define the neural network.
(train_inputs, desired_results), (test_X, test_y) = mnist.load_data()  # Load the test data


def convert_number(number: int) -> np.ndarray:
    """
    This function converts a number of the training data of the expected results to a numpy array with a one at the
    index num.

    :return: Numpy array with a 1 at the index num.
    """
    vec = np.zeros(10)
    vec[number] = 1.
    return vec


# Convert the elements of the desired results from numbers to numpy arrays.
converted_desired_results = np.array([convert_number(num) for num in desired_results])

# Declare the parameters for the neural network.
# Reshape the training input so that it is only one vector.
# train_inputs = np.ndarray([data.reshape(num_pixels) for data in train_inputs])
train_inputs = train_inputs.reshape(len(train_inputs), num_pixels)
training_data = list(zip(train_inputs, converted_desired_results))
mini_batch_size = 100
learning_rate = 1.
epochs = 5

print("Started learning.")
# Let the neural network learn.
new_weights, new_biases = neural_network.learn(training_data, mini_batch_size, epochs, learning_rate)
print("Finished learning.")

print(new_weights[0], new_biases[0])

# Test the neural network by going through the test images and counting the number of rightly classified images.
result_counter = 0
test_X = test_X.reshape(len(test_X), num_pixels)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


# print(test_X[0])
# print(sigmoid(test_X[0]))
# sigmoid(neural_network.biases[0])
# sigmoid(neural_network.weights[0])

for data, desired_result in zip(test_X, test_y):

    last_layer_activation = neural_network.feed_forward(data / 255.)
    if desired_result == np.argmax(last_layer_activation):
        result_counter += 1

print(f"{result_counter} of {len(test_X)} test images were verified correctly.")
