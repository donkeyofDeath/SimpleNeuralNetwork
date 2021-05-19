import numpy as np
import SimpleNeuralNetwork as snn
import packages.NeuralNetworkPython3.chapter1_2.Network as nn
import loadMnistData as lmd


# -------------------
# Declaring constants
# -------------------

NUM_PIXELS, training_data, verification_data = lmd.load_data()  # Load the MNIST data.

MINI_BATCH_SIZE = 10  # Size of the mini batches used in the stochastic gradient descent.
LEARNING_RATE = 3.  # Learning rate often declared by an eta.
EPOCHS = 3  # Number of epochs used in the stochastic gradient descent.
NUM_OUTPUT_NEURONS = 10  # Number of neurons in the output layer.
LAYER_SIZES = np.array([NUM_PIXELS, NUM_OUTPUT_NEURONS])  # Sizes of the layers in the neural network.

# ------------------------
# Setup the neural network
# ------------------------


weights = [np.random.rand(NUM_OUTPUT_NEURONS, NUM_PIXELS)]  # Weights used in the neural network.
biases = [np.random.rand(NUM_OUTPUT_NEURONS)]  # Biases used in the neural networks.

neural_network = snn.SimpleNeuralNetwork(LAYER_SIZES, weights, biases)  # Define the neural network.

# ----------------------------------
# Train the network and view results
# ----------------------------------

print("Started learning.")
neural_network.learn(training_data, MINI_BATCH_SIZE, EPOCHS, LEARNING_RATE)  # Let the neural network learn.
print("Finished learning.")

# Test the neural network by going through the test images and counting the number of rightly classified images.
result_counter = sum([result == np.argmax(neural_network.feed_forward(data)) for data, result in verification_data])

# Print the result of how many images are identified correctly.
print(f"{result_counter} of {len(verification_data)} test images were verified correctly by my net work.")

# ---------------------------------
# Setup and train reference network
# ---------------------------------

# Declare network.
reference_neural_network = nn.Network(LAYER_SIZES, weights, [snn.convert_array(bias_vec) for bias_vec in biases])

# Convert format of the data.
training_data = [(snn.convert_array(x), snn.convert_array(y)) for x, y in training_data]
verification_data = [(snn.convert_array(x), y) for x, y in verification_data]

# Train the network.
reference_neural_network.learn(training_data, EPOCHS, MINI_BATCH_SIZE, LEARNING_RATE, test_data=verification_data)
