import numpy as np
import SimpleNeuralNetwork as snn
import packages.NeuralNetworkPython3.chapter1_2.Network as nn
import loadMnistData as lmd


# -------------------
# Declaring constants
# -------------------

NUM_PIXELS, TRAINING_DATA, VERIFICATION_DATA = lmd.load_data()  # Load the MNIST data.

MINI_BATCH_SIZE = 10  # Size of the mini batches used in the stochastic gradient descent.
LEARNING_RATE = 3.  # Learning rate often declared by an eta.
EPOCHS = 1  # Number of epochs used in the stochastic gradient descent.
NUM_OUTPUT_NEURONS = 10  # Number of neurons in the output layer.
NUM_HIDDEN_LAYER_NEURONS = 30  # Number of neurons in a hidden layer.
# Sizes of the layers in the neural network.
LAYER_SIZES = np.array([NUM_PIXELS, NUM_HIDDEN_LAYER_NEURONS, NUM_OUTPUT_NEURONS])

# ------------------------
# Setup the neural network
# ------------------------

# Setup weights and biases.
weights = [np.random.randn(y, x) for x, y in zip(LAYER_SIZES[:-1], LAYER_SIZES[1:])]
biases = [np.random.randn(y) for y in LAYER_SIZES[1:]]

neural_network = snn.SimpleNeuralNetwork(LAYER_SIZES, weights, biases)  # Define the neural network.

# ----------------------------------
# Train the network and view results
# ----------------------------------
print("Started learning.")
neural_network.learn(TRAINING_DATA, MINI_BATCH_SIZE, EPOCHS, LEARNING_RATE, shuffle_flag=False, verification_data=VERIFICATION_DATA)  # Let the neural network learn.
print("Finished learning.")

# -------------------------------------
# Setup and train the reference network
# -------------------------------------

# Declare network using the library written by michael Nielsen.
biases = [snn.convert_array(bias_vec) for bias_vec in biases]
reference_neural_network = nn.Network(LAYER_SIZES, weights, biases)

# for old_b, new_b in zip(old_biases, new_biases):
#    print(old_b, "\n", new_b, "\n\n")

# Convert data to the format Michael Nielsen uses.
TRAINING_DATA = [(snn.convert_array(x), snn.convert_array(y)) for x, y in TRAINING_DATA]
VERIFICATION_DATA = [(snn.convert_array(x), y) for x, y in VERIFICATION_DATA]

# Train the network.
reference_neural_network.learn(TRAINING_DATA, EPOCHS, MINI_BATCH_SIZE, LEARNING_RATE, test_data=VERIFICATION_DATA,
                               shuffle_flag=False)
