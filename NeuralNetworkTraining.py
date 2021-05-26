import numpy as np
import SimpleNeuralNetwork as snn
# import packages.NeuralNetworkPython3.chapter1_2.Network as nn
import loadMnistData as lmd
import time as tm

# -------------------
# Declaring constants
# -------------------

NUM_PIXELS, TRAINING_DATA, VERIFICATION_INPUT, VERIFICATION_RESULT = lmd.load_data_2()  # Load the MNIST data.
VERIFICATION_DATA = (VERIFICATION_INPUT, VERIFICATION_RESULT)

MINI_BATCH_SIZE = 100  # Size of the mini batches used in the stochastic gradient descent.
LEARNING_RATE = 3.  # Learning rate often declared as an eta.
EPOCHS = 10  # Number of epochs used in the stochastic gradient descent.
NUM_OUTPUT_NEURONS = 10  # Number of neurons in the output layer.
NUM_HIDDEN_LAYER_NEURONS = 30  # Number of neurons in a hidden layer.
# Sizes of the layers in the neural network.
LAYER_SIZES = np.array([NUM_PIXELS, NUM_HIDDEN_LAYER_NEURONS, NUM_OUTPUT_NEURONS])

# ------------------------
# Setup the neural network
# ------------------------

neural_network = snn.SimpleNeuralNetwork(LAYER_SIZES)  # Define the neural network.

# ----------------------------------
# Train the network and view results
# ----------------------------------

start = tm.time()
# Let the neural network learn.
neural_network.learn(TRAINING_DATA, MINI_BATCH_SIZE, EPOCHS, LEARNING_RATE, verification_data=VERIFICATION_DATA)
end = tm.time()
print(f"Finished learning. My network needed: {end - start:.2f} s.")
