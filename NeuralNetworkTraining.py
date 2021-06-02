import numpy as np
import SimpleNeuralNetwork as snn
import packages.NeuralNetworkPython3.chapter3.network2 as nn2
import loadMnistData as lmd
import time as tm

# -------------------
# Declaring constants
# -------------------

NUM_PIXELS, TRAINING_DATA, VERIFICATION_INPUT, VERIFICATION_RESULT = lmd.load_data_2()  # Load the MNIST data.
VERIFICATION_DATA = (VERIFICATION_INPUT, VERIFICATION_RESULT)

MINI_BATCH_SIZE = 10  # Size of the mini batches used in the stochastic gradient descent.
LEARNING_RATE = .5  # Learning rate often declared as an eta.
REG_PARAM = 5.  # Regularization parameter referred to as lambda in formulae.
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
train_cost, train_acc, ver_cost, ver_acc = neural_network.learn(TRAINING_DATA, MINI_BATCH_SIZE, EPOCHS, LEARNING_RATE,
                                                                REG_PARAM, verification_data=VERIFICATION_DATA,
                                                                monitor_training_accuracy_flag=False,
                                                                monitor_training_cost_flag=False,
                                                                monitor_verification_accuracy_flag=False,
                                                                monitor_verification_cost_flag=False)
end = tm.time()
print(f"Finished learning. My network needed {end - start:.2f} s.\n")
print("Cost function values on the training data:\n", train_cost, "\n")
print("Training data accuracy:\n", train_acc, "\n")
print("Cost function values on the verification data:\n", ver_cost, "\n")
print("Verification data accuracy:\n", ver_acc, "\n")
