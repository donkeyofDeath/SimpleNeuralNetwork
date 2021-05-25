import unittest
import numpy as np
import SimpleNeuralNetwork as snn
import packages.NeuralNetworkPython3.chapter1_2.Network as nn
import loadMnistData as lmd
import random as rn


class MnistTestCase(unittest.TestCase):

    def setUp(self) -> None:
        """
        Set up-method for this test suite. It creates two neural networks one provided by Michael Nielsen and one by me.
        Further, it sets up the training data of the MNIST library, once in the data format used by me and once in the
        format used by Michael Nielsen.

        :return: None.
        """
        # ------------------------------------
        # Loading data and declaring constants
        # ------------------------------------

        num_pixels, self.training_data, self.verification_data = lmd.load_data()  # Load the MNIST data.

        self.mini_batch_size = 100  # Size of the mini batches used in the stochastic gradient descent.
        self.learning_rate = 3.  # Learning rate often declared by an eta.
        self.epochs = 2  # Number of epochs used in the stochastic gradient descent.
        num_output_neurons = 10  # Number of neurons in the output layer.
        num_hidden_layer_neurons = 30  # Number of neurons in a hidden layer.
        # Sizes of the layers in the neural network.
        layer_sizes = np.array([num_pixels, num_hidden_layer_neurons, num_output_neurons])

        # -------------------------
        # Setup the neural networks
        # -------------------------

        # Setup weights and biases.
        weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        biases = [np.random.randn(y) for y in layer_sizes[1:]]

        self.neural_network = snn.SimpleNeuralNetwork(layer_sizes, weights, biases)  # Define the neural network.
        # Declare network using the library written by michael Nielsen.
        biases = [snn.convert_array(bias_vec) for bias_vec in biases]  # Convert the basis to the right format.
        self.reference_neural_network = nn.Network(layer_sizes, weights, biases)

        # Convert data to the format Michael Nielsen uses.
        self.reference_training_data = [(snn.convert_array(x), snn.convert_array(y)) for x, y in self.training_data]
        self.reference_verification_data = [(snn.convert_array(x), y) for x, y in self.verification_data]

    def test_weights_and_biases(self) -> None:
        """
        This method tests if the weight and biases which were declared in the setUp method are the same for both
        networks.

        :return: None
        """
        # Test if the weights were declared correctly.
        for weight_mat, ref_weight_mat in zip(self.neural_network.weights, self.reference_neural_network.weights):
            np.testing.assert_array_almost_equal(weight_mat, ref_weight_mat)

        # Test if the biases were declared correctly.
        for bias_vec, ref_bias_vec in zip(self.neural_network.biases, self.reference_neural_network.biases):
            np.testing.assert_array_almost_equal(snn.convert_array(bias_vec), ref_bias_vec)

    def test_update_weights_and_biases(self) -> None:
        """
        This test method picks 1000 random elements of the MNIST training data. This subset is then converted to the
        data format used Michael Nielsen and then my and Nielsen's network are fed the according data via the update
        methods. Then the resulting weights and biases are compared.

        :return: None.
        """
        # subset_size = int(0.5 * np.random.randint(100, high=0.25 * len(self.training_data)))
        learning_rate = 10. * np.random.ranf()  # Random float between 0. and 10.
        mini_batch_size = 1000

        # Pick a random subset of the MNIST data with 1000 elements.
        random_mini_batch = rn.sample(self.training_data, mini_batch_size)
        # Recombine the shuffled data.
        input_data, desired_results = zip(*random_mini_batch)
        # Convert the random subset to the data format of Michael Nielsen.
        reference_mini_batch = [(snn.convert_array(x), snn.convert_array(y)) for x, y in random_mini_batch]

        # Update both networks once.
        self.neural_network.update_weights_and_biases((np.array(input_data).T, np.array(desired_results).T),
                                                      mini_batch_size, learning_rate)
        self.reference_neural_network.update_mini_batch(reference_mini_batch, learning_rate)

        # Test if the resulting weights and biases are the same.
        for weight_mat, ref_weight_mat in zip(self.neural_network.weights, self.reference_neural_network.weights):
            np.testing.assert_array_almost_equal(weight_mat, ref_weight_mat)

        for bias_vec, ref_bias_vec in zip(self.neural_network.biases, self.reference_neural_network.biases):
            np.testing.assert_array_almost_equal(snn.convert_array(bias_vec), ref_bias_vec)

    def test_learn(self) -> None:
        """
        Test the learn method which uses the stochastic gradient descent algorithm to train the neural network.

        :return: None.
        """
        # ------------------
        # Train the networks
        # ------------------

        # Let the neural network learn.
        self.neural_network.learn(self.training_data, self.mini_batch_size, self.epochs, self.learning_rate,
                                  shuffle_flag=False, verification_data=self.verification_data)

        # Train the reference network.
        self.reference_neural_network.learn(self.reference_training_data, self.epochs, self.mini_batch_size,
                                            self.learning_rate, test_data=self.reference_verification_data,
                                            shuffle_flag=False)

        # -----------------------------------
        # Compare the results of the networks
        # -----------------------------------

        for weight_mat, ref_weight_mat in zip(self.neural_network.weights, self.reference_neural_network.weights):
            np.testing.assert_array_almost_equal(weight_mat, ref_weight_mat)

        for bias_vec, ref_bias_vec in zip(self.neural_network.biases, self.reference_neural_network.biases):
            np.testing.assert_array_almost_equal(snn.convert_array(bias_vec), ref_bias_vec)


if __name__ == '__main__':
    unittest.main()
