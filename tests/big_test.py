import unittest
import numpy as np
import SimpleNeuralNetwork as snn
import packages.NeuralNetworkPython3.chapter1_2.Network as nn
import loadMnistData as lmd


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        """


        :return: None.
        """
        # ------------------------------------
        # Loading data and declaring constants
        # ------------------------------------

        num_pixels, self.training_data, self.verification_data = lmd.load_data()  # Load the MNIST data.

        self.mini_batch_size = 100  # Size of the mini batches used in the stochastic gradient descent.
        self.learning_rate = 3.  # Learning rate often declared by an eta.
        self.epochs = 3  # Number of epochs used in the stochastic gradient descent.
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

    def test_backpropagation_algorithm(self) -> None:
        """
        Test the backpropagation algorithm by comparing the result of my code to the result obtained by
        Michael Nielsen's code. This is done by picking a random training input from the training data and the
        corresponding result.

        :return: None.
        """
        rand_num = np.random.randint(len(self.training_data))  # Random integer.
        data = self.training_data[rand_num][0]  # Randomly pick input data.
        results = self.training_data[rand_num][1]  # Corresponding correct result to the input data.
        reference_data = self.reference_training_data[rand_num][0]  # Pick the data in Michael Nielsen's format.
        reference_result = self.reference_training_data[rand_num][1]  # Corresponding correct result.

        # Test if the the randomly picked data has the same entries in my and Michael Nielsen's format.
        np.testing.assert_array_almost_equal(snn.convert_array(data), reference_data)
        np.testing.assert_array_almost_equal(snn.convert_array(results), reference_result)

        # Results of the back propagation algorithm.
        partial_weights, partial_biases = self.neural_network.back_propagation_algorithm(data, results)
        ref_partial_biases, ref_partial_weights = self.reference_neural_network.backprop(reference_data,
                                                                                         reference_result)

        # Test if the result are the same. Since the biases are 1-dimensional they are converted
        for part_weight_mat, ref_part_weight_mat in zip(partial_weights, ref_partial_weights):
            np.testing.assert_array_almost_equal(part_weight_mat, ref_part_weight_mat)

        for part_bias_vec, ref_part_bias_vec in zip(partial_biases, ref_partial_biases):
            np.testing.assert_array_almost_equal(snn.convert_array(part_bias_vec), ref_part_bias_vec)

    def test_update_weights_and_biases(self) -> None:
        """

        :return: None.
        """


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
