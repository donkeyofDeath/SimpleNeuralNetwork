import unittest as ut
import SimpleNeuralNetwork as snn
import numpy as np


class SimpleNeuralNetworkTestCase(ut.TestCase):

    def setUp(self) -> None:
        """
        Sets up the two neural network which are used during the tests. Each network has three layers. The layer's of
        the first network all have the same number of neurons, while the second network's layers differ in the number
        of neurons.

        :return: None
        """
        self.first_layer = np.array([1, 1, 1, 1])  # The input is the same for both neural networks.

        self.biases = [np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1])]
        self.second_biases = [np.array([1, 1]), np.array([1])]

        self.weights = [np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
                        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])]

        self.second_weights = [np.array([[1, 1, 0, 0], [0, 0, 1, 1]]),
                               np.array([1, 1])]

        self.layer_sizes = np.array([4, 4, 4])
        self.second_layer_sizes = np.array([4, 2, 1])

        # Set up the networks with parameters from above.
        self.first_neural_network = snn.SimpleNeuralNetwork(self.first_layer, self.layer_sizes, self.weights,
                                                            self.biases)
        self.second_neural_network = snn.SimpleNeuralNetwork(self.first_layer, self.second_layer_sizes,
                                                             self.second_weights, self.second_biases)

    def tearDown(self) -> None:
        del self.first_neural_network
        del self.second_neural_network

    def test_update(self) -> None:
        """
        This method tests the update method of both neural networks.

        :return: None
        """
        self.first_neural_network.update()
        np.testing.assert_array_almost_equal(self.first_neural_network.current_layer, np.array([0.8495477739862124,
                                                                                                0.8495477739862124,
                                                                                                0.8495477739862124,
                                                                                                0.8495477739862124]))

        self.second_neural_network.update()
        np.testing.assert_array_almost_equal(self.second_neural_network.current_layer, np.array([0.9214430516601156,
                                                                                                 0.9214430516601156]))


if __name__ == '__main__':
    ut.main()
