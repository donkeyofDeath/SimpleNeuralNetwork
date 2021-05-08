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
        """
        Deletes the SimpleNeuralNetwork objects declared in the setUp method.

        :return:None
        """
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

    def test_weight_lists(self) -> None:
        """
        Tests if errors are raised correctly when setting the weight_list of a neural network.

        :return: None
        """
        # Tests if an incorrect type (should be a list) raises an according error.
        with self.assertRaises(TypeError):
            self.first_neural_network.weights = 5.

        # Tests if an error is raised when the contents of the weight list are not a numpy array.
        with self.assertRaises(TypeError):
            self.first_neural_network.weights = [1, 2, 3]

        # Tests if weight matrices with the wrong shape raise an error.
        with self.assertRaises(ValueError):
            self.first_neural_network.weights = [np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]), np.array([1])]

        # Test if the entries of the weight matrices are numbers.
        with self.assertRaises(TypeError):
            self.first_neural_network.weights = [np.array(["Hello"])]

        # Test if the error is raised when the shape of a weight matrix does not coincide with size of a corresponding
        # layer.
        with self.assertRaises(ValueError):
            self.first_neural_network.weights = [np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
                                                 np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])]
        with self.assertRaises(ValueError):
            self.second_neural_network.weights = [np.array([[1, 1, 0, 0], [0, 0, 1, 1]]),
                                                  np.array([1, 1, 1])]

        # Test if new weights are declared correctly.
        new_weights = [np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]),
                       np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])]
        self.first_neural_network.weights = new_weights
        np.testing.assert_array_almost_equal(self.first_neural_network.weights, new_weights)

    def test_bias_list(self) -> None:
        """
        Tests if errors are raised correctly when setting the bias_list of a neural network.

        :return: None
        """
        # Tests if an incorrect type (should be a list) raises an according error.
        with self.assertRaises(TypeError):
            self.first_neural_network.biases = 5.

        # Tests if an error is raised when the contents of the bias list are not a numpy array.
        with self.assertRaises(TypeError):
            self.first_neural_network.biases = [1, 2, 3]

        # Tests if weight matrices with the wrong shape raise an error.
        with self.assertRaises(ValueError):
            self.first_neural_network.biases = [np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]), np.array([1])]

        # Test if the entries of the weight matrices are numbers.
        with self.assertRaises(TypeError):
            self.first_neural_network.biases = [np.array(["Hello"])]

        # Test if the error is raised when the shape of a weight matrix does not coincide with size of a corresponding
        # layer.
        with self.assertRaises(ValueError):
            self.first_neural_network.biases = [np.array([0, 0, 0, 0, 0]), np.array([0, 0, 0, 0])]
        with self.assertRaises(ValueError):
            self.second_neural_network.biases = [np.array([0, 0]), np.array([0, 0])]

        # Test if new weights are declared correctly.
        new_biases = [np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])]
        self.first_neural_network.biases = new_biases
        np.testing.assert_array_almost_equal(self.first_neural_network.biases, new_biases)


if __name__ == '__main__':
    ut.main()
