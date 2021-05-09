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

    def test_weights(self) -> None:
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

        # Test if new weights are declared correctly.
        new_weights = [np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]),
                       np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])]
        self.first_neural_network.weights = new_weights
        np.testing.assert_array_almost_equal(self.first_neural_network.weights, new_weights)

    def test_biases(self) -> None:
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

        # Test if the entries of the weight matrices are numbers.
        with self.assertRaises(TypeError):
            self.first_neural_network.biases = [np.array(["Hello"])]

        with self.assertRaises(ValueError):
            self.first_neural_network.biases = [np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]), np.array([1])]

        # Test if new biases are declared correctly.
        new_biases = [np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])]
        self.first_neural_network.biases = new_biases
        np.testing.assert_array_almost_equal(self.first_neural_network.biases, new_biases)

    def test_layer_sizes(self) -> None:
        """


        :return: None
        """
        # Tests if an error is raised when the layer_sizes attribute is declared with the wrong type.
        with self.assertRaises(TypeError):
            self.first_neural_network.layer_sizes = "Hello"

        # Tests if an error is raised if the declared numpy array has the wrong shape.
        with self.assertRaises(ValueError):
            self.first_neural_network.layer_sizes = np.array([[1, 1]])

        # Error if other datatype than int is provided.
        with self.assertRaises(TypeError):
            self.first_neural_network.layer_sizes = np.array([1., 2.])

        # Error for handling negative ints.
        with self.assertRaises(ValueError):
            self.first_neural_network.layer_sizes = np.array([-1, -2])

        # Error if the shape of the layer_sizes does not coincide with the current layer.
        with self.assertRaises(ValueError):
            self.first_neural_network.layer_sizes = np.array([1, 1, 1])

        # Test if new layer sizes are set correctly.
        new_layer_sizes = np.array([4, 3, 1])
        self.first_neural_network.layer_sizes = new_layer_sizes
        np.testing.assert_array_almost_equal(self.first_neural_network.layer_sizes, new_layer_sizes)

    def test_check_shapes_biases(self):
        """

        :return:
        """
        # Tests if weight matrices with the wrong shape raise an error.
        self.first_neural_network.biases = [np.array([1, 1, 1, 1, 1]), np.array([1, 1, 1, 1, 1])]
        self.second_neural_network.biases = [np.array([1, 1, 1, 1, 1, 1]), np.array([1])]

        with self.assertRaises(ValueError):
            self.first_neural_network.check_shapes()

        with self.assertRaises(ValueError):
            self.second_neural_network.check_shapes()

    def test_check_shapes_weights(self):
        """

        :return:
        """
        # Tests if weight matrices with the wrong shape raise an error.
        self.first_neural_network.weights = [np.array([[1, 1], [1, 1]]), np.array([[1, 1], [1, 1]]), np.array([1])]
        self.second_neural_network.weights = [np.array([[1, 1], [1, 1]]), np.array([[1, 1], [1, 1]]), np.array([1])]

        with self.assertRaises(ValueError):
            self.first_neural_network.check_shapes()

        with self.assertRaises(ValueError):
            self.second_neural_network.check_shapes()

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

    def test_run(self) -> None:
        """


        :return: None.
        """
        first_result_list = self.first_neural_network.run()
        np.testing.assert_array_almost_equal(first_result_list[0], np.array([0.7310585786300049,
                                                                             0.7310585786300049,
                                                                             0.7310585786300049,
                                                                             0.7310585786300049]))

        np.testing.assert_array_almost_equal(first_result_list[1], np.array([0.8495477739862124,
                                                                             0.8495477739862124,
                                                                             0.8495477739862124,
                                                                             0.8495477739862124]))

        np.testing.assert_array_almost_equal(first_result_list[2], np.array([0.8640739977337843,
                                                                             0.8640739977337843,
                                                                             0.8640739977337843,
                                                                             0.8640739977337843]))

        second_result_list = self.second_neural_network.run()
        np.testing.assert_array_almost_equal(second_result_list[0], np.array([0.7310585786300049,
                                                                              0.7310585786300049,
                                                                              0.7310585786300049,
                                                                              0.7310585786300049]))

        np.testing.assert_array_almost_equal(second_result_list[1], np.array([0.9214430516601156,
                                                                              0.9214430516601156]))

        np.testing.assert_array_almost_equal(second_result_list[2], np.array([0.9449497893439537]))

    def test_sigmoid_function(self) -> None:
        """

        :return: None
        """
        np.testing.assert_almost_equal(self.first_neural_network.sigmoid_function(0.), 0.5)
        np.testing.assert_almost_equal(self.first_neural_network.sigmoid_function(0.5), 0.622459)


if __name__ == "__main__":
    ut.main()
