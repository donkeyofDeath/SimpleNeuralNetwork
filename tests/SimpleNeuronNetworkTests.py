import unittest as ut
import SimpleNeuralNetwork as snn
import numpy as np
import packages.NeuralNetworkPython3.chapter1_2.Network as nw


class SimpleNeuralNetworkTestCase(ut.TestCase):

    def setUp(self) -> None:
        """
        Sets up the two neural network which are used during the tests. Each network has three layers. The layer's of
        the first network all have the same number of neurons, while the second network's layers differ in the number
        of neurons.

        :return: None
        """
        self.learning_rate = 3.  # Learning rate, which is the variable eta in gradient descent.
        self.first_layer = np.array([1, 1, 1, 1])  # The input is the same for both neural networks.
        self.reference_first_layer = snn.convert_array(self.first_layer)

        self.biases = [np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1])]
        self.reference_biases = [snn.convert_array(array) for array in self.biases]
        self.second_biases = [np.array([1, 1]), np.array([1])]
        self.reference_second_biases = [snn.convert_array(array) for array in self.second_biases]

        self.weights = [np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
                        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])]

        self.second_weights = [np.array([[1, 1, 0, 0], [0, 0, 1, 1]]),
                               np.array([[1, 1]])]

        self.layer_sizes = np.array([4, 4, 4])
        self.second_layer_sizes = np.array([4, 2, 1])

        # Neural networks to which my results are compared.
        self.first_reference_neural_network = nw.Network(self.layer_sizes, self.weights, self.reference_biases)
        self.second_reference_neural_network = nw.Network(self.second_layer_sizes, self.second_weights,
                                                          self.reference_second_biases)

        # Set up the networks with parameters from above.
        self.first_neural_network = snn.SimpleNeuralNetwork(self.layer_sizes, weights=self.weights, biases=self.biases)
        self.second_neural_network = snn.SimpleNeuralNetwork(self.second_layer_sizes, weights=self.second_weights,
                                                             biases=self.second_biases)

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
        Test if all the according errors are raised when wrong declaration are made. Further it is tested if correct
        variables are declared correctly.

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

        # Test if new layer sizes are set correctly.
        new_layer_sizes = np.array([4, 3, 1])
        self.first_neural_network.layer_sizes = new_layer_sizes
        np.testing.assert_array_almost_equal(self.first_neural_network.layer_sizes, new_layer_sizes)

    def test_check_shapes_biases(self) -> None:
        """
        Tests if the check_shapes method raises errors correctly when setting new biases.

        :return: None
        """
        # Tests if weight matrices with the wrong shape raise an error.
        self.first_neural_network.biases = [np.array([1, 1, 1, 1, 1]), np.array([1, 1, 1, 1, 1])]
        self.second_neural_network.biases = [np.array([1, 1, 1, 1, 1, 1]), np.array([1])]

        with self.assertRaises(ValueError):
            self.first_neural_network.check_shapes()

        with self.assertRaises(ValueError):
            self.second_neural_network.check_shapes()

    def test_check_shapes_weights(self) -> None:
        """
        Tests if the checks_shapes raises the correct when new weights are set.

        :return: None
        """
        # Tests if weight matrices with the wrong shape raise an error.
        self.first_neural_network.weights = [np.array([[1, 1], [1, 1]]), np.array([[1, 1], [1, 1]]), np.array([1])]
        self.second_neural_network.weights = [np.array([[1, 1], [1, 1]]), np.array([[1, 1], [1, 1]]), np.array([1])]

        with self.assertRaises(ValueError):
            self.first_neural_network.check_shapes()

        with self.assertRaises(ValueError):
            self.second_neural_network.check_shapes()

    def test_feed_forward(self) -> None:
        """
        Tests if the feed forward method calculates the individual layers of the neural network correctly.

        :return: None.
        """
        # Check if the run method resets the parameters set by the update method correctly.
        first_result = self.first_neural_network.feed_forward(self.first_layer.reshape(len(self.first_layer), 1))

        np.testing.assert_array_almost_equal(first_result, np.array([[0.8677026536525567],
                                                                     [0.8677026536525567],
                                                                     [0.8677026536525567],
                                                                     [0.8677026536525567]]))

        # Check if the run method resets the parameters set by the update method correctly.
        second_result = self.second_neural_network.feed_forward(self.first_layer.reshape(len(self.first_layer), 1))

        np.testing.assert_array_almost_equal(second_result, np.array([[0.9481003474891515]]))

    def test_sigmoid_function(self) -> None:
        """
        Tests the sigmoid function.

        :return: None
        """
        np.testing.assert_almost_equal(self.first_neural_network.sigmoid_function(0.), 0.5)
        np.testing.assert_almost_equal(self.first_neural_network.sigmoid_function(0.5), 0.6224593312018546)

    def test_sigmoid_derivative(self) -> None:
        """
        Tests if the derivative of the sigmoid function was calculated correctly. The analytical solution is compared to
        the the result gained by using a finite difference method.

        :return: None
        """

        def finite_difference(func: callable, point: float):
            """
            Returns the the derivative of the function func at the point point.
            :param func: Function of which the derivative is calculated.
            :param point: Point at which the derivative is calculated.
            :return: The derivative of func at the point point.
            """
            delta_x = 10 ** (-6)
            return (func(point + delta_x) - func(point)) / delta_x

        # Points at which the derivative is calculated.
        point_1 = 1.
        point_2 = 0.

        # Numerically calculated derivatives.
        finite_difference_1 = finite_difference(self.first_neural_network.sigmoid_function, point_1)
        finite_difference_2 = finite_difference(self.first_neural_network.sigmoid_function, point_2)

        # Test the results.
        np.testing.assert_almost_equal(finite_difference_1, self.first_neural_network.sigmoid_derivative(point_1))
        np.testing.assert_almost_equal(finite_difference_2, self.first_neural_network.sigmoid_derivative(point_2))

    def test_cost_func_grad(self):
        """
        Test the gradient of the cost function.

        :return: None.
        """
        self.assertAlmostEqual(1 - 2, self.first_neural_network.cost_func_grad(1, 2))
        self.assertAlmostEqual(3 - 2, self.first_neural_network.cost_func_grad(3, 2))


if __name__ == "__main__":
    ut.main()
