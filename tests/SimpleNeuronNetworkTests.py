import unittest as ut
import SimpleNeuralNetwork as snn
import numpy as np
import packages.NeuralNetworkPython3.chapter1_2.Network as nw


def convert_array(array: np.array) -> np.array:
    """
    Convert a 1D numpy array into a 2D matrix with one column.

    :param array: 1D numpy array
    :return: 2D numpy column array with the same entry.
    """
    return np.reshape(array, (len(array), 1))


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
        self.reference_first_layer = convert_array(self.first_layer)

        self.biases = [np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1])]
        self.reference_biases = [convert_array(array) for array in self.biases]
        self.second_biases = [np.array([1, 1]), np.array([1])]
        self.reference_second_biases = [convert_array(array) for array in self.second_biases]

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
        self.first_neural_network = snn.SimpleNeuralNetwork(self.layer_sizes, self.weights, self.biases)
        self.second_neural_network = snn.SimpleNeuralNetwork(self.second_layer_sizes, self.second_weights,
                                                             self.second_biases)

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

    def test_update(self) -> None:
        """
        This method tests the update method of both neural networks.

        :return: None
        """
        # Set the current layer to the sigmoid function of the first layer attribute.
        self.first_neural_network.current_layer = self.first_neural_network.sigmoid_function(self.first_layer)
        np.testing.assert_array_almost_equal(self.first_neural_network.update(self.first_neural_network.weights[0],
                                                                              self.first_neural_network.biases[0]),
                                             np.array([0.8495477739862124,
                                                       0.8495477739862124,
                                                       0.8495477739862124,
                                                       0.8495477739862124]))

        # Set the current layer to the sigmoid function of the first layer attribute.
        self.second_neural_network.current_layer = self.second_neural_network.sigmoid_function(self.first_layer)
        self.second_neural_network.update(self.second_neural_network.weights[0], self.second_neural_network.biases[0])
        np.testing.assert_array_almost_equal(self.second_neural_network.current_layer, np.array([0.9214430516601156,
                                                                                                 0.9214430516601156]))

    def test_feed_forward(self) -> None:
        """
        Tests if the feed forward method calculates the individual layers of the neural network correctly.

        :return: None.
        """
        # Check if the run method resets the parameters set by the update method correctly.
        first_result = self.first_neural_network.feed_forward(self.first_layer)

        np.testing.assert_array_almost_equal(first_result, np.array([0.8640739977337843,
                                                                     0.8640739977337843,
                                                                     0.8640739977337843,
                                                                     0.8640739977337843]))

        # Check if the run method resets the parameters set by the update method correctly.
        second_result = self.second_neural_network.feed_forward(self.first_layer)

        np.testing.assert_array_almost_equal(second_result, np.array([0.9449497893439537]))

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

    def test_calculate_a_and_z(self):
        """
        Test the methods which calculates the activations and the z values of a neural network.

        :return: None.
        """
        # Calculate the activations and the z values of the first neural network.
        activations, z_values = self.first_neural_network.calculate_a_and_z(self.first_layer)

        self.assertEqual(len(activations), 3)  # Check if the list has the right length.
        # Check the activations.
        np.testing.assert_array_almost_equal(activations[0], np.array([1., 1., 1., 1.]))
        np.testing.assert_array_almost_equal(activations[1], np.array([0.8807970779778823, 0.8807970779778823,
                                                                       0.8807970779778823, 0.8807970779778823]))
        np.testing.assert_array_almost_equal(activations[2], np.array([0.8677026536525567, 0.8677026536525567,
                                                                       0.8677026536525567, 0.8677026536525567]))

        self.assertEqual(len(z_values), 2)  # Check if the list has the right length.
        # Check the z values.
        np.testing.assert_array_almost_equal(z_values[0], np.array([2., 2., 2., 2.]))
        np.testing.assert_array_almost_equal(z_values[1], np.array([1.8807970779778822, 1.8807970779778822,
                                                                    1.8807970779778822, 1.8807970779778822]))

        # Calculate the activations and the z values of the second neural network.
        activations, z_values = self.second_neural_network.calculate_a_and_z(self.first_layer)

        self.assertEqual(len(activations), 3)  # Check if the list has the right length.
        # Check the activations.
        np.testing.assert_array_almost_equal(activations[0], np.array([1., 1., 1., 1.]))
        np.testing.assert_array_almost_equal(activations[1], np.array([0.9525741268224334, 0.9525741268224334]))
        np.testing.assert_array_almost_equal(activations[2], np.array([0.9481003474891515]))

        self.assertEqual(len(z_values), 2)  # Check if the list has the right length.
        # Check the z values.
        np.testing.assert_array_almost_equal(z_values[0], np.array([3., 3.]))
        np.testing.assert_array_almost_equal(z_values[1], np.array([2.9051482536448665]))

    def test_calculate_deltas(self):
        """
        Tests if the correct values for delta re returned for the back propagation algorithm.

        :return: NOne.
        """
        z_values = [np.array([1., 1., 1., 1.]), np.array([1., 1., 1., 1.])]  # Z values.

        deltas_last_layer = np.array([0.5, 0.5, 0.5, 0.5])  # Deltas of the last layer used for the backpropagation.

        deltas = self.first_neural_network.calculate_deltas(deltas_last_layer, z_values)  # All deltas.

        np.testing.assert_array_almost_equal(deltas[0], np.array([0.09830597, 0.09830597, 0.09830597, 0.09830597]))
        np.testing.assert_array_almost_equal(deltas[1], deltas_last_layer)

        # Calculations for the second neural network.

        z_values = [np.array([1., 1.]), np.array([1.])]  # Z values.

        deltas_last_layer = np.array([0.5])  # Deltas of the last layer used for the backpropagation.

        deltas = self.second_neural_network.calculate_deltas(deltas_last_layer, z_values)

        np.testing.assert_array_almost_equal(deltas[0], np.array([0.09830596662074094, 0.09830596662074094]))
        np.testing.assert_array_almost_equal(deltas[1], np.array([0.5]))

    def test_back_propagation_algorithm(self):
        """
        Tests the back propagation algorithm for two different neural networks. It uses the code by Michael Nielsen,
        which I ported to Python 3 to check id the back propagation algorithm functions correctly.

        :return: None.
        """
        training_data = np.array([255., 255., 255., 255.])  # Training data, which serves as an input.
        desired_result = np.array([0., 0., 0., .1])  # Results, which represents the desired output of the network.

        reference_training_data = convert_array(training_data)
        reference_desired_result = convert_array(desired_result)
        # Testing the first neural network.

        # Calculate the the partial derivatives of the weights and biases of the network which is tested and the
        # reference neural network.
        partial_weights, partial_biases = self.first_neural_network.back_propagation_algorithm(training_data,
                                                                                               desired_result)
        partial_biases_ref, partial_weights_ref = self.first_reference_neural_network.backprop(reference_training_data,
                                                                                               reference_desired_result)

        # Check if the if the output have the right shapes.
        self.assertEqual(len(partial_weights), 2)
        self.assertEqual(partial_weights[0].shape, (4, 4))
        self.assertEqual(partial_weights[1].shape, (4, 4))

        # Loop through all the partial derivatives and compare the testing network to the reference network.
        for weights_der, reference_weight_der in zip(partial_weights, partial_weights_ref):
            np.testing.assert_array_almost_equal(weights_der, reference_weight_der)

        for bias_der, reference_bias_der in zip(partial_biases, partial_biases_ref):
            np.testing.assert_array_almost_equal(convert_array(bias_der), reference_bias_der)

        # Testing the second neural network.

        desired_result = np.array([0.5])  # Desired result used in the back propagation algorithm.
        reference_desired_result = convert_array(desired_result)

        # Set up partial derivatives of the weights and biases for the reference and testing networks.
        partial_weights, partial_biases = self.second_neural_network.back_propagation_algorithm(training_data,
                                                                                                desired_result)

        partial_biases_ref, partial_weights_ref = self.second_reference_neural_network.backprop(reference_training_data,
                                                                                                reference_desired_result)

        # Check if the partial derivatives have the right shapes.
        self.assertEqual(len(partial_weights), 2)
        self.assertEqual(partial_weights[0].shape, (2, 4))
        self.assertEqual(partial_weights[1].shape, (1, 2))

        # Loop through all the partial derivatives and compare them to the derivatives of the reference network.
        for weights_der, reference_weight_der in zip(partial_weights, partial_weights_ref):
            np.testing.assert_array_almost_equal(weights_der, reference_weight_der)

        for bias_der, reference_bias_der in zip(partial_biases, partial_biases_ref):
            np.testing.assert_array_almost_equal(convert_array(bias_der), reference_bias_der)

    def test_update_weights_and_biases(self):
        """
        This method tests the update_weights_and_biases method which is used in the learning algorithm to update the
        weights and biases with the help of the gradient of a mini batch. The test is performed using two different
        neural networks and their references.

        :return:None.
        """
        # --------------------------------
        # Testing the first neural network
        # --------------------------------

        # Training data for this test.
        training_data = [(np.array([255., 255., 255., 255.]), np.array([1., 0., 0., 0.])),
                         (np.array([255., 0., 255., 255.]), np.array([0., 1., 0., 0.]))]

        reference_training_data = [(convert_array(x), convert_array(y)) for x, y in training_data]

        mini_batch_size = len(training_data)  # Number of elements in a mini batch.

        # Setup the neural networks.
        self.first_neural_network.update_weight_and_biases(training_data, mini_batch_size, self.learning_rate)
        self.first_reference_neural_network.update_mini_batch(reference_training_data, self.learning_rate)

        # print(self.first_neural_network.weights)

        # Loop through all the weights and biases of the first neural network and compare them to the ones of the
        # reference network.
        for weight_mat, weight_mat_ref in zip(self.first_neural_network.weights,
                                              self.first_reference_neural_network.weights):
            np.testing.assert_array_almost_equal(weight_mat, weight_mat_ref)

        for bias_vec, bias_vec_ref in zip(self.first_neural_network.biases, self.first_reference_neural_network.biases):
            np.testing.assert_array_almost_equal(convert_array(bias_vec), bias_vec_ref)

        # ---------------------------------
        # Testing the second neural network
        # ---------------------------------

        # Training data for this test.
        training_data = [(np.array([255., 255., 255., 255.]), np.array([1.])),
                         (np.array([255., 0., 255., 255.]), np.array([0.]))]

        reference_training_data = [(convert_array(x), convert_array(y)) for x, y in training_data]

        mini_batch_size = len(training_data)  # Number of elements in a mini batch.

        # Setup the neural networks.
        self.second_neural_network.update_weight_and_biases(training_data, mini_batch_size, self.learning_rate)
        self.second_reference_neural_network.update_mini_batch(reference_training_data, self.learning_rate)

        # Loop through all the weights and biases of the first neural network and compare them to the ones of the
        # reference network.
        for weight_mat, weight_mat_ref in zip(self.second_neural_network.weights,
                                              self.second_reference_neural_network.weights):
            np.testing.assert_array_almost_equal(weight_mat, weight_mat_ref)

        for bias_vec, bias_vec_ref in zip(self.second_neural_network.biases,
                                          self.second_reference_neural_network.biases):
            np.testing.assert_array_almost_equal(convert_array(bias_vec), bias_vec_ref)

    def test_learn(self):
        """
        Tests if the learning algorithm of the neural networks is implemented correctly by comparing the output of the
        learn method with the result of the learn method written by Michael Nielsen. This is done for two different
        neural networks. I just realised that it is very difficult to test this method since it randomly shuffles the
        training data.

        :return: None.
        """

        mini_batch_size = 2  # Number of elements in a mini batch.
        epochs = 3  # Number of epochs.

        # --------------------------------
        # Testing the first neural network
        # --------------------------------

        # Training data for the first neural network.
        training_data = [(np.array([255., 255., 255., 255.]), np.array([1., 0., 0., 0.])),
                         (np.array([255., 0., 255., 255.]), np.array([0., 1., 0., 0.])),
                         (np.array([0., 255., 199., 255.]), np.array([0., 0., 1., 0.])),
                         (np.array([255., 255., 0., 255.]), np.array([0., 0., 0., 1.]))]

        # Convert the arrays into the format Michael Nielsen uses.
        reference_training_data = [(convert_array(x), convert_array(y)) for x, y in training_data]

        # Train a network and its reference using the corresponding training data.
        self.first_neural_network.learn(training_data, mini_batch_size, epochs, self.learning_rate, shuffle_flag=False)
        self.first_reference_neural_network.learn(reference_training_data, epochs, mini_batch_size, self.learning_rate,
                                                  shuffle_flag=False)

        # Compare all the weights to their reference.
        for weight_mat, weight_mat_ref in zip(self.first_neural_network.weights,
                                              self.first_reference_neural_network.weights):
            np.testing.assert_array_almost_equal(weight_mat, weight_mat_ref)

        # Compare all the biases to their reference.
        for bias_vec, bias_vec_ref in zip(self.first_neural_network.biases, self.first_reference_neural_network.biases):
            np.testing.assert_array_almost_equal(convert_array(bias_vec), bias_vec_ref)

        # ---------------------
        # Second neural network
        # ---------------------

        # Training data for this neural network.
        training_data = [(np.array([255., 255., 255., 255.]), np.array([1.])),
                         (np.array([255., 0., 255., 255.]), np.array([0.66])),
                         (np.array([0., 255., 199., 255.]), np.array([0.33])),
                         (np.array([255., 255., 0., 255.]), np.array([0.]))]

        # Convert the format of the training data to the format Michael Nielsen is using.
        reference_training_data = [(convert_array(x), convert_array(y)) for x, y in training_data]

        # Train the networks using the corresponding training data.
        self.second_neural_network.learn(training_data, mini_batch_size, epochs, self.learning_rate, shuffle_flag=False)
        self.second_reference_neural_network.learn(reference_training_data, epochs, mini_batch_size, self.learning_rate,
                                                   shuffle_flag=False)

        # Compare all the weights to their references.
        for weight_mat, weight_mat_ref in zip(self.second_neural_network.weights,
                                              self.second_reference_neural_network.weights):
            np.testing.assert_array_almost_equal(weight_mat, weight_mat_ref)

        # Compare all the biases to their references.
        for bias_vec, bias_vec_ref in zip(self.second_neural_network.biases, self.second_reference_neural_network.biases):
            np.testing.assert_array_almost_equal(convert_array(bias_vec), bias_vec_ref)


if __name__ == "__main__":
    ut.main()
