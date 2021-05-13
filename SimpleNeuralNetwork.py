import numpy as np
import random as rand


class SimpleNeuralNetwork:

    def __init__(self, layer_sizes: np.ndarray, weight_list: list, bias_list: list) -> None:
        """
        Tested.
        This is the constructor for a simple feed forward neural network object. The neural network consist of
        individual layers of different which are connected through the weights and biases via linear equation.
        The result of this linear equation is then put in a Sigmoid function to amp it to the interval [0, 1].
        For further reading checkout the book http://neuralnetworksanddeeplearning.com/ by Michael Nielsen.

        :param layer_sizes: A 1D numpy array, containing the size (number of neurons) of the individual layers.
        :param weight_list: List of weight matrix connecting the layers of the network via multiplication.
        :param bias_list: List of bias vectors added to neurons of each layer.
        """

        self.current_layer = None
        self.layer_sizes = layer_sizes
        self.weights = weight_list
        self.biases = bias_list

    @property
    def layer_sizes(self) -> np.ndarray:
        return self._layer_sizes

    @layer_sizes.setter
    def layer_sizes(self, new_layer_sizes) -> None:
        """
        Tested.
        Setter method for the layer sizes. it is tested if the new layer sizes is a numpy, has an according shape and is
        filled with positive integers. If this is not the case, an according error is raised. If no error is raised, the
        layer sizes are updated.

        :param new_layer_sizes: Numpy array containing the size (number of neurons) of each layer of the neural network.
        :return: None
        """
        # Check: The new layer sizes is a list of numpy array.
        if not isinstance(new_layer_sizes, np.ndarray):
            raise TypeError("The layer sizes have to be a numpy array.")

        # Check: The new layer size array is one-dimensional.
        if len(new_layer_sizes.shape) != 1:
            raise ValueError("The sizes of the layers has to be a one-dimensional array.")

        # Check: Check if the entries in the array are ints.
        if new_layer_sizes.dtype != int:
            raise TypeError("Size of the layers have to of type int.")

        # Check: All ints in the array are greater than zero.
        if not all(new_layer_sizes > 0):
            raise ValueError("The sizes of the layers have to be positive.")

        # Check: The layer size of the current layer corresponds to one in the new list of layer sizes.
        # if len(self.current_layer) != new_layer_sizes[self._layer_counter]:
        #    raise ValueError("The size of the current layer has to coincide with the corresponding value in the"
        #                     "layer_sizes array.")

        self._layer_sizes = new_layer_sizes

        # Print a warning in colored text to the console when the layer size is changed.
        # print("\033[93m" + "Warning: Size of the layers was changed. The shape of the weights and biases might not"
        #                   "coincide with the layer sizes anymore." + "\033[0m")

    @property
    def biases(self) -> list:
        return self._biases

    @biases.setter
    def biases(self, new_biases) -> None:
        """
        Tested.
        Setter method for the biases used in the connections of the neural networks. Before a new array of biases is
        set, it is checked if it is a numpy array, has an according shape, is filled with real numbers and if the shapes
        of the biases conform with the sizes entered in the layer_sizes field.

        :param new_biases: New basis, replacing the old biases after the checks have been, which are described above.
        :return: None
        """
        # Check the type of the new biases.
        if not isinstance(new_biases, list):
            raise TypeError("All entries of the biases have to be numpy arrays.")

        # Loop through all the array in the biases.
        for n, bias_vector in enumerate(new_biases):
            # Check type of all entries in biases.
            if not isinstance(bias_vector, np.ndarray):
                raise TypeError("All entries of the bias list have to be numpy arrays.")

            # Check if all the arrays have are one-dimensional.
            if len(bias_vector.shape) != 1:
                raise ValueError("All entries of biases have to be one-dimensional.")

            # Check if all the entries in the arrays are numbers.
            if not (bias_vector.dtype == float or bias_vector.dtype == int):
                raise TypeError("The entries of the biases have to be real numbers.")

        self._biases = new_biases

    @property
    def weights(self) -> list:
        return self._weights

    @weights.setter
    def weights(self, new_weights) -> None:
        """
        Tested.
        Setter method for the weights which are used in the connection of the layers. Before a the weights are set a
        number of checks is performed. These checks include if the newly entered weights are in a numpy array, if this
        array has the right shape, if the numpy is filled with numbers and if the shapes of the individual weight
        matrices correspond to the number of neurons declared by the layer_sizes array.

        :param new_weights: New weights which are set after the checks have been performed.
        :return: None.
        """
        # Check if the assigned object is a list.
        if not isinstance(new_weights, list):
            raise TypeError("The weights have to be a list.")

        # Loop through all the entries.
        for n, weight_matrix in enumerate(new_weights):
            # Check if each entry is a numpy array.
            if not isinstance(weight_matrix, np.ndarray):
                raise TypeError("The entries of the weight list have to be numpy arrays.")

            shape = weight_matrix.shape  # Save the shape of the matrix.

            # Check of the shape of each entry
            if not (len(shape) == 2 or len(shape) == 1):
                raise ValueError("All entries of weight list have to be one- or two-dimensional.")

            # Check if the entries a re numbers
            if not (weight_matrix.dtype == float or weight_matrix.dtype == int):
                raise TypeError("The entries of the weights have to be real numbers.")

        self._weights = new_weights

    @staticmethod
    def sigmoid_derivative(num):
        """
        Derivative of the Sigmoid function. Compatible with numpy arrays.

        :param num: A real number.
        :return: The derivative of the Sigmoid function at the point num.
        """
        return 1. / 2. * (1. + np.cosh(num))

    @staticmethod
    def sigmoid_function(num):
        """
        Tested.
        Sigmoid function compatible with numpy arrays.

        :param num: A number
        :return: The value of the sigmoid function using the num as input.
        """
        return 1. / (1. + np.exp(-num))

    def check_shapes(self) -> None:
        """
        Tested.
        This method checks if the shapes of the entries in weights and biases coincide with the entries in layer sizes.
        If this is not the case an according error is raised.

        :return: None
        """
        # Loop through all the entries in weights.
        for n, (weight_matrix, bias_vector) in enumerate(zip(self.weights, self.biases)):

            shape = weight_matrix.shape  # Shape of the matrix shape.

            # Check if the dimension of the weight matrices coincide with the layer sizes.
            if len(shape) == 2:
                if shape[0] != self.layer_sizes[n + 1] or shape[1] != self.layer_sizes[n]:
                    raise ValueError(f"Shapes {shape} of the {n}-th weight matrix does not coincide with the layer"
                                     f" sizes {(self.layer_sizes[n + 1], self.layer_sizes[n])} .")
            elif len(shape) == 1:
                if shape[0] != self.layer_sizes[n]:
                    raise ValueError(f"Shapes {shape} of the {n}-th weight matrix does not coincide with the layer"
                                     f"sizes {self.layer_sizes[n]}.")

            shape = bias_vector.shape  # Save the vector shape.

            # Check if the shape of the bias vector coincides with layer size.
            if shape[0] != self.layer_sizes[n + 1]:
                raise ValueError(f"Shape f{shape} of the {n}-th bias vector does not coincide with the {n + 1}-th layer"
                                 f"size {self.layer_sizes[n + 1]}.")

    def update(self, mat: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """
        Tested.
        Moves from on layer to the next, updating the current layer.

        :return: None.
        """
        # Update function from one layer to another.
        self.current_layer = self.sigmoid_function(np.dot(mat, self.current_layer) + bias)
        return self.current_layer

    def feed_forward(self, first_layer: np.ndarray) -> np.ndarray:
        """
        Tested.
        Runs the update() method repeatedly. How often the method is run is determined by the length of the layer_sizes
        attribute.

        :return: A list containing a one-dimensional numpy array containing the values of the corresponding layer.
        """
        self.check_shapes()  # Check the shapes.

        self.current_layer = first_layer  # Reset the current layer regardless if update was called before.

        # Return a list of numpy arrays corresponding to neurons in the according layer.
        for weight, bias in zip(self.weights, self.biases):
            self.update(weight, bias)

        return self.current_layer

    def learn(self, training_data: list, mini_batch_size: int, number_of_epochs: int, grad_step_size: float) \
            -> (list, list):
        """
        This method is the heart of this class. It "teaches" the neural network using the training data which is
        separated into mini batches of the size mini_batch_size. The weights and biases of the network are updated
        after each mini batch using gradient descent and back propagation.

        :param training_data: Tuples of training data containing the input and the desired output.
        :param mini_batch_size: Number of training examples in a mini batch.
        :param number_of_epochs: How many times the network runs through the training examples.
        :param grad_step_size: The step size used in the gradient descent. In formulas it is often the greek letter eta.
        :return: A tuple containing the weights an biases.
        """
        number_of_training_examples = len(training_data)
        for _ in range(number_of_epochs):

            rand.shuffle(training_data)  # Randomly shuffle the training data

            # Divide training data into mini batches.
            mini_batches = [training_data[x:x + mini_batch_size] for x in range(0, number_of_training_examples,
                                                                                mini_batch_size)]
            for mini_batch in mini_batches:
                # Updates the weights and biases after going through the training data of a mini batch.
                self.update_weight_and_biases(mini_batch, mini_batch_size, grad_step_size)

        return self.weights, self.biases

    def update_weight_and_biases(self, mini_batch: list, mini_batch_size: int, grad_step_size: float) -> None:
        """
        This method uses the data in a mini batch to update the weights and biases using the back propagation algorithm.

        :param mini_batch_size: Number of data examples in a mini batch.
        :param mini_batch: List of tuples containing the data and the desired result of the network corresponding to
            this data.
        :param grad_step_size: Step size used in the gradient descent. In formulae it often represented by the greek
            letter eta.
        :return: None.
        """
        # Creating the sum of the partial derivatives of the weights and biases for a mini batch, which will be
        # continuously updated via back propagation. These are later used to calculate the mean gradient of a
        # mini batch.
        weight_derivatives_sum = [np.zeros(weight.shape) for weight in self.weights]
        bias_derivatives_sum = [np.zeros(bias.shape) for bias in self.biases]

        for training_data, desired_result in mini_batch:
            # Derivatives of the cost function with regards to the individual weights and biases.
            weight_derivatives, bias_derivatives = self.back_propagation_algorithm(training_data, desired_result)

            # Update the sum of the derivatives with the new derivatives calculated by back propagation.
            weight_derivatives_sum = [delta_w_sum + delta_w for delta_w_sum, delta_w in zip(weight_derivatives_sum,
                                                                                            weight_derivatives)]
            bias_derivatives_sum = [delta_b_sum + delta_b for delta_b_sum, delta_b in zip(bias_derivatives_sum,
                                                                                          bias_derivatives)]

        const = grad_step_size / mini_batch_size  # Constant used for calculating the mean gradient.

        # Update the weights and biases of the network using back propagation.
        self.weights = [weight - const * delta_w for weight, delta_w in zip(self.weights, weight_derivatives_sum)]
        self.biases = [bias - const * delta_b for bias, delta_b in zip(self.biases, bias_derivatives_sum)]

    def back_propagation_algorithm(self, training_data: np.ndarray, desired_result: np.ndarray) -> (list, list):
        """
        The back propagation algorithm. Takes training data as an input an returns the partial derivatives of the
        weights and biases.

        :param training_data: Training data represented by a numpy array of floats.
        :param desired_result: Desired output
        :return: A tuple of lists of numpy arrays. The individual arrays are the weight matrices and bias vectors
            corresponding to a layer.
        """
        activations, z_values = self.calculate_a_and_z(training_data)  # Activations and z values.

        cost_func_grad = self.cost_func_grad(activations[-1], desired_result)  #

        delta_vec_last_layer = np.multiply(cost_func_grad,
                                           self.sigmoid_derivative(np.dot(self.weights[-1], activations[-2])
                                                                   + self.biases[-1]))

        deltas = self.calculate_deltas(delta_vec_last_layer, z_values)

        partial_weights = [np.outer(activation_vec, delta_vec) for activation_vec, delta_vec in
                           zip(activations[:-1], deltas)]

        return partial_weights, deltas

    def calculate_deltas(self, delta_last_layer: np.ndarray, z_values: list) -> list:
        """
        This method is part of the back propagation algorithm and calculates the deltas for all the layers in the neural
        network.

        :param delta_last_layer: The delta values of the last layer in the neural network.
        :param z_values: List of numpy arrays containing the z values for each layer of the network.
        :return: A matrix containing all the delta values for each layer
        """
        # This variable is used to save the current delta vector by the update_delta_vec function.
        current_delta_vec = delta_last_layer

        def update_delta_vec(weight_mat: np.ndarray, z_value_vec: np.ndarray) -> np.ndarray:
            """
            This is a help function to used to calculate the deltas in the l-th layer using the weight matrix from the
            (l + 1)-th layer and the z values of the l-th layer.

            :param weight_mat: weight matrix in the (l + 1)-th layer.
            :param z_value_vec: z values of the l-the layer.
            :return: Delta for the l-th layer.
            """
            nonlocal current_delta_vec
            current_delta_vec = np.dot(weight_mat.T, current_delta_vec) * self.sigmoid_derivative(z_value_vec)
            return current_delta_vec

        # A lot happens in the return statement. I will try to quickly sum it up. Since we are doing back propagation
        # the arguments in zip() ar reversed. Further, we drop the first element of the weights and the last element of
        # the z values since we wont be needing those during the calculation. Lastly the resulting list  is reversed
        # again to order the deltas by ascending layer index. Lastly the delta vector of the last layer is added to
        # return all the deltas.
        return [update_delta_vec(weight_mat, z_value_vec) for weight_mat, z_value_vec in
                zip(reversed(self.weights[1:]), reversed(z_values[:-1]))][::-1] + [delta_last_layer]

    def calculate_a_and_z(self, training_data: np.ndarray) -> (list, list):
        """
        This method is part of the back propagation algorithm and calculates the the activations of the neurons in a
        neural network as well as the corresponding values of z, which is the variable that is fed to sigmoid function:
        a(l+1) = sigmoid(z), z = w.a(l) + b.

        :return: Returns a tuple containing the activations and corresponding z values.
        """
        self.current_layer = training_data  # Set the current layer to the training data.

        # Calculate the activations for each layer in the neural network.
        activations = [self.sigmoid_function(training_data)] + [self.update(weight_mat, bias_vec) for
                                                                weight_mat, bias_vec in zip(self.weights, self.biases)]

        # Calculate z values using the activations. The last element is dropped since this is the output layer.
        z_values = [np.dot(weight_mat, activation) + bias_vec for weight_mat, bias_vec, activation in
                    zip(self.weights, self.biases, activations[:-1])]

        return activations, z_values

    @staticmethod
    def cost_func_grad(last_layer_activation, desired_result):
        """
        One component of the gradient of a quadratic cost function with respect to the activation in the last layer of
        the neural network.
        :param last_layer_activation:
        :param desired_result:
        :return:
        """
        return last_layer_activation - desired_result
