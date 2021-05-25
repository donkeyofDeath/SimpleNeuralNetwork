import numpy as np
import random as rand
from typing import Tuple


def convert_array(array: np.array) -> np.array:
    """
    Convert a 1D numpy array into a 2D matrix with one column.

    :param array: 1D numpy array
    :return: 2D numpy column array with the same entry.
    """
    return array.reshape((len(array), 1))


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

    # ----------
    # Properties
    # ----------

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

        self._layer_sizes = new_layer_sizes

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

    # --------------
    # Static methods
    # --------------

    @staticmethod
    def sigmoid_derivative(num):
        """
        Tested.
        Derivative of the Sigmoid function. Compatible with numpy arrays.

        :param num: A real number.
        :return: The derivative of the Sigmoid function at the point num.
        """
        return 0.5 / (1. + np.cosh(num))

    @staticmethod
    def sigmoid_function(num):
        """
        Tested.
        Sigmoid function compatible with numpy arrays.

        :param num: A number
        :return: The value of the sigmoid function using the num as input.
        """
        return 1. / (1. + np.exp(-num))

    @staticmethod
    def cost_func_grad(last_layer_activation, desired_result):
        """
        Tested.
        One component of the gradient of a quadratic cost function with respect to the activation in the last layer of
        the neural network.

        :param last_layer_activation: The activation of the neurons in the last layer of the neural network.
        :param desired_result: The desired activations of the neurons in the last layer of the neural network.
        :return: last_layer_activation - desired_result.
        """
        return last_layer_activation - desired_result

    # --------------
    # Normal Methods
    # --------------

    def check_shapes(self) -> None:
        """
        Tested.
        This method checks if the shapes of the entries in weights and biases coincide with the entries in layer sizes.
        If this is not the case an according error is raised.

        :return: None.
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

        # Reset the current layer regardless if update was called before.
        self.current_layer = first_layer

        # Return a list of numpy arrays corresponding to neurons in the according layer.
        for weight, bias in zip(self.weights, self.biases):
            self.update(weight, bias)

        return self.current_layer

    def learn(self, learning_data: list, mini_batch_size: int, number_of_epochs: int, learning_rate: float,
              shuffle_flag: bool = True, verification_data: list = None) -> None:
        """
        Tested.
        This method is the heart of this class. It "teaches" the neural network using the training data which is
        separated into mini batches of the size mini_batch_size. The weights and biases of the network are updated
        after each mini batch using gradient descent which itself is using the back propagation algorithm.

        :param learning_data: Data containing tuples of numpy arrays. The first numpy array contains the input data for
            the neural network. The second numpy array contains the desired output of the network corresponding to the
            the input.
        :param mini_batch_size: Number of inputs in a mini batch.
        :param number_of_epochs: Number of epochs that are executed.
        :param learning_rate: Learning rate used in the gradient descent. This number is often declared as the greek
            letter eta in formulae.
        :param shuffle_flag: If this flag is true the input data is shuffled. If the value of the flag is False, the
            learning data is processed as is.
        :param verification_data: This data has the same format as the learning_data. If data is provided, it is used to
            see how many images are verified correctly.
        :return: None
        """
        number_of_training_examples = len(learning_data)

        # Test if the mini batch size divides the number of training examples if not an error is raised.
        if number_of_training_examples % mini_batch_size == 0:
            number_of_mini_batches = int(number_of_training_examples / mini_batch_size)
        else:
            raise ValueError("The mini batch size has to divide the number of training examples.")

        # Iterate through all epochs.
        for index in range(number_of_epochs):
            # This line is mostly here for testing reasons.
            if shuffle_flag:
                # Randomly shuffle the training data.
                rand.shuffle(learning_data)

            # Recombine the shuffled data.
            input_data, desired_results = zip(*learning_data)

            # Divide training data into mini batches of the same size.
            input_data = np.array(input_data).reshape(number_of_mini_batches, mini_batch_size, self.layer_sizes[0])
            desired_results = np.array(desired_results).reshape(number_of_mini_batches, mini_batch_size,
                                                                self.layer_sizes[-1])

            # Updates the weights and biases after going through the training data of a mini batch.
            for input_data_mat, desired_result_mat in zip(input_data, desired_results):
                # The data needs to be transposed since to have the right format for the matrix multiplications,
                self.update_weights_and_biases((input_data_mat.T, desired_result_mat.T), mini_batch_size,
                                               learning_rate)

            # Use verification data if it provided.
            # TODO: I could probably make this much more efficient using matrix multiplication.
            if verification_data is not None:
                # Count the correctly classified results.
                counter = sum([result == np.nanargmax(self.feed_forward(data)) for data, result in verification_data])
                # Print the result of how many images are identified correctly.
                print(f"Epoch {index}: {counter} out of {len(verification_data)}.")
            else:
                print(f"Epoch {index} finished.")

    def update_weights_and_biases(self, mini_batch: Tuple[np.ndarray, np.ndarray], mini_batch_size: int,
                                  learning_rate: float) -> None:
        """
        Tested.
        Updates the weights and biases of the network using gradient descent and the back propagation algorithm.
        The algorithm is implemented to operate mostly on matrix multiplications using numpy as well as possible.

        :param mini_batch: List of tuples containing the data and the desired result of the network corresponding to
            this data. The data is represented by numpy arrays.
        :param mini_batch_size: Number of elements in a mini_batch.
        :param learning_rate: The learning rate used to train the network using gradient descent. In formulae it is
            often declared as an eta.
        :return: None.
        """
        activations = [mini_batch[0]]  # List containing the activations of all inputs for each layer.
        z_values = []  # List containing the z values of all inputs for each layer.

        # --------------------------
        # Back propagation algorithm
        # --------------------------

        # Calculate the activations and the z values.
        for weight_mat, bias_vec in zip(self.weights, self.biases):
            z_value_mat = np.dot(weight_mat, activations[-1]) + bias_vec[:, np.newaxis]
            z_values.append(z_value_mat)
            activations.append(self.sigmoid_function(z_value_mat))

        # The delta values of the last layer.
        deltas = [np.multiply(self.cost_func_grad(activations[-1], mini_batch[1]), self.sigmoid_derivative(z_values[-1]))]

        # Calculate the delta values.
        for weight_mat, z_value_mat in zip(reversed(self.weights[1:]), reversed(z_values[:-1])):
            deltas.insert(0, np.multiply(np.dot(weight_mat.T, deltas[0]), self.sigmoid_derivative(z_value_mat)))

        # -------------------------
        # Update weights and biases
        # -------------------------

        const = learning_rate / mini_batch_size  # Constant used to calculate the mean gradient.

        # Update the weights and the biases.
        self.weights = [weight_mat - const * np.dot(delta_mat, activation_mat.T) for
                        delta_mat, activation_mat, weight_mat in zip(deltas, activations[:-1], self.weights)]
        self.biases = [bias_vec - const * delta_mat.sum(axis=1) for delta_mat, bias_vec in zip(deltas, self.biases)]
