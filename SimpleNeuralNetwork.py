import numpy as np
import random as rand
from typing import Tuple, List
import loadMnistData as lmd
import time as tm


def convert_array(array: np.array) -> np.array:
    """
    Convert a 1D numpy array into a 2D matrix with one column.

    :param array: 1D numpy array
    :return: 2D numpy column array with the same entry.
    """
    return array.reshape((len(array), 1))


class SimpleNeuralNetwork:

    def __init__(self, layer_sizes: np.ndarray, weights: List[np.ndarray] = None, biases: List[np.ndarray] = None) \
            -> None:
        """
        Tested.
        This is the constructor for a simple feed forward neural network object. The neural network consist of
        individual layers of different size which are connected through the weights and biases via linear equations.
        The result of this calculation is then put in a Sigmoid function to amp it to the interval [0, 1].
        For further reading checkout the book http://neuralnetworksanddeeplearning.com/ by Michael Nielsen.
        If no weights and biases are provided the network is initialized with random weights and biases obeying a
        Gaussian distribution. The distribution of the biases have a mean value of 0 and a variance of 1, while the
        distribution of the weights also has the mean 0 but the variance is given by 1/N where N is the number of input
        neurons of a layer.

        :param layer_sizes: A 1D numpy array, containing the size (number of neurons) of the individual layers.
        :param weights: List of weight matrices (2D numpy arrays) connecting the layers of the network via
            multiplication.
        :param biases: List of bias vectors (1D numpy arrays) added to neurons of each layer.
        """

        self.current_layer = None
        self.layer_sizes = layer_sizes

        # Set weights randomly if no weights are provided.
        if weights is None:
            self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.layer_sizes[:-1],
                                                                               self.layer_sizes[1:])]
        else:
            self.weights = weights

        # Set biases randomly if no biases are provided.
        if biases is None:
            self.biases = [np.random.randn(y) for y in self.layer_sizes[1:]]
        else:
            self.biases = biases

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
    def transform_matrix(mat: np.ndarray, num_of_mats: int) -> np.ndarray:
        """
        TODO: Test this method.s
        Transposes a matrix and divides it along the column into num_of_mats many sub matrices.

        :param mat: Matrix to be turned into a array of matrices.
        :param num_of_mats: Number of matrices.
        :return: A 3D numpy array
        """
        rows, columns = mat.shape
        return mat.reshape(num_of_mats, rows // num_of_mats, columns).transpose(0, 2, 1)

    @staticmethod
    def reverse_transform_tensor(tensor: np.ndarray) -> np.ndarray:
        """
        TODO: Test this method.
        This function takes in a 3D numpy array and flattens it one the first level, making it a 2D numpy array.
        This function basically concatenates all the matrices together.

        :param tensor: A 3D numpy array which can be seen as a list of matrices.
        :return: A flattened version of the 3D numpy array which is now 2D
        """
        num_mats, num_rows, num_cols = tensor.shape
        return tensor.transpose(0, 2, 1).reshape(num_cols * num_mats, num_rows).T
        # return np.concatenate(tensor, axis=1)

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
    def calc_delta(activation_vec: np.ndarray, result_vec: np.ndarray) -> np.ndarray:
        """
        TODO: Test this method.
        This method calculates the value of delta for a cross entropy cost function.

        :param activation_vec: Numpy array containing the activations of each layer.
        :param result_vec: The desired results corresponding to each input.
        :return: The associated value of the parameter delta.
        """
        return activation_vec - result_vec

    @staticmethod
    def cross_entropy_cost(act_mat: np.ndarray, res_mat: np.ndarray) -> float:
        """
        TODO: Test this method.
        Formula for a cross entropy cost function. The nan_to_num function is used so a number is returned.

        :param act_mat: 2D Numpy array containing the activations of the last layer of the neural network.
        :param res_mat: 2D Numpy array containing the desired results for each input.
        :return: The associated cross entropy cost function.
        """
        return np.sum(np.nan_to_num(-res_mat * np.log(act_mat) - (1 - res_mat) * np.log(1 - act_mat)).sum(axis=0)) / \
               act_mat.shape[1]

    @staticmethod
    def calc_accuracy(last_layer_activation: np.ndarray, written_numbers: np.ndarray) -> int:
        """
        TODO: Test this method.
        This method takes in a 2D numpy array of inputs in this array each column represents on data input. The desired
        results is 1D numpy array of integers between 0 and 9. These numbers represent the correct output of the
        network. The method calculates how many inputs have the correct output and returns the number.

        :param last_layer_activation: 2D numpy array in which each column represents the activation in the last layer.
        :param written_numbers: 1D numpy array representing the correct output for each input (hand written number).
        :return: The number of correctly verified inputs.
        """
        return np.sum(written_numbers == np.argmax(last_layer_activation, axis=0))

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

    def feed_forward(self, first_layer: np.ndarray) -> np.ndarray:
        """
        Tested.
        Feeds a collection of inputs through the network. The input is represented by a 2D numpy array in which a
        column represents a single input.

        :return: A 2D numpy array containing all the outputs of the network. A single output is represented by a column.
        """
        self.current_layer = first_layer  # Reset the current layer regardless if update was called before.

        # Return a list of numpy arrays corresponding to neurons in the according layer.
        for weight, bias in zip(self.weights, self.biases):
            self.current_layer = self.sigmoid_function(np.dot(weight, self.current_layer) + bias[:, np.newaxis])

        return self.current_layer

    def learn(self, learning_data: list, mini_batch_size: int, number_of_epochs: int, learning_rate: float,
              reg_param: float, shuffle_flag: bool = True, verification_data: Tuple[np.ndarray, np.ndarray] = None,
              monitor_training_cost_flag=False, monitor_training_accuracy_flag=False,
              monitor_verification_cost_flag=False, monitor_verification_accuracy_flag=False) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        TODO: Test the newly defined flags.
        Tested.
        This method is the heart of this class. It "teaches" the neural network using the training data which is
        separated into mini batches of the size mini_batch_size. The weights and biases of the network are updated
        after each mini batch using gradient descent which itself is using the back propagation algorithm. A cross
        entropy cost function is used for the learning procedure as well L2 regularization. The method returns the data
        which is set by the flags to be returned. If no flags are the set method returns four empty lists.

        :param learning_data: Data containing tuples of three elements. The first element is a numpy array and contains
            the input data for the neural network. The second element is also a numpy array contains the desired output
            of the network corresponding to the input. The last entry of the tuple is a number representing the written
            number which is represented by the input data (first numpy array).
        :param mini_batch_size: Number of inputs in a mini batch.
        :param number_of_epochs: Number of epochs that are executed.
        :param learning_rate: Learning rate used in the gradient descent. This number is often declared as the greek
            letter eta in formulae.
        :param reg_param: Parameter associated with the L2 regularization in formula it is often presented by a lambda.
        :param shuffle_flag: If this flag is true the input data is shuffled. If the value of the flag is False, the
            learning data is processed as is.
        :param verification_data: This data has the same format as the learning_data. If data is provided, it is used to
            see how many images are verified correctly.
        :param monitor_training_cost_flag: Monitoring flag of the cost function of the training data for each epoch.
        :param monitor_training_accuracy_flag: Monitoring flag of the accuracy on the training data for each epoch.
        :param monitor_verification_cost_flag: Monitoring flag of the cost function of the verification for each epoch.
        :param monitor_verification_accuracy_flag: Monitoring flag of the accuracy on the verification data for each epoch.
        :return: Returns four numpy arrays containing the data according to the set flags. Flags which were not set
            return empty lists. The data is taken after a training epoch is completed. The data arrays are returned in
            the order: verification accuracy, verification cost, training accuracy, training cost.
        """

        # ------
        # Set up
        # ------

        self.check_shapes()  # Check the shapes.
        number_of_training_examples = len(learning_data)
        # Save this flag to later execute computationally heavy code only once.
        training_flag = monitor_training_cost_flag or monitor_training_accuracy_flag

        # Empty lists which are filled if the associated flags are provided.
        verification_cost, verification_accuracy, training_cost, training_accuracy = [], [], [], []

        # Test if the mini batch size divides the number of training examples if not an error is raised.
        if number_of_training_examples % mini_batch_size == 0:
            number_of_mini_batches = int(number_of_training_examples / mini_batch_size)
        else:
            raise ValueError("The mini batch size has to divide the number of training examples.")

        # ---------------------------
        # Stochastic gradient descent
        # ---------------------------

        # Iterate through all epochs.
        for index in range(number_of_epochs):
            # This line is mostly here for testing reasons.
            if shuffle_flag:
                # Randomly shuffle the training data and the according results.
                rand.shuffle(learning_data)

            # Recombine the shuffled data.
            input_data, desired_output, numbers = zip(*learning_data)

            # Divide training data into mini batches of the same size and convert them into numpy arrays.
            input_data = self.transform_matrix(np.array(input_data), number_of_mini_batches)
            desired_output = self.transform_matrix(np.array(desired_output), number_of_mini_batches)

            if training_flag:
                numbers = np.array(numbers)  # Convert back to a numpy array.
                # Declare empty array which will be filled with the output data of the network.
                output_data = np.zeros((number_of_mini_batches, self.layer_sizes[-1], mini_batch_size))

            # Updates the weights and biases after going through the training data of a mini batch.
            for n, (input_data_mat, desired_result_mat) in enumerate(zip(input_data, desired_output)):
                # Save the activation of the last layer
                act = self.update_weights_and_biases((input_data_mat, desired_result_mat), mini_batch_size,
                                                     learning_rate, reg_param, number_of_training_examples,
                                                     output_flag=training_flag)
                if training_flag:
                    output_data[n] = act  # Append the activation to the array containing all the outputs.

            # ----------
            # Monitoring
            # ----------

            # Only do this if one of the flags is raised, since this part is very computationally expensive.
            if training_flag:
                # Declare the input and output data.
                desired_output = self.reverse_transform_tensor(desired_output)
                output_data = self.reverse_transform_tensor(output_data)
                if monitor_training_accuracy_flag:
                    # Ratio of correctly verified training examples.
                    train_ratio_test = self.calc_accuracy(output_data, numbers) / number_of_training_examples
                    training_accuracy.append(train_ratio_test)
                if monitor_training_cost_flag:
                    training_cost.append(self.cross_entropy_cost(output_data, desired_output))

            # Use verification data if it is provided.
            if verification_data is not None:
                verification_output = self.feed_forward(verification_data[0])  # Count the correctly classified results.
                # Calculate the ratio of correctly identified verification examples.
                verification_ratio = self.calc_accuracy(verification_output, verification_data[1]) / len(verification_data[1])
                if monitor_verification_accuracy_flag:
                    verification_accuracy.append(verification_ratio)
                if monitor_verification_cost_flag:
                    # Convert the list of numbers to the corresponding outputs of the network and calculate the cross
                    # entropy cost function.
                    result = np.apply_along_axis(lmd.convert_number, 1, np.atleast_2d(verification_data[1]).T)
                    verification_cost.append(self.cross_entropy_cost(verification_output, result.T))

                # Print the result of how many images are identified correctly.
                print(f"Epoch {index + 1}: {100 * verification_ratio:.2f} %.")
            else:
                print(f"Epoch {index + 1} finished.")

        return np.array(training_cost), np.array(training_accuracy), np.array(verification_cost), \
               np.array(verification_accuracy)

    def update_weights_and_biases(self, mini_batch: Tuple[np.ndarray, np.ndarray], mini_batch_size: int,
                                  learning_rate: float, reg_param: float, data_size: int,
                                  output_flag: bool = False) -> np.ndarray:
        """
        Tested.
        Updates the weights and biases of the network using gradient descent and the back propagation algorithm.
        The algorithm is implemented to operate mostly on matrix multiplications using numpy as well as possible.
        A cross entropy cost function is used for the learning procedure as well L2 regularization.
  
        :param mini_batch: List of tuples containing the data and the desired result of the network corresponding to
            this data. The data is represented by numpy arrays.
        :param mini_batch_size: Number of elements in a mini_batch.
        :param learning_rate: The learning rate used to train the network using gradient descent. In formulae it is
            often declared as an eta.
        :param reg_param: The variable lambda used in the L2 regularization formula.
        :param data_size: Number of elements in the training data.
        :param output_flag: If this flag is raised the activations of the last layer of each call are returned.
        :return: If the output_flag is raised, the last layer of the activations is returned. If it isn't raised the
            method returns None.
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

        # Initialize the list of deltas with the delta values of the last layer.
        # deltas = [np.multiply(self.cost_func_grad(activations[-1], mini_batch[1]), self.sigmoid_derivative(z_values[-1]))]
        deltas = [self.calc_delta(activations[-1], mini_batch[1])]

        # Calculate the delta values.
        for weight_mat, z_value_mat in zip(reversed(self.weights[1:]), reversed(z_values[:-1])):
            deltas.insert(0, np.multiply(np.dot(weight_mat.T, deltas[0]), self.sigmoid_derivative(z_value_mat)))

        # -------------------------
        # Update weights and biases
        # -------------------------

        const = learning_rate / mini_batch_size  # Constant used to calculate the mean gradient.

        # Update the weights and the biases using the mean gradient.
        # The first term in formula of the weights is due to the L2 regularization.
        self.weights = [(1 - (learning_rate * reg_param / data_size)) * weight_mat - const *
                        np.dot(delta_mat, activation_mat.T) for delta_mat, activation_mat, weight_mat in
                        zip(deltas, activations[:-1], self.weights)]
        self.biases = [bias_vec - const * delta_mat.sum(axis=1) for delta_mat, bias_vec in zip(deltas, self.biases)]

        if output_flag:
            return activations[-1]
