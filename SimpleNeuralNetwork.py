import numpy as np


class SimpleNeuralNetwork:

    _layer_counter = 0

    def __init__(self, first_layer: np.ndarray, layer_sizes: np.ndarray, weights: list, biases: list) ->\
            None:
        """
        This is the constructor for a simple feed forward neural network object. The neural network consist of
        individual layers of different which are connected through the weights and biases via linear equation.
        The result of this linear equation is then put in a Sigmoid function to amp it to the interval [0, 1].
        For further reading checkout the book http://neuralnetworksanddeeplearning.com/ by Michael Nielsen.

        :param first_layer: The first layer of the neural network which corresponds to the input fed to the network.
        :param layer_sizes: A 1D numpy array, containing the size (number of neurons) of the individual layers.
        :param weights: Weights connecting the neurons of one layer to the next via multiplication.
        :param biases: Biases added to neurons of each layer.
        """
        self.current_layer = self.sigmoid_function(first_layer)
        self.layer_sizes = layer_sizes
        self.weights = weights
        self.biases = biases

    @property
    def layer_sizes(self) -> np.ndarray:
        return self._layer_sizes

    @layer_sizes.setter
    def layer_sizes(self, new_layer_sizes) -> None:
        """
        Setter method for the layer sizes. it is tested if the new layer sizes is a numpy, has an according shape and is
        filled with positive integers. If this is not the case, an according error is raised. If no error is raised, the
        layer sizes are updated.

        :param new_layer_sizes: Numpy array containing the size (number of neurons) of each layer of the neural network.
        :return: None
        """
        if not isinstance(new_layer_sizes, np.ndarray):
            raise TypeError("The layer sizes have to be a numpy array.")
        if len(new_layer_sizes.shape) != 1:
            raise ValueError("The sizes of the layers has to be a one-dimensional array.")
        if len(self.current_layer) != new_layer_sizes[self._layer_counter]:
            raise ValueError("The size of the current layer has to coincide with the corresponding value in the"
                             "layer_sizes array.")
        if new_layer_sizes.dtype != int:
            raise TypeError("Size of the layers have to of type int.")
        if not all(new_layer_sizes > 0):
            raise ValueError("The sizes of the layers have to be positive.")
        self._layer_sizes = new_layer_sizes

    @property
    def biases(self) -> np.ndarray:
        return self._biases

    @biases.setter
    def biases(self, new_biases) -> None:
        """
        Setter method for the biases used in the connections of the neural networks. Before a new array of biases is
        set, it is checked if it is a numpy array, has an according shape, is filled with real numbers and if the shapes
        of the biases conform with the sizes entered in the layer_sizes field.

        :param new_biases: New basis, replacing the old biases after the checks have been, which are described above.
        :return: None
        """
        if not isinstance(new_biases, list):
            raise TypeError("All entries of the biases have to be numpy arrays.")

        for n, bias_vector in enumerate(new_biases):
            shape = bias_vector.shape
            if shape[0] != self.layer_sizes[n + 1]:
                raise ValueError(f"Shape f{shape} of the {n}-th bias vector does not coincide with the {n + 1}-th layer"
                                 f"size {self.layer_sizes[n + 1]}.")
            if len(bias_vector.shape) != 1:
                raise ValueError("All entries of biases have to be one-dimensional.")
            if not (bias_vector.dtype == float or bias_vector.dtype == int):
                raise TypeError("The entries of the biases have to be real numbers.")

        self._biases = new_biases

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @weights.setter
    def weights(self, new_weights) -> None:
        """
        Setter method for the weights which are used in the connection of the layers. Before a the weights are set a
        number of checks is performed. These checks include if the newly entered weights are in a numpy array, if this
        array has the right shape, if the numpy is filled with numbers and if the shapes of the individual weight
        matrices correspond to the number of neurons delcared by the layer_sizes array.

        :param new_weights: New weights which are set after the checks have been performed.
        :return: None.
        """
        if not isinstance(new_weights, list):
            raise TypeError("The weights have to be a list.")

        for n, weight_matrix in enumerate(new_weights):
            shape = weight_matrix.shape
            if not (len(shape) == 2 or len(shape) == 1):
                raise ValueError("All entries of weight list have to be two-dimensional.")
            if not (weight_matrix.dtype == float or weight_matrix.dtype == int):
                raise TypeError("The entries of the weights have to be real numbers.")
            if len(shape) == 2:
                if shape[0] != self.layer_sizes[n + 1] or shape[1] != self.layer_sizes[n]:
                    raise ValueError(f"Shapes {shape} of the {n}-th weight matrix does not coincide with the layer sizes"
                                     f"{(self.layer_sizes[n + 1], self.layer_sizes[n])} .")
            elif len(shape) == 1:
                if shape[0] != self.layer_sizes[n]:
                    raise ValueError(f"Shapes {shape} of the {n}-th weight matrix does not coincide with the layer sizes"
                                     f"{self.layer_sizes[n]}.")

        self._weights = new_weights

    @staticmethod
    def sigmoid_function(num: np.ndarray) -> np.ndarray:
        """
        Sigmoid function compatible with numpy arrays.

        :param num: A number
        :return: The value of the sigmoid function using the num as input.
        """
        return 1. / (1. + np.exp(-num))

    def update(self) -> np.ndarray:
        """
        Moves from on layer to the next, updating the current layer.

        :return: The new layer, which is now the current layer.
        """
        self.current_layer = self.sigmoid_function(np.dot(self.weights[self._layer_counter], self.current_layer) +
                                                   self.biases[self._layer_counter])
        self._layer_counter += 1
        return self.current_layer

    def run(self) -> list:
        """
        Runs the update() method repeatedly. How often the method is run is determined by the length of the layer_sizes
        attribute.

        :return: A list containing a one-dimensional numpy array containing the values of the corresponding layer.
        """
        return [self.update() for _ in range(len(self.layer_sizes) - 1)]
