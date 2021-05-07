import numpy as np


class SimpleNeuralNetwork:

    def __init__(self, layer_sizes: np.ndarray, weights: np.array, biases: np.array):
        """
        This is the constructor for a simple feed forward neural network object. The neural network consist of
        individual layers of different which are connected through the weights and biases via linear equation.
        The result of this linear equation is then put in a Sigmoid function to amp it to the interval [0, 1].
        For further reading checkout the book http://neuralnetworksanddeeplearning.com/ by Michael Nielsen.

        :param layer_sizes: A 1D numpy array, containing the size (number of neurons) of the individual layers.
        :param weights: Weights connecting the neurons of one layer to the next via multiplication.
        :param biases: Biases added to neurons of each layer.
        """
        self.layer_sizes = layer_sizes
        self.current_layer = None
        self.weights = weights
        self.biases = biases

    @property
    def layer_sizes(self):
        return self._layer_sizes

    @layer_sizes.setter
    def layer_sizes(self, new_layer_sizes):
        """

        :param new_layer_sizes:
        :return:
        """
        if not isinstance(new_layer_sizes, np.ndarray):
            raise TypeError("The layer sizes have to be a numpy array.")
        if len(new_layer_sizes.shape) != 1:
            raise ValueError("The sizes of the layers has to be a one-dimensional array.")
        if new_layer_sizes.dtype != int:
            raise TypeError("Size of the layers have to of type int.")
        if not all(new_layer_sizes > 0):
            raise ValueError("The sizes of the layers have to be positive.")
        self._layer_sizes = new_layer_sizes

    @property
    def biases(self):
        return self._biases

    @biases.setter
    def biases(self, new_biases):
        """

        :param new_biases:
        :return:
        """
        if not isinstance(new_biases, np.ndarray):
            raise TypeError("All entries of the biases have to be numpy arrays.")
        if len(new_biases) != 2:
            raise ValueError("All entries of biases have to be one-dimensional.")
        if new_biases.dtype != float or new_biases.dtype != int:
            raise TypeError("The entries of the biases have to be real numbers.")
        for n, bias_vector in enumerate(new_biases):
            shape = bias_vector.shape
            if shape[0] != self.layer_sizes[n + 1]:
                raise ValueError(f"Shape f{shape} of the {n}-th bias vector does not coincide with the {n + 1}-th layer"
                                 f"size {self.layer_sizes[n + 1]}.")
        self._biases = new_biases

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, new_weights):
        """

        :param new_weights:
        :return:
        """
        if not isinstance(new_weights, np.ndarray):
            raise TypeError("The weights have to be a numpy array.")
        if len(new_weights.shape) != 3:
            raise ValueError("All entries of weights have to two-dimensional.")
        if new_weights.dtype != float or new_weights.dtype != int:
            raise TypeError("The entries of the weights have to be real numbers.")
        for n, weight_matrix in enumerate(new_weights):
            shape = weight_matrix.shape
            if shape[0] != self.layer_sizes[n + 1] or shape[1] != self.layer_sizes[n]:
                raise ValueError(f"Shapes f{shape} of the {n}-th weight matrix does not coincide with the layer sizes"
                                 f"{(self.layer_sizes[n + 1], self.layer_sizes[n])} .")
        self._weights = new_weights

    @staticmethod
    def sigmoid_function(num: float):
        """

        :param num:
        :return:
        """
        return 1 / (1 + np.exp(-num))

    def update(self, index: int) -> np.ndarray:
        """

        :return:
        """
        return self.sigmoid_function(np.dot(self.weights[index], self.current_layer) + self.biases[index])

    def run(self, network_input: np.ndarray) -> list:
        """

        :param: network_input
        :return:
        """
        if len(network_input) != len(self.layer_sizes[0]):
            raise ValueError(f"Network input and the input must be of the same length. They are {network_input} and"
                             f"{self.layer_sizes[0]}.")

        self.current_layer = network_input

        return [self.update(n) for n in range(len(self.layer_sizes))]
