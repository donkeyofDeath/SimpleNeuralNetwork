import numpy as np


class SimpleNeuralNetwork:

    def __init__(self, layer_sizes: np.ndarray, weights: list, biases: list):
        """

        :param layer_sizes:
        :param weights:
        :param biases:
        """
        self.layer_sizes = layer_sizes
        self.current_layer = None
        self.weights = weights
        self.biases = biases

    @property
    def biases(self):
        return self._biases

    @biases.setter
    def biases(self, new_biases):
        for n, bias_vector in enumerate(new_biases):
            if not isinstance(bias_vector, np.ndarray):
                raise TypeError("All entries of the weights have to be numpy arrays.")
            shape = bias_vector.shape
            if len(shape) != 1:
                raise ValueError("All entries of weights have to be one-dimensional.")
            if shape[0] != self.layer_sizes[n + 1]:
                raise ValueError(f"Shape f{shape} of the {n}-th bias vector does not coincide with the {n + 1}-th layer"
                                 f"size {self.layer_sizes[n + 1]}.")
        self._biases = new_biases

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, new_weights):
        for n, weight_matrix in enumerate(new_weights):
            shape = weight_matrix.shape
            if not isinstance(weight_matrix, np.ndarray):
                raise TypeError("All entries of the weights have to be numpy arrays.")
            if len(shape) != 2:
                raise ValueError("All entries of weights have to two-dimensional.")
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
