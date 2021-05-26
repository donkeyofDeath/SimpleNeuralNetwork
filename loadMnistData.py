from keras.datasets import mnist
import numpy as np
from typing import Tuple, List


def convert_number(number: int) -> np.ndarray:
    """
    This function converts a number of the training data of the expected results to a numpy array with a one at the
    index num.

    :return: Numpy array with a 1 at the index num.
    """
    vec = np.zeros(10)
    vec[number] = 1.
    return vec


def load_data() -> Tuple[int, list, list]:
    """
    Loads the training data and verification data of the MNIST library. The images of the MNIST libraries are
    represented by 1D numpy arrays with 8 Bit values representing the gray scale of the pixels in the images.


    :return: Returns the number of pixels in the images, a list of tuples of numpy arrays containing the training inputs
        and the corresponding correct output and a list of tuples of lists containing the verification input
        (numpy array) and an int representing the number described by the corresponding numpy array.
    """
    num_pixels = 784  # Number  of pixels in a image.

    # train_inputs = List of the gray scale (8 Bit) of a quadratic image with 784 pixels.
    # desired_result = List of the numbers that the images in train_input are representing.
    # test_inputs = List of the gray scale (8 Bit) of a quadratic image with 784 pixels.
    # test_results = List of numbers that the images in test_inputs are representing.
    (train_inputs, desired_results), (test_inputs, test_results) = mnist.load_data()  # Load the test data

    # Convert the elements of the desired results from numbers to numpy arrays.
    converted_desired_results = np.array([convert_number(num) for num in desired_results])
    # Reshape the training input so that each entry is a 1D vector.
    train_inputs = train_inputs.reshape(len(train_inputs), num_pixels)
    # Make the training data a list of tuples of numpy arrays.
    training_data = list(zip(train_inputs / 255., converted_desired_results))

    # Reshape the test inputs so that the entries are 1D instead of 2D.
    test_inputs = test_inputs.reshape(len(test_inputs), num_pixels)
    # Create a list of tuples with the input and the corresponding result
    verification_data = list(zip(test_inputs / 255., test_results))

    return num_pixels, training_data, verification_data


def load_data_2() -> Tuple[int, List[Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """
    Loads the training data and verification data of the MNIST library. The images of the MNIST libraries are
    represented by 1D numpy arrays with 8 Bit values representing the gray scale of the pixels in the images.


    :return: Returns the number of pixels in the images, a list of tuples of numpy arrays containing the training inputs
        and the corresponding correct output and a two numpy arrays containing the verification data and corresponding
        result.
    """
    num_pixels = 784  # Number  of pixels in a image.

    # train_inputs = List of the gray scale (8 Bit) of a quadratic image with 784 pixels.
    # desired_result = List of the numbers that the images in train_input are representing.
    # test_inputs = List of the gray scale (8 Bit) of a quadratic image with 784 pixels.
    # test_results = List of numbers that the images in test_inputs are representing.
    (train_inputs, desired_results), (test_inputs, test_results) = mnist.load_data()  # Load the test data

    # Convert the elements of the desired results from numbers to numpy arrays.
    converted_desired_results = np.array([convert_number(num) for num in desired_results])
    # Reshape the training input so that each entry is a 1D vector.
    train_inputs = train_inputs.reshape(len(train_inputs), num_pixels)
    # Make the training data a list of tuples of numpy arrays.
    training_data = list(zip(train_inputs / 255., converted_desired_results))

    # Reshape the test inputs so that the entries are 1D instead of 2D.
    test_inputs = test_inputs.reshape(len(test_inputs), num_pixels)
    print(test_inputs.T.shape)

    return num_pixels, training_data, test_inputs, test_results


if __name__ == "__main__":
    load_data_2()
