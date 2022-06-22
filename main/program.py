from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt


def mnist_dataset():
    mnist = fetch_openml('mnist_784', version=1)
    return mnist


"""
Represents a column vector as a width x height matrix 
"""


def digit_to_image(column_vector, width, height):
    return column_vector.reshape(width, height)


"""
Accepts an image as N x M matrix with values [0-1] 
"""


def display_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    plt.show()


"""
Returns x_train, x_test, y_train, y_test, where each _train, _test has been split_at a given index
"""

def split_test_train(split_at, X, y):
    return X[:split_at], X[split_at:], y[:split_at], y[split_at:]
