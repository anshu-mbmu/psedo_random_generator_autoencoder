import tensorflow as tf
import numpy as np

def load_and_preprocess_data():
    """Loads and preprocesses the Fashion MNIST dataset."""
    # Load the Fashion MNIST dataset
    (x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()

    # Normalize the pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # Reshape the data to flatten the images into vectors
    x_train = x_train.reshape((len(x_train), 28 * 28))
    x_test = x_test.reshape((len(x_test), 28 * 28))

    return x_train, x_test

# Move the assignment of x_train and x_test outside the if __name__ == "__main__": block
print("Loading and preprocessing data...")
x_train, x_test = load_and_preprocess_data()
print(f"Shape of x_train: {x_train.shape}")
print(f"Shape of x_test: {x_test.shape}")
