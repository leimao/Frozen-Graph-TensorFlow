
import tensorflow as tf
import numpy as np


def train_test_split(x, y, train_fraction = 0.9):

    # Split the data into training data and test data
    assert len(x) == len(y)
    dataset_size = len(x)
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    idx_split = int(dataset_size * train_fraction)
    x_train = x[:idx_split]
    y_train = y[:idx_split]
    x_test = x[idx_split:]
    y_test = y[idx_split:]

    return (x_train, y_train), (x_test, y_test)


class CIFAR10(object):

    def __init__(self, train_fraction = 0.9):

        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
        (self.x_train, self.y_train), (self.x_valid, self.y_valid) = train_test_split(x = self.x_train, y = self.y_train, train_fraction = train_fraction)

        assert np.array_equal(np.unique(self.y_train), np.unique(self.y_test)) == True

        self.num_classes = len(np.unique(self.y_train))

        self.input_size = list(self.x_train.shape[1:])

        # Convert integer label to binary vector
        self.y_train_onehot = tf.keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_valid_onehot = tf.keras.utils.to_categorical(self.y_valid, self.num_classes)
        self.y_test_onehot = tf.keras.utils.to_categorical(self.y_test, self.num_classes)
        # Image scaling
        self.x_train = self.x_train.astype('float32')
        self.x_valid = self.x_valid.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_valid /= 255
        self.x_test /= 255


