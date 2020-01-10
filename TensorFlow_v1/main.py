import os
import argparse
import tensorflow as tf
import numpy as np

from cnn import CNN
from cifar import CIFAR10
from utils import plot_curve, model_accuracy


def train(learning_rate, learning_rate_decay, dropout_rate, mini_batch_size,
          epochs, optimizer, random_seed, model_directory, model_filename,
          log_directory):

    np.random.seed(random_seed)

    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Load CIFAR10 dataset
    cifar10 = CIFAR10()
    x_train = cifar10.x_train
    y_train = cifar10.y_train
    y_train_onehot = cifar10.y_train_onehot
    x_valid = cifar10.x_valid
    y_valid = cifar10.y_valid
    y_valid_onehot = cifar10.y_valid_onehot

    num_classes = cifar10.num_classes
    input_size = cifar10.input_size

    print('CIFAR10 Input Image Size: {}'.format(input_size))

    model = CNN(input_size=input_size,
                num_classes=num_classes,
                optimizer=optimizer)

    train_accuracy_log = list()
    valid_accuracy_log = list()
    train_loss_log = list()

    for epoch in range(epochs):
        print('Epoch: %d' % epoch)

        learning_rate *= learning_rate_decay
        # Prepare mini batches on train set
        shuffled_idx = np.arange(len(x_train))
        np.random.shuffle(shuffled_idx)
        mini_batch_idx = [
            shuffled_idx[k:k + mini_batch_size]
            for k in range(0, len(x_train), mini_batch_size)
        ]

        # Validate on validation set
        valid_prediction_onehot = model.test(data=x_valid)
        valid_prediction = np.argmax(valid_prediction_onehot, axis=1).reshape(
            (-1, 1))
        valid_accuracy = model_accuracy(label=y_valid,
                                        prediction=valid_prediction)
        print('Validation Accuracy: %f' % valid_accuracy)
        valid_accuracy_log.append(valid_accuracy)

        # Train on train set
        for i, idx in enumerate(mini_batch_idx):
            train_loss = model.train(data=x_train[idx],
                                     label=y_train_onehot[idx],
                                     learning_rate=learning_rate,
                                     dropout_rate=dropout_rate)
            if i % 200 == 0:
                train_prediction_onehot = model.test(data=x_train[idx])
                train_prediction = np.argmax(train_prediction_onehot,
                                             axis=1).reshape((-1, 1))
                train_accuracy = model_accuracy(label=y_train[idx],
                                                prediction=train_prediction)
                print('Training Loss: %f, Training Accuracy: %f' %
                      (train_loss, train_accuracy))
                if i == 0:
                    train_accuracy_log.append(train_accuracy)
                    train_loss_log.append(train_loss)

    model.save(directory=model_directory, filename=model_filename)
    print('Trained model saved successfully')

    model.save_as_pb(directory=model_directory, filename=model_filename)
    print('Trained model saved as pb successfully')

    # The directory should not exist before calling this method
    signature_dir = os.path.join(model_directory, 'signature')
    assert (not os.path.exists(signature_dir))
    model.save_signature(directory=signature_dir)
    print('Trained model with signature saved successfully')

    plot_curve(train_losses = train_loss_log, train_accuracies = train_accuracy_log, valid_accuracies = valid_accuracy_log, \
        filename = os.path.join(log_directory, 'training_curve.png'))


def test(model_file):

    tf.reset_default_graph()

    # Load CIFAR10 dataset
    cifar10 = CIFAR10()
    x_test = cifar10.x_test
    y_test = cifar10.y_test
    y_test_onehot = cifar10.y_test_onehot
    num_classes = cifar10.num_classes
    input_size = cifar10.input_size

    model = CNN(input_size=input_size,
                num_classes=num_classes,
                optimizer='Adam')
    model.load(filepath=model_file)

    test_prediction_onehot = model.test(data=x_test)
    test_prediction = np.argmax(test_prediction_onehot, axis=1).reshape(
        (-1, 1))
    test_accuracy = model_accuracy(label=y_test, prediction=test_prediction)

    print('Test Accuracy: %f' % test_accuracy)


def main():
    # Default settings
    learning_rate_default = 0.001
    learning_rate_decay_default = 0.9
    dropout_rate_default = 0.5
    mini_batch_size_default = 64
    epochs_default = 30
    optimizer_default = 'Adam'
    random_seed_default = 0
    model_directory_default = 'model'
    model_filename_default = 'cifar10_cnn'
    log_directory_default = 'log'

    # Argparser
    parser = argparse.ArgumentParser(
        description='Train CNN on CIFAR10 dataset.')

    parser.add_argument('-train',
                        '--train',
                        help='train model',
                        action='store_true')
    parser.add_argument('-test',
                        '--test',
                        help='test model',
                        action='store_true')
    parser.add_argument('--lr',
                        type=float,
                        help='initial learning rate',
                        default=learning_rate_default)
    parser.add_argument('--lr_decay',
                        type=float,
                        help='learning rate decay',
                        default=learning_rate_decay_default)
    parser.add_argument('--dropout',
                        type=float,
                        help='dropout rate',
                        default=dropout_rate_default)
    parser.add_argument('--batch_size',
                        type=int,
                        help='mini batch size',
                        default=mini_batch_size_default)
    parser.add_argument('--epochs',
                        type=int,
                        help='number of epochs',
                        default=epochs_default)
    parser.add_argument('--optimizer',
                        type=str,
                        help='optimizer',
                        default=optimizer_default)
    parser.add_argument('--seed',
                        type=int,
                        help='random seed',
                        default=random_seed_default)
    parser.add_argument('--model_dir',
                        type=str,
                        help='model directory',
                        default=model_directory_default)
    parser.add_argument('--model_filename',
                        type=str,
                        help='model filename',
                        default=model_filename_default)
    parser.add_argument('--log_dir',
                        type=str,
                        help='log directory',
                        default=log_directory_default)

    argv = parser.parse_args()

    # Post-process argparser
    learning_rate = argv.lr
    learning_rate_decay = argv.lr_decay
    dropout_rate = argv.dropout
    mini_batch_size = argv.batch_size
    epochs = argv.epochs
    optimizer = argv.optimizer
    random_seed = argv.seed
    model_directory = argv.model_dir
    model_filename = argv.model_filename
    log_directory = argv.log_dir

    if argv.train:
        print('Training CNN on CIFAR10 dataset...')
        train(learning_rate=learning_rate,
              learning_rate_decay=learning_rate_decay,
              dropout_rate=dropout_rate,
              mini_batch_size=mini_batch_size,
              epochs=epochs,
              optimizer=optimizer,
              random_seed=random_seed,
              model_directory=model_directory,
              model_filename=model_filename,
              log_directory=log_directory)

    if argv.test:
        print('Testing CNN on CIFAR10 dataset...')
        test(model_file=os.path.join(model_directory_default,
                                     model_filename_default))


if __name__ == '__main__':

    main()
