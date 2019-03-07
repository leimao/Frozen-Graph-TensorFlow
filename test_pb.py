
import tensorflow as tf
import numpy as np
import argparse

from cifar import CIFAR10
from utils import model_accuracy
from tensorflow.python.framework import tensor_util

# If load from pb, you may have to use get_tensor_by_name heavily.

class CNN(object):

    def __init__(self, model_filepath):

        # The file path of model
        self.model_filepath = model_filepath
        # Initialize the model
        self.load_graph(model_filepath = self.model_filepath)

    def load_graph(self, model_filepath):
        '''
        Lode trained model.
        '''
        print('Loading model...')
        self.graph = tf.Graph()
        self.sess = tf.InteractiveSession(graph = self.graph)

        with tf.gfile.GFile(model_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        print('Check out the input placeholders:')
        nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
        for node in nodes:
            print(node)

        # Define input tensor
        self.input = tf.placeholder(np.float32, shape = [None, 32, 32, 3], name='input')
        self.dropout_rate = tf.placeholder(tf.float32, shape = [], name = 'dropout_rate')

        tf.import_graph_def(graph_def, {'input': self.input, 'dropout_rate': self.dropout_rate})

        print('Model loading complete!')

        '''
        # Get layer names
        layers = [op.name for op in self.graph.get_operations()]
        for layer in layers:
            print(layer)
        '''

        '''
        # Check out the weights of the nodes
        weight_nodes = [n for n in graph_def.node if n.op == 'Const']
        for n in weight_nodes:
            print("Name of the node - %s" % n.name)
            print("Value - " )
            print(tensor_util.MakeNdarray(n.attr['value'].tensor))
        '''

    def test(self, data):

        # Know your output node name
        output_tensor = self.graph.get_tensor_by_name("import/cnn/output:0")
        output = self.sess.run(output_tensor, feed_dict = {self.input: data, self.dropout_rate: 0})

        return output


def test_from_frozen_graph(model_filepath):

    tf.reset_default_graph()

    # Load CIFAR10 dataset
    cifar10 = CIFAR10()
    x_test = cifar10.x_test
    y_test = cifar10.y_test
    y_test_onehot = cifar10.y_test_onehot
    num_classes = cifar10.num_classes
    input_size = cifar10.input_size

    # Test 500 samples
    x_test = x_test[0:500]
    y_test = y_test[0:500]

    model = CNN(model_filepath = model_filepath)

    test_prediction_onehot = model.test(data = x_test)
    test_prediction = np.argmax(test_prediction_onehot, axis = 1).reshape((-1,1))
    test_accuracy = model_accuracy(label = y_test, prediction = test_prediction)

    print('Test Accuracy: %f' % test_accuracy)

def main():

    model_pb_filepath_default = './model/cifar10_cnn.pb'

    # Argparser
    parser = argparse.ArgumentParser(description = 'Load and test model from frozen graph pb file.')

    parser.add_argument('--model_pb_filepath', type = str, help = 'model pb-format frozen graph file filepath', default = model_pb_filepath_default)

    argv = parser.parse_args()

    model_pb_filepath = argv.model_pb_filepath

    test_from_frozen_graph(model_filepath=model_pb_filepath)


if __name__ == '__main__':

    main()