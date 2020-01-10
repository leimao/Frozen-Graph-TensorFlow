import tensorflow as tf
import os

from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils as saved_model_utils

class CNN(object):

    def __init__(self, input_size, num_classes, optimizer):

        self.num_classes = num_classes
        self.input_size = input_size
        self.optimizer = optimizer

        self.learning_rate = tf.placeholder(tf.float32, shape = [], name = 'learning_rate')
        self.dropout_rate = tf.placeholder(tf.float32, shape = [], name = 'dropout_rate')
        self.input = tf.placeholder(tf.float32, [None] + self.input_size, name = 'input')
        self.label = tf.placeholder(tf.float32, [None, self.num_classes], name = 'label')
        self.output = self.network_initializer()
        self.loss = self.loss_initializer()
        self.optimization = self.optimizer_initializer()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def network(self, input, dropout_rate):

        conv1 = tf.layers.conv2d(
            inputs = input,
            filters = 64,
            kernel_size = [3, 3],
            padding = 'same',
            activation = tf.nn.relu,
            name = 'conv1')

        conv2 = tf.layers.conv2d(
            inputs = conv1,
            filters = 64,
            kernel_size = [3, 3],
            padding = 'same',
            activation = tf.nn.relu,
            name = 'conv2')

        pool1 = tf.layers.max_pooling2d(
            inputs = conv2,
            pool_size = [2, 2],
            strides = [2, 2],
            name = 'pool1')

        pool1_dropout = tf.layers.dropout(
            inputs = pool1,
            rate = dropout_rate,
            training=True,
            name = 'pool1_dropout')

        conv3 = tf.layers.conv2d(
            inputs = pool1_dropout,
            filters = 128,
            kernel_size = [3, 3],
            padding = 'same',
            activation = tf.nn.relu,
            name = 'conv3')

        conv4 = tf.layers.conv2d(
            inputs = conv3,
            filters = 128,
            kernel_size = [3, 3],
            padding = 'same',
            activation = tf.nn.relu,
            name = 'conv4')

        pool2 = tf.layers.max_pooling2d(
            inputs = conv4,
            pool_size = [2, 2],
            strides = [2, 2],
            name = 'pool2')

        pool2_dropout = tf.layers.dropout(
            inputs = pool2,
            rate = dropout_rate,
            training=True,
            name = 'pool2_dropout')

        conv5 = tf.layers.conv2d(
            inputs = pool2_dropout,
            filters = 256,
            kernel_size = [3, 3],
            padding = 'same',
            activation = tf.nn.relu,
            name = 'conv5')

        pool3 = tf.layers.max_pooling2d(
            inputs = conv5,
            pool_size = [2, 2],
            strides = [2, 2],
            name = 'pool3')

        pool3_dropout = tf.layers.dropout(
            inputs = pool3,
            rate = dropout_rate,
            training=True,
            name = 'pool3_dropout')

        flat = tf.layers.flatten(
            inputs = pool3_dropout, 
            name = 'flat')

        fc1 = tf.layers.dense(
            inputs = flat,
            units = 256,
            activation = tf.nn.relu,
            name = 'fc1')

        fc1_dropout = tf.layers.dropout(
            inputs = fc1,
            rate = dropout_rate,
            training=True,
            name = 'fc1_dropout')

        fc2 = tf.layers.dense(
            inputs = fc1_dropout,
            units = self.num_classes,
            activation = None,
            name = 'fc2')

        # Give output node a 
        output = tf.identity(fc2, name='output')

        return output

    def network_initializer(self):

        with tf.variable_scope('cnn') as scope:
            ouput = self.network(input = self.input, dropout_rate = self.dropout_rate)

        return ouput

    def loss_initializer(self):

        with tf.variable_scope('loss') as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels = self.label, logits = self.output, name = 'cross_entropy')
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name = 'cross_entropy_mean')
        return cross_entropy_mean

    def optimizer_initializer(self):

        if self.optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(self.loss)

        return optimizer

    def train(self, data, label, learning_rate, dropout_rate):

        _, train_loss = self.sess.run([self.optimization, self.loss], 
            feed_dict = {self.input: data, self.label: label, self.learning_rate: learning_rate, self.dropout_rate: dropout_rate})
        return train_loss

    def validate(self, data, label):

        output, validate_loss = self.sess.run([self.output, self.loss], 
            feed_dict = {self.input: data, self.label: label, self.dropout_rate: 0.0})
        return output, validate_loss

    def test(self, data):

        output = self.sess.run(self.output, feed_dict = {self.input: data, self.dropout_rate: 0.0})

        return output

    def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, filename + '.ckpt')
        self.saver.save(self.sess, filepath)
        return filepath
    
    def save_signature(self, directory):

        signature = signature_def_utils.build_signature_def(
            inputs={'input': saved_model_utils.build_tensor_info(self.input), 'dropout_rate': saved_model_utils.build_tensor_info(self.dropout_rate)},
            outputs={'output': saved_model_utils.build_tensor_info(self.output)},
            method_name=signature_constants.PREDICT_METHOD_NAME)
        signature_map = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                         signature}
        model_builder = saved_model_builder.SavedModelBuilder(directory)
        model_builder.add_meta_graph_and_variables(self.sess,
            tags=[tag_constants.SERVING],
            signature_def_map=signature_map,
            clear_devices=True)
        model_builder.save(as_text=False) 

    def save_as_pb(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save check point for graph frozen later
        ckpt_filepath = self.save(directory=directory, filename=filename)
        pbtxt_filename = filename + '.pbtxt'
        pbtxt_filepath = os.path.join(directory, pbtxt_filename)
        pb_filepath = os.path.join(directory, filename + '.pb')
        # This will only save the graph but the variables will not be saved.
        # You have to freeze your model first.
        tf.train.write_graph(graph_or_graph_def=self.sess.graph_def, logdir=directory, name=pbtxt_filename, as_text=True)

        # Freeze graph
        # Method 1
        freeze_graph.freeze_graph(input_graph=pbtxt_filepath, input_saver='', input_binary=False, input_checkpoint=ckpt_filepath, output_node_names='cnn/output', restore_op_name='save/restore_all', filename_tensor_name='save/Const:0', output_graph=pb_filepath, clear_devices=True, initializer_nodes='')
        
        # Method 2
        '''
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()
        output_node_names = ['cnn/output']

        output_graph_def = graph_util.convert_variables_to_constants(self.sess, input_graph_def, output_node_names)

        with tf.gfile.GFile(pb_filepath, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        '''
        
        return pb_filepath

    def load(self, filepath):

        if os.path.splitext(filepath)[1] != '.ckpt':
            filepath += '.ckpt'

        self.saver.restore(self.sess, filepath)




