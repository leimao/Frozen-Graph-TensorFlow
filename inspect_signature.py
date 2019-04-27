import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

def retrieve_model_data_info(saved_model_path):
   with tf.Session() as sess:
       graph = tf.Graph()
       with graph.as_default():
           metagraph = tf.saved_model.loader.load(sess, [tag_constants.SERVING], saved_model_path)
       inputs_mapping = dict(metagraph.signature_def['serving_default'].inputs)
       outputs_mapping = dict(metagraph.signature_def['serving_default'].outputs)
       print("Print output mapping: ", outputs_mapping)
       print("Print input mapping: ", inputs_mapping)

retrieve_model_data_info('./model/signature')