import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np

from utils import wrap_frozen_graph

def main():

    # Mysterious code
    # https://leimao.github.io/blog/TensorFlow-cuDNN-Failure/
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # Dummy example copied from TensorFlow
    # https://www.tensorflow.org/guide/keras/functional#models_with_multiple_inputs_and_outputs

    num_tags = 12  # Number of unique issue tags
    num_words = 10000  # Size of vocabulary obtained when preprocessing text data
    num_departments = 4  # Number of departments for predictions

    title_input = keras.Input(
        shape=(None,), name="title"
    )  # Variable-length sequence of ints
    body_input = keras.Input(shape=(None,), name="body")  # Variable-length sequence of ints
    tags_input = keras.Input(
        shape=(num_tags,), name="tags"
    )  # Binary vectors of size `num_tags`

    # Embed each word in the title into a 64-dimensional vector
    title_features = keras.layers.Embedding(num_words, 64)(title_input)
    # Embed each word in the text into a 64-dimensional vector
    body_features = keras.layers.Embedding(num_words, 64)(body_input)

    # Reduce sequence of embedded words in the title into a single 128-dimensional vector
    title_features = keras.layers.LSTM(128)(title_features)
    # Reduce sequence of embedded words in the body into a single 32-dimensional vector
    body_features = keras.layers.LSTM(32)(body_features)

    # Merge all available features into a single large vector via concatenation
    x = keras.layers.concatenate([title_features, body_features, tags_input])

    # Stick a logistic regression for priority prediction on top of the features
    priority_pred = keras.layers.Dense(1, name="priority")(x)
    # Stick a department classifier on top of the features
    department_pred = keras.layers.Dense(num_departments, name="department")(x)

    # Instantiate an end-to-end model predicting both priority and department
    model = keras.Model(
        inputs=[title_input, body_input, tags_input],
        outputs=[priority_pred, department_pred],
    )

    model.compile(
        optimizer=keras.optimizers.RMSprop(1e-3),
        loss=[
            keras.losses.BinaryCrossentropy(from_logits=True),
            keras.losses.CategoricalCrossentropy(from_logits=True),
        ],
        loss_weights=[1.0, 0.2],
    )

    # Dummy input data
    title_data = np.random.randint(num_words, size=(1280, 10)).astype("float32")
    body_data = np.random.randint(num_words, size=(1280, 100)).astype("float32")
    tags_data = np.random.randint(2, size=(1280, num_tags)).astype("float32")

    # Dummy target data
    priority_targets = np.random.random(size=(1280, 1))
    dept_targets = np.random.randint(2, size=(1280, num_departments))

    model.fit(
        {"title": title_data, "body": body_data, "tags": tags_data},
        {"priority": priority_targets, "department": dept_targets},
        epochs=2,
        batch_size=32,
    )

    predictions = model.predict({"title": title_data[0:1], "body": body_data[0:1], "tags": tags_data[0:1]})
    predictions_priority = predictions[0]
    predictions_department = predictions[1]

    print("-" * 50)
    print("Example TensorFlow prediction reference:")
    print(predictions_priority)
    print(predictions_department)

    # Save model to SavedModel format
    tf.saved_model.save(model, "./models/complex_model")

    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(x=(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype), tf.TensorSpec(model.inputs[1].shape, model.inputs[1].dtype), tf.TensorSpec(model.inputs[2].shape, model.inputs[2].dtype)))

    # Get frozen ConcreteFunction
    # https://github.com/tensorflow/tensorflow/issues/36391#issuecomment-596055100
    frozen_func = convert_variables_to_constants_v2(full_model, lower_control_flow=False)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                        logdir="./frozen_models",
                        name="complex_frozen_graph.pb",
                        as_text=False)

    # Load frozen graph using TensorFlow 1.x functions
    with tf.io.gfile.GFile("./frozen_models/complex_frozen_graph.pb", "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    # Wrap frozen graph to ConcreteFunctions
    frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                    inputs=["x:0", "x_1:0", "x_2:0"],
                                    outputs=["Identity:0", "Identity_1:0"],
                                    print_graph=True)

    # Note that we only have "one" input and "output" for the loaded frozen function
    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Get predictions
    frozen_graph_predictions = frozen_func(x=tf.constant(title_data[0:1]), x_1=tf.constant(body_data[0:1]), x_2=tf.constant(tags_data[0:1]))
    frozen_graph_predictions_priority = frozen_graph_predictions[0]
    frozen_graph_predictions_department = frozen_graph_predictions[1]

    print("-" * 50)
    print("Example TensorFlow frozen graph prediction reference:")
    print(frozen_graph_predictions_priority.numpy())
    print(frozen_graph_predictions_department.numpy())

    # The two predictions should be almost the same.
    assert np.allclose(a=frozen_graph_predictions_priority.numpy(), b=predictions_priority, rtol=1e-05, atol=1e-08, equal_nan=False)
    assert np.allclose(a=frozen_graph_predictions_department.numpy(), b=predictions_department, rtol=1e-05, atol=1e-08, equal_nan=False)


if __name__ == "__main__":

    main()
