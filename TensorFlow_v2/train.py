import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np

from utils import get_fashion_mnist_data


def main():

    tf.random.set_seed(seed=0)

    # Get data
    (train_images, train_labels), (test_images,
                                   test_labels) = get_fashion_mnist_data()

    # Create Keras model
    model = keras.Sequential(layers=[
        keras.layers.InputLayer(input_shape=(28, 28), name="input"),
        keras.layers.Flatten(input_shape=(28, 28), name="flatten"),
        keras.layers.Dense(128, activation="relu", name="dense"),
        keras.layers.Dense(10, activation="softmax", name="output")
    ],
                             name="FCN")

    # Print model architecture
    model.summary()

    # Compile model with optimizer
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # Train model
    model.fit(x={"input": train_images}, y={"output": train_labels}, epochs=1)

    # Test model
    test_loss, test_acc = model.evaluate(x={"input": test_images},
                                         y={"output": test_labels},
                                         verbose=2)
    print("-" * 50)
    print("Test accuracy: ")
    print(test_acc)

    # Get predictions for test images
    predictions = model.predict(test_images)
    # Print the prediction for the first image
    print("-" * 50)
    print("Example prediction reference:")
    print(predictions[0])

    # Save model to SavedModel format
    tf.saved_model.save(model, "./models")

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
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
                      name="frozen_graph.pb",
                      as_text=False)


if __name__ == "__main__":

    main()
