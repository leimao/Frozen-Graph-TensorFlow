# Frozen Graph TensorFlow 2.x

Lei Mao

## Introduction

TensorFlow 1.x provided interface to freeze models via `tf.Session`. However, since TensorFlow 2.x removed `tf.Session`, freezing models in TensorFlow 2.x had been a problem to most of the users.

In this repository, several simple concrete examples have been implemented to demonstrate how to freeze models and run inference using frozen models in TensorFlow 2.x. The frozen models are also fully compatible with inference using TensorFlow 1.x, TensorFlow 2.x, ONNX Runtime, and TensorRT. 

## Usages

### Docker Container

We use TensorFlow 2.2 Docker container from DockerHub. To download the Docker image, please run the following command in the terminal.

```bash
$ docker pull tensorflow/tensorflow:2.2.0-gpu
```

To start the Docker container, please run the following command in the terminal.

```bash
$ docker run --gpus all -it --rm -v $(pwd):/mnt tensorflow/tensorflow:2.2.0-gpu
```

### Examples

#### Example 1

We would train a simple fully connected neural network to classify the Fashion MNIST data. The model would be saved as `SavedModel` in the `models/simple_model` directory for completeness. In addition, the model would also be frozen and saved as `simple_frozen_graph.pb` in the `frozen_models` directory.

To train, save, export, and run inference for the model, please run the following command in the terminal.

```bash
$ python example_1.py
```

#### Example 2

We would train a simple recurrent neural network that has multiple inputs and outputs using random data. The model would be saved as `SavedModel` in the `models/complex_model` directory for completeness. In addition, the model would also be frozen and saved as `complex_frozen_graph.pb` in the `frozen_models` directory.

To train, save, export, and run inference for the model, please run the following command in the terminal.

```bash
$ python example_2.py
```

### Convert Frozen Graph to ONNX

If TensorFlow 1.x and `tf2onnx` have been installed, the frozen graph could be converted to ONNX model using the following command.

```bash
$ python -m tf2onnx.convert --input ./frozen_models/frozen_graph.pb --output model.onnx --outputs Identity:0 --inputs x:0
```

### Convert Frozen Graph to UFF

The frozen graph could also be converted to UFF model for TensorRT using the following command. 

```bash
$ convert-to-uff frozen_graph.pb -t -O Identity -o frozen_graph.uff
```

TensorRT 6.0 Docker image could be pulled from [NVIDIA NGC](https://ngc.nvidia.com/).

```bash
$ docker pull nvcr.io/nvidia/tensorrt:19.12-py3
```

## References

* [Migrate from TensorFlow 1.x to 2.x](https://www.tensorflow.org/guide/migrate)
