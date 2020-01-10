# Frozen Graph TensorFlow 2.x

Lei Mao

## Introduction

TensorFlow 1.x provided interface to freeze models via `tf.Session`. However, since TensorFlow 2.x removed `tf.Session`, freezing models in TensorFlow 2.x had been a problem to most of the users.

In this repository, a simple concrete example has been implemented to demonstrate how to freeze models in TensorFlow 2.x. The frozen models are also fully compatible with inference using TensorFlow 1.x, TensorFlow 2.x, ONNX Runtime, and TensorRT. 

## Usages

### Docker Container

We use TensorFlow 2.1 Docker container from DockerHub. To download the Docker image, please run the following command in the terminal.

```bash
$ docker pull tensorflow/tensorflow:latest-gpu-py3
```

Currently Google forgot to upload a TensorFlow 2.1 Docker image for Python 3.6 and GPU explicitly. I will change the Docker image to explicit tag instead of `latest` once Google fixed this.


To start the Docker container, please run the following command in the terminal.

```bash
$ docker run --gpus all -it --rm -v $(pwd):/mnt tensorflow/tensorflow:latest-gpu-py3
```

### Train and Export Model

We would train a simple fully connected neural network to classify the Fashion MNIST data. The model would be saved as `SavedModel` in the `models` directory for completeness. In addition, the model would also be frozen and saved as `frozen_graph.pb` in the `frozen_models` directory.

To train and export the model, please run the following command in the terminal.

```
$ python train.py
```

We would also have a reference value for the sample inference from TensorFlow 2.x using the conventional inference protocol in the printouts.

```
Example prediction reference:
[3.9113933e-05 1.1972898e-07 5.2244545e-06 5.4371812e-06 6.1125693e-06
 1.1335548e-01 3.0090479e-05 2.8483599e-01 9.5160649e-04 6.0077089e-01]
```

### Run Inference Using Frozen Graph

To run inference using the frozen graph in TensorFlow 2.x, please run the following command in the terminal.

```bash
$ python test.py
```

We also got the value for the sample inference using frozen graph. It is (almost) exactly the same to the reference value we got using the conventional inference protocol. 

```
Example prediction reference:
[3.9113860e-05 1.1972921e-07 5.2244545e-06 5.4371812e-06 6.1125752e-06
 1.1335552e-01 3.0090479e-05 2.8483596e-01 9.5160597e-04 6.0077089e-01]
```

### Convert Frozen Graph to ONNX

If TensorFlow 1.x and `tf2onnx` have been installed, the frozen graph could be converted to ONNX model using the following command.

```bash
python -m tf2onnx.convert --input ./frozen_models/frozen_graph.pb --output model.onnx --outputs Identity:0 --inputs x:0
```

### Convert Frozen Graph to UFF

The frozen graph could also be converted to UFF model for TensorRT using the following command. 

```python
convert-to-uff frozen_graph.pb -t -O Identity -o frozen_graph.uff
```

TensorRT 6.0 Docker image could be pulled from [NVIDIA NGC](https://ngc.nvidia.com/)

```
$ docker pull nvcr.io/nvidia/tensorrt:19.12-py3
```

## References

* [Migrate from TensorFlow 1.x to 2.x](https://www.tensorflow.org/guide/migrate)
