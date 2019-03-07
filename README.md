# Convolutional Neural Network CIFAR10

Lei Mao

University of Chicago

## Introduction

This is an object-oriented implementation of convolutional neural network (CNN) using TensorFlow on CIFAR10 dataset. The CNN class could be initialized, trained, tested, saved, and loaded in a manner similar to Keras, which is highly human-readable, portable, and scalable. The classification accuracy on CIFAR10 dataset is 80%.

## Dependencies

* Python 3.5
* Numpy 1.14
* TensorFlow 1.8
* Matplotlib 2.1.1 (for demo purpose)


## Files
```bash
.
├── cifar.py
├── cnn.py
├── main.py
├── README.md
└── utils.py
```
## Features

* User-friendly CNN API wrapped
* Allows changing learning rate and dropout rate in real time
* No need for significant changes to codes in order to work for other tasks

## Usage

In Python, to build CNN classifier using the package in a Keras style:

```python
from cnn import CNN
model = CNN(input_size = input_size, num_classes = num_classes, optimizer = optimizer)
...
# Train on train set
for i, idx in enumerate(mini_batch_idx):
    train_loss = model.train(data = x_train[idx], label = y_train_onehot[idx], learning_rate = learning_rate, dropout_rate = dropout_rate)
    if i % 200 == 0:
        train_prediction_onehot = model.test(data = x_train[idx])
        train_prediction = np.argmax(train_prediction_onehot, axis = 1).reshape((-1,1))
        train_accuracy = model_accuracy(label = y_train[idx], prediction = train_prediction)
        print('Training Loss: %f, Training Accuracy: %f' % (train_loss, train_accuracy))
```

## Demo

To test the CNN classifier on CIFAR10, simply run the following command in the shell:

```bash
$ python main.py --train --test --epoch 30 --lr_decay 0.9 --dropout 0.5
```

Addition arguments:
```
Train CNN on CIFAR10 dataset.

optional arguments:
  -h, --help            show this help message and exit
  -train, --train       train model
  -test, --test         test model
  --lr LR               initial learning rate
  --lr_decay LR_DECAY   learning rate decay
  --dropout DROPOUT     dropout rate
  --batch_size BATCH_SIZE
                        mini batch size
  --epochs EPOCHS       number of epochs
  --optimizer OPTIMIZER
                        optimizer
  --seed SEED           random seed
  --model_dir MODEL_DIR
                        model directory
  --model_filename MODEL_FILENAME
                        model filename
  --log_dir LOG_DIR     log directory
```

<p align="center">
    <img src = "./training_curve.png" width="80%">
</p>

```bash
Training CNN on CIFAR10 dataset...
2018-05-10 05:22:57.239404: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2018-05-10 05:22:57.351311: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:898] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-05-10 05:22:57.351751: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1356] Found device 0 with properties: 
name: GeForce GTX TITAN X major: 5 minor: 2 memoryClockRate(GHz): 1.076
pciBusID: 0000:01:00.0
totalMemory: 11.93GiB freeMemory: 11.63GiB
2018-05-10 05:22:57.351784: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0
2018-05-10 05:22:57.537926: I tensorflow/core/common_runtime/gpu/gpu_device.cc:923] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-05-10 05:22:57.537967: I tensorflow/core/common_runtime/gpu/gpu_device.cc:929]      0 
2018-05-10 05:22:57.537978: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 0:   N 
2018-05-10 05:22:57.538215: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 11265 MB memory) -> physical GPU (device: 0, name: GeForce GTX TITAN X, pci bus id: 0000:01:00.0, compute capability: 5.2)
Epoch: 0
Validation Accuracy: 0.109800
Training Loss: 2.303014, Training Accuracy: 0.156250
Training Loss: 1.508689, Training Accuracy: 0.468750
Training Loss: 1.441969, Training Accuracy: 0.562500
Training Loss: 1.061783, Training Accuracy: 0.625000
Epoch: 1
Validation Accuracy: 0.629600
Training Loss: 1.255457, Training Accuracy: 0.546875
Training Loss: 1.148911, Training Accuracy: 0.546875
Training Loss: 0.892210, Training Accuracy: 0.765625
Training Loss: 0.906945, Training Accuracy: 0.671875
Epoch: 2
Validation Accuracy: 0.692600
Training Loss: 0.717275, Training Accuracy: 0.796875
Training Loss: 0.688842, Training Accuracy: 0.828125
Training Loss: 0.506169, Training Accuracy: 0.875000
Training Loss: 0.533918, Training Accuracy: 0.859375
Epoch: 3
Validation Accuracy: 0.758600
Training Loss: 0.548112, Training Accuracy: 0.796875
Training Loss: 0.373025, Training Accuracy: 0.875000
Training Loss: 0.625190, Training Accuracy: 0.828125
Training Loss: 0.479000, Training Accuracy: 0.906250
Epoch: 4
Validation Accuracy: 0.766000
Training Loss: 0.386557, Training Accuracy: 0.890625
Training Loss: 0.346336, Training Accuracy: 0.906250
Training Loss: 0.524307, Training Accuracy: 0.812500
Training Loss: 0.390890, Training Accuracy: 0.843750
Epoch: 5
Validation Accuracy: 0.781600
Training Loss: 0.344274, Training Accuracy: 0.906250
Training Loss: 0.261086, Training Accuracy: 0.906250
Training Loss: 0.335110, Training Accuracy: 0.906250
Training Loss: 0.303369, Training Accuracy: 0.937500
Epoch: 6
Validation Accuracy: 0.783600
Training Loss: 0.230124, Training Accuracy: 0.937500
Training Loss: 0.111131, Training Accuracy: 0.984375
Training Loss: 0.334720, Training Accuracy: 0.937500
Training Loss: 0.242996, Training Accuracy: 0.953125
Epoch: 7
Validation Accuracy: 0.787600
Training Loss: 0.181984, Training Accuracy: 0.953125
Training Loss: 0.153440, Training Accuracy: 0.984375
Training Loss: 0.098123, Training Accuracy: 0.984375
Training Loss: 0.299774, Training Accuracy: 0.921875
Epoch: 8
Validation Accuracy: 0.782600
Training Loss: 0.176800, Training Accuracy: 0.984375
Training Loss: 0.030254, Training Accuracy: 1.000000
Training Loss: 0.038182, Training Accuracy: 1.000000
Training Loss: 0.070540, Training Accuracy: 1.000000
Epoch: 9
Validation Accuracy: 0.781200
Training Loss: 0.022498, Training Accuracy: 1.000000
Training Loss: 0.012810, Training Accuracy: 1.000000
Training Loss: 0.045954, Training Accuracy: 1.000000
Training Loss: 0.015940, Training Accuracy: 1.000000
Epoch: 10
Validation Accuracy: 0.791000
Training Loss: 0.037453, Training Accuracy: 1.000000
Training Loss: 0.012426, Training Accuracy: 1.000000
Training Loss: 0.039405, Training Accuracy: 1.000000
Training Loss: 0.020274, Training Accuracy: 1.000000
Epoch: 11
Validation Accuracy: 0.786800
Training Loss: 0.034441, Training Accuracy: 1.000000
Training Loss: 0.004813, Training Accuracy: 1.000000
Training Loss: 0.011596, Training Accuracy: 1.000000
Training Loss: 0.041501, Training Accuracy: 1.000000
Epoch: 12
Validation Accuracy: 0.790000
Training Loss: 0.004985, Training Accuracy: 1.000000
Training Loss: 0.015460, Training Accuracy: 1.000000
Training Loss: 0.004123, Training Accuracy: 1.000000
Training Loss: 0.002550, Training Accuracy: 1.000000
Epoch: 13
Validation Accuracy: 0.792400
Training Loss: 0.006348, Training Accuracy: 1.000000
Training Loss: 0.002836, Training Accuracy: 1.000000
Training Loss: 0.000638, Training Accuracy: 1.000000
Training Loss: 0.004171, Training Accuracy: 1.000000
Epoch: 14
Validation Accuracy: 0.793800
Training Loss: 0.002822, Training Accuracy: 1.000000
Training Loss: 0.005613, Training Accuracy: 1.000000
Training Loss: 0.000661, Training Accuracy: 1.000000
Training Loss: 0.003778, Training Accuracy: 1.000000
Epoch: 15
Validation Accuracy: 0.789800
Training Loss: 0.018694, Training Accuracy: 1.000000
Training Loss: 0.000567, Training Accuracy: 1.000000
Training Loss: 0.014707, Training Accuracy: 1.000000
Training Loss: 0.037662, Training Accuracy: 1.000000
Epoch: 16
Validation Accuracy: 0.790600
Training Loss: 0.004663, Training Accuracy: 1.000000
Training Loss: 0.007919, Training Accuracy: 1.000000
Training Loss: 0.000787, Training Accuracy: 1.000000
Training Loss: 0.001239, Training Accuracy: 1.000000
Epoch: 17
Validation Accuracy: 0.798600
Training Loss: 0.001123, Training Accuracy: 1.000000
Training Loss: 0.000295, Training Accuracy: 1.000000
Training Loss: 0.000120, Training Accuracy: 1.000000
Training Loss: 0.000520, Training Accuracy: 1.000000
Epoch: 18
Validation Accuracy: 0.802000
Training Loss: 0.000115, Training Accuracy: 1.000000
Training Loss: 0.000026, Training Accuracy: 1.000000
Training Loss: 0.000200, Training Accuracy: 1.000000
Training Loss: 0.000138, Training Accuracy: 1.000000
Epoch: 19
Validation Accuracy: 0.804400
Training Loss: 0.000320, Training Accuracy: 1.000000
Training Loss: 0.000239, Training Accuracy: 1.000000
Training Loss: 0.000064, Training Accuracy: 1.000000
Training Loss: 0.000103, Training Accuracy: 1.000000
Epoch: 20
Validation Accuracy: 0.803200
Training Loss: 0.000096, Training Accuracy: 1.000000
Training Loss: 0.000049, Training Accuracy: 1.000000
Training Loss: 0.000032, Training Accuracy: 1.000000
Training Loss: 0.000045, Training Accuracy: 1.000000
Epoch: 21
Validation Accuracy: 0.805400
Training Loss: 0.000096, Training Accuracy: 1.000000
Training Loss: 0.000205, Training Accuracy: 1.000000
Training Loss: 0.000012, Training Accuracy: 1.000000
Training Loss: 0.000067, Training Accuracy: 1.000000
Epoch: 22
Validation Accuracy: 0.806400
Training Loss: 0.000059, Training Accuracy: 1.000000
Training Loss: 0.000062, Training Accuracy: 1.000000
Training Loss: 0.000048, Training Accuracy: 1.000000
Training Loss: 0.000129, Training Accuracy: 1.000000
Epoch: 23
Validation Accuracy: 0.805400
Training Loss: 0.000038, Training Accuracy: 1.000000
Training Loss: 0.000026, Training Accuracy: 1.000000
Training Loss: 0.000038, Training Accuracy: 1.000000
Training Loss: 0.000027, Training Accuracy: 1.000000
Epoch: 24
Validation Accuracy: 0.806000
Training Loss: 0.000027, Training Accuracy: 1.000000
Training Loss: 0.000039, Training Accuracy: 1.000000
Training Loss: 0.000014, Training Accuracy: 1.000000
Training Loss: 0.000031, Training Accuracy: 1.000000
Epoch: 25
Validation Accuracy: 0.805200
Training Loss: 0.000010, Training Accuracy: 1.000000
Training Loss: 0.000006, Training Accuracy: 1.000000
Training Loss: 0.000014, Training Accuracy: 1.000000
Training Loss: 0.000014, Training Accuracy: 1.000000
Epoch: 26
Validation Accuracy: 0.805600
Training Loss: 0.000012, Training Accuracy: 1.000000
Training Loss: 0.000026, Training Accuracy: 1.000000
Training Loss: 0.000026, Training Accuracy: 1.000000
Training Loss: 0.000009, Training Accuracy: 1.000000
Epoch: 27
Validation Accuracy: 0.805600
Training Loss: 0.000027, Training Accuracy: 1.000000
Training Loss: 0.000038, Training Accuracy: 1.000000
Training Loss: 0.000006, Training Accuracy: 1.000000
Training Loss: 0.000015, Training Accuracy: 1.000000
Epoch: 28
Validation Accuracy: 0.807000
Training Loss: 0.000013, Training Accuracy: 1.000000
Training Loss: 0.000009, Training Accuracy: 1.000000
Training Loss: 0.000010, Training Accuracy: 1.000000
Training Loss: 0.000004, Training Accuracy: 1.000000
Epoch: 29
Validation Accuracy: 0.806400
Training Loss: 0.000015, Training Accuracy: 1.000000
Training Loss: 0.000006, Training Accuracy: 1.000000
Training Loss: 0.000003, Training Accuracy: 1.000000
Training Loss: 0.000009, Training Accuracy: 1.000000
Trained model saved successfully
Testing CNN on CIFAR10 dataset...
2018-05-10 05:27:22.236105: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0
2018-05-10 05:27:22.236173: I tensorflow/core/common_runtime/gpu/gpu_device.cc:923] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-05-10 05:27:22.236257: I tensorflow/core/common_runtime/gpu/gpu_device.cc:929]      0 
2018-05-10 05:27:22.236272: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 0:   N 
2018-05-10 05:27:22.236393: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 11265 MB memory) -> physical GPU (device: 0, name: GeForce GTX TITAN X, pci bus id: 0000:01:00.0, compute capability: 5.2)
Test Accuracy: 0.793700
```

The final test accuracy is 79.4%. The running time of training for 30 epochs is less than 5 minutes on a PC using a NVIDIA GeForce GTX TITAN X GPU.

## Future Work

Add TensorBoard functionalities to the CNN class.