# Save, Load and Inference From Frozen Graph in TensorFlow 

Lei Mao

## Introduction

This repository was modified from my previous [simple CNN model](https://github.com/leimao/Convolutional_Neural_Network_CIFAR10) to classify CIFAR10 dataset. It consist training, saving model to frozen graph ``pb`` file, load ``pb`` file and do inference in TensorFlow. The tutorial with detailed description is available on my [blog](https://leimao.github.io/blog/Save-Load-Inference-From-TF-Frozen-Graph/). To the best of my knowledge, there is few similar tutorials on the internet. I wish this sample code could help you to prepare your own ``pb`` file for deployment.


## Dependencies

* Python 3.6
* Numpy 1.14
* TensorFlow 1.12
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

### Train and Test Model in TensorFlow

```bash
$ python main.py --help
usage: main.py [-h] [-train] [-test] [--lr LR] [--lr_decay LR_DECAY]
               [--dropout DROPOUT] [--batch_size BATCH_SIZE] [--epochs EPOCHS]
               [--optimizer OPTIMIZER] [--seed SEED] [--model_dir MODEL_DIR]
               [--model_filename MODEL_FILENAME] [--log_dir LOG_DIR]

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

```bash
$ python main.py --train --test --epoch 30 --lr_decay 0.9 --dropout 0.5
```

### Test Model from PB File

```bash
$ python test_pb.py --help
usage: test_pb.py [-h] [--model_pb_filepath MODEL_PB_FILEPATH]

Load and test model from frozen graph pb file.

optional arguments:
  -h, --help            show this help message and exit
  --model_pb_filepath MODEL_PB_FILEPATH
                        model pb-format frozen graph file filepath

```


```bash
$ python test_pb.py
```



## Reference

* [Save, Load and Inference From TensorFlow Frozen Graph](https://leimao.github.io/blog/Save-Load-Inference-From-TF-Frozen-Graph/)
