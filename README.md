# FedMind
A simple and easy Federated Learning framework fit researchers' mind based on [PyTorch](https://pytorch.org/).

Unlike other popular FL frameworks focusing on production, `FedMind` is designed for researchers to easily implement their own FL algorithms and experiments. It provides a simple and flexible interface to implement FL algorithms and experiments.

## Installation
The package is published on PyPI under the name `fedmind`. You can install it with pip:
```bash
pip install fedmind
```

## Usage

A configuration file in `yaml` is required to run the experiments.
You can refer to the [config.yaml](./config.yaml) as an example.

There are examples in the [examples](./exmaples) directory.


Make a copy of both the [config.yaml](./config.yaml) and [fedavg_demo.py](./example/fedavg_demo.py) to your own directory.
 You can run them with the following command:
```bash
python fedavg_demo.py
```

## Features
- Parallel training speed up.
- Serialization for limited resources.