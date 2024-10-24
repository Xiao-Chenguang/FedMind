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

There are examples in the [examples](./examples) directory.


Make a copy of both the [config.yaml](./config.yaml) and [fedavg_demo.py](./examples/fedavg_demo.py) to your own directory.
 You can run them with the following command:
```bash
python fedavg_demo.py
```

Here we recommend you to use the [UV](https://docs.astral.sh/uv/) as a python environment manager to create a clean environment for the experiments.

After install `uv`, you can create a new environment and run a `FedMind` example with the following command:
```bash
uv init FL-demo
cd FL-demo

source .uv/bin/activate
uv add fedmind torchvision

wget https://raw.githubusercontent.com/Xiao-Chenguang/FedMind/refs/heads/main/examples/fedavg_demo.py
wget https://raw.githubusercontent.com/Xiao-Chenguang/FedMind/refs/heads/main/config.yaml

uv run python fedavg_demo.py
```


## Features
This FL framework provides two client simulation modes depending on your resources:
- Parallel training speed up for powerful resources.
- Serialization for limited resources.

This is controlled by the parameter `NUM_PROCESS` which can be set in the [config.yaml](./config.yaml).
Setting `NUM_PROCESS` to 0 will use the serialization mode where each client trains sequentially in same global round.
Setting `NUM_PROCESS > 0` will use the parallel mode where `NUM_PROCESS` workers consume the clients tasks in parallel.
The recommended value for `NUM_PROCESS` is the number of **CPU cores** available.
