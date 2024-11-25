# FedMind

[![image](https://img.shields.io/pypi/v/fedmind.svg)](https://pypi.python.org/pypi/fedmind)
[![image](https://img.shields.io/pypi/l/fedmind.svg)](https://github.com/Xiao-Chenguang/FedMind//blob/main/LICENSE)
[![image](https://img.shields.io/pypi/pyversions/fedmind.svg)](https://pypi.python.org/pypi/fedmind)
[![PyPI Downloads](https://static.pepy.tech/badge/fedmind)](https://pepy.tech/projects/fedmind)
[![Actions status](https://github.com/Xiao-Chenguang/FedMind/workflows/Ruff/badge.svg)](https://github.com/Xiao-Chenguang/FedMind/actions)

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
- **Simple**: Easy to implement your own FL algorithms and experiments.
- **PyTorch**: Based on [PyTorch](https://pytorch.org/), a popular deep learning framework.
- **Multi-Platform**: Support both *Linux*, *macOS* and *Windows*.
- **CPU/GPU**: Support both **CPU** and **GPU** training.
- **Serial/Parallel**: Support both serial and parallel training modes.
- **Model Operation**: Support model level operations like `+`, `-`, `*`, `/`.
- **Reproducible**: Reproduce your experiments with the `configuration file` and `seed`.


### Serial/Parallel Training
This FL framework provides two client simulation modes depending on your resources:
- Parallel training speed up for powerful resources.
- Serialization for limited resources.

This is controlled by the parameter `NUM_PROCESS` which can be set in the [config.yaml](./config.yaml).
Setting `NUM_PROCESS` to 0 will use the serialization mode where each client trains sequentially in same global round.
Setting `NUM_PROCESS > 0` will use the parallel mode where `NUM_PROCESS` workers consume the clients tasks in parallel.
The recommended value for `NUM_PROCESS` is the number of **CPU cores** available.

#### Notice on Windows
Multiprocessing with CUDA is ***not supported on Windows***. If you are using Windows, you should set `NUM_PROCESS` to 0 to use the serialization mode with CUDA. Or you are free to use the parallel mode with CPU only.

## Acknowledgement
- This project includes code from [easydict](https://github.com/makinacorpus/easydict), licensed under the GNU Lesser General Public License (LGPL) version 3.0.
See the [easydict's LICENSE](https://github.com/makinacorpus/easydict?tab=LGPL-3.0-1-ov-file) for details.