from functools import reduce

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from fedmind.algs.fedavg import FedAvg
from fedmind.config import get_config
from fedmind.data import ClientDataset


def test_fedavg():
    # 0. Prepare necessary arguments
    config = get_config("config.yaml")

    # 1. Prepare Federated Learning DataSets
    org_ds = MNIST("dataset", train=True, download=True, transform=ToTensor())
    test_ds = MNIST("dataset", train=False, download=True, transform=ToTensor())

    effective_size = len(org_ds) - len(org_ds) % config.NUM_CLIENT
    idx_groups = torch.randperm(effective_size).reshape(config.NUM_CLIENT, -1)
    fed_dss = [ClientDataset(org_ds, idx) for idx in idx_groups.tolist()]

    fed_loader = [DataLoader(ds, batch_size=32, shuffle=True) for ds in fed_dss]
    test_loader = DataLoader(test_ds, batch_size=32)

    # 2. Prepare Model and Criterion
    classes = 10
    features = reduce(lambda x, y: x * y, org_ds[0][0].shape)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(features, 32),
        nn.ReLU(),
        nn.Linear(32, classes),
    )

    criterion = nn.CrossEntropyLoss()

    # 3. Run Federated Learning Simulation
    FedAvg(
        model=model,
        fed_loader=fed_loader,
        test_loader=test_loader,
        criterion=criterion,
        config=config,
    ).fit(config.NUM_CLIENT, config.ACTIVE_CLIENT, config.SERVER_EPOCHS)

    assert True


if __name__ == "__main__":
    test_fedavg()
