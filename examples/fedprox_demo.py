from functools import reduce

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from fedmind.algs.fedprox import FedProx
from fedmind.config import get_config
from fedmind.data import ClientDataset


def test_fedprox():
    # 0. Prepare necessary arguments
    config = get_config("config.yaml")
    config.PROX_MU = 0.1
    if config.SEED >= 0:
        torch.manual_seed(config.SEED)

    # 1. Prepare Federated Learning DataSets
    org_ds = MNIST("dataset", train=True, download=True, transform=ToTensor())
    test_ds = MNIST("dataset", train=False, download=True, transform=ToTensor())

    effective_size = len(org_ds) - len(org_ds) % config.NUM_CLIENT
    idx_groups = torch.randperm(effective_size).reshape(config.NUM_CLIENT, -1)
    fed_dss = [ClientDataset(org_ds, idx) for idx in idx_groups.tolist()]

    genetors = [
        torch.Generator().manual_seed(config.SEED + i) if config.SEED >= 0 else None
        for i in range(config.NUM_CLIENT)
    ]
    fed_loader = [
        DataLoader(ds, config.BATCH_SIZE, shuffle=True, generator=gtr)
        for ds, gtr in zip(fed_dss, genetors)
    ]
    test_loader = DataLoader(test_ds, config.BATCH_SIZE * 4)

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
    FedProx(
        model=model,
        fed_loader=fed_loader,
        test_loader=test_loader,
        criterion=criterion,
        config=config,
    ).fit(config.NUM_CLIENT, config.ACTIVE_CLIENT, config.SERVER_EPOCHS)


if __name__ == "__main__":
    test_fedprox()
