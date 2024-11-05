from functools import reduce

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from fedmind.algs.mfl import MFL
from fedmind.config import get_config
from fedmind.data import ClientDataset


def test_mfl():
    # 0. Prepare necessary arguments
    args = get_config("config.yaml")
    args.OPTIM.MOMENTUM = 0.9
    if args.SEED >= 0:
        torch.manual_seed(args.SEED)

    # 1. Prepare Federated Learning DataSets
    org_ds = MNIST("dataset", train=True, download=True, transform=ToTensor())
    test_ds = MNIST("dataset", train=False, download=True, transform=ToTensor())

    effective_size = len(org_ds) - len(org_ds) % args.NUM_CLIENT
    idx_groups = torch.randperm(effective_size).reshape(args.NUM_CLIENT, -1)
    fed_dss = [ClientDataset(org_ds, idx) for idx in idx_groups.tolist()]

    genetors = [
        torch.Generator().manual_seed(args.SEED + i) if args.SEED >= 0 else None
        for i in range(args.NUM_CLIENT)
    ]
    fed_loader = [
        DataLoader(ds, args.BATCH_SIZE, shuffle=True, generator=gtr)
        for ds, gtr in zip(fed_dss, genetors)
    ]
    test_loader = DataLoader(test_ds, args.BATCH_SIZE * 4)

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
    MFL(
        model=model,
        fed_loader=fed_loader,
        test_loader=test_loader,
        criterion=criterion,
        args=args,
    ).fit(args.NUM_CLIENT, args.ACTIVE_CLIENT, args.SERVER_EPOCHS)


if __name__ == "__main__":
    test_mfl()