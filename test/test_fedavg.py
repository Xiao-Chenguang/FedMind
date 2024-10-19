import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from flair.algs.fedavg import FedAvg
from flair.data import ClientDataset
from flair.utils import EasyDict


def test_fedavg():
    args = EasyDict()
    args.SERVER_EPOCHS = 3
    args.LR = 0.1
    args.NUM_CLIENT = 100
    args.ACTIVE_CLIENT = 10
    args.NUM_PROCESS = 5
    args.BATCH_SIZE = 32
    args.CLIENT_EPOCHS = 3  # type: ignore
    args.OPTIM = {
        "NAME": "SGD",
        "LR": 0.1,
        "MOMENTUM": 0.9,
    }

    # 1. Prepare Federated Learning DataSets
    org_ds = MNIST("dataset", train=True, download=True, transform=ToTensor())
    test_ds = MNIST("dataset", train=False, download=True, transform=ToTensor())

    effective_size = len(org_ds) - len(org_ds) % args.NUM_CLIENT  # type: ignore
    idx_groups = torch.randperm(effective_size).reshape(args.NUM_CLIENT, -1)  # type: ignore
    fed_dss = [ClientDataset(org_ds, idx) for idx in idx_groups.tolist()]

    fed_loader = [DataLoader(ds, batch_size=32, shuffle=True) for ds in fed_dss]
    test_loader = DataLoader(test_ds, batch_size=32)

    # 2. Prepare Model and Criterion
    classes = 10
    shape = org_ds[0][0].shape[0] * org_ds[0][0].shape[1] * org_ds[0][0].shape[2]
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(shape, 32),
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
        args=args,
    ).fit(args.NUM_CLIENT, args.ACTIVE_CLIENT, args.SERVER_EPOCHS)  # type: ignore

    assert True


if __name__ == "__main__":
    test_fedavg()
