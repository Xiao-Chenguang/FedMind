from flair.algs.fedavg import FedAvg
from flair.utils import EasyDict
from flair.data import ClientDataset

from torch import nn
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def test_fedavg():
    args = EasyDict()
    args.epochs = 1
    args.lr = 0.1
    args.n_clients = 100
    args.NUM_PROCESS = 5
    args.CLIENT = {}
    args.CLIENT.EPOCHS = 3  # type: ignore
    args.optim = {}
    args.optim.name = "SGD"  # type: ignore
    args.optim.lr = 0.001  # type: ignore

    org_ds = datasets.MNIST(
        "dataset",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    test_ds = datasets.MNIST(
        "dataset",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    effective_size = len(org_ds) - len(org_ds) % args.n_clients  # type: ignore
    idx_groups = torch.randperm(effective_size).reshape(args.n_clients, -1)  # type: ignore
    fed_dss = [ClientDataset(org_ds, idx) for idx in idx_groups.tolist()]
    fed_loader = [DataLoader(test_ds, batch_size=32, shuffle=True) for ds in fed_dss]
    test_loader = DataLoader(test_ds, batch_size=32)

    classes = 10
    shape = org_ds[0][0].shape[0] * org_ds[0][0].shape[1] * org_ds[0][0].shape[2]
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(shape, 32),
        nn.ReLU(),
        nn.Linear(32, classes),
    )

    criterion = nn.CrossEntropyLoss()

    FedAvg(
        model=model,
        fed_loader=fed_loader,
        test_loader=test_loader,
        criterion=criterion,
        args=args,
    ).fit(args.n_clients, 2, 2)  # type: ignore

    assert True


if __name__ == "__main__":
    test_fedavg()
