from flair.fedavg import FedAvg
from flair.utils import EasyDict

from torch import nn, optim
import torch
from torch.utils.data import DataLoader, Dataset


class DumyDataSet(Dataset):
    def __init__(self, size, shape, classes):
        self.data = torch.randn(size, shape)
        self.target = torch.randint(0, classes, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def test_fedavg():
    args = EasyDict()
    args.epochs = 1
    args.lr = 0.1
    args.n_clients = 10
    args.NUM_PROCESS = 5
    args.CLIENT = {}
    args.CLIENT.EPOCHS = 10  # type: ignore

    size = 100
    shape = 100000
    classes = 3
    datasets = [DumyDataSet(size, shape, classes) for _ in range(args.n_clients)]  # type: ignore
    fed_loader = [DataLoader(ds, batch_size=32, shuffle=True) for ds in datasets]
    test_loader = DataLoader(DumyDataSet(size, shape, classes), batch_size=32)

    model = nn.Sequential(nn.Linear(shape, 32), nn.ReLU(), nn.Linear(32, classes))

    optimizer = optim.SGD(model.parameters(), lr=args.lr)  # type: ignore
    criterion = nn.CrossEntropyLoss()

    FedAvg(
        model=model,
        fed_loader=fed_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        args=args,
    ).fit(args.n_clients, 5, 2)  # type: ignore

    assert True


if __name__ == "__main__":
    test_fedavg()
