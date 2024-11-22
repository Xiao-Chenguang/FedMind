import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from fedmind.algs.fedavg import FedAvg
from fedmind.config import get_config
from fedmind.data import ClientDataset


def test_fedavg_cuda():
    # 0. Prepare necessary arguments
    config = get_config("config.yaml")
    config.DEVICE = "cuda"
    config.ACTIVE_CLIENT = 10
    config.NUM_PROCESS = 11
    config.TEST_SUBPROCESS = True
    config.SERVER_EPOCHS = 20

    if config.SEED >= 0:
        torch.manual_seed(config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    assert torch.cuda.is_available(), "CUDA is not available"

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
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 5 * 5, 128),
        nn.ReLU(),
        nn.Linear(128, classes),
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


if __name__ == "__main__":
    test_fedavg_cuda()
