import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor

from fedmind.algs.fedavg import FedAvg
from fedmind.config import get_config
from fedmind.data import ClientDataset


def test_tiny_image_net_cuda(data_path: str = ""):
    assert data_path, "Please provide the path to the tiny-imagenet-200 dataset"
    # 0. Prepare necessary arguments
    config = get_config("config.yaml")
    config.DEVICE = "cuda"
    config.ACTIVE_CLIENT = 10
    config.NUM_PROCESS = 11
    config.TEST_SUBPROCESS = True
    config.SERVER_EPOCHS = 20
    config.LOG_LEVEL = 10
    config.SEED = 10

    if config.SEED >= 0:
        torch.manual_seed(config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    assert torch.cuda.is_available(), "CUDA is not available"

    # 1. Prepare Federated Learning DataSets
    ds_name = "zh-plus/tiny-imagenet"

    def process(examples):
        ts = ToTensor()
        examples["image"] = [ts(img).expand(3, -1, -1) for img in examples["image"]]
        return examples

    hgf_train = load_dataset(ds_name, cache_dir=data_path, split="train")
    hgf_test = load_dataset(ds_name, cache_dir=data_path, split="valid")
    hgf_train.set_transform(process, columns=["image"])  # type: ignore
    hgf_test.set_transform(process, columns=["image"])  # type: ignore

    train_image = torch.stack(hgf_train["image"]).clone() / 255.0  # type: ignore
    train_label = torch.tensor(hgf_train["label"], dtype=int)  # type: ignore
    test_image = torch.stack(hgf_test["image"]).clone() / 255.0  # type: ignore
    test_label = torch.tensor(hgf_test["label"], dtype=int)  # type: ignore
    del hgf_train, hgf_test

    org_ds = TensorDataset(train_image, train_label)
    test_ds = TensorDataset(test_image, test_label)

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
    classes = 1000
    # input size: 3x64x64
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 16 * 16, 128),
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
    mp.set_start_method("spawn")
    path = "/bask/homes/c/cxx075/Chenguang/projects/FedGA/datasets/tiny-imagenet"
    test_tiny_image_net_cuda(path)
