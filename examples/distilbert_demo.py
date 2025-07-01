import logging
import multiprocessing
from typing import Any

import torch
from datasets import load_dataset
from torch import Tensor, nn
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import DistilBertConfig
from transformers.models.distilbert.modeling_distilbert import (
    DistilBertForSequenceClassification,
)
from transformers.models.distilbert.tokenization_distilbert_fast import (
    DistilBertTokenizerFast,
)

from fedmind.algs.fedavg import FedAvg
from fedmind.config import get_config
from fedmind.data import ClientDataset
from fedmind.utils import EasyDict, StateDict


class FedAvgTextClassification(FedAvg):
    """Federated Averaging Algorithm for Text Classification."""

    def __init__(self, model, fed_loader, test_loader, criterion, config):
        super().__init__(model, fed_loader, test_loader, criterion, config)
        self.logger.info(f"Start {self.__class__.__name__}.")

    @staticmethod
    def _train_client(
        model: Module,
        gm_params: StateDict,
        train_loader: DataLoader,
        optimizer: Optimizer,
        criterion: _Loss,
        epochs: int,
        logger: logging.Logger,
        config: EasyDict,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        """Train the model with given environment.

        Args:
            model: The model to train.
            gm_params: The global model parameters.
            train_loader: The DataLoader object that contains the training data.
            optimizer: The optimizer to use.
            criterion: The loss function to use.
            epochs: The number of epochs to train the model.
            logger: The logger object to log the training process.
            config: The configuration dictionary.

        Returns:
            A dictionary containing the trained model parameters, training loss and more.
        """
        # Train the model
        model.load_state_dict(gm_params)
        cost = 0.0
        model.train()
        for epoch in range(epochs):
            logger.debug(f"Epoch {epoch + 1}/{epochs}")
            for batch in train_loader:
                labels = batch.pop("label").to(config.DEVICE)
                inputs = {k: v.to(config.DEVICE) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = model(**inputs)
                loss: Tensor = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                if loss.isnan():
                    logger.warning("Loss is NaN.")
                cost += loss.item()

        return {
            "model_update": model.state_dict(destination=StateDict()) - gm_params,
            "train_loss": cost / len(train_loader) / epochs,
        }

    @staticmethod
    def _test_server(
        model: Module,
        gm_params: StateDict,
        test_loader: DataLoader,
        criterion: _Loss,
        logger: logging.Logger,
        config: EasyDict,
        *args,
        **kwargs,
    ) -> dict:
        """Test the model.

        Args:
            model: The model to test.
            gm_params: The global model parameters.
            test_loader: The DataLoader object that contains the test data.
            criterion: The loss function to use.
            logger: The logger object to log the testing process.
            config: The configuration dictionary.

        Returns:
            The evaluation metrics dict.
        """

        total_loss = 0
        correct = 0
        total = 0
        model.load_state_dict(gm_params)
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                labels = batch.pop("label").to(config.DEVICE)
                inputs = {k: v.to(config.DEVICE) for k, v in batch.items()}
                outputs = model(**inputs)
                loss: Tensor = criterion(outputs.logits, labels)
                total_loss += loss.item()
                predicted = torch.argmax(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        logger.info(f"Test Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

        return {"test_loss": total_loss, "test_accuracy": accuracy}


def test_fedavg():
    # 0. Prepare necessary arguments
    config = get_config("config.yaml")
    if config.SEED >= 0:
        torch.manual_seed(config.SEED)

    multiprocessing.set_start_method("spawn")  # avoid deadlock of tokenizer with mp

    # 1. Prepare Federated Learning DataSets
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    org_ds = load_dataset("IMDB", split="train[:5%]", cache_dir="dataset").map(
        lambda x: tokenizer(
            x["text"], truncation=True, padding="max_length", max_length=512
        ),
        batched=True,
    )
    org_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])  # type: ignore

    test_ds = load_dataset("IMDB", split="test[:5%]", cache_dir="dataset").map(
        lambda x: tokenizer(
            x["text"], truncation=True, padding="max_length", max_length=512
        ),
        batched=True,
    )
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])  # type: ignore

    effective_size = len(org_ds) - len(org_ds) % config.NUM_CLIENT  # type: ignore
    idx_groups = torch.randperm(effective_size).reshape(config.NUM_CLIENT, -1)
    fed_dss = [ClientDataset(org_ds, idx) for idx in idx_groups.tolist()]  # type: ignore

    genetors = [
        torch.Generator().manual_seed(config.SEED + i) if config.SEED >= 0 else None
        for i in range(config.NUM_CLIENT)
    ]
    fed_loader = [
        DataLoader(ds, config.BATCH_SIZE, shuffle=True, generator=gtr)
        for ds, gtr in zip(fed_dss, genetors)
    ]
    test_loader = DataLoader(test_ds, config.BATCH_SIZE * 4)  # type: ignore
    # for batch in test_loader:
    #     for k, v in batch.items():
    #         print(f"{k} type: {type(v)}")

    # 2. Prepare Model and Criterion
    classes = 2
    model = model = DistilBertForSequenceClassification(
        DistilBertConfig(num_labels=classes)
    )

    criterion = nn.CrossEntropyLoss()

    # 3. Run Federated Learning Simulation
    FedAvgTextClassification(
        model=model,
        fed_loader=fed_loader,
        test_loader=test_loader,
        criterion=criterion,
        config=config,
    ).fit(config.NUM_CLIENT, config.ACTIVE_CLIENT, config.SERVER_EPOCHS)


if __name__ == "__main__":
    test_fedavg()
