import logging
from logging import Logger
from typing import Any

import torch
import torch.multiprocessing as mp
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from flair.utils import StateDict


def train(
    model: Module,
    gm_params: StateDict,
    train_loader: DataLoader,
    optimizer: Optimizer,
    criterion: _Loss,
    epochs: int,
    logger: Logger,
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

    Returns:
        A dictionary containing the trained model parameters.
    """
    # Train the model
    model.load_state_dict(gm_params)
    cost = 0.0
    model.train()
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss: Tensor = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if loss.isnan():
                logger.warning("Loss is NaN.")
            cost += loss.item()

    return {
        "model_update": model.state_dict(destination=StateDict()) - gm_params,
        "train_loss": cost / len(train_loader) / epochs,
    }


def test(
    model: Module,
    gm_params: StateDict,
    test_loader: DataLoader,
    criterion: _Loss,
    logger: Logger,
) -> dict[str, Any]:
    """Test the model with given environment.

    Args:
        model: The model to test.
        gm_params: The global model parameters.
        test_loader: The DataLoader object that contains the test data.
        criterion: The loss function to use.
        logger: The logger object to log the testing process.

    Returns:
        A dictionary containing the test results.
    """
    total_loss = 0.0
    correct = 0
    total = 0
    model.load_state_dict(gm_params)
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss: Tensor = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    logger.info(f"Test Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

    return {"test_loss": total_loss, "test_accuracy": accuracy}


def train_process(
    worker_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    model: Module,
    optim: dict,
    criterion: _Loss,
    epochs: int,
):
    """Train process for multi-process environment.

    Args:
        worker_id: The worker process id.
        task_queue: The task queue for task distribution.
        result_queue: The result queue for result collection.
        model: The model to train.
        optim: dictionary containing the optimizer parameters.
        criterion: The loss function to use.
        epochs: The number of epochs to train the model.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(processName)s] %(message)s",
    )
    logger = logging.getLogger(f"Worker-{worker_id}")
    logger.info(f"Worker-{worker_id} started.")
    if optim["name"] == "SGD":
        optimizer = SGD(model.parameters(), lr=optim["lr"])
    else:
        raise NotImplementedError(f"Optimizer {optim['name']} not implemented.")
    while True:
        task = task_queue.get()
        if task == "STOP":
            break
        else:
            parm, loader = task
            result = train(model, parm, loader, optimizer, criterion, epochs, logger)
            result_queue.put(result)
