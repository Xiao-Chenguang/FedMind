import logging
from typing import Any

from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from fedmind.algs.fedavg import FedAvg
from fedmind.utils import EasyDict, StateDict


class FedProx(FedAvg):
    """FedProx:
    [Federated Optimization in Heterogeneous Networks](https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html)
    """

    def __init__(
        self,
        model: Module,
        fed_loader: list[DataLoader],
        test_loader: DataLoader,
        criterion: _Loss,
        config: EasyDict,
    ):
        super().__init__(model, fed_loader, test_loader, criterion, config)
        assert hasattr(config, "PROX_MU"), "PROX_MU is not set for FedProx."

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
        mu = config.PROX_MU

        # Train the model
        model.load_state_dict(gm_params)
        cost = 0.0
        model.train()
        for epoch in range(epochs):
            logger.debug(f"Epoch {epoch + 1}/{epochs}")
            for inputs, labels in train_loader:
                inputs = inputs.to(config.DEVICE)
                labels = labels.to(config.DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss: Tensor = criterion(outputs, labels)

                # Add proximal term
                for key, param in model.named_parameters():
                    gp = gm_params[key]
                    loss += (mu / 2) * (param - gp).norm(2) ** 2

                loss.backward()
                optimizer.step()
                if loss.isnan():
                    logger.warning("Loss is NaN.")
                cost += loss.item()

        return {
            "model_update": model.state_dict(destination=StateDict()) - gm_params,
            "train_loss": cost / len(train_loader) / epochs,
        }
