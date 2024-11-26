import logging
from typing import Any

from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fedmind.server import FedAlg
from fedmind.utils import EasyDict, StateDict


def set_momentum_buffer(
    model: Module, optimizer: Optimizer, momentum_buffer: StateDict
):
    param_buffer = {p: momentum_buffer[name] for name, p in model.named_parameters()}
    for group in optimizer.param_groups:
        for p in group["params"]:
            optimizer.state[p]["momentum_buffer"] = param_buffer[p].clone()


def get_momentum_buffer(model: Module, optimizer: Optimizer) -> StateDict:
    return StateDict(
        {
            name: optimizer.state[param]["momentum_buffer"].clone()
            for name, param in model.named_parameters()
            if param in optimizer.state and "momentum_buffer" in optimizer.state[param]
        }
    )


class MFL(FedAlg):
    """The MFL algorithm.

    Original paper: Accelerating Federated Learning via Momentum Gradient Descent.

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
        self.logger.info(f"Start {self.__class__.__name__}.")

        assert config.OPTIM.NAME == "SGD", "MFL only supports Optimizer optimizer."
        assert config.OPTIM.MOMENTUM > 0, "Momentum should be greater than 0."
        self._gm_momentum = StateDict()
        self.dyn_args["momentum_buffer"] = self._gm_momentum

    def _aggregate_updates(self, updates: list[dict]) -> dict:
        """Aggregate updates to new model.

        Args:
            updates: The list of updates to aggregate.

        Returns:
            The aggregated metrics.
        """
        agg_update = sum([update["model_update"] for update in updates]) / len(updates)
        agg_momentum = sum([update["momentum"] for update in updates]) / len(updates)
        agg_loss = sum([update["train_loss"] for update in updates]) / len(updates)
        self._gm_params.add_(agg_update)
        self._gm_momentum.copy_(agg_momentum)
        self.logger.info(f"Train loss: {agg_loss:.4f}")
        return {"train_loss": agg_loss}

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
        momentum_buffer: StateDict,
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
        if len(momentum_buffer) > 0:
            set_momentum_buffer(model, optimizer, momentum_buffer)
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
                loss.backward()
                optimizer.step()
                if loss.isnan():
                    logger.warning("Loss is NaN.")
                cost += loss.item()

        return {
            "model_update": model.state_dict(destination=StateDict()) - gm_params,
            "momentum": get_momentum_buffer(model, optimizer),
            "train_loss": cost / len(train_loader) / epochs,
        }
