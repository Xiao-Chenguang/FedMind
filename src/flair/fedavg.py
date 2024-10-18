from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from flair.server import FedAlg
from flair.utils import EasyDict, StateDict


class FedAvg(FedAlg):
    """The federated averaging algorithm."""

    def __init__(
        self,
        model: Module,
        fed_loader: list[DataLoader],
        test_loader: DataLoader,
        optimizer: Optimizer,
        criterion: _Loss,
        args: EasyDict,
    ):
        super().__init__(model, fed_loader, test_loader, optimizer, criterion, args)
        self.logger.info("Start Federated Averaging.")

    def _aggregate_updates(self, updates: list[dict]):
        """Aggregate updates to new model.

        Args:
            updates: The list of updates to aggregate.
        """
        state_dict = self.model.state_dict(destination=StateDict())
        for update in updates:
            state_dict += update["model_parameters"]
        self.model.load_state_dict(state_dict)
