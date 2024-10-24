from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from fedmind.server import FedAlg
from fedmind.utils import EasyDict


class FedAvg(FedAlg):
    """The federated averaging algorithm."""

    def __init__(
        self,
        model: Module,
        fed_loader: list[DataLoader],
        test_loader: DataLoader,
        criterion: _Loss,
        args: EasyDict,
    ):
        super().__init__(model, fed_loader, test_loader, criterion, args)
        self.logger.info(f"Start {self.__class__.__name__}.")

    def _aggregate_updates(self, updates: list[dict]) -> dict:
        """Aggregate updates to new model.

        Args:
            updates: The list of updates to aggregate.

        Returns:
            The aggregated metrics.
        """
        agg_update = sum([update["model_update"] for update in updates]) / len(updates)
        agg_loss = sum([update["train_loss"] for update in updates]) / len(updates)
        self.gm_params += agg_update
        self.logger.info(f"Train loss: {agg_loss:.4f}")
        return {"train_loss": agg_loss}
