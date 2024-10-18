import logging

import torch.multiprocessing as mp
from torch import randperm
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import SGD
from torch.utils.data import DataLoader

from flair.client import test, train, train_process
from flair.utils import EasyDict, StateDict


class FedAlg:
    """The federated learning algorithm base class.

    FL simulation is composed of the following steps repeatively:
    1. Select active clients from pool and broadcast model.
    2. Synchornous clients training.
    3. Get updates from clients feedback.
    4. Aggregate updates to new model.
    5. Evaluate the new model.
    """

    def __init__(
        self,
        model: Module,
        fed_loader: list[DataLoader],
        test_loader: DataLoader,
        criterion: _Loss,
        args: EasyDict,
    ):
        self.model = model
        self.fed_loader = fed_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.args = args

        self.gm_params = self.model.state_dict(destination=StateDict())
        optim: dict = self.args.optim  # type: ignore
        if optim["name"] == "SGD":
            self.optimizer = SGD(self.model.parameters(), lr=optim["lr"])
        else:
            raise NotImplementedError(f"Optimizer {optim['name']} not implemented.")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(processName)s] %(message)s",
        )
        self.logger = logging.getLogger("Server")

        if self.args.NUM_PROCESS > 0:  # type: ignore
            self.__init_mp__()

    def __init_mp__(self):
        """Set up multi-process environment.

        Create `worker processes`, `task queue` and `result queue`.
        """

        # Create queues for task distribution and result collection
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()

        # Start client processes
        self.processes = []
        for worker_id in range(self.args.NUM_PROCESS):  # type: ignore
            args = (
                worker_id,
                self.task_queue,
                self.result_queue,
                self.model,
                self.args.optim,  # type: ignore
                self.criterion,
                self.args.CLIENT.EPOCHS,  # type: ignore
            )
            p = mp.Process(target=train_process, args=args)
            p.start()
            self.processes.append(p)

    def __del_mp__(self):
        """Terminate multi-process environment."""

        # Terminate all client processes
        for _ in range(self.args.NUM_PROCESS):  # type: ignore
            self.task_queue.put("STOP")

        # Wait for all client processes to finish
        for p in self.processes:
            p.join()

    def _select_clients(self, pool: int, num_clients: int) -> list[int]:
        """Select active clients from the pool.

        Args:
            pool: The total number of clients to select from.
            num_clients: The number of clients to select.

        Returns:
            The list of selected clients indices.
        """
        return randperm(pool)[:num_clients].tolist()

    def _aggregate_updates(self, updates: list[dict]):
        """Aggregate updates to new model.

        Args:
            updates: The list of updates to aggregate.
        """
        raise NotImplementedError("Aggregate updates method must be implemented.")

    def _evaluate(self):
        """Evaluate the model."""
        test(
            self.model,
            self.gm_params,
            self.test_loader,
            self.criterion,
            self.logger,
        )

    def fit(self, pool: int, num_clients: int, num_rounds: int):
        """Fit the model with federated learning.

        Args:
            pool: The total number of clients to select from.
            num_clients: The number of clients to select.
            num_rounds: The number of federated learning rounds.
        """
        for _ in range(num_rounds):
            self.logger.info(f"Round {_ + 1}/{num_rounds}")

            # 1. Select active clients from pool and broadcast model
            clients = self._select_clients(pool, num_clients)

            # 2. Synchornous clients training
            updates = []
            if self.args.NUM_PROCESS == 0:  # type: ignore
                # Serial simulation instead of parallel
                for cid in clients:
                    updates.append(
                        train(
                            self.model,
                            self.gm_params,
                            self.fed_loader[cid],
                            self.optimizer,
                            self.criterion,
                            self.args.CLIENT.EPOCHS,  # type: ignore
                            self.logger,
                        )
                    )
            else:
                # Parallel simulation with torch.multiprocessing
                for cid in range(num_clients):
                    self.task_queue.put((self.gm_params, self.fed_loader[cid]))
                for cid in range(num_clients):
                    updates.append(self.result_queue.get())

            # 3. Aggregate updates to new model
            self._aggregate_updates(updates)

            # 4. Evaluate the new model
            self._evaluate()

        # Terminate multi-process environment
        if self.args.NUM_PROCESS > 0:  # type: ignore
            self.__del_mp__()