import logging
import os
from typing import Any, Callable

import torch
import torch.multiprocessing as mp
import wandb
import yaml
from torch import Tensor, randperm
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from fedmind.utils import EasyDict, StateDict


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
        self.model = model.to(args.DEVICE)
        self.fed_loader = fed_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.args = args

        self.gm_params = self.model.state_dict(destination=StateDict())
        optim: dict = self.args.OPTIM
        if optim["NAME"] == "SGD":
            self.optimizer = SGD(self.model.parameters(), lr=optim["LR"])
        else:
            raise NotImplementedError(f"Optimizer {optim['NAME']} not implemented.")

        self.wb_run = wandb.init(
            mode="offline",
            project=args.get("WB_PROJECT", "fedmind"),
            entity=args.get("WB_ENTITY", "wandb"),
            config=self.args.to_dict(),
            settings=wandb.Settings(_disable_stats=True, _disable_machine_info=True),
        )

        logging.basicConfig(
            level=args.LOG_LEVEL,
            format="%(asctime)s %(levelname)s [%(processName)s] %(message)s",
        )
        self.logger = logging.getLogger("Server")
        self.logger.info(f"Get following configs:\n{yaml.dump(args.to_dict())}")

        if self.args.NUM_PROCESS > 0:
            self.__init_mp__()

    def __init_mp__(self):
        """Set up multi-process environment.

        Create `worker processes`, `task queue` and `result queue`.
        """

        # Create queues for task distribution and result collection
        mp.set_start_method("spawn")
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()

        # Start client processes
        self._processes = mp.spawn(
            self._create_worker_process,
            nprocs=self.args.NUM_PROCESS,
            join=False,  # Do not wait for processes to finish
            args=(
                self.task_queue,
                self.result_queue,
                self._train_client,
                self.model,
                self.args.OPTIM,
                self.criterion,
                self.args.CLIENT_EPOCHS,
                self.args.LOG_LEVEL,
                self.args,
            ),
        )
        self.logger.debug(f"Started {self.args.NUM_PROCESS} worker processes.")

    def __del_mp__(self):
        """Terminate multi-process environment."""

        # Terminate all client processes
        for _ in range(self.args.NUM_PROCESS):
            self.task_queue.put("STOP")

        # Wait for all client processes to finish
        assert self._processes is not None, "Worker processes no found."
        self._processes.join()

    def _select_clients(self, pool: int, num_clients: int) -> list[int]:
        """Select active clients from the pool.

        Args:
            pool: The total number of clients to select from.
            num_clients: The number of clients to select.

        Returns:
            The list of selected clients indices.
        """
        return randperm(pool)[:num_clients].tolist()

    def _aggregate_updates(self, updates: list[dict]) -> dict:
        """Aggregate updates to new model.

        Args:
            updates: The list of updates to aggregate.

        Returns:
            The aggregated metrics.
        """
        raise NotImplementedError("Aggregate updates method must be implemented.")

    def _evaluate(self) -> dict:
        """Evaluate the model.

        Returns:
            The evaluation metrics.
        """
        model: Module = self.model
        gm_params: StateDict = self.gm_params
        test_loader: DataLoader = self.test_loader
        criterion: _Loss = self.criterion
        logger: logging.Logger = self.logger

        total_loss = 0
        correct = 0
        total = 0
        model.load_state_dict(gm_params)
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.args.DEVICE)
                labels = labels.to(self.args.DEVICE)
                outputs = model(inputs)
                loss: Tensor = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        logger.info(f"Test Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

        return {"test_loss": total_loss, "test_accuracy": accuracy}

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
            if self.args.NUM_PROCESS == 0:
                # Serial simulation instead of parallel
                for cid in clients:
                    updates.append(
                        self._train_client(
                            self.model,
                            self.gm_params,
                            self.fed_loader[cid],
                            self.optimizer,
                            self.criterion,
                            self.args.CLIENT_EPOCHS,
                            self.logger,
                            self.args,
                        )
                    )
            else:
                # Parallel simulation with torch.multiprocessing
                for cid in range(num_clients):
                    self.task_queue.put((self.gm_params, self.fed_loader[cid]))
                for cid in range(num_clients):
                    updates.append(self.result_queue.get())

            # 3. Aggregate updates to new model
            train_metrics = self._aggregate_updates(updates)

            # 4. Evaluate the new model
            test_metrics = self._evaluate()

            # 5. Log metrics
            self.wb_run.log(train_metrics | test_metrics)

        # Terminate multi-process environment
        if self.args.NUM_PROCESS > 0:
            self.__del_mp__()

        # Finish wandb run and sync
        self.wb_run.finish()
        os.system(f"wandb sync {os.path.dirname(self.wb_run.dir)}")

    @staticmethod
    def _train_client(
        model: Module,
        gm_params: StateDict,
        train_loader: DataLoader,
        optimizer: Optimizer,
        criterion: _Loss,
        epochs: int,
        logger: logging.Logger,
        args: EasyDict,
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
            logger.debug(f"Epoch {epoch + 1}/{epochs}")
            for inputs, labels in train_loader:
                inputs = inputs.to(args.DEVICE)
                labels = labels.to(args.DEVICE)
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

    @staticmethod
    def _create_worker_process(
        worker_id: int,
        task_queue: mp.Queue,
        result_queue: mp.Queue,
        client_func: Callable,
        model: Module,
        optim: dict,
        criterion: _Loss,
        epochs: int,
        log_level: int,
        args: EasyDict,
    ):
        """Train process for multi-process environment.

        Args:
            worker_id: The worker process id.
            task_queue: The task queue for task distribution.
            result_queue: The result queue for result collection.
            client_func: The client function to train the model.
            model: The model to train.
            optim: dictionary containing the optimizer parameters.
            criterion: The loss function to use.
            epochs: The number of epochs to train the model.
        """
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s %(levelname)s [%(processName)s] %(message)s",
        )
        logger = logging.getLogger(f"Worker-{worker_id}")
        logger.info(f"Worker-{worker_id} started.")
        if optim["NAME"] == "SGD":
            optimizer = SGD(model.parameters(), lr=optim["LR"])
        else:
            raise NotImplementedError(f"Optimizer {optim['NAME']} not implemented.")
        while True:
            task = task_queue.get()
            if task == "STOP":
                break
            else:
                parm, loader = task
                result = client_func(
                    model, parm, loader, optimizer, criterion, epochs, logger, args
                )
                result_queue.put(result)
