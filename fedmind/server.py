import logging
import os
from copy import deepcopy
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
        self._model = model.to(args.DEVICE)
        self._fed_loader = fed_loader
        self._test_loader = test_loader
        self._criterion = criterion
        self.args = args

        self._gm_params = self._model.state_dict(destination=StateDict()) * 1
        optim: dict = self.args.OPTIM
        if optim["NAME"] == "SGD":
            self._optimizer = SGD(
                self._model.parameters(),
                lr=optim["LR"],
                momentum=optim.get("MOMENTUM", 0),
                dampening=optim.get("DAMPENING", 0),
            )
        else:
            raise NotImplementedError(f"Optimizer {optim['NAME']} not implemented.")

        self._wb_run = wandb.init(
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
        self._task_queue = mp.Queue()
        self._result_queue = mp.Queue()
        self._test_queue = mp.Queue()

        # Start client processes
        self._processes = mp.spawn(
            self._create_worker_process,
            nprocs=self.args.NUM_PROCESS,
            join=False,  # Do not wait for processes to finish
            args=(
                self._task_queue,
                self._result_queue,
                self._test_queue,
                self._train_client,
                self._test_server,
                self._model,
                self.args.OPTIM,
                self._criterion,
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
            self._task_queue.put(("STOP",))

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
        if self.args.NUM_PROCESS == 0 or not self.args.TEST_SUBPROCESS:
            return self._test_server(
                self._model,
                self._gm_params,
                self._test_loader,
                self._criterion,
                self.logger,
                self.args,
            )
        else:
            return self._test_queue.get()

    @staticmethod
    def _test_server(
        model: Module,
        gm_params: StateDict,
        test_loader: DataLoader,
        criterion: _Loss,
        logger: logging.Logger,
        args: EasyDict,
    ) -> dict:
        """Test the model.

        Args:
            model: The model to test.
            gm_params: The global model parameters.
            test_loader: The DataLoader object that contains the test data.
            criterion: The loss function to use.
            logger: The logger object to log the testing process.

        Returns:
            The evaluation metrics.
        """

        total_loss = 0
        correct = 0
        total = 0
        model.load_state_dict(gm_params)
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(args.DEVICE)
                labels = labels.to(args.DEVICE)
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
                            self._model,
                            self._gm_params,
                            self._fed_loader[cid],
                            self._optimizer,
                            self._criterion,
                            self.args.CLIENT_EPOCHS,
                            self.logger,
                            self.args,
                        )
                    )
            else:
                # Parallel simulation with torch.multiprocessing
                if self.args.TEST_SUBPROCESS:
                    self._task_queue.put(("TEST", self._gm_params, self._test_loader))
                for cid in clients:
                    self._task_queue.put(
                        ("TRAIN", self._gm_params, self._fed_loader[cid])
                    )
                for cid in clients:
                    updates.append(self._result_queue.get())

            # 3. Aggregate updates to new model
            train_metrics = self._aggregate_updates(updates)
            del updates  # Fix shared cuda tensors issue (release tensor generated by child process)

            # 4. Evaluate the new model
            test_metrics = self._evaluate()

            # 5. Log metrics
            self._wb_run.log(train_metrics | test_metrics)

        # Terminate multi-process environment
        if self.args.NUM_PROCESS > 0:
            self.__del_mp__()

        # Finish wandb run and sync
        self._wb_run.finish()
        os.system(f"wandb sync {os.path.dirname(self._wb_run.dir)}")

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
        *args_,
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
        test_queue: mp.Queue,
        train_func: Callable,
        test_func: Callable,
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
        model = deepcopy(model)
        if optim["NAME"] == "SGD":
            optimizer = SGD(model.parameters(), lr=optim["LR"])
        else:
            raise NotImplementedError(f"Optimizer {optim['NAME']} not implemented.")
        while True:
            task = task_queue.get()
            if task[0] == "STOP":
                break
            elif task[0] == "TRAIN":
                _, parm, loader = task
                result = train_func(
                    model, parm, loader, optimizer, criterion, epochs, logger, args
                )
                result_queue.put(result)
            elif task[0] == "TEST":
                _, parm, loader = task
                result = test_func(model, parm, loader, criterion, logger, args)
                test_queue.put(result)
            else:
                raise ValueError(f"Unknown task type {task[0]}")
