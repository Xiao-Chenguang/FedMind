import logging
import multiprocessing as mp
import os
from copy import deepcopy
from typing import Any, Callable, Optional

from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import SGD, Optimizer
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
        args: EasyDict,
    ):
        super().__init__(model, fed_loader, test_loader, criterion, args)
        self.logger.info(f"Start {self.__class__.__name__}.")

        assert args.OPTIM.NAME == "SGD", "MFL only supports Optimizer optimizer."
        assert args.OPTIM.MOMENTUM > 0, "Momentum should be greater than 0."
        self._momentum_buffer = None

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
        self._gm_params += agg_update
        self._gm_momentum = agg_momentum
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
        args: EasyDict,
        momentum_buffer: Optional[StateDict],
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
        if momentum_buffer is not None:
            set_momentum_buffer(model, optimizer, momentum_buffer)
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
            "momentum": get_momentum_buffer(model, optimizer),
            "train_loss": cost / len(train_loader) / epochs,
        }

    def fit(self, pool: int, num_clients: int, num_rounds: int):
        """Fit the model with federated learning.

        Args:
            pool: The total number of clients to select from.
            num_clients: The number of clients to select.
            num_rounds: The number of federated learning rounds.
        """
        for round in range(num_rounds):
            self.logger.info(f"Round {round + 1}/{num_rounds}")

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
                            self._momentum_buffer,
                        )
                    )
            else:
                # Parallel simulation with torch.multiprocessing
                if self.args.TEST_SUBPROCESS:
                    self._task_queue.put(("TEST", self._gm_params, self._test_loader))
                for cid in clients:
                    self._task_queue.put(
                        (
                            "TRAIN",
                            self._gm_params,
                            self._fed_loader[cid],
                            self._momentum_buffer,
                        )
                    )
                for cid in clients:
                    updates.append(self._result_queue.get())

            # 3. Aggregate updates to new model
            train_metrics = self._aggregate_updates(updates)
            del updates  # Fix shared cuda tensors issue (release tensor generated by child process)

            # 4. Evaluate the new model
            test_metrics = self._evaluate()

            # 5. Log metrics
            # wandb.log(train_metrics | test_metrics)
            self._wb_run.log(train_metrics | test_metrics)

        # Terminate multi-process environment
        if self.args.NUM_PROCESS > 0:
            self.__del_mp__()

        # Finish wandb run and sync
        self._wb_run.finish()
        os.system(f"wandb sync {os.path.dirname(self._wb_run.dir)}")

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
                _, parm, loader, momentum_buffer = task
                logger.debug("received train task")
                result = train_func(
                    model,
                    parm,
                    loader,
                    optimizer,
                    criterion,
                    epochs,
                    logger,
                    args,
                    momentum_buffer,
                )
                result_queue.put(result)
                logger.debug("result put to queue")
            elif task[0] == "TEST":
                _, parm, loader = task
                result = test_func(model, parm, loader, criterion, logger, args)
                test_queue.put(result)
            else:
                raise ValueError(f"Unknown task type {task[0]}")
