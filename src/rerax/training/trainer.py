import abc
from typing import Any

import chex
from flax import nnx

from rerax.tasks.base import Task


class BaseTrainerMeta(type(nnx.Module), abc.ABCMeta):
    pass


class BaseTrainer(nnx.Module, metaclass=BaseTrainerMeta):
    def __init__(self, model: nnx.Module, task: Task, optimizer: nnx.Optimizer):
        self._model = model
        self._task = task
        self._optimizer = optimizer

    @abc.abstractmethod
    def train_step(self, batch) -> dict[str, Any]:
        pass

    def fit(
        self, data_loader, num_epochs: int = 1, *, training=True
    ) -> list[dict[str, float]]:
        history_per_epoch = []
        epoch_metrics = nnx.MultiMetric(loss=nnx.metrics.Average())
        for epoch in range(num_epochs):
            epoch_metrics.reset()
            for batch in data_loader:
                metrics = self.train_step(batch)
                epoch_metrics.update(values=metrics["loss"])

        current_result = epoch_metrics.compute()
        history_per_epoch.append(current_result)

        return history_per_epoch


# nnx.jitは関数の引数をJAXに持ち込む
# nnx.Moduleを継承することで、`Trainer`クラスは自動的にJAXが扱えるオブジェクト(pytree)として登録して追跡できるようにする
class Trainer(BaseTrainer):
    def __init__(self, model: nnx.Module, task: Task, optimizer: nnx.Optimizer):
        self._model = model
        self._task = task
        self._optimizer = optimizer

    @nnx.jit
    def train_step(self, batch: dict[str, chex.Array]) -> dict[str, Any]:
        def loss_fn(model):
            outputs = model(batch)
            loss = self._task.compute_loss(outputs, batch, training=True)
            return loss, outputs

        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, outputs), grads = grad_fn(self._model)
        self._optimizer.update(self._model, grads=grads)

        metrics = self._task.compute_metrics(
            outputs,
            batch,
        )
        if "loss" not in metrics:
            metrics["loss"] = loss
        return metrics
