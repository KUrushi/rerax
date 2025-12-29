import abc
from typing import Any

import chex
from flax import nnx
from grain.python import DataLoader

from rerax.tasks.base import Task
from rerax.tracking.base import BaseTracker


class BaseTrainerMeta(type(nnx.Module), abc.ABCMeta):
    pass


class BaseTrainer(nnx.Module, metaclass=BaseTrainerMeta):
    def __init__(
        self,
        model: nnx.Module,
        task: Task,
        optimizer: nnx.Optimizer,
        tracker: BaseTracker | None = None,
    ):
        self._model = model
        self._task = task
        self._optimizer = optimizer
        self._tracker = tracker

    @abc.abstractmethod
    def train_step(self, batch) -> dict[str, Any]:
        pass

    @abc.abstractmethod
    def eval_step(self, batch) -> dict[str, Any]:
        pass

    def fit(
        self,
        train_loader: DataLoader,
        total_steps: int,
        log_freq: int = 100,
        eval_loader: DataLoader | None = None,
        eval_freq: int | None = None,
    ) -> dict[str, Any]:
        train_metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
        iterator = iter(train_loader)
        current_step = 0

        history = []
        print(f"Start training for {total_steps} steps...")

        while current_step < total_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)

            step_metrics = self.train_step(batch)

            train_metrics.update(**step_metrics)
            current_step += 1

            if current_step % log_freq == 0:
                current_result = train_metrics.compute()
                print(f"Step {current_step}: {current_result}")
                if self._tracker:
                    self._tracker.log_metrics(current_result, step=current_step)

                history.append(current_result)
                train_metrics.reset()

            if eval_loader and eval_freq and current_step % eval_freq == 0:
                # TODO:
                # self.evaluate(eval_loader, current_step)
                pass

        return {"history": history}

    def evaluate(self, data_loader: DataLoader, current_step: int) -> dict[str, Any]:
        raise NotImplementedError("evaluateはまだ未実装です")


# nnx.jitは関数の引数をJAXに持ち込む
# nnx.Moduleを継承することで、`Trainer`クラスは自動的にJAXが扱えるオブジェクト(pytree)として登録して追跡できるようにする
class Trainer(BaseTrainer):
    @nnx.jit
    def train_step(
        self,
        batch: dict[str, chex.Array],
    ) -> dict[str, Any]:
        def loss_fn(model):
            outputs = model(batch)
            loss = self._task.compute_loss(
                outputs, batch, training=True, mask=batch.get("mask")
            )
            return loss, outputs

        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, outputs), grads = grad_fn(self._model)
        self._optimizer.update(self._model, grads=grads)

        metrics = self._task.compute_metrics(outputs, batch, mask=batch.get("mask"))
        if "loss" not in metrics:
            metrics["loss"] = loss
        return metrics
