import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from rerax.tasks.base import Task
from rerax.training.trainer import Trainer


class TestTrainer:
    """NNX Trainer Architectureのテスト"""

    def test_train_step_updates_params_without_jit(self):
        """train_stepを実行すると、モデルのパラメータが更新されるのことを確認"""

        class LinearModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.linear = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, batch):
                return self.linear(batch["inputs"])

        class MSETask(Task):
            def compute_loss(self, outputs, batch, *, training=True, mask=None):
                loss = jnp.mean((outputs - batch["targets"]) ** 2)
                return loss

            def compute_metrics(self, outputs, batch, *, mask=None):
                return {"loss": self.compute_loss(outputs, batch)}

        rngs = nnx.Rngs(0)
        model = LinearModel(rngs)
        task = MSETask()
        optimizer = nnx.Optimizer(model, optax.sgd(learning_rate=0.1), wrt=nnx.Param)

        trainer = Trainer(model=model, task=task, optimizer=optimizer)

        # パラメータのコピーを取得
        initial_params = jax.tree_util.tree_map(
            lambda x: x.copy(), nnx.state(model, nnx.Param)
        )

        x = jnp.ones((1, 2))
        y = jnp.array([[10.0]])
        batch = {"inputs": x, "targets": y}

        with jax.disable_jit():
            metrics = trainer.train_step(batch)

        assert "loss" in metrics
        current_params = nnx.state(model, nnx.Param)

        params_changed = False
        flat_initial = jax.tree_util.tree_leaves(initial_params)
        flat_current = jax.tree_util.tree_leaves(current_params)

        for p_init, p_curr in zip(flat_initial, flat_current):
            if not jnp.allclose(p_init, p_curr):
                params_changed = True
                break

            assert params_changed, "Model parameters should be updated after train_step"

    def test_train_step_updates_params_with_jit(self):
        """train_stepを実行すると、モデルのパラメータが更新されるのことを確認"""

        class LinearModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.linear = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, batch):
                return self.linear(batch["inputs"])

        class MSETask(Task):
            def compute_loss(self, outputs, batch, *, training=True, mask=None):
                loss = jnp.mean((outputs - batch["targets"]) ** 2)
                return loss

            def compute_metrics(self, outputs, batch, *, mask=None):
                return {"loss": self.compute_loss(outputs, batch)}

        rngs = nnx.Rngs(0)
        model = LinearModel(rngs)
        task = MSETask()
        optimizer = nnx.Optimizer(model, optax.sgd(learning_rate=0.1), wrt=nnx.Param)

        trainer = Trainer(model=model, task=task, optimizer=optimizer)

        # パラメータのコピーを取得
        initial_params = jax.tree_util.tree_map(
            lambda x: x.copy(), nnx.state(model, nnx.Param)
        )

        x = jnp.ones((1, 2))
        y = jnp.array([[10.0]])
        batch = {"inputs": x, "targets": y}

        metrics = trainer.train_step(batch)

        assert "loss" in metrics
        current_params = nnx.state(model, nnx.Param)

        params_changed = False
        flat_initial = jax.tree_util.tree_leaves(initial_params)
        flat_current = jax.tree_util.tree_leaves(current_params)

        for p_init, p_curr in zip(flat_initial, flat_current):
            if not jnp.allclose(p_init, p_curr):
                params_changed = True
                break

            assert params_changed, "Model parameters should be updated after train_step"


class TestTrainerFit:
    """Design Doc 3.3: NNX Trainer Loopのテスト"""

    def test_fit_reduce_loss(self):
        """
        fit メソッドを実行すると、エポックごとに学習が進み、
        最終的なLossが初期Lossよりも下がっていることを確認する。
        """

        class LinearModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.linear = nnx.Linear(
                    1, 1, kernel_init=nnx.initializers.zeros, rngs=rngs
                )

            def __call__(self, batch):
                return self.linear(batch["inputs"])

        class MSETask(Task):
            def compute_loss(self, outputs, batch, *, training=True, mask=None):
                return jnp.mean((outputs - batch["targets"]) ** 2)

            def compute_metrics(self, outputs, batch, *, mask=None):
                return {"loss": self.compute_loss(outputs, batch)}

        rngs = nnx.Rngs(0)
        model = LinearModel(rngs)
        task = MSETask()

        optimizer = nnx.Optimizer(model, optax.sgd(learning_rate=0.01), wrt=nnx.Param)
        trainer = Trainer(model, task, optimizer)

        X = jnp.arange(1, 5, dtype=jnp.float32).reshape((4, 1))
        Y = X * 2.0

        # 簡易的なデータローダー
        train_loader = [{"inputs": X, "targets": Y} for _ in range(5)]

        # 重みが0なので、予測は0。正解はYなのでLossは大きい
        initial_loss = task.compute_loss(model({"inputs": X}), {"targets": Y})
        history = trainer.fit(train_loader, num_epochs=5)

        final_loss = task.compute_loss(model({"inputs": X}), {"targets": Y})
        print(f"{initial_loss = }")
        print(f"{final_loss = }")
        assert final_loss < initial_loss
        assert final_loss < 0.1
        assert (
            isinstance(history, list)
            or isinstance(history, dict)
            or isinstance(history, nnx.MultiMetric)
        )
