from typing import Any

import jax.numpy as jnp
import optax
from flax import nnx

from rerax.tasks.base import Task
from rerax.tracking.base import BaseTracker
from rerax.training.trainer import Trainer


class TestTrackerIntegration:
    """Design Doc 4.3: Experiment Tracking Abstraction のテスト"""

    def test_trainer_logs_metrics_to_tracker(self):
        """
        TrainerにTrackerを渡した場合、fit中の各ステップ(またはエポック)で
        metricsがTrackerに送信されることを確認。
        """

        # --- 1. Mock Tracker の作成 (BaseTrackerを継承) ---
        class MockTracker(BaseTracker):
            def __init__(self):
                self.logs = []

            def log_metrics(self, metrics: dict[str, float], step: int):
                self.logs.append({"step": step, "metrics": metrics})

            def log_params(self, params: dict[str, Any]):
                pass

        # --- 2. 最小構成のセットアップ ---
        class DummyModel(nnx.Module):
            def __init__(self, rngs):
                self.layer = nnx.Linear(1, 1, rngs=rngs)

            def __call__(self, batch):
                return self.layer(batch["x"])

        class DummyTask(Task):
            def compute_loss(self, outputs, batch, *, training=True, mask=None):
                return jnp.array(0.5)  # 定数Loss

            def compute_metrics(self, outputs, batch, *, mask=None):
                return {"loss": self.compute_loss(outputs, batch)}

        model = DummyModel(nnx.Rngs(0))
        task = DummyTask()
        optimizer = nnx.Optimizer(model, optax.sgd(0.1), wrt=nnx.Param)

        tracker = MockTracker()

        # Trainerにtrackerを注入
        # (Trainerの__init__シグネチャ変更が必要になるはず)
        trainer = Trainer(model, task, optimizer, tracker=tracker)

        # --- 3. 実行 ---
        # データローダー (1エポックあたり2バッチ)
        data = [{"x": jnp.ones((1, 1))} for _ in range(2)]
        

        # 3エポック回す
        trainer.fit(data, 3, log_freq=1)

        # --- 4. 検証 ---
        # 3エポック分のログが残っているはず
        assert len(tracker.logs) == 3

        # 最後のログを確認
        last_log = tracker.logs[-1]
        assert "step" in last_log
        assert "loss" in last_log["metrics"]
        # エポック番号(0, 1, 2)がステップとして記録されているか
        assert last_log["step"] == 3
