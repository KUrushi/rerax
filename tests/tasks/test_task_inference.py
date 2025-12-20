import pytest
import jax.numpy as jnp
from flax import nnx

from rerax.tasks.base import Task


class TestTaskInterface:
    """Design Doc 3.1: Task Abstraction Design の要件を満たすかテスト"""

    def test_task_is_nnx_module(self):
        """Taskは nnx.Module を継承している必要がある"""

        # 具体的な実装を持たない抽象クラスとして扱うため、
        # テスト用のダミーサブクラスを作成して検証します
        class DummyTask(Task):
            def compute_loss(self, outputs, batch, *, training=True):
                return jnp.array(0.0)

            def compute_metrics(self, outputs, batch):
                return {}

        task = DummyTask()
        assert isinstance(task, nnx.Module), "Task must inherit from nnx.Module"

    def test_task_interface_methods(self):
        """
        Taskクラスは compute_loss と compute_metrics を実装する必要がある。
        ベースクラスの時点では NotImplementedError を投げるべき。
        """
        # 抽象クラス自体をインスタンス化（またはメソッド呼び出し）して
        # 未実装メソッドがエラーを吐くか確認

        # 注意: nnx.Module の初期化仕様によってはインスタンス化の方法が異なる場合がありますが、
        # ここでは標準的な継承の動作を確認します。

        class IncompleteTask(Task):
            pass

        # nnx.Module は通常 __init__ が必要ですが、ここではメソッドの存在確認のみ
        task = IncompleteTask()

        # ダミーデータ
        outputs = jnp.array([1.0])
        batch = {"label": jnp.array([1.0])}

        # compute_loss のインターフェース確認
        with pytest.raises(NotImplementedError):
            task.compute_loss(outputs, batch, training=True)

        # compute_metrics のインターフェース確認
        with pytest.raises(NotImplementedError):
            task.compute_metrics(outputs, batch)
