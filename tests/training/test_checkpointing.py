import shutil
import tempfile

import chex
import jax
import optax
import pytest
from flax import nnx

from rerax.layers.common import Squeeze
from rerax.models.two_tower import TwoTowerModel
from rerax.tasks.retrieval import RetrievalTask
from rerax.training.trainer import Trainer


class TestCheckpointing:
    @pytest.fixture
    def checkpoint_dir(self):
        dirpath = tempfile.mkdtemp()
        yield dirpath
        shutil.rmtree(dirpath)

    def test_save_and_restore(self, checkpoint_dir):
        class SimpleTower(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                # 明示的に params コレクションの乱数を使う
                self.linear = nnx.Linear(2, 2, rngs=rngs)
                self.squeeze = Squeeze(axis=1)

            def __call__(self, x):
                return self.squeeze(self.linear(x))

        # 乱数シードを固定
        rngs = nnx.Rngs(params=0)
        model = TwoTowerModel(SimpleTower(rngs), SimpleTower(rngs))
        optimizer = nnx.Optimizer(model, optax.sgd(0.1), wrt=nnx.Param)
        task = RetrievalTask()

        trainer = Trainer(model, task, optimizer, checkpoint_dir=checkpoint_dir)

        # 保存前の状態を取得
        # nnx.stateは参照を返すので、state_originalを元にしたもの変更されると、state_originalそのものも変更されてします。
        # なので、コピーを作成する
        state_original = jax.tree_util.tree_map(
            lambda x: x, nnx.state(model, nnx.Param)
        )

        assert len(jax.tree_util.tree_leaves(state_original)) > 0, (
            "Model params should not be empty"
        )

        # チェックポイント保存
        trainer.save_checkpoint(step=100)

        state_changed = jax.tree_util.tree_map(lambda x: x + 1.0, state_original)
        nnx.update(model, state_changed)
        current_state = nnx.state(model, nnx.Param)

        # 変更されたことを確認
        with pytest.raises(AssertionError):
            chex.assert_trees_all_close(
                state_original.to_pure_dict(),
                current_state.to_pure_dict(),
                rtol=0,
                atol=1e-12,
            )

        # 復元
        trainer.restore_checkpoint(step=100)

        # 検証: 元の状態に戻っているか
        state_restored = nnx.state(model, nnx.Param)
        chex.assert_trees_all_close(state_original, state_restored)
