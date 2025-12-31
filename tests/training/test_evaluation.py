import chex
import grain.python as grain
import optax
import pytest
from flax import nnx

from rerax.data.loader import build_dataloader
from rerax.data.transforms import TwoTowerPreprocessor
from rerax.layers.common import Squeeze
from rerax.models.two_tower import TwoTowerModel
from rerax.tasks.retrieval import RetrievalTask
from rerax.training.trainer import Trainer


class InMemoryDataSource(grain.RandomAccessDataSource):
    """テスト用にオンメモリのデータをGrainのDataSourceとして扱うためのラッパークラス。"""

    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


@pytest.fixture
def eval_datasource():
    test_data = [
        {"user_id": 1, "item_id": 1},
        {"user_id": 1, "item_id": 2},
        {"user_id": 1, "item_id": 3},
        {"user_id": 2, "item_id": 1},
        {"user_id": 2, "item_id": 2},
        {"user_id": 3, "item_id": 1},
    ]

    ds = InMemoryDataSource(test_data)
    return ds


class TestEvaluation:
    def test_evaluation(self, eval_datasource):
        class TowerModel(nnx.Module):
            def __init__(self, vocab_size: int, hidden_size, *, rngs: nnx.Rngs):
                self._embedding = nnx.Embed(vocab_size, hidden_size, rngs=rngs)
                self._linear = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
                self._squeeze = Squeeze(axis=1)

            def __call__(self, x: chex.Array):
                x = self._embedding(x)
                x = self._linear(x)
                x = self._squeeze(x)
                return x

        transformations = [TwoTowerPreprocessor("user_id", "item_id")]
        data_loader = build_dataloader(
            eval_datasource, batch_size=2, transformations=transformations, num_epochs=1
        )

        rngs = nnx.Rngs(0)

        query_tower = TowerModel(vocab_size=4, hidden_size=4, rngs=rngs)
        candidate_tower = TowerModel(vocab_size=4, hidden_size=4, rngs=rngs)
        tower_model = TwoTowerModel(query_tower, candidate_tower)
        tx = optax.adam(learning_rate=0.01)
        optimizer = nnx.Optimizer(tower_model, tx, wrt=nnx.Param)
        trainer = Trainer(tower_model, task=RetrievalTask(), optimizer=optimizer)
        metrics = trainer.evaluate(data_loader, current_step=0)

        assert "loss" in metrics
        assert "recall@1" in metrics
        assert "recall@5" in metrics
