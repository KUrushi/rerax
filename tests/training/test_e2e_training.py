import chex
import optax
import grain.python as grain
import pytest
from flax import nnx

from rerax.data.loader import build_dataloader
from rerax.data.transforms import TwoTowerPreprocessor
from rerax.layers.common import Squeeze
from rerax.models.two_tower import TwoTowerModel
from rerax.tasks.retrieval import RetrievalTask
from rerax.training.trainer import Trainer


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


@pytest.mark.e2e
class TestE2ETraining:

    def test_retrieval_training(
        self, e2e_retrieval_grain_datasource: grain.RandomAccessDataSource
    ):
        transformations = [TwoTowerPreprocessor("user_id", "item_id")]
        data_loader = build_dataloader(
            e2e_retrieval_grain_datasource,
            batch_size=2,
            transformations=transformations,
        )

        rngs = nnx.Rngs(0)
        query_tower = TowerModel(
            vocab_size=3,
            hidden_size=4,
            rngs=rngs,
        )
        candidate_tower = TowerModel(
             vocab_size=3,
            hidden_size=4,
            rngs=rngs,
        )
        tower_model = TwoTowerModel(query_tower, candidate_tower)
        tx = optax.adam(learning_rate=0.01)
        optimizer = nnx.Optimizer(
            tower_model,
            tx,
            wrt=nnx.Param
        )

        trainer = Trainer(
            tower_model,
            task=RetrievalTask(),
            optimizer=optimizer
        )
        trainer.fit(data_loader, total_steps=5)
