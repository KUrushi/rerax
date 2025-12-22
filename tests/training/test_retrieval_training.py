import jax.numpy as jnp
import optax
import jax
import chex

from flax import nnx

from rerax.tasks.retrieval import RetrievalTask
from rerax.layers.common import Squeeze
from rerax.models.two_tower import TwoTowerModel
from rerax.training.trainer import Trainer

class TestTwoTower:
    def test_two_tower_training(self):
        class QueryTower(nnx.Module):
            def __init__(self, vocab_size: int, hidden_size, *, rngs: nnx.Rngs):
                self._embedding = nnx.Embed(vocab_size, hidden_size, rngs=rngs)
                self._linear =  nnx.Linear(hidden_size, hidden_size, rngs=rngs)
                self._squeeze = Squeeze(axis=1)
            
            def __call__(self, x:chex.Array):
                x = self._embedding(x)
                x = self._linear(x)
                x = self._squeeze(x)
                return x

        class CandidateTower(nnx.Module):
            def __init__(self, vocab_size: int, hidden_size, *, rngs: nnx.Rngs):
                self._embedding = nnx.Embed(vocab_size, hidden_size, rngs=rngs)
                self._linear =  nnx.Linear(hidden_size, hidden_size, rngs=rngs)
                self._squeeze = Squeeze(axis=1)

            
            def __call__(self, x:chex.Array):
                x = self._embedding(x)
                x = self._linear(x)
                x = self._squeeze(x)
                return x

            
        vocab_size = 10
        hidden_size = 5
        batch_size = 32
        key = jax.random.key(0)

        # データの準備
        ids_key, key = jax.random.split(key)
        batch_data = {
            "query_ids": jax.random.randint(ids_key, shape=(batch_size,1), minval=0, maxval=vocab_size),
            "candidate_ids": jax.random.randint(ids_key, shape=(batch_size,1), minval=0, maxval=vocab_size)
        }


        # モデルの準備
        model_key, key = jax.random.split(key)
        rngs = nnx.Rngs(model_key)
        query_tower = QueryTower(vocab_size, hidden_size, rngs=rngs)
        candidate_tower = CandidateTower(vocab_size, hidden_size, rngs=rngs)
        model = TwoTowerModel(query_tower, candidate_tower)

    
        optimizer = nnx.Optimizer(
            model,
            optax.adam(learning_rate=0.001),
            wrt=nnx.Param
        )
        

        # タスクの準備
        task = RetrievalTask()
        trainer = Trainer(model, task, optimizer, None) 

        metrics = trainer.train_step(batch=batch_data)
        assert "loss" in metrics
        assert "recall@1" in metrics

        assert not jnp.isnan(metrics["loss"])
        print(f"Training Step Metrics: {metrics}")

