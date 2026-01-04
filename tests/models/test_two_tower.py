import chex
import jax
from flax import nnx

from rerax.layers.common import Squeeze


class TestTwoTowerModel:
    def test_two_tower_output_shape(self):
        class CustomTwoTowerModel(nnx.Module):
            def __init__(self, query_tower: nnx.Module, candidate_tower: nnx.Module):
                self.query_tower = query_tower
                self.candidate_tower = candidate_tower

            def __call__(self, batch: dict[str, chex.Array]) -> dict[str, chex.Array]:
                query_ids = batch["query_ids"]
                candidate_ids = batch["candidate_ids"]

                query_embeddings = self.query_tower(query_ids)
                candidate_embeddings = self.candidate_tower(candidate_ids)
                return {
                    "query_embeddings": query_embeddings,
                    "candidate_embeddings": candidate_embeddings,
                }

        class QueryTower(nnx.Module):
            def __init__(self, vocab_size: int, hidden_size, *, rngs: nnx.Rngs):
                self._embedding = nnx.Embed(vocab_size, hidden_size, rngs=rngs)
                self._linear = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
                self._squeeze = Squeeze(axis=1)

            def __call__(self, x: chex.Array):
                x = self._embedding(x)
                x = self._linear(x)
                x = self._squeeze(x)
                return x

        class CandidateTower(nnx.Module):
            def __init__(self, vocab_size: int, hidden_size, *, rngs: nnx.Rngs):
                self._embedding = nnx.Embed(vocab_size, hidden_size, rngs=rngs)
                self._linear = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
                self._squeeze = Squeeze(axis=1)

            def __call__(self, x: chex.Array):
                x = self._embedding(x)
                x = self._linear(x)
                x = self._squeeze(x)
                return x

        vocab_size = 10
        hidden_size = 5
        batch_size = 10
        key = jax.random.key(0)
        model_key, key = jax.random.split(key)
        rngs = nnx.Rngs(model_key)
        query_tower = QueryTower(vocab_size, hidden_size, rngs=rngs)
        candidate_tower = CandidateTower(vocab_size, hidden_size, rngs=rngs)
        model = CustomTwoTowerModel(query_tower, candidate_tower)

        ids_key, key = jax.random.split(key)
        batch_data = {
            "query_ids": jax.random.randint(
                ids_key, shape=(batch_size, 1), minval=0, maxval=vocab_size
            ),
            "candidate_ids": jax.random.randint(
                ids_key, shape=(batch_size, 1), minval=0, maxval=vocab_size
            ),
        }

        output = model(batch_data)
        expected_shape = (batch_size, hidden_size)
        chex.assert_shape(output["candidate_embeddings"], expected_shape)
        chex.assert_shape(output["query_embeddings"], expected_shape)

        chex.assert_equal_shape(
            [output["candidate_embeddings"], output["query_embeddings"]]
        )
