import jax.numpy as jnp
import optax
import jax
import chex
from flax import nnx

from rerax.tasks.retrieval import RetrievalTask


class TestRetrievalTask:
    def test_compute_in_batch_softmax(self):
        """
        RetrievalTaskがIn-batch Softmax Lossを正しく計算できるか検証する
        - クエリと候補の埋め込みを受け取る
        - 行列積でスコア(ロジット)を計算する
        - 対角成分を正解ラベルとしてCrossEntropyを計算する
        """
        batch_size = 4
        hidden_size = 8

        rngs = nnx.Rngs(0)
        query_embeddings = jnp.ones((batch_size, hidden_size))
        candidate_embeddings = jnp.ones((batch_size, hidden_size))

        outputs = {
            "query_embeddings": query_embeddings,
            "candidate_embeddings": candidate_embeddings
        }

        batch = {}
        task = RetrievalTask()

        loss = task.compute_loss(outputs, batch)
        logits = jnp.matmul(query_embeddings, candidate_embeddings.T)
        expected_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=jnp.arange(batch_size)
        ).mean()

        chex.assert_trees_all_close(loss, expected_loss)
        


    def test_compute_metrics(self):
        """
        RetrievalTaskがメトリクス (Recall@K)を正しく計算できるか検証
        """
        batch_size = 10
        hidden_size = 4
        key = jax.random.key(0)
        query_key, key = jax.random.split(key)

        query_embeddings = jax.random.normal(query_key, (batch_size, hidden_size)) 
        candidate_embeddings = query_embeddings

        outputs = {
            "query_embeddings": query_embeddings,
            "candidate_embeddings": candidate_embeddings
        }

        batch = {}

        task = RetrievalTask()
        metrics = task.compute_metrics(outputs, batch)

        assert "recall@1" in metrics
        assert metrics["recall@1"] == 1.0

        assert "recall@5" in metrics
        assert metrics["recall@5"] == 1.0
        

