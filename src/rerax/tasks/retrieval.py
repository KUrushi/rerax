import jax.numpy as jnp
import optax
from rerax.tasks.base import Task

class RetrievalTask(Task):
    def compute_loss(self, outputs, batch, *, mask=None, training=True):
        query_embeddings = self._normalize(outputs['query_embeddings'])
        candidate_embeddings = self._normalize(outputs['candidate_embeddings'])
        logits = query_embeddings @ candidate_embeddings.T

        return optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=jnp.arange(logits.shape[0])
        ).mean()


    def compute_metrics(self, outputs, batch, *, mask=None):
        query_embeddings = self._normalize(outputs['query_embeddings'])
        candidate_embeddings = self._normalize(outputs['candidate_embeddings'])
        logits = query_embeddings @ candidate_embeddings.T

        prediction = jnp.argsort(-logits, axis=-1)
        targets = jnp.arange(logits.shape[0])[:, None]

        metrics = {}
        for k in (1, 5):
            correct = prediction[:,:k] == targets
            metrics[f"recall@{k}"] = correct.sum() / targets.shape[0]

        return metrics

    def _normalize(self, embeddings):
        return embeddings / jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
