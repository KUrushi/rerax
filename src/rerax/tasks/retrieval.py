import jax.numpy as jnp
import optax

from rerax.tasks.base import Task


class RetrievalTask(Task):
    def __init__(self, top_k_for_metrics: tuple[int, ...] = (1, 5)):
        self.top_k_for_metrics = top_k_for_metrics

    def compute_loss(self, outputs, batch, *, mask=None, training=True):
        logits = self._compute_logits(
            outputs["query_embeddings"], outputs["candidate_embeddings"]
        )

        return optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=jnp.arange(logits.shape[0])
        ).mean()

    def compute_metrics(self, outputs, batch, *, mask=None):
        logits = self._compute_logits(
            outputs["query_embeddings"], outputs["candidate_embeddings"]
        )
        prediction = jnp.argsort(-logits, axis=-1)
        targets = jnp.arange(logits.shape[0])[:, None]

        metrics = {}
        for k in self.top_k_for_metrics:
            correct = prediction[:, :k] == targets
            metrics[f"recall@{k}"] = correct.any(axis=-1).sum() / targets.shape[0]

        return metrics

    def _normalize(self, embeddings):
        return embeddings / jnp.linalg.norm(embeddings, axis=-1, keepdims=True)

    def _compute_logits(self, query_embs, candidate_embs):
        query_embs = self._normalize(query_embs)
        candidate_embs = self._normalize(candidate_embs)
        return query_embs @ candidate_embs.T
