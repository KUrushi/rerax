import jax.numpy as jnp
import optax
from rerax.tasks.base import Task

class RetrievalTask(Task):
    def compute_loss(self, outputs, batch, *, mask=None, training=True):

        logits = outputs['query_embeddings'] @ outputs['candidate_embeddings'].T

        return optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=jnp.arange(logits.shape[0])
        ).mean()


    def compute_metrics(self, outputs, batch, *, mask=None):
        return super().compute_metrics(outputs, batch, mask=mask)
