import chex
import jax.numpy as jnp
import optax

from rerax.tasks.base import Task


class GenerativeTask(Task):
    def compute_loss(
        self,
        outputs: chex.Array,
        batch: dict[str, chex.Array],
        *,
        mask: chex.Array | None = None,
        training: bool = True,
    ):
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=outputs, labels=batch["labels"]
        )
        if mask is not None:
            return (loss * mask).sum() / jnp.maximum(mask.sum(), 1)
        return loss.mean()

    def compute_metrics(self, outputs: chex.Array, batch: dict[str, chex.Array], *, mask: chex.Array | None = None):
        predictions = jnp.argmax(outputs, axis=2)
        correct = (batch['labels'] == predictions).astype(jnp.float32)

        if mask is not None:
            accuracy = (correct * mask).sum() / jnp.maximum(mask.sum(), 1)
        else:
            accuracy = correct.mean()

        return {"accuracy": accuracy}
