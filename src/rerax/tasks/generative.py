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
    ):
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=outputs, labels=batch["labels"]
        )
        if mask:
            loss = loss * mask
        return loss.sum() / jnp.min(mask.sum(), 1)

    def compute_metrics(self, outputs: chex.Array, batch: dict[str, chex.Array]):
        predictions = jnp.argmax(outputs, axis=2)

        return {"accuracy": (batch["labels"] == predictions).astype(jnp.float32).mean()}
