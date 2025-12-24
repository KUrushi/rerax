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
            return self._compute_masked_mean(loss, mask)
        return loss.mean()

    def compute_metrics(
        self,
        outputs: chex.Array,
        batch: dict[str, chex.Array],
        *,
        mask: chex.Array | None = None,
    ):
        predictions = jnp.argmax(outputs, axis=2)
        correct = (batch["labels"] == predictions).astype(jnp.float32)

        if mask is not None:
            accuracy = self._compute_masked_mean(correct, mask)
        else:
            accuracy = correct.mean()

        return {"accuracy": accuracy}

    def _compute_masked_mean(self, value: chex.Array, mask: chex.Array) -> chex.Array:
        """Compute mean of value masked by mask."""
        return (value * mask).sum() / jnp.maximum(mask.sum(), 1)
