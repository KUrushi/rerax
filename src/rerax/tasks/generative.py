import chex
import jax.numpy as jnp
import optax

from rerax.tasks.base import Task


class GenerativeTask(Task):
    def compute_loss(
        self, outputs: chex.Array, batch: dict[str, chex.Array], *, training=True
    ):
        return optax.softmax_cross_entropy_with_integer_labels(
            logits=outputs, labels=batch["labels"]
        ).mean()

    def compute_metrics(self, outputs: chex.Array, batch: dict[str, chex.Array]):
        predictions = jnp.argmax(outputs, axis=2)

        return {"accuracy": (batch["labels"] == predictions).astype(jnp.float32).mean()}


def build_dataloader(
    data_source, batch_size, seed=0, *, drop_remainder=True, shuffle=False
):
    pass
