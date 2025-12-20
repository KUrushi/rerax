from typing import Any

import jax.numpy as jnp
from flax import nnx

from rerax.layers.sasrec import SASRecBlock, SASRecEmbedding


class SASRec(nnx.Module):
    def __init__(
        self,
        num_items: int,
        max_len: int,
        hidden_size: int,
        num_heads: int,
        num_blocks: int,
        dropout_rate: float,
        *,
        rngs: nnx.Rngs,
    ):
        self.embed = SASRecEmbedding(
            num_items, max_len, hidden_size, dropout_rate, rngs=rngs
        )
        self.sas_blocks = nnx.List(
            [
                SASRecBlock(hidden_size, num_heads, dropout_rate, rngs=rngs)
                for _ in range(num_blocks)
            ]
        )
        self.mask = create_causal_mask(max_len)
        self.norm = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.projection = nnx.Linear(hidden_size, num_items, rngs=rngs)

    def __call__(self, batch: dict[str, Any], training: bool) -> jnp.ndarray:
        x = self.embed(batch["inputs"], training=training)
        for block in self.sas_blocks:
            x = block(x, mask=self.mask[: x.shape[1], : x.shape[1]], training=training)

        x = self.norm(x)
        x = self.projection(x)
        return x


def create_causal_mask(seq_len: int):
    mask = jnp.ones((seq_len, seq_len), dtype=bool)

    mask = jnp.tril(mask)
    return mask
