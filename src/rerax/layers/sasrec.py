import chex
from flax import nnx


class SASRecEmbedding(nnx.Module):
    def __init__(
        self,
        num_items: int,

        max_len: int,
        hidden_size: int,
        dropout_rate: float,
        *,
        rngs: nnx.Rngs,
    ):
        self.embed = nnx.Embed(num_of_items, hidden_size, rngs=rngs)
        self.pos_embed = nnx.Param(rngs.normal((max_len, hidden_size)))
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, input_ids: chex.Array, training: bool = True) -> chex.Array:
        seq_len = input_ids.shape[1]
        x = self.embed(input_ids)
        pos = self.pos_embed[:seq_len, :]
        x = x + pos
        x = self.dropout(x, deterministic=not training)
        return x


class SASRecBlock(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float,
        *,
        rngs: nnx.Rngs,
    ):
        self.ln1 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_size,
            qkv_features=hidden_size,
            decode=False,
            rngs=rngs,
        )
        self.dropout1 = nnx.Dropout(dropout_rate, rngs=rngs)

        self.ln2 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.ffn_intermediate = nnx.Linear(hidden_size, hidden_size * 4, rngs=rngs)
        self.ffn_output = nnx.Linear(hidden_size * 4, hidden_size, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(
        self, x: chex.Array, mask: chex.Array, training: bool = True
    ) -> chex.Array:
        # 1. Self-Attention Block
        h = self.ln1(x)
        h = self.attention(h, inputs_k=h, inputs_v=h, mask=mask)
        h = self.dropout1(h, deterministic=not training)
        x = x + h

        # 2. Feed Forward Block
        h = self.ln2(x)
        h = self.ffn_intermediate(h)
        h = nnx.gelu(h)
        h = self.dropout2(h, deterministic=not training)
        h = self.ffn_output(h)
        x = x + h
        return x
