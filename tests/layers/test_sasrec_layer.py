import chex
import jax.numpy as jnp
import pytest
from flax import nnx

from rerax.layers.sasrec import SASRecBlock
from rerax.models.sasrec import create_causal_mask


class TestSASRecBlock:
    @pytest.fixture
    def block(self):
        return SASRecBlock(
            hidden_size=32, num_heads=4, dropout_rate=0.1, rngs=nnx.Rngs(0)
        )

    def test_output_shape(self, block):
        """
        Blockを通しても形状が変わらないこと (ResidualConnectionのため必須)
        Input: [Batch, SeqLen, Hidden] -> Output: [Batch, SeqLen, Hidden]
        """
        batch_size, seq_len, hidden_size = 2, 10, 32
        x = jnp.ones((batch_size, seq_len, hidden_size))
        mask = create_causal_mask(seq_len)
        out = block(x, mask=mask, training=True)
        chex.assert_shape(out, (batch_size, seq_len, hidden_size))

    def test_block_causality(self, block):
        """
        ブロック単体でも「未来の干渉」がないことを確認
        """
        seq_len = 5
        hidden_size = 32
        mask = create_causal_mask(seq_len)

        x1 = jnp.ones((1, seq_len, hidden_size), dtype=jnp.float32)
        x2 = x1.at[:, -1, :].set(999.0)

        out1 = block(x1, mask=mask, training=False)
        out2 = block(x2, mask=mask, training=False)

        chex.assert_trees_all_close(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5)
        with pytest.raises(AssertionError):
            chex.assert_trees_all_close(out1[:, -1, :], out2[:, -1, :], atol=1e-5)
