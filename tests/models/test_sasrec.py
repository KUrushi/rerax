import fiddle as fdl
import jax.numpy as jnp
import pytest
from flax import nnx

from rerax.models.sasrec import SASRec


class TestSASRec:
    """Design Doc 1.2 Modern JAX Native SASRec Implementation with fiddle"""

    @pytest.fixture
    def base_config(self):
        """
        FiddleのConfigオブジェクトを返すFixture
        """
        return fdl.Config(
            SASRec,
            num_items=100,
            max_len=20,
            hidden_size=32,
            num_heads=4,
            num_blocks=2,
            dropout_rate=0.1,
            rngs=nnx.Rngs(0)
        )

    def test_fiddle_build_and_forward_shape(self, base_config):
        model = fdl.build(base_config)

        batch_size = 2
        inputs = jnp.zeros((batch_size, base_config.max_len), dtype=jnp.int32)
        batch = {"inputs": inputs}

        logits = model(batch, training=True)
        expected_shape = (batch_size, base_config.max_len, base_config.num_items)
        assert logits.shape == expected_shape

    def test_forward_inference_model(self, base_config):
        """推論モードの挙動の確認"""
        model = fdl.build(base_config)
        inputs = jnp.zeros((2, base_config.max_len), dtype=jnp.int32)
        batch = {"inputs": inputs}

        logits1 = model(batch, training=False)
        logits2 = model(batch, training=False)
        assert jnp.array_equal(logits1, logits2)


    def test_create_causal_mask(self):
        """
        create_causal_mask関数が正しい形状のマスクを生成するかを確認
        長さ L=3の場合
        [[True, False, False],
         [True, True, False],
         [True, True, True]]
        """
        from rerax.models.sasrec import create_causal_mask

        seq_len = 3
        mask = create_causal_mask(seq_len)
        expected = jnp.array(
            [[True, False, False], [True, True, False], [True, True, True]], jnp.bool
        )
        assert mask.shape == (seq_len, seq_len)
        assert jnp.array_equal(mask, expected)
