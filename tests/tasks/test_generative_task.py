import jax
import jax.numpy as jnp
import optax
import pytest

from rerax.tasks.generative import GenerativeTask


class TestGenerativeTask:
    """Generative Task実装テスト"""

    @pytest.fixture
    def task(self):
        return GenerativeTask()

    def test_compute_loss_value(self, task):
        """
        実際の計算ロジックがあっているかを検証。
        Optaxの計算結果と一致することを確認する
        """
        batch_size, seq_len, vocab_size = 2, 4, 10
        key = jax.random.key(0)
        logits = jax.random.normal(key, (batch_size, seq_len, vocab_size))
        labels = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)

        batch = {"labels": labels}
        loss_actual = task.compute_loss(logits, batch, training=True)

        loss_expected = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=labels
        ).mean()

        assert jnp.allclose(loss_actual, loss_expected, atol=1e-5)

    def test_compute_metrics_accuracy(self, task):
        """
        compute_metricsが正しく正解率(accuracy)を計算できるか検証
        """
        batch_size, seq_len, vocab_size = 2, 3, 4
        outputs = jnp.array(
            [
                # Batch 0
                [
                    [10.0, 0.0, 0.0, 0.0],  # 予測: 0
                    [0.0, 10.0, 0.0, 0.0],  # 予測: 1
                    [0.0, 0.0, 10.0, 0.0],  # 予測: 2
                ],
                # Batch 1
                [
                    [0.0, 10.0, 0.0, 0.0],  # 予測: 1
                    [0.0, 0.0, 10.0, 0.0],  # 予測: 2
                    [0.0, 0.0, 0.0, 10.0],  # 予測: 3
                ],
            ]
        )

        batch = {"labels": jnp.array([[0, 2, 2], [0, 0, 0]])}
        metrics = task.compute_metrics(outputs, batch)
        assert "accuracy" in metrics
        assert jnp.allclose(metrics["accuracy"], 1.0 / 3.0)
