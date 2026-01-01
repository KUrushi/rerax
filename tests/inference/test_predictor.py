import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from rerax.inference.predictor import Predictor
from rerax.tasks.base import Task
from rerax.training.trainer import Trainer


class SimpleModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(2, 4, rngs=rngs)

    def __call__(self, x):
        return self.linear(x["input"])


class DummyTask(Task):
    def compute_loss(self, *args, **kwargs):
        return jnp.array(0.0)

    def compute_metrics(self, *arkgs, **kwargs):
        return {}


class InMemoryDataSource:
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self._data[idx]


class TestPredictor:
    def test__restore_and_predict(self, tmp_path):
        rngs = nnx.Rngs(0)
        model_train = SimpleModel(rngs)
        optimizer = nnx.Optimizer(model_train, optax.sgd(0.1), wrt=nnx.Param)
        task = DummyTask()

        trainer = Trainer(model_train, task, optimizer, checkpoint_dir=tmp_path)
        trainer.save_checkpoint(step=100)
        expected_params = nnx.state(model_train, nnx.Param)

        rngs_new = nnx.Rngs(1)
        model_infer = SimpleModel(rngs_new)
        predictor = Predictor.from_checkpoint(model_infer, tmp_path)

        current_params = nnx.state(model_infer, nnx.Param)
        chex.assert_trees_all_close(expected_params, current_params)

        inputs = [{"input": np.random.randn(2).astype(np.float32)} for _ in range(10)]
        data_source = InMemoryDataSource(inputs)

        import grain.python as grain

        sampler = grain.IndexSampler(len(inputs), shuffle=False, num_epochs=1, seed=0)
        loader = grain.DataLoader(
            data_source=data_source,
            sampler=sampler,
            operations=[grain.Batch(batch_size=3)],
            worker_count=0,
        )

        predictions = predictor.predict(loader)
        assert isinstance(predictions, (np.ndarray, jax.Array))
        if isinstance(predictions, jax.Array):
            # JAX Arrayの場合はCPUにあるか確認 (理想はnp.array)
            assert predictions.device.platform == "cpu"

        # 2. 件数が正しいか (10件)
        assert len(predictions) == 10

        # 3. 次元数が正しいか (Linearの出力は4次元)
        assert predictions.shape == (10, 4)
