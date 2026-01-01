import chex
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
from grain.python import DataLoader


class Predictor(nnx.Module):
    def __init__(self, model: nnx.Module):
        self._model = model

    @classmethod
    def from_checkpoint(
        cls, model: nnx.Module, checkpoint_dir: str, step: int | None = None
    ) -> "Predictor":
        with ocp.CheckpointManager(checkpoint_dir) as mngr:
            if step is None:
                step = mngr.latest_step()
                if step is None:
                    raise ValueError(f"No Checkpoints found in {checkpoint_dir}")

            _, params_state = nnx.split(model, nnx.Param)
            restore_args = ocp.args.Composite(
                params=ocp.args.StandardRestore(params_state),
            )

            restored = mngr.restore(step, args=restore_args)

            nnx.update(model, restored["params"])
            return cls(model)

    @nnx.jit
    def predict_step(
        self, batch: dict[str, chex.Array]
    ) -> chex.Array | dict[str, chex.Array]:
        return self._model(batch)

    def predict(self, data_loader: DataLoader) -> chex.Array | dict[str, chex.Array]:
        results = []
        for batch in data_loader:
            step_output = self.predict_step(batch)
            step_output_cpu = jax.device_get(step_output)
            results.append(step_output_cpu)

        first_elem = results[0]
        if isinstance(first_elem, dict):
            concatenated = {}
            for k in first_elem.keys():
                concatenated[k] = jnp.concatenate([r[k] for r in results], axis=0)
            return concatenated
        else:
            return jnp.concatenate(results, axis=0)
