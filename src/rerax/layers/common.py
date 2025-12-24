import chex
from flax import nnx


class Squeeze(nnx.Module):
    """指定された軸の次元を削除する(主に (Barch, 1, Hidden) -> (Batch, Hidden))"""

    def __init__(self, axis: int = 1):
        self._axis = axis

    def __call__(self, x: chex.Array) -> chex.Array:
        chex.assert_axis_dimension(x, self._axis, 1)
        return x.squeeze(axis=self._axis)
