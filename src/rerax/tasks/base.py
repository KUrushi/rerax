from flax import nnx


class Task(nnx.Module):
    def compute_loss(self, outputs, batch, *, mask=None, training=True):
        raise NotImplementedError()

    def compute_metrics(self, outputs, batch):
        raise NotImplementedError()
