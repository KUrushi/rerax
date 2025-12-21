import grain.python as grain
import numpy as np
from rerax.data.processor import process_sequence


class ProcessSequence(grain.MapTransform):
    def __init__(self, max_len: int, id_keys: str = "input_ids") -> None:
        self._max_len = max_len
        self._id_keys = id_keys

    def map(self, element) -> dict[str, np.ndarray]:
        input_ids = element[self._id_keys]
        inputs, labels, mask = process_sequence(input_ids, max_len=self._max_len)

        return {"inputs": inputs, "labels": labels, "mask": mask}
