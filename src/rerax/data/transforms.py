import grain.python as grain
import numpy as np

from rerax.data.processor import process_sequence


class ProcessSequence(grain.MapTransform):
    def __init__(self, max_len: int, id_keys: str = "input_ids") -> None:
        self._max_len = max_len
        self._id_keys = id_keys

    def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        input_ids = element[self._id_keys]
        inputs, labels, mask = process_sequence(input_ids, max_len=self._max_len)

        return {"inputs": inputs, "labels": labels, "mask": mask}


class TwoTowerPreprocessor(grain.MapTransform):
    def __init__(
        self,
        query_key: str,
        candidate_key: str,
        *,
        target_query_key_name: str = "query_ids",
        target_candidate_key_name="candidate_ids",
    ):
        self._query_key = query_key
        self._candidate_key = candidate_key
        self._target_query_key_name = target_query_key_name
        self._target_candidate_key_name = target_candidate_key_name

    def map(self, element: dict[str, int | np.ndarray]) -> dict[str, int | np.ndarray]:
        output = element.copy()
        output.pop(self._query_key)
        output.pop(self._candidate_key)

        for key, target_key in zip(
            (self._query_key, self._candidate_key),
            (self._target_query_key_name, self._target_candidate_key_name),
        ):
            if not isinstance(element[key], list | np.ndarray):
                output[target_key] = np.array([element[key]])
            else:
                output[target_key] = element[key]

        return output
