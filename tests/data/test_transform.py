import chex
import numpy as np

from rerax.data.transforms import TwoTowerPreprocessor


class TestTransform:
    def test_dimension_expansion(self):
        expected_data = {"query_ids": np.array([123]), "candidate_ids": np.array([456])}
        data = {"user_id": 123, "item_id": 456}
        processor = TwoTowerPreprocessor("user_id", "item_id")
        actual = processor.map(data)
        chex.assert_trees_all_equal(expected_data, actual)

    def test_dimension_expansion_with_extra_data(self):
        expected_data = {
            "query_ids": np.array([123]),
            "candidate_ids": np.array([456]),
            "timestamp": 12345,
        }
        data = {"user_id": 123, "item_id": 456, "timestamp": 12345}
        processor = TwoTowerPreprocessor("user_id", "item_id")
        actual = processor.map(data)
        chex.assert_trees_all_equal(expected_data, actual)

    def test_rank1_input(self):
        expected_data = {"query_ids": np.array([123]), "candidate_ids": np.array([456])}
        data = {"user_id": np.array([123]), "item_id": np.array([456])}
        processor = TwoTowerPreprocessor("user_id", "item_id")
        actual = processor.map(data)
        chex.assert_trees_all_equal(expected_data, actual)

    def test_arbitary_key_name(self):
        expected_data = {"user_id": np.array([123]), "item_id": np.array([456])}
        data = {"query_ids": np.array([123]), "candidate_ids": np.array([456])}

        processor = TwoTowerPreprocessor(
            "query_ids",
            "candidate_ids",
            target_query_key_name="user_id",
            target_candidate_key_name="item_id",
        )
        actual = processor.map(data)
        chex.assert_trees_all_equal(expected_data, actual)
