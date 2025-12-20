import grain
import numpy
import numpy as np
import pytest

from rerax.data.loader import build_dataloader


class TestDataLoader:
    """Data Pipeline with Grainの基本機能テスト"""

    def test_build_dataloader_basic_batching(self):
        """
        単純な整数のリストをソースとして、
        指定したバッチサイズでデータが供給されるかを確認。
        """
        data_source = list(range(100))
        batch_size = 10

        loader = build_dataloader(
            data_source=data_source, batch_size=batch_size, seed=0, drop_remainder=True
        )

        batch = next(iter(loader))
        assert isinstance(batch, np.ndarray) or isinstance(batch, list)
        assert batch[0] in data_source

    def test_dataloader_structure_with_dict(self):
        """
        辞書形式のデータを扱えるか
        {'item_id': ..., 'user_id' ...}
        """
        data_source = [{"item_id": i, "user_id": i * 2} for i in range(20)]
        batch_size = 5

        loader = build_dataloader(
            data_source=data_source, batch_size=batch_size, seed=0
        )

        batch = next(iter(loader))

        assert isinstance(batch, dict)
        assert "item_id" in batch
        assert "user_id" in batch
        assert len(batch["item_id"]) == batch_size
