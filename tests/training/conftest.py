import chex
import grain.python as grain
import pytest
from flax import nnx

from rerax.layers.common import Squeeze

# def pytest_addoption(parser):
#     """コマンドラインオプション --run-e2eを追加"""
#     parser.addoption(
#         "--run-e2e", action="store_true", default=False, help="run slow e2e tests"
#     )


# def pytest_collection_modifyitems(config, items):
#     """
#     --run-e2e が指定されていない場合、@pytest.mark.e2e がついたテストをスキップする
#     """
#     if config.getoption("--run-e2e"):
#         # --run-e2e が指定されている場合はスキップしない（通常通り実行）
#         return

#     skip_e2e = pytest.mark.skip(reason="need --run-e2e option to run")

#     for item in items:
#         if "e2e" in item.keywords:
#             item.add_marker(skip_e2e)


class InMemoryDataSource(grain.RandomAccessDataSource):
    """テスト用にオンメモリのデータをGrainのDataSourceとして扱うためのラッパークラス。"""

    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


@pytest.fixture(scope="session")
def e2e_retrieval_grain_datasource():
    """
    E2Eテスト用のGrain DataSourceインスタンスを返すFixure.
    scope = "session"にすることで、テスト実行全体で1回だけ生成され、再利用される
    """

    test_data = [
        {"user_id": 1, "item_id": 1},
        {"user_id": 1, "item_id": 2},
        {"user_id": 1, "item_id": 3},
        {"user_id": 2, "item_id": 1},
        {"user_id": 2, "item_id": 2},
        {"user_id": 3, "item_id": 1},
    ]

    ds = InMemoryDataSource(test_data)
    return ds
