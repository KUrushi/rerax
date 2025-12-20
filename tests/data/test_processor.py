import numpy as np
import pytest

from rerax.data.processor import process_sequence


class TestDataProcessor:
    """Data Pipeline Transformation logic"""

    def test_process_sequence_padding(self):
        """
        短いシーケンスの場合、paddingが行われるか
        max_len = 5, item_ids=[1, 2]
        Input: [1, 2, 0, 0, 0]
        Label: [2, 0, 0, 0, 0]
        Mask: [1, 0, 0, 0, 0]
        """
        item_ids = [1, 2]
        max_len = 5
        inputs, labels, mask = process_sequence(item_ids, max_len)

        # 期待値
        expected_inputs = np.array([1, 2, 0, 0, 0])
        expected_labels = np.array([2, 0, 0, 0, 0])
        expected_mask = np.array([1, 0, 0, 0, 0])

        np.testing.assert_array_equal(inputs, expected_inputs)
        np.testing.assert_array_equal(labels, expected_labels)
        np.testing.assert_array_equal(mask, expected_mask)

    def test_process_sequence_truncation(self):
        """
        長いシーケンスの場合、新しいもの（後ろ）を残して切り詰められるか。
        max_len=3, item_ids=[1, 2, 3, 4, 5]

        直近のデータを使いたいので、[3, 4, 5] が対象になるはず。
        Input: [3, 4, 5]
        Label: [4, 5, 0] (もし5が最後なら)

        ※ SASRecの学習では、一般的に「入力シーケンス」に対して「1つずらした正解」を作ります。
        Input: [3, 4] -> Predict: 4
        Input: [3, 4, 5] -> Predict: 5

        モデルの仕様に合わせて、
        Input: [3, 4, 5]
        Label: [4, 5, <EOS/PAD>]
        とするか、Input/Labelの長さを合わせる設計にします。
        ここでは「Inputの長さ == Labelの長さ == max_len」で実装します。
        """

        item_ids = [1, 2, 3, 4, 5]
        max_len = 3

        inputs, labels, mask = process_sequence(item_ids, max_len)
        expected_inputs = np.array([3, 4, 5])
        expected_labels = np.array([4, 5, 0])
        expected_mask = np.array([1, 1, 0])
        np.testing.assert_array_equal(inputs, expected_inputs)
        np.testing.assert_array_equal(labels, expected_labels)
        np.testing.assert_array_equal(mask, expected_mask)

    def test_process_sequence_no_padding(self):
        item_ids = [1, 2, 3]
        max_len = 3

        inputs, labels, mask = process_sequence(item_ids, max_len)
        expected_inputs = np.array([1, 2, 3])
        expected_labels = np.array([2, 3, 0])
        expected_mask = np.array([1, 1, 0])
        np.testing.assert_array_equal(inputs, expected_inputs)
        np.testing.assert_array_equal(labels, expected_labels)
        np.testing.assert_array_equal(mask, expected_mask)
