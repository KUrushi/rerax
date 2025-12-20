from unittest.mock import MagicMock, patch

import pytest

from rerax.tracking.mlflow import MLflowTracker


class TestMLflowTracker:
    """Design Doc 4.3: MLflow Integraionのテスト"""

    @pytest.fixture
    def mock_mlflow(self):
        with patch("rerax.tracking.mlflow.mlflow") as mock:
            yield mock

    def test_log_metrics_calls_mlflow(self, mock_mlflow):
        """log_metricsが呼ばれると、`mlflow.log_metrics`が実行される"""
        tracker = MLflowTracker(experiment_name="test_exp")
        metrics = {"loss": 0.5, "accuracy": 0.9}
        step = 10
        tracker.log_metrics(metrics, step=step)
        mock_mlflow.log_metrics.assert_called_once_with(metrics, step=step)

    def test_log_params_calls_mlflow(self, mock_mlflow):
        """log_paramsが呼ばれると、mlflow.log_paramsが実行されるか"""
        tracker = MLflowTracker(experiment_name="test_exp")
        params = {"learning_rate": 0.01, "batch_size": 32}
        tracker.log_params(params)

        mock_mlflow.log_params.assert_called_once_with(params)

    def test_init_sets_experiment(self, mock_mlflow):
        """初期化時に実験名がセットされるか"""
        exp_name = "my_experiment"
        _ = MLflowTracker(experiment_name=exp_name)
        mock_mlflow.set_experiment.assert_called_once_with(exp_name)
