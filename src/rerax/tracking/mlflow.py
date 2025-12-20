from typing import Any

import mlflow

from rerax.tracking.base import BaseTracker


class MLflowTracker(BaseTracker):
    def __init__(self, experiment_name="test_exp"):
        mlflow.set_experiment(experiment_name)

    def log_params(self, params: dict[str, Any]):
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        mlflow.log_metrics(metrics, step=step)
