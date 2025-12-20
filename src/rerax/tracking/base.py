from typing import Any, Protocol


class BaseTracker(Protocol):
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        pass

    def log_params(self, params: dict[str, Any]):
        pass
