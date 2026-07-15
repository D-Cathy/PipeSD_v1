"""Lifecycle contract for orchestration engines."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from .contracts import Result, Task


class Engine(ABC):
    @abstractmethod
    def run(self, task: Task) -> Result:
        pass

    def cancel(self, task_id) -> bool:
        return False

    def metrics(self, task_id=None) -> Dict[str, Any]:
        return {}
