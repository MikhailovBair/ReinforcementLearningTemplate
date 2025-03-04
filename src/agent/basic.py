from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    @abstractmethod
    def get_action(self, observation: Any) -> Any:
        pass
