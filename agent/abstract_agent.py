from __future__ import annotations

from abc import abstractmethod
from typing import Any


class AbstractAgent(object):
    def __init__(self, *args: tuple[Any], **kwargs: dict) -> None:
        """Make agent."""
        pass

    @abstractmethod
    def predict_action(self, *args: tuple[Any], **kwargs: dict) -> tuple[Any, dict]:
        """Predict action given state."""
        ...

    @abstractmethod
    def save(self, *args: tuple[Any], **kwargs: dict) -> Any:
        """Save agent."""
        ...

    @abstractmethod
    # TODO what is the return type? A callable?
    def load(self, *args: tuple[Any], **kwargs: dict) -> Any:
        """Load agent."""
        ...

    @abstractmethod
    def update_agent(self, *args: tuple[Any], **kwargs: dict) -> Any | None:
        """Update agent."""
        ...
