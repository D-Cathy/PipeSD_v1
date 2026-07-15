"""Decision-policy contract shared by collaboration paradigms."""

from abc import ABC, abstractmethod

from .contracts import CollaborationContext, Decision


class Strategy(ABC):
    @abstractmethod
    def decide(self, context: CollaborationContext) -> Decision:
        pass
