"""Framework-neutral contracts shared by collaboration paradigms."""

from .channel import Channel, InProcessChannel
from .contracts import Action, CollaborationContext, Decision, Result, Task
from .engine import Engine
from .node import BackendNode, HTTPNode, Node, NodeCapabilities, NodeRequest, ensure_node
from .strategy import Strategy

__all__ = [
    "Action",
    "BackendNode",
    "Channel",
    "CollaborationContext",
    "Decision",
    "Engine",
    "HTTPNode",
    "InProcessChannel",
    "Node",
    "NodeCapabilities",
    "NodeRequest",
    "Result",
    "Strategy",
    "Task",
    "ensure_node",
]
