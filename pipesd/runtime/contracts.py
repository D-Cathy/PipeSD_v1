"""Value objects exchanged by Nodes, Strategies, and Engines."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Union

TaskId = Union[int, str]


class Action(str, Enum):
    """Framework-level actions; paradigms use only the subset they need."""

    CONTINUE = "continue"
    ACCEPT_LOCAL = "accept_local"
    SELF_VERIFY = "self_verify"
    SEND_TO_CLOUD = "send_to_cloud"
    RUN_EDGE = "run_edge"
    RUN_CLOUD = "run_cloud"
    FALLBACK = "fallback"
    REDACT = "redact"
    CALL_TOOL = "call_tool"
    STOP = "stop"


@dataclass
class Task:
    task_id: TaskId
    modality: str
    input_data: Any = None
    prompt: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    task_id: TaskId
    output: Any
    status: str = "completed"
    stop_reason: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollaborationContext:
    task: Task
    state: Dict[str, Any] = field(default_factory=dict)
    observations: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    action: Action
    target: Optional[str] = None
    reason: str = ""
    payload: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
