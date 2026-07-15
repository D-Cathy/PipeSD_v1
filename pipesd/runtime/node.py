"""Execution-location abstraction for local and remote model runtimes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional


@dataclass
class NodeCapabilities:
    modalities: Iterable[str] = field(default_factory=tuple)
    models: Iterable[str] = field(default_factory=tuple)
    supports_streaming: bool = False
    supports_logits: bool = False
    supports_kv_cache: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeRequest:
    operation: str
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)


class Node(ABC):
    """A computation unit independent of where and how it is hosted."""

    def __init__(self, node_id: str, location: str, capabilities: Optional[NodeCapabilities] = None):
        self.node_id = node_id
        self.location = location
        self._capabilities = capabilities or NodeCapabilities()

    @property
    def capabilities(self) -> NodeCapabilities:
        return self._capabilities

    def initialize(self, task) -> Any:
        return None

    @abstractmethod
    def execute(self, request: NodeRequest, state: Any = None) -> Any:
        pass

    def invoke(self, operation: str, *args, **kwargs) -> Any:
        return self.execute(NodeRequest(operation, args=args, kwargs=kwargs))

    def supports(self, operation: str) -> bool:
        return False

    def finalize(self, task_id, state: Any = None) -> None:
        return None

    def health(self) -> Dict[str, Any]:
        return {"status": "ok", "node_id": self.node_id, "location": self.location}

    def metrics(self) -> Dict[str, Any]:
        return {}


class BackendNode(Node):
    """Compatibility adapter exposing an existing Python backend as a Node."""

    def __init__(
        self,
        backend: Any,
        node_id: str = "local",
        location: str = "local",
        capabilities: Optional[NodeCapabilities] = None,
    ):
        super().__init__(node_id, location, capabilities)
        self.backend = backend

    def execute(self, request: NodeRequest, state: Any = None) -> Any:
        try:
            operation = getattr(self.backend, request.operation)
        except AttributeError as exc:
            raise ValueError(
                f"Backend on node {self.node_id!r} does not support operation {request.operation!r}."
            ) from exc
        return operation(*request.args, **request.kwargs)

    def supports(self, operation: str) -> bool:
        return callable(getattr(self.backend, operation, None))

    def metrics(self) -> Dict[str, Any]:
        if hasattr(self.backend, "runtime_metrics"):
            return dict(self.backend.runtime_metrics())
        if hasattr(self.backend, "task_metrics"):
            return dict(self.backend.task_metrics(None))
        return {}


class HTTPNode(Node):
    """Remote computation node reached through a configured Channel."""

    def __init__(
        self,
        base_url: str,
        channel: Any,
        node_id: str = "cloud",
        endpoints: Optional[Dict[str, str]] = None,
        capabilities: Optional[NodeCapabilities] = None,
    ):
        super().__init__(node_id, "remote", capabilities)
        self.base_url = base_url.rstrip("/")
        self.channel = channel
        self.endpoints = dict(endpoints or {})

    def execute(self, request: NodeRequest, state: Any = None) -> Any:
        endpoint = self.endpoints.get(request.operation, request.operation)
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        url = f"{self.base_url}{endpoint}"
        payload = request.args[0] if request.args else request.kwargs.pop("payload", None)
        headers = request.kwargs.pop("headers", None)
        tag = request.kwargs.pop("tag", None)
        if request.kwargs:
            raise ValueError(f"Unsupported HTTPNode request options: {sorted(request.kwargs)}")
        if callable(getattr(self.channel, "request", None)):
            return self.channel.request(url, payload, headers=headers, tag=tag)
        return self.channel.submit(url, payload, headers=headers, tag=tag).result()

    def supports(self, operation: str) -> bool:
        return operation in self.endpoints or operation.startswith("/")

    def health(self) -> Dict[str, Any]:
        if callable(getattr(self.channel, "health", None)):
            return dict(self.channel.health())
        return {"status": "unknown", "node_id": self.node_id, "location": self.location}


def ensure_node(value: Any, *, node_id: str = "local", location: str = "local") -> Node:
    return value if isinstance(value, Node) else BackendNode(value, node_id=node_id, location=location)
