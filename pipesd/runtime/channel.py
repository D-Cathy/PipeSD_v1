"""Transport contract plus an in-process implementation for tests and embedding."""

from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import Any, Callable, Dict, Mapping, Optional


class Channel(ABC):
    @abstractmethod
    def request(
        self,
        endpoint: str,
        payload: Any,
        headers: Optional[Dict[str, str]] = None,
        tag: Optional[str] = None,
    ) -> Any:
        pass

    def stream(self, endpoint: str, payload: Any, headers=None, tag=None):
        raise NotImplementedError(f"{type(self).__name__} does not support streaming.")

    def health(self) -> Dict[str, Any]:
        return {"status": "ok"}

    def close(self) -> None:
        return None


class InProcessChannel(Channel):
    """Dispatch requests directly to callables without a network server."""

    def __init__(self, handlers: Mapping[str, Callable[[Any], Any]]):
        self.handlers = dict(handlers)
        self.closed = False

    def request(self, endpoint, payload, headers=None, tag=None):
        if self.closed:
            raise RuntimeError("Channel is closed.")
        try:
            handler = self.handlers[endpoint]
        except KeyError as exc:
            raise KeyError(f"No in-process handler registered for {endpoint!r}.") from exc
        return handler(payload)

    def submit(self, endpoint_url, data, headers=None, tag=None):
        """Compatibility with the original asynchronous Edge channel API."""
        future = Future()
        try:
            future.set_result(self.request(endpoint_url, data, headers, tag))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def drain_tag(self, tag):
        return []

    def health(self):
        return {"status": "closed" if self.closed else "ok", "handlers": len(self.handlers)}

    def close(self):
        self.closed = True
