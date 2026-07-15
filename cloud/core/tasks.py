"""Thread-safe multi-task protocol state for the Cloud verifier."""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict

from shared.protocol import ProtocolError, error_payload, success_payload
from pipesd.runtime.node import ensure_node


@dataclass
class TaskState:
    model_state: Any
    n_past: int
    revision: int = 0
    next_sequence_no: int = 0
    last_seen: float = field(default_factory=time.monotonic)
    responses: Dict[str, dict] = field(default_factory=dict)


class TaskManager:
    def __init__(self, backend, ttl_s=600, seed=42):
        self.backend_node = ensure_node(backend, node_id="text-target", location="cloud")
        self.backend = getattr(self.backend_node, "backend", backend)
        self.ttl_s = ttl_s
        self.seed = seed
        self.tasks = {}
        self.lock = threading.RLock()
        self.model_lock = threading.Lock()

    def cleanup_expired(self):
        cutoff = time.monotonic() - self.ttl_s
        with self.lock:
            expired = [task_id for task_id, task in self.tasks.items() if task.last_seen < cutoff]
            for task_id in expired:
                del self.tasks[task_id]
        return len(expired)

    def init_task(self, payload):
        task_id = payload["task_id"]
        tokens = list(payload.get("tokens", []))
        with self.model_lock:
            model_state = self.backend_node.invoke("init", tokens)
        with self.lock:
            self.tasks[task_id] = TaskState(model_state=model_state, n_past=len(tokens))
        return success_payload(status="initialized", task_id=task_id, n_past=len(tokens), revision=0)

    def propose(self, payload):
        task_id = payload["task_id"]
        request_id = payload["request_id"]
        with self.lock:
            task = self.tasks.get(task_id)
            if task is None:
                raise ProtocolError(f"Task {task_id!r} is not initialized.")
            if request_id in task.responses:
                return task.responses[request_id]
            if payload["sequence_no"] != task.next_sequence_no:
                raise ProtocolError("Out-of-order proposal sequence.")
            if payload["base_revision"] != task.revision or payload["n_past"] != task.n_past:
                raise ProtocolError("Stale proposal revision or n_past.")
            task.last_seen = time.monotonic()
            with self.model_lock:
                accepted, final_token, n_past = self.backend_node.invoke(
                    "verify",
                    task.model_state, list(payload["tokens"]), list(payload["probs"]), self.seed
                )
            task.n_past = n_past
            task.revision += 1
            task.next_sequence_no += 1
            response = success_payload(
                task_id=task_id, request_id=request_id,
                sequence_no=payload["sequence_no"], revision=task.revision,
                n_accepted=accepted, n_speculative=len(payload["tokens"]),
                final_token=final_token, n_past=n_past,
            )
            task.responses[request_id] = response
            return response

    def exit_task(self, task_id):
        with self.lock:
            existed = self.tasks.pop(task_id, None) is not None
        return success_payload(status="exited", task_id=task_id, existed=existed)
