"""Task isolation and ordering for video-language Cloud verification."""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict

from shared.protocol import ProtocolError, success_payload


@dataclass
class VideoTaskState:
    backend_state: Any
    cache_position: int
    revision: int = 0
    next_sequence_no: int = 0
    last_seen: float = field(default_factory=time.monotonic)
    responses: Dict[str, dict] = field(default_factory=dict)


class VideoTaskManager:
    def __init__(self, backend, ttl_s=600):
        self.backend = backend
        self.ttl_s = ttl_s
        self.tasks = {}
        self.lock = threading.RLock()
        self.model_lock = threading.Lock()

    def cleanup_expired(self):
        cutoff = time.monotonic() - self.ttl_s
        with self.lock:
            expired = [key for key, task in self.tasks.items() if task.last_seen < cutoff]
            for key in expired:
                task = self.tasks.pop(key)
                self.backend.close_task(task.backend_state)
        return len(expired)

    def init_task(self, payload):
        task_id = payload["task_id"]
        started = time.perf_counter()
        with self.model_lock:
            backend_state, cache_position = self.backend.init_task(
                payload["prompt"], payload["model_family"],
                payload["evidence"], payload.get("generation", {}),
            )
        cloud_compute_s = time.perf_counter() - started
        with self.lock:
            previous = self.tasks.pop(task_id, None)
            if previous is not None:
                self.backend.close_task(previous.backend_state)
            self.tasks[task_id] = VideoTaskState(backend_state, int(cache_position))
        return success_payload(
            modality="video_text", status="initialized", task_id=task_id,
            revision=0, cache_position=int(cache_position),
            cloud_compute_s=cloud_compute_s,
        )

    def propose(self, payload):
        task_id = payload["task_id"]
        request_id = payload["request_id"]
        with self.lock:
            task = self.tasks.get(task_id)
            if task is None:
                raise ProtocolError(f"Video task {task_id!r} is not initialized.")
            if request_id in task.responses:
                return task.responses[request_id]
            if payload["sequence_no"] != task.next_sequence_no:
                raise ProtocolError("Out-of-order video proposal sequence.")
            if payload["base_revision"] != task.revision:
                raise ProtocolError("Stale video proposal revision.")
            if payload["cache_position"] != task.cache_position:
                raise ProtocolError("Stale video cache position.")
            task.last_seen = time.monotonic()
            started = time.perf_counter()
            with self.model_lock:
                accepted, override, cache_position, divergences = self.backend.verify(task.backend_state, payload)
            cloud_compute_s = time.perf_counter() - started
            backend_metrics = self.backend.task_metrics(task.backend_state)
            task.cache_position = int(cache_position)
            task.revision += 1
            task.next_sequence_no += 1
            response = success_payload(
                modality="video_text", task_id=task_id, request_id=request_id,
                sequence_no=payload["sequence_no"], revision=task.revision,
                accepted_count=int(accepted), override_token=override,
                cache_position=task.cache_position, js_divergences=list(divergences),
                cloud_compute_s=cloud_compute_s,
                **backend_metrics,
            )
            task.responses[request_id] = response
            return response

    def exit_task(self, task_id):
        with self.lock:
            task = self.tasks.pop(task_id, None)
        if task is not None:
            self.backend.close_task(task.backend_state)
        return success_payload(modality="video_text", status="exited", task_id=task_id, existed=task is not None)
