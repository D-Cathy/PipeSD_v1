"""Deployment-agnostic Edge orchestrator for video-to-text speculation."""

import sys
import time
import uuid

from shared.protocol import FinalizeRequest, ProtocolError
from shared.serialization import CONTENT_TYPE, pack_message
from shared.video_protocol import VideoInitRequest, VideoProposalRequest, VideoVerificationResponse
from pipesd.runtime.node import HTTPNode, ensure_node
from pipesd.runtime import Action, CollaborationContext, Engine, Result, Task
from .strategy import strategy_from_config


class VideoSpeculativeEdgeRole(Engine):
    def __init__(self, draft_backend, channel, config, model_family="qwen3_vl", strategy=None):
        self.edge_node = ensure_node(draft_backend, node_id="video-draft", location="edge")
        self.draft = getattr(self.edge_node, "backend", draft_backend)
        self.channel = channel
        self.cloud_node = HTTPNode(
            channel.config.server_url,
            channel,
            node_id="video-target",
            endpoints={
                "init": "/video/init",
                "propose": "/video/propose",
                "exit": "/video/exit",
            },
        )
        self.config = config
        self.model_family = model_family
        self.strategy = strategy or strategy_from_config(config)
        self.transport = None

    def run(self, task):
        if not isinstance(task, Task):
            raise TypeError("VideoSpeculativeEdgeRole.run expects a pipesd.Task.")
        if task.modality != "video":
            raise ValueError(f"Video speculative engine cannot run modality {task.modality!r}.")
        raw = self.process_task(task.task_id, task.input_data, task.prompt)
        return Result(
            task_id=task.task_id,
            output=raw["text"],
            stop_reason=raw.get("stop_reason", "max_tokens"),
            metrics=dict(raw.get("metrics", {})),
            metadata={"tokens": list(raw.get("tokens", [])), "cloud_queries": raw.get("cloud_queries", 0)},
        )

    def _request(self, endpoint, message):
        payload = pack_message(message)
        started = time.perf_counter()
        operation = endpoint.rsplit("/", 1)[-1]
        result = self.cloud_node.invoke(
            operation, payload, headers={"Content-Type": CONTENT_TYPE},
        )
        elapsed = time.perf_counter() - started
        if self.transport is not None:
            self.transport["bytes_sent"] += len(payload)
            if isinstance(result, dict):
                self.transport["bytes_received"] += len(pack_message(result))
            self.transport["request_latency_s"][endpoint] = (
                self.transport["request_latency_s"].get(endpoint, 0.0) + elapsed
            )
            self.transport["request_count"][endpoint] = (
                self.transport["request_count"].get(endpoint, 0) + 1
            )
        if not isinstance(result, dict) or result.get("error"):
            raise ProtocolError(str(result.get("error", result)))
        return result

    def process_task(self, task_id, video_path, prompt):
        total_started = time.perf_counter()
        task = Task(task_id, "video", input_data=video_path, prompt=prompt)
        self.transport = {
            "bytes_sent": 0, "bytes_received": 0,
            "request_latency_s": {}, "request_count": {},
        }
        initialize_started = time.perf_counter()
        evidence = self.edge_node.invoke("initialize", video_path, prompt)
        edge_initialize_s = time.perf_counter() - initialize_started
        init = self._request("/video/init", VideoInitRequest(
            task_id=task_id, prompt=prompt, model_family=self.model_family,
            evidence=evidence, generation={"max_new_tokens": self.config.max_new_tokens},
        ))
        revision = int(init["revision"])
        cache_position = int(init["cache_position"])
        sequence_no = 0
        output = []
        committed_for_cloud = []
        cloud_queries = 0
        edge_draft_s = 0.0
        edge_self_verify_s = 0.0
        accepted_lengths = []
        js_divergences = []
        cloud_compute_s = float(init.get("cloud_compute_s", 0.0))
        route_counts = {"edge_high": 0, "edge_mid": 0, "cloud": 0}
        cache_reused_tokens = 0
        cache_rollbacks = 0
        model_cache_length = 0

        try:
            while len(output) < self.config.max_new_tokens and not self.edge_node.invoke("is_finished"):
                remaining = self.config.max_new_tokens - len(output)
                draft_started = time.perf_counter()
                chunk = self.edge_node.invoke("draft_chunk", min(self.config.chunk_gamma, remaining))
                edge_draft_s += time.perf_counter() - draft_started
                if not chunk:
                    break
                avg_confidence = sum(item.confidence for item in chunk) / len(chunk)
                decision = self.strategy.decide(CollaborationContext(
                    task,
                    state={"generated_tokens": len(output), "remaining_tokens": remaining},
                    observations={
                        "average_confidence": avg_confidence,
                        "chunk_size": len(chunk),
                    },
                ))
                if decision.action == Action.ACCEPT_LOCAL:
                    route_counts["edge_high"] += 1
                    accepted = [item.token_id for item in chunk]
                    output.extend(accepted)
                    committed_for_cloud.extend(accepted)
                    self.edge_node.invoke("commit_tokens", accepted)
                    continue
                verify_started = time.perf_counter()
                self_verified = (
                    decision.action == Action.SELF_VERIFY
                    and self.edge_node.invoke("self_verify", chunk)
                )
                edge_self_verify_s += time.perf_counter() - verify_started
                if self_verified:
                    route_counts["edge_mid"] += 1
                    accepted = [item.token_id for item in chunk]
                    output.extend(accepted)
                    committed_for_cloud.extend(accepted)
                    self.edge_node.invoke("commit_tokens", accepted)
                    continue

                request_id = uuid.uuid4().hex
                route_counts["cloud"] += 1
                response = VideoVerificationResponse.from_dict(self._request(
                    "/video/propose",
                    VideoProposalRequest(
                        task_id=task_id, request_id=request_id, sequence_no=sequence_no,
                        base_revision=revision, cache_position=cache_position,
                        route="cloud", committed_tokens=committed_for_cloud,
                        tokens=chunk, verification_rule=self.config.verification_rule,
                        js_threshold=self.config.js_threshold,
                    ),
                ))
                if response.request_id != request_id:
                    raise ProtocolError("Video response request_id mismatch.")
                committed_for_cloud = []
                actual = [item.token_id for item in chunk[:response.accepted_count]]
                if response.override_token is not None:
                    actual.append(response.override_token)
                output.extend(actual)
                self.edge_node.invoke("commit_tokens", actual)
                self.edge_node.invoke("apply_cloud_result", response.accepted_count, response.override_token)
                revision = response.revision
                cache_position = response.cache_position
                sequence_no += 1
                cloud_queries += 1
                accepted_lengths.append(response.accepted_count)
                js_divergences.extend(response.js_divergences)
                cloud_compute_s += response.cloud_compute_s
                cache_reused_tokens = response.cache_reused_tokens
                cache_rollbacks = response.cache_rollbacks
                model_cache_length = response.model_cache_length

            tokens = output[:self.config.max_new_tokens]
            total_s = time.perf_counter() - total_started
            backend_metrics = (
                self.edge_node.metrics()
            )
            return {
                "task_id": task_id, "tokens": output[:self.config.max_new_tokens],
                "text": self.edge_node.invoke("decode", tokens),
                "cloud_queries": cloud_queries,
                "metrics": {
                    "generated_tokens": len(tokens),
                    "total_time_s": total_s,
                    "tokens_per_second": len(tokens) / total_s if total_s > 0 else 0.0,
                    "edge_initialize_s": edge_initialize_s,
                    "edge_draft_s": edge_draft_s,
                    "edge_self_verify_s": edge_self_verify_s,
                    "cloud_compute_s": cloud_compute_s,
                    "network_roundtrip_s": sum(self.transport["request_latency_s"].values()),
                    "bytes_sent": self.transport["bytes_sent"],
                    "bytes_received": self.transport["bytes_received"],
                    "average_accept_length": (
                        sum(accepted_lengths) / len(accepted_lengths) if accepted_lengths else 0.0
                    ),
                    "accepted_lengths": accepted_lengths,
                    "js_divergences": js_divergences,
                    "mean_js_divergence": (
                        sum(js_divergences) / len(js_divergences) if js_divergences else 0.0
                    ),
                    "route_counts": route_counts,
                    "request_count": dict(self.transport["request_count"]),
                    "cloud_cache_reused_tokens": cache_reused_tokens,
                    "cloud_cache_rollbacks": cache_rollbacks,
                    "cloud_model_cache_length": model_cache_length,
                    **backend_metrics,
                },
            }
        finally:
            handling_error = sys.exc_info()[0] is not None
            try:
                self._request("/video/exit", FinalizeRequest(task_id=task_id))
            except Exception:
                if not handling_error:
                    raise


class VideoSpeculativeEngine(VideoSpeculativeEdgeRole):
    """Public name for the video orchestration engine."""
