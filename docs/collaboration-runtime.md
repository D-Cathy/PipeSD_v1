# Collaboration runtime

PipeSD exposes four framework-level extension points without changing the
existing text and video command-line entry points.

## Composition model

```text
Task -> Engine -> Strategy -> Decision
          |                    |
          +---- Node <---------+
                  |
               Channel
```

- **Node** hides a model backend and its execution location.
- **Strategy** converts runtime observations into a framework `Decision`.
- **Engine** owns one paradigm's orchestration loop and returns a `Result`.
- **Channel** transports requests without leaking HTTP, gRPC, or in-process
  details into an Engine.

The current text and video speculative implementations are the first built-in
Engines. Routing, cascade, split inference, and agent collaboration remain
future Engines; they should reuse these contracts rather than add branches to
the speculative loops.

## Public value objects

```python
from pipesd import Action, CollaborationContext, Decision, Result, Task

task = Task(
    task_id="video-001",
    modality="video",
    input_data="/data/example.mp4",
    prompt="Describe the video.",
    metadata={"max_new_tokens": 64},
)
```

`Result` always contains `task_id`, `output`, `status`, `stop_reason`,
`metrics`, and `metadata`. Paradigm-specific diagnostics stay inside the last
two dictionaries.

## Nodes

Existing Python model objects can be adapted without rewriting them:

```python
from pipesd.nodes import BackendNode, NodeRequest

edge = BackendNode(existing_backend, node_id="edge", location="local")
output = edge.execute(NodeRequest("infer", args=(request,)))
```

Remote services use `HTTPNode`, so Engines invoke named operations instead of
building URLs:

```python
from pipesd.nodes import HTTPNode

cloud = HTTPNode(
    "http://cloud-host:8000",
    channel,
    endpoints={"init": "/init", "propose": "/propose", "exit": "/exit"},
)
response = cloud.invoke("init", packed_request, headers={"Content-Type": content_type})
```

A future SPEAR integration should implement a `Node` and a `Channel`; model
selection and scheduling must not be added to the speculative Engine.

## Strategies

New decision policies subclass `Strategy`:

```python
from pipesd import Action, Decision, Strategy

class PrivacyRoutingStrategy(Strategy):
    def decide(self, context):
        if context.observations.get("privacy_level") == "S3":
            return Decision(Action.RUN_EDGE, reason="private")
        return Decision(Action.RUN_CLOUD, reason="cloud_allowed")
```

Built-in speculative policies are importable from `pipesd.strategies`:

```python
from pipesd.strategies import DPStrategy, VideoConfidenceStrategy
```

The video Engine accepts a custom strategy through its `strategy=` argument.
The original threshold configuration remains supported.

## Engines

All public Engines accept a `Task` and return a `Result`:

```python
from pipesd import Task
from pipesd.engines import VideoSpeculativeEngine

engine = VideoSpeculativeEngine(
    draft_backend,
    channel,
    video_config,
    strategy=video_strategy,
)
result = engine.run(Task("video-1", "video", "/data/video.mp4", "Describe it."))

print(result.output)
print(result.metrics)
```

The legacy `process_task(...)` methods remain available during migration.

## Channels

```python
from pipesd.channels import InProcessChannel, NetworkChannel
```

`NetworkChannel` is the production HTTP transport used by the existing Edge
entry points. `InProcessChannel` dispatches to Python callables and supports
both the new synchronous `request(...)` API and the legacy
`submit(...).result()` API.

Streaming is part of the public contract but intentionally raises
`NotImplementedError` until an SSE, WebSocket, or gRPC transport is added.

## Compatibility boundary

- Existing `/init`, `/propose`, `/exit` and video endpoints are unchanged.
- Existing text and video CLI commands are unchanged.
- Existing model backend interfaces remain valid through `BackendNode`.
- Models, datasets, and experiment outputs remain external assets.
- A regular cloud API can serve routing/cascade Engines, but strict
  speculative decoding still requires logits and cache control.
