# VIVE as a reusable PipeSD video-to-text family

VIVE drafts and verifies **text tokens** conditioned on compressed video
evidence. PipeSD keeps this distinct from frame/latent speculative generation.

## Deployment split

| VIVE responsibility | PipeSD location |
| --- | --- |
| dataset/video loading | `edge/tasks/` |
| projected visual-token extraction | Edge VLM adapter |
| temporal pruning and base/extra pooling | `edge/integrations/vive.py` |
| 2B text drafting and confidence routing | `edge/families/video_speculative/` |
| HTTP transport and bandwidth model | `edge/core/channel.py` |
| task ordering, idempotency and cleanup | `cloud/core/video_tasks.py` |
| 8B visual prefill, KV cache and JS/p-q verification | `cloud/integrations/vive.py` |
| wire format | `shared/video_protocol.py` |

VIVE remains a separate checkout (Apache-2.0) and is not copied into PipeSD.
Set it up beside the project, for example `C:/Users/.../Speculative_decoding/vive`.

## Wire flow

1. `/video/init` uploads prompt plus base/extra visual evidence once. Cloud builds
   the dense verification context and owns the authoritative KV cache.
2. High-confidence chunks are accepted locally. Mid-confidence chunks use Edge
   self-verification. Their token ids accumulate in `committed_tokens`.
3. `/video/propose` first advances Cloud with `committed_tokens`, then validates
   the low-confidence candidate using sparse top-k probabilities.
4. Cloud returns accepted count, optional override token, revision and cache
   position. Edge rolls back through its backend adapter when required.
5. `/video/exit` frees task-specific visual memory and KV cache.

## Why sparse top-k probabilities

VIVE already aligns Edge top-k token strings/ids with the Cloud vocabulary.
Sending the full vocabulary distribution for every token is unnecessary and
dominates bandwidth. PipeSD therefore sends `topk_ids`, `topk_probs`, the drafted
token and confidence per position.

## Real backend contract

The Edge adapter implements `VideoDraftBackend`: initialize/compress video,
draft a chunk, self-verify, apply Cloud correction, detect EOS and decode text.

The built-in Qwen3-VL Edge backend is selected with `--draft-model-path`. It
samples RGB frames once, produces text-token top-k probabilities locally, and
sends the sampled frames during `/video/init`. CUDA is required by default;
CPU execution requires `--device cpu --allow-cpu` because it is impractically
slow for normal 2B video-model benchmarking.

The Cloud adapter supplies a verifier object to `ViveCloudBackend` with:

```python
init_task(prompt, model_family, evidence, generation) -> (state, cache_position)
verify_chunk(state, committed_tokens, candidate_tokens, rule, js_threshold)
    -> (accepted_count, override_token, cache_position, js_divergences)
close_task(state)
```

The next GPU integration step is to extract these operations from VIVE's
`ParallelSpeculativeDecoder._prefill_models`, `_cloud_verify_chunk`, cache
rollback, and cleanup logic without leaving either model in the same process.

PipeSD now includes a correctness-first Qwen3-VL Cloud backend selected with
`--video-target-model-path`. It verifies sparse draft probabilities with JS
divergence or the original `p/q` rule and returns the Cloud argmax on rejection.
It currently performs full-context forward passes; VIVE KV-cache reuse remains
the next performance optimization and does not require a protocol change.

## Supported strategies

- `token_adapter`: upload compressed base/extra projected tokens.
- `pixel_extra`: upload compressed frames and let Cloud compute dense evidence.
- verification rule `js`: reject when JS divergence exceeds the threshold.
- verification rule `specdec_original`: probabilistic `min(1, p/q)` acceptance.

Start with the Mock backend and protocol tests before loading Qwen3-VL 2B/8B.

## Executable extension points

Mock Cloud:

```bash
python -m cloud.app.main --mock
```

Mock Edge (the video path is metadata only in this mode):

```bash
python edge/app/run_video_edge.py --video sample.mp4 --server-url http://127.0.0.1:8000
```

Real Cloud deployments inject a factory without changing the runner:

```bash
python -m cloud.app.main --mock \
  --video-backend-factory my_vive_cloud:create_backend \
  --video-backend-kwargs '{"model_path":"Qwen3-VL-8B-Instruct","device":"cuda:0"}'
```

```bash
python edge/app/run_video_edge.py --video sample.mp4 \
  --draft-model-path edge/data/models/Qwen3-VL-2B-Instruct \
  --device cuda:0 --server-url http://127.0.0.1:8000
```
