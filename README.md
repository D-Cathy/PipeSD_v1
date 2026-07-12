# PipeSD

PipeSD is a unified speculative decoding framework for small-draft and large-target models.

The current recommended deployment runs both models on the same server process. The previous cloud/edge HTTP split is still kept for legacy comparison experiments, but it is no longer the main path.

## Repository layout

- `edge/`: main merged code folder. It contains the unified runner, draft model adapters, local target verifier, channels, metrics, and speculative decoding strategies.
- `cloud/`: legacy FastAPI target-model service for the old distributed HTTP deployment.

## Recommended flow

Run a smoke test without model files:

```bash
cd edge
python app/run_edge.py --deployment local --mock_models --max_generated_tokens 8
```

Run real single-server speculative decoding:

```bash
cd edge
python app/run_edge.py \
  --deployment local \
  --draft_model_path pre_models/path-to-small-model.gguf \
  --target_model_path pre_models/path-to-large-model.gguf \
  --max_generated_tokens 40
```

Use the old cloud/edge HTTP mode only when you need a distributed baseline:

```bash
cd edge
python app/run_edge.py --deployment network --server_url http://127.0.0.1:8000
```

## Key merged files

- `edge/app/run_edge.py`: unified experiment entry.
- `edge/core/local_models.py`: local draft model and target verifier adapters.
- `edge/core/channel.py`: `LocalChannel` for in-process calls and `NetworkChannel` for legacy HTTP.
- `edge/families/speculative/speculative_edge.py`: speculative decoding runner that is deployment-agnostic.

## Notes

- Ubuntu >= 22.04 and CUDA are recommended for real model runs.
- `edge/install.sh` is the current dependency reference.
- `cloud/install.sh` and `cloud/README.md` are retained for the legacy FastAPI service.
