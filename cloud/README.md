# PipeSD Cloud

FastAPI service that hosts the target model, verifies speculative tokens, and reports GPU power-integral metrics.

Cloud runs on the GPU server. It owns the target model, per-task snapshots,
speculative-token verification, cleanup, and the FastAPI service. It never loads
the draft model or datasets.

## Directory layout
- `pre_models/`: target-model GGUF files (downloaded or placed manually)

## Running the service
From the repository root:
```bash
python -m pip install -r cloud/requirements.txt
python -m cloud.app.main --target-model-path /models/target.gguf --host 0.0.0.0 --port 8000
```

For a local communication test without a GPU:

```bash
python -m cloud.app.main --mock
```

The service binds to port `8000` by default (see `APP_PORT` in `src/speculative_server.py`).

## API endpoints (high level)
- `POST /init`: initialize a versioned task with a token prefix
- `POST /propose`: send speculative tokens and/or trigger verification
- `POST /exit`: finalize a task and return aggregated metrics
- `GET /health`: service status, protocol version, and active task count

All POST endpoints use MessagePack. Deploy `shared/` unchanged beside both
packages; mismatched protocol versions are rejected.
Backend failures also return a versioned MessagePack error, so Edge reports the
original Cloud exception instead of a misleading protocol mismatch.
