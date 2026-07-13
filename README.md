# PipeSD_text

PipeSD_text is a true two-machine speculative-decoding system. Edge runs the
small draft model on a local computer; Cloud runs the large target model on a GPU
server. HTTP is the only target-verification path.

```text
Edge: dataset -> draft model -> speculative state -> NetworkChannel
                                                    |
                       /health /init /propose /exit |
                                                    v
Cloud: FastAPI -> task manager -> target verifier -> GPU/energy metrics
```

## Layout

- `edge/`: local-only application, draft model, algorithms, tasks, and tests.
- `cloud/`: server-only API, task state, target backends, and tests.
- `shared/`: versioned protocol and MessagePack serialization; copy unchanged to both hosts.
- `scripts/`: separate Edge and Cloud packaging scripts for PowerShell and POSIX shells.
- `docs/video-vive-integration.md`: incremental VIVE video-to-text integration contract.

The video family is isolated under `edge/families/video_speculative/` and
`cloud/core/video_tasks.py`; it reuses the production network channel without
changing the text decoding family.

## Quick end-to-end smoke test

Terminal 1:

```bash
python -m pip install -r cloud/requirements.txt
python -m cloud.app.main --mock
```

Terminal 2:

```bash
python -m pip install -r edge/requirements.txt
python edge/app/run_edge.py --mock_draft --server_url http://127.0.0.1:8000 --max_generated_tokens 8
```

See `edge/README.md` and `cloud/README.md` for real-model commands. The former
single-process target path has been removed from the new runner; `cloud/src/`
remains temporarily as historical experiment code and is not used by the new API.
