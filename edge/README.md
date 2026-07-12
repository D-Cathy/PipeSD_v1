# PipeSD Unified Runner

This folder is now the main merged code folder for speculative decoding experiments. It can run the small draft model and large target model on the same server process, while keeping the old HTTP channel as an optional deployment mode.

## Recommended single-server run

```bash
python app/run_edge.py \
  --deployment local \
  --draft_model_path pre_models/path-to-small-model.gguf \
  --target_model_path pre_models/path-to-large-model.gguf \
  --max_generated_tokens 40
```

## Smoke test

```bash
python app/run_edge.py --deployment local --mock_models --max_generated_tokens 8
```

## Legacy distributed run

Start the old cloud service from `../cloud`, then run:

```bash
python app/run_edge.py --deployment network --server_url http://127.0.0.1:8000
```

## Main merged pieces

- `app/run_edge.py`: unified experiment entry.
- `core/local_models.py`: draft-model and target-verifier adapters.
- `core/channel.py`: `LocalChannel` for in-process calls and `NetworkChannel` for the old service.
- `families/speculative/speculative_edge.py`: algorithm runner that talks to either channel through the same interface.
