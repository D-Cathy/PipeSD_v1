# PipeSD Edge

Edge runs on the local computer. It owns the dataset, draft model, speculative
state machine, network simulation, and HTTP client. It never loads a target model.

Install from the repository root:

```bash
python -m pip install -r edge/requirements.txt
```

Start Cloud first, then run a mock-draft communication test:

```bash
python edge/app/run_edge.py --mock_draft --server_url http://CLOUD_HOST:8000 --max_generated_tokens 8
```

Run a real draft model:

```bash
python edge/app/run_edge.py --draft_model_path /models/draft.gguf --server_url http://CLOUD_HOST:8000
```

Use `--gamma 5` to cap the number of pending draft tokens sent in one
verification. Generation stops on the model EOS token or at
`--max_generated_tokens`. Results are written to `exp/results/benchmark.json`;
official HumanEval-format completions are appended to
`exp/results/humaneval_samples.jsonl`.

HumanEval executes generated code. Only in an isolated environment, install the
official evaluator and explicitly acknowledge execution:

```bash
python edge/tasks/evaluate_humaneval.py exp/results/humaneval_samples.jsonl --allow-code-execution
```

`--bandwidth_MBps`, `--base_latency_c`, and the channel timeout control the
simulated link. Set `PIPE_SD_SERVER_URL` only for legacy tools; the new entry uses
the explicit `--server_url` argument.
