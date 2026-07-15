# app/run_edge.py
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.channel import NetworkChannel
from core.config import ChannelConfig, ExperimentConfig, ModelConfig, SpeculativeConfig
from core.local_models import MockDraftModel, LlamaCppDraftModel
from core.metrics import MetricsCollector
from families.speculative.speculative_edge import SpeculativeEdgeRole
from families.speculative.strategy import DPStrategy
from edge.app.result_io import append_jsonl, result_record
from edge.app.run_config import add_common_arguments, parse_args_with_config


def load_humaneval_data(data_path, start_idx, end_idx):
    print(f"[Data] Loading HumanEval data from {data_path} ...")
    if not os.path.exists(data_path):
        print(f"[Warn] {data_path} not found; using in-memory smoke-test prompts.")
        fallback_samples = [
            {"prompt": "def is_prime(n):\n    \"\"\"Return True if n is prime.\"\"\"", "task_id": 10},
            {"prompt": "def quick_sort(arr):\n    \"\"\"Sort the array.\"\"\"", "task_id": 11},
            {"prompt": "def fib(n):\n    \"\"\"Return nth Fibonacci number.\"\"\"", "task_id": 12},
        ]
        return fallback_samples[start_idx: end_idx + 1]

    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            prompt = item.get("prompt")
            if prompt is None:
                continue
            task_id = item.get("task_id", item.get("question_id", idx))
            samples.append({"prompt": prompt, "task_id": task_id})

    sliced_samples = samples[start_idx: end_idx + 1]
    print(f"[Data] Loaded {len(sliced_samples)} samples.")
    return sliced_samples


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PipeSD Edge speculative decoding client")
    add_common_arguments(parser)
    parser.add_argument("--algorithm", type=str, default="pipesd", choices=["cloud_only", "pipesd"])
    parser.add_argument("--mock-draft", "--mock_draft", dest="mock_draft", action="store_true", help="Use a deterministic draft model against a mock or real Cloud service")
    parser.add_argument("--draft-n-gpu-layers", "--draft_n_gpu_layers", dest="draft_n_gpu_layers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ctx-size", "--ctx_size", dest="ctx_size", type=int, default=2048)
    parser.add_argument("--start-index", "--start_index_of_sample", dest="start_index", type=int, default=0)
    parser.add_argument("--end-index", "--end_index_of_sample", dest="end_index", type=int, default=2)
    parser.add_argument("--data-path", "--data_path", dest="data_path", type=str, default="data/humaneval.jsonl")
    parser.set_defaults(
        server_url="http://127.0.0.1:8000", server_timeout_s=120.0,
        bandwidth_mbps=2.5, base_latency_s=0.05, draft_model_path="",
        device="cuda:0", chunk_size=5, max_new_tokens=30,
        output_jsonl="edge/exp/results/text_results.jsonl",
    )
    return parse_args_with_config(parser, "text", argv)


def build_runtime(args, chan_cfg, model_cfg, exp_cfg):
    print("[Mode] Edge/Cloud mode: this process loads only the draft model.")
    draft_model = MockDraftModel() if args.mock_draft else LlamaCppDraftModel(model_cfg, exp_cfg)
    return draft_model, NetworkChannel(chan_cfg)


def main(argv=None):
    args = parse_args(argv)

    chan_cfg = ChannelConfig(
        server_url=args.server_url,
        timeout_s=args.server_timeout_s,
        bandwidth_MBps=args.bandwidth_mbps,
        base_latency_c=args.base_latency_s,
    )
    model_cfg = ModelConfig(
        model_path=args.draft_model_path,
        n_gpu_layers=args.draft_n_gpu_layers,
        threads=args.threads,
        ctx_size=args.ctx_size,
    )
    spec_cfg = SpeculativeConfig(gamma=args.chunk_size)
    exp_cfg = ExperimentConfig(
        algorithm=args.algorithm,
        max_generated_tokens=args.max_new_tokens,
        data_path=args.data_path,
        start_index=args.start_index,
        end_index=args.end_index,
    )

    samples = load_humaneval_data(exp_cfg.data_path, exp_cfg.start_index, exp_cfg.end_index)
    model_node, channel = build_runtime(args, chan_cfg, model_cfg, exp_cfg)

    strategy = DPStrategy(spec_cfg, chan_cfg)
    collector = MetricsCollector(exp_dir="exp/results", filename="benchmark.json")

    runner = SpeculativeEdgeRole(
        model_node=model_node,
        channel=channel,
        strategy=strategy,
        collector=collector,
        model_config=model_cfg,
        exp_cfg=exp_cfg,
    )
    runner.load_model()

    print("\n[Main] Starting evaluation pipeline")
    for sample in samples:
        task_id = sample["task_id"]
        prompt = sample["prompt"]
        print(f"\n[Main] Processing task: {task_id}")
        collector.reset_sample()
        runner.process_task(task_id=task_id, prompt=prompt)
        record = result_record(runner.last_result, {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "modality": "text", "task_type": "humaneval",
            "draft_model_path": os.path.abspath(args.draft_model_path) if args.draft_model_path else None,
            "device": args.device, "server_url": args.server_url,
            "bandwidth_mbps": args.bandwidth_mbps, "base_latency_s": args.base_latency_s,
            "chunk_size": args.chunk_size, "max_new_tokens": args.max_new_tokens,
        })
        append_jsonl(args.output_jsonl, record)

    print("\n[Main] Evaluation finished")
    channel.close()
    print("[Main] Results saved to exp/results/benchmark.json")
    print("[Main] HumanEval samples saved to exp/results/humaneval_samples.jsonl")
    print(f"[Main] Unified task results saved to {args.output_jsonl}")


if __name__ == "__main__":
    main()

