# app/run_edge.py
import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.channel import LocalChannel, NetworkChannel
from core.config import ChannelConfig, ExperimentConfig, ModelConfig, SpeculativeConfig
from core.local_models import MockDraftModel, MockTargetVerifier, LlamaCppDraftModel, LlamaCppTargetVerifier
from core.metrics import MetricsCollector
from families.speculative.speculative_edge import SpeculativeEdgeRole
from families.speculative.strategy import DPStrategy


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
            raw_task_id = item.get("task_id", item.get("question_id", idx))
            try:
                task_id = int(str(raw_task_id).split("/")[-1])
            except Exception:
                task_id = raw_task_id
            samples.append({"prompt": prompt.strip(), "task_id": task_id})

    sliced_samples = samples[start_idx: end_idx + 1]
    print(f"[Data] Loaded {len(sliced_samples)} samples.")
    return sliced_samples


def parse_args():
    parser = argparse.ArgumentParser(description="PipeSD unified speculative decoding runner")
    parser.add_argument("--algorithm", type=str, default="pipesd", choices=["cloud_only", "pipesd"])
    parser.add_argument("--deployment", type=str, default="local", choices=["local", "network"], help="local runs draft and target models on one server; network keeps the legacy HTTP split")
    parser.add_argument("--simulation", action="store_true", help="Compatibility alias for --deployment local --mock_models")
    parser.add_argument("--mock_models", action="store_true", help="Use deterministic mock models for a smoke test")
    parser.add_argument("--draft_model_path", type=str, default="")
    parser.add_argument("--target_model_path", type=str, default="")
    parser.add_argument("--draft_n_gpu_layers", type=int, default=0)
    parser.add_argument("--target_n_gpu_layers", type=int, default=-1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ctx_size", type=int, default=2048)
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--bandwidth_MBps", type=float, default=2.5)
    parser.add_argument("--base_latency_c", type=float, default=0.05)
    parser.add_argument("--max_generated_tokens", type=int, default=30)
    parser.add_argument("--start_index_of_sample", type=int, default=0)
    parser.add_argument("--end_index_of_sample", type=int, default=2)
    parser.add_argument("--data_path", type=str, default="data/humaneval.jsonl")
    return parser.parse_args()


def build_runtime(args, chan_cfg, model_cfg, exp_cfg):
    if args.simulation:
        args.deployment = "local"
        args.mock_models = True

    if args.deployment == "local":
        print("[Mode] Unified local server mode: draft model and target model run in one process.")
        if args.mock_models:
            return MockDraftModel(), LocalChannel(chan_cfg, MockTargetVerifier())
        if not args.draft_model_path or not args.target_model_path:
            raise ValueError("Real local mode requires both --draft_model_path and --target_model_path.")
        target_verifier = LlamaCppTargetVerifier(
            model_path=args.target_model_path,
            ctx_size=args.ctx_size,
            n_gpu_layers=args.target_n_gpu_layers,
            threads=args.threads,
            seed=exp_cfg.seed,
        )
        return LlamaCppDraftModel(model_cfg, exp_cfg), LocalChannel(chan_cfg, target_verifier)

    print("[Mode] Legacy network mode: draft runner calls an HTTP target-model service.")
    draft_model = MockDraftModel() if args.mock_models else LlamaCppDraftModel(model_cfg, exp_cfg)
    return draft_model, NetworkChannel(chan_cfg)


def main():
    args = parse_args()

    chan_cfg = ChannelConfig(
        server_url=args.server_url,
        bandwidth_MBps=args.bandwidth_MBps,
        base_latency_c=args.base_latency_c,
    )
    model_cfg = ModelConfig(
        model_path=args.draft_model_path,
        n_gpu_layers=args.draft_n_gpu_layers,
        threads=args.threads,
        ctx_size=args.ctx_size,
    )
    spec_cfg = SpeculativeConfig()
    exp_cfg = ExperimentConfig(
        algorithm=args.algorithm,
        max_generated_tokens=args.max_generated_tokens,
        data_path=args.data_path,
        start_index=args.start_index_of_sample,
        end_index=args.end_index_of_sample,
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

    print("\n[Main] Evaluation finished")
    channel.close()
    print("[Main] Results saved to exp/results/benchmark.json")


if __name__ == "__main__":
    main()

