"""Generic Edge entry point for video-to-text speculative decoding backends."""

import argparse
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.channel import NetworkChannel
from core.config import ChannelConfig
from families.video_speculative.backends import MockVideoDraftBackend
from families.video_speculative.qwen3_vl_backend import Qwen3VLDraftBackend
from families.video_speculative.config import VideoSpeculativeConfig
from families.video_speculative.video_edge import VideoSpeculativeEdgeRole
from edge.app.result_io import append_jsonl, result_record
from edge.app.run_config import add_common_arguments, parse_args_with_config
from pipesd.runtime import Result


def load_factory(spec):
    module_name, separator, attribute = spec.partition(":")
    if not separator:
        raise ValueError("Backend factory must use module:callable syntax.")
    return getattr(importlib.import_module(module_name), attribute)


def load_tasks(args):
    if args.input_jsonl:
        tasks = []
        with Path(args.input_jsonl).open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                item = json.loads(line)
                video = item.get("video") or item.get("video_path") or item.get("input")
                if not video:
                    raise ValueError(f"Video manifest line {index + 1} has no video path.")
                tasks.append({
                    "task_id": item.get("task_id", f"video-{index}"),
                    "video": video,
                    "prompt": item.get("prompt", args.prompt),
                })
        return tasks
    if not args.video:
        raise ValueError("Provide --input/--video or --input-jsonl.")
    return [{"task_id": args.task_id, "video": args.video, "prompt": args.prompt}]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PipeSD video-to-text Edge client")
    add_common_arguments(parser)
    parser.add_argument("--input", "--video", dest="video")
    parser.add_argument("--input-jsonl", help="Run many videos in one process and reuse draft weights")
    parser.add_argument("--prompt", default="Please describe the video in detail.")
    parser.add_argument("--task-id", default="video-0")
    parser.add_argument("--model-family", default="qwen3_vl")
    parser.add_argument("--backend-factory", help="Optional module:callable returning a VideoDraftBackend")
    parser.add_argument("--backend-kwargs", default="{}", help="JSON kwargs passed to the backend factory")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--rlt-diff-threshold", type=float, default=0.001)
    parser.add_argument("--rlt-downsample-size", type=int, default=32)
    parser.add_argument("--high-confidence", type=float, default=0.9)
    parser.add_argument("--mid-confidence", type=float, default=0.75)
    parser.add_argument("--verification-rule", choices=["js", "specdec_original"], default="js")
    parser.add_argument("--js-threshold", type=float, default=0.4)
    parser.set_defaults(
        server_url="http://127.0.0.1:8000", server_timeout_s=300.0,
        bandwidth_mbps=0.0, base_latency_s=0.0, draft_model_path="",
        device="cuda:0", chunk_size=6, max_new_tokens=256,
        output_jsonl="edge/exp/results/video_results.jsonl",
    )
    return parser, parse_args_with_config(parser, "video", argv)


def main(argv=None):
    parser, args = parse_args(argv)

    if args.backend_factory and args.draft_model_path:
        parser.error("Use either --backend-factory or --draft-model-path, not both.")
    if args.backend_factory:
        backend = load_factory(args.backend_factory)(**json.loads(args.backend_kwargs))
    elif args.draft_model_path:
        backend = Qwen3VLDraftBackend(
            model_path=args.draft_model_path, device=args.device,
            allow_cpu=args.allow_cpu, max_frames=args.max_frames, top_k=args.top_k,
            rlt_diff_threshold=args.rlt_diff_threshold,
            rlt_downsample_size=args.rlt_downsample_size,
        )
    else:
        backend = MockVideoDraftBackend()
    channel = NetworkChannel(ChannelConfig(
        server_url=args.server_url, timeout_s=args.server_timeout_s,
        bandwidth_MBps=args.bandwidth_mbps, base_latency_c=args.base_latency_s,
    ))
    config = VideoSpeculativeConfig(
        chunk_gamma=args.chunk_size, max_new_tokens=args.max_new_tokens,
        high_conf_threshold=args.high_confidence, mid_conf_threshold=args.mid_confidence,
        verification_rule=args.verification_rule, js_threshold=args.js_threshold,
    )
    try:
        role = VideoSpeculativeEdgeRole(backend, channel, config, args.model_family)
        tasks = load_tasks(args)
        print(f"[Main] Processing {len(tasks)} video task(s); draft weights are reused in this process.")
        for task in tasks:
            result = role.process_task(task["task_id"], task["video"], task["prompt"])
            run_metadata = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "modality": "video", "task_type": "video_caption",
                "video_path": str(Path(task["video"]).resolve()),
                "prompt": task["prompt"], "model_family": args.model_family,
                "draft_model_path": str(Path(args.draft_model_path).resolve()) if args.draft_model_path else None,
                "device": args.device, "server_url": args.server_url,
                "bandwidth_mbps": args.bandwidth_mbps,
                "base_latency_s": args.base_latency_s,
                "max_frames": args.max_frames, "chunk_size": args.chunk_size,
                "max_new_tokens": args.max_new_tokens, "top_k": args.top_k,
                "rlt_diff_threshold": args.rlt_diff_threshold,
                "rlt_downsample_size": args.rlt_downsample_size,
                "verification_rule": args.verification_rule,
                "js_threshold": args.js_threshold,
            }
            standardized = Result(
                task_id=task["task_id"], output=result["text"],
                stop_reason=result.get("stop_reason", "max_tokens"),
                metrics=dict(result.get("metrics", {})),
                metadata={
                    "modality": "video", "tokens": list(result.get("tokens", [])),
                    "cloud_queries": result.get("cloud_queries", 0),
                },
            )
            record = result_record(standardized, run_metadata)
            append_jsonl(args.output_jsonl, record)
            print(json.dumps(record, ensure_ascii=False, indent=2))
            print(f"[Result] Appended to {args.output_jsonl}")
    finally:
        channel.close()


if __name__ == "__main__":
    main()
