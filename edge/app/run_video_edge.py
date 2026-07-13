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


def load_factory(spec):
    module_name, separator, attribute = spec.partition(":")
    if not separator:
        raise ValueError("Backend factory must use module:callable syntax.")
    return getattr(importlib.import_module(module_name), attribute)


def append_jsonl(path, record):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def main():
    parser = argparse.ArgumentParser(description="PipeSD video-to-text Edge client")
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompt", default="Please describe the video in detail.")
    parser.add_argument("--task-id", default="video-0")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--server-timeout-s", type=float, default=300)
    parser.add_argument("--bandwidth-mbps", type=float, default=0.0,
                        help="Simulated upload bandwidth in MB/s; 0 disables throttling")
    parser.add_argument("--base-latency-s", type=float, default=0.0,
                        help="Simulated round-trip base latency in seconds")
    parser.add_argument("--model-family", default="qwen3_vl")
    parser.add_argument("--backend-factory", help="Optional module:callable returning a VideoDraftBackend")
    parser.add_argument("--backend-kwargs", default="{}", help="JSON kwargs passed to the backend factory")
    parser.add_argument("--draft-model-path", help="Local Qwen3-VL draft model directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--rlt-diff-threshold", type=float, default=0.001)
    parser.add_argument("--rlt-downsample-size", type=int, default=32)
    parser.add_argument("--chunk-gamma", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--high-confidence", type=float, default=0.9)
    parser.add_argument("--mid-confidence", type=float, default=0.75)
    parser.add_argument("--verification-rule", choices=["js", "specdec_original"], default="js")
    parser.add_argument("--js-threshold", type=float, default=0.4)
    parser.add_argument(
        "--output-jsonl", default="edge/exp/results/video_results.jsonl",
        help="Append each completed video result to this JSONL file",
    )
    args = parser.parse_args()

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
        chunk_gamma=args.chunk_gamma, max_new_tokens=args.max_new_tokens,
        high_conf_threshold=args.high_confidence, mid_conf_threshold=args.mid_confidence,
        verification_rule=args.verification_rule, js_threshold=args.js_threshold,
    )
    try:
        result = VideoSpeculativeEdgeRole(backend, channel, config, args.model_family).process_task(
            args.task_id, args.video, args.prompt,
        )
        result["run"] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "video_path": str(Path(args.video).resolve()),
            "prompt": args.prompt,
            "model_family": args.model_family,
            "draft_model_path": str(Path(args.draft_model_path).resolve()) if args.draft_model_path else None,
            "server_url": args.server_url,
            "bandwidth_MBps": args.bandwidth_mbps,
            "base_latency_s": args.base_latency_s,
            "max_frames": args.max_frames,
            "chunk_gamma": args.chunk_gamma,
            "max_new_tokens": args.max_new_tokens,
            "top_k": args.top_k,
            "rlt_diff_threshold": args.rlt_diff_threshold,
            "rlt_downsample_size": args.rlt_downsample_size,
            "verification_rule": args.verification_rule,
            "js_threshold": args.js_threshold,
        }
        append_jsonl(args.output_jsonl, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"[Result] Appended to {args.output_jsonl}")
    finally:
        channel.close()


if __name__ == "__main__":
    main()
