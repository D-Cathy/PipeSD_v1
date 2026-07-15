"""Shared configuration and CLI helpers for Edge launchers."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


@dataclass
class RunConfig:
    modality: str
    server_url: str = "http://127.0.0.1:8000"
    server_timeout_s: float = 300.0
    bandwidth_mbps: float = 0.0
    base_latency_s: float = 0.0
    draft_model_path: str = ""
    device: str = "cuda:0"
    chunk_size: int = 4
    max_new_tokens: int = 256
    output_jsonl: str = ""


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="JSON run configuration; explicit CLI values override it")
    parser.add_argument("--server-url", "--server_url", dest="server_url", default=argparse.SUPPRESS)
    parser.add_argument(
        "--server-timeout-s", "--server_timeout_s", dest="server_timeout_s",
        type=float, default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--bandwidth-mbps", "--bandwidth_MBps", dest="bandwidth_mbps",
        type=float, default=argparse.SUPPRESS,
        help="Simulated upload bandwidth in MB/s; 0 disables throttling",
    )
    parser.add_argument(
        "--base-latency-s", "--base_latency_c", dest="base_latency_s",
        type=float, default=argparse.SUPPRESS,
        help="Simulated round-trip base latency in seconds",
    )
    parser.add_argument(
        "--draft-model-path", "--draft_model_path", dest="draft_model_path",
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--device", default=argparse.SUPPRESS)
    parser.add_argument(
        "--chunk-size", "--chunk-gamma", "--gamma", dest="chunk_size",
        type=int, default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-new-tokens", "--max_generated_tokens", dest="max_new_tokens",
        type=int, default=argparse.SUPPRESS,
    )
    parser.add_argument("--output-jsonl", dest="output_jsonl", default=argparse.SUPPRESS)


def _config_defaults(path: str, modality: str) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Run config must contain a JSON object.")
    defaults: Dict[str, Any] = {}
    common = payload.get("common", {})
    section = payload.get(modality, {})
    if common:
        defaults.update(common)
    if section:
        defaults.update(section)
    # A flat file is also accepted for small experiments.
    if "common" not in payload and modality not in payload:
        defaults.update(payload)
    return {str(key).replace("-", "_"): value for key, value in defaults.items()}


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    modality: str,
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config")
    known, _ = probe.parse_known_args(argv)
    if known.config:
        parser.set_defaults(**_config_defaults(known.config, modality))
    return parser.parse_args(argv)
