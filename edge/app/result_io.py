"""Common JSONL result serialization for Edge tasks."""

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path


def result_record(result, run):
    if is_dataclass(result):
        record = asdict(result)
    else:
        record = dict(result)
    record["run"] = dict(run)
    return record


def append_jsonl(path, record):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
