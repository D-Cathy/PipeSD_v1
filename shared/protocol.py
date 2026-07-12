from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

from .version import PROTOCOL_VERSION

TaskId = Union[int, str]


class ProtocolError(ValueError):
    pass


@dataclass
class InitRequest:
    task_id: TaskId
    tokens: List[int]
    protocol_version: str = PROTOCOL_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProposalRequest:
    task_id: TaskId
    request_id: str
    sequence_no: int
    base_revision: int
    n_past: int
    tokens: List[int]
    probs: List[List[float]]
    should_verify: bool = True
    index: int = 0
    protocol_version: str = PROTOCOL_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FinalizeRequest:
    task_id: TaskId
    protocol_version: str = PROTOCOL_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResponse:
    task_id: TaskId
    request_id: str
    sequence_no: int
    revision: int
    n_accepted: int
    n_speculative: int
    final_token: Optional[int]
    n_past: int
    protocol_version: str = PROTOCOL_VERSION

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VerificationResponse":
        if payload.get("error"):
            raise ProtocolError(str(payload["error"]))
        require_protocol(payload)
        required = (
            "task_id", "request_id", "sequence_no", "revision", "n_accepted",
            "n_speculative", "final_token", "n_past",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ProtocolError(f"Verification response is missing fields: {missing}")
        values = {key: payload[key] for key in required}
        values["protocol_version"] = payload["protocol_version"]
        return cls(**values)


def require_protocol(payload: Dict[str, Any]) -> None:
    actual = payload.get("protocol_version")
    if actual != PROTOCOL_VERSION:
        raise ProtocolError(
            f"Protocol mismatch: expected {PROTOCOL_VERSION!r}, received {actual!r}."
        )


def success_payload(**fields: Any) -> Dict[str, Any]:
    return {"protocol_version": PROTOCOL_VERSION, **fields}


def error_payload(message: str, **fields: Any) -> Dict[str, Any]:
    return success_payload(error=message, **fields)
