"""Versioned wire contract for video-understanding speculative decoding.

The speculative unit is a text token. Video evidence is initialized once and
referenced by later proposal requests, so a low-confidence chunk does not resend
the full video on every Cloud query.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

from .protocol import ProtocolError
from .version import PROTOCOL_VERSION

TaskId = Union[int, str]


@dataclass
class BinaryTensor:
    dtype: str
    shape: List[int]
    data: bytes
    codec: str = "raw"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoEvidence:
    strategy: str
    base: Optional[BinaryTensor] = None
    extra: Optional[BinaryTensor] = None
    frames: Optional[BinaryTensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoInitRequest:
    task_id: TaskId
    prompt: str
    model_family: str
    evidence: VideoEvidence
    generation: Dict[str, Any] = field(default_factory=dict)
    protocol_version: str = PROTOCOL_VERSION
    modality: str = "video_text"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SparseTokenDistribution:
    token_id: int
    topk_ids: List[int]
    topk_probs: List[float]
    confidence: float

    def validate(self) -> None:
        if len(self.topk_ids) != len(self.topk_probs):
            raise ProtocolError("topk_ids and topk_probs must have equal length.")
        if self.token_id < 0 or any(token < 0 for token in self.topk_ids):
            raise ProtocolError("Token ids must be non-negative.")


@dataclass
class VideoProposalRequest:
    task_id: TaskId
    request_id: str
    sequence_no: int
    base_revision: int
    cache_position: int
    route: str
    committed_tokens: List[int]
    tokens: List[SparseTokenDistribution]
    verification_rule: str = "js"
    js_threshold: float = 0.4
    protocol_version: str = PROTOCOL_VERSION
    modality: str = "video_text"

    def to_dict(self) -> Dict[str, Any]:
        for token in self.tokens:
            token.validate()
        return asdict(self)


@dataclass
class VideoVerificationResponse:
    task_id: TaskId
    request_id: str
    sequence_no: int
    revision: int
    accepted_count: int
    override_token: Optional[int]
    cache_position: int
    js_divergences: List[float] = field(default_factory=list)
    cloud_compute_s: float = 0.0
    cache_reused_tokens: int = 0
    cache_rollbacks: int = 0
    model_cache_length: int = 0
    protocol_version: str = PROTOCOL_VERSION
    modality: str = "video_text"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VideoVerificationResponse":
        if payload.get("error"):
            raise ProtocolError(str(payload["error"]))
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            raise ProtocolError("Video protocol version mismatch.")
        required = (
            "task_id", "request_id", "sequence_no", "revision",
            "accepted_count", "override_token", "cache_position",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ProtocolError(f"Video response is missing fields: {missing}")
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__ if key in payload})
