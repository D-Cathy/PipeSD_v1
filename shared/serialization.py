"""Strict MessagePack serialization helpers for the wire protocol."""

from dataclasses import asdict, is_dataclass
from typing import Any, Dict

import msgpack

from .protocol import ProtocolError, require_protocol

CONTENT_TYPE = "application/msgpack"


def pack_message(message: Any) -> bytes:
    if is_dataclass(message):
        message = asdict(message)
    if not isinstance(message, dict):
        raise TypeError("Protocol messages must be dataclasses or dictionaries.")
    return msgpack.packb(message, use_bin_type=True)


def unpack_message(data: bytes, *, validate_version: bool = True) -> Dict[str, Any]:
    try:
        payload = msgpack.unpackb(data, raw=False)
    except Exception as exc:
        raise ProtocolError(f"Invalid MessagePack payload: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("Protocol payload must be a mapping.")
    if validate_version:
        require_protocol(payload)
    return payload
