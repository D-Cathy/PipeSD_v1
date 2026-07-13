"""Optional NumPy/Torch tensor conversion for MessagePack video evidence."""

from typing import Any

from .protocol import ProtocolError
from .video_protocol import BinaryTensor


def encode_tensor(value: Any, *, codec: str = "raw") -> BinaryTensor:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("NumPy is required for tensor serialization.") from exc
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(value)
    data = array.tobytes(order="C")
    if codec == "zlib":
        import zlib
        data = zlib.compress(data, level=6)
    elif codec != "raw":
        raise ValueError(f"Unsupported tensor codec: {codec}")
    return BinaryTensor(dtype=str(array.dtype), shape=list(array.shape), data=data, codec=codec)


def decode_tensor(payload: BinaryTensor):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("NumPy is required for tensor deserialization.") from exc
    if isinstance(payload, dict):
        payload = BinaryTensor(**payload)
    data = payload.data
    if payload.codec == "zlib":
        import zlib
        try:
            data = zlib.decompress(data)
        except zlib.error as exc:
            raise ProtocolError(f"Invalid zlib tensor payload: {exc}") from exc
    elif payload.codec != "raw":
        raise ProtocolError(f"Unsupported tensor codec: {payload.codec}")
    array = np.frombuffer(data, dtype=np.dtype(payload.dtype))
    expected = 1
    for size in payload.shape:
        expected *= int(size)
    if array.size != expected:
        raise ProtocolError("Tensor byte length does not match declared shape.")
    return array.reshape(payload.shape).copy()
