"""Reusable video-understanding speculative decoding family."""

from .video_edge import VideoSpeculativeEdgeRole

__all__ = ["VideoSpeculativeEdgeRole"]
from .qwen3_vl_backend import Qwen3VLDraftBackend, create_qwen3_vl_draft_backend

__all__ = ["Qwen3VLDraftBackend", "create_qwen3_vl_draft_backend"]
