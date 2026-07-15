"""Built-in decision strategies."""

from edge.families.speculative.strategy import DPStrategy
from edge.families.video_speculative.strategy import VideoConfidenceStrategy

__all__ = ["DPStrategy", "VideoConfidenceStrategy"]
