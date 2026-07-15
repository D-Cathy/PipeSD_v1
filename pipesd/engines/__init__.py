"""Built-in orchestration engines."""

from edge.families.speculative.speculative_edge import TextSpeculativeEngine
from edge.families.video_speculative.video_edge import VideoSpeculativeEngine

__all__ = ["TextSpeculativeEngine", "VideoSpeculativeEngine"]
