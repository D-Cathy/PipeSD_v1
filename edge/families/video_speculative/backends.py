"""Draft backend contract and a deterministic implementation for tests."""

from abc import ABC, abstractmethod
from typing import List

from shared.video_protocol import SparseTokenDistribution, VideoEvidence


class VideoDraftBackend(ABC):
    @abstractmethod
    def initialize(self, video_path: str, prompt: str) -> VideoEvidence:
        pass

    @abstractmethod
    def draft_chunk(self, max_tokens: int) -> List[SparseTokenDistribution]:
        pass

    @abstractmethod
    def self_verify(self, chunk: List[SparseTokenDistribution]) -> bool:
        pass

    @abstractmethod
    def apply_cloud_result(self, accepted_count: int, override_token: int | None) -> None:
        pass

    def commit_tokens(self, tokens: List[int]) -> None:
        """Commit the actual accepted sequence to the local model context."""
        return None

    @abstractmethod
    def is_finished(self) -> bool:
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        pass


class MockVideoDraftBackend(VideoDraftBackend):
    def __init__(self, confidences=None):
        self.confidences = list(confidences or [0.95, 0.8, 0.3, 0.95, 0.95, 0.95])
        self.cursor = 0

    def initialize(self, video_path, prompt):
        self.cursor = 0
        return VideoEvidence(strategy="mock", metadata={"video_path": video_path})

    def draft_chunk(self, max_tokens):
        result = []
        for _ in range(max_tokens):
            if self.cursor >= len(self.confidences):
                break
            token = self.cursor + 10
            confidence = self.confidences[self.cursor]
            result.append(SparseTokenDistribution(token, [token, token + 1], [confidence, 1.0 - confidence], confidence))
            self.cursor += 1
        return result

    def self_verify(self, chunk):
        return sum(item.confidence for item in chunk) / len(chunk) >= 0.7

    def apply_cloud_result(self, accepted_count, override_token):
        return None

    def commit_tokens(self, tokens):
        return None

    def is_finished(self):
        return self.cursor >= len(self.confidences)

    def decode(self, tokens):
        return " ".join(str(token) for token in tokens)
