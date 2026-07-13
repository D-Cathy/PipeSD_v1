"""Cloud-side interfaces for video-language token verification."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class VideoTargetBackend(ABC):
    @abstractmethod
    def init_task(self, prompt: str, model_family: str, evidence: Dict[str, Any], generation: Dict[str, Any]):
        """Build dense visual context and return backend task state plus cache position."""

    @abstractmethod
    def verify(self, state: Any, proposal: Dict[str, Any]) -> Tuple[int, int | None, int, List[float]]:
        """Return accepted count, override token, cache position and JS diagnostics."""

    def close_task(self, state: Any) -> None:
        return None

    def task_metrics(self, state: Any) -> Dict[str, Any]:
        return {}


class MockVideoTargetBackend(VideoTargetBackend):
    """Deterministic backend for protocol tests without video models or CUDA."""

    def init_task(self, prompt, model_family, evidence, generation):
        return {
            "prompt": prompt,
            "model_family": model_family,
            "evidence": evidence,
            "tokens": [],
        }, 0

    def verify(self, state, proposal):
        state["tokens"].extend(int(token) for token in proposal.get("committed_tokens", []))
        accepted = 0
        divergences = []
        override = None
        for item in proposal.get("tokens", []):
            confidence = float(item.get("confidence", 0.0))
            divergence = max(0.0, 1.0 - confidence)
            divergences.append(divergence)
            if proposal.get("verification_rule", "js") == "js" and divergence > float(proposal.get("js_threshold", 0.4)):
                override = int(item["token_id"]) + 1
                break
            state["tokens"].append(int(item["token_id"]))
            accepted += 1
        if override is not None:
            state["tokens"].append(override)
        return accepted, override, len(state["tokens"]), divergences
