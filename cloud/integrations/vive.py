"""Callback adapter for running VIVE-compatible Cloud verification in PipeSD."""

from cloud.models.video_target import VideoTargetBackend


class ViveCloudBackend(VideoTargetBackend):
    """Bind PipeSD task management to a GPU-resident VIVE verifier.

    The callable object is responsible for Qwen3-VL/LLaVA loading and KV cache
    operations. Keeping that implementation external prevents PipeSD from
    copying VIVE's model forks and lets deployments upgrade them independently.
    """

    def __init__(self, verifier):
        self.verifier = verifier

    def init_task(self, prompt, model_family, evidence, generation):
        return self.verifier.init_task(
            prompt=prompt, model_family=model_family,
            evidence=evidence, generation=generation,
        )

    def verify(self, state, proposal):
        return self.verifier.verify_chunk(
            state=state,
            committed_tokens=proposal.get("committed_tokens", []),
            candidate_tokens=proposal.get("tokens", []),
            rule=proposal.get("verification_rule", "js"),
            js_threshold=float(proposal.get("js_threshold", 0.4)),
        )

    def close_task(self, state):
        self.verifier.close_task(state)
