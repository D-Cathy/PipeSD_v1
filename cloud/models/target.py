"""Target-only model backends used by the Cloud service."""

from typing import Any, Dict, List

import numpy as np

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover
    Llama = None


class MockTargetBackend:
    """Deterministic backend for protocol and local network tests."""

    def init(self, tokens: List[int]) -> Dict[str, Any]:
        return {"tokens": list(tokens)}

    def verify(self, state, tokens, probs, seed):
        accepted = len(tokens)
        final_token = (tokens[-1] + 1) % 256 if tokens else 1
        state["tokens"].extend(tokens)
        state["tokens"].append(final_token)
        return accepted, final_token, len(state["tokens"])


class LlamaCppTargetBackend:
    """Serialized llama.cpp target backend with per-task state snapshots."""

    def __init__(self, model_path, ctx_size=4096, n_gpu_layers=-1, threads=1, seed=42):
        if Llama is None:
            raise RuntimeError("llama-cpp-python is required for the real Cloud backend.")
        self.seed = seed
        self.model = Llama(
            model_path=model_path, n_ctx=ctx_size, n_gpu_layers=n_gpu_layers,
            n_threads=threads, n_threads_batch=threads, logits_all=True,
            verbose=False, seed=seed,
        )

    def init(self, tokens):
        self.model.reset()
        self.model.eval(tokens)
        return {"snapshot": self.model.save_state(), "n_past": self.model.n_tokens}

    @staticmethod
    def _softmax(values):
        values = np.asarray(values, dtype=np.float64)
        values -= np.max(values, axis=-1, keepdims=True)
        exp = np.exp(values)
        return exp / exp.sum(axis=-1, keepdims=True)

    def verify(self, state, tokens, draft_probs, seed):
        self.model.load_state(state["snapshot"])
        base = self.model.n_tokens
        if tokens:
            self.model.eval(tokens)
            target_probs = self._softmax(self.model.scores[base - 1:base - 1 + len(tokens)])
        else:
            target_probs = []
        accepted = 0
        for idx, token in enumerate(tokens):
            draft = float(draft_probs[idx][token])
            target = float(target_probs[idx][token])
            if np.random.default_rng(seed + base + idx).random() < min(1.0, target / max(draft, 1e-12)):
                accepted += 1
            else:
                break
        # Restore the accepted prefix before adding the correction token. This is
        # the authoritative cache rollback, rather than merely changing n_tokens.
        self.model.load_state(state["snapshot"])
        if accepted:
            self.model.eval(tokens[:accepted])
        if accepted < len(tokens):
            delta = np.maximum(target_probs[accepted] - np.asarray(draft_probs[accepted]), 0.0)
            delta = delta / delta.sum() if delta.sum() else target_probs[accepted]
            final_token = int(np.random.default_rng(seed + base + accepted).choice(len(delta), p=delta))
        else:
            final_token = int(self.model.sample(top_k=1, top_p=1.0, temp=0.0))
        self.model.eval([final_token])
        state["snapshot"] = self.model.save_state()
        state["n_past"] = self.model.n_tokens
        return accepted, final_token, self.model.n_tokens
