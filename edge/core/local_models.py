"""Draft-only model adapters for the Edge host."""

import numpy as np

from core.util import softmax

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover
    Llama = None


def _sample_from_probs(probs, seed=None):
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    total = probs.sum()
    if total <= 0:
        probs = np.full(probs.shape[0], 1.0 / probs.shape[0])
    else:
        probs = probs / total
    return int(np.random.default_rng(seed).choice(probs.shape[0], p=probs))


class MockDraftModel:
    """Small deterministic draft model for HTTP smoke tests."""

    def __init__(self, vocab_size=256):
        self.counter = 0
        self.vocab_size = vocab_size

    def load_model(self):
        return None

    def start_task(self, prompt):
        self.counter = 0
        return [1]

    def sample(self):
        self.counter += 1
        token_id = self.counter % self.vocab_size
        probs = np.zeros(self.vocab_size, dtype=np.float32)
        probs[token_id] = 0.95
        return token_id, probs

    def reset_kv_cache(self, accept_len=None):
        return None

    def sync_generated_tokens(self, tokens):
        self.counter = len(tokens)

    def decode(self, tokens):
        return " ".join(str(token) for token in tokens)

    def is_eos(self, token):
        return False


class LlamaCppDraftModel:
    """llama.cpp small-model adapter used only by Edge."""

    def __init__(self, config, exp_cfg):
        if Llama is None:
            raise RuntimeError("llama-cpp-python is required unless --mock_draft is set.")
        self.config = config
        self.exp_cfg = exp_cfg
        self.model = None
        self.prefix_tokens = []

    def load_model(self):
        self.model = Llama(
            model_path=self.config.model_path,
            n_threads=self.config.threads,
            n_threads_batch=self.config.threads,
            n_gpu_layers=self.config.n_gpu_layers,
            n_ctx=self.config.ctx_size,
            logits_all=True,
            use_mmap=True,
            verbose=False,
            seed=self.exp_cfg.seed,
        )

    def start_task(self, prompt):
        if self.model is None:
            self.load_model()
        self.model.reset()
        self.prefix_tokens = self.model.tokenize(prompt.encode("utf-8"), add_bos=True)
        self.model.eval(self.prefix_tokens)
        return list(self.prefix_tokens)

    def sample(self):
        probs = softmax(self.model.scores[self.model.n_tokens - 1])
        token = int(np.argmax(probs)) if self.exp_cfg.temp == 0 else _sample_from_probs(
            probs, seed=self.exp_cfg.seed + self.model.n_tokens
        )
        self.model.eval([token])
        return token, probs

    def reset_kv_cache(self, accept_len=None):
        if accept_len is None or self.model is None:
            return
        target_n_tokens = len(self.prefix_tokens) + int(accept_len)
        if target_n_tokens < self.model.n_tokens:
            self.model.n_tokens = target_n_tokens

    def sync_generated_tokens(self, tokens):
        """Rebuild draft cache after Cloud rejects or replaces draft tokens."""
        self.model.reset()
        self.model.eval(self.prefix_tokens + list(tokens))

    def decode(self, tokens):
        return self.model.detokenize(list(tokens)).decode("utf-8", errors="replace")

    def is_eos(self, token):
        return int(token) == int(self.model.token_eos())
