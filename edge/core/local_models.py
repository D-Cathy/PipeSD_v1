import msgpack
import numpy as np

from core.util import softmax

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - exercised only when llama-cpp is absent
    Llama = None


def _normalize_probs(probs):
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    total = probs.sum()
    if total <= 0:
        return np.full(probs.shape[0], 1.0 / probs.shape[0], dtype=np.float64)
    return probs / total


def _sample_from_probs(probs, seed=None):
    probs = _normalize_probs(probs)
    rng = np.random.default_rng(seed)
    return int(rng.choice(probs.shape[0], p=probs))


class MockDraftModel:
    """Small deterministic draft model for smoke tests without model files."""

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


class LlamaCppDraftModel:
    """Draft model adapter used by the unified single-server runner."""

    def __init__(self, config, exp_cfg):
        if Llama is None:
            raise RuntimeError("llama-cpp-python is required when --mock_models is not set.")
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
        scores = self.model.scores[self.model.n_tokens - 1]
        probs = softmax(scores)
        if getattr(self.exp_cfg, "temp", 0.0) == 0:
            token = int(np.argmax(probs))
        else:
            token = _sample_from_probs(probs, seed=self.exp_cfg.seed + self.model.n_tokens)
        self.model.eval([token])
        return token, probs

    def reset_kv_cache(self, accept_len=None):
        if accept_len is None or self.model is None:
            return
        target_n_tokens = len(self.prefix_tokens) + int(accept_len)
        if target_n_tokens < self.model.n_tokens:
            self.model.n_tokens = target_n_tokens


class MockTargetVerifier:
    """Target-model verifier with the same interface as the local real verifier."""

    def __init__(self):
        self.tasks = {}

    def init_task(self, task_id, tokens):
        self.tasks[task_id] = {"n_past": len(tokens)}
        return {"init": "success", "n_past": len(tokens)}

    def verify_tokens(self, data_bytes):
        payload = msgpack.unpackb(data_bytes, raw=False)
        tokens = payload.get("tokens", [])
        task_id = payload.get("task_id")
        n_past = int(payload.get("n_past", self.tasks.get(task_id, {}).get("n_past", 0)))
        final_token = (tokens[-1] + 1) if tokens else 1
        new_n_past = n_past + len(tokens) + 1
        self.tasks.setdefault(task_id, {})["n_past"] = new_n_past
        return {
            "n_accepted": len(tokens),
            "n_speculative": len(tokens),
            "final_token": final_token,
            "n_past": new_n_past,
        }

    def exit_task(self, task_id):
        self.tasks.pop(task_id, None)
        return {"status": "exited", "task_id": task_id}


class LlamaCppTargetVerifier:
    """In-process target model verifier; replaces the old cloud HTTP service."""

    def __init__(self, model_path, ctx_size=4096, n_gpu_layers=-1, threads=1, seed=42):
        if Llama is None:
            raise RuntimeError("llama-cpp-python is required when --mock_models is not set.")
        self.seed = seed
        self.model = Llama(
            model_path=model_path,
            n_threads=threads,
            n_threads_batch=threads,
            n_gpu_layers=n_gpu_layers,
            use_mlock=False,
            verbose=False,
            logits_all=True,
            n_ctx=ctx_size,
            seed=seed,
        )
        self.tasks = {}

    def init_task(self, task_id, tokens):
        self.model.reset()
        self.model.eval(tokens)
        self.tasks[task_id] = {
            "prefix": list(tokens),
            "state": self.model.save_state(),
            "n_past": self.model.n_tokens,
        }
        return {"init": "success", "n_past": self.model.n_tokens}

    def verify_tokens(self, data_bytes):
        payload = msgpack.unpackb(data_bytes, raw=False)
        task_id = payload.get("task_id")
        task = self.tasks.get(task_id)
        if task is None:
            return {"error": f"Task {task_id} was not initialized."}

        tokens = list(payload.get("tokens", []))
        draft_probs = [np.asarray(p, dtype=np.float64) for p in payload.get("probs", [])]
        n_past = int(payload.get("n_past", task["n_past"]))

        self.model.load_state(task["state"])
        if not tokens:
            final_token = self.model.sample(top_k=1, top_p=1.0, temp=0.0)
            self.model.eval([final_token])
            task["state"] = self.model.save_state()
            task["n_past"] = self.model.n_tokens
            return {"n_accepted": 0, "n_speculative": 0, "final_token": int(final_token), "n_past": task["n_past"]}

        if self.model.n_tokens > n_past:
            self.model.n_tokens = n_past
        self.model.eval(tokens)
        target_scores = self.model.scores[n_past - 1: n_past - 1 + len(tokens)]
        target_probs = softmax(target_scores)

        n_accepted = 0
        for idx, token in enumerate(tokens):
            draft_prob = float(draft_probs[idx][token]) if idx < len(draft_probs) else 0.0
            target_prob = float(target_probs[idx][token])
            ratio = target_prob / max(draft_prob, 1e-9)
            rand_val = np.random.default_rng(self.seed + n_past + idx).random()
            if ratio >= 1.0 or rand_val < ratio:
                n_accepted += 1
            else:
                break

        self.model.n_tokens = n_past + n_accepted
        if n_accepted < len(tokens):
            diff_probs = target_probs[n_accepted] - draft_probs[n_accepted]
            final_token = _sample_from_probs(np.maximum(diff_probs, 0.0), seed=self.seed + n_past + n_accepted)
        else:
            final_token = self.model.sample(top_k=1, top_p=1.0, temp=0.0)

        self.model.eval([final_token])
        task["state"] = self.model.save_state()
        task["n_past"] = self.model.n_tokens
        return {
            "n_accepted": n_accepted,
            "n_speculative": len(tokens),
            "final_token": int(final_token),
            "n_past": task["n_past"],
        }

    def exit_task(self, task_id):
        self.tasks.pop(task_id, None)
        return {"status": "exited", "task_id": task_id}

