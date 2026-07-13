import unittest
from unittest.mock import patch

import torch

from cloud.models.qwen3_vl_target import Qwen3VLTargetBackend


def distribution(*pairs, size=32):
    values = torch.zeros(size)
    for token, probability in pairs:
        values[token] = probability
    remainder = max(0.0, 1.0 - float(values.sum()))
    values[-1] += remainder
    return values


class Qwen3VLTargetBackendTests(unittest.TestCase):
    def setUp(self):
        self.backend = Qwen3VLTargetBackend("unused")

    def run_with_cloud_distribution(self, state, proposal, probs):
        logits = torch.log(probs.clamp_min(1e-12))

        def advance(current, tokens):
            current["tokens"].extend(int(token) for token in tokens)
            current["next_logits"] = logits
            return torch.stack([logits for _ in tokens])

        def crop(current, length):
            current["tokens"] = current["tokens"][:length]
            current["cache_rollbacks"] += 1

        with patch.object(self.backend, "_advance", side_effect=advance), \
                patch.object(self.backend, "_crop_cache", side_effect=crop):
            return self.backend.verify(state, proposal)

    @staticmethod
    def state(probs):
        return {
            "tokens": [], "next_logits": torch.log(probs.clamp_min(1e-12)),
            "prompt_length": 3, "past_key_values": object(),
            "cache_rollbacks": 0, "cache_reused_tokens": 0,
        }

    def test_js_accepts_matching_draft(self):
        probs = distribution((5, 0.8), (6, 0.1))
        state = self.state(probs)
        proposal = {
            "committed_tokens": [3], "verification_rule": "js", "js_threshold": 0.1,
            "tokens": [{"token_id": 5, "topk_ids": [5, 6],
                        "topk_probs": [0.8, 0.1], "confidence": 0.8}],
        }
        accepted, override, position, divergences = self.run_with_cloud_distribution(
            state, proposal, probs,
        )
        self.assertEqual((accepted, override, position), (1, None, 2))
        self.assertEqual(state["tokens"], [3, 5])
        self.assertLessEqual(divergences[0], 0.1)

    def test_js_rejects_and_uses_cloud_argmax(self):
        probs = distribution((7, 0.9), (5, 0.01))
        state = self.state(probs)
        proposal = {
            "committed_tokens": [], "verification_rule": "js", "js_threshold": 0.01,
            "tokens": [{"token_id": 5, "topk_ids": [5, 6],
                        "topk_probs": [0.9, 0.05], "confidence": 0.9}],
        }
        accepted, override, position, _ = self.run_with_cloud_distribution(state, proposal, probs)
        self.assertEqual((accepted, override, position), (0, 7, 1))
        self.assertEqual(state["tokens"], [7])
        self.assertEqual(state["cache_rollbacks"], 1)

    def test_original_rule_accepts_when_p_over_q_is_one(self):
        probs = distribution((4, 0.8))
        state = self.state(probs)
        proposal = {
            "verification_rule": "specdec_original", "tokens": [
                {"token_id": 4, "topk_ids": [4], "topk_probs": [0.5], "confidence": 0.5}
            ],
        }
        accepted, override, _, _ = self.run_with_cloud_distribution(state, proposal, probs)
        self.assertEqual((accepted, override), (1, None))

    def test_incremental_position_ids_match_only_new_tokens(self):
        class Cache:
            def get_seq_length(self):
                return 633

        class Output:
            past_key_values = Cache()
            logits = torch.zeros((1, 4, 32))

        class Model:
            def __init__(self):
                self.model = type("Core", (), {"rope_deltas": torch.tensor([[-10]])})()
                self.kwargs = None

            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return Output()

        model = Model()
        self.backend.model = model
        self.backend.device_name = "cpu"
        state = {
            "past_key_values": Cache(), "next_logits": torch.zeros(32),
            "prompt_length": 633, "tokens": [], "cache_reused_tokens": 0,
        }
        self.backend._advance(state, [1, 2, 3, 4])
        self.assertEqual(tuple(model.kwargs["position_ids"].shape), (3, 1, 4))
        self.assertEqual(model.kwargs["position_ids"][0, 0].tolist(), [623, 624, 625, 626])


if __name__ == "__main__":
    unittest.main()
