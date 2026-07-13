import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np

from families.video_speculative.qwen3_vl_backend import Qwen3VLDraftBackend


class Qwen3VLDraftBackendTests(unittest.TestCase):
    def test_requires_cuda_unless_cpu_is_explicit(self):
        backend = Qwen3VLDraftBackend("missing-model", device="cuda:0")
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CPU-only PyTorch"):
                backend._load()

    def test_cpu_requires_explicit_opt_in(self):
        backend = Qwen3VLDraftBackend("missing-model", device="cpu")
        with self.assertRaisesRegex(RuntimeError, "--allow-cpu"):
            backend._load()

    def test_incomplete_model_is_reported_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = Qwen3VLDraftBackend(directory, device="cpu", allow_cpu=True)
            with self.assertRaisesRegex(FileNotFoundError, "Incomplete"):
                backend._load()

    def test_generated_tokens_extend_qwen_multimodal_token_types(self):
        backend = Qwen3VLDraftBackend("unused", device="cpu", allow_cpu=True)
        backend.base_inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "mm_token_type_ids": torch.tensor([[0, 2, 0]]),
        }
        backend.generated_tokens = [4, 5]
        inputs = backend._generation_inputs()
        self.assertEqual(tuple(inputs["input_ids"].shape), (1, 5))
        self.assertEqual(tuple(inputs["attention_mask"].shape), (1, 5))
        self.assertEqual(inputs["mm_token_type_ids"].tolist(), [[0, 2, 0, 0, 0]])

    def test_rlt_removes_repeated_frames_but_keeps_changed_frame(self):
        backend = Qwen3VLDraftBackend("unused", rlt_diff_threshold=0.001)
        frames = np.stack([
            np.zeros((16, 16, 3), dtype=np.uint8),
            np.zeros((16, 16, 3), dtype=np.uint8),
            np.full((16, 16, 3), 255, dtype=np.uint8),
        ])
        compressed, indices, runs = backend._rlt_compress(frames, [0, 1, 2])
        self.assertEqual(compressed.shape[0], 2)
        self.assertEqual(indices, [0, 2])
        self.assertEqual(runs, [2, 1])


if __name__ == "__main__":
    unittest.main()
