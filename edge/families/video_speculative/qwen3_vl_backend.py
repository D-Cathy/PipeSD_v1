"""Qwen3-VL draft backend for real Edge video inference."""

from __future__ import annotations

from pathlib import Path
from typing import List

from shared.tensor_serialization import encode_tensor
from shared.video_protocol import SparseTokenDistribution, VideoEvidence

from .backends import VideoDraftBackend


class Qwen3VLDraftBackend(VideoDraftBackend):
    """Generate speculative text tokens with a local Qwen3-VL model.

    The raw sampled frames are sent once during ``/video/init``. This is a
    correctness-first transport representation; a later optimized backend can
    replace it with VIVE-compressed projected visual tokens without changing
    the orchestration API.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        top_k: int = 16,
        max_frames: int = 16,
        allow_cpu: bool = False,
        rlt_diff_threshold: float = 0.001,
        rlt_downsample_size: int = 32,
    ):
        self.model_path = str(Path(model_path).resolve())
        self.device_name = device
        self.top_k = max(2, int(top_k))
        self.max_frames = max(1, int(max_frames))
        self.allow_cpu = bool(allow_cpu)
        self.rlt_diff_threshold = max(0.0, float(rlt_diff_threshold))
        self.rlt_downsample_size = max(8, int(rlt_downsample_size))
        self.model = None
        self.processor = None
        self.base_inputs = None
        self.generated_tokens: List[int] = []
        self.eos_token_ids = set()
        self.finished = False

    def _load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if self.device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "Qwen3-VL Edge requested CUDA, but this Python environment has "
                "CPU-only PyTorch. Install a CUDA PyTorch build or pass "
                "--device cpu --allow-cpu explicitly."
            )
        if self.device_name == "cpu" and not self.allow_cpu:
            raise RuntimeError(
                "Qwen3-VL-2B CPU inference is disabled by default because it is "
                "very slow and memory intensive. Pass --allow-cpu to opt in."
            )
        model_dir = Path(self.model_path)
        required = ("config.json", "tokenizer.json", "model.safetensors")
        missing = [name for name in required if not (model_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete Qwen3-VL model directory; missing {missing}")

        dtype = torch.float16 if self.device_name.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        if self.device_name.startswith("cuda"):
            # With CUDA_VISIBLE_DEVICES the selected physical GPU is the
            # process current device. Some PyTorch builds reject both string
            # and torch.device arguments for these memory-stat APIs.
            torch.cuda.reset_peak_memory_stats()
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path, dtype=dtype, local_files_only=True,
        ).to(self.device_name)
        self.model.eval()
        tokenizer = self.processor.tokenizer
        eos = tokenizer.eos_token_id
        self.eos_token_ids = {int(eos)} if isinstance(eos, int) else set(eos or [])

    def _sample_frames(self, video_path):
        import numpy as np

        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video does not exist: {path}")
        try:
            import decord
        except ImportError:
            decord = None
        if decord is not None:
            reader = decord.VideoReader(str(path), ctx=decord.cpu(0))
            count = len(reader)
            if count <= 0:
                raise ValueError(f"Video has no decodable frames: {path}")
            take = min(count, self.max_frames)
            indices = np.linspace(0, count - 1, num=take, dtype=np.int64)
            frames = reader.get_batch(indices).asnumpy()
            fps = float(reader.get_avg_fps() or 0.0)
            return frames, indices.tolist(), fps, count

        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "Video decoding requires either decord or opencv-python (cv2)."
            ) from exc
        capture = cv2.VideoCapture(str(path))
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if count <= 0:
                raise ValueError(f"Video has no decodable frames: {path}")
            take = min(count, self.max_frames)
            indices = np.linspace(0, count - 1, num=take, dtype=np.int64)
            frames = []
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Failed to decode video frame {int(index)} from {path}")
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return np.stack(frames), indices.tolist(), fps, count
        finally:
            capture.release()

    def initialize(self, video_path: str, prompt: str) -> VideoEvidence:
        import torch
        from PIL import Image
        from transformers.video_utils import VideoMetadata

        self._load()
        frames, indices, fps, frame_count = self._sample_frames(video_path)
        sampled_count = len(frames)
        frames, indices, run_lengths = self._rlt_compress(frames, indices)
        pil_frames = [Image.fromarray(frame) for frame in frames]
        messages = [{
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        metadata = VideoMetadata(
            total_num_frames=len(pil_frames), fps=fps or None,
            width=int(frames.shape[2]), height=int(frames.shape[1]),
            duration=(float(frame_count) / fps) if fps > 0 else None,
            video_backend="pipesd", frames_indices=indices,
        )
        inputs = self.processor(
            text=text, videos=[pil_frames], video_metadata=[metadata], return_tensors="pt",
        )
        self.base_inputs = {
            key: value.to(self.device_name) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        self.generated_tokens = []
        self.finished = False
        return VideoEvidence(
            strategy="vive_rlt_zlib_frames",
            frames=encode_tensor(frames, codec="zlib"),
            metadata={
                "source_name": Path(video_path).name,
                "sample_indices": indices,
                "sampled_frame_count": sampled_count,
                "compressed_frame_count": len(frames),
                "rlt_run_lengths": run_lengths,
                "rlt_diff_threshold": self.rlt_diff_threshold,
                "source_frame_count": frame_count,
                "source_fps": fps,
            },
        )

    def _rlt_compress(self, frames, indices):
        """VIVE-style temporal run-length compression on sampled RGB frames."""
        import numpy as np
        from PIL import Image

        if len(frames) <= 1:
            return frames, indices, [len(frames)]

        def thumbnail(frame):
            image = Image.fromarray(frame).convert("L").resize(
                (self.rlt_downsample_size, self.rlt_downsample_size), Image.Resampling.BILINEAR,
            )
            return np.asarray(image, dtype=np.float32) / 255.0

        kept_frames = [frames[0]]
        kept_indices = [int(indices[0])]
        run_lengths = [1]
        reference = thumbnail(frames[0])
        for frame, index in zip(frames[1:], indices[1:]):
            candidate = thumbnail(frame)
            mse = float(np.mean((reference - candidate) ** 2))
            if mse <= self.rlt_diff_threshold:
                run_lengths[-1] += 1
                continue
            kept_frames.append(frame)
            kept_indices.append(int(index))
            run_lengths.append(1)
            reference = candidate
        return np.stack(kept_frames), kept_indices, run_lengths

    def _generation_inputs(self):
        import torch

        inputs = dict(self.base_inputs)
        if self.generated_tokens:
            base_length = int(inputs["input_ids"].shape[1])
            suffix = torch.tensor(
                [self.generated_tokens], dtype=inputs["input_ids"].dtype,
                device=self.device_name,
            )
            # Qwen3-VL 5.x names this tensor ``mm_token_type_ids``. Extend all
            # sequence-aligned auxiliary tensors so future Transformers
            # versions cannot silently leave one shorter than input_ids.
            for key, value in list(inputs.items()):
                if key == "input_ids" or not isinstance(value, torch.Tensor):
                    continue
                if value.ndim != 2 or value.shape[0] != suffix.shape[0] or value.shape[1] != base_length:
                    continue
                fill_value = 1 if key == "attention_mask" else 0
                extension = torch.full(
                    suffix.shape, fill_value, dtype=value.dtype, device=value.device,
                )
                inputs[key] = torch.cat((value, extension), dim=1)
            inputs["input_ids"] = torch.cat((inputs["input_ids"], suffix), dim=1)
            if "attention_mask" not in inputs:
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        return inputs

    def draft_chunk(self, max_tokens: int) -> List[SparseTokenDistribution]:
        import torch

        if self.finished or max_tokens <= 0:
            return []
        inputs = self._generation_inputs()
        prompt_length = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            result = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
            )
        new_ids = result.sequences[0, prompt_length:].tolist()
        proposals = []
        for token_id, logits in zip(new_ids, result.scores):
            probs = torch.softmax(logits[0].float(), dim=-1)
            values, ids = torch.topk(probs, k=min(self.top_k, probs.numel()))
            token_id = int(token_id)
            proposals.append(SparseTokenDistribution(
                token_id=token_id,
                topk_ids=[int(value) for value in ids.tolist()],
                topk_probs=[float(value) for value in values.tolist()],
                confidence=float(probs[token_id].item()),
            ))
            if token_id in self.eos_token_ids:
                break
        return proposals

    def self_verify(self, chunk):
        return bool(chunk) and sum(item.confidence for item in chunk) / len(chunk) >= 0.7

    def apply_cloud_result(self, accepted_count, override_token):
        # The orchestrator calls this only after a Cloud-routed chunk. Locally
        # accepted chunks are synchronized through draft_chunk proposals.
        return None

    def commit_tokens(self, tokens):
        self.generated_tokens.extend(int(token) for token in tokens)
        if any(int(token) in self.eos_token_ids for token in tokens):
            self.finished = True

    def is_finished(self):
        return self.finished

    def decode(self, tokens):
        return self.processor.tokenizer.decode(tokens, skip_special_tokens=True)

    def runtime_metrics(self):
        if not self.device_name.startswith("cuda"):
            return {"edge_peak_gpu_memory_gb": 0.0}
        import torch
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        return {"edge_peak_gpu_memory_gb": float(peak)}


def create_qwen3_vl_draft_backend(**kwargs):
    return Qwen3VLDraftBackend(**kwargs)
