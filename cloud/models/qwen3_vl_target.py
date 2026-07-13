"""Qwen3-VL Cloud verifier for video-to-text speculative decoding."""

from pathlib import Path

from shared.protocol import ProtocolError
from shared.tensor_serialization import decode_tensor

from .video_target import VideoTargetBackend


class Qwen3VLTargetBackend(VideoTargetBackend):
    """Incremental verifier with prompt KV reuse and rejection rollback."""

    def __init__(self, model_path, device="cuda:0", allow_cpu=False):
        self.model_path = str(Path(model_path).resolve())
        self.device_name = device
        self.allow_cpu = bool(allow_cpu)
        self.model = None
        self.processor = None

    def _load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if self.device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Qwen3-VL Cloud requested CUDA, but CUDA is unavailable.")
        if self.device_name == "cpu" and not self.allow_cpu:
            raise RuntimeError("Qwen3-VL-8B CPU inference is disabled; explicitly allow CPU to opt in.")
        model_dir = Path(self.model_path)
        missing = [name for name in ("config.json", "tokenizer.json") if not (model_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete Qwen3-VL Cloud model directory; missing {missing}")
        if not ((model_dir / "model.safetensors").is_file() or
                (model_dir / "model.safetensors.index.json").is_file()):
            raise FileNotFoundError("Qwen3-VL Cloud model has no safetensors weights or index.")
        dtype = torch.float16 if self.device_name.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path, dtype=dtype, local_files_only=True,
        ).to(self.device_name)
        self.model.eval()

    @staticmethod
    def _sparse_js(cloud_probs, item):
        import torch

        ids = [int(value) for value in item.get("topk_ids", [])]
        q_values = [max(0.0, float(value)) for value in item.get("topk_probs", [])]
        if len(ids) != len(q_values) or not ids:
            raise ProtocolError("Candidate top-k ids/probabilities are missing or misaligned.")
        if any(token < 0 or token >= cloud_probs.numel() for token in ids):
            raise ProtocolError("Candidate contains a token outside the Cloud vocabulary.")
        p_values = [float(cloud_probs[token].item()) for token in ids]
        p = torch.tensor(p_values + [max(0.0, 1.0 - sum(p_values))], dtype=torch.float64)
        q = torch.tensor(q_values + [max(0.0, 1.0 - sum(q_values))], dtype=torch.float64)
        p = p / p.sum().clamp_min(1e-12)
        q = q / q.sum().clamp_min(1e-12)
        middle = 0.5 * (p + q)
        p_term = torch.where(p > 0, p * torch.log(p / middle.clamp_min(1e-12)), 0.0)
        q_term = torch.where(q > 0, q * torch.log(q / middle.clamp_min(1e-12)), 0.0)
        return float((0.5 * (p_term.sum() + q_term.sum())).item())

    def init_task(self, prompt, model_family, evidence, generation):
        import torch
        from PIL import Image
        from transformers.video_utils import VideoMetadata

        if model_family != "qwen3_vl":
            raise ProtocolError(f"Qwen3-VL backend cannot serve model family {model_family!r}.")
        self._load()
        if evidence.get("strategy") not in {
            "sampled_rgb_frames", "compressed_rgb_frames", "vive_rlt_zlib_frames",
        } or not evidence.get("frames"):
            raise ProtocolError("Qwen3-VL Cloud requires RGB-frame evidence.")
        frames = decode_tensor(evidence["frames"])
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ProtocolError("Video frames must have shape [frames, height, width, 3].")
        pil_frames = [Image.fromarray(frame.astype("uint8", copy=False)) for frame in frames]
        messages = [{"role": "user", "content": [
            {"type": "video"}, {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        source_fps = float(evidence.get("metadata", {}).get("source_fps") or 0.0)
        source_count = int(evidence.get("metadata", {}).get("source_frame_count") or len(pil_frames))
        indices = evidence.get("metadata", {}).get("sample_indices") or list(range(len(pil_frames)))
        metadata = VideoMetadata(
            total_num_frames=len(pil_frames), fps=source_fps or None,
            width=int(frames.shape[2]), height=int(frames.shape[1]),
            duration=(float(source_count) / source_fps) if source_fps > 0 else None,
            video_backend="pipesd", frames_indices=[int(value) for value in indices],
        )
        encoded = self.processor(
            text=text, videos=[pil_frames], video_metadata=[metadata], return_tensors="pt",
        )
        base_inputs = {
            key: value.to(self.device_name) if isinstance(value, torch.Tensor) else value
            for key, value in encoded.items()
        }
        with torch.inference_mode():
            outputs = self.model(**base_inputs, use_cache=True)
        prompt_length = int(base_inputs["input_ids"].shape[1])
        state = {
            "past_key_values": outputs.past_key_values,
            "next_logits": outputs.logits[0, -1].float(),
            "prompt_length": prompt_length,
            "tokens": [],
            "generation": dict(generation or {}),
            "cache_rollbacks": 0,
            "cache_reused_tokens": 0,
        }
        # Protocol cache positions count generated text tokens, not the visual
        # prompt positions internal to the model cache.
        return state, 0

    def _advance(self, state, token_ids):
        import torch

        token_ids = [int(token) for token in token_ids]
        if not token_ids:
            return None
        previous_length = state["prompt_length"] + len(state["tokens"])
        ids = torch.tensor([token_ids], dtype=torch.long, device=self.device_name)
        attention_mask = torch.ones(
            (1, previous_length + len(token_ids)), dtype=torch.long, device=self.device_name,
        )
        cache_position = torch.arange(
            previous_length, previous_length + len(token_ids), device=self.device_name,
        )
        core_model = getattr(self.model, "model", None)
        rope_deltas = getattr(core_model, "rope_deltas", None)
        position_ids = cache_position.view(1, 1, -1).expand(3, 1, -1)
        if rope_deltas is not None:
            position_ids = position_ids + rope_deltas.to(
                device=self.device_name, dtype=position_ids.dtype,
            ).view(1, -1, 1)
        with torch.inference_mode():
            outputs = self.model(
                input_ids=ids, attention_mask=attention_mask,
                past_key_values=state["past_key_values"], cache_position=cache_position,
                position_ids=position_ids,
                use_cache=True,
            )
        state["past_key_values"] = outputs.past_key_values
        state["tokens"].extend(token_ids)
        state["next_logits"] = outputs.logits[0, -1].float()
        state["cache_reused_tokens"] += previous_length
        return outputs.logits[0].float()

    @staticmethod
    def _crop_cache(state, generated_length):
        target = state["prompt_length"] + int(generated_length)
        cache = state["past_key_values"]
        if hasattr(cache, "crop"):
            cache.crop(target)
        else:
            trimmed = []
            for key, value in cache:
                trimmed.append((key[:, :, :target, :], value[:, :, :target, :]))
            state["past_key_values"] = tuple(trimmed)
        state["tokens"] = state["tokens"][:generated_length]
        state["cache_rollbacks"] += 1

    def verify(self, state, proposal):
        import random

        import torch

        committed = [int(token) for token in proposal.get("committed_tokens", [])]
        if committed:
            self._advance(state, committed)
        accepted, override, divergences = 0, None, []
        rule = proposal.get("verification_rule", "js")
        threshold = float(proposal.get("js_threshold", 0.4))
        items = list(proposal.get("tokens", []))
        if not items:
            return 0, None, len(state["tokens"]), []
        candidates = [int(item["token_id"]) for item in items]
        start_generated_length = len(state["tokens"])
        first_logits = state["next_logits"]
        batch_logits = self._advance(state, candidates)
        # Distribution for candidate i is the context before candidate i:
        # cached next_logits for i=0, then batched logits at i-1.
        verify_logits = [first_logits] + [batch_logits[index - 1] for index in range(1, len(items))]
        for index, item in enumerate(items):
            token_id = int(item["token_id"])
            probs = torch.softmax(verify_logits[index], dim=-1)
            if token_id < 0 or token_id >= probs.numel():
                raise ProtocolError("Draft token is outside the Cloud vocabulary.")
            divergence = self._sparse_js(probs, item)
            divergences.append(divergence)
            if rule == "js":
                accept = divergence <= threshold
            elif rule == "specdec_original":
                q = max(float(item.get("confidence", 0.0)), 1e-12)
                accept = random.random() <= min(1.0, float(probs[token_id].item()) / q)
            else:
                raise ProtocolError(f"Unsupported video verification rule: {rule!r}")
            if accept:
                accepted += 1
            else:
                override = int(probs.argmax().item())
                break
        if override is not None:
            # The batched forward tentatively cached every candidate. Retain
            # only the accepted prefix, then append the authoritative token.
            retained = start_generated_length + accepted
            retained_next_logits = first_logits if accepted == 0 else batch_logits[accepted - 1]
            self._crop_cache(state, retained)
            state["next_logits"] = retained_next_logits
            self._advance(state, [override])
        return accepted, override, len(state["tokens"]), divergences

    def close_task(self, state):
        state.clear()

    def task_metrics(self, state):
        cache = state.get("past_key_values")
        if cache is not None and hasattr(cache, "get_seq_length"):
            model_cache_length = int(cache.get_seq_length())
        else:
            model_cache_length = state.get("prompt_length", 0) + len(state.get("tokens", []))
        return {
            "cache_reused_tokens": int(state.get("cache_reused_tokens", 0)),
            "cache_rollbacks": int(state.get("cache_rollbacks", 0)),
            "model_cache_length": model_cache_length,
        }


def create_qwen3_vl_target_backend(**kwargs):
    return Qwen3VLTargetBackend(**kwargs)
