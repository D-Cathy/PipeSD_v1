"""Narrow bridge to a separately checked-out VIVE repository.

PipeSD intentionally does not vendor VIVE's custom Transformers model sources.
This bridge imports only its visual compression function and converts the result
to the stable PipeSD wire contract.
"""

import importlib
import sys
from pathlib import Path

from shared.tensor_serialization import encode_tensor
from shared.video_protocol import VideoEvidence


class ViveBridge:
    def __init__(self, repository_path):
        self.repository_path = Path(repository_path).resolve()
        if not (self.repository_path / "decoding" / "visual_token_compress.py").exists():
            raise FileNotFoundError(f"Not a VIVE checkout: {self.repository_path}")

    def _import(self, name):
        path = str(self.repository_path)
        if path not in sys.path:
            sys.path.insert(0, path)
        return importlib.import_module(name)

    def compress_projected_tokens(
        self, projected_tokens, *, base_pool_kernel=8, extra_pool_kernel=2,
        ttm_prune_ratio=0.5, ttm_window_size=4,
    ) -> VideoEvidence:
        module = self._import("decoding.visual_token_compress")
        result = module.compress_visual_tokens(
            projected_tokens,
            base_pool_kernel=base_pool_kernel,
            extra_pool_kernel=extra_pool_kernel,
            ttm_prune_ratio=ttm_prune_ratio,
            ttm_window_size=ttm_window_size,
        )
        return VideoEvidence(
            strategy="token_adapter",
            base=encode_tensor(result["base_tokens"]),
            extra=encode_tensor(result["extra_tokens"]),
            metadata={
                "base_counts": result["base_counts"],
                "extra_counts": result["extra_counts"],
                "kept_indices": result["kept_indices"],
                "union_len": result["union_len"],
                "ttm_prune_ratio": result["ttm_prune_ratio"],
                "ttm_window_size": result["ttm_window_size"],
            },
        )
