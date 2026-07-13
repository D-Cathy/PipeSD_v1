from dataclasses import dataclass


@dataclass
class VideoSpeculativeConfig:
    chunk_gamma: int = 6
    max_new_tokens: int = 256
    high_conf_threshold: float = 0.9
    mid_conf_threshold: float = 0.75
    verification_rule: str = "js"
    js_threshold: float = 0.4
    topk: int = 50
