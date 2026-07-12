# families/speculative/strategy.py
from abc import ABC, abstractmethod
class BaseStrategy:
    def check_verify_condition(self, pending_length):
        raise NotImplementedError

class DPStrategy(BaseStrategy):
    """具体的策略实现：使用动态规划计算合并批次"""
    def __init__(self, spec_cfg, channel_cfg):
        self.spec_cfg = spec_cfg
        self.channel_cfg = channel_cfg

    def check_verify_condition(self, pending_length):
        # This runner sends one authoritative proposal at a time. A DP plan such
        # as [[0], [1, 2, 3, 4]] cannot be consumed by repeatedly selecting its
        # first segment (that degenerates into one-token verification forever).
        # Accumulate a full speculative window here; EOS/max-length flushes the
        # shorter tail in the runner.
        return pending_length >= max(1, self.spec_cfg.gamma)
