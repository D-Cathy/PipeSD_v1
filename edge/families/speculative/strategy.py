# families/speculative/strategy.py
from pipesd.runtime import Action, CollaborationContext, Decision, Strategy, Task


class BaseStrategy(Strategy):
    def check_verify_condition(self, pending_length):
        context = CollaborationContext(
            Task("compat", "text"),
            state={"pending_length": pending_length},
        )
        return self.decide(context).action == Action.SEND_TO_CLOUD

class DPStrategy(BaseStrategy):
    """具体的策略实现：使用动态规划计算合并批次"""
    def __init__(self, spec_cfg, channel_cfg):
        self.spec_cfg = spec_cfg
        self.channel_cfg = channel_cfg

    def decide(self, context):
        # This runner sends one authoritative proposal at a time. A DP plan such
        # as [[0], [1, 2, 3, 4]] cannot be consumed by repeatedly selecting its
        # first segment (that degenerates into one-token verification forever).
        # Accumulate a full speculative window here; EOS/max-length flushes the
        # shorter tail in the runner.
        pending_length = int(context.state.get("pending_length", 0))
        flush_tail = bool(context.state.get("draft_eos") or context.state.get("reached_limit"))
        if flush_tail or pending_length >= max(1, self.spec_cfg.gamma):
            return Decision(Action.SEND_TO_CLOUD, reason="draft_window_ready")
        return Decision(Action.CONTINUE, reason="accumulate_draft")
