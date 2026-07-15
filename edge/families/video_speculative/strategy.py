"""Decision policies for video speculative decoding."""

from pipesd.runtime import Action, Decision, Strategy


class VideoConfidenceStrategy(Strategy):
    def __init__(self, high_threshold=0.9, mid_threshold=0.75):
        if not 0.0 <= mid_threshold <= high_threshold <= 1.0:
            raise ValueError("Expected 0 <= mid_threshold <= high_threshold <= 1.")
        self.high_threshold = float(high_threshold)
        self.mid_threshold = float(mid_threshold)

    def decide(self, context):
        confidence = float(context.observations.get("average_confidence", 0.0))
        if confidence >= self.high_threshold:
            return Decision(Action.ACCEPT_LOCAL, reason="high_confidence")
        if confidence >= self.mid_threshold:
            return Decision(Action.SELF_VERIFY, reason="medium_confidence")
        return Decision(Action.SEND_TO_CLOUD, reason="low_confidence")


def strategy_from_config(config):
    return VideoConfidenceStrategy(
        high_threshold=config.high_conf_threshold,
        mid_threshold=config.mid_conf_threshold,
    )
