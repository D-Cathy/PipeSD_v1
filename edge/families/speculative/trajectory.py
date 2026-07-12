# families/speculative/trajectory.py
import msgpack

class DraftTrajectory:
    def __init__(self):
        self.tokens = []
        self.probs = []
            
    def append_step(self, token, prob):
        self.tokens.append(token)
        if hasattr(prob, "tolist"):
            prob = prob.tolist()
        self.probs.append(prob)
            
    def rollback(self, accept_length):
        """当云端拒绝部分词时，安全将草稿本回卷到认可长度"""
        self.tokens = self.tokens[:accept_length]
        self.probs = self.probs[:accept_length]

    def __len__(self):
        return len(self.tokens)

    def clear(self):
        self.tokens = []
        self.probs = []