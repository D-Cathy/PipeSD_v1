# families/speculative/strategy.py
from abc import ABC, abstractmethod
from core.merge import dynamic_token_scheduling_dp
class BaseStrategy:
    def check_verify_condition(self, trajectory):
        raise NotImplementedError

class DPStrategy(BaseStrategy):
    """具体的策略实现：使用动态规划计算合并批次"""
    def __init__(self, spec_cfg, channel_cfg):
        self.spec_cfg = spec_cfg
        self.channel_cfg = channel_cfg

    def check_verify_condition(self, trajectory):
        # 1. 这里直接调用你 core/merge.py 的逻辑
        # 你可以根据配置计算 d 和 C
        d_val = 0.29 / self.channel_cfg.bandwidth_MBps 
        C_val = self.channel_cfg.base_latency_c
        
        # 2. 生成合并计划
        batches, _ = dynamic_token_scheduling_dp(
            token_compute_times=[0.036] * len(trajectory),
            C=C_val, d=d_val
        )
        
        # 3. 判定逻辑：如果轨迹长度达到了 DP 计算的第一个批次长度，就触发验证
        target_batch_len = len(batches[0]) if batches else 1
        return len(trajectory) >= target_batch_len