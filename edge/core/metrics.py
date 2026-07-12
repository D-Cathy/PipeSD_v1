import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional

#它就像是守在终点线的裁判，专门负责在程序运行期间收集各种性能数字，并在最后拼装成账本。
class MetricsCollector:
    def __init__(self, exp_dir: str, filename: str):
        self.exp_dir = exp_dir
        self.saved_path = os.path.join(exp_dir, filename)
        os.makedirs(self.exp_dir, exist_ok=True)
        self.reset_sample()

    def reset_sample(self):
        """每个样本（Task）开始前重置单次数据计数"""
        self.current_metrics: Dict[str, Any] = {}
        self.token_durations: List[float] = []
        self.verify_spec_lengths: List[int] = []
        self.verify_accept_lengths: List[int] = []

    def record_token_duration(self, duration: float):
        self.token_durations.append(duration)

    def record_verification(self, spec_len: int, accept_len: int):
        self.verify_spec_lengths.append(spec_len)
        self.verify_accept_lengths.append(accept_len)

    def record_custom(self, key: str, value: Any):
        self.current_metrics[key] = value

    def _compile_diagnostics(self, output_length: int) -> Dict[str, Any]:
        if not self.verify_spec_lengths:
            return {}
        
        rejected_lengths = [s - a for s, a in zip(self.verify_spec_lengths, self.verify_accept_lengths)]
        rollback_events = sum(1 for r in rejected_lengths if r > 0)
        num_verifications = len(self.verify_spec_lengths)

        return {
            'mean_verify_spec_len': float(np.mean(self.verify_spec_lengths)) if self.verify_spec_lengths else None,
            'mean_accept_len': float(np.mean(self.verify_accept_lengths)) if self.verify_accept_lengths else None,
            'rollback_events': rollback_events,
            'rollback_rate': float(rollback_events / num_verifications) if num_verifications else 0.0,
        }

    def save_sample_result(self, task_id: int, output_text: str, prefix_len: int, total_time: float, strategy: str):
        """编译当前 Task 的全部指标并追加保存到本地 JSON 账本"""
        output_length = len(output_tokens) if 'output_tokens' in locals() else 0 # 柔性适配
        avg_token_time = float(np.mean(self.token_durations)) if self.token_durations else None
        
        acc_ratio = 0.0
        if self.verify_spec_lengths:
            acc_ratio = sum(a/s for s, a in zip(self.verify_spec_lengths, self.verify_accept_lengths) if s > 0) / len(self.verify_spec_lengths)

        diagnostics = self._compile_diagnostics(output_length)

        exp_result = {
            'task_id': task_id,
            'output_length': self.current_metrics.get('output_length', 0),
            'total_time': total_time,
            'output': output_text,
            'strategy': strategy,
            'avg_token_time': avg_token_time,
            'acc_ratio': acc_ratio,
            'verify_spec_lengths': self.verify_spec_lengths,
            'verify_accept_lengths': self.verify_accept_lengths,
            'diagnostics': diagnostics
        }
        
        # 融入自定义传入的其他增量指标（如 GPU 功耗等）
        exp_result.update(self.current_metrics)

        # 追加式落盘保护
        data = []
        if os.path.exists(self.saved_path):
            with open(self.saved_path, 'r', encoding='utf-8') as f:
                try: data = json.load(f)
                except: data = []
        
        data.append(exp_result)
        with open(self.saved_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"[Core] 任务 {task_id} 指标已成功无损落盘至: {self.saved_path}")