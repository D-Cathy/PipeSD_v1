from abc import ABC, abstractmethod
from typing import Any
from core.config import ModelConfig

class BaseInferenceRole(ABC):
    """
    分布式推理角色基类：统领多机异构拓扑的生命周期
    """
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.model = None

    @abstractmethod
    def load_model(self):
        """所有角色（无论边缘端 CPU 还是云端 4090）都必须实现的模型加载标准动作"""
        pass

    @abstractmethod
    def process_task(self, task_id: int, input_data: Any) -> Any:
        """处理推理任务的核心生命周期接口"""
        pass