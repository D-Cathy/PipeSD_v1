from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


#信道配置类
#专门定义边缘端与云端网络通信相关的物理参数。
@dataclass
class ChannelConfig:
    server_url: str = "http://127.0.0.1:8000"
    timeout_s: int = 10
    bandwidth_MBps: float = 2.5
    base_latency_c: float = 0.05
    use_env_proxy: bool = False

#模型配置类
#专门定义边缘端本地小模型（Llama-cpp）的启动和加载配置
@dataclass
class ModelConfig:
    model_path: str = ""  #我需要补充的C:\Users\11864\.ssh\PipeSD\edge\pre_models\deepseek-coder-1.3b-instruct-GGUF\deepseek-coder-1.3b-instruct.Q4_K_M.gguf
    n_gpu_layers: int = 0
    threads: int = 4
    ctx_size: int = 2048

#推测解码特有参数类
#封装了所有属于推测解码算法族（Speculative Decoding）独有的超参数。
@dataclass
class SpeculativeConfig:
    gamma: int = 5
    verify_strategy: str = "fixed-num"  # fixed-num, single-token, multiple-tokens, hybrid
    verify_num: int = 8
    verify_thresh_single: float = 0.94
    verify_thresh_multi: float = 0.9
    init_alpha: float = 0.01
    multiply_times: float = 0.7
    merge_policy: str = "dp"
    nomerge: bool = False

#大实验全局主控类
#控制整个自动化测试管线的运行范围和生成格式。
@dataclass
class ExperimentConfig:
    dataset: str = "humaneval"
    data_path: str = "data/humaneval.jsonl"
    algorithm: str = "vanilla"  # vanilla, edgeLLM, hsl, pipesd, cloud_only
    seed: int = 42
    start_index: int = 0
    end_index: int = 1
    max_generated_tokens: int = 512
    top_k: int = 40
    top_p: float = 0.95
    temp: float = 0.0
    result_tag: str = ""#运行时自己在命令行自己传入

    #top_k、top_p 和 temp（Temperature，温度），正是最核心、最经典的三大文本采样超参数。
    # 它们就像是三道层层把关的“滤网”，对概率分布进行修剪。
    #top_k（前 K 个词截断 只要排名前k的词）
    #top_p“把候选词按概率从大到小累加，当累加概率达到95% 时，后面的词就不要了。”
    #temp temp = 0 最高概率的那个词会被无限放大，其余词的概率几乎归零。准确呆板
    #temp > 1 整个词表的概率分布会被“烫平”，原本概率只有 1\%的冷门词可能会被推高到 10%。
    # 模型会变得创意拉满、天马行空，甚至开始胡言乱语（幻觉增多），适合写小说、拟标题。