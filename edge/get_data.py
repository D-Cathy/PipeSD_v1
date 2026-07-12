import os
import json
from transformers import AutoModel  # 借用一下看看能不能跑通环境

# 1. 确保 data 文件夹存在
os.makedirs("data", exist_ok=True)

try:
    from datasets import load_dataset
    print("正在从 Hugging Face 下载 OpenAI HumanEval 数据集...")
    dataset = load_dataset("openai/openai_humaneval", trust_remote_code=True)
    
    # 2. 转换为项目需要的 jsonl 格式
    output_path = "data/humaneval.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset["test"]:
            # 对齐标准 HumanEval 字段
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"🎉 成功！数据集已自动生成到: {output_path}")

except ImportError:
    # 如果没有安装 datasets 库，我们直接用官方最原始的 164 道题的硬编码保底生成
    print("由于没有 datasets 库，正在通过网络直接获取原始数据...")
    import urllib.request
    url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    import gzip
    
    response = urllib.request.urlopen(url)
    with gzip.open(response, 'rt', encoding='utf-8') as f_in:
        with open("data/humaneval.jsonl", "w", encoding="utf-8") as f_out:
            f_out.write(f_in.read())
    print("🎉 保底方案成功！原始数据集已下载并解压到: data/humaneval.jsonl")