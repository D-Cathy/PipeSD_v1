# PipeSD 端云协同框架介绍与部署运行手册

版本：2026-07-15

## 1. 项目定位

PipeSD 当前已经从单一推测解码脚本发展为可扩展的端云协同运行框架。现阶段已经落地文本推测解码和视频理解文本生成推测解码：Edge 只加载小模型，Cloud 只加载大模型，两端通过 HTTP 协议通信，并共享 Task、Result、Node、Strategy、Engine、Channel 等公共抽象。

当前能力包括：

- 文本：DeepSeek-Coder 1.3B Draft + 6.7B Target，GGUF/llama.cpp，真实双 GPU 已跑通。
- 视频：Qwen3-VL-2B Draft + Qwen3-VL-8B Target，真实双 GPU已跑通。
- 协议：`/health`、`/init`、`/propose`、`/exit` 和 `/video/*`。
- 统一入口：`python -m edge.app.run text|video`。
- 统一 JSON Config、统一 Result JSONL、Mock 后端和网络模拟。
- 视频 Cloud KV cache 复用、拒绝回滚和 RLT + zlib 视觉帧压缩。

## 2. 总体架构

```text
Edge（本地或边缘 GPU）                 Cloud（GPU 服务器）
┌──────────────────────┐             ┌────────────────────────┐
│ Draft 小模型          │             │ Target 大模型           │
│ Task / Dataset Loader │   HTTP      │ FastAPI                 │
│ Strategy              ├────────────►│ Task Manager            │
│ Speculative Engine    │ init/       │ Token Verification      │
│ Network Channel       │ propose/    │ KV Cache / Rollback     │
│ Result Writer         │ exit        │ GPU Inference           │
└──────────────────────┘             └────────────────────────┘
```

服务器双 GPU 模拟仍然使用两个独立进程，只是运行在同一台服务器，并用 `127.0.0.1` 通信。例如 GPU 3 放小模型、GPU 6 放大模型，因此仍保留真实端云协议边界。

## 3. 四个公共核心抽象

### 3.1 Model / Node

Node 表示一个可以执行操作的计算单元。Engine 不需要知道模型位于本机、远端服务器还是其他运行时，只需要向 Node 发出操作请求。

- `BackendNode`：包装本地 Python 模型对象。
- `HTTPNode`：通过 Channel 调用远程 Cloud。
- `NodeRequest`：描述 operation、args 和 kwargs。
- `NodeCapabilities`：描述节点支持的操作。

### 3.2 Strategy

Strategy 是协同决策逻辑，也是未来最重要的扩展点。文本 Strategy 决定何时把 Draft chunk 发给 Cloud；视频 Strategy 根据高、中、低置信度选择本地接受、自验证或 Cloud 验证。

### 3.3 Engine

Engine 负责运行协同循环、维护状态并聚合 Result。当前 TextSpeculativeEngine 和 VideoSpeculativeEngine 已接入统一 `Task → Result` 接口，同时保留旧 `process_task` 调用。

### 3.4 Channel

Channel 负责节点之间怎么传。当前生产通道是 HTTP NetworkChannel，支持超时、代理绕过、带宽和时延模拟；测试可使用 InProcessChannel。

## 4. 目录结构

```text
PipeSD_runtime_test/
├── pipesd/                     # 公共 SDK 与运行时抽象
│   ├── runtime/                # Task/Result/Node/Strategy/Engine/Channel
│   ├── nodes/
│   ├── strategies/
│   ├── engines/
│   └── channels/
├── edge/
│   ├── app/                    # run.py、旧入口、配置和结果工具
│   ├── core/                   # Channel、模型适配器、指标
│   ├── families/speculative/   # 文本推测解码
│   ├── families/video_speculative/ # 视频推测解码
│   └── tests/
├── cloud/
│   ├── app/main.py             # FastAPI Cloud 服务
│   ├── core/                   # 文本/视频任务状态管理
│   └── models/                 # Target 后端
├── shared/                     # 协议、序列化、版本、张量传输
├── configs/                    # 文本/视频配置示例
└── docs/
```

## 5. 统一 Config

Config 是 PipeSD 的运行与实验配置，不是模型自身的 `config.json`。它把长命令保存成可复用 JSON，并分成 `common` 和 `text`/`video`：

```json
{
  "common": {
    "server_url": "http://127.0.0.1:8000",
    "chunk_size": 4,
    "max_new_tokens": 16,
    "bandwidth_mbps": 0,
    "base_latency_s": 0
  },
  "text": {
    "draft_model_path": "/path/to/draft.gguf",
    "data_path": "/path/to/humaneval.jsonl"
  }
}
```

优先级为：程序默认值 < Config 文件 < 命令行显式参数。优势是减少错误、统一文本和视频参数、复现实验，并为未来 Routing、Cascade、Agent、Split Inference 提供统一配置入口。

## 6. 环境与资产检查

```bash
cd /home/guoqiuyuan/PipeSD_runtime_test
python -c "import torch, transformers; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(transformers.__version__)"
python -c "import decord; from PIL import Image; print('video dependencies OK')"
nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv
```

必须选择显存足够且未被其他用户占用的 GPU。不要随意结束不属于自己的进程。Qwen3-VL-8B 在 24 GiB GPU 上需要较充足的空闲显存。

## 7. 文本推测解码部署

### 7.1 设置实际路径

```bash
export TEXT_DRAFT="/home/guoqiuyuan/PipeSD_text/edge/pre_models/deepseek-coder-1.3b-instruct.Q4_K_M.gguf"
export TEXT_TARGET="/实际路径/deepseek-coder-6.7b-instruct.Q4_K_M.gguf"
export TEXT_DATA="/实际路径/humaneval.jsonl"
test -f "$TEXT_DRAFT" && echo "Draft OK"
test -f "$TEXT_TARGET" && echo "Target OK"
test -f "$TEXT_DATA" && echo "Data OK"
```

### 7.2 终端一：GPU 6 启动 Cloud

```bash
CUDA_VISIBLE_DEVICES=6 python -m cloud.app.main \
  --target-model-path "$TEXT_TARGET" \
  --target-n-gpu-layers -1 \
  --target-ctx-size 4096 \
  --target-threads 4 \
  --host 127.0.0.1 \
  --port 8000
```

### 7.3 终端二：检查并运行 Edge

```bash
curl http://127.0.0.1:8000/health

CUDA_VISIBLE_DEVICES=3 python -m edge.app.run text \
  --draft-model-path "$TEXT_DRAFT" \
  --draft-n-gpu-layers -1 \
  --device cuda:0 \
  --server-url http://127.0.0.1:8000 \
  --server-timeout-s 1200 \
  --bandwidth-mbps 0 \
  --base-latency-s 0 \
  --data-path "$TEXT_DATA" \
  --chunk-size 4 \
  --max-new-tokens 64 \
  --start-index 0 \
  --end-index 2 \
  --output-jsonl edge/exp/results/text_results.jsonl
```

### 7.4 文本结果

```bash
tail -n 1 edge/exp/results/text_results.jsonl
tail -n 1 exp/results/humaneval_samples.jsonl
python -m json.tool exp/results/benchmark.json | tail -n 80
```

## 8. 视频推测解码部署

### 8.1 设置路径

```bash
export VIDEO_DRAFT="/home/guoqiuyuan/PipeSD_video/assets/vive/models/Qwen3-VL-2B-Instruct"
export VIDEO_TARGET="/home/guoqiuyuan/PipeSD_video/assets/vive/models/Qwen3-VL-8B-Instruct"
export VIDEO_ROOT="/home/guoqiuyuan/PipeSD_video/assets/vive/videos/VideoDetailCaption/Test_Videos"
test -f "$VIDEO_DRAFT/config.json" && echo "Draft OK"
test -f "$VIDEO_TARGET/config.json" && echo "Target OK"
```

### 8.2 终端一：启动视频 Cloud

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=6 python -m cloud.app.main \
  --mock \
  --video-target-model-path "$VIDEO_TARGET" \
  --video-device cuda:0 \
  --host 127.0.0.1 \
  --port 8000
```

`--mock` 只让未使用的文本 Target 使用 Mock；视频 Target 仍是真实 Qwen3-VL-8B。第一次 `/video/init` 时执行延迟加载。

### 8.3 终端二：运行单视频

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=4 python -m edge.app.run video \
  --task-id video-single-test \
  --input "$VIDEO_ROOT/v_0rr7iGHamw0.mp4" \
  --draft-model-path "$VIDEO_DRAFT" \
  --device cuda:0 \
  --server-url http://127.0.0.1:8000 \
  --server-timeout-s 1200 \
  --bandwidth-mbps 0 \
  --base-latency-s 0 \
  --max-frames 8 \
  --chunk-size 4 \
  --max-new-tokens 16 \
  --top-k 16 \
  --rlt-diff-threshold 0.001 \
  --rlt-downsample-size 32 \
  --verification-rule js \
  --js-threshold 0.4 \
  --output-jsonl edge/exp/results/video_results.jsonl
```

## 9. 视频批量运行与模型复用

Cloud 服务不退出时，8B Target 只加载一次。要复用 2B Draft，使用一个 JSONL 清单让多个视频在同一 Edge 进程运行：

```json
{"task_id":"video-0","video":"/path/to/a.mp4","prompt":"Please describe the video in detail."}
{"task_id":"video-1","video":"/path/to/b.mp4","prompt":"Please describe the video in detail."}
```

```bash
CUDA_VISIBLE_DEVICES=4 python -m edge.app.run video \
  --input-jsonl configs/video-tasks.local.jsonl \
  --draft-model-path "$VIDEO_DRAFT" \
  --device cuda:0 \
  --server-url http://127.0.0.1:8000 \
  --server-timeout-s 1200 \
  --max-frames 8 \
  --chunk-size 4 \
  --max-new-tokens 16 \
  --top-k 16 \
  --output-jsonl edge/exp/results/video_results.jsonl
```

控制台应只出现一次 `Loading weights`。如果 Edge 进程退出，操作系统会释放显存，下次仍需加载；跨独立命令复用需要未来增加常驻 Edge API 服务。

## 10. 统一结果结构

```json
{
  "task_id": "HumanEval/0",
  "output": "...",
  "status": "completed",
  "stop_reason": "max_tokens",
  "metrics": {},
  "metadata": {"tokens": [], "modality": "text"},
  "run": {"modality": "text", "chunk_size": 4, "max_new_tokens": 16}
}
```

视频使用相同顶层字段。文本还保留 HumanEval 官方 completion JSONL 和 `benchmark.json`。

## 11. 常见故障

### 模型路径不存在

```bash
find /home/guoqiuyuan -type f -name "*.gguf" 2>/dev/null
find /home/guoqiuyuan -type f -path "*/Qwen3-VL-8B-Instruct/config.json" 2>/dev/null
```

### 数据集进入 fallback

使用 `find` 查找真实 `humaneval.jsonl`，然后修正 `--data-path`。

### CUDA OOM

Cloud 日志中的 `GPU 0` 是当前进程的可见设备 0；如果使用了 `CUDA_VISIBLE_DEVICES=6`，它实际上对应物理 GPU 6。先运行 `nvidia-smi`，选择真正空闲的卡。帧数降低无法解决模型加载阶段只剩几十 MiB 的问题。

### Connection refused

Cloud 未启动、端口错误或进程已经退出。先执行：

```bash
curl http://127.0.0.1:8000/health
```

### 输出代码不完整

通常是 `max_new_tokens` 太小，可从 16 提高到 64 或 128。

## 12. 别人如何调用 PipeSD

命令行调用：

```bash
python -m edge.app.run text --config configs/text-local.json
python -m edge.app.run video --config configs/video-local.json
```

SDK 调用：

```python
from pipesd import Task

result = engine.run(Task(
    task_id="sample-0",
    modality="video",
    input_data="/path/to/video.mp4",
    prompt="Please describe the video in detail.",
))
print(result.output)
print(result.metrics)
```

扩展新范式时通常实现或包装 Node、编写 Strategy，并由专用 Engine 执行协同循环；Channel 和 Task/Result 可以继续复用。

## 13. 当前边界与下一步

- 当前统一的是 Edge 入口，Cloud 参数尚未完全并入同一份 Config。
- 视频 Draft 可在批量任务内复用，但尚无常驻 Edge API。
- 文本和视频 HTTP 路由仍分别为 `/init` 和 `/video/init`。
- 下一步可以增加严格配置校验、常驻 Edge 服务、Cloud Config 和任务注册表。
- 之后可以增加 RoutingEngine、CascadeEngine、AgentEngine 和 SplitInferenceEngine。

## 14. 最简检查清单

1. 确认代码目录和 Python 环境。
2. 用 `nvidia-smi` 选择两张真正空闲的 GPU。
3. 用 `test -f` 检查模型、数据和视频路径。
4. 终端一启动 Cloud 并保持运行。
5. 终端二调用 `/health`。
6. 终端二启动统一 Edge 入口。
7. 检查 Cloud 的 init/propose/exit 请求。
8. 检查 JSONL、benchmark 和 GPU 显存。
9. 完成后按 Ctrl+C 停止 Cloud，释放大模型显存。
