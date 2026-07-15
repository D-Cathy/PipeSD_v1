# 文本与视频统一运行入口

PipeSD 保留原有两个入口，同时新增统一入口：

```bash
python -m edge.app.run text --config configs/text-speculative-dual-gpu.example.json
python -m edge.app.run video --config configs/video-speculative-dual-gpu.example.json
```

命令行参数会覆盖配置文件。公共参数统一为 `--draft-model-path`、
`--server-url`、`--server-timeout-s`、`--device`、`--chunk-size`、
`--max-new-tokens`、`--bandwidth-mbps`、`--base-latency-s` 和
`--output-jsonl`。旧的下划线参数及 `--gamma`、`--chunk-gamma` 仍作为兼容别名。

## 服务器双 GPU 文本模拟

终端一使用 GPU 6 常驻 Cloud 大模型：

```bash
CUDA_VISIBLE_DEVICES=6 python -m cloud.app.main \
  --target-model-path /path/to/deepseek-coder-6.7b-instruct.Q4_K_M.gguf \
  --target-n-gpu-layers -1 --host 127.0.0.1 --port 8000
```

终端二使用 GPU 3 运行 Edge 小模型：

```bash
CUDA_VISIBLE_DEVICES=3 python -m edge.app.run text \
  --config configs/text-speculative-dual-gpu.example.json
```

设置 `CUDA_VISIBLE_DEVICES=3` 后，进程内部看到的选中设备仍是 `cuda:0`。

## 视频模型只加载一次

Cloud 服务不退出时，8B Target 模型只加载一次。要让 2B Draft 模型也复用，
用 `configs/video-tasks.example.jsonl` 的格式准备多个视频，然后执行：

```bash
CUDA_VISIBLE_DEVICES=3 python -m edge.app.run video \
  --config configs/video-speculative-dual-gpu.example.json \
  --input-jsonl /path/to/video_tasks.jsonl
```

模型在任务循环之外创建，因此清单内所有任务共享同一份 Draft 权重。退出 Python
进程后显存会被操作系统释放，下一次独立启动仍需加载；跨独立命令复用需要常驻
Edge API 服务，不应假装能够由普通命令永久持有显存。

## 结果

文本和视频都会向 `--output-jsonl` 追加任务记录，包含任务编号、输出、指标和
`run` 运行配置。文本仍额外保留 HumanEval 官方 completion JSONL 与历史 benchmark
JSON，确保既有评测流程兼容。
