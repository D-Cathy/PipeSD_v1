# PipeSD 文本与视频推测解码运行手册

本文档记录当前已经验证通过的文本和视频端云推测解码启动顺序、命令和验收方法。

## 1. 总体部署关系

```text
Edge
├── 加载小模型 Draft Model
├── 读取文本数据或视频
├── 生成 draft token chunk
└── 通过 HTTP 请求 Cloud
          │
          ▼
Cloud
├── 加载大模型 Target Model
├── 管理任务状态与 KV cache
├── 验证 draft token
└── 返回接受长度和必要的覆盖 token
```

文本协议：

```text
/init  /propose  /exit  /health
```

视频协议：

```text
/video/init  /video/propose  /video/exit  /health
```

## 2. 当前验证过的模型

### 文本

```text
Edge：DeepSeek-Coder-1.3B-Instruct GGUF
Cloud：DeepSeek-Coder-6.7B-Instruct GGUF
数据：HumanEval JSONL
```

### 视频

```text
Edge：Qwen3-VL-2B-Instruct
Cloud：Qwen3-VL-8B-Instruct
任务：VideoDetailCaption / 视频描述
```

模型和数据不提交到 GitHub，运行时通过绝对路径传入。

---

# 第一部分：运行文本推测解码

文本模式推荐采用：

```text
Windows 本地电脑：1.3B Draft Model
GPU 服务器：6.7B Target Model
SSH 隧道：Windows 18001 → 服务器 8000
```

## 3. 服务器：查找文本大模型

登录 GPU 服务器后执行：

```bash
find /home/guoqiuyuan \
  -type f \
  -iname "*6.7b*.gguf" \
  2>/dev/null
```

根据实际输出设置模型路径：

```bash
export TARGET_GGUF="/home/guoqiuyuan/PipeSD_new/cloud/models/deepseek-coder-6.7b-instruct.Q4_K_M.gguf"
```

检查：

```bash
ls -lh "$TARGET_GGUF"
```

## 4. 服务器：启动真实文本 Cloud

进入上传后的重构代码：

```bash
conda activate base
cd /home/guoqiuyuan/PipeSD_runtime_test
```

如果端口 8000 上已有视频 Cloud，先在原终端按 `Ctrl+C` 停止。

启动真实 6.7B Target Model：

```bash
CUDA_VISIBLE_DEVICES=6 python -m cloud.app.main \
  --target-model-path "$TARGET_GGUF" \
  --host 127.0.0.1 \
  --port 8000
```

成功标志：

```text
Application startup complete
Uvicorn running on http://127.0.0.1:8000
```

该终端保持运行。

## 5. Windows：建立 SSH 隧道

新开 PowerShell：

```powershell
$Ssh = "$env:WINDIR\System32\OpenSSH\ssh.exe"
& $Ssh -N -L 18001:127.0.0.1:8000 guoqiuyuan@222.20.97.217
```

输入服务器密码后，窗口没有继续输出是正常现象。不要关闭该窗口。

检查隧道：

```powershell
curl.exe --noproxy "*" http://127.0.0.1:18001/health
```

预期返回类似：

```json
{
  "status": "ok",
  "protocol_version": "1.0",
  "active_tasks": 0,
  "active_video_tasks": 0
}
```

## 6. Windows：运行真实文本 Edge

另开 PowerShell：

```powershell
cd "C:\Users\11864\Speculative_decoding\github_source\PipeSD_new"
```

设置本地 1.3B 模型：

```powershell
$DraftModel = "C:\Users\11864\Speculative_decoding\github_source\PipeSD_new\edge\models\deepseek-coder-1.3b-instruct.Q4_K_M.gguf"
Test-Path $DraftModel
```

应输出 `True`。

绕过系统代理：

```powershell
$env:NO_PROXY = "127.0.0.1,localhost"
$env:no_proxy = $env:NO_PROXY
```

运行一条 HumanEval 样本：

```powershell
python edge\app\run_edge.py `
  --draft_model_path $DraftModel `
  --server_url http://127.0.0.1:18001 `
  --server_timeout_s 300 `
  --data_path "edge\data\humaneval.jsonl" `
  --gamma 4 `
  --max_generated_tokens 16 `
  --start_index_of_sample 0 `
  --end_index_of_sample 0
```

## 7. 文本结果检查

Edge 成功标志：

```text
[Data] Loaded 1 samples.
[Speculative] Draft model backend is ready.
[Speculative] Initializing target verifier
[Main] Evaluation finished
```

Cloud 应出现：

```text
POST /init 200 OK
POST /propose 200 OK
POST /exit 200 OK
```

查看指标：

```powershell
Get-Content "exp\results\benchmark.json" -Tail 30
```

查看 HumanEval completion：

```powershell
Get-Content "exp\results\humaneval_samples.jsonl" -Tail 1
```

`n_ctx_seq (2048) < n_ctx_train (16384)` 是上下文容量提示，不是错误。

## 8. 文本 Mock 测试

服务器启动 Mock Cloud：

```bash
cd /home/guoqiuyuan/PipeSD_runtime_test
python -m cloud.app.main --mock --host 127.0.0.1 --port 8000
```

Edge 使用 Mock Draft：

```powershell
python edge\app\run_edge.py `
  --mock_draft `
  --server_url http://127.0.0.1:18001 `
  --server_timeout_s 300 `
  --max_generated_tokens 8 `
  --start_index_of_sample 0 `
  --end_index_of_sample 0
```

---

# 第二部分：运行视频推测解码

当前真实模型烟雾测试是在同一台 GPU 服务器上使用不同 GPU 完成的：

```text
GPU 3：Qwen3-VL-2B Edge
GPU 6：Qwen3-VL-8B Cloud
HTTP：127.0.0.1:8000
```

该方式用于验证算法和端云协议。真正两机部署时，只需把 Edge 命令移到 Edge 主机，并将 `--server-url` 改为 Cloud 地址或 SSH 隧道地址。

## 9. 服务器：设置视频模型和数据路径

```bash
export MODEL_ROOT="/home/guoqiuyuan/PipeSD_video/assets/vive/models"
export VIDEO_ROOT="/home/guoqiuyuan/PipeSD_video/assets/vive/videos"
```

设置测试视频：

```bash
export VIDEO_PATH="$VIDEO_ROOT/VideoDetailCaption/Test_Videos/v_0rr7iGHamw0.mp4"
```

检查：

```bash
ls -lh "$MODEL_ROOT/Qwen3-VL-2B-Instruct/config.json"
ls -lh "$MODEL_ROOT/Qwen3-VL-8B-Instruct/config.json"
ls -lh "$VIDEO_PATH"
```

## 10. 服务器终端一：启动真实视频 Cloud

```bash
conda activate base
cd /home/guoqiuyuan/PipeSD_runtime_test
```

设置 8B 绝对路径：

```bash
export TARGET_MODEL_DIR="/home/guoqiuyuan/PipeSD_video/assets/vive/models/Qwen3-VL-8B-Instruct"
```

启动：

```bash
CUDA_VISIBLE_DEVICES=6 python -m cloud.app.main \
  --mock \
  --video-target-model-path "$TARGET_MODEL_DIR" \
  --video-device cuda:0 \
  --host 127.0.0.1 \
  --port 8000
```

这里的 `--mock` 只表示文本 Target 使用 Mock；因为提供了 `--video-target-model-path`，视频使用的仍然是真实 8B 模型。

## 11. 服务器终端二：启动真实视频 Edge

```bash
conda activate base
cd /home/guoqiuyuan/PipeSD_runtime_test
```

不同终端的环境变量不共享，需要重新设置：

```bash
export MODEL_ROOT="/home/guoqiuyuan/PipeSD_video/assets/vive/models"
export VIDEO_PATH="/home/guoqiuyuan/PipeSD_video/assets/vive/videos/VideoDetailCaption/Test_Videos/v_0rr7iGHamw0.mp4"
```

运行：

```bash
CUDA_VISIBLE_DEVICES=3 python edge/app/run_video_edge.py \
  --task-id "video-real-test" \
  --video "$VIDEO_PATH" \
  --draft-model-path "$MODEL_ROOT/Qwen3-VL-2B-Instruct" \
  --device cuda:0 \
  --server-url http://127.0.0.1:8000 \
  --server-timeout-s 1200 \
  --bandwidth-mbps 0 \
  --base-latency-s 0 \
  --max-frames 8 \
  --rlt-diff-threshold 0.001 \
  --rlt-downsample-size 32 \
  --chunk-gamma 4 \
  --max-new-tokens 16 \
  --top-k 16 \
  --verification-rule js \
  --js-threshold 0.4 \
  --output-jsonl edge/exp/results/video_results.jsonl
```

## 12. 视频结果检查

Edge 应输出 JSON：

```json
{
  "task_id": "video-real-test",
  "tokens": [],
  "text": "...",
  "cloud_queries": 2,
  "metrics": {
    "generated_tokens": 16,
    "accepted_lengths": [4, 4],
    "cloud_cache_reused_tokens": 2556,
    "bytes_sent": 2700000
  }
}
```

具体数值可能随模型、视频和系统负载变化。

Cloud 应出现：

```text
POST /video/init 200 OK
POST /video/propose 200 OK
POST /video/exit 200 OK
```

查看最后一条结果：

```bash
tail -n 1 edge/exp/results/video_results.jsonl
```

重点检查：

```text
generated_tokens
cloud_queries
accepted_lengths
route_counts
cloud_cache_reused_tokens
cloud_cache_rollbacks
bytes_sent
text
```

`cloud_cache_rollbacks = 0` 不代表错误。如果所有 draft chunk 都完整接受，就不需要回滚。

## 13. 视频 Mock 测试

启动 Mock Cloud：

```bash
cd /home/guoqiuyuan/PipeSD_runtime_test
python -m cloud.app.main --mock --host 127.0.0.1 --port 8000
```

不提供 `--draft-model-path` 时，视频 Edge 自动使用 Mock Draft：

```bash
python edge/app/run_video_edge.py \
  --task-id "video-mock-test" \
  --video "mock.mp4" \
  --server-url http://127.0.0.1:8000 \
  --chunk-gamma 2 \
  --max-new-tokens 8 \
  --output-jsonl edge/exp/results/video_mock_results.jsonl
```

Mock 模式不需要真实视频文件、Qwen3-VL 或 GPU。

---

# 第三部分：常见问题

## 14. `Connection refused`

检查 Cloud 是否启动：

```bash
curl http://127.0.0.1:8000/health
```

Windows 通过隧道检查：

```powershell
curl.exe --noproxy "*" http://127.0.0.1:18001/health
```

## 15. `Read timed out`

第一次加载大模型可能较慢，增大：

```text
--server_timeout_s 300
```

视频可以使用：

```text
--server-timeout-s 1200
```

## 16. `Incomplete Qwen3-VL Cloud model directory`

不要依赖另一个终端设置的变量。每个终端分别执行：

```bash
export TARGET_MODEL_DIR="/home/guoqiuyuan/PipeSD_video/assets/vive/models/Qwen3-VL-8B-Instruct"
```

检查：

```bash
ls "$TARGET_MODEL_DIR/config.json"
ls "$TARGET_MODEL_DIR/tokenizer.json"
```

## 17. Windows 请求进入 Privoxy

使用 SSH 隧道和 NO_PROXY：

```powershell
$env:NO_PROXY = "127.0.0.1,localhost"
$env:no_proxy = $env:NO_PROXY
```

健康检查使用：

```powershell
curl.exe --noproxy "*" http://127.0.0.1:18001/health
```

## 18. 端口被占用

同一端口不能同时启动文本 Cloud 和视频 Cloud。测试时先停止旧服务：

```text
Ctrl+C
```

也可以使用不同端口，例如：

```text
文本 Cloud：8000
视频 Cloud：8001
```

对应修改 Edge 的 `--server_url` 或 `--server-url`。

## 19. 推荐验收顺序

```text
1. 运行单元测试
2. 启动 Mock Cloud
3. 运行 Mock Edge
4. 运行一条真实文本样本
5. 运行一个真实视频、生成 16 token
6. 再扩大数据量
```

不要在第一次验证时直接运行完整数据集。
