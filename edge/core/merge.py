# core/merge.py
import sys
from typing import List, Tuple

def dynamic_token_scheduling_dp(
    token_compute_times: List[float],
    C: float,
    d: float,
    verbose: bool = False,
) -> Tuple[List[List[int]], float]:
    """
    使用动态规划（DP）求解最优的 Token 合并与异步发送调度计划。

    Args:
        token_compute_times: 端侧小模型生成每个 Token 所需的本地计算时间列表（秒）。
        C: 泛化信道单次数据包发射的固定网络传播与握手开销（秒，即 RTT 相关的固定底噪）。
        d: 单个 Token 数据包的纯物理网线传输耗时（体积 MB / 带宽 MBps）。
        verbose: 是否打印调试日志。

    Returns:
        A tuple (batches, min_completion_time):
            batches: 最佳的合并拆分计划列表，例如 [[0], [1, 2], [3, 4, 5]] 代表哪些词合并在一起发。
            min_completion_time: 理论预测达到的最小整体推断完成时间。
    """
    N = len(token_compute_times)
    if N == 0:
        return [], 0.0

    # 1. 计算每个 Token 在本地被连续自回归生出来时的绝对就位时间点
    T_ready = [0.0] * N
    T_ready[0] = token_compute_times[0]
    for i in range(1, N):
        T_ready[i] = T_ready[i - 1] + token_compute_times[i]

    # 2. 初始化 DP 账本：DP[i] 记录处理完前 0..i 个 Token 的最优（最小）总时间
    DP = [0.0] * N
    # P[i] 记录达成 DP[i] 最佳效益时，最后一个合并批次（Batch）的起始索引
    P = [0] * N

    for i in range(N):
        min_total_time = sys.float_info.max
        best_j = 0

        # 遍历所有可能的切分点 j，尝试将 j..i 划分为同一个网络发送批次
        for j in range(i + 1):
            prev_batch_finish_time = DP[j - 1] if j > 0 else 0.0
            data_ready_time = T_ready[i]
            # 关键：木桶效应拦截点。发送时机取决于“前一批发完”和“当前批的最后一个字本地生成完”的较晚者
            batch_start_time = max(prev_batch_finish_time, data_ready_time)
            
            batch_size = i - j + 1
            batch_duration = C + batch_size * d  # 固定网络底噪 + 线性体积传输耗时
            current_total_time = batch_start_time + batch_duration

            if current_total_time < min_total_time:
                min_total_time = current_total_time
                best_j = j

        DP[i] = min_total_time
        P[i] = best_j

    # 3. 从终点逆向回溯，重组出最完美的批次划分计划
    batches: List[List[int]] = []
    current_idx = N - 1
    while current_idx >= 0:
        batch_start_idx = P[current_idx]
        batch = list(range(batch_start_idx, current_idx + 1))
        batches.append(batch)
        current_idx = batch_start_idx - 1

    batches.reverse()
    min_completion_time = DP[N - 1]

    if verbose:
        print(f"[DP 优化器] 最佳合并批次划分: {batches}, 预测最佳推断耗时: {min_completion_time:.6f}s")

    return batches, min_completion_time