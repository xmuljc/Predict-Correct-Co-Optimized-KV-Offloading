# CASS：Correction-Aware SSD Scheduling 实验任务说明

## 1. 目标

实现并验证 **Correction-Aware SSD Scheduling（CASS）**。

CASS 面向动态稀疏 Attention 的预取失误场景：

- 已命中的 KV Blocks 已经位于 GPU 或可立即使用；
- 漏预测的 KV Blocks 需要从 SSD 执行 correction read；
- CASS 在 correction read 进行期间，先计算 Ready Blocks；
- correction blocks 到达后继续计算，并通过 online softmax 合并结果。

需要回答的核心问题：

> CASS 是否能够减少 correction read 引起的 GPU 等待，并降低单步 decode latency？

---

## 2. 核心对比方案

必须实现以下三种模式。

### 2.1 Wait-All Baseline

```text
发起 correction read
→ 等待全部 correction blocks 到达
→ 对全部选中 blocks 一次性计算 Attention
```

### 2.2 Async-I/O Baseline

```text
异步发起 correction read
→ 仍然等待全部 blocks 到达
→ 一次性计算 Attention
```

用于排除“收益仅来自异步 I/O”的可能。

### 2.3 CASS

```text
识别 Ready Blocks 和 Correction Blocks
→ 同时执行：
   1. GPU 计算 Ready Blocks
   2. SSD 读取 Correction Blocks
→ Correction Blocks 到达
→ 继续计算
→ 使用 online softmax 合并
```

可额外提供 Oracle：

```text
假设所有 blocks 已提前就绪
```

作为延迟下界。

---

## 3. 实现范围

第一阶段只做可控实验，不直接修改完整 llama.cpp 推理流程。

优先实现：

```text
core/
  attention_state.*
  block_state.*
  cass_scheduler.*

harness/
  mock_ssd.*
  experiment_runner.*
  trace_recorder.*
  metrics.*

tests/
  test_online_softmax.*
  test_scheduler.*
  test_correctness.*
```

要求：

1. 每个核心模块必须有单元测试；
2. 所有调度事件必须写入 trace；
3. 每次实验必须输出 JSON metrics；
4. 固定随机种子，保证结果可复现；
5. 第一阶段允许使用 mock SSD latency；
6. 第一阶段不实现完整模型推理。

---

## 4. 正确性要求

CASS 不能直接相加两个局部 softmax 输出。

必须维护 online softmax 状态：

```text
m：当前最大 score
l：指数和
o：未归一化输出累积
```

Ready Blocks 和 Correction Blocks 分别计算后，通过 online softmax 合并。

至少验证：

```text
max_abs_error
mean_abs_error
relative_error
```

比较对象：

```text
CASS 输出 vs Wait-All 输出
```

建议：

- score 最大值和 softmax 归一化状态使用 FP32；
- 第一阶段先使用 FP32 验证；
- 后续再测试 FP16 / BF16。

---

## 5. 实验变量

至少覆盖以下参数。

### Ready Block 数量

```text
2, 4, 8, 16
```

### Correction Block 数量

```text
1, 2, 4, 8
```

### Block Size

```text
16, 32, 64, 128 tokens
```

### SSD 延迟

```text
100, 300, 500, 1000 us
```

### Correction Block 到达方式

```text
all_at_once
uniform
random
```

第一轮快速验证可以固定：

```text
block_size = 64
ready_blocks = [2, 4, 8]
correction_blocks = [1, 2, 4]
ssd_latency_us = [300, 500, 1000]
```

---

## 6. 必须记录的指标

每组实验至少记录：

```text
total_latency_us
attention_latency_us
io_latency_us
io_blocking_time_us
ready_attention_time_us
correction_attention_time_us
merge_time_us
scheduler_overhead_us
kernel_launch_count
overlap_time_us
overlap_ratio
max_abs_error
```

其中：

```text
overlap_ratio =
overlap_time_us / io_latency_us
```

加速比：

```text
speedup =
baseline_total_latency / cass_total_latency
```

---

## 7. Trace 事件

每次实验至少记录以下事件：

```text
experiment_start
selection_done
correction_io_submit
ready_attention_start
ready_attention_end
correction_block_arrive
correction_attention_start
correction_attention_end
online_merge_start
online_merge_end
experiment_end
```

每个事件包含：

```json
{
  "event": "ready_attention_start",
  "timestamp_us": 0,
  "layer": 0,
  "block_ids": [1, 2, 3]
}
```

---

## 8. 实验流程

### Phase 1：数学正确性

1. 生成固定的 Q、K、V；
2. 随机划分 Ready Blocks 和 Correction Blocks；
3. 分别运行 Wait-All 和 CASS；
4. 比较输出误差；
5. 单元测试必须通过。

### Phase 2：Mock SSD 实验

1. 使用 sleep、定时器或事件模拟 SSD 延迟；
2. 实现 Wait-All、Async-I/O 和 CASS；
3. 扫描实验变量；
4. 每组重复 50 次；
5. 输出均值、P50、P95。

### Phase 3：真实 SSD 实验

1. 使用 liburing 发起异步读取；
2. 使用 pinned memory 作为中间缓冲区；
3. 使用 CUDA stream 和 event 管理传输与计算；
4. 验证 SSD I/O 与 GPU 计算是否真实重叠；
5. 记录 GPU、CPU、SSD 资源占用。

### Phase 4：端到端集成

仅在前三个阶段验证有效后执行：

1. 接入现有 SolidAttention harness；
2. 使用真实 Block Selection Trace；
3. 对比原始调度器与 CASS；
4. 统计每 token decode latency。

---

## 9. 最关键的实验

必须优先完成以下对比：

| Ready Blocks | Correction Blocks | SSD Latency | Block Size |
|---:|---:|---:|---:|
| 8 | 1 | 300 us | 64 |
| 8 | 2 | 500 us | 64 |
| 4 | 4 | 500 us | 64 |
| 2 | 8 | 1000 us | 64 |

需要输出：

```text
Wait-All latency
Async-I/O latency
CASS latency
CASS speedup
I/O blocking reduction
scheduler overhead
correctness error
```

---

## 10. 成功标准

CASS 被认为有效，需要同时满足：

1. CASS 输出与 Wait-All 输出在浮点容差内一致；
2. CASS 的平均 latency 低于 Wait-All；
3. I/O blocking time 明显下降；
4. P95 latency 不明显恶化；
5. 调度和合并开销小于被隐藏的 I/O 时间；
6. 至少找到一组稳定有效的运行区间。

建议目标：

```text
decode latency reduction >= 10%
io blocking reduction >= 20%
```

如果没有达到目标，也必须给出：

- CASS 失效的参数范围；
- 额外开销来源；
- Ready Block 计算不足的原因；
- 是否需要 micro-batching。

---

## 11. 消融实验

至少完成以下消融：

```text
Wait-All
Async-I/O only
CASS without micro-batching
CASS with micro-batching
```

Micro-batching 可测试：

```text
batch_size = 1, 2, 4, 8 blocks
```

用于判断每个 block 单独启动 kernel 是否导致额外开销过大。

---

## 12. 输出文件

建议输出目录：

```text
outputs/
  env/
    hardware.txt
    software.txt

  traces/
    *.jsonl

  metrics/
    *.json

  logs/
    *.log

  figures/
    latency_vs_io.pdf
    speedup_vs_correction_blocks.pdf
    overlap_ratio.pdf
    overhead_breakdown.pdf

  reports/
    cass_summary.md
```

每组实验的 JSON 至少包含：

```json
{
  "mode": "cass",
  "ready_blocks": 8,
  "correction_blocks": 2,
  "block_size": 64,
  "ssd_latency_us": 500,
  "total_latency_us": 0,
  "io_blocking_time_us": 0,
  "overlap_ratio": 0,
  "speedup": 0,
  "max_abs_error": 0
}
```

---

## 13. 一键运行要求

最终提供：

```bash
bash scripts/setup.sh
bash scripts/run_unit_tests.sh
bash scripts/run_mock_experiments.sh
bash scripts/run_real_ssd_experiments.sh
bash scripts/generate_report.sh
```

其中真实 SSD 实验暂未完成时，脚本必须明确退出并提示缺失依赖，不能静默跳过。

---

## 14. 最终需要回答的问题

最终报告必须明确回答：

1. CASS 是否有效？
2. CASS 在哪些条件下有效？
3. CASS 在哪些条件下反而变慢？
4. Ready Block 计算最多能覆盖多少 correction read？
5. micro-batching 是否必要？
6. 主要额外开销来自哪里？
7. Mock SSD 结果是否能在真实 NVMe SSD 上复现？
8. CASS 是否能够带来真实端到端加速？
