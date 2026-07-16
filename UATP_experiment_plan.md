# UATP：Uncertainty-Aware Two-Stage KV Prefetching 实验设计

## 1. 实验目标

实现并验证 **UATP（Uncertainty-Aware Two-Stage KV Prefetching）**。

UATP 面向 SolidAttention 的 KV Block 预取场景，目标是：

- 提高 Selected Block 的预测召回率；
- 减少 correction read；
- 控制错误预取带来的额外 SSD 读取；
- 根据预测不确定性动态调整预取数量；
- 最终降低 SSD blocking time 和 decode latency。

核心问题：

> 相比固定 Top-K 的历史相似度预取，UATP 能否以可控的额外读取量换取更高召回率和更少 correction read？

---

## 2. 方法设计

UATP 分为两个阶段。

### 2.1 第一阶段：高召回候选生成

综合以下信息生成候选 Block 集合：

- 最近若干 decode step 的 Selected Blocks；
- Block 最近访问频率；
- Block 连续命中次数；
- 相邻层 Selected Blocks；
- 当前层历史选择变化趋势。

输出：

```text
Candidate Blocks
```

该阶段优先保证 Recall，不要求候选集合很小。

### 2.2 第二阶段：候选重排序

为每个 KV Block 保存轻量代表信息，例如：

```text
mean key
max-pooled key
centroid
低维投影后的 block summary
```

使用当前 Query 或可提前获得的 Query 表征对候选块打分：

```text
score_i = similarity(query, block_summary_i)
```

最终根据分数、置信度和预算选择预取集合。

### 2.3 动态预取预算

不固定预取 Top-K。

根据预测不确定性动态决定预取数量：

```text
置信度高   → 减少预取数量
置信度低   → 扩大预取集合
SSD 队列拥塞 → 缩小预算
miss 代价高 → 扩大预算
```

可测试的不确定性指标：

- Top-1 与 Top-K 分数差距；
- Top-K 边界分数差距；
- 分数熵；
- 分数方差；
- 最近预测稳定度。

---

## 3. 对比方案

至少实现以下 Baseline。

### B0：No Prefetch

不做预测预取，所有 Selected Blocks 均按需读取。

### B1：Previous-Step Reuse

直接使用上一 decode step 的 Selected Blocks 作为预取结果。

### B2：History Top-K

根据最近若干步的访问频率或历史相似度，固定预取 K 个 Block。

### B3：Summary Rerank

固定候选数量，使用 Block Summary 重排序，但预取预算固定。

### UATP

两阶段候选生成 + Summary 重排序 + 动态预取预算。

---

## 4. 实验数据

优先使用真实 Qwen Selected-Block Trace。

每个 transition 至少包含：

```text
request_id
token_id
layer_id
previous_selected_blocks
current_selected_blocks
candidate_blocks
block_access_history
query_or_query_summary
```

数据划分：

```text
train / validation / test
```

若方法不需要训练，也必须按请求划分测试集，禁止同一请求同时出现在参数选择和最终测试中。

---

## 5. 核心指标

### 5.1 Block Recall

```text
Recall =
命中的真实 Selected Blocks
/
真实 Selected Blocks 总数
```

### 5.2 Block Precision

```text
Precision =
命中的真实 Selected Blocks
/
预取 Blocks 总数
```

### 5.3 Correction Rate

```text
Correction Rate =
未被预取的真实 Selected Blocks
/
真实 Selected Blocks 总数
```

### 5.4 Prefetch Amplification

```text
Prefetch Amplification =
预取 Blocks 数量
/
真实 Selected Blocks 数量
```

### 5.5 Wasted Prefetch Ratio

```text
Wasted Prefetch Ratio =
未被使用的预取 Blocks
/
预取 Blocks 总数
```

### 5.6 SSD Blocking Time

使用实测值或经过真实硬件标定的 cost model 估计：

```text
ssd_blocking_time_us
```

### 5.7 Decode Latency

最终端到端验证时记录：

```text
mean
P50
P95
P99
```

---

## 6. 主要实验

### 6.1 固定预算公平对比

保持所有方法预取 Block 数量相同，比较：

- Recall
- Precision
- Correction Rate
- SSD Blocking Time

目的：证明 UATP 的预测质量更高。

### 6.2 固定 Recall 对比

调节各方法预算，使 Recall 接近，比较：

- Prefetch Amplification
- Wasted Prefetch Ratio
- SSD 读取字节数

目的：证明 UATP 达到同样 Recall 时读取更少。

### 6.3 固定 SSD 预算对比

限制单位 token 的最大预取字节数，比较：

- Recall
- Correction Rate
- Decode Latency

目的：验证在相同 I/O 预算下的收益。

### 6.4 动态预算实验

比较：

```text
Fixed Top-K
Entropy-based Budget
Margin-based Budget
Cost-aware Budget
```

重点观察：

- 不确定场景下是否自动扩大预算；
- 确定场景下是否减少错误预取；
- 平均预算是否可控。

---

## 7. 消融实验

至少完成以下消融：

```text
History Only
History + Frequency
History + Cross-Layer
History + Summary Rerank
History + Summary Rerank + Dynamic Budget
```

还应测试：

- 不同历史窗口长度；
- 不同候选集合大小；
- 不同 Block Summary；
- 不同不确定性指标；
- 是否使用相邻层信息；
- 是否使用在线反馈更新。

---

## 8. 参数扫描

### 历史窗口

```text
1, 2, 4, 8, 16 steps
```

### 候选集合大小

```text
K, 2K, 4K, 8K
```

### 最终预取预算

```text
0.5K, K, 1.25K, 1.5K, 2K
```

### Block Summary 维度

```text
16, 32, 64, 128
```

### 不确定性阈值

根据 validation set 自动选择，不允许在 test set 上调参。

---

## 9. Cost-Aware 决策

建议定义每个 Block 的期望收益：

```text
expected_gain_i =
P(block_i needed) × miss_cost_i
-
prefetch_cost_i
```

其中：

- `P(block_i needed)`：预测概率或归一化置信度；
- `miss_cost_i`：漏读后 correction read 的预计代价；
- `prefetch_cost_i`：预取该 Block 的 SSD 读取代价。

在 SSD 预算约束下，选择期望收益最高的 Blocks。

需要分别测试：

```text
仅按概率排序
概率 × miss cost
概率 × miss cost - prefetch cost
```

---

## 10. 实验流程

### Phase 1：Trace Replay

1. 读取真实 Qwen Selected-Block Trace；
2. 实现所有 Baseline 和 UATP；
3. 扫描参数；
4. 输出逐 transition 结果；
5. 统计平均值、P50、P95；
6. 找到最佳策略及有效区间。

### Phase 2：真实 SSD 标定

1. 使用真实 NVMe 测量不同 Block 数量的读取延迟；
2. 测量随机读和批量读；
3. 建立 SSD latency lookup table；
4. 用实测表替代固定 latency 假设。

### Phase 3：真实 Qwen 集成

1. 接入在线 selector；
2. 每个 decode step 实时生成候选与置信度；
3. 执行真实预取和 correction read；
4. 比较 Baseline 与 UATP 的端到端 decode latency；
5. 统计 SSD 读取字节和 GPU 等待时间。

---

## 11. 输出格式

建议目录：

```text
experiments/uatp/
  configs/
  traces/
  metrics/
  logs/
  figures/
  reports/
```

每个 transition 输出：

```json
{
  "request_id": 0,
  "token_id": 0,
  "layer_id": 0,
  "policy": "uatp",
  "candidate_count": 0,
  "prefetch_count": 0,
  "hit_count": 0,
  "miss_count": 0,
  "recall": 0,
  "precision": 0,
  "wasted_prefetch_ratio": 0,
  "ssd_bytes": 0,
  "estimated_blocking_us": 0
}
```

---

## 12. 建议图表

### 图 1：Recall–Prefetch Amplification 曲线

横轴：

```text
Prefetch Amplification
```

纵轴：

```text
Recall
```

### 图 2：Correction Rate–SSD Read Bytes

横轴：

```text
SSD Read Bytes
```

纵轴：

```text
Correction Rate
```

### 图 3：不同不确定性下的动态预算

横轴：

```text
Prediction Uncertainty
```

纵轴：

```text
Prefetch Block Count
```

### 图 4：端到端延迟

比较：

```text
No Prefetch
History Top-K
Summary Rerank
UATP
```

---

## 13. 成功标准

UATP 被认为有效，需要至少满足：

1. 在相同预取预算下，Recall 高于 History Top-K；
2. 在相同 Recall 下，错误预取和 SSD 读取量更低；
3. Correction Rate 明显下降；
4. 动态预算不会导致平均预取量失控；
5. 真实 Qwen 推理中 SSD Blocking Time 和 Decode Latency 下降。

建议目标：

```text
Recall 提升 >= 5%
Correction Rate 降低 >= 20%
SSD Blocking Time 降低 >= 15%
Prefetch Amplification 增长 <= 20%
```

---

## 14. 最终报告必须回答

1. 哪种候选生成信号最有效？
2. Block Summary 是否带来稳定收益？
3. 哪种不确定性指标最可靠？
4. 动态预算相比固定 Top-K 是否更优？
5. Recall 提升是否以过多错误预取为代价？
6. UATP 在哪些层、上下文长度和任务类型下最有效？
7. Trace Replay 的收益能否在真实 SSD 和真实 Qwen 中复现？
8. UATP 是否真正减少了 correction read 和端到端延迟？
