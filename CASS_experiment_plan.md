# Correction-Aware SSD Scheduling（CASS）实验设计

## 1. 实验目标

验证 CASS 是否能够在发生 correction read 时，通过重叠 SSD I/O 与 Ready Block 的 Attention 计算，降低：

- I/O Blocking Time
- 单步 Decode Latency
- 端到端推理延迟

核心判断标准：

\[
T_{CASS} < T_{Baseline}
\]

并且：

\[
	ext{被隐藏的 I/O 时间} > 	ext{额外调度与合并开销}
\]

---

## 2. 对比方案

### Baseline：Wait-All

流程：

```text
发起 correction read
        ↓
等待所有 correction blocks 到达
        ↓
统一执行 Attention
```

### CASS

流程：

```text
识别 Ready Blocks 与 Correction Blocks
        ↓
同时执行
├─ GPU：计算 Ready Blocks
└─ SSD：读取 Correction Blocks
        ↓
Correction Blocks 到达
        ↓
增量计算并通过 online softmax 合并
```

### Oracle

假设所有需要的 KV Blocks 均已提前到达 GPU，用作理论延迟下界。

---

## 3. 控制变量

Baseline 与 CASS 必须保持以下条件一致：

- 相同 Query
- 相同 KV 数据
- 相同真实 Selected Blocks
- 相同 Ready / Correction 划分
- 相同 SSD 延迟
- 相同 Block Size
- 相同 Attention Kernel
- 相同硬件与运行环境

实验中只改变调度策略。

---

## 4. 实验变量

### 4.1 Ready Block 数量

```text
2、4、8、16
```

### 4.2 Correction Block 数量

```text
1、2、4、8
```

### 4.3 SSD 读取延迟

```text
100 μs、300 μs、500 μs、1000 μs
```

### 4.4 Block Size

```text
16、32、64、128 tokens
```

### 4.5 Correction Block 到达模式

- 同时到达
- 均匀到达
- 随机到达

---

## 5. 核心指标

### 5.1 Decode Latency

单轮 decode 的完整执行时间。

### 5.2 I/O Blocking Time

GPU 因等待 correction blocks 而空闲的时间。

### 5.3 Overlap Ratio

\[
	ext{Overlap Ratio}
=
rac{	ext{被 Attention 计算覆盖的 I/O 时间}}
{	ext{总 correction I/O 时间}}
\]

### 5.4 额外开销

记录：

- kernel 启动次数
- online softmax 合并时间
- 状态保存与恢复时间
- CUDA event / stream 同步时间
- 调度器执行时间

### 5.5 正确性指标

对比 Baseline 与 CASS 输出：

\[
	ext{Max Error}
=
\max |O_{CASS}-O_{Baseline}|
\]

同时记录平均绝对误差和相对误差。

---

## 6. 实验流程

```text
生成固定 Attention Trace
        ↓
人为指定 Ready / Correction Blocks
        ↓
分别运行 Baseline、CASS、Oracle
        ↓
每组实验重复 50～100 次
        ↓
统计平均值、P50、P95
```

第一阶段使用 mock SSD latency。

第二阶段接入真实：

```text
liburing + NVMe SSD + CUDA Stream
```

---

## 7. 消融实验

### 7.1 仅启用异步 I/O

异步发起 correction read，但仍等待所有 Block 到齐后统一计算。

目的：证明收益不是单纯来自异步 I/O。

### 7.2 CASS 不使用 Micro-batching

每到达一个 correction block，就立即启动一次 Attention Kernel。

### 7.3 CASS 使用 Micro-batching

多个 correction blocks 到达后聚合计算。

目的：验证 kernel 启动与同步开销是否抵消收益。

### 7.4 移除 Online Softmax 增量合并

等待所有块到齐后统一计算。

目的：验证分阶段计算本身带来的收益。

---

## 8. 最小实验矩阵

| Ready Blocks | Correction Blocks | SSD 延迟 | Block Size |
|---:|---:|---:|---:|
| 8 | 1 | 300 μs | 64 |
| 8 | 2 | 500 μs | 64 |
| 4 | 4 | 500 μs | 64 |
| 2 | 8 | 1000 μs | 64 |

优先完成以上四组，即可初步判断 CASS 的有效区间。

---

## 9. 成功标准

CASS 被认为有效，需要满足：

1. 平均 Decode Latency 低于 Baseline
2. P95 延迟不明显恶化
3. I/O Blocking Time 明显下降
4. 输出误差处于浮点容差范围内
5. 节省时间大于额外调度与合并开销

建议目标：

```text
平均 Decode Latency 降低 ≥ 10%
I/O Blocking Time 降低 ≥ 20%
```

---

## 10. 建议绘制的图表

### 图 1：SSD 延迟与 Decode Latency

- 横轴：Correction I/O Latency
- 纵轴：End-to-End Decode Latency
- 曲线：Baseline、CASS、Oracle

### 图 2：Correction Block 数量与加速比

- 横轴：Correction Block Number
- 纵轴：Speedup

\[
	ext{Speedup}
=
rac{T_{Baseline}}{T_{CASS}}
\]

### 图 3：Ready Block 数量与 Overlap Ratio

- 横轴：Ready Block Number
- 纵轴：Overlap Ratio

### 图 4：额外开销分解

展示：

- Kernel Launch
- State Save / Restore
- Online Merge
- Synchronization
- Scheduling

---

## 11. 最终结论需要回答的问题

实验结束后，需要明确回答：

1. CASS 在什么条件下有效？
2. CASS 在什么条件下反而变慢？
3. Ready Block 需要达到多少，才能覆盖 correction read？
4. SSD 延迟多大时，CASS 收益最明显？
5. Micro-batching 是否必要？
6. 额外调度开销是否可控？
7. CASS 能否带来真实端到端加速？
