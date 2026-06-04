> 继续讨论下一个问题：大 SRAM + DDR 替代 HBM 的路线，在 LLM 时代是否还成立？

我的判断是：

> **“大 SRAM + DDR 替代 HBM”这条路线在 LLM 时代不是完全失效，但它从“通用替代 HBM 的主路线”退化成了“特定场景下成立的架构选择”。对大规模云端 LLM 推理，尤其是 decode 和长上下文 serving，它很难完全替代 HBM；但对中小模型、强复用 prefill、权重分片足够细、或者端侧/低成本系统，它仍然有价值。**

Graphcore 当年的逻辑非常清楚：**SRAM 解决带宽，DDR 解决容量**。HC2021 第 19 页明确说，GPU/TPU 试图用 HBM 同时解决 bandwidth 和 capacity，而 IPU 选择用 SRAM 解决 bandwidth，用 DDR 解决 capacity；理由是 HBM 昂贵、容量有限，并会增加处理器热设计压力。

但 LLM 时代的问题是：**LLM 的“带宽问题”和“容量问题”并没有被干净地分开**。尤其 decode 阶段，权重读取、KV cache 读取、跨芯片通信和低 batch serving 纠缠在一起，导致“DDR 只负责容量、SRAM 负责带宽”的边界变得不稳定。

---

# 1. Graphcore 这条路线原本的成立条件

Graphcore 的核心假设可以抽象成：

```text
模型/数据总容量需求很大
  → 用便宜 DDR / Streaming Memory 承担

计算时的 working set 可以放进片上 SRAM
  → 用 SRAM 提供高带宽、低能耗

SRAM 足够大时
  → 可以增加数据复用窗口
  → 显著降低 DRAM bandwidth 需求
```

这不是随便说的。HC2021 第 22 页专门给了一个粗略模型，论证 **sufficient on-die SRAM collapses required DRAM bandwidth**，也就是片上 SRAM 越大，在同样 compute rate 下所需 DRAM bandwidth 越低。

从架构角度，这个思路是合理的：

```text
HBM 路线：
  用高带宽外存直接喂计算

Graphcore 路线：
  用大 SRAM 扩大 reuse window
  尽量让计算在 SRAM working set 内完成
  DDR 只做较低频率的数据补给
```

这在很多传统训练/推理 workload 上是有吸引力的，尤其是：

```text
CNN
BERT fixed-shape training
中等规模模型
稀疏/图结构中可被静态调度的数据访问
可通过 pipeline / recomputation / sharding 管理状态的训练任务
```

而且 GC200 的片上 SRAM 确实很夸张：1472 个 tile，每 tile 624 KiB，总计约 896 MiB SRAM；HC2021 同页也给出 250 TFLOP/s、62 TB/s memory bandwidth、7.8 TB/s inter-tile bandwidth 等指标。

---

# 2. 但 LLM 推理把问题拆成了 prefill 和 decode

判断这条路线是否成立，不能笼统说“LLM 推理”，必须拆成：

```text
Prefill:
  输入 prompt 一次性并行处理
  大矩阵乘 GEMM 较多
  batch/sequence 维度较大时，权重复用好
  更偏 compute-bound

Decode:
  每次生成一个 token
  batch 小时大量 GEMV / skinny GEMM
  每 token 都要读权重
  每 token 都要读 KV cache
  更偏 memory-bandwidth-bound
```

这两阶段对“大 SRAM + DDR”的态度完全不同。

---

# 3. Prefill 阶段：Graphcore 路线仍然有一定合理性

prefill 的特点是：

```text
输入 token 多
矩阵乘规模较大
权重可以被多个 token 复用
activation / intermediate 有较强局部性
```

如果 SRAM 足够大，prefill 可以从 Graphcore 思路中受益：

```text
把 activation / partial result / tensor slice 留在片上 SRAM
通过 BSP exchange 做静态数据重排
减少外部 DDR 访问
```

对于 prefill：

```text
大 SRAM 的作用很明显：
  增大 tile-local working set
  提升片上复用
  降低外存带宽压力
```

所以如果 workload 是：

```text
固定 batch
固定 sequence length
离线/准离线推理
大 prefill 占比
编译器能做静态规划
```

那么 IPU-like 的 **large SRAM + DDR** 仍然可能成立。

但问题是，云端 LLM serving 的真正痛点往往不在 prefill，而在 decode。

---

# 4. Decode 阶段：DDR 替代 HBM 非常困难

decode 的核心问题是：

```text
每生成一个 token
几乎所有层都要访问权重
还要访问历史 KV cache
```

如果 batch 很小，权重复用不足，计算形态接近 GEMV。此时最要命的是：

```text
每 token 需要搬大量权重和 KV
外存带宽直接决定 tokens/s
```

这时候 Graphcore 的“DDR 只负责容量”会遇到挑战：

```text
如果权重不能常驻 SRAM
  每 token 需要从 DDR streaming weight
  DDR bandwidth 很可能不够

如果 KV cache 不能常驻 SRAM
  长上下文 decode 每 token 都要从 DDR 读 KV
  DDR bandwidth 也可能不够
```

而 LLM 的模型权重非常大。我们做一个粗略量级判断：

```text
7B FP16 权重 ≈ 14 GB
70B FP16 权重 ≈ 140 GB
175B FP16 权重 ≈ 350 GB
```

单颗 GC200 约 896 MiB SRAM，远放不下完整模型。即使用 16 颗 IPU，也只有约：

```text
16 × 896 MiB ≈ 14 GiB SRAM
```

也就是说，16 颗 IPU 的总 SRAM 大概也就刚好接近 FP16 7B 模型权重级别，尚未考虑：

```text
code
activation
temporary buffer
KV cache
communication buffer
fragmentation
tile-local mapping overhead
```

所以对 70B 级别模型，**权重不可能全驻留在 IPU SRAM 中**，必须依赖 DDR/跨芯片分片/流式加载。

这就是关键：
Graphcore 的 SRAM 很大，但 LLM 把“很大”的尺度重新定义了。

---

# 5. KV cache 是 LLM 时代额外打破平衡的因素

Graphcore HC2021 的“SRAM collapse DRAM bandwidth”模型更多是在讨论 **model streamed from DRAM** 的 inference 简化模型。

但 LLM decode 还有一个非常关键的状态：**KV cache**。

KV cache 的特点是：

```text
随 sequence length 线性增长
随 batch size 线性增长
每层都有
decode 每步都要读历史 K/V
```

对于长上下文：

```text
KV cache 可能比 activation 大得多
甚至成为主要 memory traffic
```

这会让“SRAM 负责带宽，DDR 负责容量”的边界再次崩溃：

```text
KV cache 如果放 DDR：
  decode 每 token 读 DDR，带宽压力巨大

KV cache 如果放 SRAM：
  SRAM 容量很快被长上下文吃掉
  还会挤占权重、activation、workspace
```

尤其是 LLM serving 中还有：

```text
continuous batching
不同请求不同上下文长度
请求进出动态变化
KV cache 分配/回收/碎片
```

这和 IPU 的 BSP + 静态映射 + 静态 exchange 并不天然契合。

---

# 6. 为什么 GPU/HBM 在 LLM 时代反而更强？

从 Graphcore 视角看，HBM 是贵、容量有限、耗电。但 LLM serving 让 HBM 的价值变得非常直接：

```text
HBM 同时提供：
  高容量
  高带宽
  低于 DDR 的访问延迟/能耗
  更简单的统一外存编程模型
```

GPU/HBM 的优势不是因为 HBM 完美，而是因为 LLM decode 需要一种东西：

```text
大量模型状态和 KV cache
必须以很高带宽被反复访问
```

这个时候，HBM 的“同时解决容量和带宽”反而正中靶心。

Graphcore 说 “capacity determines what an AI can do; bandwidth just limits how fast.”
但 LLM serving 中，“how fast” 直接决定服务成本和用户体验：

```text
tokens/s
TTFT
TPOT
P99 latency
吞吐/美元
吞吐/瓦
```

所以 bandwidth 不只是“快一点慢一点”，而是商业可用性的核心。

---

# 7. 什么时候“大 SRAM + DDR”在 LLM 时代仍然成立？

我认为它在以下场景仍然成立，甚至可能很有价值。

## 7.1 中小模型 + 量化后权重可部分/全部驻留 SRAM

比如：

```text
1B ~ 7B 量化模型
INT8 / FP8 / INT4
有限上下文
batch/并发可控
```

如果权重和部分 KV 能放进聚合 SRAM，那么大 SRAM 路线就很强：

```text
DDR 只做加载/换入
decode 主要命中 SRAM
外存带宽压力显著降低
```

这对端侧、边缘服务器、小型私有部署尤其有吸引力。

---

## 7.2 Prefill-heavy workload

如果 workload 是：

```text
长 prompt 批处理
embedding / reranking / encoder-like transformer
离线生成
prefill 占比远高于 decode
```

大 SRAM 有机会通过更好的数据复用降低 DDR bandwidth。

---

## 7.3 MoE 中专家局部激活，且调度足够静态

MoE 理论上可能适合大 SRAM：

```text
每 token 只激活少数 experts
可以把热 expert / 当前 expert shard 放 SRAM
冷 expert 放 DDR
```

但这里有一个前提：

```text
routing 足够可预测
负载均衡能做好
expert dispatch/gather 不把 exchange 搞爆
```

如果是在线动态 MoE，BSP/静态 exchange 会遇到挑战。

---

## 7.4 成本优先，而非极致 tokens/s

Graphcore 的 DDR 经济性论证在成本敏感场景仍有意义。HC2021 第 20 页强调 HBM 成本高，DDR-based system 可以把节省的钱用于更多处理器。

如果目标不是最高单节点吞吐，而是：

```text
低成本部署
中等 SLA
低功耗密度
大容量但低带宽访问
```

那么 DDR + 大 SRAM 仍然有商业空间。

---

# 8. 什么时候这条路线不成立？

以下场景很难成立：

## 8.1 大模型 decode，尤其 70B+

```text
权重远大于 SRAM
KV cache 远大于 SRAM
每 token 需要大量外存读取
DDR 带宽不足
```

这种场景 HBM 很难替代。

---

## 8.2 低 batch / 低延迟 serving

小 batch decode 下权重复用差：

```text
batch 小 → GEMV → arithmetic intensity 低
```

这时外存带宽决定吞吐。DDR 的低成本并不能弥补带宽不足。

---

## 8.3 长上下文

长上下文的 KV cache 访问会持续增长：

```text
context 越长
每 token attention KV traffic 越大
```

如果 KV 在 DDR，DDR 带宽会成为 P99 latency 风险。

---

## 8.4 动态 continuous batching

云端 serving 不是固定图：

```text
请求不断进出
sequence length 不同
batch composition 动态变化
KV cache 动态分配
```

这削弱了静态编译、静态映射、BSP schedule 的优势。

---

# 9. 更本质的判断：SRAM 不是 HBM 的替代，而是 HBM 的放大器

我觉得在 LLM 时代，更准确的架构结论是：

> **大 SRAM 很重要，但它很难完全替代 HBM；更现实的是用大 SRAM 降低 HBM/DRAM traffic，提升片上复用，而不是把 HBM 彻底拿掉。**

也就是说，最稳健的路线可能不是：

```text
SRAM + DDR 替代 HBM
```

而是：

```text
HBM 提供高带宽大容量主存
SRAM 提供片上 working set / KV tile / weight tile / activation buffer
编译器最大化 SRAM reuse
```

这也是现代 GPU、TPU、很多 AI ASIC 的现实路线：

```text
HBM 作为高带宽 backing store
片上 SRAM / scratchpad 做局部复用
```

Graphcore 的极端路线是：

```text
尽量不用 HBM
用更多 SRAM + DDR + 编译器调度来绕开 HBM
```

这在 LLM decode 面前比较吃亏，因为 decode 对 backing store bandwidth 太敏感。

---

# 10. 对云端推理芯片设计的启发

如果你设计云端 LLM 推理芯片，我不建议简单复制 Graphcore 的 “No HBM” 叙事。更合理的是分层判断：

## 第一层：模型状态能否驻留？

```text
权重 + KV cache + runtime buffer
是否能放进片上 SRAM / 近端 HBM？
```

如果不能，外存 bandwidth 就是核心约束。

## 第二层：decode 的 arithmetic intensity

粗略看：

```text
decode batch 越小
算术强度越低
越依赖内存带宽
```

这时 DDR 会很危险。

## 第三层：KV cache 放在哪里？

这是 LLM 推理芯片的一级问题：

```text
KV in SRAM:
  快，但容量吃紧

KV in HBM:
  相对现实，带宽仍高

KV in DDR:
  容量大但慢，长上下文/低延迟困难
```

## 第四层：SRAM 的角色

SRAM 应该用于最有价值的地方：

```text
hot KV block
attention tile
weight tile
activation / partial sum
collective buffer
prefetch staging buffer
```

而不是幻想全部状态都靠 SRAM cover。

---

# 11. 如果用一句话评价 Graphcore 这条路线

Graphcore 的判断在 2021 年并不荒谬，它准确抓住了：

```text
数据搬运能耗高
HBM 昂贵
SRAM 近计算很有价值
未来 AI 可能稀疏/不规则
```

但 LLM 时代的发展强化了另一件事：

```text
大模型推理不是只有 working set 复用问题，
而是巨大模型状态和 KV cache 必须被高带宽持续访问的问题。
```

所以：

```text
SRAM + DDR
  适合降低成本、提升片上复用、服务中小模型/静态 workload

HBM + SRAM
  更适合大模型、高吞吐、低延迟、长上下文 LLM serving
```

---

# 12. 最终结论

我会把结论分成三档：

| 场景                                       | 大 SRAM + DDR 替代 HBM 是否成立 | 判断                                       |
| ---------------------------------------- | -----------------------: | ---------------------------------------- |
| CNN/BERT-like 静态图训练/推理                   |                      较成立 | SRAM 复用窗口有效，DDR 容量便宜                     |
| 中小 LLM / 量化模型 / prefill-heavy            |                     部分成立 | SRAM 可显著减轻 DDR 压力                        |
| 70B+ LLM decode / 长上下文 / 低 batch serving |                     很难成立 | 权重和 KV cache 高带宽访问需要 HBM 级 backing store |

**一句话总结：在 LLM 时代，大 SRAM 依然非常重要，但 DDR 很难全面替代 HBM；对云端大模型 decode 来说，SRAM 更像是降低 HBM 压力的片上复用层，而不是取代 HBM 的主存方案。**
