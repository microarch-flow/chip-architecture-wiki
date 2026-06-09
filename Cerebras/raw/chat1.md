> HC2021.pdf 是Cerebras在Hot Chip 2021上的演讲材料

很好，这份 **Hot Chips 2021** 材料和前面那份 WSE-3 白皮书刚好形成互补：白皮书更偏 **core / fabric / WSE-3 架构细节**，而 HC2021 更偏 **CS-2 时代的系统级执行模型：Weight Streaming + MemoryX + SwarmX + 稀疏训练 + 集群扩展**。📌

我先给你一个总览，然后我们后续可以逐页深入。

---

# 1. 这份 HC2021 材料的核心主题

标题是 **“Multi-Million Core, Multi-Wafer AI Cluster”**，所以它讨论的重点不是单颗 WSE 的微架构，而是：

> **如何把一颗 wafer-scale chip 扩展成一个多 wafer / 多 CS-2 的 AI training cluster，同时保持类似单节点的编程体验。**

材料中的主线非常清晰：

```text
模型越来越大
   ↓
传统多 GPU 分布式训练复杂度爆炸
   ↓
Cerebras 把问题拆成三部分：
   1. MemoryX：解决模型容量
   2. WSE/CS-2：解决片上执行
   3. SwarmX：解决多 CS-2 扩展
   ↓
再叠加 unstructured sparsity 加速
```

HC2021 中给出的 WSE-2 参数是：**46,225 mm² silicon、2.6 万亿晶体管、850,000 AI optimized cores、40GB on-chip memory、20PB/s memory bandwidth、220Pb/s fabric bandwidth、TSMC 7nm**。这对应的是 CS-2 / WSE-2，而不是前面白皮书里的 CS-3 / WSE-3。

---

# 2. HC2021 与 WSE-3 白皮书的关系

可以这样理解：

| 文档             |           时代 | 重点           | 关键词                                                      |
| -------------- | -----------: | ------------ | -------------------------------------------------------- |
| Hot Chips 2021 | WSE-2 / CS-2 | 系统级训练架构      | Weight Streaming、MemoryX、SwarmX、Sparse Training          |
| WSE-3 白皮书      | WSE-3 / CS-3 | 单 wafer 架构细节 | 900k cores、5nm、48KB/core、2D mesh、24 colors、dataflow core |

两份文档不是冲突，而是层次不同。

HC2021 更多在回答：

> **“有了一颗超大 wafer 芯片之后，怎么训练 100B、1T、10T、甚至 120T 参数模型？”**

WSE-3 白皮书更多在回答：

> **“这颗 wafer 芯片内部的 core、SRAM、fabric、静态路由、dataflow trigger 到底怎么工作？”**

---

# 3. Cerebras 的系统架构：不是单芯片，而是三件套

HC2021 中最重要的系统抽象是这张：

```text
MemoryX  <──weights/gradients──>  CS-2  <──dataset──> Dataset Server
             ↑
          Optimizer Compute
```

在第 14 页，Cerebras 明确说 Weight Streaming Execution Model 的特点是：

* weights stored externally off-wafer；
* weights streamed onto wafer to compute layer；
* activations only are resident on wafer；
* execute one layer at a time；
* gradients streamed out of wafer；
* weight update occurs in MemoryX。

这说明 **CS-2/WSE 不是一个完整训练系统的全部**，而是系统里的 **compute fabric**。

更准确地说：

```text
MemoryX = 参数存储 + optimizer/update compute
CS-2/WSE = activation resident + layer compute
SwarmX = 多 CS-2 的 weight broadcast / gradient reduction fabric
Dataset Server = 输入样本供给
```

这个系统分工非常关键。

---

# 4. Weight Streaming：Cerebras 最核心的执行模型

HC2021 的第 8、14、15、16 页反复讲 Weight Streaming。它不是一个小优化，而是整个 Cerebras 系统的基础执行模型。

## 4.1 它解决什么问题？

Cerebras 认为极大模型有两个容量问题：

```text
1. How do you store the giant model?
2. How do you run that giant model on a chip?
```

第 18 页直接这样提出问题。第 19 页则说 MemoryX 可以提供 **4TB–2.4PB capacity**，支持 **200B–120T weights with optimizer state**，并使用 DRAM + flash hybrid storage，还有内部 optimizer compute。

这背后的思想是：

> **模型参数容量和计算芯片容量解耦。**

也就是：

```text
传统 GPU:
  模型权重 / optimizer state / activation / gradient 都要被切到很多 GPU HBM 中

Cerebras:
  权重和 optimizer state 主要放 MemoryX
  activation 放 WSE SRAM
  权重按层流入 WSE
  梯度流回 MemoryX 做 update
```

---

## 4.2 它为什么能掩盖 latency？

很多人看到 weight streaming 第一反应会质疑：

> 权重都从外部流进来，延迟不会很高吗？

HC2021 第 16 页给出的答案是 **pipeline**：

* coarse-grained pipelining：forward / delta / gradient fully pipelined；
* fine-grained pipelining：weight update 和下一轮 forward pass overlap；
* 目标是 removing latency-sensitive communication。

所以 Cerebras 的设想不是“低延迟随机访问权重”，而是：

```text
按层顺序流式访问权重
通过训练流水线把 MemoryX 的权重流入、梯度流出、optimizer update 与 CS-2 计算重叠
```

这更像一个 **流式训练数据通路**，不是 GPU 那种 HBM resident model execution。

---

# 5. WSE 是 MatMul array：这是最重要的硬件映射心智模型

第 20 页非常关键：它直接说 **“The Wafer is the MatMul array”**。图中展示了：

* Core Array；
* Sequence 方向；
* Feature 方向；
* Weights 横向流入；
* Partial Sum Ring 纵向/环式累计；
* activation 存在 local memory；
* weights never stored；
* fits MatMuls up to size 100k × 100k；
* no overhead of splitting MatMul across multiple devices。

这张图可以转化成你的架构语言：

```text
Physical topology:
  2D wafer core array

Logical tensor layout:
  feature / hidden dimension 映射到一个方向
  sequence / batch 映射到另一个方向

Communication pattern:
  weight broadcast
  partial-sum ring reduction

Storage:
  activation resident in local SRAM
  weight streamed through fabric, not stored

Execution:
  one layer at a time
  MatMul as streamed weight × resident activation
```

这和前面 WSE-3 白皮书中 **Hidden dimension split over x-direction，Batch/Sequence split over y-direction** 的描述是同一个思想，只是 HC2021 讲得更系统级。

---

# 6. Cerebras 的“无需模型并行”到底是什么意思？

HC2021 第 36 页说：

> Workload maps to multiple CS-2s the same way as for one；
> 40GB SRAM fits enormous layers of up to 100k hidden dim；
> no complexity of partitioning individual layers or running model-parallel。

这句话容易被误解。

它不是说：

> “所有权重都放进一颗 CS-2 的片上 SRAM。”

而是说：

> **单个 layer 的 activation / partial sum / compute mapping 可以放在 WSE 上完成，不需要像 GPU 集群那样把一个 MatMul layer 再切成 tensor parallel / pipeline parallel / sequence parallel 的复杂组合。**

真正支撑它的机制是：

```text
权重容量问题：
  交给 MemoryX

单层计算问题：
  交给 WSE 的巨大 MatMul array

多系统吞吐问题：
  交给 SwarmX 做 data parallel
```

所以 Cerebras 的“简单”来自一个很强的假设：

> **只要单个 WSE 能容纳一个 layer 的 activation/partial sum 布局，就可以把模型容量问题从片上 SRAM 中移走。**

---

# 7. SwarmX：多 CS-2 扩展为什么说是 data parallel？

第 23 页展示了 SwarmX：MemoryX 通过 SwarmX 把 weights broadcast 到多个 CS-2，gradients 在回程中 reduce。材料明确说：

* data parallel training across CS-2s；
* weights are broadcast to all CS-2s；
* gradients are reduced on way back；
* multi-system scaling with the same execution model as single system；
* compute scaling independent from capacity。

这非常有意思。

传统 GPU 训练中，scale-out 可能涉及：

```text
data parallel
tensor parallel
pipeline parallel
sequence parallel
ZeRO / optimizer state sharding
activation checkpointing
通信/计算 overlap
```

Cerebras 试图把它简化成：

```text
每个 CS-2 运行同一个模型映射
不同 CS-2 处理不同 data batch
MemoryX/SwarmX 负责权重广播和梯度归约
```

也就是：

```text
多 CS-2 之间主要是 data parallel
单 CS-2 内部解决 layer execution
MemoryX 解决参数/optimizer state
```

这就是它所谓 **push-button scaling** 的来源。

---

# 8. 稀疏加速：HC2021 比白皮书讲得更激进

HC2021 第 27 页列出一些稀疏训练研究，认为有 2× 到 10×+ 的机会。第 28 页说 Cerebras 架构为 sparse compute 设计，包括：

* fine-grained dataflow cores；
* compute only for non-zero data；
* high bandwidth memory；
* high bandwidth interconnect；
* capable of accelerating dynamic and unstructured sparsity。

第 29 页提出关键等价关系：

```text
Sparse GEMM is one AXPY per non-zero weight
```

第 31 页则把 WSE 描述为 **Sparse MatMul array**：

```text
dense activations resident on wafer
sparse weight stream only contains non-zero weights
each core performs AXPY with activations, one weight at a time
```

第 32 页还给出了 GPT-3 layer 12k×12k MatMul 上，unstructured sparsity factor 与 measured speedup 接近线性的图。

这里我们要同时看到两个点：

## 正面理解

Cerebras 的确比 GPU 更自然支持：

```text
fine-grained unstructured sparsity
dynamic sparsity
weight-stream filtered nonzero compute
AXPY-level sparse execution
```

因为 GPU tensor core 通常更喜欢 dense GEMM 或结构化稀疏，而 Cerebras 的 dataflow trigger 粒度更细。

## 批判性理解

这个收益成立依赖三个条件：

```text
1. 模型真的能在目标精度下保持高非结构化稀疏率；
2. 稀疏导致的负载不均衡能被 fabric/microthread/scheduler 吃掉；
3. sparse metadata / routing / reduction 的 overhead 低于跳过零值带来的收益。
```

这也是我们后续可以重点讨论的地方。

---

# 9. HC2021 和白皮书之间的几个关键差异

## 差异 1：WSE-2 vs WSE-3

HC2021 是 WSE-2：

```text
7nm
850,000 cores
40GB SRAM
20PB/s memory BW
220Pb/s fabric BW
2.6T transistors
```

白皮书是 WSE-3：

```text
5nm
900,000 cores
>4T transistors
21PB/s memory BW
214Pb/s fabric BW
8-wide FP16 SIMD per core
```

注意一个细节：WSE-3 的 fabric bandwidth 写的是 **214 Pb/s**，HC2021 的 WSE-2 是 **220 Pb/s**。这不一定代表 WSE-3 fabric 变弱，可能和统计口径、可用路由、实现频率、公开材料取整有关；我们不能仅凭这两个数做性能结论。

---

## 差异 2：HC2021 更强调训练，白皮书更强调架构

HC2021 的核心是：

```text
training extreme-scale models
MemoryX stores weights + optimizer
gradients stream back
weight update in MemoryX
SwarmX data parallel scaling
```

白皮书虽然也讲训练，但更深入：

```text
core architecture
DSR
microthreads
static routing colors
16-bit data + 16-bit control packet
5-port router
wafer-scale fabric
```

---

## 差异 3：HC2021 的“120T 参数”是系统容量，不是片上容量

第 7、10、19、38 页都提到 **120T parameter capacity**，但这来自 MemoryX 的外部存储能力，而不是 CS-2 片上 SRAM。

所以千万不要理解成：

```text
CS-2 片上 SRAM 能装 120T 参数
```

正确理解是：

```text
MemoryX 可以存储 120T 参数级别模型和 optimizer state
CS-2 负责流式计算每一层
```

---

# 10. 从芯片架构角度看，Cerebras 的真正创新在哪里？

我认为不是单一的“芯片很大”，而是这几个设计一起成立：

## 10.1 晶圆级 many-core

```text
850k / 900k small cores
distributed local SRAM
fine-grained dataflow
```

## 10.2 片上 SRAM 替代 HBM resident execution

```text
activation resident
weight streaming
partial sum local accumulation / ring reduction
```

## 10.3 静态/半静态数据流 fabric

```text
2D mesh
broadcast / multicast
partial sum ring
static route / color
```

## 10.4 存算解耦

```text
MemoryX stores model + optimizer state
CS-2/WSE computes
SwarmX scales data parallel training
```

## 10.5 非结构化稀疏作为架构放大器

```text
sparse weights → fewer packets → fewer FMACs → less reduction traffic
```

这五个东西缺一个，Cerebras 的故事就不完整。

---

# 11. 对你做架构探索 DSL/IR 的启发

这份 HC2021 对你非常有价值，因为它逼着 DSL 不能只描述“硬件峰值算力”。

至少要能表达下面这些对象：

```text
Hardware DSL:
  - core array topology
  - local SRAM per core
  - fabric topology
  - broadcast/reduction capability
  - static route/color resources
  - external memory system: MemoryX-like disaggregated storage
  - multi-system interconnect: SwarmX-like broadcast/reduce fabric

Mapping IR:
  - tensor dimension → physical dimension mapping
  - activation residency
  - weight stream schedule
  - gradient stream-back schedule
  - partial-sum reduction pattern
  - layer-by-layer execution order

Event IR:
  - weight arrival triggers compute
  - command packet triggers reduction
  - gradient stream-out
  - optimizer update event in external memory
  - overlap relation: update vs next forward pass
```

尤其是这一点很重要：

> Cerebras 的 event 不是传统意义上“某个 core 在某个时间执行 op”，而是 **data packet / control packet arrival 触发某个 tensor instruction**。

这对你的 Event IR 设计非常有启发。

---

# 12. 我建议下一轮从这个问题开始深入

这份材料最值得先讨论的是：

> **Weight Streaming 到底是架构创新，还是把 HBM resident model 的问题转移到了外部 MemoryX 带宽/流水线？**

因为这个问题能直接牵出 Cerebras 的所有关键设计：

```text
1. 为什么 activation resident，而不是 weight resident？
2. 为什么 WSE 适合大 batch training？
3. 为什么小 batch inference decode 可能没那么自然？
4. MemoryX 带宽是不是系统瓶颈？
5. optimizer update 放 MemoryX 是优势还是限制？
6. SwarmX data parallel 为什么能 near-linear？
```

一句话总结：**HC2021 讲的是 Cerebras 的系统级训练范式：用 MemoryX 解耦模型容量，用 WSE 做 wafer-scale layer compute，用 SwarmX 做多 CS-2 data parallel，并用 weight streaming 和稀疏数据流把传统分布式训练复杂度尽量隐藏起来。**
