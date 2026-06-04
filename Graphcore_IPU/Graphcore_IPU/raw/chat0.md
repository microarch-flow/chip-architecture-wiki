> 现在发给你看的是Graphcore官方的文档，我已经看过一遍了，但是由于我英文不行，有些点理解的不够深入，现在你先总结一下这个文档中的信息，然后我们再逐一讨论

可以，我们先把这份 **Graphcore IPU Programmer’s Guide** 当成“官方架构 + 编程模型入口文档”来读。它不是最底层的微架构白皮书，但它已经把 IPU 的核心设计哲学讲得很清楚：**多 tile、分布式 SRAM、显式数据搬运、BSP 执行、Poplar 静态编译调度**。

# 1. 这份文档的核心结论

一句话概括：

> **IPU 不是 GPU，也不是 TPU systolic-array 风格的单一大矩阵引擎，而是一颗由大量独立 tile 组成、依靠分布式 SRAM 和显式 exchange fabric 执行图程序的 many-core AI processor。**

它的基本抽象是：

```text
Host
  ↓ 编译 / 加载 / 控制执行
IPU Program
  ↓ lowered
多个 IPU / 多个 Tile 上的 per-tile program
  ↓ 执行模型
Compute → Sync → Exchange → Compute → ...
```

这和你之前关心的 **Tensix / dataflow / 分布式 SRAM / 编译器调度 / 确定性架构** 非常相关。

---

# 2. IPU 硬件架构：many-tile + distributed SRAM

## 2.1 IPU 由大量 tile 组成

文档中说，IPU 由很多独立处理单元组成，称为 **tiles**。以 GC200 为例，一颗 IPU 有 **1472 个 tile**。每个 tile 包含：

```text
Tile
 ├── 一个多线程处理器
 ├── 本地 SRAM
 ├── 本地代码存储
 └── exchange fabric 接口
```

这点很关键：**tile 不是简单 PE，而更像一个小型可编程处理器 + local memory。**

它和 GPU SM 的差异在于：

| 对比项  | GPU                                | Graphcore IPU                       |
| ---- | ---------------------------------- | ----------------------------------- |
| 执行模型 | SIMT / warp / thread block         | BSP + vertices + compute sets       |
| 存储模型 | cache + global memory              | tile-local SRAM + explicit exchange |
| 控制   | 大量线程共享 SIMT 执行模型                   | 每个 tile/vertex 可跑不同代码               |
| 数据搬运 | load/store 到 global memory / cache | 显式 copy / exchange                  |

文档还明确强调：**IPU 不是 SIMD device**，没有一个单一机器级 instruction stream 广播给所有 tile。同步点之间，不同 tile/thread 可以执行完全不同的代码。

---

## 2.2 IPU 内存体系：In-Processor-Memory + Streaming Memory

Graphcore 把内存分成两类：

```text
In-Processor-Memory
  = IPU 内部 SRAM
  = 每个 tile 一份 local SRAM
  = 分布式、独立地址空间

Streaming Memory
  = 外部 DDR
  = 作为大容量存储
  = tile 不能直接 load/store
```

文档里最重要的一句话是：

> tile 的 load/store 指令只能访问本 tile 的 local memory，不能直接访问其他 tile memory，也不能直接访问 Streaming Memory。

所以 IPU 的存储模型不是共享内存，而是：

```text
Tile local SRAM 之间：
  通过 exchange fabric 显式搬运

Streaming Memory ↔ Tile local SRAM：
  也通过显式 copy / exchange 完成
```

这点和传统 GPU/HBM 模型差别非常大。GPU 程序员通常认为有一个 global memory，线程通过 load/store 访问；而 IPU 更像是：

```text
数据先被编译器/程序映射到 tile local memory
计算只在本地数据上进行
需要远端数据时，通过 exchange phase 显式搬运
```

---

## 2.3 GC200 的 local SRAM 很大

文档给了 GC200 的数据：

```text
每 tile SRAM: 624 KB
tile 数量: 1472
总 In-Processor-Memory: 接近 900 MB
```

这是 IPU 非常激进的设计点。它不像 GPU 那样依赖大 HBM + cache，而是试图把大量模型状态、activation、中间 tensor 放在片上 SRAM 里。

对架构理解来说，这意味着：

```text
IPU 的优势：
  片上 SRAM 容量大
  SRAM 聚合带宽高
  显式数据位置可控
  编译器可做强静态优化

IPU 的代价：
  编程/编译复杂度高
  数据映射很重要
  tile 间通信显式化
  对动态图/动态 shape 不一定友好
```

---

# 3. IPU 执行模型：BSP，而不是自由异步 dataflow

这是这份文档最核心的部分。

Graphcore IPU 使用 **Bulk Synchronous Parallel, BSP** 执行模型。一个 step 包含三个阶段：

```text
1. Local tile compute
2. Global cross-tile synchronisation
3. Data exchange
```

也就是：

```text
Compute → Sync → Exchange → Compute → Sync → Exchange → ...
```

## 3.1 Compute phase

在 compute phase：

```text
每个 tile 只访问自己的 local memory
所有 tile 并行计算
tile 之间不通信
```

这意味着 compute phase 本质上是 **embarrassingly local** 的。

## 3.2 Sync phase

所有 tile 必须到达同步点，才能进入 exchange。

这会带来一个很重要的性能问题：

```text
一个 step 的 compute 时间 ≈ 最慢 tile 的 compute 时间
快 tile 需要等待慢 tile
```

所以 IPU 编译器/调度器必须尽量做 **load balance**。

## 3.3 Exchange phase

在 exchange phase：

```text
tile 之间搬运数据
tile 与 Streaming Memory 搬运数据
通过 exchange fabric 进行通信
```

这不是普通 NoC 中“任意时刻 packet/flit 异步流动”的模型，而是更偏 **阶段化通信**：

```text
先大家算
再大家同步
再大家搬
```

所以 IPU 的 exchange fabric 很强，但它的使用方式非常受 BSP 模型约束。

---

# 4. Tile 微架构：小型多线程处理器 + AMP 单元

每个 tile 内部有一个多线程处理器，有两种模式：

```text
Supervisor mode
Worker mode
```

可以粗略理解为：

```text
Supervisor thread:
  负责 tile 上的顶层控制程序

Worker threads:
  执行具体 vertex/codelet 任务
```

文档中说，线程按照固定 round-robin 顺序逐条指令执行。这意味着它不是乱序 CPU，而是一个面向可预测执行的小型多线程处理器。

指令类型包括：

```text
控制流指令
内存访问指令
整数和浮点算术指令
超越函数指令，例如 exp
随机数生成指令
支持 stochastic rounding
```

每个 tile 还有一个 **AMP unit，accumulating matrix product unit**，可以每周期执行最多 **64 个 MAC**。

这个设计很有意思：
Graphcore 没有选择一个巨大的全局 systolic array，而是把小矩阵/向量能力分散到每个 tile。

可以理解成：

```text
GPU:
  大量 SIMD lanes / tensor cores 组织在 SM 里

TPU:
  大 systolic array

Graphcore IPU:
  每个 tile 有较小的矩阵乘单元
  通过大量 tile 聚合吞吐
```

这和你之前讨论的 **many small tiles vs one/few giant tensor cores** 是同一个设计分叉。

---

# 5. Poplar 编程模型：global program + variables + compute sets + vertices

IPU 的软件栈核心是 **Poplar graph library**。这份文档主要解释 Poplar 如何把高级程序映射到 IPU。

## 5.1 Program

一个 IPU program 可以运行在一个或多个 IPU 上。运行前，用户选择一组 IPU，编译后这组 IPU 在执行期间不会变化。

程序看起来像普通控制流程序：

```text
while (...)
  Copy(...)
  Execute(compute_set)
  if (...)
    Copy(...)
  else
    Copy(...)
```

但是背后会被 lowering 成：

```text
per-tile code
explicit sync
explicit exchange
local compute
```

---

## 5.2 Variable

Poplar 里的 variable 是全局作用域的数组，可以被理解为底层存储对象。

但物理上，一个 variable 可以被切分映射到多个 tile：

```text
Variable X
 ├── elements [0:127] → tile 0
 ├── elements [128:255] → tile 1
 ├── elements [256:383] → tile 2
 └── ...
```

这个映射叫 **tile mapping**。

这里要注意一个关键概念：

```text
Tensor 是 variable 的 view
Variable 是真实底层存储
```

也就是说，tensor transpose / slice / concat 很多时候只是 view；但如果你把 transposed view copy 到另一个 variable，就会真正触发数据重排和搬运。

这个设计和 MLIR / tensor IR 里的 view、layout、copy 概念很像。

---

## 5.3 Compute set 与 vertex

IPU 的计算不是直接说“执行一个 op”，而是通过：

```text
Compute set
  ├── vertex 0
  ├── vertex 1
  ├── vertex 2
  └── ...
```

每个 vertex 是一个小任务，绑定到特定 tile 上，运行一段代码，称为 **codelet**。

所以：

```text
Compute set = 一批并行 vertex 的集合
Vertex = 绑定到某个 tile 的细粒度任务
Codelet = vertex 执行的代码
```

这套模型和你自己的架构探索 IR 很相关。因为它把一个算子拆成：

```text
数据放在哪里
每个 tile 执行什么 vertex
vertex 读写哪些 variable slice
什么时候 copy
什么时候 sync
```

这实际上就是一种 **显式 schedule representation**。

---

# 6. 编译流程：从 global program lowering 到 per-tile program

文档中有一个非常关键的过程：

```text
global IPU program
  ↓
explicit sync / exchange / compute steps
  ↓
per-tile program
  ↓
conventional compiler 编译每个 tile 的代码
```

lowering 后，每个 tile 的程序具备几个特征：

```text
只读写本 tile memory
包含显式同步指令
包含显式通信 routines
执行 compute set 时，只运行映射到本 tile 的 vertices
```

这说明 IPU 的编译器承担了非常重的职责：

```text
变量切分
tile mapping
通信插入
同步插入
liveness 分析
memory allocation
per-tile code generation
```

这和 GPU 编译器差别很大。GPU 也做调度和优化，但 GPU 硬件提供 shared global memory illusion；IPU 没有这个 illusion，所以编译器必须显式管理数据位置和数据移动。

---

# 7. Variable liveness：用静态分析压缩 SRAM 占用

文档专门讲了 **Variable liveness**。

虽然 variable 是 global scope，但它们的物理内存可以复用：

```text
如果 variable A 和 variable B 不会在同一时间 live
那么它们可以共享同一段 tile memory / Streaming Memory
```

这和编译器里的 register allocation / buffer reuse 很像。

对 IPU 非常重要，因为：

```text
片上 SRAM 很大，但仍然有限
ML graph 中 activation / temporary tensor 很多
编译器必须通过 liveness 分析降低 peak live memory
```

文档还说 PopVision Graph Analyser 可以显示程序执行过程中的 live memory 和 always-live memory。

这对你做架构探索也很有启发：
仅仅估算 tensor 总大小是不够的，更应该估算：

```text
peak live memory
always-live memory
temporary memory
activation stash
通信 buffer
```

---

# 8. Host / Device 通信：Host 不直接喂 tile，而是经 DDR/Streaming Memory

文档描述了 IPU-Machine 中 host 到 IPU 的通信路径：

```text
Host CPU memory
  ↓ RDMA over Ethernet
Host-side NIC
  ↓ Ethernet
IPU-Machine NIC
  ↓
IPU-Machine DDR / Streaming Memory
  ↓ exchange phase
Tile local memory
```

所以 host stream 数据不是直接进入 tile compute，而是经过 Streaming Memory，再由 IPU 在 exchange 阶段搬进 tile memory。

这意味着：

```text
Host-I/O 路径和 tile compute 路径是解耦的
Streaming Memory 既是大容量存储，也是 host/device 通信中转
```

文档还提到可以把 IPU 内的 tile 分成两组：

```text
I/O tile group
Compute tile group
```

让一部分 tile 专门负责 I/O，从而和 compute 并行。这是 IPU 内有限形式的 task parallelism。

---

# 9. Programming tools：Poplar SDK 软件栈

Graphcore 的软件栈支持多个层级：

```text
高层 ML framework:
  TensorFlow / PyTorch / Keras / ONNX

Graphcore 中间层:
  PopART / PopTorch / PopLibs

底层:
  Poplar C++ API
  Vertex code / codelet
  Assembly
```

用户可以：

```text
1. 直接用 PyTorch / TensorFlow
2. 写 Poplar C++ 程序
3. 写自定义 vertex/codelet
4. 给 ML framework 加自定义 IPU op
```

文档还强调：**整个程序总是 fully compiled**。这带来：

```text
优点：
  编译器有全局视野
  可以做强优化
  可以安排 memory liveness / communication / tile mapping

缺点：
  编译时间可能很长
  大模型程序可能编译几分钟
  对动态 workload 适配较差
```

---

# 10. 常见算法技术：Replication、Gradient Accumulation、Recomputation、Pipeline

后半部分讲的是如何在 IPU 上高效跑 ML workload。

## 10.1 Replication

Replication 就是多个 IPU 或多组 IPU 运行同一个 program。

例如：

```text
16 个 IPU
分成 4 个 replica
每个 replica 使用 4 个 IPU 跑同一个模型程序
```

训练时它就是 data parallel：

```text
每个 replica 处理不同 batch
各自算 gradient
通过 AllReduce 聚合 gradient
每个 replica 更新同样的权重
```

推理时更简单：

```text
每个 replica 处理不同输入
提升吞吐
```

文档还提到 replicas 之间可以通过 IPU-Fabric 做 collective operation，例如 AllReduce。

---

## 10.2 Replicated tensor sharding

这点很重要。

在 data parallel training 中，某些 tensor，比如权重和 optimizer state，在每个 replica 上内容相同。为了省内存，可以把这些 replicated tensor shard 到多个 replica 上：

```text
Replica 0 存一部分
Replica 1 存一部分
Replica 2 存一部分
Replica 3 存一部分
```

需要完整 tensor 时，再做 AllGather。

这本质是：

```text
用通信换内存
```

对大模型训练/推理都很关键。它和 ZeRO、FSDP 的思想接近。

---

## 10.3 Gradient accumulation

Gradient accumulation 是把大 batch 拆成多个 micro-batch 顺序执行：

```text
zero accumulated gradients

for N micro-batches:
  forward
  backward
  accumulate gradients

weight update
```

它解决的是：

```text
单次放不下大 batch
或者 pipeline 需要足够多 micro-batch 来提高利用率
```

在 IPU 上，gradient accumulation 还和 pipeline utilization 强相关。

---

## 10.4 Recomputation

Recomputation 是典型的 memory-compute tradeoff：

```text
不保存所有中间 activation
backward 需要时重新计算 forward activation
```

好处：

```text
降低 activation memory / stash memory
```

代价：

```text
增加额外 compute
延长执行时间
```

IPU 片上 SRAM 虽然很大，但由于每个 tile local memory 固定，而且多 IPU pipeline 需要 stash activation，所以 recomputation 是很自然的优化手段。

---

## 10.5 Model parallelism and pipelining

文档将模型分到多个 IPU：

```text
IPU 0: layers 0~k
IPU 1: layers k~m
IPU 2: layers m~n
...
```

然后用 micro-batch 流水执行：

```text
micro-batch 0 forward on IPU0
micro-batch 0 forward on IPU1
micro-batch 1 forward on IPU0
...
```

训练时还要 backward，因此有 forward pipeline 和 backward pipeline。

文档给了一个 pipeline utilization 的直觉：

```text
如果 gradient accumulation steps TGA 远大于 pipeline stages n
pipeline utilization 接近 1
```

特殊情况下利用率可近似为：

```text
TGA / (TGA + n - 1)
```

例如：

```text
4 stages, TGA = 1    → 利用率约 25%
4 stages, TGA = 1000 → 利用率约 99.7%
```

这说明 IPU pipeline 特别依赖足够多的 micro-batch 来填满流水。

---

## 10.6 Activation stash

训练 pipeline 有一个额外问题：

```text
forward 时产生的 activation
要等到对应 micro-batch backward 时才用
```

所以需要存一段时间，文档称为 **activation stash**。

这会带来显著 memory pressure：

```text
pipeline stage 越多
micro-batch 越多
stash 越长
memory 占用越大
```

文档说普通 schedule 中，N 个 IPU pipeline 的最大 stash length 可能是：

```text
2 × N - 1
```

而 interleaved schedule 可以把最大 stash 降到：

```text
N
```

但代价是 forward/backward 不平衡，整体可能更慢。

---

# 11. 从架构角度看，IPU 的设计哲学是什么？

我认为这份文档背后的设计哲学可以概括成四句话：

## 11.1 把 memory hierarchy 显式化

IPU 不想让 cache 去“猜”数据在哪里，而是让编译器和程序明确知道：

```text
数据在哪个 tile
什么时候搬
搬到哪里
谁消费它
何时释放
```

这提高了确定性，但增加了编译器复杂度。

---

## 11.2 用大量 distributed SRAM 替代大 shared memory illusion

IPU 的片上 SRAM 非常大，但不是统一共享 SRAM，而是：

```text
大量 tile-local SRAM
通过 exchange fabric 连接
```

这会带来很高的聚合 SRAM 带宽，但代价是数据布局和通信调度变成核心问题。

---

## 11.3 用 BSP 简化全局同步和通信模型

IPU 没有让 tile 间任意异步通信无限复杂化，而是采用：

```text
Compute phase
Sync phase
Exchange phase
```

这使得编译器和 profiler 更容易建立全局执行视图，也提高了确定性。

但代价是：

```text
barrier 等待
load imbalance 敏感
通信和计算 overlap 受限
```

---

## 11.4 用编译器承担复杂性

IPU 的硬件哲学不是“硬件自动调度一切”，而是：

```text
硬件提供大量小计算核心 + SRAM + fabric
编译器负责 mapping / scheduling / communication / memory reuse
```

这和 Groq、Tenstorrent、Cerebras、TPU/GPU 都可以形成非常好的对比。

---

# 12. 对你研究 AI 推理芯片架构的启发

结合你现在关注的云端推理芯片，我觉得这份文档至少有 5 个值得深入讨论的点：

| 主题                                   | 为什么重要                                   |
| ------------------------------------ | --------------------------------------- |
| **分布式 SRAM**                         | IPU 是“片上大 SRAM + 显式搬运”的代表案例             |
| **BSP 执行模型**                         | 适合分析确定性、barrier 成本、tile load balance    |
| **exchange fabric**                  | 它不是传统 mesh NoC 教科书模型，而是服务于 BSP exchange |
| **Poplar graph lowering**            | 对你设计 architecture DSL + event IR 很有参考   |
| **pipeline / stash / recomputation** | 对大模型训练和推理的 memory-compute tradeoff 很关键  |

尤其是 IPU 的这套模型可以映射到你的探索框架：

```text
Architecture DSL:
  Tile
  Local SRAM
  Exchange fabric
  Streaming Memory
  IPU-Link / GW-Link

Event / Schedule IR:
  ComputeSet
  Vertex
  Copy
  Sync
  Exchange
  StreamCopy
  AllReduce
  AllGather

Compiler analysis:
  Tile mapping
  Variable liveness
  Peak live memory
  Stash memory
  Pipeline utilization
```

---

# 13. 我建议后续逐一讨论的顺序

我建议我们不要一开始就评价“IPU 好不好”，而是按下面顺序深挖：

## 第一讲：IPU 的执行模型

重点讨论：

```text
Compute → Sync → Exchange
BSP 到底意味着什么？
它和 GPU / Tenstorrent / Groq / TPU 有什么本质不同？
```

## 第二讲：IPU tile 到底是什么

重点讨论：

```text
tile 是 PE、core、SM，还是 mini-processor？
AMP unit 和 tensor core/systolic array 有什么区别？
```

## 第三讲：IPU memory architecture

重点讨论：

```text
In-Processor-Memory
Streaming Memory
local address space
explicit copy
为什么 tile 不能直接 load/store 其他 tile memory？
```

## 第四讲：Poplar 编译模型

重点讨论：

```text
variable / tensor view / tile mapping / compute set / vertex / codelet
global program 如何 lowering 成 per-tile program？
```

## 第五讲：IPU 对 LLM 推理是否合适

重点讨论：

```text
权重能不能放 SRAM？
KV cache 怎么处理？
decode 阶段适不适合 BSP？
attention / all-reduce / GQA 对 IPU 是利好还是限制？
```

---

**一句话总结：这份文档展示的 IPU 本质上是一种“编译器管理的 many-tile BSP dataflow 机器”，核心不是单个矩阵阵列有多大，而是如何把数据、计算、同步、通信全部显式化。**
