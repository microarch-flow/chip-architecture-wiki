> Cerebras_Architecture.pdf、HC2022.pdf、HC2024.pdf是剩下的材料，现在开始分析

好，我们正式开始。
我建议第一轮先不钻某一个细节，而是先建立 **Cerebras 架构总心智模型**。后面我们再逐层深挖：core、SRAM、fabric、weight streaming、MemoryX/SwarmX、training、inference、以及和 GPU/Graphcore/Groq/Tenstorrent 的差异。🚀

# 1. Cerebras 的一句话定位

**Cerebras WSE 本质上不是“超大 GPU”，而是一台 wafer-scale 的 distributed-SRAM dataflow computer。**

它的核心不是传统 GPU 的：

```text
HBM + cache hierarchy + SM/Tensor Core + warp/SIMT execution
```

而更像：

```text
90 万个小型可编程 dataflow core
+ 每 core 本地 SRAM
+ wafer-scale 2D mesh fabric
+ 静态路由 / broadcast / multicast / reduce
+ 编译器控制的数据流执行
+ 外部 MemoryX 权重流式输入
+ SwarmX 多系统扩展
```

Cerebras 自己反复强调三条主线：

```text
1. Core architecture：用 fine-grained dataflow core 支持非结构化稀疏
2. Scale-up：用 wafer-scale chip 放大 Moore’s Law
3. Scale-out：用 MemoryX + SwarmX 简化多系统扩展
```

这三点在官网白皮书、HC2021、HC2022、IEEE Micro 文章里是一致的。

---

# 2. Cerebras 架构演进：WSE-1 → WSE-2 → WSE-3

先把版本关系捋清楚。

| 版本           |                        年份/材料 |   工艺 | core 数 |     片上 SRAM | memory BW | fabric BW | 重点                   |
| ------------ | ---------------------------: | ---: | -----: | ----------: | --------: | --------: | -------------------- |
| WSE-1        |                         2019 | 16nm | 约 400k | 未在这些材料中完整展开 |         — |         — | 第一代 wafer-scale 验证   |
| WSE-2 / CS-2 | HC2021 / HC2022 / IEEE Micro |  7nm |   850k |        40GB |    20PB/s |   220Pb/s | 训练、稀疏、MemoryX、SwarmX |
| WSE-3 / CS-3 |                 白皮书 / HC2024 |  5nm |   900k |        44GB |    21PB/s |   214Pb/s | 更强 core，训练 + 推理叙事    |

HC2021 给出的 WSE-2 参数是 **46,225 mm²、2.6T transistors、850,000 cores、40GB SRAM、20PB/s memory bandwidth、220Pb/s fabric bandwidth、TSMC 7nm**。

HC2024 给出的 WSE-3 参数是 **46,225 mm²、4T transistors、900,000 AI cores、125PFLOPS AI compute、44GB on-chip memory、21PB/s memory bandwidth、214Pb/s fabric bandwidth、TSMC 5nm**。

这里有一个很重要的判断：

> **Cerebras 的代际升级不是改变基本架构范式，而是在同一个 wafer-scale dataflow + distributed SRAM 范式下提高 core 数量、数据通路能力、SRAM 容量和系统软件成熟度。**

也就是说，WSE-2 到 WSE-3 不是类似 GPU 架构大换代，而更像：

```text
same architecture philosophy
+ better process
+ more transistors
+ stronger SIMD datapath
+ slightly more SRAM
+ inference story becomes more explicit
```

---

# 3. 第一性原理：Cerebras 到底想解决什么问题？

它其实在解决两个不同但相关的问题。

## 3.1 训练问题：大模型分布式训练太复杂

传统 GPU 集群训练大模型时，往往需要混合使用：

```text
data parallel
tensor parallel
pipeline parallel
sequence parallel
optimizer state sharding
activation checkpointing
communication overlap
```

HC2022 里专门批判了 GPU cluster 的 hybrid parallelism：data parallel 受参数内存限制，pipeline parallel 有通信和 activation memory 问题，tensor parallel 有复杂 partitioning 和通信问题。

Cerebras 的回答是：

```text
把单个 device 做到足够大
让一个 WSE 可以承载单层超大 MatMul
把权重容量问题交给外部 MemoryX
多 WSE 只做 data parallel
```

这就是它所谓：

> **large model on single device + data-parallel-only scale-out**

## 3.2 推理问题：生成式推理被 memory bandwidth 卡住

HC2024 把叙事转向 inference。它认为生成式推理慢，是因为生成 1000 个 token 需要 1000 次 serial passes through the model，每次 pass 都要从 memory 读取模型参数，因此 low memory bandwidth 成为瓶颈。

它用 WSE-3 对 H100 做了一个极端对比：

| 项                |      WSE-3 |                     H100 | Cerebras 声称优势 |
| ---------------- | ---------: | -----------------------: | ------------: |
| chip size        | 46,225 mm² |                  814 mm² |           57× |
| cores            |    900,000 | 16,896 FP32 + 528 Tensor |           52× |
| on-chip memory   |       44GB |                   0.05GB |          880× |
| memory bandwidth |     21PB/s |                0.003PB/s |         7000× |

HC2024 的结论是：wafer-scale + SRAM 能“完全移除 memory bandwidth bottleneck”。这当然带有营销表达，但它揭示了 Cerebras 推理路线的核心：**把模型尽量放进 wafer SRAM，而不是每 token 从 HBM 读权重。** 

---

# 4. Cerebras 最核心的架构分解

我建议你把 Cerebras 拆成 6 个层次理解。

```text
Level 1: Core
Level 2: Local SRAM
Level 3: Wafer-scale fabric / NoC
Level 4: MatMul mapping
Level 5: Weight Streaming + MemoryX
Level 6: SwarmX scale-out
```

下面逐层讲。

---

# 5. Level 1：Core 是“小 processor + tensor datapath + dataflow trigger”

Cerebras core 不是纯 MAC array，也不是传统 CPU core，而是一个混合体：

```text
general-purpose processor foundation
+ tensor instruction datapath
+ DSR tensor descriptor
+ fabric-triggered handler lookup
+ microthread scheduler
```

HC2022 给出的 WSE-2 core 参数：

```text
core area: 228µm × 170µm
process: TSMC N7
logic:SRAM area ratio = 50:50
logic: 110,000 standard cells
SRAM: 48KB
clock: 1.1GHz
peak power: 30mW
```



它有几个关键部件：

| 部件                        | 作用                               |
| ------------------------- | -------------------------------- |
| 16 GPRs                   | 通用控制和标量操作                        |
| 44 DSRs / WSE-3 为 48 DSRs | 描述 tensor operand                |
| tensor datapath           | 执行 FP16/BF16/INT 等 tensor op     |
| 48KB SRAM                 | 数据 + 指令                          |
| fabric input/output       | 接收数据/控制 packet                   |
| dataflow trigger          | 根据 packet 触发 handler instruction |
| microthreads              | 硬件支持多个 tensor context 交错执行       |

白皮书和 IEEE Micro 文章都强调：DSR 可以描述 tensor 地址、长度、shape、stride，也可以描述 fabric-streaming tensor、FIFO、circular buffer；core 接收 fabric 上的数据和控制后，根据输入信息 lookup 要执行的指令。

所以 Cerebras core 的心智模型应该是：

```text
不是：一个线程按 PC 顺序一直跑

而是：core 预加载一组 handler / tensor ops，
      fabric packet 到达后触发某个 tensor operation，
      tensor operation 由 DSR 描述的数据结构自动展开执行。
```

这点非常重要。

它更接近：

> **可编程 dataflow PE，而不是传统顺序 processor。**

---

# 6. Level 2：Distributed SRAM 是 Cerebras 的根基

每个 core 有 48KB SRAM。WSE-2 总计 40GB，WSE-3 总计 44GB。Cerebras 不把这称为 cache hierarchy，而是把它作为每个 core 独立寻址的本地 SRAM。

HC2022 对 WSE-2 local memory 给了更具体结构：

```text
48KB per PE
8 banks × 6KB
32b wide per bank
single port
2 × 64b read + 1 × 64b write per cycle
256B software-managed cache
```



这和 GPU 的差异非常大：

| 项             | GPU                                        | Cerebras                       |
| ------------- | ------------------------------------------ | ------------------------------ |
| 主容量           | HBM                                        | SRAM distributed across wafer  |
| local storage | register/shared memory/cache               | 每 core 48KB SRAM               |
| 数据共享          | cache / shared memory / HBM / interconnect | 显式通过 fabric                    |
| 带宽来源          | HBM + cache reuse                          | aggregate local SRAM bandwidth |
| 适合            | dense GEMM，高复用                             | 低复用、GEMV、AXPY、稀疏、流式数据流         |

Cerebras 的核心论点是：

> GPU 的 datapath bandwidth 远大于 HBM bandwidth，所以要靠数据复用；Cerebras 则把 memory 分布到 datapath 旁边，使 memory bandwidth 接近 datapath operand bandwidth。

这就是为什么它总强调 **all BLAS levels full performance**。
GEMM 数据复用高，GPU 很适合；但 GEMV、DOT、AXPY 数据复用低，GPU 更容易 memory-bound。Cerebras 认为 Sparse GEMM 可以分解为每个 nonzero weight 对应一次 AXPY，所以如果 AXPY 都能全速跑，就能自然支持 fine-grained unstructured sparsity。

---

# 7. Level 3：Fabric 是 NoC，而且是静态路由 dataflow NoC

这个可以明确回答：

> **Cerebras fabric 是 NoC。**

证据非常直接。材料明确说它是：

```text
2D mesh topology
each core has a fabric router
5-port router: north/south/east/west + core
32b/cycle bidirectional links
single-cycle latency between neighboring cores
lossless flow control with low buffering
24 configurable static routing colors
dedicated buffering per color
hardware broadcast/multicast
```



这满足 NoC 的完整定义：

```text
nodes + routers + links + topology + packet + routing + flow control
```

但它不是通用 CPU coherent NoC。它的特点是：

| 维度                  | Cerebras fabric                              |
| ------------------- | -------------------------------------------- |
| topology            | 2D mesh                                      |
| routing             | static routing                               |
| packet              | 32-bit fine-grained packet                   |
| data/control        | packet 携带 16-bit data + 16-bit control/index |
| flow control        | lossless, low buffering                      |
| broadcast/multicast | 硬件支持                                         |
| route abstraction   | colors                                       |
| 编译器角色               | 极强，静态规划通信                                    |

所以它更准确的名字是：

> **wafer-scale static-routed dataflow NoC。**

这和 Graphcore exchange fabric 的差异在于：Graphcore 更强调 BSP exchange；Cerebras 更强调 packet arrival 触发 compute，fabric 本身直接参与 dataflow execution。

---

# 8. Level 4：Cerebras 怎么把 MatMul 映射到 wafer？

Cerebras 最关键的图是：**the wafer is the MatMul array**。

Transformer 的 activation tensor 可以抽象成：

```text
activation[B][S][H]
```

其中：

```text
B = batch
S = sequence
H = hidden / feature
```

Cerebras 的映射大概是：

```text
H dimension → wafer x direction
B/S dimension → wafer y direction
```

每个 core 的 local SRAM 存一小块 activation tensor。权重从 wafer 边缘或输入通道 broadcast 进来，沿列/行传到相应 cores。每个 nonzero weight packet 到达后，触发对应 core 用本地 activation 做 FMAC/AXPY。白皮书中明确说 hidden dimension split over fabric x-direction，batch and sequence split over y-direction。

一个简化图：

```text
              H dimension →
        ┌──────────────────────────┐
 B/S ↓  │ core core core core core │
        │ core core core core core │
        │ core core core core core │
        │ core core core core core │
        └──────────────────────────┘

weights broadcast →
partial sums reduce along ring / rows
activations resident in local SRAM
```

执行过程是：

```text
1. activation tensor 分布在各 core SRAM
2. weight stream 进入 wafer
3. weight packet 携带 data + index/control
4. nonzero weight 到达后触发 FMAC
5. partial sum 存在 local accumulator/cache
6. PSUM/FSUM command 触发 reduction
7. partial sums 通过 ring pattern 归约
8. 输出 activation 保持适合下一层的 layout
```

HC2022 里对 sparse GEMM 的描述非常清楚：nonzero weights broadcast over columns of cores，每个 individual weight triggers FMACs，zero weights 不发送、不计算、不占片上 weight memory。

---

# 9. Level 5：Weight Streaming 是训练系统的核心

Cerebras 的训练路线不是：

```text
把模型权重全部放在 WSE SRAM
```

而是：

```text
weights stored externally in MemoryX
weights streamed onto wafer layer by layer
activations resident on wafer
gradients streamed back
weight update occurs in MemoryX
```

HC2021 和 HC2022 都反复讲这个执行模型。HC2024 也继续说 MemoryX stores/streams weights to CS-3s，SwarmX performs broadcast/reduce。

这有一个关键含义：

> **Cerebras 的训练容量来自 MemoryX，不是来自 WSE SRAM。**

所以当它说“120T parameter capacity on a CS-2 system”时，不能理解为 120T 参数都在 wafer SRAM 里，而是 MemoryX 系统能承载模型权重和 optimizer state，WSE 负责每层流式计算。

这套模型的优点是：

```text
1. 参数容量和 compute 解耦
2. 单层 MatMul 可以在一个 WSE 上完整执行
3. 避免 GPU cluster 上复杂 tensor/pipeline parallel
4. 权重流式访问适合训练中的大 batch / pipeline
5. gradient 回流后在 MemoryX 做 optimizer update
```

但代价也很明显：

```text
1. MemoryX 带宽必须足够
2. 执行模型依赖 layer-by-layer streaming
3. 小 batch / low reuse 场景下 amortization 可能不够
4. 系统软件必须强力调度 streaming、compute、update overlap
```

所以 Weight Streaming 不是“免费解决容量问题”，而是把问题重构成：

> **外部权重流 + wafer activation residency + pipeline overlap。**

---

# 10. Level 6：SwarmX 是 scale-out 的关键

Cerebras 的 scale-out 不是 GPU 集群常见的 tensor parallel / pipeline parallel，而是：

```text
MemoryX broadcasts weights to all CS systems
each CS processes different data batch
gradients are reduced on return path
SwarmX performs broadcast/reduce
```

IEEE Micro 文章说，SwarmX 位于 MemoryX 和 CS-2 之间，broadcast weights to all CS-2 systems，并 reduce gradients from all CS-2s；内部使用 tree topology 来做模块化低开销扩展。

这就是它所谓的：

> **data parallel only scale-out。**

对比 GPU：

```text
GPU:
  model too big → tensor parallel + pipeline parallel + ZeRO + data parallel

Cerebras:
  single WSE handles layer mapping
  MemoryX handles parameter capacity
  SwarmX handles broadcast/reduce
  multi-WSE primarily data parallel
```

这是非常强的架构主张。
它的本质不是“通信消失了”，而是：

> **把模型并行通信转换成更规则的 broadcast/reduce 通信。**

这对硬件非常友好，因为 broadcast/reduce 可以用专用 fabric/tree 做得更高效、更确定。

---

# 11. Cerebras 的训练架构和推理架构其实是两套叙事

这是读这些材料时最重要的批判点之一。

## 11.1 训练叙事：weights external，activations resident

训练时，Cerebras 的主要模式是：

```text
MemoryX holds weights + optimizer state
WSE holds activations / deltas / partial sums
weights stream in
gradients stream out
optimizer update in MemoryX
```

这适合：

```text
large batch training
layer-by-layer execution
weight reuse across batch/sequence
pipeline overlap
data-parallel scale-out
```

## 11.2 推理叙事：models fit in SRAM，avoid HBM memory wall

HC2024 的推理叙事发生了变化。
它强调生成式推理是 memory bandwidth problem，并展示 WSE-3 的 44GB SRAM 能放下 Llama3.1-8B 的 FP16 权重；对 Llama3.1-70B，它展示 140GB FP16 权重可用 4×WSE-3 的 176GB SRAM 承载。

这意味着推理时它更像：

```text
weights resident in wafer SRAM
token-by-token decode avoids repeated HBM reads
```

而不是训练时的：

```text
weights streamed from MemoryX every layer
```

这是一个非常关键的转变。

所以我们后续讨论 inference 时，要严格区分：

| 场景                          | 权重位置                    | 核心瓶颈                              | Cerebras 方案                      |
| --------------------------- | ----------------------- | --------------------------------- | -------------------------------- |
| 训练                          | MemoryX                 | 大模型容量、分布式并行复杂度                    | weight streaming + data parallel |
| 大 batch inference / prefill | 可能 resident 或 streaming | GEMM/attention compute + memory   | wafer-scale parallelism          |
| single-user decode          | 尽量 SRAM resident        | 权重读取带宽、KV cache、serial dependency | SRAM bandwidth + pipeline        |
| 70B+ inference              | 多 WSE resident          | 跨 WSE layer/pipeline/通信           | 多 wafer mapping                  |

这也是我后面最想和你深入讨论的部分：
**Cerebras 对训练的解释很自洽；对推理的 HC2024 叙事非常吸引人，但需要仔细拆解它的条件、batch、KV cache、latency、throughput 和系统成本。**

---

# 12. 从架构哲学看，Cerebras 和 GPU 的本质差别

## GPU 的哲学

```text
小/中等芯片
高频
HBM
SIMT/SIMD
dense tensor core
cache/register/shared memory 做复用
多 GPU 通过 NVLink/IB 扩展
软件用复杂并行策略填满硬件
```

GPU 的核心假设是：

> **workload 有足够 reuse，可以靠 blocking/tiling/cache/tensor core 获得高利用率。**

## Cerebras 的哲学

```text
极大芯片
大量小 core
大量分布式 SRAM
静态路由 fabric
dataflow trigger
weight/activation 显式布局
专用 broadcast/reduce 扩展
```

Cerebras 的核心假设是：

> **把 memory 放到 compute 旁边，把通信留在 wafer 内，把分布式复杂度尽量前移到编译器和系统架构中。**

这两条路线的根本区别是：

| 维度    | GPU                                  | Cerebras                                                |
| ----- | ------------------------------------ | ------------------------------------------------------- |
| 扩展方式  | scale-out first                      | scale-up first                                          |
| 内存    | HBM-centered                         | SRAM-distributed                                        |
| 并行    | SIMT + tensor core                   | dataflow PE array                                       |
| 通信    | 多级 interconnect                      | wafer 2D mesh + SwarmX                                  |
| 软件复杂度 | runtime/kernel/parallel strategy 很复杂 | compiler mapping / static route / dataflow schedule 很复杂 |
| 适合负载  | dense GEMM，通用 AI                     | 大模型训练、低复用/稀疏/推理 memory wall 场景                          |
| 风险    | 通信复杂、HBM wall                        | 良率、散热、编译映射、灵活性、成本                                       |

一句话：

> **GPU 是“用软件和并行策略驯服多芯片系统”；Cerebras 是“把系统尽量做成一颗巨大的、可静态规划的数据流芯片”。**

---

# 13. 对你的架构探索 DSL / IR 的启发

Cerebras 对你现在做的 architecture DSL 非常有价值，因为它验证了你之前提出的几层抽象。

## 13.1 Hardware DSL 不能只描述 peak compute

必须描述：

```text
core array topology
local SRAM per core
SRAM banking
fabric topology
router port
static route/color
broadcast/multicast support
reduction pattern
external memory system
scale-out broadcast/reduce fabric
```

## 13.2 Mapping IR 必须描述 tensor layout

Cerebras 最关键的是：

```text
activation[B][S][H] 如何映射到 wafer 2D mesh
H 映射到 x
B/S 映射到 y
weight 如何 broadcast
partial sum 如何 reduce
output layout 如何服务下一层
```

这说明 mapping IR 不能只写：

```text
op = matmul
tile = 16x16
```

还要写：

```text
tensor dimension → physical dimension
data residency
stream direction
reduction topology
layout transform cost
```

## 13.3 Event IR 必须支持 data-triggered execution

Cerebras 的事件不是简单的：

```text
time t: core executes op
```

而是：

```text
packet arrives on color c
→ lookup handler
→ DSR selects tensor
→ tensor datapath executes
→ output packet emitted / partial sum accumulated
```

所以你的 Event IR 里需要表达：

```text
data arrival event
control packet event
handler lookup
tensor operation
fabric tensor operand
microthread availability
backpressure / output availability
priority
```

这和你之前说 event 应该由 compiler/scheduler 产生，并与 hardware DSL 绑定，是非常一致的。

---

# 14. 第一轮结论：Cerebras 真正的架构贡献是什么？

我认为不是“芯片大”这么简单，而是这 5 个东西组合在一起：

```text
1. Wafer-scale scale-up
   用一整片 wafer 做一个逻辑芯片，减少传统多芯片边界。

2. Distributed SRAM
   把 memory 分布到每个 core 附近，获得巨大 aggregate bandwidth。

3. Fine-grained dataflow core
   通过 packet-triggered execution 支持非结构化稀疏和动态执行。

4. Static-routed fabric
   2D mesh + colors + broadcast/multicast + reduction，使通信可规划、低开销。

5. Disaggregated training system
   MemoryX 解决容量，SwarmX 解决 data-parallel broadcast/reduce，WSE 解决 layer compute。
```

所以 Cerebras 不是一个单点创新，而是一个系统级架构选择：

> **用 wafer-scale SRAM dataflow machine，把传统 GPU 集群中的 HBM wall、model parallel complexity、chip-to-chip communication 尽量转换成片上 fabric、静态映射、权重流和 broadcast/reduce。**

---

# 15. 下一步建议：先深挖 core，还是先深挖 fabric？

我建议下一步从 **core 微架构** 开始，因为 Cerebras 后面所有东西都建立在这个 core 上：

```text
1. 为什么它说 core 是 general processor？
2. tensor instruction 到底怎么执行？
3. DSR 和普通寄存器的区别是什么？
4. fabric packet 如何触发 instruction lookup？
5. microthread 是不是类似 GPU warp？
6. 它为什么能支持 unstructured sparsity？
```

一句话总结：**Cerebras 的本质是 wafer-scale distributed-SRAM dataflow NoC machine；它用极端 scale-up 和显式数据流，把 GPU 集群里的很多动态复杂性转移到编译器、静态路由、MemoryX/SwarmX 和 wafer-scale fabric 中。**


