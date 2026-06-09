> Cerebras_Wafer-Scale_Architecture.pdf,这是Cerebras官网的白皮书的内容

很好，这份白皮书信息密度其实挺高，尤其适合从 **core → memory → fabric → weight streaming → MatMul mapping** 这条链路拆。我们先把它抽象成一个可讨论的架构模型，然后逐个问题深入。📌

# 0. 这份 Cerebras 白皮书的核心主张

这份文档围绕 **WSE-3 / CS-3** 展开，Cerebras 的架构主张可以浓缩成一句话：

> **用 wafer-scale 的超大规模二维 mesh，把大量小型 dataflow core、分布式 SRAM、静态路由 fabric 和权重流式输入结合起来，尽量避免传统多 GPU 系统中的 HBM 容量、HBM 带宽、芯片间通信和模型并行复杂度。** 

它不是传统意义上的：

* GPU；
* TPU-style systolic array；
* 大 tensor core；
* cache-coherent manycore CPU；

而更接近：

> **超大规模 distributed-SRAM dataflow machine。**

---

# 1. 白皮书里的关键参数

先把关键架构数字抓出来：

| 项目                   | WSE-3 / CS-3 描述                                          |
| -------------------- | -------------------------------------------------------- |
| 工艺                   | TSMC 5nm                                                 |
| 芯片面积                 | 46,225 mm²                                               |
| 晶体管数                 | 超过 4 万亿                                                  |
| core 数量              | 900,000 个                                                |
| 单 core 面积            | 约 38,000 μm²                                             |
| 单 core SRAM          | 48 KB                                                    |
| core 逻辑              | 约 110,000 standard cells                                 |
| 每 core SIMD          | 8-wide FP16 SIMD                                         |
| fabric 拓扑            | 2-D mesh                                                 |
| router 端口            | 5-port：东西南北 + core                                       |
| packet 粒度            | 16-bit data + 16-bit control = 32-bit packet             |
| routing              | static routing                                           |
| local routes         | 每 core 24 个 static route，称为 colors                       |
| 片上总 memory bandwidth | 21 PB/s                                                  |
| 片上 fabric bandwidth  | 214 Pb/s                                                 |
| 单跳延迟                 | 相邻节点 single clock cycle latency                          |
| 权重存储                 | 外部 MemoryX，权重流式输入                                        |
| 计算模型                 | weight streaming + dataflow-triggered AXPY / Sparse-GEMM |

这个参数表已经能说明它的设计哲学：**不是把单个 core 做强，而是把 core 做小、SRAM 做近、fabric 做密、系统做巨大。**

---

# 2. 第一层理解：Cerebras core 到底是什么？

白皮书明确说，Cerebras core 是一个 **small-core design**。每个 core 大约一半面积是 SRAM，一半面积是 logic。每个 core 有 **48 KB 本地 SRAM**，并且这些 SRAM 是 **per-core independently addressed**，不是传统共享内存。跨 core 的共享必须通过 fabric 显式完成。

这点非常关键。

它不像 GPU：

```text
HBM → L2 → SM shared memory / register file → Tensor Core
```

也不像 TPU：

```text
HBM → unified buffer → systolic array
```

Cerebras 更像：

```text
Core_i:
  local SRAM + local instruction + local dataflow datapath + local router

Core_j:
  local SRAM + local instruction + local dataflow datapath + local router

Core_i <-> router/fabric <-> Core_j
```

也就是说：

> **每个 core 是一个极小的、带本地 SRAM 的可编程 dataflow processing element。**

---

# 3. 第二层理解：为什么它强调 SRAM，而不是 HBM？

白皮书里有一个很重要的说法：传统 GPU 的 compute datapath bandwidth 往往比 DRAM/HBM bandwidth 高很多，所以 operand 必须被复用很多次，否则 datapath 吃不饱。Cerebras 的路线是把 SRAM 完全分布到 core 旁边，让本地 memory bandwidth 接近 core datapath operand bandwidth，从而减少对高数据复用的依赖。

这背后的架构含义是：

## 传统 GPU 的问题

GPU 很强，但它依赖：

```text
高算力 + 高复用 + 高 batch + 高 arithmetic intensity
```

所以 GEMM 很舒服，但 GEMV、AXPY、稀疏计算、小 batch decode 这类低复用 workload 就容易变成 memory-bound。

## Cerebras 的反向选择

Cerebras 说：

```text
既然稀疏、GEMV、AXPY 这些低复用计算需要巨大 memory bandwidth，
那我就把 memory 分布到每个 core 旁边。
```

所以它的路线不是：

> **让一个 HBM 给很多 compute 喂数据。**

而是：

> **让每个小 compute 都有自己的本地 SRAM，并用 fabric 显式搬运数据。**

这和你之前反复讨论的 **many-tile 提升 aggregate SRAM→compute bandwidth** 是高度一致的，只是 Cerebras 把这个想法推到 wafer-scale。

---

# 4. 第三层理解：它的 fabric 是不是 NoC？

我认为：**是 NoC，而且是非常典型的 AI dataflow NoC。**

白皮书明确说：

* fabric 是 **2-D mesh topology**；
* 每个 core 有一个 fabric router；
* router 是 **five-port design**；
* 东西南北四个方向是 32-bit bidirectional interface；
* 还有一个 port 面向 core；
* 使用 lossless flow control；
* 使用 low buffering；
* 相邻节点 single cycle latency；
* static routing；
* 每个 core 支持 24 个 local static routes，也就是 colors；
* 支持 broadcast / multicast。

这基本上已经满足 NoC 的核心定义了：

```text
nodes + routers + links + topology + routing + flow control
```

但它不是 CPU/GPU 常见的“通用 packet NoC”。它更像：

> **静态路由、编译器参与、细粒度 packet、面向 dataflow trigger 的专用 ML fabric。**

这和你之前对 Graphcore exchange fabric 的理解可以放在一起比较：

| 项         | Graphcore IPU             | Cerebras WSE                             |
| --------- | ------------------------- | ---------------------------------------- |
| fabric 类型 | 静态/编译器控制倾向很强              | 静态路由 2-D mesh                            |
| 通信粒度      | bulk exchange / BSP 语义更明显 | 16-bit data + 16-bit control 的细粒度 packet |
| 同步模型      | BSP 更强                    | dataflow trigger 更强                      |
| 计算触发      | exchange 后计算              | packet 到达可触发 core lookup / FMAC          |
| router    | 文档公开较少                    | 明确 five-port router                      |
| 特点        | tile 间 exchange           | wafer-scale dataflow fabric              |

所以我们可以先定一个结论：

> **Cerebras fabric 是 NoC，但不是为 cache coherence 或 general-purpose message passing 设计的 NoC，而是为 ML dataflow、广播、多播、静态路由、稀疏触发设计的 wafer-scale NoC。**

---

# 5. 第四层理解：weight streaming 是 Cerebras 的关键机制

白皮书最有意思的一部分是 **Weight Streaming**。

Cerebras 不是把完整模型权重都放在 WSE-3 上，而是把权重放在外部 **MemoryX**，然后按 layer / weight stream 到 CS-3。文档说，权重不会存储在 CS-3 上，甚至不是临时存储，而是被流进来后直接触发计算，用完就丢弃。

这个机制非常关键。

它说明 Cerebras 的片上 SRAM 主要不是拿来放所有权重，而是放：

```text
activations
partial sums
local tensors
temporary data
instructions
control structures
```

而权重变成：

```text
external MemoryX → stream into wafer → trigger compute → discard
```

这是一种很特殊的模型：

```text
weights are streamed
activations are resident
compute is dataflow-triggered
```

它和 GPU 常见方式完全不同：

```text
GPU:
  weights in HBM
  activations in HBM/cache/register/shared memory
  tensor core loads operands repeatedly

Cerebras:
  activations resident on wafer SRAM
  weights stream through fabric
  weight packet triggers AXPY/FMCA
  weights not retained
```

---

# 6. 第五层理解：为什么它把 MatMul 看成 AXPY / Sparse-GEMM？

白皮书里有个非常重要的架构视角：

> Sparse-GEMM 可以看成每个非零 matrix element 对应一次 AXPY。

也就是：

```text
y += weight * activation_vector
```

当 weight 是非零时，才广播这个 weight；当 weight 是零时，直接跳过，不发送、不计算。

这解释了 Cerebras 为什么强调：

* fine-grained dataflow；
* unstructured sparsity；
* 16-bit data + 16-bit control packet；
* weight packet 触发 FMAC；
* sender 端过滤 zero；
* receiver 端只看到 nonzero data。

它不是像很多 GPU sparse tensor core 那样要求结构化稀疏，而是试图利用 **fine-grained unstructured sparsity**。

当然，这里后续值得批判性讨论：
**非结构化稀疏在真实 LLM 中到底能稳定带来多少收益？稀疏模式怎么训练？精度损失如何控制？稀疏带来的负载不均衡怎么处理？**

这会是 Cerebras 架构成败的核心问题之一。

---

# 7. 第六层理解：WSE 上的 MatMul 映射方式

白皮书描述了 transformer MatMul 的逻辑映射：

```text
activation tensor: [Batch, Sequence, Hidden]

Hidden dimension → fabric x-direction
Batch + Sequence → fabric y-direction
```

也就是说，它把整个 wafer 当成一个巨大的 **Wafer MatMul Array**：

```text
x 方向切 hidden
y 方向切 batch/sequence
```

然后：

1. activations 先分布在对应 core 的 SRAM 中；
2. weights 通过 broadcast fabric 发给各列；
3. weight 到达后触发 FMAC；
4. partial sums 通过 ring pattern 做 reduction；
5. reduction 和下一行 weight 的 FMAC 通过 microthreads overlap；
6. 最终 output activation 仍然按下一层需要的分布方式存放。

这个设计非常漂亮，核心是：

> **让每一层的输出分布自然成为下一层的输入分布，减少中间重排。**

这点对你的架构 DSL/IR 很有启发：
mapping 不只是算一个 op 的 tile partition，而是要考虑：

```text
operator_i output layout == operator_i+1 input layout
```

否则每层之间的数据重排会吞掉大量收益。

---

# 8. 我建议我们后续重点讨论 6 个问题

## 问题 1：Cerebras 的 core 是“processor”还是“dataflow PE”？

白皮书说它是 general processor，有 GPR、load/store、branch、six-stage pipeline；但实际 ML 执行又高度依赖 tensor instruction、DSR、dataflow trigger、microthread。
这个混合形态很有意思。

## 问题 2：48 KB SRAM per core 是否真的等价于“DRAM equivalent capacity”？

它总 SRAM 很大，但分布式 SRAM 的可用性取决于 mapping。
不是所有 SRAM 都能像统一大内存一样任意访问。

## 问题 3：static routing + 24 colors 到底怎么理解？

这很适合和你现在的 NoC 学习结合。
Cerebras 的 colors 本质上像是：

```text
statically configured virtual routes / logical channels
```

但它和传统 virtual channel 不是完全一回事。

## 问题 4：weight streaming 对训练有利，但对 LLM inference decode 是否同样有利？

训练和大 batch inference 可以 amortize weight streaming；
但小 batch decode 可能 amortization 不足。
这是 Cerebras 用于 LLM 推理时最值得质疑的点。

## 问题 5：Cerebras 的“无需 model parallel”到底成立到什么程度？

它说大模型可以在单 chip 上跑，不需要复杂 model parallel distribution。
但严格来说，它是：

```text
weights externalized + activations resident + wafer-scale compute
```

不是说所有权重都在片上。

所以“无需 model parallel”成立，但前提是接受 weight streaming 这种模型。

## 问题 6：它对你的架构探索方法有什么启发？

我觉得至少有 4 点：

```text
1. 架构 DSL 必须支持 distributed local memory，而不是只写 global SRAM size。

2. NoC DSL 必须能表达 static routes / multicast / broadcast / reduction pattern。

3. Mapping IR 必须表达 tensor layout across physical topology。

4. Event IR 需要表达 data-triggered compute，而不仅是固定时间发起的 compute event。
```

---

# 一句话总结

**Cerebras WSE-3 本质上是 wafer-scale 的 distributed-SRAM dataflow NoC machine：用 90 万个小 core、超大二维 mesh、静态路由、权重流式输入和数据触发计算，换取极高片上带宽与大模型映射简化。**

接下来我建议我们先深入第一个问题：**Cerebras core 到底是传统 processor、SIMD PE，还是 dataflow core？**
