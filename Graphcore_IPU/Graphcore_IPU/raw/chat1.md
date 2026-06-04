> 这是HC2021的资料，同样帮我梳理分析一下，等到将我收集的资料全部整理后，我们才开始讨论

可以，这份 **Hot Chips 2021 Graphcore Colossus Mk2 IPU** 比前一份 *IPU Programmer’s Guide* 更偏“架构宣讲 + 设计取舍论证”。前一份是 **怎么编程、怎么执行**，这份则是在回答：

> Graphcore 为什么要这样设计 IPU？Colossus Mk2 相比 Mk1 改了什么？为什么不用 HBM？为什么强调 SRAM、BSP、Exchange、fine-grained parallelism？

下面我先按“架构师视角”梳理，不展开批判性讨论，等你资料收集齐后我们再统一对比分析。

# 1. 这份 HC2021 资料的核心观点

一句话概括：

> **Graphcore 把 Colossus Mk2 IPU 定义为一种面向未来 AI workload 的 fine-grained parallel processor：用大量小 tile + 超大分布式 SRAM + stateless all-to-all exchange + BSP 执行模型，来降低数据搬运能耗和内存成本。**

它的主张不是“我有最高 TOPS”，而是：

```text
AI 还在早期阶段，算法形态不稳定
未来模型会更大、更稀疏、更图结构化
后 Dennard 时代要让 memory 靠近 logic
后 Moore 时代要靠多芯片并行扩展
```

所以 IPU 的目标是：

```text
不要假设未来 AI 一定是大矩阵 dense GEMM
不要把所有资源押注在大 systolic array / Tensor Core 上
而是提供大量细粒度、可编程、近存储的计算 tile
```

这就是它和 GPU / TPU 设计哲学的根本差异。

---

# 2. IPU 的软件抽象：Graph + Codelet + Control Program

资料第 3 页给了 IPU 的软件抽象，这和 Programmer’s Guide 中的 Poplar 模型是一致的，但讲得更“体系结构化”。

IPU 软件抽象由四部分组成：

```text
1. 一个声明式、带 loop 的二分图
   compute vertices
   tensor vertices
   directed edges
   I/O pipes

2. 一组 atomic codelets
   定义 compute vertex 如何处理 tensor slice

3. 一个 control program
   条件执行 compute vertex sets

4. 一个 host program
   负责终止 / 管理 I/O pipes
```

这里的关键词是 **bipartite graph**：

```text
Tensor vertex 代表数据
Compute vertex 代表计算
Edge 代表数据依赖 / 数据流向
```

这和你现在设计的 **architecture DSL + event/schedule IR** 很有关系。Graphcore 的抽象其实就是：

```text
数据对象：tensor vertex
计算事件：compute vertex / codelet
控制流：control program
外部输入输出：I/O pipe
```

它不是普通 CPU 程序模型，也不是 GPU kernel-only 模型，而是强烈图化、静态化、编译器友好的模型。

---

# 3. IPU 的硬件抽象：many tiles + local memory + all-to-all exchange

资料第 4 页总结了硬件抽象：

```text
Many tiles
  每个 tile 包含 multi-threaded processor + local memory

Tiles communicate via an all-to-all, stateless exchange

Any codelet is executable atomically by a single tile thread

A tensor vertex may be distributed over many tiles

Bulk synchronous alternation between local compute and global communication
```

这里最重要的不是“all-to-all”这个词，而是 **stateless exchange**。

它意味着 Graphcore 并不把 tile 间通信建模成传统 NoC 的 packet queue / router arbitration / dynamic routing，而是更像：

```text
编译器预先知道谁在什么时候发
编译器预先知道谁在什么时候收
硬件按照精确周期搬运数据
```

这和传统 mesh NoC 非常不一样，后面第 16 页还会展开。

---

# 4. Colossus Mk1 vs Mk2：核心升级

第 5 页给了 Mk1 和 Mk2 的关键参数对比：

| 项目                   | Colossus Mk1 / GC2 | Colossus Mk2 / GC200 |
| -------------------- | -----------------: | -------------------: |
| 工艺                   |           TSMC N16 |              TSMC N7 |
| 晶体管                  |             23.65B |               59.33B |
| Tile 数               |               1216 |                 1472 |
| 每 tile SRAM          |            256 KiB |              624 KiB |
| 总 SRAM               |            304 MiB |              896 MiB |
| FP16 峰值              |        125 TFLOP/s |          250 TFLOP/s |
| 片上 memory bandwidth  |            62 TB/s |              62 TB/s |
| inter-tile bandwidth |           7.8 TB/s |             7.8 TB/s |
| inter-chip bandwidth |           320 GB/s |             320 GB/s |

最值得注意的是：

```text
Mk2 的计算翻倍：125 → 250 TFLOP/s
SRAM 容量大幅增加：304 MiB → 896 MiB
tile 数增加不算特别大：1216 → 1472
inter-tile / inter-chip 带宽基本没变
```

所以 Mk2 的重点不是简单堆更多 tile，而是：

```text
更大 tile memory
更高计算密度
更适合放下大 workspace / activation / model fragment
```

这和第 6 页 Mk1 的经验教训强相关。

---

# 5. Mk1 的经验教训：这页非常重要

第 6 页是整份资料里最有架构价值的一页。Graphcore 自己总结了 Mk1 的 lessons：

## 5.1 功能放得太多，软件没完全用起来

他们说 Mk1 作为 MVP 放了比软件生命周期内能点亮的更多功能，例如 sparse tensor arithmetic。

这说明：

```text
硬件特性 != 可用性能
软件栈成熟度决定架构价值释放程度
```

对 AI 加速器尤其关键。你做架构探索时，不能只看“硬件能不能做”，还要问：

```text
编译器能不能稳定生成？
框架能不能表达？
用户愿不愿意改代码？
调试 profiling 是否可接受？
```

## 5.2 256 KiB tile memory 太紧

他们明确说，花了大量时间调代码和 workspace 来适配 256 KiB tile memory。

这解释了为什么 Mk2 每 tile SRAM 增加到 624 KiB。

对 IPU 这种分布式 SRAM 架构来说，限制不是“总 SRAM 有多少”，而是：

```text
每个 tile local memory 是否足够容纳：
  code
  vertex state
  input slice
  output slice
  temporary workspace
  communication buffer
```

也就是说，**per-tile SRAM 容量是编译器映射难度的关键参数**。

## 5.3 大模型多芯片映射需要专家知识

Graphcore 承认，把大模型高效映射到多芯片需要 computer expertise，多数 AI programmer 需要 rich automation。

这其实是 IPU 路线的核心痛点：

```text
硬件越显式
确定性越强
但软件自动化要求越高
```

这和你之前讨论的“复杂性不会消失，只会转移”完全一致。

## 5.4 Whole-graph compilation 会越来越慢

他们说 whole-graph compilation 最初最简单，但随着模型变大，编译不可避免变慢。

这对 LLM 时代尤其关键。大图、动态 shape、长上下文、复杂 control flow 都会挑战全图编译。

## 5.5 BSP 对电源裕量优化不友好

这点非常有意思。

他们说 bulk synchrony 让 Vdd margin / supply transient 的调优更难。

直觉是：

```text
BSP 会导致大量 tile 在相近时间进入 compute 或 exchange
全芯片活动模式更同步
电流瞬态更集中
电源网络压力更大
```

这说明 BSP 带来确定性的同时，也可能带来 **power delivery / IR drop / transient current** 方面的挑战。

---

# 6. IPU-POD512 / M2000：系统级扩展

第 7 页介绍了 M2000 IPU-Machine：

```text
4 × Colossus Mk2 IPU
约 1 PFLOP/s peak
本地 proxy host
最大 512 GiB DDR
1.2 Tb/s inter-chassis breakout
1.5 kW TDP
典型应用约 1 kW
```

这说明 Graphcore 的扩展单元不是单卡 GPU 形式，而是 **disaggregated AI accelerator**：

```text
Host / proxy host / IPU machine / IPU-POD
```

这和 GPU 服务器的 tight CPU-GPU node 模式不同。Graphcore 显然希望通过 IPU-Fabric 组成更大的 Pod。

---

# 7. 和 GPU / TPU 的结构对比

第 8 页给了 IPU、GPU、TPU 的结构对比。Graphcore 想强调的是：

```text
IPU core 数多
SIMD width 小
on-die memory 巨大
off-die memory 不采用 HBM，而是 Host/DDR 路线
```

它的结论是：

> IPU is a fine-grained parallel processor with huge distributed SRAM on die.

这句话是 Graphcore 对自己架构的定位。

可以这样理解：

| 架构  | 粒度                    | 存储哲学                      | 典型优势                           |
| --- | --------------------- | ------------------------- | ------------------------------ |
| GPU | 中等粒度 SM + 大 SIMD/SIMT | HBM + cache/shared memory | dense tensor throughput + 通用生态 |
| TPU | 粗粒度 systolic array    | HBM + systolic dataflow   | 大矩阵乘效率                         |
| IPU | 细粒度 tile many-core    | 大量分布式 SRAM + 显式 exchange  | 不规则/稀疏/图结构/近存储并行               |

Graphcore 在这份资料中试图把 IPU 定位成 **未来 AI 算法不确定性下的更通用 AI processor**。

---

# 8. Die 结构：Tile memory 占大头

第 9 页展示了 Colossus Mk2 die 的结构：

```text
59.33B active transistors
7nm
823 mm²
1.325 GHz global mesochronous clock
23/24 tile redundancy
```

图中面积大致分成：

```text
Tile Memory
Tile Logic
Exchange
Uncore
Link
```

从饼图看，**Tile Memory 占据了很大比例**。这说明 IPU 是一个非常典型的：

```text
memory-heavy AI processor
```

不是“算力阵列为中心”，而是“分布式 memory + 近存储 compute”为中心。

**23/24 tile redundancy** 可以理解为为了良率做 tile 冗余：物理上可能有更多 tile，启用其中大部分，屏蔽少量坏 tile。这在 823mm² 大芯片上非常合理。

---

# 9. Tile processor：MAIN path + AUX path

第 10 页讲 tile processor 微架构，比 Programmer’s Guide 更具体。

每个 tile processor：

```text
32-bit instructions
single or dual issue
two execution paths
barrel threaded
```

两个执行路径：

## MAIN path

负责：

```text
control flow
integer / address arithmetic
multi-load/store
```

## AUX path

负责：

```text
floating-point arithmetic
vector operators
matrix operators
transcendentals
random number generation
```

并且 AUX path 可以和 MAIN path co-issue。

这说明 tile 不是简单 MAC PE，而是有相当完整的：

```text
控制流 + 地址生成 + load/store + scalar/vector/matrix + transcendental + RNG
```

所以 IPU tile 更像一个 AI-oriented microcore。

---

# 10. N+1 barrel threading：7 contexts，6 pipeline slots

第 11 页讲的是非常关键的 tile 内线程模型。

```text
7 program contexts
6 round-robin pipeline slots
```

其中：

## Supervisor program

```text
执行 control program 的一个 fragment
负责 orchestrate vertices 更新
在没有让给 worker 的 slots 中执行
通过 RUN 指令 dispatch worker
```

## Worker program

```text
就是一个 codelet
负责更新一个 vertex
在 1 个 slot 中运行
速度是 1/6 clock
通过 EXIT 把 slot 还给 supervisor
```

最关键的一句话是：

> Hiding the pipeline from Workers makes vertex execution easy for a compiler to predict, hence to load balance.

这点非常重要。

由于 worker 看不到复杂 pipeline，编译器更容易预测一个 vertex/codelet 的执行时间，从而更容易做 tile 间负载均衡。

这就是 IPU 的确定性设计：

```text
牺牲一些动态硬件调度能力
换取编译器可预测性
```

---

# 11. Sparse Load / Store：为稀疏和不规则数据准备

第 12 页强调：

```text
896 MiB on-die SRAM
47 TB/s data-side bandwidth
支持 arbitrarily-structured data，只要它 fit on chip
```

ld/st 支持 sparse gather，并且可以和 arithmetic 并行全速执行，通过 compact pointer lists：

```text
16-bit absolute offsets to a base
4-bit cumulative delta offsets to a base
```

这说明 Graphcore 设计时非常重视：

```text
稀疏访问
图数据
不规则 tensor slice
pointer-based gather
```

这和 TPU/GPU 主要围绕 dense GEMM 优化的路线不同。

但这里先不展开评价，因为后续我们可以专门讨论：

```text
这种 sparse load/store 在真实大模型 workload 中到底有多大价值？
对 Transformer / MoE / graph neural network / recommender 是否有优势？
```

---

# 12. MatMul / Conv：AMP/SLIC 数据路径

第 13 页给了 FP16/FP32 matmul 和 direct convolution 的计算能力：

| Multiply | Accumulate datapath | Accumulate memory |       Burst |
| -------- | ------------------- | ----------------- | ----------: |
| f16      | f32                 | f16               | 250 TFLOP/s |
| f16      | f32                 | f32               | 125 TFLOP/s |
| f32      | f32                 | f32               |  62 TFLOP/s |

它强调：

```text
f16 × f16 products
intermediates are f32
最后可 cast
```

这和训练相关，因为训练往往需要 f32 accumulation 或 stochastic rounding 来保持数值稳定。

这里还有两个术语值得后续细挖：

```text
AMP: Accumulating Matrix Product
SLIC: 可能是 slice/矩阵子结构相关的数据通路组织
```

目前这份材料没有把 AMP/SLIC 的底层阵列细节完全展开，后续可以结合更多白皮书或 ISA 文档补齐。

---

# 13. Random numbers and stochastic rounding

第 14 页说每个 tile 每 cycle 可以生成 128 random bits：

```text
private context per worker thread
enhanced xoroshiro128+ PRNG
6th-order Irwin Hall Gaussian shaper
```

支持指令：

```text
生成 uniform / Gaussian random vector
按指定概率 randomly puncture vector
对 down-cast 做 full-speed stochastic rounding
```

这主要是为了训练：

```text
dropout
随机采样
初始化
stochastic rounding
低精度训练稳定性
```

Graphcore 把这些做成硬件支持，体现了它面向训练 workload 的设计背景。

---

# 14. Global Program Order：BSP 执行顺序

第 15 页又回到 BSP，给了更精确的信息：

```text
Tile processors execute asynchronously until they need to exchange data.

BSP:
repeat { Sync; Exchange; Compute }

Each tile executes a list of atomic codelets in one compute phase.

Hardware global synchronization:
  ~150 cycles on chip
  15 ns / hop between chips
```

这里有几个关键点：

## 14.1 tile 内 compute 是异步推进的

不是所有 tile 每条指令 lock-step，而是：

```text
每个 tile 自己跑自己的 codelet list
直到需要 exchange
```

## 14.2 exchange 前有全局同步

这让程序全局顺序清晰：

```text
所有 tile 完成 compute
所有 tile sync
所有 tile exchange
再进入下一 compute
```

## 14.3 同步成本并不为零

片内约 150 cycles，跨芯片 15ns/hop。

对于大 compute phase，这个开销可能很小；但如果 compute phase 很碎，BSP barrier 就可能变成明显开销。

---

# 15. Exchange Mechanics：这页是理解 IPU fabric 的关键

第 16 页非常重要，它解释了 IPU 的 exchange 并不是普通动态 NoC。

关键参数和机制：

```text
Exchange spine: 1600 × 36b
每个 tile 和 I/O block 有一个 36b pipelined send channel
每个 tile 有一个 1600-way receive mux
每 tile 32b/cycle send and receive
```

最重要的描述是：

```text
Poplar compiler schedules transmit, receive and select at precise cycles from sync,
knowing all pipeline delays.
```

也就是说：

```text
从 sync 开始算第几个 cycle
哪个 tile 发
哪个 tile 收
receive mux 选择哪一路
这些都由编译器静态排好
```

它还明确说：

```text
No queues
No arbiters
No packet overheads
Just data moving at full bandwidth and minimum energy
```

这和传统 NoC 的差异非常大。

## 15.1 传统 NoC 思路

```text
packet/flit
router
buffer/VC
routing
arbitration
flow control
可能出现 contention
```

## 15.2 IPU Exchange 思路

```text
静态时分调度
确定性 pipeline delay
receiver mux 按时间选择
没有动态队列/仲裁
没有 packet header overhead
```

所以它更接近：

```text
compiled communication fabric
```

而不是 runtime-routed packet NoC。

这点后面非常值得和 Tenstorrent NoC、Cerebras wafer fabric、Groq deterministic routing 做对比。

---

# 16. Chip Power：Graphcore 的能耗论证

第 17 页给了 pJ/flop 数据：

| Multiply | Accumulate datapath | Accumulate memory | pJ/flop |
| -------- | ------------------- | ----------------- | ------: |
| f16      | f32                 | f16               |     1.3 |
| f16      | f32                 | f32               |    1.75 |
| f32      | f32                 | f32               |     3.3 |

并给出两点解释：

```text
Distributed SRAM keeps most on-die transport to <1mm.

Large SRAM collapses required DRAM bandwidth,
and moves DRAM power out of the logic die thermal envelope.
```

这就是 Graphcore 的核心能耗主张：

```text
数据搬运比计算贵
所以要把数据放在离 compute 很近的 SRAM
并减少 HBM/DRAM 参与
```

注意它不是说“不需要 DRAM”，而是说：

```text
带宽靠 SRAM
容量靠 DDR
```

---

# 17. System Power：系统级能效对比

第 18 页用 IPU Pod16、DGX、TPUv3 Pod16 做了系统级对比：

| 项目                | IPU Colossus Mk2 Pod16 |     A100 DGX |  TPUv3 Pod16 |
| ----------------- | ---------------------: | -----------: | -----------: |
| Chip TDP          |                   300W |     400/500W |         450W |
| Chips in system   |                     16 |            8 |           16 |
| System TDP        |                  7000W |        6500W |        9300W |
| System W/chip     |                   437W |         812W |         581W |
| System burst FP16 |           4000 TFLOP/s | 2496 TFLOP/s | 1968 TFLOP/s |
| Nominal TFLOP/W   |                   0.57 |         0.38 |         0.21 |

Graphcore 的结论是：

```text
IPU 相比 GPU 有约 1.5x net efficiency advantage
这暗示大约 3x transport energy advantage
```

这里要注意，这属于厂商宣讲口径，后续我们需要结合真实 workload、软件成熟度、利用率来批判性分析。

---

# 18. Why No HBM：Graphcore 对 HBM 的反对逻辑

第 19 页是 Graphcore 架构选择的核心辩护。

他们的观点：

```text
Memory capacity determines what an AI can do;
bandwidth just limits how fast.
```

他们认为 GPU/TPU 用 HBM 同时解决 capacity 和 bandwidth，但 HBM 有问题：

```text
昂贵
容量有限
给 processor thermal envelope 增加 100W+
```

IPU 的方案是：

```text
用 SRAM 解决 bandwidth
用 DDR 解决 capacity
```

也就是：

```text
Bandwidth problem → on-die SRAM
Capacity problem → off-chip DDR / host DDR / Streaming Memory
```

这点非常适合和你之前关注的 HBM、CPO、端侧大模型带宽问题结合讨论。

---

# 19. DRAM Economics：不用 HBM 的成本论证

第 20 页继续从经济性论证：

```text
40GB HBM 约会让 packaged reticle-sized processor 成本变成 3 倍
DDR-based systems 可以把省下的钱花在更多 processors 上
```

它还给了一个硅面积对比：

```text
8 cm² processor + 32GB DRAM 所需总 silicon：

HBM2: 53 cm²
DDR4: 23 cm²
```

Graphcore 的商业逻辑是：

```text
与其把钱花在昂贵 HBM 上
不如用 DDR 做容量
把成本预算投入更多 IPU 芯片
```

这个观点在 2021 年有其逻辑，但到了 LLM 推理时代，需要重新评估：

```text
decode 阶段权重/KV cache 带宽压力
batch size
模型大小
SRAM 容量是否足以 collapse DRAM bandwidth
DDR streaming weights 是否可接受
```

后续我们可以专门批判这部分。

---

# 20. Placing Model State：不同规模模型怎么放

第 21 页讨论 model state 和 optimizer state 的放置：

```text
Small model:
  model state 放 processor/SRAM
  optimizer state 可放 DRAM

Intermediate model:
  sharding + pipeline
  model state 分布在多个 processor
  optimizer state 在 DRAM

Large model:
  model state 和 optimizer state 都可能更多依赖 DRAM
  streaming weight states
```

Graphcore 的主张：

```text
Off-chip DDR bandwidth suffices for:
  distributed optimizer states at all scales
  streaming weight states for large models
```

这说明它假设：

```text
训练中 optimizer state 很大，但访问频率/带宽需求可以被 DDR 支撑
大模型权重可以 stream
SRAM 用于计算 working set
```

这和 GPU/HBM 逻辑完全不同。GPU 更倾向：

```text
模型权重、activation、KV cache 尽可能放 HBM
HBM 同时承担容量和带宽
```

---

# 21. SRAM 如何 collapse DRAM bandwidth：第 22 页的公式

第 22 页很关键，它给了一个粗略推理模型。

场景：

```text
inference with model streamed from DRAM
所有值按 2 Bytes 计算
```

图里设：

```text
n samples
Q quanta
F features
w fragment << nQF
W weights in DRAM
```

片上 SRAM 需要：

```text
4nQF Bytes SRAM
```

计算量：

```text
2nQw flops
```

最后给了关系：

```text
DRAM bandwidth (Bytes/s)
= 4F × Compute rate (flop/s) / SRAM Capacity (Bytes)
```

核心直觉是：

> SRAM 越大，能同时缓存/复用的数据块越大，则为了维持同样 compute rate 所需的 DRAM bandwidth 越低。

这就是 Graphcore 的“SRAM collapse DRAM bandwidth”论证。

换成架构语言：

```text
大 SRAM 增加 temporal reuse window
增大 on-chip working set
降低 off-chip weight streaming pressure
```

不过这个模型非常粗糙，后续需要结合真实 Transformer 推理来修正：

```text
prefill GEMM 是否有足够 batch/sequence 复用权重？
decode GEMV 是否缺少权重复用？
KV cache 是否能 fit SRAM？
GQA/MQA 对 KV cache 带宽有何影响？
```

---

# 22. Hardware helping Software：硬件如何帮助软件快速演进

第 23 页是 Graphcore 自我总结：

```text
Native graph abstraction
Codelet-level parallelism
Pipeline-oblivious threads
BSP eliminates concurrency hazards
Stateless all-to-all Exchange
Cacheless, uniform, near/far memory
```

它还展示了 SDK 在 7 个月内对 BERT-L、RN50、EfficientNet-B4 的 tuning 提升。

这页想表达：

```text
硬件机制简单、确定、无 cache、无动态 concurrency hazard
所以软件栈可以快速优化
```

但这里也有另一面：

```text
软件必须承担 mapping / scheduling / liveness / exchange 的复杂性
```

所以它是一个典型的 **hardware-software co-design tradeoff**。

---

# 23. Key Takeaways：官方总结

第 24 页的官方结论是：

```text
Colossus 是 Graphcore 对新型 AI processor 架构 IPU 的实现

IPU 最小化数据搬运能耗
从而在功耗预算内放更多 processor silicon

IPU 最小化 AI 模型的内存成本
从而在成本预算内放更多 processor silicon

IPU 的细粒度并行减少了对未来 AI 模型和数据并行性的假设
```

这就是 Graphcore 的完整架构叙事：

```text
AI 未来不确定
所以不要只为 dense GEMM 优化
要用 fine-grained programmable parallelism
要用 distributed SRAM 降低数据搬运
要用 DDR 降低容量成本
要用 BSP + static exchange 提高确定性和软件可控性
```

---

# 24. 这份资料相比 Programmer’s Guide 新增了哪些信息？

| 主题       | Programmer’s Guide                               | HC2021 资料                                                      |
| -------- | ------------------------------------------------ | -------------------------------------------------------------- |
| 目标       | 解释怎么编程、怎么执行                                      | 解释为什么这样设计                                                      |
| IPU 抽象   | Poplar program / variable / compute set / vertex | graph abstraction / codelet / control program / I/O pipe       |
| Tile     | multi-threaded processor + local memory          | MAIN/AUX path、dual issue、barrel threading                      |
| Thread   | supervisor / worker                              | 7 contexts、6 slots、worker 1/6 clock                            |
| Exchange | BSP exchange phase                               | 1600×36b spine、receiver mux、no queues/arbiters/packet overhead |
| Memory   | In-Processor-Memory + Streaming Memory           | 为什么不用 HBM、SRAM collapse DRAM bandwidth                         |
| 系统       | IPU-Machine / IPU-Fabric                         | M2000、POD512、系统功耗/成本论证                                         |
| 架构哲学     | 显式搬运 + BSP                                       | 降低 transport energy、减少对未来模型并行性的假设                              |

---

# 25. 目前我建议暂存的“后续讨论问题”

等你资料收集完，我们可以围绕这些问题展开：

## A. IPU 的 exchange fabric 到底是不是 NoC？

它没有 queue、arbiter、packet overhead，更像 compiled TDM fabric。
后续可以和：

```text
Tenstorrent NoC
Cerebras Swarm fabric
Groq deterministic fabric
FlexNoC
传统 mesh NoC
```

做对比。

## B. BSP 是优势还是限制？

优势：

```text
确定性强
调试/profiling 清晰
通信可静态调度
concurrency hazard 少
```

限制：

```text
barrier 等待
load imbalance 敏感
细碎算子同步开销高
异步 overlap 能力有限
供电瞬态压力更集中
```

## C. 大 SRAM + DDR 替代 HBM 的路线，在 LLM 时代是否还成立？

这个要重点讨论：

```text
训练 vs 推理
prefill vs decode
dense GEMM vs GEMV
KV cache
batch size
long context
MoE
权重复用窗口
```

## D. Fine-grained parallelism 对未来 AI 是否真的更稳健？

Graphcore 认为未来 AI 有 sparse / graph / irregular data，所以 IPU 更合适。
但现实是 2020 之后 Transformer dense GEMM/HBM 生态成为主流。

这里需要分析：

```text
架构前瞻判断是否正确？
软件生态是否拖累？
LLM workload 是否偏离 IPU 假设？
还是 IPU 的商业/系统路线出了问题？
```

## E. 对你的架构 DSL / event IR 的启发

IPU 可以抽象成：

```text
Architecture DSL:
  Tile
  Local SRAM
  MAIN/AUX datapath
  Exchange spine
  Receive mux
  Streaming Memory
  IPU-Link

Event IR:
  ComputeSet
  Vertex
  Codelet
  Copy
  Sync
  Exchange
  StreamCopy
  AllReduce
  Pipeline stage
  Recomputation
```

这会是一个非常好的参考案例。

---

**一句话总结：HC2021 这份资料把 Graphcore IPU 的“设计哲学”讲清楚了：它不是追求最大矩阵阵列，而是用细粒度 many-tile、超大分布式 SRAM、静态 exchange 和 BSP 执行模型，试图在功耗、成本和未来 AI workload 不确定性之间找到另一条路线。**
