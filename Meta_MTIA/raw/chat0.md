> 我现在继续了解Meta MTIA v1/v2相关的信息

你这个方向很值得继续，而且 **Meta MTIA 比 NeuronCore 更适合做“公开资料级别的架构拆解”**：Meta 的 ISCA’23 / ISCA’25 论文给出了 PE 内部结构、NoC、SRAM/LPDDR 层次、异步 dataflow 执行模型、PyTorch eager mode 支持，以及模型-芯片协同优化经验。相比 AWS NeuronCore，MTIA 公开得更“体系结构论文化”。✨

下面我按你真正关心的方式整理：**资料来源 → 架构主线 → 控制流/数据流 → v1/v2 对比 → 如何学习**。

---

# 1. 先明确命名：MTIA v1 / v2 / 2i / 100 / 200

Meta 的命名有点混乱：

| 常见叫法                        | 更正式/后续叫法               | 主要定位                                                       |
| --------------------------- | ---------------------- | ---------------------------------------------------------- |
| **MTIA v1**                 | **MTIA 1 / MTIA 100**  | 第一代推荐系统推理加速器                                               |
| **Next-gen MTIA / MTIA v2** | **MTIA 2i / MTIA 200** | 第二代推荐系统推理加速器，已大规模部署                                        |
| **MTIA 300/400/450/500**    | 后续路线                   | 300 面向 R&R training，400/450/500 转向 GenAI / GenAI inference |

Meta 2026 年博客明确说，ISCA’23 和 ISCA’25 两篇论文分别介绍了前两代 MTIA：**MTIA 100 和 MTIA 200，也就是原来的 MTIA 1 和 MTIA 2i**；后续 MTIA 300/400/450/500 则是更快节奏的后续芯片路线。([AI.Meta][1])

---

# 2. 最值得收集的资料

## S 级：必须收集

| 资料                                                                                           | 类型                    | 为什么重要                                                |
| -------------------------------------------------------------------------------------------- | --------------------- | ---------------------------------------------------- |
| **MTIA: First Generation Silicon Targeting Meta’s Recommendation Systems**                   | ISCA’23 论文            | v1 架构主资料，讲 accelerator/platform/software/performance |
| **MTIA v1: Meta’s first-generation AI inference accelerator**                                | Meta 官方博客             | v1 的公开规格、PE grid、memory、system design                |
| **Next Gen MTIA — Recommendation Inference Accelerator**                                     | Hot Chips 2024 slides | v2/2i 的公开架构演讲，规格和改进点集中                               |
| **Meta’s Second Generation AI Chip: Model-Chip Co-Design and Productionization Experiences** | ISCA’25 论文            | v2/2i 最重要资料，包含 PE 内部结构、NoC、memory hierarchy、生产化经验    |
| **Four MTIA Chips in Two Years**                                                             | Meta 2026 博客          | MTIA 300/400/450/500 路线图，理解后续从 R&R 到 GenAI 的转向       |

其中 **ISCA’25 的 MTIA 2i 论文** 是当前最值得精读的资料。它不是单纯宣传，而是直接讲了 MTIA 2i 的 PE 内部结构、NoC、Local Memory、DPE、RE、SIMD Engine、Command Processor、Fabric Interface，以及模型协同优化和生产化问题。([aisystemcodesign.github.io][2])

---

# 3. MTIA 的核心设计点：不是 LLM 芯片，而是 R&R / DLRM 芯片

MTIA v1/v2 的主战场不是大模型通用训练，也不是 GPT 类 LLM 推理，而是 **ranking and recommendation，尤其是 Meta 内部 DLRM / recommendation inference**。

Meta v1 博客说，MTIA v1 是为 Meta 内部 workload 设计的第一代 ASIC，目标是服务 recommendation models；它和 silicon、PyTorch、recommendation models 做了 full-stack co-design。([AI.Meta][3])

ISCA’25 论文也明确说，MTIA 2i 面向 Meta 四类 AI workload 中的 **recommendation model inference**，这是 Meta 当前推理 workload 的主要部分；对于不适合 MTIA 2i 的模型，Meta 仍然使用 GPU。([aisystemcodesign.github.io][2])

这点很关键：

```text
MTIA v1/v2 的设计目标不是：
  尽可能高的 dense GEMM 峰值
  尽可能强的 LLM decode
  替代所有 GPU workload

而是：
  用更低 TCO / 更好 perf/W
  服务 Meta 内部大量 recommendation inference
  特别是 batch 受限、embedding/MLP/feature interaction 混合的模型
```

---

# 4. MTIA v1 架构心智模型

Meta v1 官方博客给出的高层结构非常清楚：

```text
MTIA v1
  ├── Control subsystem
  │   └── firmware，管理 compute/memory resources，和 host 通信，编排 job execution
  ├── 8x8 PE grid
  │   └── 64 个 Processing Elements
  ├── Shared on-chip SRAM
  │   └── 128 MB，所有 PE 共享
  ├── Off-chip LPDDR5
  │   └── 最多 128 GB
  └── Mesh interconnect
      └── 连接 PEs、memory blocks
```

v1 的公开规格包括：TSMC 7nm、800 MHz、102.4 TOPS INT8、51.2 TFLOPS FP16、TDP 25W；芯片由 PE grid、on-chip/off-chip memory 和 interconnect 组成。([AI.Meta][3])

每个 PE 有两个 RISC-V processor cores，其中一个带 vector extension，还有若干 fixed-function units，用于矩阵乘、accumulation、data movement、nonlinear function 等关键操作；每个 PE 有 128KB local SRAM。([AI.Meta][3])

所以 v1 可以抽象成：

```text
Host CPU
  ↓ PCIe
MTIA control subsystem / firmware
  ↓ job orchestration
8x8 PE grid
  ├── PE local SRAM: 128 KB / PE
  ├── RISC-V scalar core
  ├── RISC-V vector core
  ├── fixed-function units
  └── mesh / shared SRAM / LPDDR5
```

和 NeuronCore 的差异是：**MTIA 更像一个 8x8 many-PE accelerator，而不是单个巨大异构 core**。

---

# 5. MTIA 2i / v2 架构：公开信息更有价值

MTIA 2i 仍然是 8×8 PE array，但整体规格和 PE 内部都增强很多。ISCA’25 表格给出：TSMC 5nm、1.35GHz、85W TDP、PCIe Gen5 x8、177 TFLOPS FP16/BF16、354 TOPS INT8；每 PE local memory 从 128KB 增到 384KB；shared on-chip SRAM 从 128MB 增到 256MB；LPDDR5 从 32–64GB 增到 64–128GB；on-chip SRAM bandwidth 从 0.8TB/s 增到 2.7TB/s；off-chip LPDDR5 bandwidth 为 204.8GB/s。([aisystemcodesign.github.io][2])

| 维度                  |                         MTIA v1 |          MTIA 2i / v2 |
| ------------------- | ------------------------------: | --------------------: |
| 工艺                  |                        TSMC 7nm |              TSMC 5nm |
| 频率                  |                          800MHz |               1.35GHz |
| PE 阵列               |                             8×8 |                   8×8 |
| 每 PE Local Memory   |                           128KB |                 384KB |
| Shared on-chip SRAM |                           128MB |                 256MB |
| LPDDR5              |     32–64GB / 最高 128GB 公开口径略有差异 |              64–128GB |
| On-chip SRAM BW     |                         0.8TB/s |               2.7TB/s |
| LPDDR5 BW           |                         176GB/s |             204.8GB/s |
| FP16/BF16 GEMM      |                51.2 TFLOPS FP16 |  177 TFLOPS FP16/BF16 |
| INT8 GEMM           |                      102.4 TOPS |              354 TOPS |
| TDP                 | 25W typical / 35W board-level口径 | 65W typical / 85W TDP |

Meta 官方 2024 博客也强调，v2 相比 v1：dense compute 约 3.5×，sparse compute 约 7×，PE local storage 约 3×，on-chip SRAM 翻倍且带宽 3.5×，LPDDR5 容量翻倍，NoC 带宽翻倍。([AI.Meta][4])

---

# 6. MTIA 2i 的 PE 内部结构：这部分最值得你精读

ISCA’25 论文公开了 PE 内部模块：

```text
MTIA 2i PE
  ├── RISC-V scalar core
  ├── RISC-V vector core, 64B wide vector extension
  ├── Local Memory, 384 KB
  ├── Command Processor, CP
  ├── Memory Layout Unit, MLU
  ├── Dot Product Engine, DPE
  ├── Reduction Engine, RE
  ├── SIMD Engine, SE
  ├── Fabric Interface, FI
  └── PE Interconnect
```

论文说明：每个 PE 包含两个 RISC-V cores 和相关外设，以及一组专门用于计算或数据搬运的 fixed-function units；PE local memory 为 384KB；PE Interconnect 连接 RISC-V cores、peripherals、Local Memory 和 custom hardware blocks；fixed-function units 可以直接访问 Local Memory，并形成 coarse-grained pipeline，数据可以从一个 unit 传到下一个 unit。([aisystemcodesign.github.io][2])

这句话非常有价值，因为它说明 MTIA PE 不是“RISC-V + 外设”那么简单，而是：

```text
RISC-V cores 负责控制 / 发指令
fixed-function units 负责高吞吐数据通路
Local Memory 是 PE 内数据暂存中心
Command Processor 负责依赖检查和调度
FI 负责 PE 内 Local Memory 和 NoC/SRAM/LPDDR 之间的数据搬运
```

---

# 7. MTIA 2i 的控制流：RISC-V 生成 custom instructions，CP 调度 fixed-function units

这是你最关心的部分。

MTIA 2i 的执行模型不是 GPU 那种 warp/SIMT，也不是纯 systolic array。ISCA’25 论文直接说：**PE 上的执行遵循 asynchronous dataflow model**。程序员为 RISC-V cores 写代码，由 RISC-V cores 生成一系列 custom instructions，这些 instructions 在 fixed-function units 上以 dataflow 方式执行；DMA transfers 和 computations 在依赖满足后发生。([aisystemcodesign.github.io][2])

控制流可以抽象成：

```text
Kernel code on RISC-V cores
  ↓
RISC-V scalar/vector core 执行控制代码
  ↓
生成 custom instructions
  ↓
Command Processor 接收 custom instructions
  ↓
CP 负责：
  - fixed-function units 的 dependency checking
  - scheduling
  - custom instruction tracking
  - Local Memory arbitration
  - circular buffer abstraction
  ↓
DPE / RE / SE / MLU / FI 异步执行
```

Command Processor 的作用很关键：它编排 fixed-function units，做 dependency checking、scheduling、tracking，并仲裁 RISC-V cores 和 fixed-function units 对 Local Memory 的访问；它还提供 Circular Buffer abstraction，并把 circular buffer 管理和 dependency tracking 下沉到硬件。([aisystemcodesign.github.io][2])

所以 MTIA 2i 的控制流可以总结为：

> **RISC-V cores 是可编程控制面，Command Processor 是 PE 内 micro-scheduler，fixed-function units 是数据面。**

这和你之前理解的 NeuronCore 不同：

| 架构                 | 控制流心智模型                                                          |
| ------------------ | ---------------------------------------------------------------- |
| NeuronCore-v2      | 多 engine sequencer，各 engine 有独立 instruction stream               |
| MTIA 2i PE         | RISC-V cores 生成 custom instructions，CP 调度 fixed-function units   |
| Tenstorrent Tensix | RISC-V-like/control core + unpack/math/pack/data movement engine |
| SambaNova RDU      | compiler 生成 dataflow pipeline，PCU/PMU/switch 执行                  |
| Graphcore IPU      | tile processor 执行 vertices，BSP exchange 同步                       |

---

# 8. MTIA 2i 的数据流：Local Memory → DPE → RE → SE/neighbor PE

PE 内部的数据流也比较清楚。

## 8.1 GEMM / FC 类数据流

DPE 用于 GEMM。论文说 DPE 操作两个 tensor：第一个 tensor 被读取并缓存在 DPE 中，第二个 tensor 从 Local Memory streaming 进入，与第一个 tensor 的所有 rows 做 dot product；DPE 包含两个 32×32B×32 MAC tiles，FP16/BF16 输入、FP32 输出时每 PE 约 2.76 TFLOPS/s；还支持 2:4 weight sparsity，理论额外 2× throughput。([aisystemcodesign.github.io][2])

抽象成：

```text
Local Memory
  ├── tensor A -> read/cache in DPE
  └── tensor B -> stream into DPE
       ↓
DPE: dot product / GEMM
       ↓
RE: accumulation / reduction
       ↓
SE or neighboring PE or Local Memory
```

## 8.2 Reduction / activation / nonlinear 数据流

Reduction Engine 存储矩阵乘结果并累加；它可以接收/累加结果、转发到邻近 PE，或者把结果传到 SIMD Engine 做后续计算，比如 activation。([aisystemcodesign.github.io][2])

SIMD Engine 用于 vector operations，包括 quantization 和 nonlinear functions；它可以从 RE 接收数据，也可以直接从 Local Memory 读取，还带有 lookup tables 用于近似 nonlinear functions。([aisystemcodesign.github.io][2])

因此一个 FC + activation 可以抽象为：

```text
FI/DMA:
  shared SRAM / LPDDR / NoC -> PE Local Memory

DPE:
  Local Memory operands -> GEMM

RE:
  accumulate partial sums
  possibly reduce across neighboring PEs

SE:
  quantization / nonlinear / activation

FI/DMA:
  output -> NoC / shared SRAM / LPDDR
```

## 8.3 MLU：layout transformation 单元

MLU 用于 transpose、concatenate、reshape 等 memory-layout transformation。([aisystemcodesign.github.io][2])

这说明 MTIA 2i 很重视 recommendation / PyTorch workload 中实际存在的 layout 问题，不只是矩阵乘。对你的架构探索来说，这个模块很值得关注：**很多 accelerator 论文忽略 layout transform，但生产模型里它会实实在在消耗 latency 和 memory bandwidth**。

---

# 9. MTIA 的 NoC 和 memory hierarchy：SRAM 是设计核心，LPDDR 是容量补充

MTIA 2i 的 NoC 连接 64 PEs、host、Control Core 和 memory subsystem；NoC 通过 die 四边的 crossbars 连接到 shared on-chip SRAM 和 off-chip memory controllers。论文说 NoC 是 non-blocking architecture，用来减少不同 initiators 之间的干扰；源端使用 flow control，并通过 leaky-bucket traffic shaping 和 packet fragmentation 平滑流量、防止突发拥塞。([aisystemcodesign.github.io][2])

这比很多厂商资料有用得多，因为它至少告诉你：

```text
NoC 不是只为 PE-to-PE 服务；
它也服务：
  PE ↔ shared SRAM
  PE ↔ LPDDR memory controllers
  PE ↔ host/control subsystem
  PE ↔ PE
```

MTIA 2i 的 memory hierarchy 很有特点：

```text
PE Local Memory:
  384KB / PE
  固定功能单元近端数据暂存，供 DPE/RE/SE/MLU/FI 使用

Shared on-chip SRAM:
  256MB
  全 PE 共享
  2.7TB/s
  可被分为 hardware-managed cache LLC 和 software-managed scratch LLS

LPDDR5:
  64–128GB
  204.8GB/s
  用于大容量模型参数/embedding/数据
```

ISCA’25 论文明确说，MTIA 2i 没用 HBM，而是使用 large SRAM + LPDDR DRAM，以降低成本和功耗；SRAM 提供 2.7TB/s，而 LPDDR 只有 204GB/s，约 13×差距。([aisystemcodesign.github.io][2])

更关键的是，256MB global SRAM 可以按 32MB granularity 分成：

```text
LLC: hardware-managed cache
LLS: software-managed scratch memory
```

Meta 的 autotuning framework 会自动决定 LLS/LLC 的大小，典型策略是：让 LLS 容纳整个 activation buffer，剩余 SRAM 用作 LLC；如果 activation buffer 放不下，则比较较小 batch size 的 LLS 方案和当前 batch size 的 LLC 方案。([aisystemcodesign.github.io][2])

这个设计非常值得你学习，因为它不是“全硬件 cache”也不是“全软件 scratchpad”，而是一个混合方案：

| 层级              | 类型                       | 作用                                |
| --------------- | ------------------------ | --------------------------------- |
| PE Local Memory | 显式近端 scratch             | 供 PE 内 fixed-function pipeline 使用 |
| LLS             | software-managed scratch | pin activation buffer，避免写回/污染     |
| LLC             | hardware-managed cache   | 缓存 weights / sparse accesses      |
| LPDDR5          | 大容量 DRAM                 | 放模型/embedding/不常驻数据               |

---

# 10. 为什么 MTIA v2 适合 R&R，但不适合 LLM decode？

这点对你很关键。

ISCA’25 论文说，MTIA 2i 设计点依赖 recommendation models 的 locality；相对较小的模型有显著 locality，可以最大化 SRAM reuse。论文也明确承认，当模型复杂度和大小超过 SRAM 容量时，有限的 off-chip bandwidth 会导致性能急剧下降；他们认为 2 GF/sample 对 MTIA 2i 不可达，因为 GEMM 会变成 DRAM bandwidth-bound，dense/sparse networks 的 latency 也会过高。对于 Llama2-7B，论文说 prefill 满足 600ms TTFT，但 decode 无法满足每 token 60ms 的 latency requirement。([aisystemcodesign.github.io][2])

这句话非常重要：

```text
MTIA 2i 的强项：
  推荐模型中较强 locality
  小 batch 但 activation / working set 可放 SRAM
  embedding + dense MLP + feature interaction 的混合 workload
  低 TCO / perf/W

MTIA 2i 的弱项：
  超大模型权重/KV/activation 超出 SRAM
  LPDDR bandwidth 不足
  LLM decode 每 token 重复读取大量权重/KV，容易 bandwidth-bound
```

这和你之前对 LLM decode 的判断一致：**decode 阶段不是看峰值 FLOPS，而是看权重/KV/cache/data movement 的有效带宽与 batch/reuse。**

---

# 11. 模型-芯片协同：MTIA 论文真正有价值的地方

MTIA 2i 论文最有价值的不只是 PE 结构，而是生产化经验。

Meta 说，为了适应 MTIA 2i，他们做了：

* data placement autotuning；
* kernel tuning；
* batch size autotuning；
* request coalescing；
* model sharding；
* graph fusion；
* SRAM hit-rate 优化；
* model changes rejection / model transformation。

例如论文说，dense networks 可以通过 buffer placement、graph optimization、fusion、GEMM 的额外 LLC tiling 等达到超过 95% SRAM hit rate；sparse networks 也可以让 40–60% accesses 命中 SRAM。([aisystemcodesign.github.io][2])

对于复杂 GEMM，论文描述了一个很硬核的优化：activation 通常能放在 PE distributed Local Memory 中，于是他们 preload activations from LLS，并 across PE columns broadcast weights，解耦 activation loading 和 weight loading，减少 PE contention；通过硬件 broadcast read 支持消除读取 weights 时的 NoC contention，并把 weight tiles 预取到 LLC 隐藏 DRAM latency，某些形状 latency 提升 45%，并达到超过 95% DRAM bandwidth。([aisystemcodesign.github.io][2])

这对你的架构探索方法非常有启发：

```text
不是芯片规格决定一切，而是：
  模型结构
  activation liveness
  SRAM placement
  operator scheduling
  graph fusion
  DMA schedule
  NoC contention
  PE Local Memory capacity
  request coalescing
共同决定最终线上 P99 latency / throughput / TCO
```

---

# 12. 与 Graphcore / Cerebras / SambaNova / NeuronCore 的对比

| 维度         | MTIA v1/v2                       | Graphcore IPU                       | Cerebras WSE           | SambaNova RDU            | AWS NeuronCore             |
| ---------- | -------------------------------- | ----------------------------------- | ---------------------- | ------------------------ | -------------------------- |
| 基本目标       | Meta R&R inference               | graph/dataflow ML                   | wafer-scale AI/HPC     | reconfigurable dataflow  | cloud Trainium/Inferentia  |
| 基本单元       | 8×8 PE grid                      | tile                                | PE on wafer            | PCU/PMU/switch           | NeuronCore                 |
| PE/core 控制 | RISC-V cores + CP                | tile processor                      | PE program/routing     | compiler dataflow        | per-engine sequencer       |
| 数据面        | DPE/RE/SE/MLU/FI                 | local SRAM + exchange               | PE mesh/dataflow       | dataflow pipeline        | Tensor/Vector/GPSIMD       |
| 存储         | PE LM + shared SRAM + LPDDR      | distributed SRAM + streaming memory | massive SRAM + MemoryX | SRAM + HBM + DDR         | SBUF/PSUM + HBM            |
| 互连         | custom NoC + crossbars           | Exchange fabric                     | wafer-scale fabric     | programmed switch fabric | device/interconnect fabric |
| 编译/软件      | PyTorch eager + kernels/autotune | Poplar                              | CSL                    | SambaFlow                | Neuron SDK/NKI             |
| 强项         | 低 TCO R&R inference              | graph ML                            | extreme scale on wafer | operator fusion          | AWS cloud integration      |

一句话：

> **MTIA 更像“面向 Meta recommendation workload 的 RISC-V-controlled many-PE accelerator”：PE 内部是 Local Memory + fixed-function dataflow pipeline，PE 外部是 shared SRAM/LPDDR/NoC，软件栈围绕 PyTorch eager、kernel tuning 和模型协同优化展开。**

---

# 13. 建议你的资料文件夹分类

```text
Meta-MTIA/
  00-index/
    reading-order.md
    mtia-v1-v2-spec-table.md
    compare-with-graphcore-cerebras-sambanova-neuron.md

  01-mtia-v1/
    ISCA-2023-MTIA-first-generation/
    Meta-blog-MTIA-v1/
    architecture-notes.md

  02-mtia-v2-2i/
    ISCA-2025-model-chip-codesign/
    HotChips-2024-next-gen-mtia/
    Meta-blog-next-generation-mtia/
    architecture-notes.md

  03-pe-microarchitecture/
    risc-v-control-cores.md
    command-processor.md
    local-memory.md
    dpe-dot-product-engine.md
    re-reduction-engine.md
    se-simd-engine.md
    mlu-memory-layout-unit.md
    fi-fabric-interface.md

  04-memory-hierarchy/
    pe-local-memory.md
    shared-sram-lls-llc.md
    lpddr5.md
    data-placement-autotuning.md

  05-noc-and-data-movement/
    noc-crossbars.md
    source-flow-control.md
    packet-fragmentation.md
    broadcast-read.md
    weight-prefetch.md

  06-software-stack/
    pytorch-eager-mode.md
    kernel-generation.md
    graph-fusion.md
    request-coalescing.md
    model-sharding.md

  07-workload-analysis/
    dlrm.md
    hstu.md
    dhen.md
    llm-limitations.md
    prefill-vs-decode.md

  08-roadmap/
    mtia-300.md
    mtia-400.md
    mtia-450.md
    mtia-500.md
```

---

# 14. 推荐阅读顺序

## 第一阶段：建立全局模型

先读：

```text
1. Meta 官方 MTIA v1 blog
2. Meta 官方 next-gen MTIA blog
3. Hot Chips 2024 Next Gen MTIA slides
```

目标回答：

```text
MTIA 为什么不是 GPU？
为什么它服务 R&R inference？
为什么使用 SRAM + LPDDR，而不是 HBM？
为什么仍然保留 PyTorch eager mode？
```

## 第二阶段：精读 v2/2i 微架构

重点读 ISCA’25：

```text
1. Table 2: MTIA 2i vs MTIA 1 specs
2. Section 3.1: Architecture Overview
3. Section 3.2: Processing Element
4. Section 3.3: New Features
5. Section 3.6: Unique Memory Hierarchy
```

目标画出：

```text
PE control flow:
  RISC-V -> custom instructions -> CP -> fixed-function units

PE data flow:
  FI -> Local Memory -> DPE -> RE -> SE/neighbor PE -> FI

Chip data flow:
  LPDDR -> shared SRAM/LLC/LLS -> NoC -> PE Local Memory
```

## 第三阶段：读模型-芯片协同

重点读：

```text
Section 4: Model-Chip Co-design
Section 6: case study
Section 7: production results
Section 8: limitations
```

目标回答：

```text
为什么 activation buffer 要 pin 在 LLS？
为什么 graph fusion 能减少 working set？
为什么复杂 GEMM 变成 DRAM bandwidth-bound？
为什么 LLM decode 不适合 MTIA 2i？
```

---

# 15. 从你的架构探索 DSL/IR 角度怎么抽象 MTIA？

## Architecture DSL

```text
MTIADevice:
  PEGrid:
    shape: 8x8
    PE:
      local_memory: 384KB
      risc_v_scalar_core
      risc_v_vector_core
      command_processor
      dpe
      re
      se
      mlu
      fabric_interface

  GlobalMemory:
    shared_sram:
      capacity: 256MB
      partition:
        LLS: software-managed scratch
        LLC: hardware-managed cache
    lpddr5:
      capacity: 64-128GB
      bandwidth: 204.8GB/s

  Interconnect:
    custom_noc
    edge_crossbars
    source_flow_control
    packet_fragmentation
    traffic_shaping
```

## Event / Schedule IR

```text
ComputeEvent:
  target: DPE / SE / RISC-V vector
  op: GEMM / activation / quant / nonlinear
  tile_shape
  datatype
  sparsity_mode

MemoryEvent:
  LPDDR -> LLC
  LLC/LLS -> PE Local Memory
  PE Local Memory -> DPE
  RE -> SE
  PE -> neighboring PE
  output -> shared SRAM / LPDDR

ControlEvent:
  RISC-V issue custom instruction
  CP dependency tracking
  CP circular buffer management
  eager-mode job launch
  work queue descriptor broadcast

NoCEvent:
  PE read shared SRAM
  broadcast weights across PE columns
  source throttling
  packet fragmentation
```

## Cost Model

```text
CostModel:
  DPE utilization
  SE utilization
  FI bandwidth
  PE local memory capacity pressure
  shared SRAM hit rate
  LLS/LLC partition choice
  LPDDR bandwidth pressure
  NoC contention
  job launch overhead
  graph fusion benefit
  batch/request coalescing effect
```

这比单纯看 TOPS 有用得多。

---

# 一句话总结

**MTIA v1/v2 的核心不是“Meta 也做了 AI 芯片”，而是一个非常典型的生产级 recommendation inference accelerator 案例：8×8 RISC-V-controlled PE grid、PE 内 Local Memory + DPE/RE/SE/MLU/FI 异步 dataflow、PE 外 shared SRAM + LPDDR + custom NoC，再通过 PyTorch eager、kernel autotuning、graph fusion、SRAM placement 和 request coalescing 把真实线上 R&R workload 跑出低 TCO。**

[1]: https://ai.meta.com/blog/meta-mtia-scale-ai-chips-for-billions/ "Four MTIA Chips in Two Years: Scaling AI Experiences for Billions"
[2]: https://aisystemcodesign.github.io/papers/MTIA-ISCA25.pdf "Meta's Second Generation AI Chip:  Model-Chip Co-Design and Productionization Experiences"
[3]: https://ai.meta.com/blog/meta-training-inference-accelerator-AI-MTIA/ "MTIA v1: Meta’s first-generation AI inference accelerator"
[4]: https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/ "Our next generation Meta Training and Inference Accelerator"
