> HC2021_SN10.pdf、Reconfigurable_Dataflow_Architecture.pdf、HC2024_SN40L.pdf、SambaNova_SN40L.pdf、RDARuntime.pdf 我搜集到的一些SambaNova的资料，你用同样的方式帮我分析一下

可以，我们按之前分析 Cerebras / Graphcore / M100 那种方式来做：**先建立心智模型 → 再拆 core / memory / fabric / runtime / compiler → 最后做架构定位和批判性分析**。这次资料非常好，覆盖了 SambaNova 从 **RDA 理念、SN10、SN40L 到 RDARuntime** 的完整栈。🚀

---

# 0. 一句话先定性 SambaNova

**SambaNova RDA 本质上是一个 CGRA 风格的空间数据流 AI 加速器：它不是像 GPU 那样把算子拆成 kernel 依次执行，也不是像 TPU 那样主要围绕大型 systolic array，而是把模型图中的一段子图空间映射成“计算单元 + 存储单元 + 片上通信路径”的可重构流水线。**

它的核心卖点不是单个 MAC 阵列多强，而是：

> **把 compute、memory、interconnect 都变成 compiler 可以显式编排的资源，从而做 aggressive fusion、pipeline parallelism 和大模型/多模型 memory hierarchy 管理。**

---

# 1. 资料地图：这几份文档各自负责什么

| 资料                                           | 主要价值                                               | 我建议的阅读定位                       |
| -------------------------------------------- | -------------------------------------------------- | ------------------------------ |
| **Reconfigurable Dataflow Architecture 白皮书** | 讲 RDA 的设计动机、dataflow 和 spatial programming 哲学      | 先看，用来建立总心智模型                   |
| **HC2021 SN10**                              | 第一代公开架构资料，讲 SN10 RDU、PCU/PMU/AG、switch、DataScale   | 用来理解 RDA 原始微架构                 |
| **HC2024 SN40L**                             | 新一代语言模型优化 RDU，讲三层 memory、PCU、PMU、interconnect、AGCU | 用来理解 SN40L 相比 SN10 怎么面向 LLM 演进 |
| **SambaNova SN40L paper**                    | 更完整论文版，重点是 CoE、streaming dataflow、三层 memory、性能数据   | 用来分析真实性能路径和设计取舍                |
| **RDARuntime paper**                         | 讲 RDA 的 runtime/OS：资源管理、virtual RDU、OS-bypass、调度   | 用来理解这种空间数据流架构为什么需要专门 runtime   |

RDA 白皮书明确说 SambaNova 的核心思想是：RDU 没有传统意义上的固定 ISA，而是针对每个模型进行专门编程，生成类似 application-specific accelerator 的映射；SambaFlow 负责把 PyTorch/TensorFlow 等模型图抽取、优化、映射到 RDU 上。

---

# 2. 第一层心智模型：SambaNova RDA 到底是什么？

我建议先这样理解：

```text
传统 GPU:
  model graph -> many kernels -> each kernel runs on SMs -> intermediate tensors often materialized

TPU:
  model graph -> compiler schedule -> systolic array + HBM/UB transfers

SambaNova RDA:
  model graph segment -> spatial dataflow pipeline
                       -> PCU computes
                       -> PMU buffers/transforms
                       -> RDN routes vector/scalar/control streams
                       -> AGCU talks to HBM/DDR/host/remote RDU
```

RDA 白皮书反复强调的问题是：传统 core-based architecture 的 compute 可以被编程，但**通信路径主要由 cache / memory hierarchy 隐式管理**；而 RDA 希望把“数据如何从一个中间计算流到下一个中间计算”也显式编程和优化。

所以它不是单纯的 “another NPU”。

更准确地说，它是：

> **可重构空间流水线机器。**

其中：

* **PCU = Pattern Compute Unit**：负责 arithmetic / SIMD / systolic compute。
* **PMU = Pattern Memory Unit**：负责片上 scratchpad、address generation、tensor transformation。
* **Switch / RDN = Reconfigurable Dataflow Network**：负责把 vector/scalar/control stream 在片上路由。
* **AGCU = Address Generation and Coalescing Unit**：负责连接片上 dataflow 世界和 HBM/DDR/host/remote RDU。
* **SambaFlow compiler**：负责把 graph 映射成空间布局、通信 pattern、pipeline schedule。
* **RDARuntime**：负责资源管理、虚拟化、调度、multi-tenant、低延迟配置。

---

# 3. SN10：第一代公开 RDU 的架构骨架

SN10 是理解 SambaNova 的起点。HotChips 2021 资料给出的 SN10 参数很关键：

* TSMC 7nm
* 40B transistors
* 640 PCU
* 640 PMU
* > 300 BF16 TFLOPS
* > 300 MB on-chip memory
* 150 TB/s on-chip memory bandwidth
* BF16 with FP32 accumulation，同时支持 FP32、Int32、Int16、Int8 

SN10-8R 系统层面是：

* 8 个 RDU
* 12 TB DDR4-2667 memory
* 48 DDR4 channels
* 32 PCIe Gen4 x16 links 

这说明 SambaNova 从一开始就不是只做 chip，而是 **chip + node + rack + compiler/runtime** 的 vertically integrated design。

---

# 4. SN40L：从“通用 RDA”转向“LLM / GenAI optimized RDA”

SN40L 是这批资料中最值得深挖的部分。HotChips 2024 资料给出的单 RDU 关键参数是：

* TSMC 5nm
* 102B transistors
* 1,040 RDU cores
* 638 BF16 TFLOPS
* 520 MB on-chip memory
* 64 GB HBM
* 1.5 TB high-capacity memory
* 三层 dataflow memory hierarchy 

这里要注意一个口径问题：HC2024 的另一页以系统/节点级方式描述三层 memory：on-chip SRAM、RDU high-bandwidth memory、RDU high-capacity DDR memory，并给出 **8GB SRAM、1TB HBM、24TB DDR、25.6TB/s、1600GB/s** 等数字。这个看起来不是单颗 RDU 口径，而是更大系统聚合口径；单 RDU 口径更接近 520MB SRAM、64GB HBM、1.5TB DDR。

这点很重要，因为 SambaNova 很喜欢在 **socket / node / rack** 不同层级上讲 memory capacity，分析时必须先确认口径。

---

# 5. SN40L 的核心结构：PCU + PMU + RDN + AGCU

## 5.1 PCU：既像 vector，又像 systolic

SN40L 的 PCU 可以配置成：

* systolic array
* SIMD vector unit with M lanes
* 支持 BF16、FP32、INT32、INT8
* 有 cross-lane reduction tree
* tail stage 支持 transcendental functions、casting、stochastic rounding 

这很关键。

它不是 GPU SM，也不是 TPU 那种固定矩阵阵列。它更像：

```text
configurable compute pipeline =
  header + SIMD/systolic body + reduction + tail special function
```

所以 PCU 的定位是：**可重构的数据流计算 stage**。

这解释了为什么 SambaNova 可以把 softmax、layernorm、transpose、GEMM、elementwise 等混合图段 fuse 成较大的 dataflow pipeline。它不只是在做 GEMM，而是在做 **graph segment execution**。

---

## 5.2 PMU：SambaNova 的片上 memory 不是普通 SRAM bank

PMU 是我认为 SambaNova 最值得深入研究的地方之一。SN40L PMU 的公开描述包括：

* programmer-managed scratchpad
* 支持 concurrent reads and writes
* fragmentable address-generation pipeline
* 每周期可产生 4 个地址
* data alignment crossbar 支持 transpose、dilation、downcast 等 tensor transformations
* address predication 支持多个 PMU 组合存储大 tensor 

这说明 PMU 不是“被动 SRAM”。

它更像：

```text
PMU = SRAM banks
    + address generator
    + alignment / permutation crossbar
    + predicate / fragmentation logic
    + vector/scalar/control stream ports
```

也就是说，PMU 是 **memory + data layout transform + stream buffer + address engine** 的组合。

这和普通 GPU shared memory 的差异很大：

| 对比项              | GPU shared memory          | SambaNova PMU                              |
| ---------------- | -------------------------- | ------------------------------------------ |
| 管理方式             | kernel/thread block 管理     | compiler/runtime 空间映射管理                    |
| 作用               | block-local scratchpad     | graph pipeline stage buffer                |
| 地址模式             | 程序线程生成 load/store          | PMU 内部 address generation pipeline         |
| tensor transform | 通常靠指令/线程搬运实现               | PMU data alignment crossbar 直接支持           |
| 跨算子融合            | 受 kernel/thread/block 边界限制 | 作为 stage buffer 支撑 coarse-grained pipeline |

所以 PMU 是 SambaNova dataflow 的关键。

---

## 5.3 RDN / Interconnect：不是普通 NoC，而是 dataflow fabric

SN40L 的 unit-level interconnect 由 3 个物理网络组成：

* **Vector network**：承载数据，packet switched
* **Scalar network**：承载地址和 metadata，packet switched
* **Control network**：承载 flow control、synchronization、graph orchestration token

同时支持：

* point-to-point
* one-to-many multicast
* many-to-one transmission
* 2D dimension-order routing
* software override
* packet-level end-to-end credits
* tensor-level software control tokens 

这个结构非常有意思。

它不是简单的 mesh NoC，而是 **面向 dataflow stream 的多平面互联**：

```text
Vector plane  : tensor data stream
Scalar plane  : address / metadata stream
Control plane : synchronization / orchestration token
```

这和我们之前讨论的 NoC 很贴合：SambaNova 明确把 traffic class 分开了，而不是所有东西都挤在一个 packet network 里。

SN10 时代已经有 programmable packet-switched interconnect、独立 data/control bus、programmable routing 和 programmable counters，用于灵活分配 on-chip bandwidth 给 concurrent streams。

SN40L 则进一步把 vector/scalar/control 三类网络显式化。

---

## 5.4 AGCU：片上 dataflow 世界和外部 memory / IO 的 portal

AGCU 是理解 SN40L memory hierarchy 的关键组件。

SN40L paper 描述 AGCU 是 RDU tile 访问本地 HBM/DDR、host memory、remote RDU device memory、remote RDU tile 的 reconfigurable dataflow bridge；tile 侧暴露 RDN vector/scalar/control ports，TLN 侧生成 read/write request 并做 coalescing。它还支持 P2P，把数据直接 stream 到其他 socket 的 RDU tile，不经过 DDR/HBM，可用于 AllReduce 等 collective。

HotChips 2024 还说 AGCU 负责：

* DDR/HBM/Host memory address generation
* RDU 间 P2P communication
* graph control interface
* 支持无 host involvement 的 graph schedule orchestration
* Segment Lookaside Buffer 做虚实地址转换和 memory access management 

所以 AGCU 可以理解为：

```text
AGCU = DMA/LDST engine
     + address generator
     + coalescer
     + memory translation
     + graph launch/orchestration engine
     + P2P endpoint
```

它不是简单 DMA。

在 SambaNova 这种架构里，AGCU 是 **dataflow graph 和外部 memory hierarchy 的边界控制器**。

---

# 6. SambaNova 真正想解决的问题：operator fusion + memory wall

SN40L paper 里最核心的论证是：

小模型 / expert 模型经常 operational intensity 较低，而且图中有复杂 access pattern，比如 transpose、shuffle、many-to-one、one-to-many。GPU 的 operator fusion 受 kernel model、SM 间通信、shared cache/HBM materialization 限制，很难把复杂图段完全 fuse。论文用一个 Monarch FFT 风格图说明：不 fusion 时 operational intensity 只有 39.5 ops/byte，部分 fusion 到 102.6，完全 spatial fusion 到 410.4。

SambaNova 的解决方式是：

```text
把多个 operator 做成 coarse-grained spatial pipeline
tensor tile 在 PCU/PMU/RDN 中流动
中间结果不必频繁 materialize 到 HBM
PMU 作为 stage buffer
RDN 处理复杂 many-to-one / one-to-many / reorder
```

这和 GPU 的差异不是“谁的算力更大”，而是：

> **GPU 擅长把单个 kernel 在大量 SM 上 temporal multiplex；RDA 擅长把一段 graph 映射成空间流水线。**

---

# 7. SN40L 为什么加三层 memory：SRAM + HBM + DDR

SN40L 的三层 memory 是为 GenAI / CoE 做的：

```text
On-chip SRAM:
  支撑 dataflow pipeline，保存 intermediate / stage buffer

HBM:
  放当前活跃模型 / router / 当前 expert，支撑高带宽 decode

DDR:
  放大量 expert / 大模型参数，支撑低成本大容量和快速 model switching
```

SN40L paper 讲 CoE 时很清楚：Samba-CoE 有 150 个 Llama2-7B expert，总参数超过 1T，部署在一个 8-socket SN40L node 上。所有 expert 权重放在高容量 DDR 中，router 权重放在 HBM 中；一次推理分为三步：运行 router、从 DDR copy expert 到 HBM、运行 expert。

这个设计的目标不是传统意义上的“所有权重永远在 HBM”。

它的策略是：

```text
DDR = expert warehouse
HBM = active expert cache
SRAM = dataflow execution buffer
```

这非常适合 Composition of Experts，但不一定等价适合单个超大 dense model 的低延迟 serving。

---

# 8. CoE：SambaNova 的系统级叙事

SN40L paper 的中心不是普通 LLM，而是 **Composition of Experts, CoE**。

CoE 和 MoE 的区别：

| 项             | MoE                            | CoE                     |
| ------------- | ------------------------------ | ----------------------- |
| expert 形态     | 一个模型内部的专家层                     | 多个独立模型组成系统              |
| 训练方式          | 通常整体训练/微调                      | expert 可独立训练/微调         |
| router        | 模型内部 routing                   | 系统级 router model        |
| 硬件挑战          | sparse activation + all-to-all | model switching + 多模型驻留 |
| SambaNova 适配点 | 也可以支持                          | SN40L 重点宣传对象            |

论文说 Samba-CoE 由多个 expert 和一个 router 组成，router 根据输入 prompt 分配到最相关 expert，比如 math、code、law、Chinese 等；论文中的简化实验用 Llama2-7B 派生 expert，但概念不限于 Llama2。

这背后的系统观点是：

> 未来企业 AI 不一定是一个巨大的 monolithic LLM，而可能是大量 specialized smaller models 的组合。

而 SN40L 的三层 memory 正是为这个假设优化。

---

# 9. RDARuntime：为什么 RDA 需要一个“AI accelerator OS”

RDARuntime paper 很有价值，因为它解释了这种架构不只是 compiler 问题，还需要 runtime/OS。

论文明确说 RDARuntime 不是传统 OS，但它在 RDU 上解决 OS 应该解决的问题：硬件抽象、交互接口、硬件管理、稳定性、多租户、低 overhead。RDU 被描述为一种 CGRA，介于 GPU 固定 core 和 FPGA gate-level reconfigurable 之间，核心组件是 PCU、PMU、AGCU，并由 on-chip network 连接。

RDARuntime 和 CUDA runtime 的关键差异也很重要：

* GPU 通常每张卡暴露为一个 device file，进程显式选择 GPU。
* RDARuntime 向 OS 暴露单个 virtual device。
* 进程请求抽象 compute resources。
* Runtime 负责把 virtual RDU 映射到 physical RDU。
* 仍然允许 tile affinity、RDU IO channel hint 这类类似 CPU affinity 的控制。

这很像我们之前讨论的 architecture DSL / event IR / resource scheduler 的现实工业版本：

```text
Application / ML framework
        ↓
SambaFlow compiler -> PEF
        ↓
RDARuntime
        ↓
virtual RDU resource allocation
        ↓
physical RDU tiles / memory / network / AGCU
```

RDARuntime 还借鉴了 DPDK / OS-bypass：初始化、设备管理、部分 interrupt handling 用 kernel driver；运行时把请求的硬件区域动态映射到应用虚拟内存，从而兼顾 kernel resource management 和 low-latency userspace configuration。

这说明 SambaNova 的软件栈复杂度非常高。

---

# 10. 和 GPU / TPU / Cerebras / Tenstorrent 的定位对比

| 架构          | 核心范式                                               | 强项                                                                    | 弱点 / 代价                                      |
| ----------- | -------------------------------------------------- | --------------------------------------------------------------------- | -------------------------------------------- |
| GPU         | SIMT + kernel execution                            | 生态、通用性、峰值算力、HBM 带宽                                                    | kernel 边界、复杂 fusion、SM 间通信依赖 cache/HBM       |
| TPU         | systolic array + compiler-managed memory           | 大 GEMM / dense tensor                                                 | 非 GEMM / 动态图 / 复杂 access pattern 不一定优雅       |
| Cerebras    | wafer-scale spatial fabric + massive SRAM          | 超大片上 SRAM、低跨芯片通信、编译静态化                                                | 系统形态特殊，yield/封装/部署成本高                        |
| Tenstorrent | many tiles + local SRAM + NoC + programmable cores | tile-level dataflow、灵活、多核扩展                                           | 编译/调度复杂，NoC/内存规划困难                           |
| SambaNova   | CGRA-style reconfigurable dataflow                 | graph-level fusion、PMU tensor transform、三层 memory、CoE model switching | compiler/runtime 复杂，映射质量决定性能，动态 workload 挑战大 |

SambaNova 和 Tenstorrent 有相似点：都是 tile + local memory + NoC + compiler mapping。

但关键差异是：

* Tenstorrent 更像 **manycore + dataflow runtime + programmable Tensix cores**。
* SambaNova 更像 **CGRA + spatially configured graph pipeline**。

SambaNova 和 Cerebras 也有相似点：都强调 spatial dataflow、compiler-managed communication、大 on-chip memory。

但关键差异是：

* Cerebras 是 wafer-scale，把规模做到极致。
* SambaNova 是 socket/node/rack system，用 PMU/RDN/AGCU + DDR/HBM/SRAM 组合解决 memory wall。

---

# 11. 批判性看法：SambaNova 的优势和风险

## 优势

**第一，RDA 对复杂 graph fusion 的理论优势是真实的。**
如果一个 workload 有大量 elementwise、reduce、transpose、layout transform、small GEMM、pipeline parallelism，那么 RDA 的空间流水线确实可能比 GPU kernel-by-kernel 更高效。

**第二，PMU 是很强的架构点。**
PMU 不是 SRAM，而是 programmable memory transformation unit。这使 SambaNova 可以把 memory layout transformation 纳入 dataflow pipeline。

**第三，三层 memory 对 CoE 很合理。**
DDR 存大量 expert，HBM 缓存当前 expert，SRAM 做 pipeline buffer，这比“所有东西必须塞进 HBM”更适合多模型系统。

**第四，RDARuntime 的 virtual RDU 抽象很工业化。**
它不是论文玩具，而是在解决真实 multi-tenant、resource scheduling、low-latency configuration 问题。

---

## 风险 / 代价

**第一，compiler 是生死线。**
RDA 的硬件灵活性越高，映射空间越大。PCU/PMU/RDN/AGCU 的 allocation、routing、bandwidth model、pipeline balance 都要靠 compiler 做好。SN40L paper 自己也提到 software 必须管理 tile-level communication、HBM、DDR、die-to-die、P2P、host 等多种 bandwidth，static bandwidth model 很关键。

**第二，动态 shape / 动态 control flow 可能不友好。**
RDA 适合静态图段空间映射。越动态，越需要重新编排或 fallback，runtime overhead 和 mapping overhead 就会冒出来。

**第三，CoE 叙事依赖 workload 假设。**
如果未来主流企业 AI 真的是大量 small expert + router，SN40L 的 DDR/HBM/SRAM 设计很有价值。
但如果主流仍然是少量 dense frontier model + long-context serving，那么 CoE model switching 的优势就没那么直接。

**第四，生态壁垒很高。**
GPU 的最大优势不是硬件，而是 CUDA / PyTorch / Triton / TensorRT / NCCL / profiling / deployment 生态。SambaNova 要用 full-stack 垂直集成来换性能，但这也意味着用户迁移成本高。

---

# 12. 我建议后续按这个顺序深入

我们可以像之前分析 Cerebras 一样分阶段推进：

1. **先深挖 core：PCU + PMU**

   * PCU 的 systolic / SIMD dual mode 怎么理解？
   * PMU 为什么是 RDA 的关键？
   * PMU 和 GPU shared memory、TPU unified buffer、TT local SRAM 有什么差异？

2. **再深挖 fabric：RDN / TLN / AGCU**

   * vector/scalar/control 三网络怎么分工？
   * programmable routing 和 DOR 怎么共存？
   * many-to-one / multicast / reorder 怎么实现？
   * AGCU 和 DMA/NIC/NoC endpoint 的关系是什么？

3. **再深挖 compiler/runtime**

   * SambaFlow 到底在编译什么？
   * spatial mapping 和传统 kernel scheduling 的差异
   * RDARuntime virtual RDU、resource allocation、OS-bypass、PEF

4. **最后做架构定位**

   * RDA vs GPU tensor core
   * RDA vs TPU systolic array
   * RDA vs Cerebras fabric
   * RDA vs Tenstorrent Tensix
   * 对你自己的架构探索 DSL / event IR 有什么启发

---

## 一句话总结

**SambaNova RDA 的核心不是“更大的矩阵乘法阵列”，而是把模型图的一段计算、存储、通信都空间化、流水线化、可编译化；SN40L 则是在这个基础上用 SRAM + HBM + DDR 三层 memory 专门服务 GenAI/CoE 的 model switching 和 streaming dataflow。**

下一步我建议直接进入：**“深挖 PCU + PMU：SambaNova 的 core 到底怎么工作？”**
