# AWS Inferentia2 / NeuronCore-v2

一句话定位：**compiler-managed、software-managed SRAM 为中心的异构多引擎 AI tile**。和 Graphcore/Cerebras/SambaNova 那种"赌新颖架构范式"的初创不同，AWS 走的是另一条路——**架构相对保守（systolic + 向量/标量引擎 + scratchpad SRAM），真正的护城河是垂直整合（自研芯片 + Nitro + Graviton + 云）+ anchor 客户 + 成本**。NeuronCore-v2 同时用于推理芯片 Inferentia2 和训练芯片 Trainium1。

## 架构总览（自绘推导图）
下图为基于公开资料推导的整体架构图（非官方实现图）：左为 Inferentia2 芯片级视图，右为单个 NeuronCore-v2 内部视图，含 4 引擎/4 指令流、SBUF/PSUM、典型数据流与访问限制。

![AWS Inferentia2 / NeuronCore-v2 推导架构图（自绘）](assets/Architecture.png)

![Inferentia2 芯片（官方框图）：2×NeuronCore-v2 + 2×HBM + DMA + Collective Communication + Host PCIe + 2×NeuronLink-v2](assets/inferentia2_chip.png)

> **公开信息边界（重要）**：AWS 只公开到 **NKI（Neuron Kernel Interface）可见的片上层级**——engine、SBUF/PSUM、DMA、semaphore；**没有**公开到 NoC/router/packet/事务层。所以两个 NeuronCore-v2 之间的片上互连拓扑、core-to-core 是否绕 HBM、collective engine 微架构等，本文只能保守表述，不能脑补成 GPU NVLink 语义。

## 关键参数

**Inferentia2 芯片级**

| 项 | 值 |
|---|---|
| Compute | 2 × NeuronCore-v2 |
| HBM | 32 GiB，820 GiB/s（2 个 stack） |
| 算力 | 380 INT8 TOPS / 190 FP16·BF16·cFP8·TF32 TFLOPS / 47.5 FP32 TFLOPS |
| DMA | 32 个 DMA engine，约 1 TB/s，支持 inline 压缩/解压 |
| Collective | 6 个 CC-Core；2 条 NeuronLink-v2（chip-to-chip collective） |
| 实例 | 一个 Inf2 instance 最多 12 颗 Inferentia2 |

**NeuronCore-v2（单 core）**

| 引擎 / 存储 | 规格 |
|---|---|
| Tensor Engine | 128×128 systolic array，~92 TFLOPS（BF16/FP16/TF32/cFP8）/ ~23 TFLOPS（FP32） |
| Vector Engine | 128 lanes，reduce / 多输入 elementwise；每 32 partition 内可 32×32 transpose + shuffle |
| Scalar Engine | 128 lanes，activation/nonlinear（每个 output 只依赖一个 input element） |
| GPSIMD Engine | 8 个 512-bit 可编程处理器，跑 C code，custom op / 不规则算子 |
| SBUF | 24 MiB，128 partitions × 192 KiB，主数据存储 |
| PSUM | 2 MiB，128 partitions × 16 KiB，Tensor Engine 累加 buffer，FP32 |

## 执行模型：多 sequencer、多指令流、显式同步
NeuronCore-v2 不是"一个控制器驱动多个 FU"，也不是 GPU 的 SIMT warp scheduler。**四个 compute engine（Tensor/Vector/Scalar/GpSimd）各有自己的 sequencer，执行四条独立 instruction stream，异步并行**：

```text
Neuron Compiler / NKI kernel
  ↓ 为每个 engine 生成各自的 instruction stream
Tensor seq → Tensor datapath
Vector seq → Vector datapath
Scalar seq → Scalar datapath
GpSimd seq → 可编程 SIMD datapath
  ↓ engine 间数据依赖
hardware atomic semaphore（编译器自动插入同步）
```

每条 instruction stream 里有两类指令：**control 指令**（sequencer 用私有 32-bit scalar register 做 scalar ALU、动态条件、地址计算、branch、触发 DMA）和 **data-path 指令**（engine 专用 datapath 操作 SBUF/PSUM 中的 tensor）。这更像一个**异构多引擎 dataflow-ish tile**，但不是静态 VLIW（没有统一 bundle issue）。

> 注意：NeuronCore-v2 官方只有 **4 个引擎**（Tensor/Vector/Scalar/GpSimd），**没有独立的 "Sync Engine"**；触发 DMA、同步是每个引擎指令流里的 control 指令能力 + 硬件 atomic semaphore，真正的搬运由设备级 32 个 DMA engine 执行。

![NeuronCore-v2：Tensor / Vector / Scalar / GPSIMD 四类引擎 + On-chip SRAM](assets/neuroncore_v2_block.png)

## 片上 memory：SBUF / PSUM 是 2D，软件管理
两块 software-managed SRAM（**不是 cache**），都组织成 2D、各 128 partitions：
- **SBUF**（24 MiB）：主数据存储，input/output/中间结果
- **PSUM**（2 MiB）：Tensor Engine 的 landing/accumulation buffer，每 partition 8 banks，FP32 read-accumulate-write，**累加只能由 TensorE 控制**（VectorE/ScalarE 可读写 PSUM，TensorE 对 PSUM 只写）

关键心智模型——**P 维并行、F 维时间展开**：
```text
SBUF/PSUM tensor shape: [P, F]
  P = partition 维（≤128）= 并行维（每周期跨 128 partition 读/算/写 128 元素）
  F = free 维 = 时间/streaming 维
```
所以 NeuronCore 的 tile 不是普通 `[M,N]`，而是 `{partition_axis P≤128, free_axis F, buffer: SBUF|PSUM}`。

![SBUF/PSUM 数据沿 P 维并行进入 compute engine、沿 F 维随时间流过 engine pipeline](assets/sbuf_streaming.png)

**端口冲突约束**（编译器负责串行化，不是任意 crossbar）：VectorE 与 GPSIMD 不能并行访问 SBUF；VectorE 与 ScalarE 不能并行访问 PSUM。

## Tensor Engine 数据流：LoadStationary + MultiplyMoving
NKI 的 `nc_matmul(stationary, moving)` 在硬件上不是简单 `x@y`，而是两类 ISA 指令：
- **LoadStationary**：从 SBUF 读 `stationary[K,M]`，缓存进 Tensor Engine 内部
- **MultiplyMoving**：从 SBUF 读 `moving[K,N]`，与已加载的 stationary 相乘，结果写 `output[M,N]` 到 PSUM

![matmul：数学视图 x[M,K]@y[K,N]=out[M,N] ↔ Tensor Engine 视图 stationary/moving/PSUM/copy-to-SBUF](assets/matmul_mxkxn.png)

K > 128 时用 **PSUM 近存累加**：多个 K-tile 连续累加进同一 PSUM bank，最后经 Vector/Scalar 做 post-op（bias/activation/cast）再写回 SBUF。这与 GPU tensor core 的 accumulator register file 不同——是**显式 PSUM SRAM accumulation**。

![SBUF → Tensor Engine（128 行 × 128 列）→ PSUM（128 partitions）](assets/tensor_engine_sram.png)

典型单 core 数据流总览：
```text
HBM --DMA--> SBUF --> Tensor/Vector/Scalar/GpSimd --> PSUM 或 SBUF --DMA--> HBM
其中 HBM↔SBUF 走 DMA；PSUM↔SBUF 走 engine 指令
```

## 多 core / 多 chip 通信
两个 NeuronCore-v2 **计算上基本独立、共享 device 级 HBM**，但**没有公开的 SBUF-to-SBUF 直连 datapath**。跨 core/chip 数据流应保守建模为经 **DMA / HBM / collective**：
```text
张量并行：Core 局部 partial → CC-Core / Collective Communication → reduce/gather → 写回 HBM/SBUF
流水并行：Core0 输出 --DMA--> HBM --DMA--> Core1 SBUF（保守路径，不假设 SBUF 直达）
跨 chip ：NeuronLink-v2 做 chip-to-chip collective
```
即"分布式 scratchpad + 共享 off-chip memory + 专用 collective engine"，**不是统一 cache hierarchy，也不要建模成普通 NoC router**。

## 软件栈
- **Neuron SDK**：编译器 + runtime + 框架集成（PyTorch/TensorFlow/JAX）
- **Neuron Compiler**：把 graph 拆成 Tensor/Vector/Scalar/GpSimd/DMA 的多 engine 并行序列，自动插入 semaphore 同步、管理 SBUF/PSUM layout
- **NKI（Neuron Kernel Interface）**：暴露"编译器可见的硬件约束"（tile shape、partition 维、SBUF/PSUM layout、PSUM 累加、DMA pattern、engine overlap），是反推 NeuronCore 微架构最有价值的入口——普通产品页只够建名词表

## 芯片家族与迭代
NeuronCore-v2 是 Inferentia2（推理）与 Trainium1（训练）共用的核心。AWS 自研芯片矩阵：

| 线 | 用途 | 代表 |
|---|---|---|
| Inferentia | 推理 | Inferentia1（2019）→ **Inferentia2（2023，NeuronCore-v2）** |
| Trainium | 训练 | Trainium1（2022，NeuronCore-v2）→ Trainium2（2024，约 4× Trn1）→ Trainium3（re:Invent 2025，3nm） |
| Graviton | 通用 CPU（Arm） | Graviton1–4 |
| Nitro | 虚拟化/IO offload + 安全 | 贯穿 EC2 |

迭代逻辑：**沿同一 software-managed SRAM + 多引擎范式演进**，配合 NeuronLink 把多芯片组成大集群；不追求架构范式革命，而追求与云、与 anchor 客户深度协同优化。

## 战略定位与商业背景
- **2015 年收购 Annapurna Labs**（以色列芯片设计公司，约 $3.5 亿）——AWS 自研硅的起点，如今 Graviton/Nitro/Inferentia/Trainium 都出自这里
- **垂直整合**：芯片 + Nitro（offload/安全）+ Graviton（CPU）+ EC2/云服务一体设计，目标是降本、降低对 Nvidia 的依赖、把性价比握在自己手里
- **anchor 客户 Anthropic**：与 Annapurna **共同设计 Trainium**（几乎每日协作，用 Claude 训练负载反馈塑造下一代芯片）；**Project Rainier**（2025 年 10 月启用，印第安纳，近 50 万颗 Trainium2 专训 Claude）；Anthropic 承诺十年 **$1000 亿+**、最高 5GW，覆盖 Trainium2→Trainium4
- 这与三家初创的处境形成鲜明对照：**AWS 不需要靠单颗芯片的架构惊艳去赢，它有现成的云需求、内部 workload 和巨型 anchor 客户来吸收产能并迭代**

## 从中该获取的经验教训

### AWS 的下注（和初创相反的一条路）
- **架构保守，整合激进**：NeuronCore-v2 没有 wafer-scale、没有 CGRA reconfigurable fabric、没有 weight-streaming 之类的激进范式——就是 systolic + 向量/标量 + 可编程 SIMD fallback + 软件管理 SRAM。赌注不在硅片新颖度，而在**垂直整合 + 规模 + 客户绑定**
- **GPSIMD 是务实的逃生舱**：8 个可编程 512-bit 处理器跑 C code，承接 mask/sampling/特殊 reshape/未来新算子——承认"硬连线覆盖不了所有 op"，保留通用兜底
- **complexity 给编译器**：和三家初创一样，把多引擎调度/同步/SBUF-PSUM layout 全压给 Neuron Compiler；但 AWS 有持续的内部/客户 workload 来打磨这条 toolchain，迭代闭环比初创健康

### 与 Graphcore / Cerebras / SambaNova 的对照
- 四者都做"片上 SRAM + 编译器管理数据搬运 + 异构引擎"，但**护城河来源完全不同**：三家初创赌"架构领先 → 性能领先 → 赢客户"，结果普遍撞上软件生态 + workload 演化 + 商业兑现三重墙；AWS 反过来——**先有云 + 内部需求 + anchor 客户（Anthropic），再用够用且持续优化的芯片去填**
- 启示：在 AI 加速器赛道，**"谁能稳定吸收产能并形成迭代闭环"可能比"谁的架构最优雅"更决定生死**。AWS 用 Annapurna + 云 + Anthropic 解决了初创最缺的两样东西——出货量和软件打磨的真实负载

---
*本文基于 raw/ 下官方资料（Trainium/Inferentia2 NKI 架构指南、Inferentia2/NeuronCore-v2 架构页）与对应 chat 梳理；战略/商业背景来自公开报道（截至 2026-06）。架构图取自官方文档：`inferentia2_chip`/`neuroncore_v2_block` 来自架构页，`tensor_engine_sram`/`matmul_mxkxn`/`sbuf_streaming`（Fig 46/47/58）来自 NKI 架构指南。AWS 未公开 device 内部 NoC/fabric 层级，跨 core 数据路径为保守推断。*
