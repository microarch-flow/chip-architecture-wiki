> 现在开始分析core

好，我们从 **Cerebras core** 开始拆。这个 core 是理解 Cerebras 的入口，因为它不是 GPU SM、不是 TPU systolic cell、也不是传统 RISC core，而是一个非常特殊的：

> **small programmable dataflow tensor core**
> 小型可编程数据流张量核心。

我先给结论：

> **Cerebras core = 小型通用处理器底座 + 本地 SRAM + tensor datapath + DSR 张量描述寄存器 + fabric-triggered dataflow scheduler + microthread 机制。**

它的设计目标不是单 core 很强，而是让 **几十万/九十万个 core** 能在 wafer 级规模上以低开销执行稀疏、低复用、流式张量计算。

---

# 1. Core 的基本形态：不是“大核”，而是“极小 PE”

以 WSE-2 为例，HC2022 给出的 core 物理参数是：

| 项                |                     数值 |
| ---------------- | ---------------------: |
| core 面积          |          228μm × 170μm |
| 工艺               |                TSMC N7 |
| logic : SRAM 面积比 |                50 : 50 |
| logic            | 110,000 standard cells |
| SRAM             |                   48KB |
| 频率               |                 1.1GHz |
| 峰值功耗             |                   30mW |



这个面积非常小。它的设计思路不是：

```text
一个复杂大 core + 很强乱序/缓存/预测/向量单元
```

而是：

```text
一个足够可编程的小 core
+ 非常近的 SRAM
+ 简单但高带宽的 tensor datapath
+ fabric 触发执行
+ 复制几十万/九十万份
```

所以它更像 **dataflow PE**，不是传统 CPU core。

---

# 2. Core 的高层结构

可以先画一个抽象图：

```text
                    ┌──────────────────────────┐
Fabric Input ──────►│ Dataflow Trigger / Ctrl   │
                    │                          │
                    │  ┌────────────────────┐  │
                    │  │ Instruction Lookup │  │
                    │  └────────────────────┘  │
                    │                          │
                    │  ┌──────────────┐        │
                    │  │ Tensor Ctrl  │◄── DSR │
                    │  └──────────────┘        │
                    │                          │
                    │  ┌──────────────┐        │
                    │  │ Tensor       │        │
                    │  │ Datapath     │        │
                    │  │ FMAC/SIMD    │        │
                    │  └──────────────┘        │
                    │                          │
                    │  ┌──────────────┐        │
                    │  │ 48KB SRAM    │        │
                    │  │ + SW Cache   │        │
                    │  └──────────────┘        │
                    │                          │
Fabric Output ◄─────│ Output / Reduction / Send │
                    └──────────────────────────┘
```

这个图里最重要的是：**fabric input 不是普通 IO，而是 execution trigger**。

也就是说，Cerebras core 的执行不是简单的：

```text
PC fetch → decode → execute → memory → writeback
```

而是更像：

```text
fabric packet 到达
  → 根据 color/control/index 查 handler
  → DSR 描述 tensor operand
  → tensor datapath 执行一段 tensor operation
  → 输出到本地 SRAM / accumulator / fabric
```

官网白皮书明确说，fabric 同时传输 data 和 associated control；core 收到数据后，硬件根据 fabric 收到的信息触发 instruction lookup，整个 compute fabric 因此成为 dataflow engine。

---

# 3. Core 里有“通用处理器”，但它不是重点

HC2022 说 Cerebras core 支持一般处理器指令，包括：

```text
arithmetic
logical
load/store
compare
branch
```

还有：

```text
16 general purpose registers
compact 6-stage pipeline
independent instructions for each core
48KB memory stores data and instructions
```



这说明它确实不是硬连线 MAC cell，而是有一定可编程性的。

但要注意：**通用处理器部分不是为了像 CPU 那样跑复杂控制流**。它的主要作用是：

```text
1. 支持不同 NN op 的控制逻辑；
2. 支持 handler / kernel 级别的本地程序；
3. 配合 tensor instruction 做张量计算；
4. 让每个 core 可以有独立执行行为；
5. 支持未来模型结构变化。
```

所以它是：

> **programmable control substrate**
> 可编程控制底座。

不是：

> **general-purpose CPU core**
> 通用 CPU 核。

---

# 4. Tensor datapath：WSE-2 是 4×FP16 FMAC，WSE-3 升级到 8-way FP16 SIMD

WSE-2 的 HC2022 材料中，core datapath 是：

```text
fine-grained 64b datapath
4 × FP16 FMAC
```

并且 tensor 是 instruction 的 first-class operand。例子是：

```text
fmach [fpsum] = [fpsum], [fwd_wgt], r_in
```

其中 `[fpsum]`、`[fwd_wgt]` 是 tensor operand，`r_in` 是 scalar/GPR operand。

到了 WSE-3，HC2024 明确给出 core 支持：

```text
8-way SIMD for 16b data, FP/BF16
16-way SIMD for 8b data, Fixed/INT8
fast non-linear function instructions
fine-grained dataflow scheduling
native unstructured sparsity acceleration
48KB SRAM per core
512B local cache per core
48 DSRs
```



所以代际变化大致是：

| 项               |                WSE-2 |                WSE-3 |
| --------------- | -------------------: | -------------------: |
| 16-bit datapath |     4×FP16/BF16 FMAC |       8-way 16b SIMD |
| 8-bit datapath  |             材料中未重点强调 |       16-way 8b SIMD |
| DSR             |                   44 |                   48 |
| local cache     |                 256B |                 512B |
| SRAM            |            48KB/core |            48KB/core |
| 架构范式            | dataflow tensor core | dataflow tensor core |

这里可以看到，WSE-3 是在同一个 core 范式上增强 tensor datapath 和 cache，而不是换成 GPU 式 tensor core。

---

# 5. DSR：这是 Cerebras core 最关键的抽象之一

DSR 是 **Data Structure Register**，我建议你把它理解成：

> **硬件可理解的 tensor descriptor register。**

它不是普通标量寄存器，也不是向量寄存器。它里面存的是 tensor 的描述信息，例如：

```text
pointer / base address
length
shape
size
dimension
stride
FIFO / circular buffer 描述
fabric-streaming tensor 描述
```

WSE-2 有 44 个 DSR，WSE-3 有 48 个 DSR。白皮书说 DSR 可以描述最多 4-D tensor、fabric-streaming tensor、FIFO、circular buffer；背后有硬件状态机使用 DSR，在 datapath 上 full performance 地遍历 tensor。

这个东西非常重要。因为它把一条 tensor instruction 从：

```text
for i in range(N):
    z[i] = z[i] + w[i] * a
```

变成类似：

```text
FMAC tensor described by DSR_z
     with tensor described by DSR_w
     and scalar/register a
```

也就是说：

```text
一条指令描述一个 tensor operation
DSR 描述 tensor 的遍历方式
硬件状态机负责展开
```

这和普通 SIMD/vector 的差异是：

| 机制                     | 描述方式                        | 执行方式                   |
| ---------------------- | --------------------------- | ---------------------- |
| 普通 SIMD                | vector register 固定宽度        | 一条指令处理固定 lanes         |
| Vector ISA             | VL + vector register        | 一条指令处理 vector length   |
| Cerebras DSR tensor op | DSR 描述 memory/fabric tensor | 硬件状态机遍历 tensor operand |

它更接近一种 **tensor streaming descriptor ISA**。

---

# 6. Local SRAM：不是 cache，是 core 私有 scratchpad

每个 core 有 48KB SRAM，WSE-2 的结构是：

```text
8 banks × 6KB
每 bank 32-bit wide
single port
支持 2×64-bit read + 1×64-bit write / cycle
另有 256B software-managed cache
```



注意这个细节：

> **48KB SRAM 同时放 data 和 instructions。**

这说明每个 core 的程序和数据都是编译期/配置期放进去的。它不像 GPU 有统一巨大 HBM + cache hierarchy，而是：

```text
每个 core 独立本地 SRAM
跨 core 数据共享必须显式走 fabric
```

这带来两个后果：

## 好处

```text
1. SRAM 到 datapath 距离极短，能量低；
2. aggregate SRAM bandwidth 极大；
3. 不需要 cache tag/coherence；
4. 可预测性强；
5. 适合 compiler-managed data layout。
```

## 代价

```text
1. 程序员/编译器必须管理数据分布；
2. 没有传统共享内存的便利；
3. tensor layout 与 physical topology 强绑定；
4. 如果 mapping 不好，远端通信/重排成本很高。
```

所以 Cerebras core 的 SRAM 本质上是：

> **compiler-managed local memory / scratchpad**
> 而不是自动 cache。

---

# 7. Dataflow trigger：它如何让“数据到达即计算”？

这是 Cerebras core 最有特色的地方。

材料里反复强调：

```text
Data and control transmitted on fabric
Triggers lookup and execution of handler instructions
Lookup based on input fabric color or control information
Native unstructured sparsity harvesting by filtering out zeros
```



可以抽象成：

```text
packet = {
    data: 16-bit weight / activation / value,
    control/index: 16-bit color/index/command
}

on_packet_arrival(packet):
    handler = lookup(packet.color/control/index)
    execute(handler, packet.data)
```

在 sparse GEMM 中：

```text
nonzero weight packet 到达
  → 触发 FMAC handler
  → 使用本地 activation tensor
  → fpsum += weight * activation
```

而 zero weight：

```text
sender 端直接过滤
不发送 packet
receiver 端什么都不知道
自然不计算
```

这就是它所谓的 **native unstructured sparsity acceleration**。

关键点是：
Cerebras 不是在 receiver 端判断 “if weight == 0 then skip”，而是在 sender / MemoryX / stream 端就不发 zero。这样它同时减少：

```text
1. compute
2. fabric traffic
3. receiver scheduling overhead
4. power
```

这是它和 GPU 稀疏计算很大的差异。

---

# 8. Microthread：不是 GPU warp，更像“tensor context interleaving”

Cerebras core 支持 8 个 simultaneous tensor operations，称为 **microthreads**。材料说这些是 independent tensor contexts，硬件可以 cycle-by-cycle 切换；scheduler 监控 input/output tensor availability 和 priority，选择 eligible microthread 执行。每个 microthread 可以访问 core 的全部 registers 和 memory，没有静态资源分区。

这和 GPU warp 很容易混淆，但本质不同。

| 项    | GPU warp                  | Cerebras microthread                        |
| ---- | ------------------------- | ------------------------------------------- |
| 粒度   | 多线程 SIMT execution group  | 单 core 内多个 tensor operation context         |
| 目的   | 隐藏延迟、提高 occupancy         | 在 dataflow 动态输入下填 pipeline bubble           |
| 调度依据 | warp ready / scoreboard   | tensor input/output availability + priority |
| 资源   | 每 warp/thread 分配寄存器等      | microthread 可访问 core 全资源，无静态分区              |
| 执行对象 | thread instruction stream | tensor operation / handler context          |

更准确地说，microthread 是：

> **在单个 core 内对多个 tensor 操作上下文做 cycle-level interleaving。**

它解决的问题是：dataflow/sparse execution 下，某个 tensor op 可能因为输入没到、输出 buffer 满、fabric backpressure、reduction 等原因暂时不能继续，这时 core 可以切换到另一个 ready 的 tensor context，避免 datapath 空转。

---

# 9. Core 执行一次 sparse weight FMAC，大概发生什么？

我们用 Cerebras 的 GEMM with sparse input 场景来串起来。

假设要做：

```text
C[j][i] += W[i][k] * A[j][k]
```

其中 W 是 sparse weight stream，A 是本地 activation chunk。

一次 nonzero weight 到达某个 core 后：

```text
Step 1: Fabric packet 到达 core
        packet 携带 weight value + index/control/color

Step 2: Dataflow trigger 根据 packet 信息查 handler
        例如：这是某个 sparse GEMM weight packet

Step 3: Handler 指向一条 tensor FMAC 指令
        fmac [fpsum] = [fpsum], [activation_tensor], weight_scalar

Step 4: DSR 给出 activation tensor 和 fpsum tensor 的地址/shape/stride

Step 5: tensor control 硬件状态机遍历 tensor
        从 48KB SRAM 读 activation
        从 SW cache/SRAM 读写 accumulator

Step 6: datapath 执行 SIMD/FMCA
        WSE-2: 4×FP16 FMAC
        WSE-3: 8-way 16b SIMD

Step 7: partial sum 累积在本地
        后续 PSUM/FSUM command 触发 reduction
```

这就是 Cerebras core 的完整“数据到达触发张量计算”路径。

---

# 10. 为什么这个 core 适合 sparse / AXPY / GEMV，而 GPU 不自然？

核心原因是 **memory bandwidth 与执行粒度**。

GPU tensor core 非常适合 dense GEMM，因为 dense GEMM 有高数据复用：

```text
一个 tile 的 A/B 被多次复用
arithmetic intensity 高
HBM 带宽压力被复用摊薄
```

但 sparse GEMM 拆成很多 AXPY 后，计算强度变低：

```text
每个 nonzero weight 触发一个 vector-scalar multiply-add
数据复用少
内存访问/调度/稀疏索引开销变重要
```

Cerebras 的 core 试图用三件事解决：

```text
1. activation resident in local SRAM
2. weight packet 只发送 nonzero
3. AXPY 在本地 SRAM 带宽下执行
```

材料也明确说 Sparse-GEMM 可以看成 one AXPY per nonzero weight，Cerebras 的高 memory bandwidth 让它能在所有 BLAS levels 上保持 full performance。

所以 core 不是为了 dense GEMM 峰值极致优化，而是为了：

> **在低复用、细粒度、稀疏、动态数据流场景下仍然保持高利用率。**

---

# 11. 和几类典型 core 对比

## 11.1 vs GPU SM

| 项    | GPU SM                     | Cerebras Core                |
| ---- | -------------------------- | ---------------------------- |
| 执行模型 | SIMT warp                  | dataflow-triggered tensor op |
| 存储   | register/shared/L1/L2/HBM  | 48KB local SRAM              |
| 通信   | shared memory/cache/NVLink | wafer fabric packet          |
| 稀疏   | 多依赖结构化稀疏/库                 | nonzero packet 触发            |
| 调度   | warp scheduler             | microthread scheduler        |
| 编程   | CUDA kernel                | graph/compiler mapping       |

GPU SM 强在通用性、dense tensor throughput、软件生态。
Cerebras core 强在 local SRAM bandwidth、packet-triggered sparse execution、超大规模 replication。

---

## 11.2 vs TPU systolic cell

| 项    | TPU systolic array     | Cerebras Core                     |
| ---- | ---------------------- | --------------------------------- |
| 基本单元 | MAC cell / systolic PE | programmable dataflow tensor core |
| 数据流  | 规则 systolic wavefront  | packet-triggered dataflow         |
| 稀疏   | 不天然适合非结构化稀疏            | 天然面向 nonzero packet               |
| 控制   | 阵列级控制                  | 每 core 独立 handler/microthread     |
| 存储   | unified buffer + array | per-core SRAM                     |

TPU 更适合规则 dense matrix multiplication。
Cerebras 更偏不规则/稀疏/大规模分布式数据流。

---

## 11.3 vs Graphcore IPU tile

Graphcore IPU tile 也是大量小核 + 本地 SRAM + 编译器管理，但差异是：

| 项    | Graphcore IPU tile | Cerebras Core                 |
| ---- | ------------------ | ----------------------------- |
| 规模   | 单芯片千级 tile         | wafer 上几十万/九十万 core           |
| 执行模型 | BSP / exchange 更明显 | packet-triggered dataflow 更明显 |
| 通信   | exchange fabric    | 2D mesh static-routed fabric  |
| 稀疏   | 有支持但不是核心叙事         | unstructured sparsity 是核心叙事   |
| 权重策略 | 更偏本地/分布式存储         | training 中 weight streaming   |

它们都反 GPU cache-HBM 路线，但 Cerebras 更极端。

---

# 12. Core 的真实限制

不能只看 Cerebras 的宣传。这个 core 也有明显 trade-off。

## 12.1 单 core 很弱

一个 core 只有很小 datapath。WSE-2 是 4×FP16 FMAC，WSE-3 是 8-way 16b SIMD。单 core 不强，靠数量堆。

这意味着：

```text
只要 mapping 不能充分铺满 wafer，
单 core 性能不能救场。
```

---

## 12.2 强依赖编译器 mapping

因为每个 core 只有本地 SRAM，没有共享内存幻觉，所以编译器必须决定：

```text
tensor chunk 放哪里
weight stream 发到哪里
partial sum 怎么 reduce
fabric color 怎么配置
哪个 packet 触发哪个 handler
microthread 如何组织
```

这比 GPU kernel 编程更静态、更全局。

---

## 12.3 对 workload shape 有偏好

它最适合：

```text
large tensor
规则全局 mapping
activation 可 resident
weight 可 stream/broadcast
reduction pattern 可静态规划
```

不适合：

```text
大量小 op
频繁动态 shape
复杂数据依赖
随机访存
不规则控制流
需要通用共享内存语义的 workload
```

---

## 12.4 稀疏收益依赖算法侧

Core 支持 fine-grained unstructured sparsity，但如果模型没有足够稀疏，或者稀疏导致精度下降，那么这个优势就打折。

所以硬件能力是：

```text
可以高效利用非结构化稀疏
```

不是：

```text
自动让所有模型稀疏 10× 且无精度损失
```

---

# 13. 用你的 DSL/IR 视角描述 Cerebras core

如果我们要在你的 architecture DSL 中建模 Cerebras core，我会这样拆：

```yaml
core:
  type: programmable_dataflow_tensor_core

  local_memory:
    kind: scratchpad_sram
    capacity: 48KB
    banks: 8
    bank_width: 32b
    read_ports_effective: 2x64b
    write_ports_effective: 1x64b
    cache:
      kind: software_managed
      capacity: 256B_or_512B

  scalar_control:
    gpr: 16
    pipeline: 6_stage
    ops:
      - arithmetic
      - logical
      - load_store
      - compare
      - branch

  tensor_datapath:
    wse2:
      fp16_fmac: 4
      datapath_width: 64b
    wse3:
      simd_16b: 8
      simd_8b: 16

  tensor_descriptor:
    name: DSR
    count:
      wse2: 44
      wse3: 48
    describes:
      - tensor_address
      - length
      - shape
      - stride
      - fifo
      - circular_buffer
      - fabric_streaming_tensor

  scheduling:
    mode: dataflow_triggered
    trigger:
      - fabric_data_packet
      - fabric_control_packet
      - fabric_color
    microthreads: 8
    policy:
      - input_available
      - output_available
      - priority

  fabric_interface:
    role:
      - receive_data
      - receive_control
      - emit_data
      - participate_reduction
```

对你的 Event IR 来说，core 事件不能只写 `compute(matmul)`，而要拆成：

```text
packet_arrival(color, payload)
handler_lookup(color/control)
tensor_op_issue(handler, DSRs)
local_sram_read
simd_fmac
accumulator_update
fabric_output_or_reduce
```

这非常适合你之前说的 **event 是编译调度产物，并且必须和架构 DSL 绑定**。

---

# 14. Core 的一句话心智模型

最后给你一个最短心智模型：

> **Cerebras core 是一个带 48KB 本地 SRAM 的小型可编程 tensor dataflow PE：平时预加载 handler 和 tensor descriptor，fabric 上的 data/control packet 到达后触发 tensor op，用本地 SRAM 中的 activation/accumulator 执行 FMAC，并通过 microthread 在动态稀疏流中隐藏 bubble。**

下一步建议继续分析：**DSR + tensor instruction 到底该怎么理解？它和普通 SIMD/vector ISA、GPU tensor core、TPU systolic array 的差异是什么。**
