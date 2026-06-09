> 这些pdf资料是我收集的关于AWS NeuronCore的信息，我需要你帮我分析AWS自研芯片的架构信息，比如Inferentia2中的两个NeuronCore-v2 间是否会有数据流动？单个NeuronCore-v2 中的控制流和数据流是怎样的？

可以，我们先把 **AWS Inferentia2 / NeuronCore-v2** 抽象成“架构师视角”的结构，而不是只复述官方文档。我的结论先放前面：

> **Inferentia2 的两个 NeuronCore-v2 是“相互独立的异构 compute unit”，但它们不是完全孤岛。它们之间可以发生数据流动，不过更像是通过 DMA / HBM / collective communication 这样的系统级数据搬运路径发生，而不是两个 NeuronCore 的 Tensor/Vector datapath 之间有直接的流水线互联。**

---

# 1. Inferentia2 顶层架构：两个独立 NeuronCore + 共享设备级搬运/通信资源

官方资料给出的 Inferentia2 芯片级结构是：

| 层级              | 组件                                                         |
| --------------- | ---------------------------------------------------------- |
| Compute         | 2 个 NeuronCore-v2                                          |
| Device Memory   | 32 GiB HBM，总带宽 820 GiB/s                                   |
| Data Movement   | 约 1 TB/s DMA 带宽，支持 inline memory compression/decompression |
| Communication   | NeuronLink-v2，用于 chip-to-chip collective                   |
| Programmability | 动态 shape、control flow、GPSIMD custom operators              |

Inferentia2 文档明确说，一个 Inf2 instance 最多有 12 颗 Inferentia2 芯片，每颗芯片有 2 个 NeuronCore-v2；单芯片给出 380 INT8 TOPS、190 FP16/BF16/cFP8/TF32 TFLOPS、47.5 FP32 TFLOPS。

NKI 架构指南还补充了更接近硬件拓扑的信息：Trainium/Inferentia2 device 有 2 个 NeuronCore-v2、2 个 HBM stack、32 个 DMA engine、6 个 CC-Core 用于 collective communication；Inferentia2 有 2 条 NeuronLink-v2。

所以从 DSL/架构描述角度，Inferentia2 可以先抽象成：

```text
Inferentia2 Chip
├── HBM stack x2
├── NeuronCore-v2 x2
│   ├── SBUF 24 MiB
│   ├── PSUM 2 MiB
│   ├── Tensor Engine
│   ├── Vector Engine
│   ├── Scalar Engine
│   ├── GPSIMD Engine
│   └── Sync Engine
├── DMA engines x32
├── Collective Communication Cores x6
├── Host PCIe
└── NeuronLink-v2 x2
```

---

# 2. 两个 NeuronCore-v2 之间是否会有数据流动？

## 2.1 会，但不是“Tensor Engine 直接连 Tensor Engine”

我倾向于这样判断：

| 问题                                                  | 判断                                                      |
| --------------------------------------------------- | ------------------------------------------------------- |
| 两个 NeuronCore-v2 是否完全独立？                            | **计算上基本独立**                                             |
| 两个 NeuronCore-v2 是否共享同一颗芯片的 HBM 地址空间/设备内存系统？        | **大概率是共享 device-level HBM 资源**                          |
| 两个 NeuronCore-v2 的 SBUF 是否彼此直连？                     | **官方资料没有显示直接 SBUF-to-SBUF datapath**                    |
| 两个 core 之间是否能传 tensor/activation/KV/partial result？ | **能，但应通过 DMA / HBM / collective 路径表达**                  |
| 是否有类似 GPU SM 间共享 L2 cache 的 coherent path？          | **没有看到这种模型，NeuronCore 更接近 software-managed SRAM + DMA** |

关键证据是：NKI 架构指南说每个 device 有 32 个 DMA engine，可以在 **within and across devices** 搬运数据；单个 NeuronCore-v2 又配有 16 个 DMA engine，可在系统中任意 addressable memories 之间搬运数据，文档重点讨论 local SBUF 与 HBM 之间的搬运。 

这说明 **数据流动的语义不是“core datapath 互联”，而是“addressable memory 之间的 DMA transfer / collective transfer”**。

---

## 2.2 我建议把两个 NeuronCore 之间的数据流分成三类

### A. Pipeline parallel / layer split：activation 从 Core0 到 Core1

例如：

```text
Core0 执行 Layer i
Core0 PSUM/SBUF 得到 activation
Core0 DMA store 到 HBM
Core1 DMA load 到 Core1 SBUF
Core1 执行 Layer i+1
```

或者如果硬件/运行时允许 source/destination 是两个 core 的 SBUF address，则可以抽象成：

```text
Core0 SBUF -> DMA -> Core1 SBUF
```

但在公开文档里，AWS 重点描述的是 **HBM ↔ local SBUF**，没有明确展示 “Core0 SBUF 直达 Core1 SBUF” 的物理路径，所以做 DSL 时我建议保守建模为：

```text
NC0.SBUF -> DMA -> HBM/device memory -> DMA -> NC1.SBUF
```

这更稳，不会过度假设。

---

### B. Tensor parallel：需要 collective communication

如果两个 NeuronCore 分担同一个 GEMM / Attention head / MLP shard，那么中间可能需要：

```text
all-reduce / reduce-scatter / all-gather
```

Inferentia2 顶层图里有 **Collective Communication** block，NKI 指南中也提到 6 个 CC-Core 用于 collective communication；NeuronLink-v2 则用于 chip-to-chip collective communication。 

因此：

```text
Core0 partial output
Core1 partial output
        ↓
CC-Core / collective fabric
        ↓
reduced / gathered result
        ↓
写回 HBM 或各自 SBUF
```

这类路径应该在 DSL 里作为 **collective engine**，不要建模成普通 NoC router。

---

### C. Shared HBM model state / KV cache：两个 core 访问同一 device memory

Inferentia2 的 HBM 被描述为 device memory，用于 storing model state，总容量 32 GiB、带宽 820 GiB/s。

因此两个 NeuronCore 很可能都能通过 DMA 访问 device-level HBM 中的 tensor。数据流不是 core-to-core，而是：

```text
HBM tensor shard / weight / KV
        ↓ DMA
Core0.SBUF

HBM tensor shard / weight / KV
        ↓ DMA
Core1.SBUF
```

这更像 **分布式 scratchpad + 共享 off-chip memory**，不是统一 cache hierarchy。

---

# 3. 单个 NeuronCore-v2 的控制流：多 sequencer、多 instruction stream、显式同步

NeuronCore-v2 是一个 fully-independent heterogeneous compute unit，包含 Tensor / Vector / Scalar / GPSIMD 四类主引擎，以及 software-managed SRAM。

更重要的是，NKI 架构指南明确说：

> 每个 compute engine 都有自己的 sequencer，负责 instruction fetch、decode、issue；四个 compute engine 执行四条独立 instruction stream，并且异步并行运行。数据依赖通过硬件 atomic semaphore 显式同步，编译器根据 NKI kernel 中的数据依赖自动插入同步。

所以控制流可以这样理解：

```text
NeuronCore-v2 control model

Tensor Sequencer  -> Tensor Engine datapath
Vector Sequencer  -> Vector Engine datapath
Scalar Sequencer  -> Scalar Engine datapath
GPSIMD Sequencer  -> GPSIMD Engine datapath / programmable processors
Sync Sequencer    -> mostly DMA trigger / control instructions

各 sequencer 独立 fetch/decode/issue
各 engine 异步执行
跨 engine dependency 由 atomic semaphore 约束
```

这和 GPU SM 的 SIMT warp scheduler 不一样，也和 TPU v1 单一 systolic control 不一样。它更像：

> **一个小型 VLIW/dataflow-ish 异构 tile，但不是静态 VLIW；每个 engine 有独立 sequencer，编译器负责把 operator 拆成多条 engine stream，并插入 semaphore。**

---

# 4. 单个 NeuronCore-v2 的数据流：HBM → SBUF → Engine → PSUM/SBUF → HBM

## 4.1 本地 memory hierarchy

单个 NeuronCore-v2 内部有两个软件管理 SRAM：

| Buffer |     容量 | 用途                                | 组织方式                               |
| ------ | -----: | --------------------------------- | ---------------------------------- |
| SBUF   | 24 MiB | 主数据存储，input/output/intermediate   | 128 partitions，每 partition 192 KiB |
| PSUM   |  2 MiB | Tensor Engine accumulation buffer | 128 partitions，每 partition 16 KiB  |

文档明确说 SBUF 和 PSUM 都是二维 memory，均有 128 partitions。

核心数据路径是：

```text
HBM
 ↓ DMA load
SBUF
 ↓
Tensor / Vector / Scalar / GPSIMD
 ↓
PSUM or SBUF
 ↓ DMA store
HBM
```

其中 HBM ↔ SBUF 通过 DMA；PSUM ↔ SBUF 通过 compute engine ISA instruction 完成。

---

## 4.2 SBUF 的关键心智模型：P 维并行，F 维时间展开

SBUF/PSUM 是 2D memory：

```text
SBUF tensor shape: [P, F]

P dimension = partition dimension = 并行维
F dimension = free dimension      = 时间/streaming 维
```

文档在 Figure 58 附近说明，compute engine 访问 SBUF/PSUM 时，每周期可以跨 128 个 partition 读 128 个元素、计算、再写 128 个元素；因此 P axis 是并行访问维，F axis 是时间展开维。

这对你做 DSL 很关键。NeuronCore-v2 的 tensor tile 不是普通 `[M, N]` 二维数组，而应该建模成：

```text
Tile {
  partition_axis: P,  // <= 128, maps to SBUF/PSUM partitions
  free_axis: F,       // stream over time
  dtype,
  buffer: SBUF | PSUM
}
```

---

# 5. Tensor Engine 数据流：SBUF → systolic array → PSUM

Tensor Engine 是 NeuronCore-v2 的主算力单元，内部是 128 × 128 systolic array。它从 SBUF stream 输入，输出写到 PSUM。文档给出的吞吐是 BF16/FP16/TF32/cFP8 最大约 92 TFLOPS，FP32 约 23 TFLOPS。

## 5.1 matmul 的真实硬件视角

NKI 里的：

```python
nc_matmul(x, y)
```

在硬件视角不是简单的 `x @ y`，而是被映射成：

```text
stationary[K, M] -> LoadStationary
moving[K, N]     -> MultiplyMoving
output[M, N]     -> PSUM
```

每次 `nc_matmul(stationary, moving)` 实际有两类 ISA instruction：

| 指令                 | 作用                                            |
| ------------------ | --------------------------------------------- |
| LoadStationary, LS | 从 SBUF 读取 stationary，缓存到 Tensor Engine 内部     |
| MultiplyMoving, MM | 从 SBUF 读取 moving，与已加载的 stationary 相乘，结果写 PSUM |

文档明确说 TensorE 必须从 SBUF 读输入矩阵、写输出矩阵到 PSUM；每个 matmul 会执行 LoadStationary 和 MultiplyMoving 两类指令。

---

## 5.2 PSUM 是 Tensor Engine 的 landing buffer

TensorE 的输出不是直接回 SBUF，而是进入 PSUM：

```text
SBUF.stationary  ─┐
                  ├── TensorE 128x128 systolic ──> PSUM
SBUF.moving      ─┘
```

PSUM 支持近存累加。文档说明：VectorE 和 ScalarE 对 PSUM 有读写访问，TensorE 只有写访问；PSUM 是 TensorE 的 landing buffer，支持 read-accumulate-write，每个 4B element 可累加，但累加机制只能由 TensorE 控制。

这意味着 K 维大于 128 时，可以这样：

```text
K tile 0: TensorE writes PSUM bank
K tile 1: TensorE accumulates into same PSUM bank
K tile 2: TensorE accumulates into same PSUM bank
...
final PSUM -> Vector/Scalar copy/convert/post-op -> SBUF
```

这点和 GPU tensor core 的 accumulator register file 不同。NeuronCore-v2 是显式 PSUM SRAM accumulation。

---

# 6. Vector / Scalar / GPSIMD 的数据流

## 6.1 Vector Engine：128 lanes，适合 reduction / elementwise-with-multiple-inputs

Vector Engine 有 128 条并行 vector lanes，每条 lane 从一个 SBUF/PSUM partition stream 数据，执行数学操作，再写回 SBUF/PSUM。

它适合：

```text
LayerNorm
Reduction
Pooling
tensor_tensor
tensor_scalar
softmax 中部分 reduce
```

文档还说 VectorE 支持 limited cross-partition data movement：每组 32 partition 里可以做 32x32 transpose 和 32-partition shuffle。

所以 VectorE 不只是 ALU，也带一点小型 reshape/shuffle 能力。

---

## 6.2 Scalar Engine：128 lanes，适合 activation / nonlinear / scalar map

ScalarE 也是 128 lanes，但语义上是：

```text
out[p][f] = func(in[p][f])
```

适合：

```text
GELU
sqrt
exp-like nonlinear
activation + scale + bias
activation_reduce
```

官方单页说明 ScalarEngine 面向每个 output element 只依赖一个 input element 的计算；VectorEngine 面向 output element 依赖多个 input elements 的计算。

这两个 engine 的区别可以理解为：

| Engine  | 语义                                           |
| ------- | -------------------------------------------- |
| ScalarE | map / activation / nonlinear                 |
| VectorE | reduce / elementwise multi-input / vector op |
| TensorE | matmul / conv / Tensor reshape               |
| GPSIMD  | fallback / custom op / irregular op          |

---

## 6.3 GPSIMD Engine：8 个 512-bit programmable vector processors

GPSIMD 是 NeuronCore-v2 相比前代很重要的增强。官方文档说它包含 8 个 fully-programmable 512-bit vector processors，可以执行 general-purpose C code，并访问 embedded on-chip SRAM，用于 custom operators。

这说明 AWS 的设计不是“所有 op 都必须硬连线”，而是：

```text
常规大算子：Tensor / Vector / Scalar
不规则或新 op：GPSIMD custom operator
```

这对于 LLM 推理很重要，因为 mask、sampling、特殊 reshape、某些 tokenizer/后处理或者未来模型中的新 op，可能可以落到 GPSIMD。

---

# 7. 单 core 内部互联/访问约束：不是任意 crossbar

这一点对你最重要：NeuronCore-v2 内部不是“所有 engine 任意访问所有 buffer 且完全并行”。

文档明确给出两个访问限制：

1. VectorE 和 GPSIMD 不能并行访问 SBUF
2. VectorE 和 ScalarE 不能并行访问 PSUM

这些 serialization 由 Neuron Compiler 插入。

同时，文档又说 SBUF 可以同时给某些 engine 的 tensor read/write interface 提供峰值带宽，例如 VectorE/ScalarE/TensorE 或 GPSIMD/ScalarE/TensorE 组合；PSUM 也可以给 VectorE/TensorE 或 ScalarE/TensorE 同时提供峰值带宽。

所以我会把单 core 内部端口抽象成：

```text
SBUF ports:
  - TensorE read path
  - VectorE read/write path
  - ScalarE read/write path
  - GPSIMD read/write path
  - DMA read/write path

PSUM ports:
  - TensorE write/accumulate path
  - VectorE read/write path
  - ScalarE read/write path

Restrictions:
  - VectorE.SBUF conflicts with GPSIMD.SBUF
  - VectorE.PSUM conflicts with ScalarE.PSUM
```

这比简单画成一个全互联 crossbar 更准确。

---

# 8. 我对 AWS NeuronCore-v2 架构风格的判断

如果和你之前研究的 TPU / TT / SambaNova / Cerebras 对比，我会这样定位 AWS NeuronCore-v2：

| 架构                 | 核心风格                                                                                               |
| ------------------ | -------------------------------------------------------------------------------------------------- |
| TPU v1             | 大 systolic array + 统一 buffer + host-driven                                                         |
| Tenstorrent Tensix | many tiles + local SRAM + NoC + RISC controller                                                    |
| SambaNova RDU      | dataflow fabric + PCU/PMU + reconfigurable routing                                                 |
| Cerebras WSE       | wafer-scale many cores + local memory + fabric                                                     |
| AWS NeuronCore-v2  | **少量大 core，每 core 内部是 heterogeneous multi-engine + software-managed SRAM + DMA/collective system** |

NeuronCore-v2 不像 GPU 那样依赖 cache，也不像 Cerebras 那样把大量小 core 铺满 wafer。它更像一个 **operator-level heterogeneous tile**：

```text
一个 NeuronCore-v2 ≈ 
  128x128 Tensor systolic array
+ 128-lane Vector engine
+ 128-lane Scalar activation engine
+ 8-core programmable GPSIMD
+ 24 MiB SBUF
+ 2 MiB PSUM
+ per-engine sequencer
+ DMA/sync/semaphore
```

---

# 9. 对你问题的直接回答

## Q1：Inferentia2 中两个 NeuronCore-v2 间是否会有数据流动？

**会，但不要理解成两个 core 内部 datapath 直接互联。**

更合理的路径是：

```text
Core0.SBUF / PSUM
   ↓ engine / DMA
HBM or device address space
   ↓ DMA
Core1.SBUF
```

或者对于 tensor parallel / multi-chip collective：

```text
Core partial result
   ↓
CC-Core / Collective Communication
   ↓
HBM / SBUF / remote device
```

官方资料没有显示 “Core0 TensorE output 直接流到 Core1 TensorE input” 这种硬连线，所以建模时应避免假设 direct core-to-core streaming link。

---

## Q2：单个 NeuronCore-v2 的控制流是怎样的？

```text
NKI / compiled kernel
   ↓
per-engine instruction streams
   ├── Tensor sequencer
   ├── Vector sequencer
   ├── Scalar sequencer
   ├── GPSIMD sequencer
   └── Sync sequencer
        ↓
control instructions:
   - scalar register ALU
   - dynamic condition/address calculation
   - branch
   - DMA trigger

datapath instructions:
   - read/write SBUF/PSUM
   - TensorE matmul
   - VectorE reduce/elementwise
   - ScalarE activation
   - GPSIMD custom program

dependency:
   - atomic semaphore
   - compiler inserted synchronization
```

---

## Q3：单个 NeuronCore-v2 的数据流是怎样的？

典型 matmul + post-op 数据流：

```text
1. DMA load weights/activation
   HBM -> SBUF

2. TensorE compute
   SBUF.stationary -> LoadStationary
   SBUF.moving     -> MultiplyMoving
   TensorE output  -> PSUM

3. K-tile accumulation
   PSUM bank accumulates FP32 partial sums

4. post-processing
   PSUM -> VectorE/ScalarE -> SBUF
   例如 bias/add/activation/cast/reshape

5. DMA store
   SBUF -> HBM

6. 若跨 core
   HBM / collective path -> other NeuronCore.SBUF
```

---

# 10. 给你的 DSL 建模建议

我建议你不要把 NeuronCore-v2 建模成普通 tile + NoC router，而是建模成 **heterogeneous engine cluster**：

```yaml
NeuronCoreV2:
  local_memory:
    SBUF:
      size: 24MiB
      partitions: 128
      partition_size: 192KiB
      axes: [P, F]
    PSUM:
      size: 2MiB
      partitions: 128
      partition_size: 16KiB
      banks_per_partition: 8
      accumulation_dtype: fp32

  engines:
    TensorE:
      type: systolic_array
      shape: [128, 128]
      input: SBUF
      output: PSUM
      instructions: [LoadStationary, MultiplyMoving]
    VectorE:
      lanes: 128
      read_write: [SBUF, PSUM]
      supports: [reduce, elementwise, transpose_32x32, partition_shuffle]
    ScalarE:
      lanes: 128
      read_write: [SBUF, PSUM]
      supports: [activation, scale_bias_activation, activation_reduce]
    GPSIMD:
      processors: 8
      simd_width: 512b
      read_write: [SBUF]
      supports: [custom_operator, irregular_op]
    SyncEngine:
      supports: [dma_trigger, control]

  synchronization:
    type: atomic_semaphore
    inserted_by: compiler

  memory_movement:
    HBM_to_SBUF: DMA
    SBUF_to_HBM: DMA
    PSUM_to_SBUF: engine_instruction
    SBUF_to_PSUM: TensorE_or_engine_instruction

  conflicts:
    - [VectorE.SBUF, GPSIMD.SBUF]
    - [VectorE.PSUM, ScalarE.PSUM]
```

一句话总结：**AWS NeuronCore-v2 是“software-managed SRAM + 多异构 engine + per-engine sequencer + DMA/collective”的架构，两个 core 之间有系统级数据流，但公开资料不支持把它理解为 core datapath 直连。**
