> 继续分析：DSR + tensor instruction 到底该怎么理解？它和普通 SIMD/vector ISA、GPU tensor core、TPU systolic array 的差异是什么

好，这个问题是理解 Cerebras core 的关键。我们可以把它拆成两层：

> **DSR 解决“tensor 在哪里、长什么样、怎么流动”的问题；tensor instruction 解决“对这个 tensor 做什么操作”的问题。**

它和 SIMD/vector/GPU tensor core/TPU systolic array 的核心差异在于：**Cerebras 把 tensor operand 直接做成 ISA 层的一等对象，并且让这个 tensor operand 可以来自本地 SRAM，也可以来自 fabric stream。**

---

# 1. 先给一个直觉模型：DSR 像“硬件可读的 tensor iterator”

普通标量指令大概是：

```asm
add r1, r2, r3
```

意思是：

```text
r1 = r2 + r3
```

这里 operand 是 **寄存器**。

普通 load/store 是：

```asm
load r1, [addr]
```

这里 operand 是 **内存地址**。

Cerebras tensor instruction 更像：

```text
fmac [fpsum] = [fpsum], [fwd_wgt], r_in
```

它的 operand 里出现了 `[fpsum]`、`[fwd_wgt]` 这种 tensor operand。HC2022 明确说 Cerebras 的 ISA 将 tensors 作为 first-class operands，并给出类似 `fmach [fpsum] = [fpsum], [fwd_wgt], r_in` 的例子；这些 tensor operand 由 44 个 DSR 描述，DSR 可以描述 tensor 地址、长度、维度、stride，也可以描述 FIFO 和 fabric tensor。

所以 DSR 可以理解成：

```text
DSR = tensor descriptor + tensor iterator + streaming endpoint descriptor
```

它不是保存 tensor 数据本身，而是保存：

```text
这个 tensor 从哪里来？
base address 是多少？
shape 是什么？
stride 怎么走？
长度是多少？
是 SRAM tensor、FIFO、circular buffer，还是 fabric-streaming tensor？
```

---

# 2. DSR 不是普通寄存器，也不是 vector register

这一点很重要。

## 2.1 普通 GPR 存“值”

例如：

```text
r0 = 3
r1 = address
r2 = loop counter
```

GPR 里面直接是标量值。

## 2.2 SIMD/vector register 存“一组数据”

例如：

```text
v0 = [a0, a1, a2, a3, ...]
v1 = [b0, b1, b2, b3, ...]
v2 = v0 + v1
```

vector register 里面保存实际数据。

## 2.3 DSR 存“如何访问 tensor 的描述”

DSR 里面不是 tensor 数据，而是 tensor 的 **访问描述符**：

```text
DSR_fpsum:
  base = local_sram_addr_0
  rank = 3D
  shape = [B_tile, S_tile, H_tile]
  stride = [...]
  length = ...
  mode = local_memory_tensor

DSR_fabric_wgt:
  mode = fabric_streaming_tensor
  color = ...
  expected_index = ...
  format = FP16/BF16
```

官方材料说 DSR 包含 pointer 以及 tensor 的 length、shape、size 等信息，并且硬件状态机会使用 DSR 在 datapath 上 full performance 地遍历整个 tensor。

所以最准确的类比是：

> **DSR ≈ 硬件化的 tensor descriptor / iterator / stream descriptor。**

---

# 3. Tensor instruction 到底做了什么？

Cerebras 的 tensor instruction 不是“一条指令只做一个 scalar op”，而是：

> **一条指令描述一个 tensor-level operation，硬件根据 DSR 自动展开成一串 datapath 操作。**

举一个抽象例子：

```text
fmac [Z] = [Z], [X], a
```

含义可能是：

```c
for i in tensor_range(DSR_Z, DSR_X):
    Z[i] = Z[i] + X[i] * a
```

但这个循环不一定真的由软件一条条执行，而是：

```text
1. 指令指出操作类型：FMAC
2. DSR_Z 描述累加 tensor
3. DSR_X 描述输入 tensor
4. a 来自 fabric packet 或 GPR
5. tensor control 硬件状态机展开访问
6. SIMD/FMCA datapath 执行
```

Cerebras 官方文章说，DSR 后面有硬件状态机，能够使用 DSR 并在 datapath 上 full performance 地 sequence through the full tensor。

所以 Cerebras tensor instruction 的本质是：

```text
operation + tensor descriptor operands + hardware tensor iterator
```

而不是：

```text
operation + fixed vector registers
```

---

# 4. 为什么 Cerebras 要这样设计？

核心原因是 Cerebras 的计算不是围绕“寄存器文件中的 vector”展开，而是围绕：

```text
local SRAM resident tensor
+
fabric-streaming tensor
+
packet-triggered compute
```

展开。

在 Cerebras 中，activation 通常被分布在各 core 的本地 SRAM 中，权重以 packet 形式通过 fabric broadcast/stream 进来。core 收到 data/control packet 后，会触发 handler instruction lookup，再使用 DSR 描述的 tensor operand 做计算。白皮书明确说，fabric 同时传输 data 和 associated control，core 收到后根据 fabric 输入触发 instruction lookup，并且只发送 nonzero data 来触发计算。

所以 Cerebras core 的常见执行形态是：

```text
weight packet arrives
  ↓
lookup handler
  ↓
handler issues tensor FMAC
  ↓
DSR selects local activation tensor + local accumulator tensor
  ↓
datapath does AXPY/FMCA
```

这和传统处理器从 instruction stream 主动 load operands 很不一样。
Cerebras 更像：

> **数据来了，触发一段已经配置好的 tensor computation。**

---

# 5. 一个具体例子：Sparse GEMM 中的 DSR + tensor instruction

假设一层 MatMul：

```text
Y = X @ W
```

Cerebras 把 sparse GEMM 看成：

```text
每个 nonzero weight 触发一次 AXPY
```

也就是：

```text
Y[:, i] += W[k, i] * X[:, k]
```

在一个 core 上，可能有：

```text
DSR_act:
  指向本 core SRAM 中的一段 activation X[:, k]

DSR_acc:
  指向本 core SRAM/cache 中的一段 partial sum Y[:, i]

DSR_wgt:
  描述来自 fabric 的 streaming weight packet
```

当某个 nonzero weight `w = W[k, i]` 到达：

```text
fabric packet:
  data = w
  index/control = k, i, color, command
```

core 做：

```text
handler = lookup(packet.color/control)

execute:
  fmac [acc] = [acc], [act], w
```

这条 tensor instruction 逻辑上等价于：

```c
for j in local_BS_tile:
    acc[j] += act[j] * w
```

关键是：

1. `act[j]` 在本地 SRAM；
2. `acc[j]` 可能在 software-managed cache 或 SRAM；
3. `w` 是 fabric packet 带来的 scalar；
4. zero weight 根本不发送，因此不会触发这条 tensor instruction。

这就是为什么 Cerebras 说它能支持 fine-grained unstructured sparsity。HC2021/HC2022 都强调 sparse compute 由 fine-grained dataflow core 触发，并且只对 nonzero data 执行计算。

---

# 6. 和普通 SIMD ISA 的区别

普通 SIMD 的典型模型是：

```text
load vector from memory
load vector from memory
vector add/mul/fmac
store vector back
```

例如：

```asm
vload v0, [x]
vload v1, [y]
vfma  v2, v0, v1
vstore [z], v2
```

它的核心抽象是：

```text
vector register = 一组 lane 数据
SIMD instruction = 对这些 lane 同时做同一种操作
```

Cerebras 的抽象不同：

```text
DSR = tensor 的地址/shape/stride/stream 描述
tensor instruction = 对 DSR 描述的 tensor 执行一个操作
```

对比一下：

| 维度           | 普通 SIMD                    | Cerebras DSR + tensor instruction                 |
| ------------ | -------------------------- | ------------------------------------------------- |
| operand      | vector register            | tensor descriptor / fabric tensor / local tensor  |
| 数据位置         | 通常先 load 到 vector register | 可以直接从 SRAM/fabric tensor 流入 datapath              |
| 粒度           | 固定 SIMD 宽度                 | tensor-level，由 DSR 展开                             |
| shape/stride | 软件 loop 管理                 | DSR + hardware state machine 管理                   |
| 触发方式         | instruction stream 主动执行    | fabric packet/control 可触发 handler                 |
| 主要目标         | 提升数据并行吞吐                   | 支持 local tensor + stream tensor + sparse dataflow |

所以 Cerebras 不是“没有 SIMD”。WSE-2 core 里有 4×FP16 FMAC，WSE-3 有 8-way 16b SIMD 和 16-way 8b SIMD。区别是：**SIMD 是底层 datapath，DSR/tensor instruction 是上层 ISA 和调度抽象**。WSE-3 的 core 明确包含 8-way 16b SIMD、16-way 8b SIMD、48KB SRAM、512B local cache 和 48 个 DSR。

一句话：

> **Cerebras 底层仍然用 SIMD datapath，但程序员/编译器看到的不是简单 vector register，而是 DSR 描述的 tensor stream。**

---

# 7. 和 Vector ISA 的区别，比如 RISC-V V / ARM SVE

Vector ISA 比 SIMD 更灵活，因为它有 vector length、mask、stride load/store 等机制。

典型 vector ISA 是：

```text
setvl N
vload v0, base_x
vload v1, base_y
vfma v2, v0, v1
vstore base_z, v2
```

它的核心抽象是：

```text
一条 vector instruction 处理 VL 个元素
```

这和 Cerebras 有些相似，因为两者都不是单 scalar。但差异很大：

| 维度   | Vector ISA                            | Cerebras DSR/tensor ISA                     |
| ---- | ------------------------------------- | ------------------------------------------- |
| 执行对象 | vector register 中的元素                  | memory/fabric tensor                        |
| 长度控制 | VL / predicate / mask                 | DSR 的 shape/length/stride                   |
| 数据来源 | load/store 从 memory 到 vector register | local SRAM tensor 或 fabric streaming tensor |
| 控制流  | processor instruction stream          | fabric packet 可以触发 handler                  |
| 通信   | 通常不内建 NoC 语义                          | fabric tensor 是 operand 类型之一                |
| 稀疏   | 通常靠 mask/index/软件库                    | zero 不发包，nonzero packet 触发计算                |

所以 DSR 更像是：

```text
vector descriptor + tensor layout descriptor + fabric stream endpoint
```

而不是传统 Vector ISA 里的 VL 寄存器。

---

# 8. 和 GPU Tensor Core 的区别

GPU Tensor Core 的抽象一般是：

```text
D = A × B + C
```

通常在 warp / warpgroup 层面执行 MMA 指令：

```text
mma.sync / wgmma
```

它的强项是：

```text
dense tile matrix multiply
```

比如：

```text
16×16×16
64×N×K
```

GPU Tensor Core 的 operand 需要经过：

```text
HBM → L2 → shared memory/register → tensor core
```

它依赖：

```text
tiling
blocking
coalescing
shared memory reuse
warp-level scheduling
```

Cerebras tensor instruction 不是这种“固定 tile MMA engine”。它更像：

```text
对 DSR 描述的一段 tensor 执行 FMAC/AXPY/reduction/nonlinear 等操作
```

特别是在 sparse GEMM 中，Cerebras 是：

```text
一个 nonzero weight packet
  → 触发一个 AXPY-style tensor FMAC
```

而不是：

```text
凑齐一个 dense tile
  → 送进 tensor core 做 MMA
```

对比：

| 维度          | GPU Tensor Core                 | Cerebras Tensor Instruction                 |
| ----------- | ------------------------------- | ------------------------------------------- |
| 主要操作        | dense MMA tile                  | tensor FMAC / AXPY / reduction / stream ops |
| operand     | register/shared memory tile     | DSR 描述的 local/fabric tensor                 |
| 最佳 workload | dense GEMM，高复用                  | sparse/AXPY/GEMV/低复用/streaming              |
| 稀疏支持        | 多为结构化稀疏或库级处理                    | nonzero packet dataflow trigger             |
| 数据流         | 软件 tiling + cache/shared memory | 编译器静态 layout + fabric stream                |
| 控制粒度        | warp/CTA/kernel                 | core-local handler + fabric event           |

所以可以说：

> **GPU Tensor Core 是 dense tile MMA accelerator；Cerebras tensor instruction 是 dataflow tensor-operand instruction。**

---

# 9. 和 TPU systolic array 的区别

TPU systolic array 的典型模型是：

```text
activation 从一个方向流入
weight 从另一个方向流入
partial sum 在阵列中传递/累加
```

它非常适合规则 dense GEMM。其核心是：

```text
规则时空数据流 + 大矩阵阵列 + high utilization dense compute
```

Cerebras 的 wafer 级 MatMul array 表面上也像一个巨大阵列，但底层不同：

| 维度       | TPU Systolic Array             | Cerebras WSE                        |
| -------- | ------------------------------ | ----------------------------------- |
| 基本单元     | 简单 MAC cell / systolic PE      | 可编程 dataflow core                   |
| 数据流      | 规则 wavefront                   | packetized fabric dataflow          |
| operand  | dense activation/weight stream | local tensor + fabric packet/tensor |
| 稀疏       | 不天然适合细粒度非结构化稀疏                 | nonzero packet 天然触发                 |
| 控制       | 阵列级控制较强                        | 每 core 有 handler/DSR/microthread    |
| topology | 专用矩阵阵列                         | wafer-scale 2D mesh NoC             |
| 灵活性      | 高效但相对固定                        | 更灵活，但编译复杂                           |

TPU 的优势是规则 dense GEMM 的面积/能效极强。
Cerebras 的优势是把每个 PE 做成可编程 dataflow core，能处理 fabric-triggered sparse tensor operations。

一句话：

> **TPU 是“规则 dense systolic wave”；Cerebras 是“可编程 packet-triggered tensor dataflow”。**

---

# 10. DSR 的核心价值：把循环控制从软件变成硬件状态机

你可以把一个 tensor op 分成三部分：

```text
1. 算什么：FMAC / ADD / REDUCE / nonlinear
2. 算哪些元素：shape / length / stride / index
3. 数据从哪来，到哪去：SRAM / fabric / FIFO / circular buffer
```

在传统 CPU/GPU 里，第 2 和第 3 点很大程度靠软件 loop、address generation、load/store、shared memory 编排完成。

Cerebras 用 DSR 把第 2 和第 3 点硬件化：

```text
DSR 负责：算哪些元素 + 怎么访问 + 怎么流动
tensor instruction 负责：对这些元素做什么计算
```

这带来几个好处：

## 10.1 指令开销低

一条 tensor instruction 可以覆盖一段 tensor，而不是每几个元素发一条 scalar/SIMD 指令。

## 10.2 地址生成开销低

shape/stride/length 在 DSR 中，硬件状态机自动遍历。

## 10.3 适合 streaming

operand 可以是 fabric-streaming tensor，不一定要先 load 到 register file。

## 10.4 适合 sparse trigger

nonzero packet 到来后可以直接触发预配置 tensor operation。

## 10.5 适合小 core

每个 core 很小，如果用复杂软件 loop 管理所有细节，控制开销太高。DSR 帮它用较少控制逻辑驱动 tensor datapath。

---

# 11. 但 DSR 也有代价：灵活性被“描述符表达能力”限制

DSR 很强，但不是万能。

它能高效描述：

```text
规则 tensor
固定 shape/stride
FIFO
circular buffer
fabric stream
静态映射后的 tensor chunk
```

但如果 workload 需要：

```text
高度不规则 pointer chasing
复杂动态数据结构
频繁变化的 shape/layout
不可预测控制流
随机访存
```

DSR 的优势就下降。因为 DSR 的本质是：

> **把规则或半规则的数据访问模式预先描述给硬件。**

所以它特别适合神经网络里的 tensor computation，但不适合通用程序。

这也是 Cerebras core 和 CPU/GPU 的哲学差异：
它牺牲了通用动态性，换取 wafer-scale 上可静态规划、低开销、高带宽的数据流执行。

---

# 12. 从编译器视角看 DSR + tensor instruction

编译器需要把 PyTorch/TensorFlow 里的 op lowering 成每个 core 的局部程序和 DSR 配置。

例如一个全局 MatMul：

```python
Y = X @ W
```

编译器要决定：

```text
1. X 的哪一块放到哪个 core 的 SRAM
2. Y 的 partial sum 放到哪个 core
3. W 的 nonzero stream 从哪里进来
4. fabric color / route 怎么配置
5. 每个 core 的 DSR_act / DSR_acc / DSR_wgt 怎么配置
6. 到达什么 packet 触发什么 handler
7. PSUM/FSUM command 触发哪条 reduction tensor instruction
```

这和 GPU 编译器的区别是：

```text
GPU 编译器/库主要生成 kernel + tiling + memory movement
Cerebras 编译器生成全 wafer 的 placement + route + DSR + handler + schedule
```

所以 DSR 是硬件 ISA 概念，但它真正发挥作用依赖 compiler。官方文章也说每个 core 的 instructions 由 Cerebras software stack/compiler 生成，从 TensorFlow/PyTorch lowering 而来。

---

# 13. 对你的架构探索 DSL/IR 的启发

如果你要在自己的 DSL 里表达类似 Cerebras 的机制，不能只写：

```yaml
core:
  simd_width: 8
  sram: 48KB
```

还要表达：

```yaml
tensor_descriptor:
  count: 48
  supported_tensor_rank: 4
  address_modes:
    - local_sram
    - fabric_stream
    - fifo
    - circular_buffer
  fields:
    - base
    - length
    - shape
    - stride
    - stream_color
    - index_semantics

tensor_instruction:
  operands:
    - tensor_descriptor
    - scalar_register
    - fabric_payload
  operations:
    - fmac
    - reduce
    - nonlinear
    - move
  issue_trigger:
    - instruction_stream
    - fabric_data_packet
    - fabric_control_packet
```

Event IR 也要从：

```text
compute_tile(matmul)
```

细化成：

```text
configure_DSR(core, DSR_act, layout)
configure_DSR(core, DSR_acc, layout)
packet_arrival(core, color, weight_value, index)
handler_lookup(core, color)
tensor_fmac(core, DSR_acc, DSR_act, weight_value)
command_arrival(core, PSUM/FSUM)
tensor_reduce(core, fabric_tensor_operand)
```

这样才能描述 Cerebras 这种 **packet-triggered tensor dataflow**。

---

# 14. 最重要的对比总结

| 架构                                | 核心 operand 抽象                           | 典型执行                         | 最适合                                  |
| --------------------------------- | --------------------------------------- | ---------------------------- | ------------------------------------ |
| SIMD                              | vector register                         | 同一 op 作用于固定 lanes            | 简单数据并行                               |
| Vector ISA                        | vector register + VL/mask               | 可变长度 vector loop             | 长向量/规则循环                             |
| GPU Tensor Core                   | dense matrix tile                       | warp/CTA 组织 MMA              | dense GEMM                           |
| TPU Systolic                      | array stream                            | 规则 wavefront MAC             | dense MatMul                         |
| Cerebras DSR + tensor instruction | tensor descriptor + fabric/local tensor | packet 触发 tensor op，DSR 硬件遍历 | distributed tensor、AXPY、稀疏、streaming |

---

# 15. 一句话总结

**DSR 是 Cerebras core 里的硬件 tensor descriptor/iterator；tensor instruction 是对这些 DSR 描述的 local/fabric tensor 执行操作的指令。它不是普通 SIMD/vector register 模型，也不是 GPU dense MMA tile，更不是 TPU 固定 systolic wave，而是为 wafer-scale dataflow、权重流、非结构化稀疏和本地 SRAM 张量计算设计的 ISA 抽象。**
