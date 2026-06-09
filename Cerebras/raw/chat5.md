> 现在继续深挖 fabric

好，继续深挖 **Cerebras fabric**。这部分非常关键，因为 Cerebras 不是“90 万个 core 摆在一起”就能工作，真正让 wafer-scale 成立的是它的 **wafer-scale 2D mesh dataflow fabric**。🌐

我先给结论：

> **Cerebras fabric 是一个静态路由、细粒度 packet、支持 broadcast/multicast/reduction、跨 reticle/die 边界统一延展的 wafer-scale NoC。它不是通用 cache-coherent NoC，而是为 ML dataflow、weight broadcast、partial-sum reduction 和 sparse trigger 专门设计的 fabric。**

---

# 1. Cerebras fabric 的基本参数

从 HC2022、WSE-3 白皮书和 HC2024 可以抽出这些关键信息：

| 项                   | Cerebras fabric                                           |
| ------------------- | --------------------------------------------------------- |
| 拓扑                  | 2D mesh                                                   |
| router 位置           | 每个 core 内/旁边一个 fabric router                              |
| router 端口           | 5-port：N/S/E/W + local core                               |
| link 宽度             | 32b/cycle bidirectional                                   |
| packet 粒度           | 16b data + 16b control/index = 32b packet                 |
| 路由方式                | static routing                                            |
| local static routes | 每 core 24 个 colors                                        |
| buffer              | 每个 color 有 dedicated buffering                            |
| flow control        | lossless flow control, low buffering                      |
| 单跳延迟                | 相邻 core single-cycle latency                              |
| 通信能力                | unicast / broadcast / multicast / reduction pattern       |
| WSE-2 fabric BW     | 220 Pbit/s                                                |
| WSE-3 fabric BW     | 214 Pbit/s                                                |
| 跨 die               | scribe line 上高层金属短线，source-synchronous parallel interface |
| 缺陷处理                | redundancy + training + auto-correction state machine     |

HC2022 明确写到：fabric 是低开销 2D mesh，每个 router 有 5 个端口连接四邻居和 core，32b/cycle 双向传输，payload 携带 16b data 和 16b index，core 间单周期延迟，低 buffer flow control，24 个可配置 static routing colors，每个 color 有专用 buffer 且 non-blocking，所有 colors 在同一物理 link 上 time-multiplex，并支持硬件 broadcast/multicast。

WSE-3 白皮书给出的 WSE-3 总 fabric bandwidth 是 **214 Pb/s**，WSE-2/HC2021 给出的是 **220 Pb/s**。

---

# 2. 它是不是 NoC？是，但不是通用 NoC

你之前讨论 Graphcore exchange fabric 时已经抓住一个关键点：**NoC 不一定都长成“复杂 router + 动态路由 + cache coherence”那种形式。**

Cerebras fabric 完全满足 NoC 的基本定义：

```text
node/core + router + link + topology + packet + routing + flow control
```

但它的目标不是 CPU/GPU coherent traffic，也不是通用多 master 共享内存互连，而是：

```text
1. weight / activation / control packet 的细粒度传输
2. packet 到达触发 core 内 handler
3. weight broadcast / multicast
4. partial sum reduction
5. 静态编译规划的数据流通信
6. sparse nonzero packet 触发计算
```

所以更准确地说：

> **它是 wafer-scale static-routed dataflow NoC。**

和你之前学习的典型 NoC 对照：

| 维度                  | 传统 CPU/GPU NoC                          | Cerebras fabric                 |
| ------------------- | --------------------------------------- | ------------------------------- |
| 主要流量                | cache line、memory request、coherence msg | 16b data + control/index packet |
| 路由                  | 常见 XY/adaptive/dynamic                  | static routing                  |
| 端点                  | core/cache/memory controller            | dataflow core                   |
| 数据粒度                | 32B/64B cache line 或更大 packet           | 32b fine-grained packet         |
| 共享内存语义              | 常有                                      | 没有传统共享内存                        |
| 编译器控制               | 较弱                                      | 极强                              |
| broadcast/multicast | 不一定核心                                   | 核心能力                            |
| 目标                  | 通用通信                                    | ML dataflow                     |

---

# 3. Router 微架构：5-port 但非常“瘦”

Cerebras router 的基本结构是：

```text
          North
            ▲
            │ 32b
            │
West ◄──  Fabric Router  ──► East
            │
            │
            ▼
          South

          Local Core Port
```

五个端口：

```text
N / S / E / W / Core
```

每个方向是 32b/cycle bidirectional link。白皮书说每个 core 里有 fabric router，四个 cardinal directions 都是 32-bit bidirectional interface，加一个面向 core 的 port；router 使用 lossless flow control 和 low buffering，邻居之间 single clock cycle latency。

这个 router 的设计点很明显：

```text
1. 不做复杂动态路由
2. 不做大 buffer
3. 不做复杂 cache coherence
4. 靠 static route 和 compiler 规划降低硬件开销
5. 通过 wafer 内短线获得低能耗/低延迟
```

这就是为什么它能放到每个 core 上，而且规模扩到 90 万 core。

如果 router 设计成 GPU/CPU 那种复杂 NoC router，面积、功耗、验证复杂度都会爆炸。

---

# 4. Packet 粒度：32-bit fine-grained packet

Cerebras 的 fundamental data packet 是：

```text
16-bit data element + 16-bit control/index = 32-bit packet
```

其中 16-bit data 通常是 FP16/BF16 数据，control/index 用于表达控制、索引、命令或路由相关语义。WSE-3 白皮书明确说 fundamental packet 是 single 16-bit data element，加上 16-bit control information 组成 32-bit fine-grained packet。

这和传统 NoC 的差异非常大。

传统 NoC 可能传：

```text
cache line: 64B
memory request packet
DMA burst
coherence packet
```

Cerebras 传的是：

```text
一个 FP16/BF16 weight
+ 一个 index/control
```

为什么这么细？

因为它的执行模型是：

```text
一个 nonzero weight packet 到达
→ 触发 core 执行 AXPY/FMCA
```

如果 packet 粒度太大，就不适合 fine-grained sparse trigger。
如果 packet 粒度太小，路由开销又会上升。Cerebras 选择 32b，是因为它的 link 本身就是 32b/cycle，packet 和 link 宽度对齐。

---

# 5. Static routing：这是 Cerebras fabric 的核心设计选择

Cerebras 明确说 fabric 使用 **entirely static routing**。原因是 neural network communication 具有较强静态结构，适合编译期规划。

这意味着：

```text
router 不需要根据每个 packet 的完整目的地址动态算路由
router 只需根据预配置 route/color 做转发
```

可以想象成：

```text
color 3:
  input west → output east
  input core → output north
  multicast to east + north
  ...

color 7:
  input north → output core
  input west → output south
  ...
```

具体实现细节官方没有完全公开，但从材料可以判断：

> **color 本质上是一个局部静态 route ID / logical route resource。**

它不是传统意义上的“目的地址路由”，而更像：

```text
packet carries color/control
router has per-color configured forwarding behavior
fabric uses colors to compose global communication path
```

这点非常像你之前分析 Graphcore exchange fabric 时说的：
**不是 packet 到 router 后由复杂 header 逻辑临时决定去哪，而是编译器提前配置好路径，运行时 packet 只沿着预设通路流动。**

---

# 6. Colors：最容易误解，也最关键

官方说每个 core 有 **24 local static routes called colors**，所有 colors 彼此 nonblocking，并 time-multiplex 到同一 physical link 上；每个 color 有 dedicated buffering。

我们可以拆解。

## 6.1 color 不是颜色，是“静态逻辑通道/路由配置”

可以把 color 理解为：

```text
color = statically configured local route + buffer context + packet class
```

它至少承担三类语义：

```text
1. 路由语义：这个 packet 沿哪条预配置路径走
2. 通道语义：不同 color 有独立 buffer，避免互相阻塞
3. 控制语义：core 可以根据 input fabric color/control lookup handler
```

材料里也说 core 的 instruction lookup 可以基于 fabric color 或 control information。

所以 color 同时影响：

```text
router forwarding
core-side handler dispatch
buffer isolation
dataflow stream identity
```

---

## 6.2 color 和传统 Virtual Channel 的区别

它有点像 VC，但不能完全等同。

| 维度     | Virtual Channel          | Cerebras color                                |
| ------ | ------------------------ | --------------------------------------------- |
| 主要目的   | 避免 deadlock / 提高吞吐 / QoS | 静态 route + dataflow stream + buffer isolation |
| 路由     | 仍可能动态/XY/adaptive        | static route                                  |
| 语义     | 网络层通道                    | 网络 + 编译映射 + handler 语义                        |
| buffer | per-VC buffer            | per-color dedicated buffer                    |
| 配置     | 通常固定协议                   | 编译器/配置期设置 route                               |

所以更准确的说法：

> **color 既像 virtual channel，又像 statically configured route label，还像 dataflow stream ID。**

---

## 6.3 “nonblocking between colors” 怎么理解？

官方说 all colors are nonblocking between one another，同时又说 all colors are time-multiplexed onto same physical links。

这两句话看似矛盾，其实可以这样理解：

```text
物理 link 仍然共享，所以同一周期同一方向不可能无限并发；
但每个 color 有独立 buffer/route context，
一个 color 的 backpressure 不会把其他 color 的 buffer 状态直接堵死。
```

也就是说，“nonblocking”更可能指：

```text
per-color buffering / flow-control independent
避免 head-of-line blocking across colors
```

而不是说：

```text
24 个 color 可以在同一根 32b link 上同一周期同时传 24 个 packet
```

因为它明确说 colors 是 time-multiplexed onto same physical link。

---

# 7. Flow control：lossless + low buffering

Cerebras 使用 lossless flow control with low buffering。

这说明它不希望 packet drop/retry，因为 dataflow compute 中 packet 可能就是触发计算的事件，丢包代价很高。

但它又不想放大 buffer，因为 90 万 router 上每个 router 都放大 buffer 会很贵。所以设计目标是：

```text
1. 静态路由减少拥塞不确定性
2. 编译器规划流量，避免超订阅
3. per-color buffer 减少 HOL blocking
4. lossless backpressure 保证正确性
5. low buffering 控制面积/功耗
```

从 NoC 角度看，这是典型的：

> **以编译期流量规划换运行时硬件复杂度。**

---

# 8. Broadcast / multicast：Cerebras fabric 的一等能力

神经网络通信天然有大量 fan-out，比如：

```text
1. 一个 weight 广播给同一列/多列 core
2. command 广播触发 reduction
3. control packet 同步多个 core 行为
```

所以 Cerebras router 支持 native broadcast and multicast。白皮书明确说，神经网络通信具有高 fan-out，因此 fabric router 设计了 native broadcast/multicast 能力。

这和普通 NoC 很不一样。普通 NoC 如果没有硬件 multicast，broadcast 可能需要：

```text
source 复制 N 份 packet
分别 unicast 到 N 个目的地
```

这样带宽和功耗都很差。

Cerebras 的做法更像：

```text
packet 到 router 后按静态配置 fork 到多个方向
```

示意：

```text
          North
            ▲
            │
West ───► Router ───► East
            │
            ▼
          Core
```

一个 weight packet 可以沿途复制，使得权重流高效覆盖一列或一片 core。

---

# 9. Reduction：不是 router 自动 all-reduce，而是 fabric + core tensor op + static route 配合

Cerebras 的 partial sum reduction 很关键。材料里讲 Sparse GEMM 时说：

```text
1. 每个 core 得到 partial sum
2. PSUM/FSUM command broadcast 给各列/各 core
3. command 到达后触发 partial sum reduction
4. reduction compute 用 fabric tensor operands 的 tensor instruction
5. partial sums 通过 ring pattern across wafer 通信
6. ring pattern 由 static routing colors 设置
7. reduction 可与下一组 weight 的 FMAC overlap
```



这说明 reduction 不是一个完全隐藏在 router 里的“网络内计算 all-reduce”。更准确地说是：

```text
static-routed fabric 提供通信路径
core tensor instruction 执行 reduction computation
PSUM/FSUM command 触发 reduction 阶段
microthread 负责 overlap
```

所以它是：

> **fabric-assisted reduction / statically scheduled reduction pattern。**

而不是 GPU NCCL 那种运行时 collective，也不是 switch 内部透明 reduce。

---

# 10. 跨 die / reticle 边界：wafer-scale 的真正难点

WSE 不是一个普通单 die。它由 84 个 die 区域构成，但没有切开封装成多个芯粒，而是在 wafer 上保留整片。WSE-2 的材料里说从 core 到 die 再到 wafer：每个 die 约 17mm × 30mm，66 × 154 cores，约 10,156 cores，总共 12 × 7 die = 84 die。

问题来了：die/reticle 边界怎么连？

材料说：

```text
fabric crosses less than 1mm of scribe line
using high-level metal layers
source-synchronous parallel interface
only a few short wires per interface
more than 1 million wires total
defects unavoidable
redundancy built into protocol
training and error detection discover errors
auto-correction state machines use redundant links
```



HC2024 还强调 WSE-3 把跨 reticle boundary 的连接扩展到了 5nm，并且和 die-level fabric/system software 协同；每个 die 内是 2D mesh，跨 die 边界 full performance 延展，die level 和 wafer level 是 uniform fabric，并内置 redundancy 绕过 failure，软件看到的是 fully uniform 2D mesh。

这点非常厉害，也非常关键。

普通 chiplet 之间是：

```text
die → package substrate / interposer / UCIe / SerDes / parallel D2D
```

Cerebras 是：

```text
die region → scribe line 高层金属短线 → neighbor die region
```

距离小于 1mm，能耗和带宽都更接近片上互连。

---

# 11. “软件看到 uniform 2D mesh”背后的含义

HC2024 说软件总是看到 fully uniform 2D mesh。

这句话对你的 DSL 非常重要。真实物理层有：

```text
1. die boundaries
2. scribe line crossing
3. defective links
4. redundant links
5. training/auto-correction
```

但软件/编译器看到的是：

```text
logical uniform 2D mesh
```

这意味着 Cerebras 下面至少有一层：

```text
physical fabric → defect-aware repaired fabric → logical fabric exposed to compiler
```

它和你之前说的三层非常吻合：

```text
Physical connectivity
Legal reachability
Scheduled dataflow
```

对于 Cerebras，更准确是：

```text
Physical topology:
  wafer-scale mesh with die boundaries, possible bad links

Repaired logical topology:
  uniform 2D mesh after redundancy/training/auto-correction

Compiler route topology:
  static colors and multicast/reduction paths configured on logical mesh

Scheduled dataflow:
  weight broadcast, command packet, PSUM/FSUM ring, sparse packet trigger
```

这对你做架构 DSL/IR 是一个非常好的例子。

---

# 12. 为什么 wafer fabric 比 GPU interconnect 能效高？

HC2022 给过一个对比：用和 A100 等面积的 WSE-2 sub-fabric 来比较，GPU estimate 是 826 mm²、0.6 TB/s、0.7 GB/s/mm²、10 pJ/bit、60W；WSE-2 sub-fabric 是 826 mm²、4.3 TB/s、5.2 GB/s/mm²、0.15 pJ/bit、6W；对应 7× 带宽、7× 单位面积带宽、66× pJ/bit 优势、10× 功耗优势。

这个对比的本质不是 router 多神奇，而是物理距离：

```text
WSE:
  wafer 内高层金属短线，<1mm

GPU multi-chip:
  die → package → NVLink/SerDes → switch → package → die
```

长距离高速 SerDes 的能耗远大于片上短线。Cerebras 用 wafer-scale 把很多“chip-to-chip”通信变成“on-wafer short-wire”通信。

所以它的 fabric 优势来自：

```text
1. 物理距离短
2. 无 SerDes
3. 静态路由降低控制开销
4. 低 buffer
5. 细粒度 packet 与 ML 数据类型匹配
```

---

# 13. Fabric 和 core 是耦合设计，不是独立 NoC IP

Cerebras fabric 不能像普通 NoC IP 那样孤立理解，因为它和 core 的 dataflow trigger 是绑定的。

白皮书说 fabric 传输 data 和 associated control，core 收到后 lookup instruction；整个 compute fabric 因此成为 dataflow engine。

也就是说：

```text
fabric packet 不只是 transport data
fabric packet 也是 execution event
```

这点非常重要。

在 GPU/CPU NoC 中：

```text
packet 到达 L2/cache/controller
→ 完成 memory transaction
```

在 Cerebras 中：

```text
packet 到达 core
→ 触发 handler
→ tensor instruction 执行
→ 可能产生新的 fabric packet
```

所以它的 fabric 更像数据流机器的神经系统：

```text
communication = synchronization = scheduling trigger = compute activation
```

---

# 14. Fabric 在 MatMul 中的具体角色

以 Cerebras 的 wafer MatMul array 为例，fabric 负责三类流量：

## 14.1 Weight broadcast

```text
weights 从 wafer 边缘/输入通道进入
沿着 static route 广播到对应 columns/cores
nonzero weights 才发送
```

作用：

```text
把同一个 weight 分发给持有相关 activation 的 core
```

## 14.2 Command broadcast

例如：

```text
PSUM command
FSUM command
synchronization / reduction command
nonlinear op command
```

作用：

```text
触发某个阶段的 computation 或 reduction
```

## 14.3 Partial sum reduction

```text
每个 core 有 partial sum
通过 ring pattern across row/wafer 做 reduce
FSUM column 存 final result
```

作用：

```text
完成 MatMul 的 sum over hidden/reduction dimension
```

材料中明确说 reduction 是由 command packet 触发，PSUM/FSUM 控制，ring pattern 由 static routing colors 设置，并且能与下一行 weight 的 FMAC overlap。

---

# 15. Cerebras fabric 为什么适合稀疏？

稀疏场景下，传统架构会遇到几个问题：

```text
1. 稀疏 index overhead
2. 不规则访存
3. load imbalance
4. 小粒度计算调度开销
5. zero skipping 省下的计算被控制/访存吃掉
```

Cerebras 的 fabric + core 组合试图这样解决：

```text
1. zero 在 sender 端过滤，根本不发
2. nonzero packet 携带 data + index/control
3. packet 到达直接触发 compute
4. weight broadcast 减少复制开销
5. local SRAM 中 activation resident，避免远程取 activation
6. microthread 填补动态稀疏带来的 bubble
```

HC2021 材料也说 WSE 的 AXPY full performance、低 fixed overhead、高 bandwidth interconnect 能使 unstructured sparsity speedup 接近线性，至少在 GPT-3 layer 12k×12k MatMul 的展示中如此。

但这里也要批判性看：

> fabric 能高效“承载”稀疏流量，不等于真实模型总能给你 10× 有效稀疏收益。

收益仍然依赖：

```text
模型是否能稀疏
稀疏是否保持精度
nonzero 分布是否导致局部拥塞
reduction 是否成为瓶颈
metadata/index/control overhead 是否可控
```

---

# 16. Fabric 的瓶颈在哪里？

Cerebras 的材料强调优势，但从架构角度看，fabric 也有潜在瓶颈。

## 16.1 Bisection bandwidth 和热点

2D mesh 的天然问题是：

```text
长距离通信延迟随 Manhattan distance 增加
全局通信可能压中间 bisection
broadcast/reduction 可能形成热点
```

Cerebras 通过静态 mapping、broadcast/multicast、ring reduction、tensor layout 来缓解，但不是物理上没有问题。

---

## 16.2 Static routing 的灵活性限制

静态路由低开销，但意味着：

```text
运行时自适应能力弱
拥塞绕行能力有限
动态 shape / 动态通信模式不友好
```

如果 workload 通信模式可预测，静态路由很好；如果通信模式高度动态，静态 route 会受限。

---

## 16.3 Color 数量有限

每 core 24 colors 很强，但不是无限。复杂模型中同时存在：

```text
weight streams
activation streams
gradient streams
partial sum reductions
commands
sync
data movement
pipeline overlap
```

都要映射到有限 color / buffer / link time slots 上。编译器必须做资源分配。

---

## 16.4 Fine-grained packet 的 overhead

32b packet 很适合 sparse trigger，但如果 workload 是大块 dense transfer，细粒度 packet 的控制开销比例可能不低。Cerebras 的优势是它的 link 很宽、短线能效高，但 packet granularity 的 trade-off 仍然存在。

---

## 16.5 与 local SRAM layout 强绑定

fabric 流量是否高效，强依赖：

```text
tensor dimension → physical mesh dimension
core SRAM placement
weight broadcast direction
reduction direction
output layout
```

layout 错了，fabric 再强也会被重排/远距离搬运拖垮。

---

# 17. 和 Graphcore fabric 的对比

你之前问过 Graphcore exchange fabric 是否是 NoC。Cerebras 和 Graphcore 很适合对比：

| 维度                  | Graphcore IPU exchange fabric | Cerebras WSE fabric              |
| ------------------- | ----------------------------- | -------------------------------- |
| 规模                  | 芯片级上千 tile                    | wafer 级 85万/90万 core             |
| 拓扑公开度               | 公开材料较抽象                       | 明确 2D mesh、5-port router         |
| 路由                  | 编译器强控制                        | static routing colors            |
| 同步模型                | BSP/exchange 更显式              | dataflow trigger 更显式             |
| packet 语义           | exchange data                 | data + control/index 触发 compute  |
| broadcast/multicast | 有通信支持但不是最核心叙事                 | 明确是核心能力                          |
| reduction           | exchange 后计算/同步               | PSUM/FSUM + ring + fabric tensor |
| 架构哲学                | bulk-synchronous many-tile    | fine-grained dataflow wafer mesh |

一句话：

> **Graphcore 更像“编译器排好的 BSP exchange machine”；Cerebras 更像“packet-triggered dataflow mesh machine”。**

---

# 18. 和 GPU NVLink/NVSwitch 的对比

Cerebras HC2024 用 GPU memory wall 做推理叙事，其中一个核心对比是：GPU 要靠多 GPU 聚合 HBM bandwidth，但多 GPU 聚合需要 NVLink/NVSwitch，带来高功耗和通信复杂度。HC2024 中展示 DGX-H100 通过 8×H100 聚合 memory bandwidth，并需要高速 serial links 和 switch。

从 fabric 角度看：

| 维度      | GPU NVLink/NVSwitch                | Cerebras wafer fabric            |
| ------- | ---------------------------------- | -------------------------------- |
| 物理范围    | package/server/chip-to-chip        | wafer 内                          |
| link 类型 | 高速 SerDes/parallel D2D/switch      | 高层金属短线                           |
| 通信粒度    | GPU memory/tensor parallel traffic | fine-grained data/control packet |
| 拓扑      | 多 GPU network                      | 2D mesh                          |
| 软件      | tensor parallel / collective       | static route / dataflow          |
| 能耗      | 长距离高速 IO 高                         | 短距离片上低                           |
| 灵活性     | 更通用                                | 更专用                              |

这就是 Cerebras “scale-up first” 的根本优势：
把很多 GPU 需要跨芯片做的事情，压到 wafer 内部完成。

---

# 19. 用 NoC 术语重写 Cerebras fabric

如果用标准 NoC 术语，它大概是：

```yaml
network:
  topology: 2D_mesh
  node_count: 850k_or_900k
  router:
    ports: [north, south, east, west, local]
    radix: 5
    link_width: 32b_per_cycle_bidirectional
    buffering: low
    flow_control: lossless
    virtual_resources:
      name: colors
      count: 24
      buffer_per_color: true
  packet:
    width: 32b
    fields:
      data: 16b
      control_or_index: 16b
  routing:
    type: static
    configured_by: compiler_or_runtime_configuration
  collective_support:
    broadcast: native
    multicast: native
    reduction: static_route_plus_core_tensor_instruction
  physical:
    die_to_die:
      crossing: scribe_line_high_level_metal
      interface: source_synchronous_parallel
      redundancy: true
      training_and_auto_correction: true
  abstraction:
    software_view: uniform_2D_mesh
```

这套 YAML 对你的 DSL 很接近。

---

# 20. 对你的架构 DSL / Event IR 的启发

Cerebras fabric 对你有几个非常直接的启发。

## 20.1 NoC DSL 必须表达 static route

不能只写：

```yaml
topology: mesh
routing: xy
```

还要能写：

```yaml
routes:
  - color: 3
    path: ...
    multicast: ...
    buffer: dedicated
```

## 20.2 NoC DSL 必须表达 broadcast/multicast/reduction pattern

AI fabric 不是只做 point-to-point。

```yaml
collectives:
  broadcast:
    hardware_supported: true
  multicast:
    hardware_supported: true
  reduce:
    pattern: ring
    compute_location: core
    trigger: command_packet
```

## 20.3 Event IR 必须把 packet arrival 当成事件

Cerebras 的核心事件是：

```text
packet_arrival(color, data, control)
```

它会触发：

```text
handler_lookup
tensor_fmac
tensor_reduce
fabric_output
```

## 20.4 Physical topology 和 logical topology 要分开

Cerebras 有坏 link/冗余/auto-correction，但软件看到 uniform mesh。你的 DSL 应该能表达：

```yaml
physical_topology:
  mesh_with_defects

repair_layer:
  redundancy
  route_around_failures

logical_topology:
  uniform_2D_mesh

scheduled_dataflow:
  static_colors
```

这非常符合你之前的分层思路。

---

# 21. Fabric 的一句话心智模型

> **Cerebras fabric 是把 90 万个 small dataflow cores 组织成一台 wafer-scale dataflow computer 的神经系统：它用 2D mesh + 5-port router + 32b fine-grained packet + static colors + broadcast/multicast + ring reduction，把权重、命令和 partial sums 变成可编译、可静态调度、可触发计算的数据流。**

下一步建议继续深挖：**colors 到底该怎么类比：是 VC、route ID、stream ID，还是 handler dispatch tag？** 这个问题非常适合结合你现在正在学的 NoC router/allocator/VC/backpressure 来讨论。
