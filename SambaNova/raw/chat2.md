> RDN / switch / vector-scalar-control 三网络如何把 PCU 和 PMU 串成 dataflow fabric，每个网络的拓扑是什么？电路该怎么设计才能可配置

可以，继续深挖。这里先给结论：**SambaNova 的 RDN 不是一个“普通 NoC + 三种 packet type”，而是三张并行物理网络：vector / scalar 是 packet-switched mesh，control 是 circuit-switched bit-level routing fabric。它们共同把 PCU、PMU、AGCU 串成一个可编译的 spatial dataflow pipeline。** 🚀

---

# 1. RDN 的拓扑到底是什么？

公开资料里最明确的说法是：

> **RDN 是连接 PCU、PMU、AGCU 的 on-chip programmable interconnect，由 vector、scalar、control 三个 physical fabrics 组成；vector/scalar 是 packet-switched，control 是 circuit-switched；RDN 用 non-blocking switches 组成 mesh。** 

所以可以先画成：

```text
          S ---- S ---- S ---- S
          |      |      |      |
         PMU    PCU    PMU    PCU
          |      |      |      |
          S ---- S ---- S ---- S
          |      |      |      |
         PCU    PMU    PCU    PMU
          |      |      |      |
          S ---- S ---- S ---- S
```

但每个 `S` 不是一张网，而是三张并行网络的 switch：

```text
Switch S =
  Vector Switch Plane   // tensor data
  Scalar Switch Plane   // address / metadata
  Control Switch Plane  // token / sync / orchestration
```

更准确：

```text
        +-----------------------------+
        | Switch tile                 |
        |                             |
        |  vector router/crossbar     |
        |  scalar router/crossbar     |
        |  control bit router         |
        |                             |
        |  config tables / counters   |
        +-----------------------------+
```

HotChips 2024 明确说 SN40L 是 **mesh interconnect made of 3 physical networks**，并支持 point-to-point、one-to-many multicast、many-to-one transmission、2D DOR + software override。

---

# 2. 三张网络各自负责什么？

## 2.1 Vector network：搬 tensor data

Vector network 是主数据平面，负责搬 tensor vector：

```text
PMU vector out -> switch mesh -> PCU vector in
PCU vector out -> switch mesh -> PMU vector in
AGCU vector out -> switch mesh -> PMU / PCU
```

它的特点：

* packet-switched
* 高带宽
* 承载 tensor payload
* 支持 point-to-point / multicast / many-to-one
* packet-level credit flow control
* packet 里带 metadata，例如 sequence ID

SN40L paper 明确说 vector fabric 是 tensor data 的 primary conduit；vector packet 带 sequence ID，用于 many-to-one stream 和重排序。sequence ID 可由 unit 的 vector output port 生成，然后 PMU 用 sequence ID 参与 write address 计算，从而把多源 packet 写回正确逻辑顺序。

所以 vector 网络不是只负责“送到目的地”，还配合 PMU 实现：

```text
many sources -> one PMU -> reorder by sequence_id -> logical tensor order
```

---

## 2.2 Scalar network：搬地址、metadata，也可搬小数据

Scalar network 是元数据平面：

```text
PMU scalar ALU -> scalar network -> other PMU / PCU
AGCU address/control metadata -> scalar network
```

它的特点：

* packet-switched
* 带宽小于 vector，但更适合 word-level metadata
* 主要搬地址、loop index、predicate、metadata
* 有时也可以搬小数据或控制信息

SN40L paper 明确说 scalar fabric 主要传输 metadata，例如 address；但有些情况下也可以承载 data 或 control。

这个设计很像把 dataflow pipeline 拆成两类 stream：

```text
vector stream = 数据本体
scalar stream = 数据如何被解释 / 地址如何生成 / 当前循环状态
```

这对 PMU 很关键，因为 PMU 的 scalar ALU pipeline 可以从 scalar RDN 吃 operand，也可以把计算出的 scalar 输出到 scalar RDN；这让复杂地址计算可以跨多个 PMU 组合完成。

---

## 2.3 Control network：搬 token，不搬数据

Control network 是最不像普通 NoC 的部分。

它的特点：

* circuit-switched
* bit-level granularity
* 单 bit wire bundle，可单独 route
* 传 flow control、synchronization、graph orchestration token
* 通常对应 counter done event / loop done event

SN40L paper 明确说 control fabric 是 **circuit-switched**，由一组 single-bit wires 构成，每根 wire 可以单独 route；control token 通常对应 counter “done” event，用于粗粒度 flow control 和 graph orchestration。

直觉上：

```text
PCU counter done
      |
      v
control network token
      |
      v
enable downstream PMU / PCU / AGCU
```

它不是 packet 网络，因为 control token 的需求是：

* 极低开销
* 低延迟
* 固定路径
* 不需要大 payload
* 需要可预测同步

所以用 circuit-switched bit wire 很合理。

---

# 3. 为什么要分成 vector / scalar / control 三网？

因为 dataflow graph 里天然有三类 traffic：

```text
1. 大块 tensor 数据流
2. 地址/metadata/loop index
3. 同步/启动/完成 token
```

如果都放一张 packet NoC，会出现三个问题：

## 问题 A：大数据包阻塞控制 token

tensor data packet 很大。如果 control token 也走同一网络，就可能被大流量 backpressure 卡住，影响 pipeline orchestration。

## 问题 B：地址/metadata 和数据带宽需求不同

PMU/PCU 对地址 metadata 的需求是低粒度、低带宽、高时序敏感；vector tensor 则是高带宽 bulk stream。混在一起不利于物理设计。

## 问题 C：dataflow 需要“数据路径”和“控制路径”分离

在 spatial pipeline 中，很多 stage 是：

```text
收到 control token -> 开始读 PMU
PMU 生成 scalar address -> 输出 vector data
PCU 消费 vector data -> counter done -> 发 control token
```

所以三网分离可以让 pipeline 更接近硬件流水线，而不是软件 kernel launch。

SN10 时代也已经有类似思想：switch 是 programmable packet-switched interconnect，具有独立 data/control buses，以适配真实应用中的不同 traffic class，并支持 programmable routing 和 programmable counters。

---

# 4. 一个 PCU/PMU dataflow 是怎么被三网串起来的？

以：

```text
PMU_A, PMU_B -> PCU_GEMM -> PMU_C -> PCU_ACT -> PMU_D
```

为例。

实际运行不是“PCU 发 load 指令去读 PMU”，而更像：

```text
Control network:
  token_start -> PMU_A / PMU_B 开始发流

Scalar network:
  PMU_A/PMU_B 内部地址生成需要的 scalar metadata
  或把 loop index / address operand 传给其他 PMU

Vector network:
  PMU_A vector data -> PCU_GEMM
  PMU_B vector data -> PCU_GEMM
  PCU_GEMM output -> PMU_C
  PMU_C output -> PCU_ACT
  PCU_ACT output -> PMU_D

Control network:
  PCU_GEMM counter done -> PMU_C / PCU_ACT token
  PCU_ACT done -> next stage token
```

可以画成：

```text
 control:  start ───────► PMU_A / PMU_B
 scalar:   addr/meta ───► PMU address pipelines
 vector:   A/B data ────► PCU_GEMM ───► PMU_C ───► PCU_ACT
 control:                 done ───────► next stage
```

这就是 **dataflow fabric**：
不是一个 core 主动 fetch 所有数据，而是 compiler 配好路径后，数据和 token 自己在网络里流。

---

# 5. Switch 内部电路大概怎么设计？

公开资料没有给出 transistor-level 或完整 RTL，所以这里要分清楚：**下面是基于公开描述和常规 NoC/CGRA 实现推导的合理电路结构，不是 SambaNova 官方 RTL。**

一个 switch 大概率可以抽象成：

```text
                North
                  |
      West -- [ Switch ] -- East
                  |
                South
                  |
               Local ports
             PCU / PMU / AGCU
```

每个方向都有三套端口：

```text
North.vector / North.scalar / North.control
South.vector / South.scalar / South.control
East.vector  / East.scalar  / East.control
West.vector  / West.scalar  / West.control
Local.vector / Local.scalar / Local.control
```

---

## 5.1 Vector / Scalar switch：packet router + crossbar

Vector/scalar 都是 packet-switched，所以内部基本结构类似 NoC router：

```text
Input ports
  -> input FIFO / skid buffer
  -> route lookup / DOR
  -> VC or stream arbitration
  -> crossbar
  -> output credit check
  -> output link
```

简化图：

```text
         in_N ─┐
         in_S ─┤
         in_E ─┤       +----------+
         in_W ─┼──────►| crossbar |────► out_N
       local ──┘       +----------+────► out_S
                         ▲       └────► out_E
                         │       └────► out_W
                    arbitration    └──► local
```

每个 packet header 至少需要包含：

```text
dest / flow_id
maybe sequence_id
packet type / length
credit / stream info
payload
```

但 SambaNova 的重点是：它支持两种 routing：

1. **硬件 2D dimension-order routing, DOR**
2. **software-controlled static flow routing**

SN40L paper 说 vector/scalar packet 可以动态使用 2D DOR，也可以使用静态 flow route；静态 flow routing 中，软件给 packet stream 分配 flow ID，flow ID 随 packet 携带，在每个 switch port 被 decode，并在转发前被重新分配，从而支持 multicast。

所以 switch 里至少需要：

```text
if mode == DOR:
    out_port = dimension_order_route(current_xy, dest_xy)
else:
    entry = route_table[in_port][flow_id]
    out_ports = entry.output_port_mask
    next_flow_id = entry.next_flow_id
```

---

## 5.2 Static flow routing 怎么支持 multicast？

普通 DOR 是单播：

```text
packet -> one output port
```

但 static flow route 可以让一个输入复制到多个输出：

```text
flow_id = 17
switch route table:
  input west, flow 17 ->
      output east with flow 23
      output south with flow 9
```

电路上就是：

```text
          input packet
               |
         route table lookup
               |
       output_port_mask = 01010
               |
       replicate packet to selected outputs
```

例如：

```text
             ┌──► out_E, flow_id=23
in_W flow17 ─┤
             └──► out_S, flow_id=9
```

这就是 one-to-many multicast。

为什么需要重写 flow ID？因为同一条逻辑 stream 在不同分叉后，后续路径不同；每段路径可以有本地 flow ID，减少全局路由表复杂度。

---

## 5.3 Many-to-one 怎么做？

Many-to-one 不只是网络收敛，还要保证逻辑顺序。SambaNova 的机制是：

```text
多个 vector source -> 同一个 PMU
每个 packet 带 sequence_id
PMU 用 sequence_id 参与 write address 计算
写到正确位置
```

公开论文明确说 vector packet 的 sequence ID 是支持任意 many-to-one streams 的主要机制；vector output port 有可编程逻辑生成 sequence ID，PMU 用它计算 write address 来 reorder packet。

所以电路上不是靠网络保证全局有序，而是：

```text
Network only delivers packets
PMU reorders using sequence_id
```

这很聪明，因为 NoC 保序成本很高，尤其 many-to-one 时更难。把顺序语义下沉到 PMU 的 address generation，就更适合 dataflow。

---

# 6. Control switch：为什么是 circuit-switched？

Control network 是 bit-level circuit-switched，可理解为“可配置的 bit-wire 交叉连接”。

一个 control switch 可以很像小型可编程 crossbar：

```text
control_in[N/S/E/W/local bits]
          |
          v
   programmable mux matrix
          |
          v
control_out[N/S/E/W/local bits]
```

例如每个 output bit 有一个 mux：

```text
out_E_bit0 = mux(sel_E0,
                 in_W_bit3,
                 in_N_bit1,
                 local_done_bit,
                 const0,
                 const1)
```

也可以支持 bit-level route：

```text
PCU0.done -> PMU3.enable
PMU3.done -> PCU7.start
AGCU.memcpy_done -> PCU12.start
```

这类控制线不适合 packet-switched，因为 packet router 会有：

* header overhead
* buffering
* arbitration latency
* credit/backpressure
* jitter

而 control token 更像同步电路：

```text
done pulse -> routed wire -> enable pulse
```

所以用 circuit-switched bit fabric 更符合 spatial dataflow。

---

# 7. “可配置”到底配置了什么？

RDN 的可配置不是 FPGA 那种 LUT 级别重构，而是 CGRA/NoC 级别配置。主要配置项包括：

## 7.1 Vector/scalar route table

```text
flow_id -> output_port_mask
flow_id -> next_flow_id per output
flow_id -> priority / arbitration class
```

用于 static flow routing / multicast。

## 7.2 DOR / static route 模式

```text
route_mode = DOR | STATIC_FLOW
```

DOR 用于一般 packet，static route 用于 compiler 已知的 dataflow stream。

## 7.3 Credit / flow-control 参数

SN40L 资料说 vector/scalar 每跳使用 credit flow control，同时 packet stream 还有 end-to-end flow control，结合 coarse-grained software tokens、fine-grained hardware credits 和硬件 forward-progress guarantee。

所以可能有：

```text
per output credits
per flow credits
FIFO depth threshold
stall propagation policy
```

## 7.4 Control bit routing

```text
control_out[i] = selected control_in[j]
```

或者：

```text
control token route:
  source event -> sink enable
```

## 7.5 Sequence ID generation

vector output port 有可编程 sequence ID 生成逻辑。配置项可能类似：

```text
seq_id = base + f(loop_i, loop_j, tile_id, source_id)
```

PMU 再用 sequence ID 计算 write address。

---

# 8. 为什么说它不是普通 Mesh NoC？

普通 mesh NoC 的目标是：

```text
任意节点之间传 packet，保证吞吐、延迟、公平性、死锁避免
```

RDN 的目标是：

```text
把编译器已知的数据流图映射成稳定的硬件 pipeline
```

所以 RDN 需要的能力不同：

| 能力          | 普通 NoC                | SambaNova RDN                      |
| ----------- | --------------------- | ---------------------------------- |
| 单播          | 需要                    | 需要                                 |
| 多播          | 可选                    | 很重要                                |
| many-to-one | 常见但不一定语义化             | 明确支持，并配合 sequence ID               |
| 控制 token    | 通常作为 packet           | 独立 circuit-switched control fabric |
| 路由          | 动态为主                  | DOR + compiler static flow routing |
| 有序性         | packet/order QoS      | PMU 用 sequence ID 重排               |
| 目标          | general communication | spatial dataflow pipeline          |

---

# 9. 如果自己设计一个类似 RDN，电路上可以怎么落地？

可以按三层设计。

## Level 1：基础 mesh router

每个 switch：

```text
5 ports = N/S/E/W/Local
每 port:
  vector in/out
  scalar in/out
  control in/out
```

vector/scalar：

```text
input FIFO
route compute
allocator
crossbar
output credit
```

control：

```text
bit-level configurable mux/crossbar
```

---

## Level 2：支持 static flow

在 vector/scalar router 里加 route table：

```text
route_table[in_port][flow_id] = {
    output_mask,
    next_flow_id_for_each_output,
    priority,
}
```

这样支持：

```text
point-to-point:
  output_mask = one-hot

multicast:
  output_mask = multi-hot

flow relabel:
  next_flow_id changes per branch
```

---

## Level 3：支持 dataflow semantics

加三类机制：

```text
1. sequence_id generator at unit output
2. PMU write-address generation using sequence_id
3. control token route for stage enable/done
```

这三个机制让网络从“传 packet”变成“执行 dataflow graph”。

---

# 10. 对你最重要的架构理解

SambaNova RDN 的关键不是 mesh，而是：

```text
mesh topology
+ three traffic planes
+ programmable static routes
+ multicast
+ sequence-ID-based many-to-one reorder
+ circuit-switched control tokens
+ compiler PnR
```

真正把 PCU/PMU 串成 dataflow fabric 的，是 **compiler place-and-route**。公开论文明确说三个 RDN fabric 的 routing tables 都由软件通过 compiler 内的 PnR layer 配置。

所以从架构探索角度，RDN 不能只建模成：

```yaml
noc:
  topology: mesh
  bandwidth: X
```

更应该建模成：

```yaml
rdn:
  topology: 2d_mesh
  planes:
    vector:
      type: packet_switched
      payload: tensor_data
      routing: [dor, static_flow]
      multicast: true
      credits: hop_and_e2e
      sequence_id: true
    scalar:
      type: packet_switched
      payload: metadata_address_scalar
      routing: [dor, static_flow]
      multicast: true
    control:
      type: circuit_switched
      granularity: bit
      payload: tokens
      route_config: mux_matrix
  switch:
    ports: [N, S, E, W, local]
    route_table_per_flow: true
    flow_id_rewrite: true
```

---

# 一句话总结

**RDN 是三张并行的 2D mesh dataflow 网络：vector packet mesh 搬 tensor，scalar packet mesh 搬地址/metadata，control circuit-switched bit mesh 搬同步 token；可配置性主要来自 compiler 配置 switch route table、flow ID 重写、multicast 路径、sequence ID 生成和 control bit routing。**
