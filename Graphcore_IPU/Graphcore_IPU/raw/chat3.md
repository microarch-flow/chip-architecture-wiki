> 我可以这么理解吗？标准的noc，会有router，router中有一定的硬件逻辑电路，能够根据header flit中的信息来选择将数据从某个口发送出去。他们这套中，没有很复杂的逻辑，节点间交叉时由很多mux组成，编译器决定了某个cycle时某个mux中数据来了后对应的选择信号是什么，这样就提前知道这拍的数据会就像哪里，多个类似mux的组件组成fabric。我理解只有这样才能实现他们描述的功能，你觉得呢？

可以，**你的理解基本正确，而且已经抓到 Graphcore exchange fabric 的本质了**。我会稍微修正两个点，让这个心智模型更精确。🙂

# 1. 你的理解可以这样表述

标准 packet-switched NoC 大概是：

```text
Header flit 到达 router
  ↓
router 解析目的地址 / virtual channel / routing 信息
  ↓
route computation
  ↓
switch / VC allocation
  ↓
crossbar 选择输出端口
  ↓
转发 flit
```

而 Graphcore IPU exchange 更像：

```text
全局 Sync
  ↓
从 sync 后第 0 cycle 开始计时
  ↓
编译器已经安排好：
  哪个 tile 在哪个 cycle 发送
  哪个 tile 在哪个 cycle 接收
  receive mux 在哪个 cycle 选择哪一路
  ↓
fabric 中的数据按固定 pipeline delay 流动
  ↓
接收端按预定 select signal 把数据接走
```

这正好对应 HC2021 第 16 页的信息：Graphcore 明确说 Poplar 编译器会从 sync 开始，以精确 cycle 调度 transmit、receive 和 select，并且知道所有 pipeline delay；addressing 依赖 **time and select state**，没有 queues、arbiters、packet overheads；每个 tile 还有一个 **1600-way receive mux**。

所以你说的“由很多 mux 组成，编译器决定某个 cycle 某个 mux 的选择信号”这个方向是对的。

---

# 2. 但要避免一个过度简化：不一定是任意交叉点都有 mux

你说“节点间交叉时由很多 mux 组成”，这个可以作为**直觉模型**，但物理上可能不是一个完整的二维 crossbar matrix：

```text
1472 sender × 1472 receiver
```

如果真的每个交叉点都放 mux / switch point，面积和布线会非常夸张。

从 HC2021 给出的结构看，更准确的模型是：

```text
每个 tile / IO block 有一个 36b pipelined send channel
这些 send channels 进入 exchange spine
每个 tile 有一个 1600-way receive mux
receiver 在指定 cycle 选择某一路 send channel
```

也就是：

```text
TX0 ───────────────┐
TX1 ───────────────┤
TX2 ───────────────┤
...                ├── pipelined exchange spine
TX1599 ────────────┘
        │      │      │
        ▼      ▼      ▼
      RX mux RX mux RX mux ...
      Tile0  Tile1  Tile2
```

所以不是“每个 router 根据 header 决定下一跳”，而是更像：

```text
每个 receiver 端有一个大 mux
mux select 由静态 schedule 决定
数据流在 spine 中按固定时序传播
```

当然，中间物理实现一定还有 repeater、pipeline register、列方向 transport、时钟/mesochronous 边界等，但从架构语义上，不是 packet router。

---

# 3. 它的“地址”不是 header 地址，而是时间地址

这是最关键的区别。

传统 NoC 里，数据自己携带目的信息：

```text
packet header:
  dst = tile_900
  route = ...
  vc = ...
```

Graphcore exchange 里，官方说：

```text
Addressing is by time and select state
```

这句话非常重要。它意味着：

```text
某个数据在第 t 个 cycle 出现在某条 send channel 上
接收 tile 的 mux 在第 t + latency 个 cycle 选择这条 channel
于是这个数据就被“送到”目标 tile
```

所以目的地不是靠 header flit 动态解释出来的，而是靠：

```text
时间位置 + 接收选择状态
```

来隐式决定。

这有点像火车调度：

```text
传统 NoC：
  每辆车带导航系统，到每个路口现场决定怎么走

IPU exchange：
  时刻表提前排好，到某个时间点，某个站台打开，车自然到对应目的地
```

---

# 4. 为什么这样就能“提前知道这拍的数据会去哪里”？

因为 BSP 给了它一个全局共同起点。

IPU 执行是：

```text
Compute → Sync → Exchange → Compute
```

Programmer’s Guide 也说明，IPU 的 step 包含 local tile compute、global cross-tile synchronisation 和 data exchange；所有 tile 同步后才进入 exchange phase。

有了 sync，所有 tile 可以共享一个类似：

```text
exchange_cycle = cycles_since_last_sync
```

的时间基准。

于是编译器可以生成类似这样的 schedule：

```text
After Sync:

cycle 0:
  tile_3.TX sends word A
  tile_900.RX_SELECT = tile_3

cycle 1:
  tile_7.TX sends word B
  tile_102.RX_SELECT = tile_7

cycle 2:
  tile_3.TX sends word C
  tile_901.RX_SELECT = tile_3
```

如果 tile_3 到 tile_900 的 pipeline latency 是 L，那么真实接收选择可能是：

```text
cycle 0:
  tile_3 send A

cycle 0 + L:
  tile_900 receive mux selects tile_3 path
```

这就是“编译器知道 pipeline delay”的意义。

---

# 5. 和传统 router 的差异可以压缩成一句话

传统 NoC router 是：

```text
数据到了以后，硬件根据 header 决定往哪里走
```

IPU exchange 是：

```text
数据还没到之前，编译器已经决定这个 cycle 哪个 receiver 选择哪条数据通路
```

或者更短：

```text
传统 NoC：data-driven routing
IPU exchange：time-driven selection
```

这个区别非常本质。

---

# 6. “没有复杂逻辑”也要谨慎理解

你说“没有很复杂的逻辑”，大方向对，但不是说硬件很简单到只有 mux。

它仍然需要不少硬件：

```text
pipelined transport
send channel
receive mux
select state machinery
global sync mechanism
mesochronous timing handling
I/O block integration
physical wiring / repeaters / registers
```

但它少掉了传统 packet NoC 中最复杂、最耗能、最动态的部分：

```text
packet header decode
dynamic route computation
VC allocation
switch allocation
input/output buffering
runtime arbitration
congestion handling
backpressure protocol
```

HC2021 第 16 页明确强调没有 queues、arbiters、packet overheads，这就是在说它不是 runtime packet network。

所以更准确地说：

> **不是“没有复杂硬件”，而是“没有传统 router-centric NoC 的动态控制复杂性”。**

---

# 7. 你这个模型对架构探索很有价值

你可以把 IPU exchange fabric 抽象成一个 **scheduled mux fabric**：

```text
fabric = send_channels + deterministic pipeline + receive_muxes + compiler_schedule
```

它的性能模型不是：

```text
latency = router_hops × router_latency + queueing_delay + arbitration_delay
```

而是：

```text
exchange_time =
  schedule_length
  + deterministic_pipeline_latency
  + sync_overhead
```

它的瓶颈也不是传统 NoC 的“某个 router 拥塞”，而是：

```text
每个 tile 每 cycle send/receive 带宽限制
receive mux 选择冲突
同一时间多个 sender 想给同一 receiver
同一 sender 同一 cycle只能发一个 word
exchange schedule 被拉长
compute phase load imbalance 导致 sync 等待
```

这点非常重要：**IPU 的通信瓶颈是 schedule packing 问题，而不是 router queueing 问题。**

---

# 8. 我会给你的理解打 90 分

你的原话：

> 标准 noc 有 router，router 根据 header flit 信息选择输出；Graphcore 没有复杂逻辑，节点间交叉由很多 mux 组成，编译器决定某 cycle 某 mux 的选择信号，从而提前知道数据去哪里。

我会改成更精确版本：

> **传统 packet NoC 由 router 在运行时根据 packet/header 做路由、仲裁和转发；Graphcore IPU exchange fabric 则由 Poplar 编译器在 BSP sync 之后按精确 cycle 静态调度发送、接收和 receiver mux select。数据不依赖 header 动态寻路，而是依赖“时间 + select state”被送到目标 tile。因此它可以省掉 queues、arbiters 和 packet overhead，但把通信调度复杂性转移给编译器。**

这就是最准确的理解。✅
