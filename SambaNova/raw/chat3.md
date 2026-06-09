> 与RDN直接互联的除了RDN外，local连的是PMU还是PCU？

不是“local 只连 PMU 或只连 PCU”。更准确地说：

> **RDN switch 的 local 端口连接的是本地 dataflow unit，而这个 unit 可以是 PCU、PMU，也可以是 AGCU。**

公开资料里有两个关键证据：

1. SN40L paper 明确说：**RDN 是促进 PCU、PMU、AGCU 之间通信的片上可编程互联**，并且 AGCU 在 tile 侧也像一个 dataflow core 一样暴露 RDN vector/scalar/control ports。
2. SN40L 的 tile 图里，`S` 是 mesh switch，旁边分布着 `PMU PCU` 对，以及边缘的 `AGCU`；图注也说这个 tile block diagram 包含 PCU、PMU、RDN switches、AGCUs。

所以心智模型应该从：

```text
Switch local -> PMU ?
Switch local -> PCU ?
```

改成：

```text
Switch local -> Unit RDN port

Unit 可以是：
  PCU
  PMU
  AGCU
```

---

# 1. 一个 switch 的 local 端口不是固定语义

在普通 NoC 里，我们常说：

```text
router local port -> tile
```

但 SambaNova 的 tile 不是一个 CPU core tile，而是一片 **PCU / PMU / AGCU / Switch 的 sea-of-units**。

所以更像：

```text
        N
        |
W --- [ S ] --- E
        |
        S
        |
      local
        |
      PCU or PMU or AGCU
```

也可以是：

```text
[ S ] local -> PCU
[ S ] local -> PMU
[ S ] local -> AGCU
```

SN10 的图也支持这个理解：它画的是一片 programmable interconnect，里面有多个 `S Switch`，并散布 `PMU`、`PCU`、`AG`、`Coalescing Unit`，而不是每个 switch 后面固定挂一种单元。

---

# 2. 更精确的理解：PCU/PMU 自己也有 RDN ports

从 RDN 描述看，scalar/vector packet 从 switch 进入 unit 后会落到 unit 的 input FIFO，然后从 output FIFO 出来；AGCU 也明确在 tile 侧暴露 vector/scalar/control ports。

所以 PCU/PMU 与 RDN 的连接应理解为：

```text
PCU:
  vector_in_fifo  <--- RDN vector fabric
  scalar_in_fifo  <--- RDN scalar fabric
  control_in      <--- RDN control fabric

  vector_out_fifo ---> RDN vector fabric
  scalar_out_fifo ---> RDN scalar fabric
  control_out     ---> RDN control fabric
```

```text
PMU:
  vector_in_fifo  <--- RDN vector fabric
  scalar_in_fifo  <--- RDN scalar fabric
  control_in      <--- RDN control fabric

  vector_out_fifo ---> RDN vector fabric
  scalar_out_fifo ---> RDN scalar fabric
  control_out     ---> RDN control fabric
```

```text
AGCU:
  vector/scalar/control ports <-> RDN
  TLN side <-> HBM / DDR / host / peer RDU
```

也就是说，**local 端口连接的不是“RDN 之外的另一个 RDN”，而是某个 dataflow functional unit 的 RDN-facing port**。

---

# 3. PMU 和 PCU 是不是一对一绑定？

**看公开图，SN40L 中 PCU 和 PMU 经常成对相邻出现，但不能理解成固定私有绑定。**

SN40L tile 图里很多地方是：

```text
PMU PCU
  S
```

或者：

```text
PMU PCU
S   S
```

这说明物理布局上 PMU 和 PCU 可能按 pair / cluster 靠近放置，利于短线连接和常见 producer-consumer pattern。但 RDN 的作用正是让任意 PCU/PMU/AGCU 之间通过 switch mesh 通信，而不是让某个 PCU 只能访问旁边那个 PMU。

所以应区分两层：

```text
物理邻近：
  一个 PCU 旁边常有一个 PMU，local switch 可能靠近这两个 unit。

逻辑连接：
  compiler 可通过 RDN 把任意 PCU/PMU/AGCU 串成 dataflow graph。
```

---

# 4. 更合理的电路抽象

我会把一个局部 region 抽象成这样：

```text
          S -------- S
          |          |
        PMU        PCU
          |          |
          S -------- S
```

或者更简化：

```text
        +---------+
        |  Mesh   |
        | Switch  |
        +---------+
        /    |    \
     PCU    PMU   AGCU
```

但第二张图只是逻辑抽象，不代表一个 switch 必然同时直接挂三种 unit。真实实现更可能是：

```text
每个 switch 有若干 local/unit ports；
某些 local port 接 PCU；
某些 local port 接 PMU；
边缘或特定位置的 switch 接 AGCU；
PCU/PMU 也可能靠近多个 switch，以获得更多注入/弹出带宽。
```

公开资料没有给出端口数、每个 switch 到几个 unit、是否一个 unit 多端口连多个 switch 这些 RTL 级细节，所以这里不能下死结论。

---

# 5. 你可以这样记

最准确的表述是：

```text
RDN switch 的 local side 连接的是 RDU 内部 dataflow unit 的 network interface。

这个 unit 可以是：
  - PCU：计算 stage
  - PMU：存储 / layout transform / stage buffer
  - AGCU：off-chip / TLN portal

PCU 和 PMU 在物理布局上常相邻，但不是逻辑上私有绑定。
```

**一句话总结：RDN 的 local 不是固定连 PMU 或 PCU，而是连接本地 dataflow unit；在 SN40L/SN10 里这些 unit 包括 PCU、PMU 和 AGCU，RDN mesh 负责把它们按 compiler 的 graph mapping 串起来。**
