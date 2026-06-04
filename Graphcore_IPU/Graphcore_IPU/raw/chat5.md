> Graphcore 这套BSP方案里面，在Exchange后是否还需要Sync来确保所有数据都交换完成，不然可能会出现数据没有交换完成计算就开始了

需要，但这里要非常精确地区分两种“Sync”：

> **从语义上看，Exchange 结束后必须有一个“exchange completion boundary”，确保所有本轮数据搬运已经完成，下一轮 Compute 才能使用新数据。
> 但从 Graphcore 文档的表述看，它不一定被单独命名为一个额外的 Sync phase，而是被包含在 BSP step 的阶段边界 / exchange schedule 结束条件里。**

也就是说，你担心的现象：

```text
数据还没交换完
Compute 就开始读新数据
```

**在正确的 IPU BSP 执行中不应该发生**。否则整个编程模型就不成立了。

---

# 1. 官方描述里阶段顺序有一点“表述差异”

Programmer’s Guide 里把一个 step 写成：

```text
local tile compute
global cross-tile synchronisation
data exchange
```

也就是：

```text
Compute → Sync → Exchange
```

并且文档说，所有 tile 完成 compute 后进入同步；同步完成后，IPU 进入 exchange phase，在 tile 间复制数据。

但紧接着它又说：

```text
After the exchange phase, the process repeats:
the tiles move into a new compute phase,
performing computation using their local data
and the new data received during the exchange.
```

也就是：

```text
Exchange 完成后，才进入新的 Compute
```



HC2021 的写法则是：

```text
repeat { Sync; Exchange; Compute }
```



这两个写法其实是一回事，只是从不同起点描述循环：

```text
Compute → Sync → Exchange → Compute → Sync → Exchange → ...
```

或者：

```text
Sync → Exchange → Compute → Sync → Exchange → Compute → ...
```

---

# 2. 是否需要“Exchange 后再 Sync”？

如果你说的是 **全局 barrier 意义上的 Sync**，那从文档看，IPU 的显式全局同步点主要放在 **Exchange 之前**：

```text
Compute 完成
  ↓
所有 tile 同步
  ↓
Exchange 开始
```

这个 Sync 的作用是确保：

```text
1. 所有 producer tile 都已经完成本地计算
2. 要发送的数据已经稳定在 local memory
3. 所有 tile 都从同一个时间基准进入 exchange
```

这对于 Graphcore 的静态调度 exchange 非常关键，因为 Poplar 编译器会从 sync 开始，以精确 cycle 调度 transmit、receive 和 select，并且知道所有 pipeline delay。

但是，**Exchange 后一定还需要一个“完成边界”**。否则 receiver 读到的可能不是新数据。

只不过这个边界不一定是一个像 compute 后那样的“所有 tile 到达 barrier 然后再放行”的独立 phase，而更可能是：

```text
Exchange schedule 有固定长度
硬件/编译器知道最后一个数据何时到达
所有 tile 在 exchange schedule 结束后进入下一 compute
```

也就是说：

```text
Exchange completion is deterministic.
```

---

# 3. 为什么不一定需要额外的动态 Sync？

因为 IPU exchange 不是动态 packet NoC。

传统 NoC 中，如果 packet 动态路由，你确实可能需要确认：

```text
所有 packet 都到了吗？
有没有还堵在 router buffer 里？
有没有还在排队？
有没有下游 backpressure？
```

但 Graphcore exchange 的特点是：

```text
no queues
no arbiters
no packet overheads
addressing by time and select state
compiler schedules transmit / receive / select at precise cycles
```



所以它的完成条件不是“运行时观察网络是否 drain 完”，而是：

```text
编译器已知本次 exchange schedule 的长度
硬件已知 pipeline latency
到达预定结束 cycle 后，本轮 exchange 必然完成
```

这就像一个静态流水线：

```text
cycle 0: 发送第一批数据
cycle K: 发送最后一批数据
cycle K + L: 最后一批数据到达 receiver
cycle K + L + guard: 下一轮 compute 可以开始
```

其中 `L` 是确定的 pipeline delay / fabric latency。

---

# 4. 更准确的执行心智模型

我建议你把 IPU BSP step 理解成下面这样：

```text
Compute phase N
  - 每个 tile 执行本地 codelets
  - 只读写本 tile local memory

Sync boundary
  - 等所有 tile 完成 compute N
  - 建立 exchange 的共同时间起点

Exchange phase N
  - 按编译器静态 schedule 搬运数据
  - sender / receiver / mux select 都按 cycle 执行
  - 最后一笔数据到达时间可预测

Exchange completion boundary
  - 本轮 exchange schedule drain 完成
  - 新数据已经写入目标 tile local memory

Compute phase N+1
  - 使用 exchange N 后的新数据
```

所以如果你问：

> Exchange 后是否需要确保所有数据完成？

答案是：**必须确保。**

如果你问：

> 这个确保是否表现为一个额外的显式全局 Sync phase？

答案是：**官方资料没有把它描述成一个单独的额外 Sync；它更像是 exchange phase 的 deterministic completion / phase boundary。**

---

# 5. 为什么 HC2021 写成 repeat { Sync; Exchange; Compute }？

这个写法暗含了：

```text
Sync 是为了开始 Exchange
Exchange 完成后自然进入 Compute
Compute 完成后进入下一轮 Sync
```

也就是：

```text
Sync_k
Exchange_k
Compute_k
Sync_{k+1}
Exchange_{k+1}
Compute_{k+1}
```

如果从 Compute 的角度看，就是：

```text
Compute_k
Sync_{k+1}
Exchange_{k+1}
Compute_{k+1}
```

所以你看到的“Exchange 后没有 Sync”，不是说可以不等数据完成，而是说：

```text
Exchange 本身是一个全局、确定长度、静态调度的阶段
阶段结束即意味着本轮通信完成
```

---

# 6. 和传统 NoC 的区别

传统 packet NoC 里，可能需要某种 completion / fence / acknowledgement，因为：

```text
packet 可能排队
路径可能不同
拥塞可能不同
到达时间可能变化
```

Graphcore exchange 里则是：

```text
发送时间确定
选择信号确定
pipeline delay 确定
没有队列和动态仲裁
```

所以 completion 可以由时间表保证，而不是由 ack 协议保证。

这也是它能做到低 overhead 的关键之一。

---

# 7. 你的问题可以转化为一个架构约束

如果你在自己的 DSL / simulator 里建模 IPU-like BSP，我建议明确写成：

```yaml
phase_order:
  - compute
  - sync_before_exchange
  - exchange
  - exchange_completion_boundary
  - compute

semantics:
  compute_reads:
    visible_data: data_after_previous_exchange
  exchange_starts_only_after:
    all_tiles_compute_done: true
  next_compute_starts_only_after:
    all_exchange_writes_committed: true
```

性能模型上可以写成：

```text
T_step =
  max_i(T_compute_i)
  + T_sync
  + T_exchange_schedule
  + T_exchange_drain_or_guard
```

其中 `T_exchange_drain_or_guard` 可能已经包含在 `T_exchange_schedule` 里，不一定要单独列出来。

---

# 8. 最终回答

你的判断是对的：**如果 Exchange 后不保证所有数据交换完成，下一轮 Compute 就可能读到旧数据或未定义数据，BSP 语义会被破坏。**

但在 Graphcore IPU 里，这个保证很可能不是靠“Exchange 后再来一次动态全局 Sync”实现，而是靠：

```text
1. Exchange 前的 global sync 建立共同起点
2. 编译器精确调度每个 cycle 的 transmit / receive / select
3. fabric pipeline latency 已知
4. exchange phase 有确定结束时间
5. 下一轮 compute 只在 exchange phase 完成后开始
```

**一句话总结：IPU 的 Exchange 后一定有完成语义，但它更像静态调度通信阶段的确定性结束边界，而不是传统 NoC 中靠额外 barrier/ack 来确认所有 packet 已经送达。**
