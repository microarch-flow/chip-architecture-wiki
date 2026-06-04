> 我们继续讨论下一个问题：BSP 是优势还是限制

是的，我们继续这个问题。我的结论先放前面：

> **BSP 对 Graphcore IPU 来说既是优势，也是限制；它的优势来自“确定性、静态调度、低硬件复杂度”，限制来自“全局 barrier、负载不均衡、动态性差、通信/计算 overlap 受限”。**

换句话说，BSP 不是简单的好或坏，它是 Graphcore 这套架构的**核心设计契约**：硬件用它换来了可预测、低开销的 exchange；软件则必须接受同步粒度和调度复杂度。🔥

---

# 1. BSP 在 IPU 中到底是什么？

Graphcore 文档里的 BSP 可以抽象成：

```text
repeat {
  Sync;
  Exchange;
  Compute;
}
```

HC2021 明确写到，tile processor 在需要交换数据前可以异步执行；BSP 模型就是重复执行 **Sync → Exchange → Compute**，每个 tile 在一个 compute phase 中执行一串 atomic codelets；片上硬件全局同步约 150 cycles，跨芯片约 15ns/hop。

Programmer’s Guide 中也说，每个 step 包含：

```text
local tile compute
global cross-tile synchronisation
data exchange
```

并且整个 IPU 可以被看成串行执行一系列 step，每个 step 内部所有 tile 并行。

所以 IPU 不是所有 tile 每条指令 lock-step，而是：

```text
一个 compute phase 内：
  tile 各自跑自己的 codelet list

到 exchange 前：
  全部 tile sync

exchange phase：
  按编译器安排搬数据

然后进入下一轮 compute
```

---

# 2. BSP 的第一大优势：让通信可以静态调度

这是 BSP 最大的价值。

我们刚才讨论 exchange fabric 时说过，Graphcore 的 exchange 不是传统 packet NoC，而是编译器按 cycle 精确调度 transmit、receive、select。

如果没有 BSP，这件事很难成立。因为你不知道：

```text
tile A 什么时候算完
tile B 什么时候准备好接收
tile C 是否还在执行上一段 compute
```

BSP 通过全局同步把所有 tile 拉到同一个时间基准：

```text
所有 tile 到达 sync
  ↓
从同一个 exchange_cycle = 0 开始
  ↓
编译器可以精确安排第几拍谁发、谁收
```

这就是为什么 Graphcore 可以做到：

```text
no queues
no arbiters
no packet overhead
```

所以，**BSP 是 stateless exchange 的前提**。

没有 BSP，exchange fabric 就很可能不得不退化成传统 NoC：

```text
需要 packet header
需要 buffer
需要 arbitration
需要 flow control
需要处理运行时拥塞
```

这会显著增加硬件面积、功耗和不可预测性。

---

# 3. 第二大优势：降低并发错误和硬件复杂度

HC2021 里有一句很关键的话：**BSP eliminates concurrency hazards**。Graphcore 把这列为“hardware helping software”的机制之一。

这句话可以这么理解：

传统异步并行系统里，tile 之间可能出现很多复杂问题：

```text
producer 还没写完，consumer 已经读了
consumer 还没准备好，producer 已经发了
多个 sender 同时竞争一个 receiver
buffer 满导致反压
消息乱序
死锁 / 活锁
```

BSP 把这些问题阶段化：

```text
Compute phase:
  只访问本地数据，不跨 tile 通信

Sync:
  确保上一轮 compute 都结束

Exchange:
  只做数据搬运，不做普通 compute

Next Compute:
  使用 exchange 后的新数据
```

这带来一个很干净的语义：

```text
本轮 compute 看到的是上一轮 exchange 后的稳定状态
本轮 exchange 搬的是本轮 compute 已经完成后的结果
```

所以程序正确性更容易推理，编译器也更容易做数据依赖分析。

---

# 4. 第三大优势：性能可预测，方便 profile 和 load balance

IPU 的执行时间可以粗略写成：

```text
T_step =
  max(T_compute_tile_i)
  + T_sync
  + T_exchange_schedule
```

这比传统动态 NoC 的性能模型简单很多。传统 NoC 可能要考虑：

```text
packet contention
router queueing
VC 分配
拥塞传播
动态 backpressure
cache miss
memory ordering
```

IPU 的 BSP 模型下，主要看：

```text
每个 tile 的 compute 是否均衡
exchange schedule 是否足够紧凑
sync 粒度是否合理
```

这也是为什么 PopVision 可以比较清楚地展示 sync / exchange / compute 的时间分布。Programmer’s Guide 中提到，虽然最后 lowering 成 per-tile program，但由于 tile 间同步，仍然可以从全局视角看执行，PopVision Graph Analyser 就是这种视角。

这对架构探索非常有用，因为你可以把性能拆成：

```text
compute imbalance
communication schedule length
barrier overhead
memory pressure
```

而不是陷在“到底哪个 router 哪个 cycle 拥塞了”的细节里。

---

# 5. 第四大优势：配合分布式 SRAM，形成非常清晰的数据流模型

IPU 的每个 tile 只能 load/store 自己的 local memory，Streaming Memory 和其他 tile memory 都要通过显式搬运。Programmer’s Guide 里说，IPU 的 SRAM 被拆成多个 disjoint memory units，每个 tile 一个地址空间；tile 的 load/store 只能访问本 tile local memory，代码也存储在 local tile memory。

所以 BSP 和分布式 SRAM 天然配合：

```text
Compute phase:
  local SRAM → local compute → local SRAM

Exchange phase:
  tile SRAM ↔ tile SRAM
  Streaming Memory ↔ tile SRAM
```

这相当于把数据流分成两个显式事件：

```text
计算事件 compute
搬运事件 exchange
```

这和你之前提出的 event IR 非常契合。IPU 实际上就是把执行模型强行规整成：

```text
compute event
sync event
communication event
compute event
...
```

---

# 6. 但 BSP 的第一大限制：全局 barrier 会放大负载不均衡

BSP 最大的问题就是 barrier。

一个 step 内，如果有 1472 个 tile：

```text
Tile 0 compute: 100 cycles
Tile 1 compute: 110 cycles
...
Tile 900 compute: 300 cycles
```

那么这个 compute phase 的时间不是平均值，而是：

```text
max = 300 cycles
```

其他 tile 算完后只能等。

所以 IPU 编译器必须非常努力地做：

```text
vertex 切分
tile mapping
codelet cost estimation
load balance
```

否则会出现：

```text
大量 tile 等少数慢 tile
```

这就是 BSP 的经典限制。

Graphcore 的 N+1 barrel threading 设计里也提到，隐藏 worker pipeline 可以让 vertex execution 更容易被编译器预测，从而方便 load balance。这个设计背后其实就是在服务 BSP：如果每个 vertex 时间可预测，就更容易避免 barrier 前等待。

---

# 7. 第二大限制：同步粒度太细时，sync/exchange overhead 会吞掉收益

如果一个 compute phase 很大，比如几万 cycle，那么：

```text
150 cycles sync
```

可能不算什么。

但如果 compute set 很碎：

```text
compute 200 cycles
sync 150 cycles
exchange 100 cycles
```

那同步开销就非常明显了。

所以 BSP 架构特别怕：

```text
细碎 op
频繁小通信
短 compute phase
不规则依赖导致 frequent barriers
```

这对现代模型有现实影响。比如 transformer 中很多算子可以 fuse 成大 compute phase，但如果编译器没有融合好，或者 workload 本身存在细粒度依赖，那么 BSP step 会变多，barrier 成本上升。

---

# 8. 第三大限制：通信和计算 overlap 受限

传统异步 NoC 或 GPU 系统里，理想情况下可以：

```text
tile/core A compute
tile/core B communication
DMA prefetch
compute 与 communication overlap
```

IPU 的 BSP 基本语义是阶段化：

```text
Compute phase
Exchange phase
Compute phase
```

这会让通信/计算 overlap 没那么自然。

当然，Graphcore 文档里也提到一种有限形式：可以把 IPU 内 tile 分成两组，一组专门做 I/O，一组做 compute，从而实现某种 I/O overlap。Programmer’s Guide 说，IPU 可分为 I/O tile group 和 compute tile group，I/O tile group 执行 data stream 或 Streaming Memory copy，compute tile group 执行一般 compute。

但这不是任意细粒度的通信计算重叠，而是比较受控的 task parallelism。

所以 BSP 的代价是：

```text
通信更确定
但 overlap 自由度更低
```

---

# 9. 第四大限制：动态 workload 不友好

BSP + 静态 exchange 对编译期可知的计算图非常友好：

```text
CNN
BERT fixed sequence
静态 shape transformer
训练图
规则 tensor program
```

但对以下场景挑战较大：

```text
动态 shape
运行时决定路由的 MoE
稀疏图结构随输入变化
自回归 decode 中变长请求
复杂 control flow
在线 serving 的 continuous batching
```

原因是：

```text
通信 pattern 越动态
越难提前排 exchange schedule

控制流越分歧
越难保证所有 tile 在合适时间进入 sync

shape 越动态
越容易触发重新编译或保守 schedule
```

这不代表 IPU 不能做动态控制流，而是说它的最高效模式依赖编译器提前知道足够多的信息。

这也是为什么 Graphcore 当年强调 AI 未来会稀疏、图结构化，但真正进入 LLM serving 时代后，GPU/HBM/动态 batching 生态反而更强。这个我们后面专门讨论 LLM 适配时可以展开。

---

# 10. 第五大限制：BSP 会带来供电瞬态问题

这个点很容易被忽略，但 HC2021 第 6 页 Graphcore 自己提到了：bulk synchrony 让 Vdd margin / supply transient 的 tuning 更难。

原因很直观：

```text
大量 tile 同步进入 compute
大量 tile 同步进入 exchange
全芯片活动模式相对一致
电流变化更同步
```

这可能导致：

```text
瞬态电流更尖
IR drop / Ldi/dt 压力更强
电源裕量难压低
```

传统异步系统里，不同单元活动更错开，电流峰值可能被自然摊平。BSP 在时间上更整齐，确定性更强，但电源侧可能更难受。

这是一个非常典型的架构副作用：**执行确定性会改变功耗时序分布**。

---

# 11. BSP 对不同 workload 的影响

可以粗略分类：

| Workload 类型                    |      BSP 友好程度 | 原因                              |
| ------------------------------ | ------------: | ------------------------------- |
| 静态 CNN                         |             高 | 图规则、通信模式固定、易 fusion             |
| BERT-like fixed shape training |             高 | shape 固定、batch 规则、全图编译有效        |
| 大 batch dense GEMM             |           中到高 | compute phase 足够大，sync 被摊薄      |
| 小 batch inference              |             中 | compute phase 可能变短，barrier 比例上升 |
| LLM prefill                    |             中 | GEMM 大，但权重/activation/KV 布局复杂   |
| LLM decode                     |          偏低到中 | 小 batch GEMV、KV cache、动态请求、长上下文 |
| MoE 动态路由                       |            偏低 | runtime routing 和负载不均衡明显        |
| 图神经网络 / irregular sparse       | 理论上有潜力，但依赖编译器 | 数据结构不规则，若 pattern 静态则可调度，若动态则困难 |

所以 BSP 最适合：

```text
静态图
规则通信
compute phase 足够粗
tile workload 可均衡
```

最不适合：

```text
运行时动态通信
频繁细粒度同步
强负载不均衡
在线 serving 中请求形态多变
```

---

# 12. 和 GPU / Tenstorrent / Groq 对比

## 12.1 GPU

GPU 更像：

```text
大量动态线程
cache/HBM
warp scheduling
kernel-level synchronization
runtime memory system
```

GPU 的优势是灵活，能处理动态 workload，生态强；代价是性能可预测性较差，cache/memory contention 复杂。

## 12.2 Graphcore IPU

IPU 是：

```text
tile-local compute
global BSP sync
static exchange
compiler-managed memory/data movement
```

优势是确定性和低 transport overhead；代价是 barrier 和编译器复杂度。

## 12.3 Tenstorrent

Tenstorrent 早期 dataflow 更像：

```text
tile 间 streaming
NoC packet/routing
producer-consumer 更异步
```

它不像 IPU 那样强 BSP 全局阶段化；这可能让 overlap 更灵活，但也增加 runtime NoC/队列/背压/死锁分析复杂度。

## 12.4 Groq

Groq 也很 deterministic，但更像：

```text
全局静态时序化的数据流机器
编译器安排每个 cycle 的数据流
```

Graphcore 和 Groq 都把复杂性转移给编译器，但 Graphcore 的单位是 many-tile BSP + exchange phase，Groq 更偏严格流式 scheduled execution。

---

# 13. 对你架构探索方法的启发

BSP 可以作为你 DSL / IR 里的一个明确执行语义选项，而不是默认假设。

比如：

```yaml
execution_model:
  type: BSP
  phases:
    - sync
    - exchange
    - compute
  communication:
    allowed_only_in_exchange: true
  compute:
    local_memory_only: true
  step_time_model:
    compute: max(tile_compute_cycles)
    sync: fixed_or_topology_dependent
    exchange: scheduled_communication_cycles
```

同时你要在探索里显式评估：

```text
1. compute imbalance:
   max(tile_cycles) / avg(tile_cycles)

2. barrier ratio:
   sync_cycles / total_step_cycles

3. exchange ratio:
   exchange_cycles / total_step_cycles

4. schedule packing efficiency:
   useful_words / theoretical_exchange_capacity

5. phase granularity:
   average compute cycles per BSP step

6. dynamic shape penalty:
   是否需要多版本编译 / 保守 schedule / runtime fallback
```

这会比简单估算 TOPS 和带宽更接近真实架构行为。

---

# 14. 最关键的判断

我会这样评价 BSP：

## BSP 是优势，当你的目标是：

```text
确定性
低通信硬件复杂度
静态可调度
低 packet/queue/arbiter overhead
可 profile
规则图程序
大 compute phase
```

## BSP 是限制，当你的 workload 需要：

```text
细粒度异步通信
强 compute/communication overlap
运行时动态路由
动态 shape
低延迟小 batch serving
不均衡 sparse/MoE
```

所以对 Graphcore 来说，BSP 不是一个附属选择，而是整个 IPU 成立的前提：

```text
BSP
  ↓
静态 exchange 成立
  ↓
无 queue / arbiter / packet overhead 成立
  ↓
低数据搬运能耗成立
  ↓
大规模 distributed SRAM + many tile 模型成立
```

但反过来：

```text
BSP
  ↓
barrier 等待
  ↓
load balance 压力
  ↓
动态 workload 适配困难
  ↓
全图编译和调度复杂
```

---

# 15. 我的最终观点

**BSP 是 Graphcore IPU 的核心优势，也是它的核心边界。**

它让 IPU 拥有一种非常漂亮的架构闭环：

```text
分布式 SRAM
  + local compute
  + BSP sync
  + static exchange
  + Poplar 全图编译
```

但这个闭环要求 workload 足够静态、可切分、可均衡、通信可预测。
一旦进入高度动态的 LLM serving，尤其是 decode、continuous batching、长上下文 KV cache、MoE 动态路由，BSP 的限制就会越来越明显。

**一句话总结：BSP 对 IPU 是“用全局阶段化换确定性和低通信开销”的设计选择；它不是错，但它强烈偏好静态、规则、可均衡的大粒度图计算。**
