> 深挖 PCU + PMU：SambaNova 的 core 到底怎么工作？

我们这次把 **PCU + PMU** 当成 SambaNova RDU 的“core”来拆，但先纠正一个心智模型：**SambaNova 的 core 不是 CPU/GPU 那种取指-译码-执行的 core，也不是 TPU 那种固定 systolic array tile；它更像 CGRA 里的一个“可配置计算/存储/搬运 stage”。**

也就是说：

```text
一个 RDU tile 里反复出现：

  PMU  -- data stream -->  PCU  -- data stream -->  PMU
   |                         |                         |
address generation      SIMD / systolic           layout transform
scratchpad              reduction / tail          buffering
```

PCU 负责“算”，PMU 负责“存 + 产生地址 + 重排数据 + 作为流水线 stage buffer”。真正让 SambaNova 成立的不是 PCU 单独多强，而是 **PCU + PMU + RDN switch 被 compiler 组合成 graph-level pipeline**。RDA 白皮书也明确说，RDU 由 tiled array 的 reconfigurable processing and memory units 组成，PCU 执行 innermost-parallel operation，PMU 是 specialized scratchpad，switch fabric 连接 scalar/vector/control 三类网络。

---

# 1. 先建立一个最重要的直觉

## GPU 的 core 视角

GPU 上你一般这样想：

```text
kernel launch
  -> thread blocks
    -> warps
      -> SIMT lanes
        -> load/store through register/shared/L2/HBM
```

GPU 的基本执行单元是 **thread / warp / block**。

---

## SambaNova 的 core 视角

SambaNova 上更应该这样想：

```text
graph segment
  -> spatial pipeline
    -> PMU stage buffer
    -> PCU compute stage
    -> PMU layout transform / buffer
    -> PCU compute stage
    -> ...
```

SambaNova 的基本执行单元不是 thread，而是 **dataflow stage**。

所以 PCU/PMU 的工作方式不是：

> “每个 core 执行一段指令。”

而是：

> “compiler 把一段 graph 的每个 operator / buffer / transform 映射到不同 PCU/PMU 上，然后数据像流一样穿过这些资源。”

这就是 SambaNova 所谓 **spatial programming**：配置 RDU 的物理资源，让数据在芯片 fabric 上并行流动。白皮书里说 SambaFlow 会结合 instruction sequence 和 allocated resource location，给 compute graph 创建一个 workload-specific pipelined accelerator。

---

# 2. PCU：Pattern Compute Unit 到底是什么？

## 2.1 PCU 不是单一形态的 MAC array

SN40L 文档明确说，PCU 的 datapath 分为：

```text
Header -> Body -> Tail
```

其中：

* **Header**：消费输入 dataflow，驱动 body。
* **Body**：可以配置成 output-stationary systolic array，也可以配置成多级 pipelined SIMD core。
* **Tail**：执行特殊 element-wise function，并驱动输出 FIFO。

所以 PCU 的心智模型应该是：

```text
                    +-------------------------+
Vector In FIFO ---> | Header                  |
Scalar In FIFO ---> |                         |
Counters ------->   | Body:                   |
Control ------->    |   mode A: systolic      |
                    |   mode B: SIMD pipeline |
                    |                         |
                    | optional reduction tree |
                    |                         |
                    | Tail: exp/cast/rng/etc. |
                    +-------------------------+
                         |
                         v
                    Vector Out FIFO
                    Scalar Out FIFO
```

它有点像 **可配置向量流水线 + 可配置小型 systolic 计算阵列 + 特殊函数尾部** 的组合。

---

## 2.2 PCU 的两种主要工作模式

### 模式 A：systolic array mode

当执行 GEMM-like operation 时，PCU body 可以配置成 **output-stationary systolic array**。SN40L paper 说 PCU 可以作为 2D systolic array，输入通过 broadcast buffer 以 left-to-right 和 top-to-bottom 方式流入，累加结果通过 tail unit drain 到 output FIFO。

可以这样理解：

```text
A stream ---> [ MAC ][ MAC ][ MAC ] ---> 
              [ MAC ][ MAC ][ MAC ] ---> partial/output
B stream ---> [ MAC ][ MAC ][ MAC ] --->
```

这里的重点是 **output stationary**：

* partial sum 留在阵列内部；
* A/B 以 stream 方式流动；
* 结果最后 drain 出去。

这和 TPU systolic array 的直觉类似，但 SambaNova 的 PCU 不是整颗芯片级的大 systolic array，而是分布在 mesh 中的很多小 compute stage。SN40L 单 socket 有 1040 个 PCU 和 1040 个 PMU。

---

### 模式 B：SIMD pipeline mode

当执行 elementwise、vector op、部分 reduce、activation、normalization 相关算子时，PCU body 可以配置成 **多级 SIMD vector pipeline**。HotChips 2024 资料说 PCU 可以配置成 systolic array 或 M lanes 的 SIMD vector unit，支持 BF16、FP32、INT32、INT8，支持 arithmetic、logical、bitwise operation，并带 cross-lane reduction tree 和 tail stage。

这更像：

```text
lane0: x0 -> ALU -> reg -> ALU -> reg -> tail
lane1: x1 -> ALU -> reg -> ALU -> reg -> tail
lane2: x2 -> ALU -> reg -> ALU -> reg -> tail
...
laneM: xM -> ALU -> reg -> ALU -> reg -> tail
```

SIMD mode 下，PCU 不只是做 `add/mul`。它还可以支持：

* arithmetic operation
* logical / bitwise operation
* lane-wise operation
* optional cross-lane reduction
* tail stage special function

SN10 资料里也有类似描述：PCU 是 SN10 的 compute engine，包含 reconfigurable SIMD datapath，支持 dense/sparse tensor algebra，支持 FP32/BF16/integer formats，有 programmable counters 做 loop iterator，tail unit 加速 exp、sigmoid 等 common functions。

---

## 2.3 Header / counters / control：PCU 为什么不像普通 SIMD？

普通 SIMD/vector unit 主要是“执行 vector instruction”。但 SambaNova PCU 是 dataflow stage，所以它还要处理 **流控制和循环控制**。

PCU 里有：

* vector input FIFO
* scalar input FIFO
* counters
* control inputs / outputs
* vector output FIFO
* scalar output FIFO

其中 counters 用来跟踪 loop iteration，达到 programmed maximum 后生成 control events，表示某个 loop 完成。

这非常关键。它说明 PCU 不是每拍都被一个传统 instruction stream 驱动，而是被配置成一个 **有限状态的数据流计算 stage**：

```text
while counter < bound:
    consume vector/scalar streams
    execute configured pipeline
    emit vector/scalar streams
    maybe emit control token
```

所以 PCU 更接近：

> **可编程硬件 pipeline stage + 本地 loop controller + stream interface。**

---

## 2.4 Tail stage 为什么重要？

Tail stage 支持：

* transcendental functions
* random number generation
* stochastic rounding
* format conversion / casting

SN40L paper 说 tail operation 可以和 body compute fuse and pipeline。

这意味着很多 GPU 上可能拆成多个 kernel 或者走特殊函数单元的操作，在 SambaNova 上可以作为 PCU pipeline 的尾部 stage：

```text
SIMD body:    y = x * scale + bias
Tail stage:   y = exp(y)
Output FIFO:  stream to next PMU/PCU
```

这对 softmax、layernorm、activation、quant/dequant、cast 都很重要。

---

# 3. PMU：Pattern Memory Unit 到底是什么？

PMU 是 SambaNova 最有特色的部分之一。它不能简单理解成 SRAM。

更准确地说：

> **PMU = banked SRAM + read/write address generator + data alignment / permutation crossbar + predicate/banking logic + stream interface。**

HotChips 2024 资料说 SN40L PMU 是 programmer-managed scratchpad，支持 concurrent reads and writes；有 fragmentable address-generation pipeline，每周期可产生 4 个地址；data alignment crossbars 支持 transpose、dilation、downcast；address predication 支持多个 PMU 组合存储一个大 tensor。

---

## 3.1 PMU 的基本结构

可以这样画：

```text
                  scalar stream
                       |
                       v
              +-------------------+
Vector In --> | Write Data Align  |
              +-------------------+
                       |
                       v
              +-------------------+
              | Scratchpad Banks  |
              | bank0 bank1 ...   |
              +-------------------+
                       |
                       v
              +-------------------+
Vector Out <- | Read Data Align   |
              +-------------------+

              +-------------------+
              | Scalar ALU Pipe   |
              | read addr gen     |
              | write addr gen    |
              +-------------------+

              +-------------------+
              | Addr Pred / Bank  |
              +-------------------+
```

它内部至少有几类关键资源：

| 模块                    | 作用                                                     |
| --------------------- | ------------------------------------------------------ |
| Scratchpad banks      | 存 tensor tile、weight、metadata、中间结果                     |
| Scalar ALU pipeline   | 生成 read/write address                                  |
| Read/Write Data Align | 做 layout transform / vector permute / unaligned access |
| Address predication   | 判断这个地址是否属于当前 PMU                                       |
| Bank mapping          | 控制地址映射到哪个 SRAM bank                                    |
| FIFO + control        | 和 RDN / PCU 进行流式连接                                     |

---

## 3.2 PMU 首先是 stage buffer

在 spatially fused kernel 里，PMU 的第一职责是 **stage buffer**。

比如一个 graph：

```text
GEMM0 -> Mul -> Transpose -> GEMM1
```

SambaNova 不希望：

```text
GEMM0 output -> HBM
HBM -> Mul
Mul output -> HBM
HBM -> Transpose
Transpose output -> HBM
HBM -> GEMM1
```

而是希望：

```text
PMU(W0/I0) -> PCU(GEMM0) -> PMU(T0)
PMU(T0)    -> PCU(Mul)   -> PMU(T1)
PMU(T1)    -> PMU/align transpose -> PCU(GEMM1)
```

SN40L paper 说，PMU 用于存储 on-chip tensors，包括 inputs、parameters、metadata、intermediate results；论文 Figure 4 中的蓝色 blocks 都对应 PMU。

所以 PMU 是 graph pipeline 里的 **中间结果缓冲器**，不是某个线程块私有的 shared memory。

---

## 3.3 PMU 其次是 address-generation engine

PMU 的 Scalar ALU Pipeline 很关键。SN40L paper 说 PMU 有多个 integer ALU stages，可配置为并发生成 read/write addresses；这些 ALU 支持 bitfield extraction、shift-and-set 等常用于地址计算的复杂指令，从而用更少 ALU stage、更低 latency 生成复杂地址。

这意味着 PMU 可以硬件化很多复杂 tensor access pattern：

```text
read_addr  = base + f(i, j, k, stride, tile_id, permutation)
write_addr = base + g(i, j, k, stride, layout, bank_map)
```

普通 GPU 中，这类地址计算通常由每个线程执行 load/store 地址表达式完成；SambaNova 则把它变成 **PMU 内部的配置化 address pipeline**。

这对以下场景很重要：

* transpose
* gather/scatter-ish pattern
* dilation
* strided access
* blocked layout
* tensor tiling
* multi-PMU sharding
* sparse / graph-like access pattern

SN10 资料也说 PMU 有 Address ALUs，用于 high-throughput address calculation for arbitrarily complex accesses。

---

## 3.4 PMU 的 read/write address pipeline 可以切分

这一点很有架构味道。SN40L paper 提到，spatially fused kernel 中的 stage buffer 需要 concurrent read/write，而且读写可能有不同 access pattern；有些情况下复杂写会带来简单读，或者反过来。PMU 的 scalar ALU pipeline 可以被软件切分成独立的 read 和 write address generation pipelines，并由软件配置给读/写各分配多少 stages。

这说明 PMU 不是固定的：

```text
1 read AGU + 1 write AGU
```

而更像：

```text
N-stage scalar ALU pipeline
  -> compiler decides:
       read_addr_gen uses r stages
       write_addr_gen uses w stages
       r + w <= N
```

这个设计非常适合 dataflow compiler，因为 compiler 可以根据这个 tensor 的读写模式做 trade-off：

```text
方案 A:
  写入时复杂 layout transform
  读取时连续读，喂给 PCU

方案 B:
  写入时简单顺序写
  读取时复杂 address pattern，完成 transpose/permute
```

这点和你之前关注的 **架构 DSL / event IR** 很相关：PMU 的 address pipeline allocation 应该是一个真实硬件资源约束，而不是抽象 bandwidth 数字。

---

## 3.5 PMU 还是 layout transformation unit

PMU 的 Data Alignment Unit 支持：

* transpose
* cross-lane vector permute
* vector-unaligned access
* LUT
* data format conversion
* data layout conversion

SN40L paper 还特别说，transpose tensor 会以一种 diagonally striped format 写入 scratchpad banks，使得同一个 tensor 可以 full bandwidth 读出 regular format 或 transposed format；这使 Figure 3 里的 transpose 可以作为 buffer 间 read-write access pattern optimization 来实现。

这个点非常重要。

在 GPU 上，transpose 往往是一个显式 kernel 或者 fusion 难点：

```text
SM A writes output
transpose requires cross-SM exchange
often materialize to shared cache / L2 / HBM
```

而 SambaNova 的目标是：

```text
写 PMU 时就按特殊 bank layout 放
读 PMU 时直接读成转置后的 stream
```

也就是说，transpose 不一定是一个“计算算子”，而可以变成 **PMU 的地址/布局行为**。

这是 RDA 和传统 NPU/GPU 的本质差异之一。

---

# 4. PCU + PMU 合起来怎么执行一个算子？

下面用几个典型算子拆。

---

## 4.1 Elementwise：Map / Zip

比如：

```text
y = x * scale + bias
```

在 SambaNova 里可能是：

```text
PMU_x:
  stream x vector

PMU_scale / scalar stream:
  stream scale / bias

PCU:
  SIMD pipeline:
    mul -> add
  tail:
    optional cast / stochastic rounding

PMU_y:
  receive y stream
```

心智模型：

```text
PMU(x) ---> PCU SIMD lanes ---> PMU(y)
              |
          scalar scale/bias
```

这里 PCU 主要用 SIMD mode。

如果有多维 tensor，PCU counters 管 loop，PMU 生成地址，RDN 负责把 stream 接起来。

---

## 4.2 Reduce：sum / mean / norm

比如 LayerNorm 里的 mean：

```text
mean = reduce_sum(x) / N
```

PCU 可以用两层 reduction：

```text
lane-wise SIMD reduction
    +
optional cross-lane reduction tree
```

SN40L 资料明确说 PCU 有 cross-lane reduction tree，用于沿 vectorized dimension reduce。

所以流程可能是：

```text
PMU(x) -> PCU SIMD lanes -> cross-lane reduction -> scalar/vector output
```

如果 reduce 范围大于一个 PCU 能覆盖的维度，就需要多个 PCU 做 partial reduction，再通过 RDN many-to-one 或 PMU stage buffer 合并。

所以对 LayerNorm / Softmax 的 dataflow，可以拆成：

```text
PMU input
  -> PCU reduce max / mean
  -> scalar stream
  -> PCU map/sub/exp
  -> PCU reduce sum
  -> PCU divide
  -> PMU output
```

这解释了为什么 SambaNova 喜欢展示 Map / Zip / Reduce，因为这些 pattern 正好对应 PCU/PMU/RDN 的组合。

---

## 4.3 GEMM：systolic mode + 多 PCU 并行

GEMM 可以映射到 PCU systolic mode：

```text
PMU(A tile) ---> 
                 PCU systolic array ---> PMU(C tile)
PMU(B tile) ---> 
```

如果矩阵更大，可以多个 PCU 并行：

```text
A/B tiles sharded across PMUs
       |
       v
PCU0 PCU1 PCU2 PCU3 ...
       |
       v
partial/output tensors stored in PMUs
```

SN40L paper 说 operation 可以跨多个 PCU 做 data parallel、tensor parallel、pipeline parallel；data parallel 是 partition inputs/outputs，tensor parallel 是 fork into data-parallel streams then join，pipeline parallel 是 chaining multiple PCUs to fuse operations and increase operational intensity。

所以 SambaNova 的 GEMM 不是“一个巨大的中心阵列吃掉所有矩阵”，而是：

> **很多小 PCU systolic stage + PMU feed/store + RDN stream/fork/join。**

---

## 4.4 Transpose：PMU 做，不一定 PCU 做

这是最能体现 PMU 价值的算子。

传统视角：

```text
Transpose 是一个 operator，需要 compute core 执行。
```

SambaNova 视角：

```text
Transpose 可以是 PMU 的 read/write layout transform。
```

可能过程：

```text
PCU(GEMM0) output stream
     |
     v
PMU_T0 write data align:
  写入 diagonally striped / banked layout
     |
     v
PMU_T0 read data align:
  以 transposed order full bandwidth 读出
     |
     v
PCU(GEMM1)
```

所以 transpose 没有必要占用大量 PCU compute；它更多占用：

* PMU bank bandwidth
* data align crossbar
* address generation pipeline
* RDN bandwidth

这也是为什么 SambaNova paper 用 Monarch FFT 里的 transpose 来论证 GPU fusion 的困难，以及 RDA spatial fusion 的优势。论文说复杂 access pattern 会严重限制 GPU fusion，而 fully spatially fused 可以显著提高 operational intensity。

---

# 5. 一个完整 graph segment 的执行例子

用论文里的简化图：

```text
W0, I0 -> Gemm0 -> Mul -> Transpose -> Gemm1 -> Out
```

SambaNova 可能会映射成：

```text
PMU_W0  ----\
             \
PMU_I0  -----> PCU_Gemm0 ----> PMU_T0 ----> PCU_Mul ----> PMU_T1
                                                        |
                                                        v
                                                  PMU layout transform
                                                        |
PMU_W1  -------------------------------------------\     |
                                                     \    v
                                                      PCU_Gemm1 ---> PMU_Out
```

更细一点：

```text
1. PMU_W0 / PMU_I0
   - 存 W0 / I0 tile
   - 生成读地址
   - vector stream 喂 PCU

2. PCU_Gemm0
   - 配置成 systolic mode
   - 输出 partial / tile result

3. PMU_T0
   - stage buffer
   - concurrent write from Gemm0 and read to Mul
   - 可能调整 layout

4. PCU_Mul
   - 配置成 SIMD mode
   - scalar/vector stream 输入 scale
   - 输出到下一 PMU

5. PMU_T1
   - 写入时或读取时完成 transpose-friendly layout
   - data align crossbar 做 reorder

6. PCU_Gemm1
   - 再配置成 systolic mode
   - 输出最终结果
```

这个执行方式的核心收益是：

```text
中间 tensor 不反复落 HBM
operator 之间不是 kernel boundary
transpose / layout transform 融入 PMU
多个 operator 成为 coarse-grained pipeline
```

---

# 6. PCU 和 PMU 的本质分工

我建议你记成下面这张表：

| 组件             | 本质职责                                        | 更像什么                                            | 不是简单的什么                     |
| -------------- | ------------------------------------------- | ----------------------------------------------- | --------------------------- |
| **PCU**        | 执行 stream 上的 compute stage                  | configurable SIMD/systolic pipeline             | 不是 GPU SM，也不是固定 tensor core |
| **PMU**        | 存 tensor tile、生成地址、重排 layout、做 stage buffer | programmable scratchpad + address/layout engine | 不是普通 SRAM bank              |
| **RDN switch** | 连接 PCU/PMU stream                           | dataflow fabric                                 | 不是只服务 load/store 的 NoC      |
| **AGCU**       | 连接 off-chip memory / host / peer RDU        | DMA + AGU + coalescer + graph portal            | 不是普通内存控制器                   |

PCU 和 PMU 的关系可以总结为：

```text
PCU 负责把 stream 变成新 stream；
PMU 负责让 stream 以正确的顺序、布局、带宽出现。
```

这句话很关键。

---

# 7. 和 GPU SM / Tensor Core / TPU array 的差异

## 7.1 对比 GPU SM

GPU SM 的核心抽象是：

```text
threads execute instructions
```

SambaNova PCU/PMU 的核心抽象是：

```text
streams flow through configured hardware stages
```

GPU 做 fusion 通常受限于：

* kernel launch 边界
* thread block grid 固定
* SM 间数据交换需要 L2/HBM
* shared memory 只在 block 内
* transpose/shuffle 跨 block 很难优雅 fuse

SambaNova 则把：

* compute stage 放 PCU
* intermediate buffer 放 PMU
* layout transform 放 PMU
* inter-stage communication 放 RDN

所以它更适合 **graph-level fusion**。

---

## 7.2 对比 GPU Tensor Core

Tensor Core 是矩阵乘专用单元：

```text
warp -> mma instruction -> tensor core
```

PCU 的 systolic mode 可以做 GEMM，但 PCU 不是只有 GEMM。它还能切到 SIMD pipeline、reduction、tail special function。因此 PCU 更像：

```text
Tensor Core + Vector ALU pipeline + reduction + activation/cast tail
```

但这不是说 PCU 单点比 Tensor Core 更强，而是说它服务的抽象层级不同：Tensor Core 服务一个 kernel 内的 matrix instruction，PCU 服务 spatial dataflow pipeline 的一个 compute stage。

---

## 7.3 对比 TPU systolic array

TPU 更像：

```text
large systolic array + compiler-managed memory hierarchy
```

SambaNova 更像：

```text
many small configurable compute/memory stages + mesh dataflow fabric
```

TPU 的强项是大 GEMM 的高效性；SambaNova 的强项是 **GEMM + elementwise + reduce + transpose + layout transform 组成的一整段 graph**。

---

# 8. 对你的架构探索 DSL / IR 的启发

SambaNova PCU/PMU 对你的架构 DSL 很有参考价值，因为它说明一个 AI accelerator 的 tile 不应该只描述：

```yaml
compute_tflops:
sram_capacity:
noc_bw:
```

更应该描述：

```yaml
tile:
  compute_units:
    modes:
      - systolic
      - simd_pipeline
    reduction:
      cross_lane: true
    tail:
      exp: true
      cast: true
      stochastic_rounding: true

  memory_units:
    scratchpad:
      banks: N
      concurrent_read_write: true
    address_generation:
      stages: K
      fragmentable_read_write: true
      addr_per_cycle: 4
    data_alignment:
      transpose: true
      permute: true
      unaligned_access: true
      format_conversion: true
    banking:
      programmable_bank_bits: true
      address_predication: true

  stream_ports:
    vector:
    scalar:
    control:
```

更进一步，你的 event IR 里可能需要表达：

```yaml
event:
  type: spatial_fused_kernel
  stages:
    - pcu_gemm
    - pmu_stage_buffer
    - pcu_simd
    - pmu_transpose_layout
    - pcu_gemm
  resources:
    pcus: [...]
    pmus: [...]
    rdn_routes: [...]
  constraints:
    pmu_read_addr_stages:
    pmu_write_addr_stages:
    bank_conflict_model:
    vector_stream_bw:
    scalar_stream_bw:
    control_tokens:
```

也就是说，SambaNova 告诉我们：**PMU 的 address generation / layout transform 能力，是架构探索里必须显式建模的资源**。否则你只能看到 SRAM 容量和带宽，却解释不了为什么某些 graph fusion 能成立、某些 fusion 会失败。

---

# 9. 关键误区

## 误区一：把 PMU 当 SRAM

PMU 不是 SRAM。SRAM 只是 PMU 的一部分。PMU 真正的价值在于：

```text
SRAM banks + address ALU + alignment crossbar + predicate + stream control
```

---

## 误区二：把 PCU 当 Tensor Core

PCU 可以像 systolic array，但它不是固定 Tensor Core。它还可以变成 SIMD pipeline，并带 reduction 和 tail special function。

---

## 误区三：认为 SambaNova 只是“更多片上 SRAM”

片上 SRAM 很重要，但 SambaNova 的关键不是只有大 SRAM，而是：

```text
大分布式 PMU + 可编程地址生成 + 数据重排 + PCU/PMU/RDN 空间流水线
```

---

## 误区四：认为所有算子都在 PCU 上做

不是。很多“算子”在 SambaNova 上会变成：

* PMU read/write pattern
* data alignment
* address generation
* RDN communication pattern

比如 transpose 就可能主要是 PMU 的职责，而不是 PCU 的职责。

---

# 10. 最短总结

**PCU 是 SambaNova 的可配置计算流水线：既能做 systolic GEMM，也能做 SIMD/reduce/tail special function；PMU 是可编程 memory pipeline：既存 tensor，又负责地址生成、banking、layout transform 和 stage buffering。PCU 让数据“算起来”，PMU 让数据“以正确形态流起来”；二者通过 RDN 被 compiler 编排成 graph-level spatial pipeline。**

下一步最值得继续深挖的是：**RDN / switch / vector-scalar-control 三网络如何把 PCU 和 PMU 串成 dataflow fabric**。
