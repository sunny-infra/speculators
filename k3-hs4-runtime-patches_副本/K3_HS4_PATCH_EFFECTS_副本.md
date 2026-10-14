# K3 D-Spark Hidden-State Runtime Patches：修改内容与作用说明

> 记录对象：`k3-hs4-runtime-patches` 中的 6 个运行时补丁。
> 目标场景：Kimi-K3 / K3 W4A8 在 vLLM-Ascend 上作为 D-Spark 的 hidden-state teacher 运行。
> 相关部署：TP16 / DP4 / EP，抽取 layer `3, 24, 48, 72, 90, 93` 的 hidden states。

## 1. 整体链路

这组补丁服务于 vLLM 的 `extract_hidden_states` speculative method 和 `ExampleHiddenStatesConnector`：

```text
Kimi-K3 target model forward
    -> 在指定 layer 抽取 hidden states
    -> HiddenStateCacheSpec / cache-only attention layer
    -> vLLM KV cache planner 分配 cache tensor
    -> ExampleHiddenStatesConnector 写出 safetensors
    -> D-Spark 训练侧读取并重构 teacher logits
```

6 个补丁总共修改 5 个运行时 Python 文件。其中 `vllm/v1/core/kv_cache_utils.py` 被两个 KV-cache 相关补丁连续修改：

| Patch | 修改文件 | 主要问题 | 主要效果 |
|---|---|---|---|
| `patch_kimi_k3_final_hidden.sh` | `vllm_ascend/models/kimi_k3.py` | 最后一层 hidden state 抽取位置错误 | 修正 layer 93 的数值语义，保存 final RMSNorm 之前且 AttnRes 已合并的 residual stream |
| `patch_force_moe_alltoall.sh` | `vllm_ascend/ascend_forward_context.py` | A3 上 MoE DispatchV2 通信路径失败 | 增加环境开关，强制使用 ALLTOALL，绕开 MC2/DispatchV2 bug |
| `patch_ascend_sampler_logprobs_mode.sh` | `vllm_ascend/worker/model_runner_v1.py` | AscendSampler 未继承 vLLM `logprobs_mode` 配置 | 显式传入 ModelConfig 的 `logprobs_mode`，保留 raw logits/正确采样语义 |
| `patch_hidden_state_packed_kv_capacity.sh` | `vllm/v1/core/kv_cache_utils.py` | packed/hybrid KV cache 容量和准入计算错误 | 让 packed 路径的并发数、请求内存和物理分配保持一致 |
| `patch_hidden_state_separate_kv_pool.sh` | 同一个 `kv_cache_utils.py` | 当前 Ascend runner 不能消费 packed `offset/block_stride`，且 hidden-state page 与普通 KV 差异极大 | 为 Ascend 新增普通、无 stride、按 native page 分桶的独立 hidden-state pool |
| `patch_hs_connector_atomic_publish.sh` | `vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py` | 直接写最终 safetensors 可能被读者读到半写文件 | 先写临时文件，再 `os.replace` 原子发布 |

`patch_hidden_state_packed_kv_capacity.sh` 必须先于 `patch_hidden_state_separate_kv_pool.sh` 应用，后者的安装脚本会检查前者的标记是否存在。

---

## 2. 各 patch 的具体修改

### 2.1 `patch_kimi_k3_final_hidden.sh`

**修改目标：**

```text
/usr/local/python3.11.15/lib/python3.11/site-packages/vllm_ascend/models/kimi_k3.py
```

**修改内容：**

1. 原始逻辑会把 layer `num_hidden_layers`，也就是 K3 的最后一层 hidden state，和普通中间层一样提前 append 到 `aux_hidden_states`。
2. 此时输出 AttnRes 的 pending block residual 还没有合并，因此拿到的 layer 93 tensor 数值语义不对。
3. patch 后：
   - 禁止原来 raw final-layer capture；
   - 在 `_apply_attention_residual(...)` 完成后、最终 `self.norm(...)` RMSNorm 之前，再 append layer 93 hidden state。

**效果：**

```text
最后一层 output-AttnRes materialized residual stream
    -> final RMSNorm 之前
    -> append 到 aux_hidden_states
```

D-Spark 训练侧后续会用 BF16 verifier 的 final norm 和 lm_head 重建 teacher logits，因此 layer 93 必须在这个 pre-final-norm 位置导出。该 patch 不改变模型权重，只修正抽取位置的框架逻辑。

---

### 2.2 `patch_force_moe_alltoall.sh`

**修改目标：**

```text
/usr/local/python3.11.15/lib/python3.11/site-packages/vllm/ascend_forward_context.py
```

准确文件为：

```text
vllm_ascend/ascend_forward_context.py
```

**修改内容：**

在 `_select_a3_moe_comm_method(...)` 中新增环境变量开关：

```text
VLLM_ASCEND_FORCE_MOE_ALLTOALL
```

当该变量设为 `1` 时，选择函数立即返回：

```python
MoECommType.ALLTOALL
```

从而跳过 MC2 / `MoeDistributeDispatchV2` 路径。

**效果：**

在 4 节点 A3 拓扑的小 batch/profile/prefill 场景下，DispatchV2 可能出现 MTE invalid-GM 或 cross-device timeout。该 patch 是 opt-in 修复：

```bash
export VLLM_ASCEND_FORCE_MOE_ALLTOALL=1
```

启动脚本同时导出：

```bash
export K3_FORCE_MOE_ALLTOALL=1
export VLLM_ASCEND_FORCE_MOE_ALLTOALL=1
```

其中这个补丁实际读取的是 `VLLM_ASCEND_FORCE_MOE_ALLTOALL`。默认不设置该变量时，原有行为不变。

---

### 2.3 `patch_ascend_sampler_logprobs_mode.sh`

**修改目标：**

```text
vllm_ascend/worker/model_runner_v1.py
```

**修改内容：**

原始：

```python
self.sampler = AscendSampler()
```

修改为：

```python
self.sampler = AscendSampler(
    logprobs_mode=self.model_config.logprobs_mode
)
```

**效果：**

vLLM 的 `ModelConfig` 支持 `logprobs_mode`，例如 `raw_logprobs` / `raw_logits`。Ascend worker 原来没有把该配置传给 `AscendSampler`，导致 Ascend 路径使用自己的默认值。

patch 后，vLLM-Ascend 采样行为遵循服务配置。如果外部训练或校验流程依赖 raw logits，该 patch 能避免采样器把 raw logits 过早转换或处理成其他形式。

---

### 2.4 `patch_hidden_state_packed_kv_capacity.sh`

**修改目标：**

```text
vllm/v1/core/kv_cache_utils.py
```

**修改内容 1：修正 packed 路径的 max concurrency 计算**

在 packed layout 下，一个 request 会为每个 logical KV group 各取一组 block ID。patch 后使用：

```python
blocks_per_request = sum(
    cdiv(
        group.kv_cache_spec.max_memory_usage_bytes(vllm_config),
        group.kv_cache_spec.page_size_bytes,
    )
    for group in kv_cache_groups
)
```

再计算：

```python
max_concurrency = num_blocks / blocks_per_request
```

**修改内容 2：修正 packed 路径的 admission / request required memory**

patch 后按照物理 allocator 真实布局计算：

```python
bytes_per_block = _pool_bytes_per_block(...)
blocks_needed = sum(...)
required = bytes_per_block * blocks_needed
```

避免 legacy generic hybrid formula 把大 hidden-state page 按普通 attention layer 数错误放大。

**修改内容 3：保留 HiddenStateCacheSpec 的 native page**

在 cross-layer packing 或后续 separate-pool 路径中，hidden-state cache 不再强行走普通 attention 的 common page 对齐，而是保留自身较大的 native page。

**效果：**

- vLLM KV cache planner 对 `HiddenStateCacheSpec` 的容量判断不再虚高；
- request admission 不再误拒物理上可容纳的配置；
- long context hidden-state extractor 可以与 K3 MLA/SWA/Mamba 等 hybrid cache 共存时得到正确容量计算；
- 该 patch 是后续 separate hidden-state pool patch 的显式前置依赖。

---

### 2.5 `patch_hidden_state_separate_kv_pool.sh`

**修改目标：**

```text
vllm/v1/core/kv_cache_utils.py
```

这是当前部署中降低长序列显存压力最关键的补丁。

**修改内容 1：新增开关和 selector**

新增函数：

```python
_use_separate_hidden_state_pool(...)
```

当 KV transfer 配置中包含：

```json
{
  "enable_separate_hidden_state_pool": "True"
}
```

且同时存在普通 attention/Mamba cache 和 hidden-state cache group 时，选择独立的 hidden-state pool。

**修改内容 2：禁止 hidden-state separate path 走 packed planner**

在 separate path 下，`_use_packed_kv_cache_config(...)` 返回 false。

原因：当前 pin 版本的 vLLM-Ascend runner 不消费 upstream packed `KVCacheTensor.offset/block_stride`。如果让 Ascend runner 处理这类 packed view descriptor，可能出现重复分配或错误 reshape。

**修改内容 3：普通 cache 与 hidden-state cache 分开建 physical slot**

新增：

```python
_separate_hidden_state_pool_slots(...)
```

它将 group 分为：

```text
regular attention / Mamba groups
    -> 按 page size 分桶，可共享 physical tensor slot

hidden-state group
    -> 独立分桶，不与普通 attention cache 混用
```

普通层仍然使用一个全局 `BlockPool` 和同一个 block ID 空间；hidden state 不再作为普通 attention tensor 的别名。

**修改内容 4：生成普通、无 stride 的 `KVCacheTensor`**

separate-planner 输出类似：

```python
KVCacheTensor(
    size=page_size * num_blocks,
    shared_by=shared_by,
)
```

这些是普通完整 tensor descriptor，而不是依赖 `offset` / `block_stride` 的 packed backing tensor view。

**修改内容 5：同步修正 bytes-per-block、并发和 admission**

该 path 中：

```python
bytes_per_block = sum(page_size for page_size, _ in slots)
```

请求所需 block 数也按每个 logical group 的实际 page 数求和，使 admission 和物理 tensor 分配完全一致。

**效果：**

- Ascend runner 能使用现有 attention / Mamba / cache-only 分支正常分配和 reshape tensor；
- hidden-state tensor 不与普通 KV tensor alias；
- 避免 packed `offset/block_stride` 在 Ascend 上无法消费的问题；
- 大 hidden-state page 不再被强行对齐到普通 attention page；
- 相同长序列所需 block ID 数显著减少，避免大量 padding 和重复分配；
- 搭配 `--num-gpu-blocks-override 352`，可把 KV cache pool 固定在一个可控范围内运行。

---

### 2.6 `patch_hs_connector_atomic_publish.sh`

**修改目标：**

```text
vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py
```

**修改内容：**

将最终路径的直接写入：

```python
save_file(tensors, filename)
```

改成：

```text
1. 生成 filename.<pid>.<time_ns>.tmp
2. event.synchronize()
3. save_file(tensors, tmp_path)
4. os.replace(tmp_path, filename)
5. 异常时清理 tmp 文件
```

**效果：**

训练侧只会在最终路径看到不存在或完整写完的 safetensors 文件，不会读到 partial file。这对共享文件系统、尤其 no-lock NFS 上的生产者/消费者模式很重要。

---

## 3. 为什么 hidden-state 模式显存占用比正常推理大

### 3.1 正常推理只需要普通 KV cache

正常 vLLM 推理时，NPU 显存主要由以下部分组成：

```text
模型权重
forward 中间 activation
attention / Mamba 等 KV cache pool
runtime / workspace / framework buffer
```

K3 本身是大型 MoE 模型，正常 forward activation 已经很大，但 cache 侧主要存 attention 需要的 K/KV 状态。MLA/K3 的普通 attention cache 通常经过模型自身的压缩或特定 layout，与完整 hidden vector 不同。

而 `extract_hidden_states` 模式还必须额外把指定层的 hidden state 缓存起来。

### 3.2 hidden-state 是完整的“普通模型激活态”，不是压缩后的 attention KV

当前配置抽取 6 层：

```text
3, 24, 48, 72, 90, 93
```

vLLM 的 `ExtractHiddenStatesModel` 会创建一个 `CacheOnlyAttentionLayer`，并把 6 层 hidden state 组织成一个伪 attention cache：

```text
num_heads = num_hidden_states = 6
head_size = target model hidden_size
kv_cache_torch_dtype = bfloat16
```

因此每个 token 的 hidden-state cache 大小近似为：

```text
bytes_per_token
  = num_hidden_states
  × hidden_size
  × dtype_bytes
  = 6
  × hidden_size
  × 2   # bfloat16
```

也就是：

```text
6 × hidden_size × 2 bytes / token
```

这 6 层 hidden vector 是完整 residual stream / model activation，不是普通 attention 中为了 decode 检索而压缩后的 K/V。因此它每个 token 的状态通常显著大于常规 attention cache。

### 3.3 长 sequence 时额外显存线性增长

忽略 block padding 时，hidden-state cache 的额外显存为：

```text
hidden_state_cache_bytes
  = sequence_length
  × 6
  × hidden_size
  × 2
```

加上 block padding：

```text
num_hidden_blocks ≈ ceil(sequence_length / block_size)

hidden_state_cache_bytes
  = num_hidden_blocks
  × block_size
  × 6
  × hidden_size
  × 2
```

所以长序列场景下：

- 每个 token 都会留下 6 份 BF16 hidden vector；
- 显存随 sequence length 线性增加；
- 该存储在写出给 connector 前，必须留在 vLLM 的 cache pool 中；
- 这些临时 activation 不只是 forward 中的占用，还被复制/插入到 KV cache shaped buffer，作为 persistent cache 形式消耗 NPU 显存。

### 3.4 hidden-state 模式是在普通 KV cache 之外额外增加 pool

启用 `extract_hidden_states` 后，同一个 vLLM worker 同时分配：

```text
普通 target model KV cache
+ hidden-state cache-only layer
+ forward activation / MoE 中间态
+ connector 和 runtime buffer
```

因此它不可能比普通推理更省显存，至少多出 6 层 hidden state 的 cache 部分。

---

## 4. KV-cache patch 如何降低长序列推理时的显存占用

### 4.1 它降低的是“额外浪费”，不是不可避免的 6 层 hidden state 本身

必须先说明边界：

> 6 个指定层 hidden state 的物理数据量是业务需求决定的，patch 无法让这部分数据消失。

patch 降低的是 planner / allocator 额外造成的显存浪费，包括：

1. page 对齐引起的 padding；
2. 大 hidden-state page 被映射到普通 attention page 后导致的低效 block；
3. hidden-state block 数被放大；
4. Ascend 无法使用 packed view 时造成的重复 backing allocation；
5. admission / concurrency 计算错误导致 vLLM 选择更大的 pool 或误判容量。

---

### 4.2 问题一：native hidden-state page 被对齐到普通 attention page

`HiddenStateCacheSpec` 是一个很大的 page spec：

```text
6 × hidden_size × 2 bytes / token
```

而普通 attention cache 的 page size 往往比它小。

如果没有 patch，legacy generic hybrid planner 会尝试把 hidden-state spec 对齐到普通 attention 的 common page：

```python
new_bs = max(common_page // per_token, 1)
```

由于 hidden state 的 `per_token` 很大，`common_page // per_token` 可能变成很小的数，甚至是 1。后果是：

```text
一个 hidden-state block 实际只能保存很少 token
长序列需要大量 hidden-state block ID
每个 block 还带有 page padding
```

例如长上下文 18K token 时，如果隐藏态 block 只能保存几个 token，就需要数千个 block ID，而不是 `ceil(18432 / native_block_size)` 这么少。

**patch 的改进：**

separate-pool 路径保留 HiddenStateCacheSpec 自身 native page：

```text
block_size 保持不变
page_size 保持不变
每个 block 保存 native_block_size 个 token
```

因此长序列所需的 hidden-state block ID 数从：

```text
ceil(L / very_small_aligned_block_size)
```

降到：

```text
ceil(L / native_block_size)
```

这是长序列显存占用下降的主要来源之一。

---

### 4.3 问题二：hidden-state 和普通 cache 混在 shared pool 中造成放大

普通 hybrid planner 会把多个 logical group 组织到共享 physical tensor slots，同一个 global block ID 会索引所有 physical slot。对普通同构 attention 来说，所有 logical group 的 token/block 进度类似，这种共享是合理的。

但 hidden-state page 与普通 attention page 差异巨大。如果 hidden state 被强行走同一套 page/物理布局，会出现两类问题：

```text
1. hidden-state page 导致长序列需要大量 global block ID；
2. 每个 global block ID 都会在 regular attention / Mamba physical slot 上保留一个 page。
```

第二点尤其重要：浪费不只是 hidden-state tensor 本身。为了让普通 cache 和 hidden cache 复用同一个 global BlockPool，普通 attention 的每个 physical slot 也会按 `num_blocks` 分配。例如 native block 本应保存 `block_size` 个 token，如果退化成每 block 1 个 token，同样 18K sequence 的 global block 数最多会放大约 `block_size` 倍；普通 attention storage 也会随着这些 block ID 被同步放大。

**patch 的改进：**

`patch_hidden_state_separate_kv_pool.sh` 将物理 slot 分成：

```text
regular attention / Mamba slots
hidden-state slot
```

并用公式：

```python
bytes_per_block = sum(page_size for page_size, _ in slots)
```

计算一个 global block 在所有 physical slot 上的真实总 bytes。普通层共享普通 slot，hidden state 使用独立 slot。这样：

- hidden-state 的大 page 不再被强转为普通 attention page；
- hidden-state tensor 不与普通 attention / Mamba tensor alias；
- admission 与 allocation 使用同一套真实 layout；
- native page 让长序列只需要较少 global block ID，从而同时避免 regular slot 和 hidden slot 的 block 数被同步放大。

---

### 4.4 问题三：Ascend runner 不能安全消费 packed `offset/block_stride`

upstream vLLM 的 packed planner 可以用一个 backing tensor 加上：

```text
KVCacheTensor.offset
KVCacheTensor.block_stride
```

表示多个 page-size bucket 的 view。

但当前 pin 版本的 vLLM-Ascend runner 不实现这些 packed view 语义。如果直接使用，Ascend 侧可能按 descriptor 分配完整 backing，造成：

```text
同一物理 backing 被分配多次
```

或者 reshape 时不能正确理解 tensor layout。

**patch 的改进：**

separate-pool path 禁用 packed planner，改输出普通、无 stride 的 descriptor：

```python
KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
```

Ascend runner 可以直接按现有 attention / Mamba / cache-only 代码分配和 reshape，避免重复 backing allocation。

---

### 4.5 问题四：admission / concurrency 误算导致 pool 过大或长序列失败

patch 前的 generic formula 对 hidden-state spec 的处理偏保守，可能：

- 把一个可运行配置误判为无法容纳；
- 在 admission 中将 hidden-state block 数乘上最大 attention group size；
- 在 concurrency 中使用不符合物理分配的 block 假设。

patch 后：

```python
required = bytes_per_block × blocks_needed
```

其中：

```text
bytes_per_block = physical slot page sizes 之和
blocks_needed = 每个 logical group 实际需要的 page 数之和
```

这使 vLLM 可以准确知道长 request 的显存需求，避免：

```text
1. 物理可容纳但被 admission 拒绝；
2. 实际只需较少 block 却请求过大的 cache pool；
3. max concurrency 计算错误。
```

当前启动参数中的：

```text
--num-gpu-blocks-override 352
```

也在这个基础上进一步固定全局 block 数，避免 vLLM 根据错误估算自动吃掉更多显存。patch 让这个 override 与真实 physical layout 配套有效。

---

## 5. KV-cache 相关 patch 的联合降低显存风险的效果总结

当前 KV transfer 配置为：

```json
{
  "kv_connector": "ExampleHiddenStatesConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "shared_storage_path": "<HIDDEN_STATES_PATH>",
    "enable_cross_layers_blocks": "False",
    "enable_separate_hidden_state_pool": "True"
  }
}
```

其中：

```text
enable_cross_layers_blocks = False
```

表示不启用跨层打包 packed path。

```text
enable_separate_hidden_state_pool = True
```

表示使用 Ascend-safe separate pool。

联合效果如下：

1. `HiddenStateCacheSpec` 保持 native block/page size；
2. hidden-state pool 与普通 attention / Mamba pool 分开；
3. `kv_cache_utils.py` 中的 admission、并发和字节数计算与实际物理分配一致；
4. Ascend runner 只处理普通完整 tensor，不处理无法消费的 packed view；
5. 长序列所需 block ID 和 padding 大幅减少；
6. 配合 `num_gpu_blocks_override`，可控制 KV pool 显存规模；
7. 长上下文 hidden-state teacher 在 16 卡 A3 上可稳定运行。

---

## 6. 当前验证状态

在 `fyf-dspark-train-1006-infer-v00` 中执行：

```bash
bash apply_all.sh --verify
```

结果：

```text
6 个补丁均显示 Verified existing
返回码 0
```

结果示例：

```text
Verified existing K3 final-hidden patch
Verified existing force-ALLTOALL patch
Verified existing AscendSampler logprobs patch
Verified existing packed KV capacity patch
Verified existing separate hidden-state KV pool patch
Verified existing connector atomic-publish patch
All K3 hs4 runtime patches completed successfully.
```

历史镜像 hash 校验：

```bash
bash verify_historical_hashes.sh
```

结果不是完全一致：

```text
5 个唯一目标文件中：
4 个 DIFF
1 个 OK
```

这不是补丁未合入的表现；`apply_all.sh --verify` 已确认 6 个补丁标记均存在。主要原因是当前 pod 使用的镜像 tag 与 2026-09-11 历史 hs4 环境不一致，目标基础文件在非补丁区域也存在差异。

---

## 7. 一句话总结

这 6 个补丁分别修了：

```text
1. layer 93 hidden state 的抽取语义
2. A3 MoE 通信 fallback
3. Ascend sampler 的 logprobs 配置传递
4. packed/hybrid KV cache admission 和并发计算
5. Ascend-safe 独立 hidden-state pool
6. hidden-state safetensors 原子发布
```

其中，hidden-state 模式显存更高，是因为它必须在普通 KV cache 之外额外保存 6 个完整 BF16 hidden vector。KV-cache 相关 patch 通过保留 native page、拆分 regular/hidden physical pool、去掉 Ascend 无法安全消费的 packed stride，并修正容量计算，减少长序列下的 padding、误分配和重复 backing allocation，从而把显存占用控制在相对合理且稳定的范围。
