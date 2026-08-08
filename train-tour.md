# 概述
<font style="color:rgb(37, 43, 58);">投机解码是一项可做到无损效果的大语言模型推理加速技术：由体量更小、推理速度更快的草稿模型预先生成多枚 Token 候选，再交由主基座校验模型（verifier）通过单次前向运算完成结果核验。该方案能够可无损加速大模型推理，显著降低推理时延。</font><font style="color:rgba(0, 0, 0, 0.9);">DSpark 通过</font>**<font style="color:rgba(0, 0, 0, 0.9);">半自回归生成架构</font>**<font style="color:rgba(0, 0, 0, 0.9);">（并行主干+Markov顺序头建立轻量token依赖）与</font>**<font style="color:rgba(0, 0, 0, 0.9);">置信度感知的动态验证调度</font>**<font style="color:rgba(0, 0, 0, 0.9);">（Confidence Head估计存活概率+硬件感知前缀调度器自适应分配验证预算），在保持目标模型输出分布无损的前提下，将投机解码的接受率从快速衰减转为几乎平坦，成为当前最具代表性与实用价值的SOTA算法。</font>

# <font style="color:rgba(0, 0, 0, 0.9);">数据准备</font>
草稿模型（draft model）训练的核心目标是拟合主模型（target model）的输出分布所以数据至关重要。数据集一般有两个来源，一是开源数据集，二是业务数据集。开源数据的answer部分主要来自人工标注或者各类主模型蒸馏，为了保证数据的输出分布与我们训练的主模型一致需要对开源数据进行重采样工作（将开源数据集的Question部分输入给主模型，重新采样主模型生成的answer以保证数据集的输出分布与主模型一致）。业务数据如果是待训练的主模型直接生成的业务数据则不需要重采样过程，业务数据理论上领域垂直程度越高训练效果越好。准备好的数据集需采用json或jsonl格式保存以方便下一轮的预处理工作。

本次GLM-5.2模型的Dspark训练采用开源数据集Open-perfectblend中的前60w条数据，经过GLM-5.2主模型重采样而来。[https://www.modelscope.cn/datasets/finghtingsun/perfectblend-for-dspark](https://www.modelscope.cn/datasets/finghtingsun/perfectblend-for-dspark)



# 环境准备
当前Dspark草稿模型训练均采用在线训练模式，即同时拉起推理和训练服务，训练作为客户端实时向推理服务发送请求，推理服务器实时生成训练需要的hidden state作为训练的输入。



## 训练环境
### vllm-ascend环境安装
#### 基础镜像
| 项目 | 值 |
| --- | --- |
| 社区镜像 | `quay.io/ascend/vllm-ascend:v0.22.1rc1-a3` |
| CANN 版本 | 9.0.0 |
| Python | 3.12.13 |
| torch | 2.10.0+cpu |
| torch_npu | 2.10.0 |


#### 代码准备
**vllm** — fix-gguf-test 分支：

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout e5588e49bc2642670116664a7fc4096e27adb179
VLLM_TARGET_DEVICE=empty pip install -U -e . --no-deps
```

**vllm-ascend** — main 分支：

```bash
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout 6e784075dcc36b603296f03a50cdc005cffe5c61

export CPLUS_INCLUDE_PATH=/usr/include/c++/12:/usr/include/c++/12/`uname -i`-openEuler-linux
COMPILE_CUSTOM_KERNELS=1 pip install -e . --no-build-isolation --no-deps
```

#### Python 依赖
```bash
pip install transformers==5.14.1
```

#### 推理脚本
8 机 DP=8, TP=16（共 128 NPU），使用 `scripts/launch_vllm.py` 启动 hidden states 提取模式。

**node0（主节点）：**

```bash
#!/bin/bash
set -euo pipefail
LOG_DIR="<log_dir>"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/serve_dp8_node0_${TS}.log"
exec > "$LOG" 2>&1

cd <speculators_dir>

nic_name="eth0"
local_ip="<node0_ip>"
node0_ip="<node0_ip>"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=512
export HCCL_CONNECT_TIMEOUT=3600
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export VLLM_VERSION="0.23.1"
export VLLM_ASCEND_ENABLE_NZ=0

python scripts/launch_vllm.py \
    <model_path> \
    --target-layer-ids 8 23 39 55 70 \
    --hidden-states-path <hidden_states_path> \
    -- --served-model-name '<model_path>' \
    --host 0.0.0.0 \
    --port 8078 \
    --tensor-parallel-size 16 \
    --data-parallel-size 8 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 0 \
    --data-parallel-address $node0_ip \
    --data-parallel-rpc-port 12345 \
    --enable-expert-parallel \
    --seed 1024 \
    --max-num-seqs 512 \
    --max-model-len 4100 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --hf-overrides '{"use_index_cache": true}' \
    --enforce-eager
```

**node1（worker 节点，需加 **`--headless`**）：**

```bash
#!/bin/bash
set -euo pipefail
LOG_DIR="<log_dir>"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/serve_dp8_node1_${TS}.log"
exec > "$LOG" 2>&1

cd <speculators_dir>

nic_name="eth0"
local_ip="<node1_ip>"
node0_ip="<node0_ip>"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=512
export HCCL_CONNECT_TIMEOUT=3600
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export VLLM_VERSION="0.23.1"
export VLLM_ASCEND_ENABLE_NZ=0

python scripts/launch_vllm.py \
    <model_path> \
    --target-layer-ids 8 23 39 55 70 \
    --hidden-states-path <hidden_states_path> \
    -- --served-model-name '<model_path>' \
    --headless \
    --port 8078 \
    --tensor-parallel-size 16 \
    --data-parallel-size 8 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 1 \
    --data-parallel-address $node0_ip \
    --data-parallel-rpc-port 12345 \
    --enable-expert-parallel \
    --seed 1024 \
    --max-num-seqs 512 \
    --max-model-len 4100 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --hf-overrides '{"use_index_cache": true}' \
    --enforce-eager
```

> 其余 node2~node7 与 node1 类似，仅 `local_ip` 和 `--data-parallel-start-rank` 不同（2~7）。
>

---

### speculators环境安装
代码分支位于：

```shell

cd 

pip install -e ./hs_connectors 
pip install -e . 
#如果安装速度过慢可借助镜像源 -i https://pypi.antfin-inc.com/simple/
```

## 测试环境
### 基础信息
| 项目 | 值 |
| --- | --- |
| 社区镜像 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` |
| CANN 版本 | 9.0.1 |
| Python | 3.12.13 |
| torch | 2.10.0+cpu |
| torch_npu | 2.10.0.post2 |


### 代码准备
**vllm** — releases/v0.26.0 分支：

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout d02df748b
VLLM_TARGET_DEVICE=empty pip install -U -e . --no-deps
```

增大引擎启动握手超时（W8A8 模型 721GB 加载需要更长时间）：

```python
# vllm/v1/engine/core.py 第 93 行
HANDSHAKE_TIMEOUT_MINS = 15  # 原值 5
```

**vllm-ascend** — main 分支：

```bash
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout e3bb5f570
```

### 问题排查与 PR 合入
在 GLM-5.2 DSpark 推测解码部署过程中，遇到了两类关键问题：head_dim=192 导致的 RoPE 计算错误，以及 DSpark 接受率低下。

---

#### 问题一：head_dim=192 导致 RoPE UB overflow
在 GLM-5.2（MLA attention, `head_dim=192`, `qk_head_dim=256`, `qk_rope_head_dim=64`）上启动 DSpark 推测解码服务时，服务启动阶段出现 UB (undefined behavior) overflow 报错，导致推理无法正常进行。

##### 分析问题
GLM-5.2 的 attention head_dim 为 192，而 vllm-ascend 的 `rope.py` 中 RoPE (Rotary Position Embedding) 实现的 threshold 硬编码为 256。当 head_dim 小于 threshold 时，RoPE 的 Triton kernel 会在超出 head_dim 范围的内存区域进行读写操作，触发 UB overflow。

具体位置：`vllm_ascend/ops/triton/rope.py` 中的 threshold 判断逻辑。

##### 合入 PR #12740
**PR**: [vllm-ascend #12740](https://github.com/vllm-project/vllm-ascend/pull/12740)

将 RoPE threshold 从 256 修改为 192，与 GLM-5.2 的 `head_dim=192` 匹配，消除 UB overflow。

```python
# vllm_ascend/ops/triton/rope.py
# 修改前: threshold = 256
# 修改后: threshold = 192
```

---

#### 问题二：DSpark 接受率偏低
在 GLM-5.2 W8A8 量化模型上启用 DSpark 推测解码后，在开源数据集gsm8k上，测得 token 接受长度不到3，性能提升不明显。

##### 分析问题
`rope.py` 中 `sin_offsets` 的计算存在错误，导致 RoPE 位置编码不正确，draft model 生成的 token 与 target model 的分布存在系统性偏差。

##### 合入 PR #12963
**PR**: [vllm-ascend #12963](https://github.com/vllm-project/vllm-ascend/pull/12963) — RoPE sin_offsets 修复

修正 `rope.py` 中 `sin_offsets` 的计算公式，使 RoPE 位置编码与标准实现一致。

```python
# vllm_ascend/ops/triton/rope.py
# 修正 sin_offsets 计算
```

应用方式：手动修改 `rope.py` 中的 sin_offsets 计算逻辑。

**效果**：接受长度大幅提升至 **4.89**，达到训练集水平（4.1-4.5），推测解码正常工作。

---





# 训练过程与超参选取
## 草稿模型结构
| **参数** | **值** |
| :--- | :--- |
| `model_type` | qwen3 |
| `num_hidden_layers` | **5** |
| `hidden_size` | 6144 |
| `intermediate_size` | **12288**（SiLU MLP） |
| `num_attention_heads` | **64** |
| `num_key_value_heads` | **64** |
| `head_dim` | **192** |
| Attention 宽度 | 64×192=1228864×192=12288（Q/K/V 投影输出维） |
| `hidden_act` | silu |
| `layer_types` | 5 层全为 `full_attention` |
| `use_sliding_window` | false |
| `max_position_embeddings` | 1048576 |
|  RoPE | `rope_theta=8e6`，`rope_type=default` |
| `rms_norm_eps` | 1e-5 |
| Attention 实现（训练） | `sdpa` |
|  |  |
|  |  |


  


## 训练脚本
```shell
#!/bin/bash
# Single-node GLM-5.2 dspark training via torchrun (16 NPUs).
# Usage: bash train-glm52-1node.sh
set -euo pipefail

MASTER_ADDR="33.182.140.216"
MASTER_PORT="29501"
NNODES=1
NPROC_PER_NODE=16
NODE_RANK=0
LOCAL_IP="33.182.140.216"
NIC_NAME="eth0"

export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export HCCL_CONNECT_TIMEOUT=3600
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export VLLM_VERSION="0.22.1"
export VLLM_ASCEND_ENABLE_NZ=0

export HCCL_IF_IP="${LOCAL_IP}"
export GLOO_SOCKET_IFNAME="${NIC_NAME}"
export TP_SOCKET_IFNAME="${NIC_NAME}"
export HCCL_SOCKET_IFNAME="${NIC_NAME}"

echo "[$(date '+%F %T')] starting single-node train: local_ip=${LOCAL_IP} master=${MASTER_ADDR}:${MASTER_PORT} nproc=${NPROC_PER_NODE}"

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  scripts/train.py \
  --verifier-name-or-path /mnt/sfs_turbo/models/ZhipuAI/GLM-5.2/ \
  --speculator-type dspark \
  --num-layers 5 \
  --block-size 8 \
  --data-path /xxx \
  --vllm-endpoint http://severhost:port/v1 \
  --save-path ./output/checkpoints-glm52 \
  --epochs 8 \
  --lr 0.0006 \
  --scheduler-type cosine \
  --total-seq-len 8192 \
  --draft-arch qwen3 \
  --draft-hidden-act silu \
  --target-layer-ids 8 23 39 55 70 \
  --max-anchors 600 \
  --markov-rank 256 \
  --enable-confidence-head \
  --confidence-head-with-markov \
  --loss-fn '{"ce": 0.1, "tv": 0.9}' \
  --confidence-head-alpha 1.0 \
  --checkpoint-freq 0.1 \
  --on-missing generate \
  --on-generate delete \
  --seed 42 \
  --log-freq 10 \
  --prefetch-factor 4 \
  --num-workers 12 \
  --request-timeout 3600 \
  --max-retries 8 \
  --trust-remote-code \
  --hidden-states-path ./hiddenstates-glm52 \
  --draft-attn-impl sdpa \
  --fsdp-shard \
  --dflash-decay-gamma 8 \
  --scheduler-total-steps 22000

```



```shell
#!/bin/bash
# Dual-node GLM-5.2 dspark training via torchrun.
# Usage: bash train-glm52-2node.sh <node_rank>
#   node_rank=0 -> master 33.182.140.216
#   node_rank=1 -> worker 33.215.117.43
set -euo pipefail

NODE_RANK="${1:?usage: $0 <node_rank>}"
MASTER_ADDR="33.182.140.216"
MASTER_PORT="29501"
NNODES=2
NPROC_PER_NODE=16
NIC_NAME="eth0"


if [[ "${NODE_RANK}" == "0" ]]; then
  LOCAL_IP="33.182.140.216"
elif [[ "${NODE_RANK}" == "1" ]]; then
  LOCAL_IP="33.215.117.43"
else
  echo "node_rank must be 0 or 1, got: ${NODE_RANK}" >&2
  exit 1
fi

export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export HCCL_CONNECT_TIMEOUT=3600
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export VLLM_VERSION="0.22.1"
export VLLM_ASCEND_ENABLE_NZ=0

# Multi-node networking
export HCCL_IF_IP="${LOCAL_IP}"
export GLOO_SOCKET_IFNAME="${NIC_NAME}"
export TP_SOCKET_IFNAME="${NIC_NAME}"
export HCCL_SOCKET_IFNAME="${NIC_NAME}"

echo "[$(date '+%F %T')] starting train: node_rank=${NODE_RANK} local_ip=${LOCAL_IP} master=${MASTER_ADDR}:${MASTER_PORT}"

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  scripts/train.py \
  --verifier-name-or-path /mnt/sfs_turbo/models/ZhipuAI/GLM-5.2/ \
  --speculator-type dspark \
  --num-layers 5 \
  --block-size 8 \
  --data-path /xxx \
  --vllm-endpoint http://severhost:port/v1 \
  --save-path ./output/checkpoints-glm52-60w \
  --epochs 6 \
  --lr 0.0002 \
  --scheduler-type cosine \
  --total-seq-len 4096 \
  --draft-arch qwen3 \
  --draft-hidden-act silu \
  --target-layer-ids 2 20 39 58 75 \
  --max-anchors 600 \
  --markov-rank 256 \
  --enable-confidence-head \
  --confidence-head-with-markov \
  --loss-fn '{"ce": 0.3, "tv": 0.7}' \
  --confidence-head-alpha 1.0 \
  --checkpoint-freq 0.1 \
  --on-missing generate \
  --on-generate delete \
  --seed 42 \
  --log-freq 10 \
  --prefetch-factor 4 \
  --num-workers 12 \
  --request-timeout 1800 \
  --max-retries 8 \
  --trust-remote-code \
  --hidden-states-path ./hiddenstate-glm52 \
  --draft-attn-impl sdpa \
  --fsdp-shard \
  --dflash-decay-gamma 10 \
  --train-data-ratio 0.95 \
  --noise-std 0.01 \
  --full-attention-indices 0 1 2 3 4

```

## 训练主要可调参数与含义
| 参数名 | 功能 | 建议 |
| --- | --- | --- |
| verifier-name-or-path  | 主模型路径读取可服用的embedding层和lm_head，向推理发送请求的model name | 因为同时承担了model_name的职责所以推理服务器启动时需要指定model name为该路径 |
| block-size | 并行预测的块大小，也就是预测步长 | 一般选择7或8，对预测步长有较高要求的可以选择16，但是size越大对显存占用越高且训练收敛越慢 |
| data-path | 训练数据集路径 | |
| vllm-endpoint | 推理服务器的api | 虽然写的是v1但是代码内会默认补齐为v1/chat/completion |
| epochs  | 训练轮次 | 看loss一般3个step之后收敛会变慢，但是考虑到续训及其他调参空间可以设置为6～8 |
| target-layer-ids | 主模型的hidden state | 从主模型取hidden state，不同模型层数不一样取值也有不同，一般是上中下均匀分布取层，以便均匀的学到相关特征 |
| max-anchors | 切分数据的max-anchors数量 | anchors数量越多理论上数据的利用率越高，但相应带来的显存占用和训练时长也会相应上升 |
| loss-fn | ce和tv复合损失的加权值 | 详见“CE 和 TV 各自优化什么” |
| num-workers | 在线训练时每卡的请求数量 | 默认为8（即A3单机下128并发），实际需要按推理服务的性能动态调整 |
| draft-attn-impl&fsdp-shard | attention后端选择 | 因为torch_npu尚未完全支持flex_attention，所以推荐sdpa后端配合fsdp-shard使用 |
| dflash-decay-gamma | 块内位置 loss 衰减指数 γ | 默认4，在训练中后段发现末尾几个token接受率一直偏低可适当增加至8或10 |
| full-attention-indices | 指定哪些层为full-attention | 当前代码仓默认attention机制为swa，如果需要使用full-att或者混合注意力需显示指定 |
| weight-decay，muon-* | 正则与 Muon 超参 | 建议默认 |


## CE 和 TV 各自优化什么
|  | **CE** | **TV** |
| --- | :--- | :--- |
| 目标 | draft 去拟合 verifier 的 argmax 硬标签 | 最小化分布总变差 1−∑vmin⁡(pv,qv)1−∑_v_min(_p__v_,_q__v_) |
| 与接受率关系 | 间接（只推“对的那个 token”） | 直接（TV 对齐 speculative decoding 的 overlap/accept） |
| 梯度特点 | 峰值尖锐，利于 top-1 准 | 管整个分布形状；overlap 已高时梯度偏弱 |
| 量纲 | 典型数值常比 TV **大** | 落在 [0,1] |


## 
# 训练结果与测试方法
## 接受率与接受长度曲线（两个颜色是因为中间中断续训过）
<img src="https://intranetproxy.alipay.com/skylark/lark/0/2026/png/193556417/1786007027026-48ae8f63-d4c3-44d2-880f-4d33b293d93f.png" width="1680" title="" crop="0,0,1,1" id="u5c4ca443" class="ne-image">

## loss曲线
<img src="https://intranetproxy.alipay.com/skylark/lark/0/2026/png/193556417/1786007040853-d171f8c3-1532-4ac6-b264-80fe4ad96896.png" width="1680" title="" crop="0,0,1,1" id="u28bf92ec" class="ne-image">

## 接受率实测
| <font style="color:black;">数据集</font> | <font style="color:black;">Position0</font> | <font style="color:black;">Position1</font> | <font style="color:black;">Position2</font> | <font style="color:black;">Position3</font> | <font style="color:black;">Position4</font> | <font style="color:black;">Position5</font> | <font style="color:black;">Position6</font> | <font style="color:black;">Position7</font> | **<font style="color:black;">Accept Length</font>** |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| <font style="color:black;">GSM8K</font> | <font style="color:black;">92.36%</font> | <font style="color:black;">84.33%</font> | <font style="color:black;">76.80%</font> | <font style="color:black;">69.59%</font> | <font style="color:black;">63.00%</font> | <font style="color:black;">57.05%</font> | <font style="color:black;">51.58%</font> | <font style="color:black;">46.29%</font> | **<font style="color:black;">6.41</font>** |
| <font style="color:black;">MATH500</font> | <font style="color:black;">93.04%</font> | <font style="color:black;">85.32%</font> | <font style="color:black;">77.62%</font> | <font style="color:black;">70.61%</font> | <font style="color:black;">64.12%</font> | <font style="color:black;">58.10%</font> | <font style="color:black;">52.48%</font> | <font style="color:black;">47.02%</font> | **<font style="color:black;">6.48</font>** |
| <font style="color:black;">Aime2025</font> | <font style="color:black;">92.23%</font> | <font style="color:black;">83.18%</font> | <font style="color:black;">74.46%</font> | <font style="color:black;">66.56%</font> | <font style="color:black;">59.47%</font> | <font style="color:black;">53.26%</font> | <font style="color:black;">47.31%</font> | <font style="color:black;">41.70%</font> | **<font style="color:black;">6.18</font>** |
| <font style="color:black;">MBPP</font> | <font style="color:black;">86.38%</font> | <font style="color:black;">71.80%</font> | <font style="color:black;">58.60%</font> | <font style="color:black;">47.51%</font> | <font style="color:black;">38.57%</font> | <font style="color:black;">31.23%</font> | <font style="color:black;">25.41%</font> | <font style="color:black;">20.59%</font> | **<font style="color:black;">4.8</font>** |
| <font style="color:black;">HumanEval</font> | <font style="color:black;">87.56%</font> | <font style="color:black;">73.77%</font> | <font style="color:black;">61.12%</font> | <font style="color:black;">50.62%</font> | <font style="color:black;">41.83%</font> | <font style="color:black;">34.80%</font> | <font style="color:black;">29.03%</font> | <font style="color:black;">24.37%</font> | **<font style="color:black;">5.03</font>** |
| <font style="color:black;">MT-Bench</font> | <font style="color:black;">78.39%</font> | <font style="color:black;">60.07%</font> | <font style="color:black;">45.76%</font> | <font style="color:black;">35.77%</font> | <font style="color:black;">29.04%</font> | <font style="color:black;">23.94%</font> | <font style="color:black;">20.37%</font> | <font style="color:black;">17.33%</font> | **<font style="color:black;">4.11</font>** |
| <font style="color:black;">SWE-bench</font> | <font style="color:black;">79.57%</font> | <font style="color:black;">61.44%</font> | <font style="color:black;">47.14%</font> | <font style="color:black;">36.20%</font> | <font style="color:black;">28.17%</font> | <font style="color:black;">22.16%</font> | <font style="color:black;">17.70%</font> | <font style="color:black;">14.38%</font> | **<font style="color:black;">4.07</font>** |


## 测试方法
### 测试服务拉起
4 机 DP=4, TP=16（共 64 NPU），使用 `vllm serve` 直接启动 DSpark 推测解码。

**node0（主节点）：**

```bash
#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.1/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=${ATB_CXX_ABI:-1}
export PATH=/usr/local/python3.12.13/bin:$PATH

set -eo pipefail

LOG_DIR="<log_dir>"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/serve_dspark4_node0_${TS}.log"
exec > "$LOG" 2>&1

nic_name="eth0"
local_ip="<node0_ip>"
node0_ip="<node0_ip>"

export PYTHONPATH=<vllm_dir>:<vllm_ascend_dir>:$PYTHONPATH
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=512
export HCCL_CONNECT_TIMEOUT=3600
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export VLLM_VERSION="0.23.1"
export VLLM_ASCEND_ENABLE_NZ=0
export TRITON_DEBUG=0

vllm serve <model_path> \
    --served-model-name '<model_path>' \
    --host 0.0.0.0 \
    --port 8078 \
    --tensor-parallel-size 16 \
    --data-parallel-size 4 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 0 \
    --data-parallel-address $node0_ip \
    --data-parallel-rpc-port 12345 \
    --enable-expert-parallel \
    --speculative-config '{"method":"dspark","model":"<dspark_model_path>","num_speculative_tokens":8}' \
    --compilation-config '{"cudagraph_capture_sizes":[9, 18, 36, 72, 144, 288, 576], "cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --seed 1024 \
    --max-num-seqs 16 \
    --max-model-len 40960 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --hf-overrides '{"use_index_cache": true}' \
    --enable-chunked-prefill
```

**node1（worker 节点，需加 **`--headless`**）：**

```bash
#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.1/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=${ATB_CXX_ABI:-1}
export PATH=/usr/local/python3.12.13/bin:$PATH

set -eo pipefail

LOG_DIR="<log_dir>"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/serve_dspark4_node1_${TS}.log"
exec > "$LOG" 2>&1

nic_name="eth0"
local_ip="<node1_ip>"
node0_ip="<node0_ip>"

export PYTHONPATH=<vllm_dir>:<vllm_ascend_dir>:$PYTHONPATH
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=512
export HCCL_CONNECT_TIMEOUT=3600
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export VLLM_VERSION="0.23.1"
export VLLM_ASCEND_ENABLE_NZ=0
export TRITON_DEBUG=0

vllm serve <model_path> \
    --served-model-name '<model_path>' \
    --host 0.0.0.0 \
    --port 8078 \
    --tensor-parallel-size 16 \
    --data-parallel-size 4 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 1 \
    --data-parallel-address $node0_ip \
    --data-parallel-rpc-port 12345 \
    --enable-expert-parallel \
    --speculative-config '{"method":"dspark","model":"<dspark_model_path>","num_speculative_tokens":8}' \
    --compilation-config '{"cudagraph_capture_sizes":[9, 18, 36, 72, 144, 288, 576], "cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --seed 1024 \
    --max-num-seqs 16 \
    --max-model-len 40960 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --hf-overrides '{"use_index_cache": true}' \
    --enable-chunked-prefill \
    --headless
```

> 其余 node2、node3 与 node1 类似，仅 `local_ip` 和 `--data-parallel-start-rank` 不同（2、3）。
>
> **W8A8 量化模型**需额外添加 `--quantization ascend` 参数。
>

### 测试命令
主节点上运行 `vllm bench serve` 进行推测解码接受率测试。以 GSM8K 为例：

```bash
#!/bin/bash

vllm bench serve \
    --backend openai-chat \
    --trust-remote-code \
    --model <model_path> \
    --tokenizer <model_path> \
    --dataset-name custom \
    --dataset-path <gsm8k_dataset_path> \
    --request-rate 10000 \
    --num-prompts 1319 \
    --max-concurrency 16 \
    --metric-percentiles "50,90,99" \
    --temperature 0 \
    --custom-output-len 3600 \
    --endpoint /v1/chat/completions \
    --base-url http://localhost:8078
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `--model` / `--tokenizer` | 主模型路径（与 serve 脚本中一致） |
| `--dataset-path` | 预处理后的数据集 JSONL 文件路径 |
| `--num-prompts` | 测试样本数（GSM8K 全量 1319） |
| `--max-concurrency` | 最大并发请求数（需与 `--max-num-seqs` 匹配） |
| `--custom-output-len` | 每条请求最大输出长度 |
| `--base-url` | vLLM 服务地址 |

