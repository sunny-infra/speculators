# K3 D-Spark 训推部署说明

本文档记录 `k3-dspark-hs4-r0-0911` 至 `k3-dspark-hs4-r3-0911` 使用的推理镜像、模型镜像、运行时补丁和 TP16/DP4 启动方式。

该服务是 D-Spark 训练使用的 Kimi K3 teacher / hidden-state producer。只有 rank 0 提供 OpenAI 兼容接口，rank 1～3 以 headless 模式参与数据并行。

## 1. 镜像和资源

推理镜像：

```text
vllm-ascend:release_0.26.0_0817_202608261510_aarch64
```

模型镜像：

```text
kimi-k3-w4a8-moe-norope
```

每个节点的资源和服务拓扑：

| 项目 | 配置 |
|---|---|
| 节点数 | 4 |
| 每节点 NPU | 16 × 910A3 |
| 每节点 CPU | 128 核 |
| 每节点内存 | 1600 GB |
| Tensor Parallel | 16 |
| Data Parallel | 4，每节点一个 local DP rank |
| Pipeline Parallel | 1 |
| Expert Parallel | 开启 |
| 服务端口 | 18088，仅 rank 0 对外提供接口 |
| DP RPC 端口 | 29500 |
| Served model name | `kimi-k3-w4a8` |


## 2. 本地文件

主要文件：

```text
k3-hs4-runtime-patches/
├── apply_all.sh
├── verify_historical_hashes.sh
├── serve_hidden_states.sh
├── K3_HS4_DEPLOYMENT.md
└── patches/
    ├── patch_kimi_k3_final_hidden.sh
    ├── patch_force_moe_alltoall.sh
    ├── patch_ascend_sampler_logprobs_mode.sh
    ├── patch_hidden_state_packed_kv_capacity.sh
    ├── patch_hidden_state_separate_kv_pool.sh
    └── patch_hs_connector_atomic_publish.sh
```

## 3. Patch 安装

将 `k3-hs4-runtime-patches` 目录复制到四个推理节点。每个节点都需要执行补丁安装，且必须在启动 vLLM 之前完成。

```bash
cd /path/to/k3-hs4-runtime-patches

# 按依赖顺序安装全部补丁
bash apply_all.sh

# 验证补丁结构及语法
bash apply_all.sh --verify

# 与 2026-09-11 hs4 环境中的五个文件进行 SHA256 校验
bash verify_historical_hashes.sh
```

`verify_historical_hashes.sh` 全部显示 `OK` 表示安装后的文件与原 hs4 服务逐字节一致。

补丁具有以下保护：

- 自动通过 Python package metadata 定位 `vllm` 和 `vllm-ascend`。
- 修改前在目标文件旁创建 `.bak.<时间>.<PID>` 备份。
- 使用临时文件和 `os.replace` 原子替换。
- 如果固定镜像的源码结构不匹配，立即退出，不继续写入。
- 重复执行时只校验已有补丁，不重复插入代码。

实际修改的五个运行时文件是：

```text
/usr/local/python3.11.15/lib/python3.11/site-packages/vllm_ascend/models/kimi_k3.py
/usr/local/python3.11.15/lib/python3.11/site-packages/vllm_ascend/ascend_forward_context.py
/usr/local/python3.11.15/lib/python3.11/site-packages/vllm_ascend/worker/model_runner_v1.py
/usr/local/python3.11.15/lib/python3.11/site-packages/vllm/v1/core/kv_cache_utils.py
/usr/local/python3.11.15/lib/python3.11/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py
```

## 4. 启动前配置

四个节点都要设置相同的四个当前 Pod IP。不要使用 2026-09-11 的历史 IP。

```bash
export HS_GROUP=hs4
export NODE0_IP=<rank-0-current-pod-ip>
export NODE1_IP=<rank-1-current-pod-ip>
export NODE2_IP=<rank-2-current-pod-ip>
export NODE3_IP=<rank-3-current-pod-ip>
```

脚本会根据传入的 rank 选择本机 IP，并设置以下关键环境变量：

```text
HCCL_IF_IP=<local-pod-ip>
VLLM_HOST_IP=<local-pod-ip>
GLOO_SOCKET_IFNAME=eth0
TP_SOCKET_IFNAME=eth0
HCCL_SOCKET_IFNAME=eth0
VLLM_ENGINE_READY_TIMEOUT_S=7200
HCCL_CONNECT_TIMEOUT=7200
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
HCCL_BUFFSIZE=800
HCCL_OP_EXPANSION_MODE=AIV
K3_FORCE_MOE_ALLTOALL=1
VLLM_ASCEND_FORCE_MOE_ALLTOALL=1
```

确认共享目录在四个节点上均可读写


如果部署到新的训练目录，需要同步修改 `serve_hidden_states.sh` 中的 `HIDDEN_STATES_PATH` 和 `LOG_DIR`。

## 5. 启动服务

完整启动脚本为：

```text
/path/to/k3-hs4-runtime-patches/serve_hidden_states.sh
```

先在 rank 0 节点启动 API server：

```bash
cd /path/to/k3-hs4-runtime-patches

nohup bash serve_hidden_states.sh 0 \
  >/tmp/hs4-bootstrap-r0.log 2>&1 &
```

看到 rank 0 开始等待其他 DP rank 后，再分别启动 rank 1～3：

```bash
# rank 1 节点
nohup bash serve_hidden_states.sh 1 \
  >/tmp/hs4-bootstrap-r1.log 2>&1 &

# rank 2 节点
nohup bash serve_hidden_states.sh 2 \
  >/tmp/hs4-bootstrap-r2.log 2>&1 &

# rank 3 节点
nohup bash serve_hidden_states.sh 3 \
  >/tmp/hs4-bootstrap-r3.log 2>&1 &
```

rank 1～3 会自动追加：

```text
--headless --data-parallel-start-rank <rank>
```

启动脚本的关键 vLLM 参数：

```text
--quantization ascend
--dtype bfloat16
--tensor-parallel-size 16
--pipeline-parallel-size 1
--data-parallel-size 4
--data-parallel-size-local 1
--enable-expert-parallel
--max-num-seqs 3
--max-model-len 18433
--max-num-batched-tokens 18448
--gpu-memory-utilization 0.98
--num-gpu-blocks-override 352
--kv-cache-dtype bfloat16
--enforce-eager
--no-enable-chunked-prefill
--language-model-only
--skip-mm-profiling
```

提取的 hidden-state layer IDs：

```text
3, 24, 48, 72, 90, 93
```

KV connector 使用：

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

## 6. 日志和健康检查

实际 vLLM 日志：

在四个 rank 全部 ready 后检查 rank 0：

```bash
curl -fsS "http://${NODE0_IP}:18088/v1/models"
```

发送一个最小 completion 请求：

```bash
curl -fsS "http://${NODE0_IP}:18088/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kimi-k3-w4a8",
    "prompt": "Hello",
    "max_tokens": 1,
    "temperature": 0
  }'
```

训练侧只需登记 rank 0 endpoint：

```text
http://<rank-0-current-pod-ip>:18088/v1
```

原 hs4 endpoint pool 使用的 `max_concurrency` 是 `1`。

## 7. 停止和重启

停止服务时，在每个节点终止对应的 `vllm serve` 进程。再次启动前不需要重复安装补丁，但建议重新检查：

```bash
cd /path/to/k3-hs4-runtime-patches
bash apply_all.sh --verify
bash verify_historical_hashes.sh
```

更换推理镜像后不要直接复用补丁。先执行各 patch 的 `--dry-run`，确认源码锚点和目标版本仍然匹配。


## 5. 启动训练服务

完整启动脚本为：

预训练
```text
/path/to/k3-hs4-runtime-patches/train-k3-dspark.sh
```
增训/微调
```text
/path/to/k3-hs4-runtime-patches/train-k3-dspark-resume.sh
```

在新 rank 0 的speculators路径下启动：

```bash
cd /path/to/speculators

nohup bash train-k3-dspark.sh 0 \
  >/tmp/hs4-bootstrap-r0.log 2>&1 &
```

多机训练修改 torchrun的nnodes 即可
