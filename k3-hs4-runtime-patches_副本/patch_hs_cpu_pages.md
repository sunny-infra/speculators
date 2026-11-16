# patch_hs_cpu_pages.py

把 `extract_hidden_states` 的 Hidden State 从 NPU hybrid KV 挪到 **TP0 CPU pages**。不减 Eagle 层、不降 bf16。只改 serve 节点 site-packages；改完必须重启 vLLM。

原理与实测见 [hs_off_npu_kv.md](../developer/hs_off_npu_kv.md)。

## 适用镜像

补丁按字符串锚点改源码，**只在下列镜像上验证过**。换 tag 后必须先 `--dry-run`，对不上就不要硬打。

| 项 | 值 |
|---|---|
| 容器镜像 | `vllm:release_0.30.0_1013_202610211952_aarch64` |
| 架构 / 卡 | aarch64，Ascend 910A3 |
| Python | 3.11.15（`/usr/local/python3.11.15`） |
| 默认 site-packages | `/usr/local/python3.11.15/lib/python3.11/site-packages` |
| vLLM（pip） | `0.1.dev100+g022658354.empty` |
| vllm-ascend（pip） | `0.1.dev100+g2facb7ef2` |
| 引擎启动日志 | `V1 LLM engine (v0.1.dev100+g022658354)` |
| torch / torch_npu | `2.10.0+cpu` / `2.10.0.post4` |
| CANN | `9.2.0`（`/usr/local/Ascend/cann-9.2.0`） |
| npu-smi | `25.5.1.1` |


未验证、不要直接打：

那些树上没有当前 `ExampleHiddenStatesConnector` / `CacheOnlyAttentionLayer` 锚点，会报 `expected 1 occurrence, found 0`。

## 新容器标准流程

在**每一台 serve 节点**上、拉起 `launch_vllm` **之前**执行。训练节点不用打这份补丁。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /path/to/speculators-sunny

python scripts/patch_hs_cpu_pages.py --dry-run
python scripts/patch_hs_cpu_pages.py
```

成功时会打印 `patched <file>` + `backup <file>.bak.<timestamp>`，或 `already patched`。可选文件缺失打印 `skip optional missing`，不算失败。

Kimi-K3 w4a8 另打（和 HS 显存无关，缺了会 load 失败）：

```bash
python scripts/patch_kimi_mm_rot_proj.py
```

然后用仓库里的 `scripts/launch_vllm.py`（默认 `hs_store_device=cpu`）。改的是磁盘上的 `.py`，**已在跑的 vLLM 不会热更新**，必须重启 serve。

不要再跑 `scripts/patch_hs_chunked_gather.py`：那是 `hs_store_device=npu` 时的分块 gather，和 CPU pages 互斥。

## 命令行

```text
python scripts/patch_hs_cpu_pages.py [--dry-run] [--<file-key> PATH]
```

| 参数 | 说明 |
|---|---|
| `--dry-run` | 只检查能否打上，不写文件、不备份 |
| `--model-runner` 等 | 覆盖单个文件路径；默认见下表 |

可重复执行：connector 已含 `HS_CPU_WORKER_REFUSE`、`HS_CPU_BLOCK_TABLE`、`kv_cache_config.num_blocks` 和 `{filename}.incomplete` 时跳过。更旧的 `HS_CPU_PAGES` / `HS_CPU_PREFIX_HASH` / `HS_CPU_UPSTREAM` 安装会就地升级。

退出码：`0` 成功（含 already patched / skip optional）；任一**必选**文件缺失或锚点不匹配为 `1`。

## 改哪些文件

默认根目录：`/usr/local/python3.11.15/lib/python3.11/site-packages`。

| `--` 参数 | 相对 site-packages 的路径 | 必选 |
|---|---|---|
| `--model-runner` | `vllm_ascend/worker/model_runner_v1.py` | 是（Ascend） |
| `--cache-only` | `vllm/model_executor/models/extract_hidden_states.py` | 是 |
| `--proposer` | `vllm/v1/spec_decode/extract_hidden_states.py` | 是 |
| `--connector` | `vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py` | 是 |
| `--kv-transfer` | `vllm/config/kv_transfer.py` | 否 |
| `--gpu-model-runner` | `vllm/v1/worker/gpu_model_runner.py` | 否 |
| `--attn-utils` | `vllm/v1/worker/gpu/attn_utils.py` | 否 |
| `--v2-speculator` | `vllm/v1/worker/gpu/spec_decode/extract_hidden_states.py` | 否 |
| `--v2-base-speculator` | `vllm/v1/worker/gpu/spec_decode/speculator.py` | 否 |
| `--v2-init-speculator` | `vllm/v1/worker/gpu/spec_decode/__init__.py` | 否 |
| `--v2-model-runner` | `vllm/v1/worker/gpu/model_runner.py` | 否 |

当前验证镜像上上表文件**全部存在**。没有 MR V2 / GPU runner 的树会 skip optional，只要 4 个必选打上即可跑 Ascend `launch_vllm`。

`scripts/hs_cpu_block_table.py` 只给单测用，逻辑已镜像进 connector 补丁，**不用**拷进 site-packages。

## 补丁阶段（marker）

connector 按顺序叠加，完整安装应同时具备：

| marker | 作用 |
|---|---|
| `HS_CPU_PAGES` | HS 不进 NPU BlockPool；TP0 逐步 DtoH 到 CPU page |
| `HS_CPU_PREFIX_HASH` | 早期 hash LRU 前缀索引（随后被 block table 取代为 SoT） |
| `HS_CPU_UPSTREAM` | 与 vllm-sunny `feat/hs-cpu-pages` 对齐：prefix 开关在 `register_kv_caches` 锁定 |
| `HS_CPU_BLOCK_TABLE` | Host HS 用与 NPU MLA 相同的 `block_id`；`num_blocks` 取 `KVCacheConfig.num_blocks` |
| `HS_CPU_WORKER_REFUSE` | 有洞/NaN 由 Worker 拒落盘并写 `{path}.incomplete`；EngineCore 只回路径 |

## 打完如何确认

```bash
SP=/usr/local/python3.11.15/lib/python3.11/site-packages
CONN=$SP/vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py

grep -n "HS_CPU_PAGES\|HS_CPU_BLOCK_TABLE\|HS_CPU_WORKER_REFUSE" \
  $SP/vllm_ascend/worker/model_runner_v1.py \
  $SP/vllm/v1/spec_decode/extract_hidden_states.py \
  "$CONN"
```

重启 serve 后日志应出现：

```text
HS_CPU_PAGES: hidden states stored on TP0 CPU, not in NPU KV
HS_CPU_BLOCK_TABLE: num_blocks=<N> block_size=128 prefix=True tp0=True
```

`N` 必须大于 0（Ascend 以 `KVCacheConfig.num_blocks` 为准，不是 `cache_config.num_gpu_blocks`）。GLM-5.2-w4a8-moe、util 0.90、64k 上大约 `2195–2268` blocks，KV 约 `4.3x`。

`launch_vllm.py` 注入的 connector extra：

- `hs_store_device=cpu`
- `hs_gather_chunk_tokens=4096`
- `num_writer_threads=2`

长序列建议同时传 `--max-num-batched-tokens 8192`，不要 `--no-enable-chunked-prefill`。

## 失败与回滚

- `expected 1 occurrence, found 0/N`：镜像/包版本与锚点不一致。不要手工改 site-packages 凑合；换回上表镜像或更新本脚本。
- `not found`（必选文件）：site-packages 不在默认 Python 3.11.15 路径，用 `--model-runner` 等显式传入。
- 回滚：同目录 `*.bak.YYYYMMDD_HHMMSS` 是打补丁前的副本，拷回去后重启 vLLM。
- 普通 `vllm serve`、Eagle/MTP/ngram **不走**这些分支；`hs_store_device=npu` 可短路回 NPU KV（需另打 `patch_hs_chunked_gather.py`）。
