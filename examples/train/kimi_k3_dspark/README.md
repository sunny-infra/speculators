# Kimi-K3 DSpark training (multi-node, online extraction)

Kimi-K3 does not fit on a single node, so this example keeps hidden-state extraction and draft training on disjoint node groups and streams hidden states between them through a Mooncake store:

```text
extraction (2 nodes)                 training (1 node)
Kimi-K3 vLLM TP8 + expert parallel -> torchrun DSpark DDP, 4 ranks
                 \                    /
                  Mooncake streaming store
```

This is a worked example for one 4-GPU-per-node NVL72 rack, not a portable harness. Values are hardcoded so the commands stay readable; adapt them rather than configuring them.

`k3_draft_mla_config.json` is the draft **decoder** config — a `K3Config` (`model_type: k3_dspark`) that `--draft-config` loads as the speculator's `transformer_layer_config`. It is Dense MLA (64 heads, YaRN, `v_head_dim=128`), not Qwen3 GQA. Everything above the decoder (block size, vocab size, mask token, target layers, the Markov and confidence heads) is set by the flags in `launch_train_node.sh`.

`--draft-config` fully defines the decoder, so do not also pass `--draft-arch` (those flags conflict). `--fc-norm` is set explicitly because the K3 per-aux RMSNorm is not implied by `--draft-config` alone at the CLI layer; `from_training_args` also forces it for `k3_dspark`.

The previous GQA decoder (`k3_draft_layer_config.json`) is kept for comparison. Point `--draft-config` at it instead, and drop `--fc-norm`.

## Requirements

- Two environments: one with vLLM plus `hs_connectors` and `mooncake-transfer-engine` (extraction), one with `speculators` plus `mooncake-transfer-engine` (training).
- Verifier weights at `/mnt/shared/weights/kimi-k3`, or set `MODEL_PATH`.
- `--trust-remote-code` is required: Kimi-K3's HuggingFace config is a multimodal wrapper (`kimi_k3` → `text_config` / `kimi_linear`).

## The components

### 1. Prepare data

```bash
SOURCE_JSONL=/path/to/data.jsonl \
  sbatch --export=ALL,SOURCE_JSONL examples/train/kimi_k3_dspark/prepare_data.sbatch
```

Writes the Arrow dataset and token frequencies to `runs/kimi_k3_dspark/data`. Everything below expects the prepared data to exist.

### 2. Mooncake master

One per run, anywhere both groups can reach:

```bash
mooncake_master --rpc_port 50051 --metrics_port 9003 \
  --rpc_thread_num 8 --enable_disk_eviction=false --logtostderr=true
```

### 3. Extractor (both nodes of the TP8 pair)

```bash
NODE_RANK=0 EXTRACT_ADDR=<head-fabric-ip> MOONCAKE_MASTER=<host>:50051 \
  FABRIC_SUBNET=<subnet-prefix> \
  bash examples/train/kimi_k3_dspark/launch_extractor_node.sh
```

`NODE_RANK=0` on the head, `1` on the other; `EXTRACT_ADDR` is the head's fabric IP on both. Rank 1 runs `--headless`. Wait for `/health` on port 8000 before starting training.

### 4. Trainer

```bash
MOONCAKE_MASTER=<host>:50051 VLLM_ENDPOINT=http://<head-fabric-ip>:8000/v1 \
  FABRIC_SUBNET=<subnet-prefix> \
  bash examples/train/kimi_k3_dspark/launch_train_node.sh
```

Add `MAX_STEPS=20` for a smoke run. Other useful overrides: `MODEL_PATH`, `DATA_DIR`, `RUN_NAME`.

## Running all four under Slurm

`run.sbatch` does exactly the above on a 3-node allocation: nodes 0-1 extract, node 2 trains, the Mooncake master shares node 0. It resolves the extractor head's fabric address, waits for `/health`, then launches training and stops the extractors and master when training exits.

```bash
FABRIC_SUBNET=<subnet-prefix> sbatch --export=ALL examples/train/kimi_k3_dspark/run.sbatch
```

Logs and checkpoints land in `runs/kimi_k3_dspark/<run-name>/`.

## Integrity and failure handling

Every Mooncake sample carries a versioned shape/dtype/CRC32 manifest. The producer rejects non-finite tensors before publication; the consumer verifies checksum and finiteness before training. Invalid round trips are regenerated twice, independently of the HTTP retry budget. An exhausted sample is dropped; if that leaves a rank with no valid samples it runs a locally empty zero-loss batch and still participates in DDP. Twenty consecutive failed round trips in one worker trip a synchronized circuit breaker.

## MLA vs GQA draft

`launch_train_node.sh` trains the TorchSpec-shaped MLA draft (`K3DSparkModel` / `k3_dspark`): 64 heads, `intermediate_size=14336`, YaRN (`factor=32`, `mscale_all_dim=1.0`), per-aux-layer `fc_norm`, and capture ids `7 31 47 63 87`. Checkpoints save `architectures: ["K3DSparkModel"]`.

The GQA decoder in `k3_draft_layer_config.json` (96 heads, SWA 2048, ids `24 48 72 88 92`) is a different geometry and will not match a `K3DSparkModel` loader.

## Aux hidden-state stream

This example uses speculators' regular residual-stream aux hiddens from `scripts/launch_vllm.py`. TorchSpec's published `lightseekorg/kimi-k3-dspark` was trained on Kimi-K3 AttnRes softmax-mixture taps (`aux_hidden_stream: "attn_res"`), which need a patched vLLM. The training code is the same either way; acceptance length will not match that checkpoint unless the extractor emits the same tap.

## Loss coefficients

speculators `tv_loss` is `½·Σ|p−q|` (total variation). TorchSpec's `l1` term is `Σ|p−q|`. The example `--loss-fn '{"ce":0.1,"tv":0.9}'` is therefore an L1 coefficient of 0.45, not TorchSpec stage-1's 0.9. Use `"tv": 1.8` to match that recipe. `block_size` here is 8; TorchSpec stage-1 used 7.
