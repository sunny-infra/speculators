# K3 D-Spark hs4 runtime patches

This bundle reproduces the runtime changes used by `k3-dspark-hs4-r0-0911`
through `k3-dspark-hs4-r3-0911`. It is intended for the pinned image:

```text
hcr.meta-guian02.guian.hw-a3.local/antsys/vllm:release_0.26.0_0817_202608261510_aarch64
```

The scripts patch five installed Python files. Every write creates a timestamped
backup next to the target and uses an atomic rename. A structural mismatch exits
before modifying the file.

## Apply on every DP node

Copy this directory into each inference Pod, then run before starting vLLM:

```bash
cd k3-hs4-runtime-patches
bash apply_all.sh
bash apply_all.sh --verify
bash verify_historical_hashes.sh
```

The scripts are idempotent. Re-running them verifies an existing installation.
The last command additionally checks byte-for-byte SHA256 parity with the five
files deployed to hs4 on 2026-09-11.

## Start the TP16/DP4 service

On every node, set all four current Pod IPs and the local rank:

```bash
export HS_GROUP=hs4
export NODE0_IP=<rank-0-pod-ip>
export NODE1_IP=<rank-1-pod-ip>
export NODE2_IP=<rank-2-pod-ip>
export NODE3_IP=<rank-3-pod-ip>
```

Start rank 0 first:

```bash
nohup bash serve_hidden_states.sh 0 >/tmp/hs4-bootstrap-r0.log 2>&1 &
```

Then start the other nodes with their corresponding rank:

```bash
nohup bash serve_hidden_states.sh 1 >/tmp/hs4-bootstrap-r1.log 2>&1 &
nohup bash serve_hidden_states.sh 2 >/tmp/hs4-bootstrap-r2.log 2>&1 &
nohup bash serve_hidden_states.sh 3 >/tmp/hs4-bootstrap-r3.log 2>&1 &
```

Only rank 0 exposes the OpenAI-compatible endpoint on port `18088`. The script
writes the actual vLLM logs under `logs/teachers/hs4/vllm_node<RANK>.log`.

## Required runtime settings

Enable the MoE fallback and the separate hidden-state pool:

```bash
export K3_FORCE_MOE_ALLTOALL=1
export VLLM_ASCEND_FORCE_MOE_ALLTOALL=1
```

The vLLM KV connector must include:

```json
{
  "kv_connector": "ExampleHiddenStatesConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "shared_storage_path": "/path/to/hidden-states",
    "enable_cross_layers_blocks": "False",
    "enable_separate_hidden_state_pool": "True"
  }
}
```

Restart vLLM after applying the patches. For TP16/DP4, start rank 0 first and
then ranks 1–3 with `--headless --data-parallel-start-rank <rank>`.

## Individual review

Each patch supports a read-only preview and installed-state verification:

```bash
bash patches/patch_kimi_k3_final_hidden.sh --dry-run
bash patches/patch_kimi_k3_final_hidden.sh --verify
```

The KV-cache scripts have an order dependency: apply the packed-capacity patch
before the separate-pool patch.
