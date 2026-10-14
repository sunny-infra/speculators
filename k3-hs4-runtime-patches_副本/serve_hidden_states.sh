#!/usr/bin/env bash
# Four-node K3 W4A8 HS producer for Inferact continuation (TP16 / DP4 / EP).
# Set HS_GROUP=hs4 and NODE0_IP..NODE3_IP to this group's current Pod IPs.
# Run: bash dspark_kimi_k3_hs.sh RANK (0, 1, 2, 3); start rank 0 first.
# Add each group's rank-0 URL to the training pool only after readiness checks.
#
# Engine image:
# hcr.meta-guian02.guian.hw-a3.local/antsys/vllm:release_0.26.0_0817_202608261510_aarch64
# Teacher CSI image (mounted at /home/admin/model-csi/model):
# hcr.meta-guian02.guian.hw-a3.local/aistudio/modelhub_69400046_kimi-k3-w4a8-norope:142600122_20260824234539
# Reuse the existing K3 HS runtime patches: final hidden pre-RMSNorm,
# packed KV capacity, separate HS pool, and forced MoE ALLTOALL.

set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh

SERVICE_RANK=${1:?usage: bash dspark_kimi_k3_hs.sh RANK}
HS_GROUP=${HS_GROUP:?set HS_GROUP to hs4}
NODE_IPS=(
  "${NODE0_IP:?set NODE0_IP}"
  "${NODE1_IP:?set NODE1_IP}"
  "${NODE2_IP:?set NODE2_IP}"
  "${NODE3_IP:?set NODE3_IP}"
)
case "$SERVICE_RANK" in
  0|1|2|3) ;;
  *) echo "RANK must be 0, 1, 2 or 3" >&2; exit 2 ;;
esac
LOCAL_IP=${NODE_IPS[$SERVICE_RANK]}

SHARED_ROOT=/a3_inference/itask/workdir/shared/fyf02367500/k3-dspark-shared
RUN_ID=m1-k3-inferact-48701646-s18432-20260910
MODEL_PATH=/home/admin/model-csi/model
HIDDEN_STATES_PATH="/a3_inference/itask/workdir/fyf02367500/k3-dspark-train-0902/k3-dspark-train/m1-k3-weight/hidden-states"
LOG_DIR="/a3_inference/itask/workdir/fyf02367500/k3-dspark-train-0902/k3-dspark-train/m1-k3-weight/logs/teachers/${HS_GROUP}"
mkdir -p "$LOG_DIR" "$HIDDEN_STATES_PATH"

export HCCL_IF_IP=$LOCAL_IP
export VLLM_HOST_IP=$LOCAL_IP
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export HCCL_CONNECT_TIMEOUT=7200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export HCCL_BUFFSIZE=800
export HCCL_OP_EXPANSION_MODE=AIV
export K3_FORCE_MOE_ALLTOALL=1
export VLLM_ASCEND_FORCE_MOE_ALLTOALL=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:${LD_LIBRARY_PATH:-}
unset RANK NODE_RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK MASTER_ADDR MASTER_PORT

SPECULATIVE_CONFIG='{"method":"extract_hidden_states","num_speculative_tokens":1,"draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":[3,24,48,72,90,93]}}}'
KV_TRANSFER_CONFIG='{"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer","kv_connector_extra_config":{"shared_storage_path":"'"$HIDDEN_STATES_PATH"'","enable_cross_layers_blocks":"False","enable_separate_hidden_state_pool":"True"}}'

set --
if [[ "$SERVICE_RANK" != 0 ]]; then
  set -- --headless --data-parallel-start-rank "$SERVICE_RANK"
fi

echo "[$(date '+%F %T')] starting K3 HS service, group=$HS_GROUP rank=$SERVICE_RANK ip=$LOCAL_IP"

exec vllm serve "$MODEL_PATH" \
  --served-model-name kimi-k3-w4a8 \
  --host 0.0.0.0 \
  --port 18088 \
  --trust-remote-code \
  --quantization ascend \
  --dtype bfloat16 \
  --tokenizer-mode auto \
  --tensor-parallel-size 16 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 4 \
  --data-parallel-size-local 1 \
  --data-parallel-address "${NODE_IPS[0]}" \
  --data-parallel-rpc-port 29500 \
  --enable-expert-parallel \
  --max-num-seqs 3 \
  --max-model-len 18433 \
  --max-num-batched-tokens 18448 \
  --gpu-memory-utilization 0.98 \
  --num-gpu-blocks-override 352 \
  --cpu-offload-gb 0 \
  --kv-cache-dtype bfloat16 \
  --seed 1024 \
  --enforce-eager \
  --mamba-cache-mode none \
  --additional-config '{"enable_cpu_binding":true,"enable_flashcomm1":true,"enable_shared_expert_dp":false,"enable_fused_mc2":0,"enable_prefill_mc2":false,"recompute_scheduler_enable":false}' \
  --no-enable-chunked-prefill \
  --speculative-config "$SPECULATIVE_CONFIG" \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  --language-model-only \
  --skip-mm-profiling \
  "$@" \
  > "$LOG_DIR/vllm_node${SERVICE_RANK}.log" 2>&1
