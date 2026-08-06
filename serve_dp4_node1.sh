#!/bin/bash
set -euo pipefail
LOG_DIR="/mnt/sfs_turbo/yc02324691/codes/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/serve_dp4_node1_${TS}.log"
exec > "$LOG" 2>&1

cd /mnt/sfs_turbo/yc02324691/codes/speculators-sunny

nic_name="eth0"
local_ip="33.182.141.60"
node0_ip="33.182.141.211"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export VLLM_VERSION="0.22.1"
export VLLM_ASCEND_ENABLE_NZ=0

python scripts/launch_vllm.py \
    /mnt/sfs_turbo/models/ZhipuAI/GLM-5.2/ \
    --target-layer-ids 8 23 39 55 70 \
    --hidden-states-path ./hiddenstates-glm52-w8a8 \
    -- --served-model-name '/mnt/sfs_turbo/models/ZhipuAI/GLM-5.2/' \
    --port 8078 \
    --tensor-parallel-size 16 \
    --data-parallel-size 4 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 1 \
    --data-parallel-address $node0_ip \
    --data-parallel-rpc-port 12345 \
    --enable-expert-parallel \
    --seed 1024 \
    --max-num-seqs 128 \
    --max-model-len 8196 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --hf-overrides '{"use_index_cache": true}' \
    --headless \
    --enforce-eager
