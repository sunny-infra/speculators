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
  --data-path /mnt/sfs_turbo/yc02324691/codes/speculators-sunny/data/glm52-perfect/prepare_data_60w \
  --vllm-endpoint http://33.182.141.211:8078/v1 \
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
  --hidden-states-path /mnt/sfs_turbo/yc02324691/codes/speculators-sunny/hiddenstates-glm52-w8a8 \
  --draft-attn-impl sdpa \
  --fsdp-shard \
  --dflash-decay-gamma 10 \
  --train-data-ratio 0.95 \
  --noise-std 0.01 \
  --full-attention-indices 0 1 2 3 4
