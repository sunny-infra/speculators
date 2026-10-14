set -euo pipefail

MASTER_ADDR="xxx"
MASTER_PORT="29501"
NNODES=1
NPROC_PER_NODE=16
NODE_RANK=0
LOCAL_IP="xxx"
NIC_NAME="xxx"

export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=512
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=1800
export HCCL_EVENT_TIMEOUT=1800
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=0

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
  --verifier-name-or-path xxx \
  --from-pretrained xxx \
  --speculator-type dspark \
  --block-size 8 \
  --data-path xxxx \
  --vllm-endpoint http://xxx:xxx/v1 \
  --save-path xxx \
  --epochs 5 \
  --lr 0.00001 \
  --scheduler-type cosine \
  --total-seq-len 65536 \
  --max-anchors 512 \
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
  --prefetch-factor 2 \
  --num-workers 2 \
  --request-timeout 1800 \
  --max-retries 8 \
  --trust-remote-code \
  --hidden-states-path xxx \
  --draft-attn-impl sdpa \
  --fsdp-shard \
  --dflash-decay-gamma 10 \
  --train-data-ratio 0.98 \
  --noise-std 0.01 \
  --scheduler-warmup-ratio 0.04
