
MASTER_ADDR="xxx"
MASTER_PORT="29101"
NNODES=1
NPROC_PER_NODE=16
NODE_RANK=0
LOCAL_IP="xxx"
NIC_NAME="xxx"

export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=600
export HCCL_CONNECT_TIMEOUT=3600
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
  --speculator-type dspark \
  --num-layers 5 \
  --block-size 8 \
  --data-path xxx \
  --vllm-endpoint  http://xxx:xx/v1 \
  --save-path xxx \
  --epochs 3 \
  --lr 0.0002 \
  --scheduler-type cosine \
  --total-seq-len 65536 \
  --draft-arch glm5 \
  --draft-hidden-act silu \
  --target-layer-ids 2 22 38 58 74 \
  --max-anchors 800 \
  --markov-rank 256 \
  --enable-confidence-head \
  --confidence-head-with-markov \
  --loss-fn '{"ce": 0.3, "tv": 0.7}' \
  --confidence-head-alpha 0.6 \
  --checkpoint-freq 0.1 \
  --on-missing generate \
  --on-generate delete \
  --seed 42 \
  --log-freq 10 \
  --prefetch-factor 6 \
  --num-workers 10 \
  --request-timeout 3600 \
  --max-retries 8 \
  --trust-remote-code \
  --hidden-states-path xxx \
  --draft-attn-impl sdpa \
  --fsdp-shard \
  --dflash-decay-gamma 10 \
  --train-data-ratio 0.99 \
  --noise-std 0.01 \
  --full-attention-indices 0 1 2 3 4

