#!/bin/bash
# Online Eagle3 long-sequence training (32k) with Ulysses sequence parallel.
#
# Topology (example, 8 NPUs on one node):
#   - Verifier pool: 2 NPUs for Ascend vLLM (TP/DP as needed)
#   - Train pool:    4 NPUs with --sp-size 4  (DP=1 × SP=4)
#   - For 64k: use --total-seq-len 65536 --sp-size 8 (8 train NPUs)
#
# Requirements:
#   - --draft-attn-impl sdpa  (window kernel; no flex attention)
#   - verifier max_model_len >= SEQ_LENGTH
#   - fast shared storage for on-the-fly HS files
#   - attention / KV head counts divisible by SP_SIZE
#
# Ascend note: set ASCEND_RT_VISIBLE_DEVICES (or device env for your stack)
# instead of CUDA_VISIBLE_DEVICES when running on NPU.

set -euo pipefail

# ============ Configuration ============
MODEL="Qwen/Qwen3-8B"
DATASET="sharegpt"
OUTPUT_DIR="./output_eagle3_32k"
VLLM_PORT=8000
DRAFT_VOCAB_SIZE=32000
MAX_SAMPLES=5000
SEQ_LENGTH=32768
SP_SIZE=4
LOGITS_CHUNK_SIZE=512
SLIDING_WINDOW=2048
EPOCHS=2
LR=1e-4
TTT_STEPS=3

# Separate verifier / train device pools (adjust for your cluster)
VLLM_DEVICES="0,1"
TRAIN_DEVICES="2,3,4,5"
NUM_TRAIN_GPUS=4
# =======================================

if (( NUM_TRAIN_GPUS % SP_SIZE != 0 )); then
  echo "NUM_TRAIN_GPUS ($NUM_TRAIN_GPUS) must be divisible by SP_SIZE ($SP_SIZE)" >&2
  exit 1
fi

echo "=== Step 1: Preparing data (seq-length=${SEQ_LENGTH}) ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

echo "=== Step 2: Launching vLLM verifier (max context >= ${SEQ_LENGTH}) ==="
# launch_vllm.py currently forces --no-enable-chunked-prefill for the HS
# connector. Give the verifier enough memory / TP for long prompts.
CUDA_VISIBLE_DEVICES="$VLLM_DEVICES" python scripts/launch_vllm.py "$MODEL" \
    -- --tensor-parallel-size 2 \
       --port "$VLLM_PORT" \
       --gpu-memory-utilization 0.90 \
       --max-model-len "$SEQ_LENGTH" &
VLLM_PID=$!

cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM server ready."

echo "=== Step 3: Train with Ulysses SP=${SP_SIZE}, seq=${SEQ_LENGTH} ==="
CUDA_VISIBLE_DEVICES="$TRAIN_DEVICES" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --sp-size "$SP_SIZE" \
    --draft-attn-impl sdpa \
    --draft-attn-kernel window_sdpa \
    --sliding-window "$SLIDING_WINDOW" \
    --logits-chunk-size "$LOGITS_CHUNK_SIZE" \
    --ttt-steps "$TTT_STEPS" \
    --on-missing generate \
    --on-generate delete

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
echo "For 64k: SEQ_LENGTH=65536 SP_SIZE=8 NUM_TRAIN_GPUS=8 (same flags)."
