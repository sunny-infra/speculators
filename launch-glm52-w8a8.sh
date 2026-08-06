export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export VLLM_VERSION="0.22.1"
export VLLM_ASCEND_ENABLE_NZ=0

python scripts/launch_vllm.py \
  /mnt/sfs_turbo/models/GLM5.2-W8A8  \
  --target-layer-ids 8 23 39 55 70 \
  --hidden-states-path ./hiddenstates-glm52-w8a8-test \
  -- --served-model-name '/mnt/sfs_turbo/models/ZhipuAI/GLM-5.2/' \
  --port 8088 \
  --data-parallel-size 1 \
  --tensor-parallel-size 16 \
  --seed 1024 \
  --max-num-seqs 128 \
  --max-model-len 8196 \
  --trust-remote-code \
  --gpu-memory-utilization 0.92 \
  --quantization ascend \
  --enforce-eager \
  --enable-expert-parallel