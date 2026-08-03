# Train Eagle-3 Online at 32k / 64k (NPU / SDPA)

This guide covers **online** Eagle-3 training when flex attention is unavailable
(e.g. Ascend NPU) and you need true long context (32k → 64k), not just packing
many short samples.

## Architecture

```
Verifier NPU pool (vLLM)  --HS files-->  Train NPU pool (DP × SP)
                                          SP ranks shard one sequence
                                          Ulysses all-to-all on attention heads
                                          window SDPA (no dense O(S²) mask)
```

| Role | Recommendation |
|------|----------------|
| Verifier | Separate devices; `max_model_len ≥` train seq; TP as needed for memory |
| Train | `WORLD_SIZE = DP × SP`; for 32k use `SP=4`, for 64k use `SP=8` |
| Attention | `--draft-attn-impl sdpa --draft-attn-kernel window_sdpa` |
| Loss memory | `--logits-chunk-size 512` (or 1024) |
| Storage | Fast shared disk for per-sample hidden-state safetensors |

## Required flags

```bash
torchrun --nproc_per_node 4 scripts/train.py \
  --draft-attn-impl sdpa \
  --draft-attn-kernel window_sdpa \
  --sliding-window 2048 \
  --total-seq-len 32768 \
  --sp-size 4 \
  --logits-chunk-size 512 \
  --ttt-steps 3 \
  --on-missing generate \
  ...
```

Constraints:

- `WORLD_SIZE % sp_size == 0`
- `num_attention_heads` and `num_key_value_heads` divisible by `sp_size`
- `--sp-size > 1` is **incompatible with `--fsdp-shard`** (use DDP)
- Do **not** use `simple_flex_attention` or `draft_attn_kernel=dense` with SP

## Data prep

```bash
python scripts/prepare_data.py \
  --model "$MODEL" \
  --data "$DATASET" \
  --output "$OUTPUT_DIR" \
  --seq-length 32768   # or 65536
```

## Ready-to-run example

See [`examples/train/eagle3_qwen3_8b_online_32k_sp4.sh`](../../../examples/train/eagle3_qwen3_8b_online_32k_sp4.sh).

For 64k, set `SEQ_LENGTH=65536`, `SP_SIZE=8`, and `NUM_TRAIN_GPUS=8` (or
`DP>1` with `WORLD_SIZE` a multiple of 8).

## What each piece does

1. **Window SDPA** — document + sliding-window attention without materializing
   dense `[S, KV]` masks; TTT uses a windowed KV cache instead of
   `ttt_steps × S` concat.
2. **Chunked logits** — teacher/draft `S×V` tensors are computed in chunks.
3. **Ulysses SP** — each rank holds `S/sp_size` tokens; attention all-to-alls
   to head-parallel full sequence, then gathers back.
4. **Online HS** — vLLM still produces hidden states on demand; train ranks in
   an SP group each read the full HS file then slice (v1). Optimize to
   slice-only reads if I/O becomes the bottleneck.

## Verifier caveats

`launch_vllm.py` forces `--no-enable-chunked-prefill` for the hidden-states
connector. Long prompts may need higher TP / memory utilization on the
verifier pool. If verifier OOM persists, evaluate making the connector
compatible with chunked prefill.

## Verification checklist (run on the training server)

### 1. Unit tests (CPU / single process)

```bash
cd speculators-sy
pytest tests/unit/models/test_eagle3_window_attention.py \
       tests/unit/models/test_eagle3_chunked_metrics.py \
       tests/unit/train/test_sequence_parallel_shard.py \
       tests/unit/train/test_draft_config_init.py -q
```

Expect: window attention matches dense mask on short sequences; chunked loss
matches full logits; SP pad/shard keeps absolute `position_ids`.

### 2. Single-card window SDPA smoke (16k, no SP)

```bash
# After prepare_data with --seq-length 16384 and a live or cached HS path:
torchrun --standalone --nproc_per_node 1 scripts/train.py \
  --verifier-name-or-path "$MODEL" \
  --data-path "$OUTPUT_DIR" \
  --draft-attn-impl sdpa \
  --draft-attn-kernel window_sdpa \
  --sliding-window 2048 \
  --total-seq-len 16384 \
  --logits-chunk-size 512 \
  --ttt-steps 3 \
  --epochs 1 --max-steps 5 \
  --on-missing generate --on-generate delete \
  --vllm-endpoint "http://localhost:8000/v1" \
  --save-path "$OUTPUT_DIR/ckpt_16k_sp1"
```

Pass criteria: no OOM on dense-mask allocation; loss is finite; step completes.

### 3. Multi-card Ulysses SP smoke (32k, SP=4)

```bash
bash examples/train/eagle3_qwen3_8b_online_32k_sp4.sh
# or equivalent torchrun with --sp-size 4 --total-seq-len 32768
```

Pass criteria:

- logs show `sp_size=4`
- each train rank local seq ≈ `32768/4`
- 1 epoch (or `--max-steps 10`) completes without HCCL/attention shape errors
- optional: compare `SP=1` vs `SP=2` loss on a tiny fixed seed run (relative gap small)

### 4. Scale to 64k

Same as step 3 with `--total-seq-len 65536 --sp-size 8` and 8 train devices
(`WORLD_SIZE` multiple of 8). Verifier `max_model_len >= 65536`.

### 5. Negative checks

```bash
# Should fail fast:
torchrun --nproc_per_node 2 scripts/train.py ... --sp-size 2 --fsdp-shard
torchrun --nproc_per_node 2 scripts/train.py ... --sp-size 2 --draft-attn-impl simple_flex_attention
```
