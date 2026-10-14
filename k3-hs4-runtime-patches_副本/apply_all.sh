#!/usr/bin/env bash
# Apply or verify all K3 hs4 runtime patches in dependency order.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="$SCRIPT_DIR/patches"

case "${1:-}" in
  "") MODE="" ;;
  --verify|--status) MODE="--verify" ;;
  *) echo "Usage: $0 [--verify|--status]" >&2; exit 2 ;;
esac

run_patch() {
  local script=$1
  echo "==> $script ${MODE}"
  bash "$PATCH_DIR/$script" $MODE
}

run_patch patch_kimi_k3_final_hidden.sh
run_patch patch_force_moe_alltoall.sh
run_patch patch_ascend_sampler_logprobs_mode.sh
run_patch patch_hidden_state_packed_kv_capacity.sh
run_patch patch_hidden_state_separate_kv_pool.sh
run_patch patch_hs_connector_atomic_publish.sh

echo "All K3 hs4 runtime patches completed successfully."
