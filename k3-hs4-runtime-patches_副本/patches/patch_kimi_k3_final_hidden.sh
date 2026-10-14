#!/usr/bin/env bash
# Patch the vLLM-Ascend Kimi-K3 target so extract_hidden_states exports layer
# id 93 after output AttnRes materialization but before the final RMSNorm.
#
# D-Spark training reconstructs teacher logits with the BF16 verifier's final
# norm and lm_head.  The unpatched runtime appends raw layer-93 hidden_states
# before the pending block_residual is merged, which has the right shape but
# the wrong numerical meaning.
#
# Usage on every inference node (the script creates a timestamped backup):
#   bash patch_kimi_k3_final_hidden.sh --dry-run
#   bash patch_kimi_k3_final_hidden.sh
#   bash patch_kimi_k3_final_hidden.sh --verify
#
# Override the installed source path when needed:
#   K3_MODEL_FILE=/path/to/vllm_ascend/models/kimi_k3.py \
#     bash patch_kimi_k3_final_hidden.sh

set -euo pipefail

MODE=apply
case "${1:-}" in
    "") ;;
    --dry-run) MODE=dry-run ;;
    --verify|--status) MODE=verify ;;
    *) echo "Usage: $0 [--dry-run|--verify|--status]" >&2; exit 2 ;;
esac

export MODE
if [[ -n "${K3_MODEL_FILE:-}" ]]; then
    export K3_MODEL_FILE
fi

python3 - <<'PY'
from __future__ import annotations

import difflib
import os
import shutil
import tempfile
import time
from importlib.metadata import distribution
from pathlib import Path

configured_path = os.environ.get("K3_MODEL_FILE")
if configured_path:
    path = Path(configured_path)
else:
    site_root = Path(distribution("vllm-ascend").locate_file(""))
    path = site_root / "vllm_ascend/models/kimi_k3.py"
mode = os.environ["MODE"]
marker = "K3_DSPARK_FINAL_HIDDEN_PRE_NORM"

if not path.is_file():
    raise SystemExit(f"ERROR: Kimi-K3 runtime source not found: {path}")

raw_capture = """            if not self.dspark_aux_capture_materialized and (layer_idx + 1) in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states)
"""
raw_capture_replacement = """            if (
                not self.dspark_aux_capture_materialized
                and (layer_idx + 1) in self.aux_hidden_state_layers
                and (layer_idx + 1) != self.config.num_hidden_layers
            ):
                aux_hidden_states.append(hidden_states)
"""

finalize = """        hidden_states = _apply_attention_residual(
            hidden_states,
            block_residual,
            self.output_attn_res_proj,
            self.output_attn_res_norm,
        )
        hidden_states = self.norm(hidden_states)
"""
finalize_replacement = """        hidden_states = _apply_attention_residual(
            hidden_states,
            block_residual,
            self.output_attn_res_proj,
            self.output_attn_res_norm,
        )
        # K3_DSPARK_FINAL_HIDDEN_PRE_NORM: layer id num_hidden_layers is the
        # output-AttnRes-materialized residual stream before final RMSNorm.
        if (
            not self.dspark_aux_capture_materialized
            and self.config.num_hidden_layers in self.aux_hidden_state_layers
        ):
            aux_hidden_states.append(hidden_states)
        hidden_states = self.norm(hidden_states)
"""


def verify_patched(source: str) -> None:
    checks = {
        "capability marker": source.count(marker) == 1,
        "unpatched raw final-layer capture absent": raw_capture not in source,
        "patched raw capture present once": source.count(raw_capture_replacement) == 1,
        "patched final AttnRes capture present once": source.count(finalize_replacement) == 1,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise SystemExit(
            "ERROR: K3 final-hidden patch verification failed for "
            f"{path}: {', '.join(failures)}"
        )
    compile(source, str(path), "exec")


source = path.read_text(encoding="utf-8")
if marker in source:
    verify_patched(source)
    print(f"Verified existing K3 final-hidden patch ({marker}): {path}")
    raise SystemExit(0)

if mode == "verify":
    raise SystemExit(f"ERROR: K3 final-hidden patch is not installed: {path}")

for label, needle in (("raw layer-93 capture", raw_capture), ("final AttnRes block", finalize)):
    count = source.count(needle)
    if count != 1:
        raise SystemExit(
            f"ERROR: expected exactly one {label} in {path}, found {count}; "
            "the installed vLLM-Ascend version differs, so this patch was not applied"
        )

patched = source.replace(raw_capture, raw_capture_replacement, 1)
patched = patched.replace(finalize, finalize_replacement, 1)
compile(patched, str(path), "exec")
verify_patched(patched)

if mode == "dry-run":
    print(f"[dry-run] patch applies cleanly: {path}")
    print(
        "".join(
            difflib.unified_diff(
                source.splitlines(keepends=True),
                patched.splitlines(keepends=True),
                fromfile=str(path),
                tofile=f"{path} (patched)",
            )
        ),
        end="",
    )
    raise SystemExit(0)

backup = path.with_suffix(
    path.suffix + f".bak.{time.strftime('%Y%m%d_%H%M%S')}.{os.getpid()}"
)
shutil.copy2(path, backup)
temporary_path = None
try:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        temporary_file.write(patched)
        temporary_file.flush()
        os.fsync(temporary_file.fileno())
    shutil.copymode(path, temporary_path)
    os.replace(temporary_path, path)
finally:
    if temporary_path is not None:
        temporary_path.unlink(missing_ok=True)
verify_patched(path.read_text(encoding="utf-8"))
print(f"Patched Kimi-K3 final hidden capture: {path}")
print(f"  backup: {backup}")
print("Restart the vLLM process before generating any training hidden states.")
PY
