#!/usr/bin/env bash
# Preserve vLLM's configured logprobs mode when constructing AscendSampler.

set -euo pipefail

MODE=apply
case "${1:-}" in
  "") ;;
  --dry-run) MODE=dry-run ;;
  --verify|--status) MODE=verify ;;
  *) echo "Usage: $0 [--dry-run|--verify|--status]" >&2; exit 2 ;;
esac
export MODE
[[ -z "${ASCEND_MODEL_RUNNER_FILE:-}" ]] || export ASCEND_MODEL_RUNNER_FILE

python3 - <<'PY'
from __future__ import annotations

import difflib
import os
import shutil
import tempfile
import time
from importlib.metadata import distribution
from pathlib import Path

configured = os.environ.get("ASCEND_MODEL_RUNNER_FILE")
if configured:
    path = Path(configured)
else:
    root = Path(distribution("vllm-ascend").locate_file(""))
    path = root / "vllm_ascend/worker/model_runner_v1.py"

mode = os.environ["MODE"]
marker = "K3_DSPARK_ASCEND_SAMPLER_LOGPROBS_MODE"
old = "        self.sampler = AscendSampler()\n"
new = """        # K3_DSPARK_ASCEND_SAMPLER_LOGPROBS_MODE: preserve raw_logits.
        self.sampler = AscendSampler(
            logprobs_mode=self.model_config.logprobs_mode
        )
"""

if not path.is_file():
    raise SystemExit(f"ERROR: Ascend model runner not found: {path}")

source = path.read_text(encoding="utf-8")
if marker in source:
    if source.count(marker) != 1 or source.count(new) != 1:
        raise SystemExit(f"ERROR: malformed sampler patch: {path}")
    compile(source, str(path), "exec")
    print(f"Verified existing AscendSampler logprobs patch: {path}")
    raise SystemExit(0)

if mode == "verify":
    raise SystemExit(f"ERROR: AscendSampler logprobs patch is not installed: {path}")
if source.count(old) != 1:
    raise SystemExit(
        f"ERROR: expected one AscendSampler constructor in {path}; "
        "the installed vLLM-Ascend version differs"
    )

patched = source.replace(old, new, 1)
compile(patched, str(path), "exec")

if mode == "dry-run":
    print(f"[dry-run] patch applies cleanly: {path}")
    print("".join(difflib.unified_diff(
        source.splitlines(keepends=True),
        patched.splitlines(keepends=True),
        fromfile=str(path),
        tofile=f"{path} (patched)",
    )), end="")
    raise SystemExit(0)

backup = path.with_suffix(
    path.suffix + f".bak.{time.strftime('%Y%m%d_%H%M%S')}.{os.getpid()}"
)
shutil.copy2(path, backup)
temporary_path = None
try:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(patched)
        handle.flush()
        os.fsync(handle.fileno())
    shutil.copymode(path, temporary_path)
    os.replace(temporary_path, path)
finally:
    if temporary_path is not None:
        temporary_path.unlink(missing_ok=True)

print(f"Patched AscendSampler logprobs mode: {path}")
print(f"  backup: {backup}")
PY
