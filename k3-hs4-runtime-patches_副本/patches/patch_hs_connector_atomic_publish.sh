#!/usr/bin/env bash
# Publish hidden-state safetensors via a temporary file and atomic rename.

set -euo pipefail

MODE=apply
case "${1:-}" in
  "") ;;
  --dry-run) MODE=dry-run ;;
  --verify|--status) MODE=verify ;;
  *) echo "Usage: $0 [--dry-run|--verify|--status]" >&2; exit 2 ;;
esac
export MODE
[[ -z "${VLLM_CONNECTOR_FILE:-}" ]] || export VLLM_CONNECTOR_FILE

python3 - <<'PY'
from __future__ import annotations

import difflib
import os
import re
import shutil
import tempfile
import time
from importlib.metadata import distribution
from pathlib import Path

configured = os.environ.get("VLLM_CONNECTOR_FILE")
if configured:
    path = Path(configured)
else:
    root = Path(distribution("vllm").locate_file(""))
    path = root / "vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py"

mode = os.environ["MODE"]
marker = "atomic_publish_patch"
if not path.is_file():
    raise SystemExit(f"ERROR: ExampleHiddenStatesConnector not found: {path}")

source = path.read_text(encoding="utf-8")
if marker in source:
    required = ("save_file(tensors, tmp_path)", "os.replace(tmp_path, filename)")
    if source.count(marker) != 1 or any(item not in source for item in required):
        raise SystemExit(f"ERROR: malformed atomic-publish patch: {path}")
    compile(source, str(path), "exec")
    print(f"Verified existing connector atomic-publish patch: {path}")
    raise SystemExit(0)

if mode == "verify":
    raise SystemExit(f"ERROR: connector atomic-publish patch is not installed: {path}")

patched = source
if re.search(r"(?m)^import time\b", patched) is None:
    if re.search(r"(?m)^import os\b", patched):
        patched = re.sub(r"(?m)^(import os\b.*)$", r"\1\nimport time", patched, count=1)
    else:
        patched = "import time\n" + patched

pattern = re.compile(
    r"(?P<indent>[ \t]*)try:\n"
    r"(?P=indent)    event\.synchronize\(\)\n"
    r"(?P=indent)    save_file\(tensors, filename\)\n"
    r"(?P=indent)finally:\n"
    r"(?P=indent)    if lock_fd is not None:\n"
    r"(?P=indent)        os\.close\(lock_fd\).*?\n",
    re.MULTILINE,
)
match = pattern.search(patched)
if not match:
    raise SystemExit(
        f"ERROR: expected connector write block not found in {path}; "
        "the installed vLLM version differs"
    )

indent = match.group("indent")
replacement = "\n".join([
    f"{indent}# {marker}: write temp then os.replace so readers on",
    f"{indent}# nolock NFS never observe a partially written safetensors.",
    f"{indent}# (Adapted from sunyu/speculators-main patch_hidden_states_connector.sh)",
    f'{indent}tmp_path = f"{{filename}}.{{os.getpid()}}.{{time.time_ns()}}.tmp"',
    f"{indent}try:",
    f"{indent}    event.synchronize()",
    f'{indent}    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)',
    f"{indent}    save_file(tensors, tmp_path)",
    f"{indent}    os.replace(tmp_path, filename)",
    f"{indent}finally:",
    f"{indent}    if os.path.exists(tmp_path):",
    f"{indent}        try:",
    f"{indent}            os.remove(tmp_path)",
    f"{indent}        except OSError:",
    f"{indent}            pass",
    f"{indent}    if lock_fd is not None:",
    f"{indent}        os.close(lock_fd)  # releases LOCK_EX",
    "",
])
patched = patched[:match.start()] + replacement + patched[match.end():]
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

print(f"Patched connector atomic publish: {path}")
print(f"  backup: {backup}")
PY
