#!/usr/bin/env bash
# Add an opt-in A3 MoE communication fallback to the installed vLLM-Ascend.
#
# The release_0.26.0_0817 image used by the Kimi-K3 D-Spark extractor selects
# the MC2/DispatchV2 path for small profile and request batches.  On the current
# 4-node A3 topology that path fails in MoeDistributeDispatchV2 with an MTE
# invalid-GM/cross-device-timeout error.  This patch leaves default serving
# behaviour unchanged and only forces the existing ALLTOALL implementation
# when VLLM_ASCEND_FORCE_MOE_ALLTOALL=1 is exported before vLLM starts.
#
# Usage on every inference node:
#   bash patch_force_moe_alltoall.sh --dry-run
#   bash patch_force_moe_alltoall.sh
#   bash patch_force_moe_alltoall.sh --verify

set -euo pipefail

MODE=apply
case "${1:-}" in
    "") ;;
    --dry-run) MODE=dry-run ;;
    --verify|--status) MODE=verify ;;
    *) echo "Usage: $0 [--dry-run|--verify|--status]" >&2; exit 2 ;;
esac

export MODE
if [[ -n "${ASCEND_FORWARD_CONTEXT_FILE:-}" ]]; then
    export ASCEND_FORWARD_CONTEXT_FILE
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

configured_path = os.environ.get("ASCEND_FORWARD_CONTEXT_FILE")
if configured_path:
    path = Path(configured_path)
else:
    site_root = Path(distribution("vllm-ascend").locate_file(""))
    path = site_root / "vllm_ascend/ascend_forward_context.py"

mode = os.environ["MODE"]
marker = "K3_DSPARK_FORCE_MOE_ALLTOALL"
env_name = "VLLM_ASCEND_FORCE_MOE_ALLTOALL"

if not path.is_file():
    raise SystemExit(f"ERROR: vLLM-Ascend forward-context source not found: {path}")

function_head = """def _select_a3_moe_comm_method(
    num_tokens: int,
    mc2_tokens_capacity: int,
    vllm_config: VllmConfig,
    is_draft_model: bool = False,
) -> MoECommType:
    skip_fused_mc2 = is_draft_model and _moe_quant_mismatch
"""
function_replacement = """def _select_a3_moe_comm_method(
    num_tokens: int,
    mc2_tokens_capacity: int,
    vllm_config: VllmConfig,
    is_draft_model: bool = False,
) -> MoECommType:
    # K3_DSPARK_FORCE_MOE_ALLTOALL: opt-in fallback for A3 runtimes whose
    # MoeDistributeDispatchV2 small-token path fails during profile or prefill.
    if _FORCE_MOE_ALLTOALL:
        logger.warning_once(
            "VLLM_ASCEND_FORCE_MOE_ALLTOALL=1: forcing A3 MoE communication "
            "to ALLTOALL; MC2/DispatchV2 is disabled for this process."
        )
        return MoECommType.ALLTOALL

    skip_fused_mc2 = is_draft_model and _moe_quant_mismatch
"""

constants_head = """_MC2_TOKENS_PER_RANK_LIMIT = 512
"""
constants_replacement = """_MC2_TOKENS_PER_RANK_LIMIT = 512
_FORCE_MOE_ALLTOALL_ENV = "VLLM_ASCEND_FORCE_MOE_ALLTOALL"


def _read_force_moe_alltoall() -> bool:
    raw = os.environ.get(_FORCE_MOE_ALLTOALL_ENV, "0").strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"{_FORCE_MOE_ALLTOALL_ENV} must be one of "
        f"0/1/false/true/no/yes/off/on, got {raw!r}"
    )


_FORCE_MOE_ALLTOALL = _read_force_moe_alltoall()
"""


def verify_patched(source: str) -> None:
    checks = {
        "capability marker": source.count(marker) == 1,
        "environment switch": env_name in source,
        "strict environment parser": source.count(constants_replacement) == 1,
        "ALLTOALL return": "return MoECommType.ALLTOALL" in source,
        "os import": "import os\n" in source,
        "unpatched A3 function head absent": function_head not in source,
        "patched A3 function head present once": source.count(function_replacement) == 1,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise SystemExit(
            "ERROR: force-ALLTOALL patch verification failed for "
            f"{path}: {', '.join(failures)}"
        )
    compile(source, str(path), "exec")


source = path.read_text(encoding="utf-8")
if marker in source:
    verify_patched(source)
    print(f"Verified existing force-ALLTOALL patch ({marker}): {path}")
    raise SystemExit(0)

if mode == "verify":
    raise SystemExit(f"ERROR: force-ALLTOALL patch is not installed: {path}")

count = source.count(function_head)
if count != 1:
    raise SystemExit(
        f"ERROR: expected exactly one A3 selector in {path}, found {count}; "
        "the installed vLLM-Ascend version differs, so this patch was not applied"
    )
if source.count(constants_head) != 1:
    raise SystemExit(
        f"ERROR: expected exactly one MC2 token-limit constant in {path}; "
        "the installed vLLM-Ascend version differs, so this patch was not applied"
    )

patched = source
if "import os\n" not in patched:
    import_needle = "import math\n"
    if patched.count(import_needle) != 1:
        raise SystemExit(f"ERROR: cannot add os import safely in {path}")
    patched = patched.replace(import_needle, "import math\nimport os\n", 1)
patched = patched.replace(function_head, function_replacement, 1)
patched = patched.replace(constants_head, constants_replacement, 1)
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
print(f"Patched opt-in A3 force-ALLTOALL selector: {path}")
print(f"  backup: {backup}")
print("Restart vLLM with VLLM_ASCEND_FORCE_MOE_ALLTOALL=1 to enable it.")
PY
