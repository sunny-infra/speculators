#!/usr/bin/env bash
# Verify byte-for-byte parity with the files deployed to hs4 on 2026-09-11.

set -euo pipefail

python3 - <<'PY'
from __future__ import annotations

import hashlib
from importlib.metadata import distribution
from pathlib import Path

vllm_root = Path(distribution("vllm").locate_file(""))
ascend_root = Path(distribution("vllm-ascend").locate_file(""))

expected = {
    ascend_root / "vllm_ascend/models/kimi_k3.py":
        "e9b5017a672d2961f6e99ffa27df81fa748960098b0ea5c4ee7b5c56c0cae988",
    ascend_root / "vllm_ascend/ascend_forward_context.py":
        "7730b854fa34ef632e89e5497ebb3ae4fa0d6918d1639ac87162e1cc5d134afe",
    ascend_root / "vllm_ascend/worker/model_runner_v1.py":
        "060fd2e124e2a758d9ae5ae9b5befd67c1326a0b7258e9ec40a5af4eec5b06c1",
    vllm_root / "vllm/v1/core/kv_cache_utils.py":
        "2d73d1453f05254653f1b73be5c469098d9ec31fb2f2a7bbb1a356c27a0e5c8d",
    vllm_root / "vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py":
        "910a972fae1c0f100c6d592ca432a2dbdd4f6f1d915100d1db1e4354939602a1",
}

failed = False
for path, wanted in expected.items():
    if not path.is_file():
        print(f"MISSING  {path}")
        failed = True
        continue
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    state = "OK" if got == wanted else "DIFF"
    print(f"{state:7} {got}  {path}")
    failed |= got != wanted

if failed:
    raise SystemExit(1)
PY
