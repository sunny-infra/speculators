#!/usr/bin/env bash
# Make vLLM's packed KV-cache admission/capacity math mirror its physical
# packed allocator.  This is required by extract_hidden_states deployments
# where a large HiddenStateCacheSpec shares the cache plan with K3 MLA/SWA
# layers: the legacy generic formula can charge that hidden-state page once
# per layer slot even though _get_kv_cache_config_packed allocates it once.

set -euo pipefail

MODE=apply
case "${1:-}" in
    "") ;;
    --dry-run) MODE=dry-run ;;
    --verify|--status) MODE=verify ;;
    *) echo "Usage: $0 [--dry-run|--verify|--status]" >&2; exit 2 ;;
esac

export MODE
[[ -z "${VLLM_KV_CACHE_UTILS_FILE:-}" ]] || export VLLM_KV_CACHE_UTILS_FILE

python3 - <<'PY'
from __future__ import annotations

import difflib
import os
import shutil
import tempfile
import time
from importlib.metadata import distribution
from pathlib import Path
from typing import NoReturn


configured = os.environ.get("VLLM_KV_CACHE_UTILS_FILE")
if configured:
    path = Path(configured)
else:
    site_root = Path(distribution("vllm").locate_file(""))
    path = site_root / "vllm/v1/core/kv_cache_utils.py"

mode = os.environ["MODE"]
concurrency_marker = "K3_DSPARK_PACKED_KV_CONCURRENCY"
admission_marker = "K3_DSPARK_PACKED_KV_ADMISSION"
native_hidden_marker = "K3_DSPARK_HIDDEN_CACHE_NATIVE_PAGE"

concurrency_anchor = """    num_layer_per_group = max(
        len(group.layer_names) for group in kv_cache_config.kv_cache_groups
    )
"""
concurrency_patch = """    # K3_DSPARK_PACKED_KV_CONCURRENCY: mirror the physical packed
    # allocator.  Every physical packed block contains one slot from each
    # page-size bucket.  All logical groups draw distinct block IDs from the
    # shared pool, so a request consumes the sum of each group's page count.
    if _use_packed_kv_cache_config(vllm_config, kv_cache_config.kv_cache_groups):
        blocks_per_request = sum(
            cdiv(
                group.kv_cache_spec.max_memory_usage_bytes(vllm_config),
                group.kv_cache_spec.page_size_bytes,
            )
            for group in kv_cache_config.kv_cache_groups
        )
        return kv_cache_config.num_blocks / blocks_per_request

    num_layer_per_group = max(
        len(group.layer_names) for group in kv_cache_config.kv_cache_groups
    )
"""

admission_anchor = """    if not kv_cache_groups:
        return 0

    if len(kv_cache_groups) == 1 and isinstance(
"""
admission_patch = """    if not kv_cache_groups:
        return 0

    # K3_DSPARK_PACKED_KV_ADMISSION: use the same bytes-per-block layout as
    # _get_kv_cache_config_packed.  The prior generic hybrid formula multiplied
    # a one-layer HiddenStateCacheSpec by the largest attention group size,
    # rejecting configurations that the physical packed allocator can hold.
    if _use_packed_kv_cache_config(vllm_config, kv_cache_groups):
        bytes_per_block = _pool_bytes_per_block(vllm_config, kv_cache_groups)
        blocks_needed = sum(
            cdiv(
                group.kv_cache_spec.max_memory_usage_bytes(vllm_config),
                group.kv_cache_spec.page_size_bytes,
            )
            for group in kv_cache_groups
        )
        required = bytes_per_block * blocks_needed
        logger.info_once(
            "Packed KV cache admission: %d bytes/block x %d blocks = %.2f GiB",
            bytes_per_block,
            blocks_needed,
            required / (1 << 30),
        )
        return required

    if len(kv_cache_groups) == 1 and isinstance(
"""

hidden_page_anchor = """    # Add hidden-state layers back with page aligned to the common page.
    if hidden_specs:
        common_page = get_uniform_page_size([g.kv_cache_spec for g in groups])
        for name, spec in hidden_specs.items():
            per_token = spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
            new_bs = max(common_page // per_token, 1)
            aligned = replace(spec, block_size=new_bs, page_size_padded=common_page)
            groups.append(KVCacheGroupSpec([name], aligned))

    return groups
"""
hidden_page_patch = """    # K3_DSPARK_HIDDEN_CACHE_NATIVE_PAGE: the packed allocator supports
    # independent page-size buckets.  Preserve the hidden-state cache's native
    # block/page size when cross-layer packing is explicitly enabled; aligning
    # its very large per-token state to an attention page shrinks the hidden
    # block to only a few tokens and charges every global attention slot for
    # thousands of hidden-only block IDs.
    if hidden_specs:
        kv_transfer_config = vllm_config.kv_transfer_config
        extra_config = (
            kv_transfer_config.kv_connector_extra_config
            if kv_transfer_config is not None
            else {}
        )
        use_native_hidden_page = (
            str(extra_config.get("enable_cross_layers_blocks", "False")).lower()
            == "true"
        )
        if use_native_hidden_page:
            for name, spec in hidden_specs.items():
                groups.append(KVCacheGroupSpec([name], spec))
        else:
            common_page = get_uniform_page_size([g.kv_cache_spec for g in groups])
            for name, spec in hidden_specs.items():
                per_token = (
                    spec.num_kv_heads
                    * spec.head_size
                    * get_dtype_size(spec.dtype)
                )
                new_bs = max(common_page // per_token, 1)
                aligned = replace(
                    spec, block_size=new_bs, page_size_padded=common_page
                )
                groups.append(KVCacheGroupSpec([name], aligned))

    return groups
"""


def fail(message: str) -> NoReturn:
    raise SystemExit(f"ERROR: {message}")


def validate_python(source: str) -> None:
    try:
        compile(source, str(path), "exec")
    except SyntaxError as error:
        fail(f"Python syntax validation failed for {path}: {error}")


def verify(source: str) -> None:
    concurrency_section = source.split(concurrency_marker, 1)[1].split(
        "num_layer_per_group = max(", 1
    )[0]
    checks = {
        "packed concurrency marker exactly once": source.count(concurrency_marker) == 1,
        "packed admission marker exactly once": source.count(admission_marker) == 1,
        "native hidden-page marker exactly once": source.count(native_hidden_marker) == 1,
        # A later Ascend-safe separate-pool patch intentionally adds another
        # branch with the same return expression. Scope this check to the
        # packed marker instead of imposing a fragile global count.
        "packed concurrency branch present": (
            "return kv_cache_config.num_blocks / blocks_per_request"
            in concurrency_section
        ),
        "admission branch exactly once": source.count(
            "Packed KV cache admission: %d bytes/block x %d blocks = %.2f GiB"
        ) == 1,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        fail(f"packed KV capacity patch verification failed for {path}: " + ", ".join(failures))
    validate_python(source)


if not path.is_file():
    fail(f"vLLM kv_cache_utils.py not found: {path}")

source = path.read_text(encoding="utf-8")
validate_python(source)

if (
    concurrency_marker in source
    and admission_marker in source
    and native_hidden_marker in source
):
    verify(source)
    print(f"Verified existing packed KV capacity patch: {path}")
    raise SystemExit(0)

if mode == "verify":
    fail(f"packed KV capacity patch is not installed: {path}")

patched = source
if concurrency_marker not in patched:
    if patched.count(concurrency_anchor) != 1:
        fail("expected exactly one get_max_concurrency anchor; installed vLLM differs")
    patched = patched.replace(concurrency_anchor, concurrency_patch, 1)
if admission_marker not in patched:
    if patched.count(admission_anchor) != 1:
        fail("expected exactly one packed admission anchor; installed vLLM differs")
    patched = patched.replace(admission_anchor, admission_patch, 1)
if native_hidden_marker not in patched:
    if patched.count(hidden_page_anchor) != 1:
        fail("expected exactly one hidden-state page alignment anchor; installed vLLM differs")
    patched = patched.replace(hidden_page_anchor, hidden_page_patch, 1)
verify(patched)

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
temporary_path: Path | None = None
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

verify(path.read_text(encoding="utf-8"))
print(f"Patched packed KV admission/capacity accounting: {path}")
print(f"  backup: {backup}")
print("Restart vLLM before testing the long-context hidden-state extractor.")
PY
