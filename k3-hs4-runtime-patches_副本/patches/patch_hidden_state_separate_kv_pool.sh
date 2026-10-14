#!/usr/bin/env bash
# Add a non-packed, page-size-aware KV-cache planner for the
# extract_hidden_states connector.
#
# vLLM's packed planner is efficient, but the Ascend runner in the pinned
# release does not consume KVCacheTensor.offset/block_stride.  Enabling it
# would therefore allocate the full packed backing once per descriptor.  This
# patch keeps the proven Ascend allocation/reshape path: normal attention and
# Mamba slots retain their existing shared tensors, while HiddenStateCacheSpec
# slots receive independent tensors at their native (large) page size.

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
markers = (
    "K3_DSPARK_SEPARATE_HIDDEN_POOL_SELECT",
    "K3_DSPARK_SEPARATE_HIDDEN_POOL_NATIVE_PAGE",
    "K3_DSPARK_SEPARATE_HIDDEN_POOL_CONCURRENCY",
    "K3_DSPARK_SEPARATE_HIDDEN_POOL_BYTES",
    "K3_DSPARK_SEPARATE_HIDDEN_POOL_PLANNER",
    "K3_DSPARK_SEPARATE_HIDDEN_POOL_ADMISSION",
)


def fail(message: str) -> NoReturn:
    raise SystemExit(f"ERROR: {message}")


def validate_python(source: str) -> None:
    try:
        compile(source, str(path), "exec")
    except SyntaxError as error:
        fail(f"Python syntax validation failed for {path}: {error}")


def replace_once(source: str, anchor: str, replacement: str, label: str) -> str:
    count = source.count(anchor)
    if count != 1:
        fail(f"expected exactly one {label} anchor, found {count}; installed vLLM differs")
    return source.replace(anchor, replacement, 1)


def verify(source: str) -> None:
    checks = {f"{marker} exactly once": source.count(marker) == 1 for marker in markers}
    checks.update(
        {
            "selector key exactly twice": source.count(
                'extra_config.get("enable_separate_hidden_state_pool", "False")'
            ) == 1,
            "non-packed descriptors have no stride": source.count(
                "KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)"
            ) >= 1,
            "planner selected before packed": source.index(
                "K3_DSPARK_SEPARATE_HIDDEN_POOL_PLANNER"
            ) < source.index("elif _use_packed_kv_cache_config"),
            "packed disabled for separate pool": (
                "if _use_separate_hidden_state_pool(vllm_config, kv_cache_groups):\n"
                "        return False"
            ) in source,
        }
    )
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        fail(
            f"separate hidden-state pool verification failed for {path}: "
            + ", ".join(failures)
        )
    validate_python(source)


if not path.is_file():
    fail(f"vLLM kv_cache_utils.py not found: {path}")

source = path.read_text(encoding="utf-8")
validate_python(source)

present = [marker in source for marker in markers]
if all(present):
    verify(source)
    print(f"Verified existing separate hidden-state KV pool patch: {path}")
    raise SystemExit(0)
if any(present):
    fail("partial separate hidden-state pool patch detected; restore a known backup first")
if mode == "verify":
    fail(f"separate hidden-state KV pool patch is not installed: {path}")

# The three accounting/native-page fixes are an explicit prerequisite.  They
# remain useful for other packed layouts; this patch adds a safer Ascend-only
# path selected by a different connector flag.
for prerequisite in (
    "K3_DSPARK_PACKED_KV_CONCURRENCY",
    "K3_DSPARK_PACKED_KV_ADMISSION",
    "K3_DSPARK_HIDDEN_CACHE_NATIVE_PAGE",
):
    if source.count(prerequisite) != 1:
        fail(
            f"missing prerequisite marker {prerequisite}; apply "
            "patch_hidden_state_packed_kv_capacity.sh first"
        )

patched = source

selector_anchor = '''def _use_packed_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> bool:
'''
selector_patch = '''# K3_DSPARK_SEPARATE_HIDDEN_POOL_SELECT: the pinned Ascend runner does
# not implement packed KVCacheTensor offset/block_stride views.  Opt into a
# page-size-aware collection of ordinary tensors instead.  Each group still
# draws unique IDs from the one global BlockPool, exactly like the standard
# hybrid layout.
def _use_separate_hidden_state_pool(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> bool:
    kv_transfer_config = vllm_config.kv_transfer_config
    extra_config = (
        kv_transfer_config.kv_connector_extra_config
        if kv_transfer_config is not None
        else {}
    )
    enabled = (
        str(
            extra_config.get("enable_separate_hidden_state_pool", "False")
        ).lower()
        == "true"
    )
    has_hidden = any(
        isinstance(group.kv_cache_spec, HiddenStateCacheSpec)
        and bool(group.layer_names)
        for group in kv_cache_groups
    )
    has_regular = any(
        not isinstance(group.kv_cache_spec, HiddenStateCacheSpec)
        and bool(group.layer_names)
        for group in kv_cache_groups
    )
    return enabled and has_hidden and has_regular


def _separate_hidden_state_pool_slots(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> list[tuple[int, list[str]]]:
    """Return ordinary physical tensor slots without mixing hidden/regular.

    Layers from distinct regular groups may share one tensor slot because
    their block IDs come from the same global allocator and cannot collide.
    Hidden-state slots are bucketed independently even if a future model has
    an equal page size, preventing Ascend's attention/Mamba branches from
    aliasing the extractor tensor.
    """
    regular_groups = [
        group
        for group in kv_cache_groups
        if not isinstance(group.kv_cache_spec, HiddenStateCacheSpec)
        and group.layer_names
    ]
    hidden_groups = [
        group
        for group in kv_cache_groups
        if isinstance(group.kv_cache_spec, HiddenStateCacheSpec)
        and group.layer_names
    ]
    assert regular_groups and hidden_groups

    slots: list[tuple[int, list[str]]] = []
    for groups in (regular_groups, hidden_groups):
        buckets = _bucket_layers_by_page_size(groups)
        for page_size, shared_slots in buckets.items():
            for shared_by in shared_slots:
                assert page_size > 0 and shared_by
                slots.append((page_size, shared_by))
    all_layers = [name for group in kv_cache_groups for name in group.layer_names]
    slotted_layers = [name for _, shared_by in slots for name in shared_by]
    assert sorted(all_layers) == sorted(slotted_layers)
    return slots


def _use_packed_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> bool:
    if _use_separate_hidden_state_pool(vllm_config, kv_cache_groups):
        return False
'''
patched = replace_once(patched, selector_anchor, selector_patch, "packed selector")

native_anchor = '''        use_native_hidden_page = (
            str(extra_config.get("enable_cross_layers_blocks", "False")).lower()
            == "true"
        )
'''
native_patch = '''        # K3_DSPARK_SEPARATE_HIDDEN_POOL_NATIVE_PAGE: both the packed
        # upstream layout and the Ascend-safe separate layout can retain the
        # extractor's native page.  The latter does not rely on offset/stride.
        use_native_hidden_page = (
            _use_separate_hidden_state_pool(
                vllm_config,
                [*groups, *(KVCacheGroupSpec([name], spec) for name, spec in hidden_specs.items())],
            )
            or str(extra_config.get("enable_cross_layers_blocks", "False")).lower()
            == "true"
        )
'''
patched = replace_once(patched, native_anchor, native_patch, "native hidden page")

concurrency_anchor = '''    # K3_DSPARK_PACKED_KV_CONCURRENCY: mirror the physical packed
'''
concurrency_patch = '''    # K3_DSPARK_SEPARATE_HIDDEN_POOL_CONCURRENCY: one request draws
    # distinct global block IDs for every logical KV group.  Each ID indexes
    # every ordinary tensor slot, so request consumption is the sum of group
    # page counts, as for the packed layout.
    if _use_separate_hidden_state_pool(
        vllm_config, kv_cache_config.kv_cache_groups
    ):
        blocks_per_request = sum(
            cdiv(
                group.kv_cache_spec.max_memory_usage_bytes(vllm_config),
                group.kv_cache_spec.page_size_bytes,
            )
            for group in kv_cache_config.kv_cache_groups
        )
        return kv_cache_config.num_blocks / blocks_per_request

    # K3_DSPARK_PACKED_KV_CONCURRENCY: mirror the physical packed
'''
patched = replace_once(
    patched, concurrency_anchor, concurrency_patch, "concurrency"
)

pool_bytes_anchor = '''    if _use_packed_kv_cache_config(vllm_config, kv_cache_groups):
        # buckets = {page_size: [[layer_names], [layer_names], ...]}
        buckets = _bucket_layers_by_page_size(kv_cache_groups)
        return sum(ps * len(slots) for ps, slots in buckets.items())
'''
pool_bytes_patch = '''    # K3_DSPARK_SEPARATE_HIDDEN_POOL_BYTES: every global block owns one
    # page in every ordinary physical tensor slot.
    if _use_separate_hidden_state_pool(vllm_config, kv_cache_groups):
        return sum(
            page_size
            for page_size, _ in _separate_hidden_state_pool_slots(kv_cache_groups)
        )
    if _use_packed_kv_cache_config(vllm_config, kv_cache_groups):
        # buckets = {page_size: [[layer_names], [layer_names], ...]}
        buckets = _bucket_layers_by_page_size(kv_cache_groups)
        return sum(ps * len(slots) for ps, slots in buckets.items())
'''
patched = replace_once(
    patched, pool_bytes_anchor, pool_bytes_patch, "pool bytes"
)

planner_anchor = '''    elif _use_packed_kv_cache_config(vllm_config, kv_cache_groups):
        # DeepSeek V4 uses the packed layout by default. Other multi-group
'''
planner_patch = '''    # K3_DSPARK_SEPARATE_HIDDEN_POOL_PLANNER: retain ordinary, unstrided
    # descriptors so the pinned Ascend runner can allocate/reshape them with
    # its existing attention, Mamba and cache-only code paths.
    elif _use_separate_hidden_state_pool(vllm_config, kv_cache_groups):
        slots = _separate_hidden_state_pool_slots(kv_cache_groups)
        bytes_per_block = sum(page_size for page_size, _ in slots)
        num_blocks = may_override_num_blocks(
            vllm_config, available_memory // bytes_per_block
        )
        kv_cache_tensors = [
            KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
            for page_size, shared_by in slots
        ]
    elif _use_packed_kv_cache_config(vllm_config, kv_cache_groups):
        # DeepSeek V4 uses the packed layout by default. Other multi-group
'''
patched = replace_once(patched, planner_anchor, planner_patch, "planner")

admission_anchor = '''    # K3_DSPARK_PACKED_KV_ADMISSION: use the same bytes-per-block layout as
'''
admission_patch = '''    # K3_DSPARK_SEPARATE_HIDDEN_POOL_ADMISSION: match the physical
    # ordinary-tensor planner.  A request consumes the sum of logical-group
    # block IDs from the one global BlockPool.
    if _use_separate_hidden_state_pool(vllm_config, kv_cache_groups):
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
            "Separate hidden-state KV pool admission: %d bytes/block x %d "
            "blocks = %.2f GiB",
            bytes_per_block,
            blocks_needed,
            required / (1 << 30),
        )
        return required

    # K3_DSPARK_PACKED_KV_ADMISSION: use the same bytes-per-block layout as
'''
patched = replace_once(patched, admission_anchor, admission_patch, "admission")

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
print(f"Patched Ascend-safe separate hidden-state KV pool: {path}")
print(f"  backup: {backup}")
print("Restart vLLM before testing the long-context hidden-state extractor.")
PY
