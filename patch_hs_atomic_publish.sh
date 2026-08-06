#!/bin/bash
# =============================================================================
# patch_hs_atomic_publish.sh
#
# Make online HS transfer reliable on nolock NFS (SFS Turbo), following the
# working approach in sunyu/speculators-main (tmp + atomic rename), adapted to
# the newer async ExampleHiddenStatesConnector and hs_connectors FileTransfer.
#
# WHAT THIS PATCHES
# -----------------
# 1) vLLM ExampleHiddenStatesConnector._write_tensors
#    BEFORE: save_file(tensors, filename)   # grows final path in place
#    AFTER:  save_file(tmp) -> os.replace(tmp, filename)
#            Final path appears only when the file is complete.
#
# 2) hs_connectors FileTransfer reader (train side)
#    BEFORE: size-stable wait (workaround for in-place writes) / flock
#    AFTER:  wait until final path appears, then load+validate with retries.
#            With (1), appearance == complete; retries cover NFS visibility.
#
# WHY
# ---
# /mnt/sfs_turbo is mounted with nolock: fcntl.flock is local-only and cannot
# sync serve→train. Upstream connector returns the path before disk write
# finishes and writes the final name in place → train sees missing/partial
# files → empty batches → loss=0.
#
# USAGE
# -----
#   # Review only (no writes):
#   bash patch_hs_atomic_publish.sh --dry-run
#
#   # Apply (edits files in place; creates .bak.<timestamp> next to each):
#   bash patch_hs_atomic_publish.sh
#
#   # Override paths if needed:
#   VLLM_CONNECTOR=/path/to/example_hidden_states_connector.py \
#   TRANSFER_PY=/path/to/hs_connectors/transfer.py \
#   bash patch_hs_atomic_publish.sh
#
# AFTER APPLYING
# --------------
# 1) Restart ALL vLLM serve DP nodes (patch is in connector process memory).
# 2) Restart train (imports FileTransfer at worker start).
# 3) Strongly recommended: set BOTH serve and train --hidden-states-path to the
#    SAME absolute directory, e.g.
#      /mnt/sfs_turbo/yc02324691/codes/speculators-sunny/hiddenstates-glm52-w8a8
#    (serve_dp4_node*.sh currently uses ./hiddenstates-glm52-w8a8)
#
# Safe to re-run: skips sections already containing the marker comments.
# =============================================================================

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default: the connector tree used by current GLM serve (see serve logs).
DEFAULT_VLLM_CONNECTOR="/mnt/sfs_turbo/yc02324691/codes/fix-gguf-test/vllm/vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py"
DEFAULT_TRANSFER_PY="${SCRIPT_DIR}/hs_connectors/src/hs_connectors/transfer.py"

VLLM_CONNECTOR="${VLLM_CONNECTOR:-$DEFAULT_VLLM_CONNECTOR}"
TRANSFER_PY="${TRANSFER_PY:-$DEFAULT_TRANSFER_PY}"

export DRY_RUN VLLM_CONNECTOR TRANSFER_PY

python3 - <<'PY'
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"
VLLM_CONNECTOR = Path(os.environ["VLLM_CONNECTOR"])
TRANSFER_PY = Path(os.environ["TRANSFER_PY"])

MARKER_WRITE = "atomic_publish_patch"
MARKER_READ = "atomic_publish_reader_patch"


def backup(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + f".bak.{time.strftime('%Y%m%d_%H%M%S')}")
    if not DRY_RUN:
        bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return bak


def write_or_preview(path: Path, new_src: str, label: str) -> None:
    if DRY_RUN:
        print(f"[dry-run] would patch {label}: {path}")
        print(f"[dry-run] new size={len(new_src)} bytes (old={path.stat().st_size})")
        return
    bak = backup(path)
    path.write_text(new_src, encoding="utf-8")
    print(f"Patched {label}: {path}")
    print(f"  backup: {bak}")


# ---------------------------------------------------------------------------
# 1) vLLM connector: atomic tmp + rename in _write_tensors
# ---------------------------------------------------------------------------

def patch_vllm_connector(path: Path) -> None:
    if not path.is_file():
        print(f"ERROR: connector not found: {path}", file=sys.stderr)
        sys.exit(1)

    src = path.read_text(encoding="utf-8")
    if MARKER_WRITE in src:
        print(f"Already patched ({MARKER_WRITE}): {path}")
        return

    # Match the current async connector body (save_file + close lock_fd).
    pattern = re.compile(
        r"(?P<indent>[ \t]*)try:\n"
        r"(?P=indent)    event\.synchronize\(\)\n"
        r"(?P=indent)    save_file\(tensors, filename\)\n"
        r"(?P=indent)finally:\n"
        r"(?P=indent)    if lock_fd is not None:\n"
        r"(?P=indent)        os\.close\(lock_fd\).*?\n",
        re.MULTILINE,
    )
    # Ensure `import time` BEFORE computing match spans (insertion shifts offsets).
    if re.search(r"(?m)^import time\b", src) is None:
        if re.search(r"(?m)^import os\b", src):
            src = re.sub(r"(?m)^(import os\b.*)$", r"\1\nimport time", src, count=1)
        else:
            src = "import time\n" + src

    match = pattern.search(src)
    if not match:
        print(
            f"ERROR: expected _write_tensors save_file block not found in {path}\n"
            "  Look for: event.synchronize(); save_file(tensors, filename)\n"
            "  vLLM version may differ — adjust this script before applying.",
            file=sys.stderr,
        )
        sys.exit(1)

    indent = match.group("indent")
    replacement = "\n".join(
        [
            f"{indent}# {MARKER_WRITE}: write temp then os.replace so readers on",
            f"{indent}# nolock NFS never observe a partially written safetensors.",
            f"{indent}# (Adapted from sunyu/speculators-main patch_hidden_states_connector.sh)",
            f"{indent}tmp_path = f\"{{filename}}.{{os.getpid()}}.{{time.time_ns()}}.tmp\"",
            f"{indent}try:",
            f"{indent}    event.synchronize()",
            f"{indent}    os.makedirs(os.path.dirname(filename) or \".\", exist_ok=True)",
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
        ]
    )

    new_src = src[: match.start()] + replacement + src[match.end() :]
    write_or_preview(path, new_src, "vLLM connector write path")


# ---------------------------------------------------------------------------
# 2) Train FileTransfer: wait for published final path, then load+validate
# ---------------------------------------------------------------------------

NEW_TRANSFER = r'''"""Abstraction for hidden-states transfer between vLLM and the trainer."""

from __future__ import annotations

import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from safetensors.torch import load_file

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable


# atomic_publish_reader_patch: with connector tmp+os.replace, the final
# safetensors path appears only after a complete write. Do not use flock
# (broken on nolock NFS). Wait for the final path, then load+validate.


def _wait_for_published_file(
    file_path: Path,
    timeout: float,
    *,
    poll_interval: float = 0.2,
) -> bool:
    """Wait until the final safetensors path exists with size > 0."""
    if timeout <= 0:
        try:
            return file_path.exists() and file_path.stat().st_size > 0
        except OSError:
            return False

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if file_path.exists() and file_path.stat().st_size > 0:
                return True
        except OSError:
            pass
        time.sleep(poll_interval)

    try:
        return file_path.exists() and file_path.stat().st_size > 0
    except OSError:
        return False


def _validate_hs(data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "hidden_states" not in data or "token_ids" not in data:
        raise ValueError(
            f"Hidden-states file missing required keys; got {sorted(data)}"
        )
    if data["token_ids"].numel() == 0:
        raise ValueError("Hidden-states token_ids is empty")
    if data["hidden_states"].numel() == 0:
        raise ValueError("Hidden-states tensor is empty")
    return data


def _load_hs_file(
    file_path: Path,
    *,
    appear_timeout: float = 0.0,
    load_retries: int = 8,
    retry_interval: float = 1.0,
) -> dict[str, torch.Tensor] | None:
    """Load a hidden-states safetensors file.

    Args:
        file_path: Path to the ``.safetensors`` file (or vLLM handle path).
        appear_timeout: Seconds to wait for the final path to appear.
            Use ``0`` for cached files (immediate miss → generate).
            Use a large value for freshly generated handles (async publish).
        load_retries: Retries for transient NFS / incomplete visibility.
        retry_interval: Sleep between load retries.
    """
    if appear_timeout > 0:
        if not _wait_for_published_file(file_path, appear_timeout):
            return None
    elif not file_path.exists():
        return None

    last_err: Exception | None = None
    for attempt in range(max(1, load_retries)):
        try:
            if not file_path.exists() or file_path.stat().st_size <= 0:
                if appear_timeout <= 0:
                    return None
                time.sleep(retry_interval)
                continue
            return _validate_hs(load_file(file_path))
        except Exception as e:  # noqa: BLE001 — retry transient NFS/read races
            last_err = e
            if attempt + 1 >= load_retries:
                raise
            time.sleep(retry_interval)

    if last_err is not None:
        raise last_err
    return None


class HiddenStatesTransfer(ABC):
    """Interface for reading hidden states produced by vLLM."""

    def setup(self) -> None:  # noqa: B027
        """Lazy initialization (safe to call from dataloader worker)."""

    @abstractmethod
    def get_cached(self, file_idx: int) -> dict[str, torch.Tensor] | None:
        """Return a previously cached sample, or ``None``."""

    @abstractmethod
    def get_generated(self, handle: str) -> dict[str, torch.Tensor] | None:
        """Retrieve a freshly generated sample by its vLLM-returned handle."""

    def cache(self, handle: str, file_idx: int) -> None:  # noqa: B027
        """Persist a generated sample to the cache location."""

    def delete(self, handle: str) -> None:  # noqa: B027
        """Clean up a generated sample (e.g. delete a temp file)."""


class HiddenStatesBackend(ABC):
    """Plugin interface for hidden-states transfer backends.

    Each backend registers itself via ``@HiddenStatesBackend.register(name)``
    and implements these four static hooks so that scripts (``train.py``,
    ``launch_vllm.py``) can discover and configure backends without hardcoding.
    """

    registry: ClassVar[dict[str, type[HiddenStatesBackend]]] = {}

    @classmethod
    def register(
        cls,
        name: str,
    ) -> Callable[[type[HiddenStatesBackend]], type[HiddenStatesBackend]]:
        def decorator(
            subclass: type[HiddenStatesBackend],
        ) -> type[HiddenStatesBackend]:
            if name in cls.registry:
                raise ValueError(f"Backend '{name}' is already registered.")
            cls.registry[name] = subclass
            return subclass

        return decorator

    @staticmethod
    @abstractmethod
    def add_train_args(parser: argparse.ArgumentParser) -> None:
        """Add backend-specific CLI arguments to ``train.py``."""
        ...

    @staticmethod
    @abstractmethod
    def add_launch_args(parser: argparse.ArgumentParser) -> None:
        """Add backend-specific CLI arguments to ``launch_vllm.py``."""
        ...

    @staticmethod
    @abstractmethod
    def from_train_args(
        args: argparse.Namespace,
        data_path: str,
    ) -> HiddenStatesTransfer:
        """Construct a :class:`HiddenStatesTransfer` from parsed train args."""
        ...

    @staticmethod
    @abstractmethod
    def build_kv_transfer_config(args: argparse.Namespace) -> dict[str, Any]:
        """Construct the ``kv_transfer_config`` dict for ``vllm serve``."""
        ...


# ---------------------------------------------------------------------------
# File-based backend (shared filesystem)
# ---------------------------------------------------------------------------


class FileTransfer(HiddenStatesTransfer):
    """File-system based hidden-states transfer (shared filesystem)."""

    # Wait for async connector to publish the final path (tmp+rename).
    GENERATED_APPEAR_TIMEOUT: float = 600.0

    def __init__(self, hidden_states_path: Path):
        self.hidden_states_path = hidden_states_path

    def get_cached(self, file_idx: int) -> dict[str, torch.Tensor] | None:
        # Cached files are already finalized — do not wait on miss.
        path = self.hidden_states_path / f"hs_{file_idx}.safetensors"
        return _load_hs_file(path, appear_timeout=0.0)

    def get_generated(self, handle: str) -> dict[str, torch.Tensor] | None:
        # API returns the handle before publish completes; wait for final path.
        return _load_hs_file(
            Path(handle), appear_timeout=self.GENERATED_APPEAR_TIMEOUT
        )

    def cache(self, handle: str, file_idx: int) -> None:
        self.hidden_states_path.mkdir(parents=True, exist_ok=True)
        target = self.hidden_states_path / f"hs_{file_idx}.safetensors"
        shutil.move(handle, target)
        lock = Path(str(handle) + ".lock")
        if lock.exists():
            lock.unlink(missing_ok=True)

    def delete(self, handle: str) -> None:
        Path(handle).unlink(missing_ok=True)
        lock = Path(str(handle) + ".lock")
        if lock.exists():
            lock.unlink(missing_ok=True)


@HiddenStatesBackend.register("file")
class FileBackend(HiddenStatesBackend):
    """Shared-filesystem backend using safetensors files."""

    @staticmethod
    def add_train_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--hidden-states-path",
            type=str,
            default=None,
            help=(
                "The path where cached hidden states files are stored. (Default: "
                "args.data_path / 'hidden_states')"
            ),
        )

    @staticmethod
    def add_launch_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--hidden-states-path",
            type=str,
            default="/tmp/hidden_states",  # noqa: S108
            help="The directory to save hidden states to. Default '/tmp/hidden_states'",
        )

    @staticmethod
    def from_train_args(
        args: argparse.Namespace,
        data_path: str,
    ) -> FileTransfer:
        hs_path = (
            Path(args.hidden_states_path)
            if args.hidden_states_path
            else Path(data_path) / "hidden_states"
        )
        return FileTransfer(hs_path)

    @staticmethod
    def build_kv_transfer_config(args: argparse.Namespace) -> dict[str, Any]:
        return {
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": args.hidden_states_path,
            },
        }
'''


def patch_transfer(path: Path) -> None:
    if not path.is_file():
        print(f"ERROR: transfer.py not found: {path}", file=sys.stderr)
        sys.exit(1)

    src = path.read_text(encoding="utf-8")
    if MARKER_READ in src:
        print(f"Already patched ({MARKER_READ}): {path}")
        return

    # Full-file replace keeps the patch reviewable and avoids fragile partial
    # edits across prior size-stable / flock experiments.
    write_or_preview(path, NEW_TRANSFER, "hs_connectors FileTransfer reader")


def main() -> None:
    print(f"DRY_RUN={DRY_RUN}")
    print(f"VLLM_CONNECTOR={VLLM_CONNECTOR}")
    print(f"TRANSFER_PY={TRANSFER_PY}")
    print("---")
    patch_vllm_connector(VLLM_CONNECTOR)
    patch_transfer(TRANSFER_PY)
    print("---")
    print("Done.")
    if DRY_RUN:
        print("Re-run without --dry-run to apply. Then restart serve + train.")
    else:
        print("Next: restart all vLLM DP nodes, then restart training.")


if __name__ == "__main__":
    main()
PY
