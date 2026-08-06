"""Abstraction for hidden-states transfer between vLLM and the trainer."""

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
