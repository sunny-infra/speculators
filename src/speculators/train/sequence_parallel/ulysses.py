"""Ulysses sequence-parallel all-to-all for ``[B, H, S, D]`` tensors."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.autograd.function import Function

from speculators.train.distributed import get_sp_group, get_sp_size


def ulysses_enabled() -> bool:
    return get_sp_size() > 1


def _all_to_all_4d(
    input_: torch.Tensor,
    scatter_idx: int,
    gather_idx: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """All-to-all on a 4D tensor.

    For Ulysses with SDPA layout ``[B, H, S, D]``:

    * ``scatter_idx=1, gather_idx=2``: ``[B, H, S/N, D] → [B, H/N, S, D]``
    * ``scatter_idx=2, gather_idx=1``: ``[B, H/N, S, D] → [B, H, S/N, D]``
    """
    if scatter_idx == gather_idx:
        raise ValueError("scatter_idx and gather_idx must differ")

    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_

    inp = input_.contiguous()
    if inp.shape[scatter_idx] % world_size != 0:
        raise ValueError(
            f"Dim {scatter_idx} size {inp.shape[scatter_idx]} not divisible by "
            f"sp world_size={world_size}"
        )
    send_list = [chunk.contiguous() for chunk in inp.chunk(world_size, dim=scatter_idx)]
    recv_list = [torch.empty_like(send_list[0]) for _ in range(world_size)]
    dist.all_to_all(recv_list, send_list, group=group)
    return torch.cat(recv_list, dim=gather_idx)


class SeqAllToAll4D(Function):
    """Differentiable 4D all-to-all (Ulysses)."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        group: dist.ProcessGroup | None,
        input_: torch.Tensor,
        scatter_idx: int,
        gather_idx: int,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        if group is None or dist.get_world_size(group) == 1:
            return input_
        return _all_to_all_4d(input_, scatter_idx, gather_idx, group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        if ctx.group is None or dist.get_world_size(ctx.group) == 1:
            return None, grad_output, None, None
        # Reverse scatter/gather
        grad_input = _all_to_all_4d(
            grad_output, ctx.gather_idx, ctx.scatter_idx, ctx.group
        )
        return None, grad_input, None, None


def seq_all_to_all(
    input_: torch.Tensor,
    scatter_idx: int,
    gather_idx: int,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Convenience wrapper around :class:`SeqAllToAll4D`."""
    if group is None:
        group = get_sp_group()
    return SeqAllToAll4D.apply(group, input_, scatter_idx, gather_idx)


def gather_document_ids(
    local_document_ids: torch.Tensor,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """All-gather 1-D document ids across the SP group → full sequence ids."""
    if group is None:
        group = get_sp_group()
    if group is None or dist.get_world_size(group) == 1:
        return local_document_ids

    world_size = dist.get_world_size(group)
    local = local_document_ids.contiguous().reshape(-1)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local, group=group)
    return torch.cat(gathered, dim=0)
