"""Metrics and loss functions for Eagle3 draft model."""

from collections.abc import Callable
from functools import partial

import torch

from speculators.models.metrics import (
    LossConfig,
    _EPS,
    compound_loss,
    compute_accuracy_single_step,
    exp_loss_decay,
    kl_div_loss,
)


def align_for_step(
    logits: torch.Tensor,  # shape: [1, total_seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, total_seq_len, draft_vocab_size]
    loss_mask: torch.Tensor | None,  # shape: [1, total_seq_len]
    prev_correct: torch.Tensor | None,  # shape: [1, total_seq_len]
    ttt_step: int,
):
    """Align logits, targets, loss_mask, and prev_correct for a given ttt_step.

    There are no target values for the last ttt_step tokens, so we mask them out
    before computing the loss/accuracy. Likewise, there are no logits for the first
    ttt_step tokens, so we mask them out.
    This is equivalent to shifting the target values by ttt_step + 1 to the left
    which puts them in the correct position for the generated tokens.
    e.g.
        indices of targets = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        indices of logits for ttt_step_0 = [1, 2, 3, 4, 5, 6, 7, 8, 9] # no shift
        indices of logits for ttt_step_1 = [2, 3, 4, 5, 6, 7, 8, 9, 10] # shift by 1
        indices of logits for ttt_step_2 = [3, 4, 5, 6, 7, 8, 9, 10, 11] # shift by 2
    The indices for the loss_mask need to be kept in line with the targets indices
    """
    logits = logits[:, :-ttt_step] if ttt_step > 0 else logits
    # shape: [1, total_seq_len - ttt_step, draft_vocab_size]
    targets = targets[:, ttt_step:]
    # shape: [1, total_seq_len - ttt_step, draft_vocab_size]
    if loss_mask is not None:
        loss_mask = loss_mask[:, ttt_step:]
        # shape: [1, total_seq_len - ttt_step]
    if prev_correct is not None:
        # Align with draft starts
        prev_correct = prev_correct[:, :-ttt_step] if ttt_step > 0 else prev_correct
        # shape: [1, total_seq_len - ttt_step]
    return logits, targets, loss_mask, prev_correct


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None,
    prev_correct: torch.Tensor | None,
    ttt_step: int,
    ttt_step_loss_decay: float,
    loss_config: LossConfig | None = None,
) -> tuple[torch.Tensor, dict]:
    """Compute metrics for a given ttt_step.

    Args:
        logits: The logits for the current ttt_step.
        targets: The targets for the current ttt_step.
        loss_mask: The loss mask for the current ttt_step.
        prev_correct: The previous correct predictions for the current ttt_step.
        ttt_step: The current ttt_step.
        ttt_step_loss_decay: The loss decay for the current ttt_step.
        loss_config: Mapping of ``{name: (loss_fn, weight)}``.

    Effects:
        Modifies prev_correct in place.

    Returns:
        Loss value and metrics dictionary.
    """
    if loss_config is None:
        loss_config = {"kl_div": (kl_div_loss, 1.0)}
    s_logits, s_targets, s_loss_mask, s_prev_correct = align_for_step(
        logits, targets, loss_mask, prev_correct, ttt_step
    )

    seq_len = s_logits.shape[1]
    if s_loss_mask is None:
        s_loss_mask = torch.ones(1, seq_len, device=s_logits.device, dtype=torch.bool)

    pos_idx = torch.full(
        (1, seq_len), ttt_step, device=s_logits.device, dtype=torch.long
    )

    s_loss, term_losses = compound_loss(
        s_logits,
        s_targets,
        s_loss_mask,
        pos_idx,
        loss_config=loss_config,
        decay_fn=partial(exp_loss_decay, gamma=ttt_step_loss_decay),
    )

    pred_ids = torch.argmax(s_logits, dim=-1)
    target_ids = torch.argmax(s_targets, dim=-1)

    full_correct, full_total, cond_correct, cond_total = compute_accuracy_single_step(
        pred_ids, target_ids, s_loss_mask, s_prev_correct
    )

    ones = torch.tensor(1.0, device=s_loss.device)
    s_metrics = {}
    s_metrics[f"loss_{ttt_step}_sum"] = s_loss.detach().clone()
    s_metrics[f"loss_{ttt_step}_total"] = ones
    for term_name, term_val in term_losses.items():
        s_metrics[f"{term_name}_{ttt_step}_sum"] = term_val
        s_metrics[f"{term_name}_{ttt_step}_total"] = ones.clone()
    s_metrics[f"full_acc_{ttt_step}_sum"] = full_correct
    s_metrics[f"full_acc_{ttt_step}_total"] = full_total
    s_metrics[f"cond_acc_{ttt_step}_sum"] = cond_correct
    s_metrics[f"cond_acc_{ttt_step}_total"] = cond_total

    return s_loss, s_metrics


def compute_metrics_chunked(
    hidden_states: torch.Tensor,  # [1, total_seq_len, hidden]
    norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    target_fn: Callable[[int, int], torch.Tensor],
    loss_mask: torch.Tensor | None,
    prev_correct: torch.Tensor | None,
    ttt_step: int,
    ttt_step_loss_decay: float,
    loss_config: LossConfig | None = None,
    chunk_size: int = 512,
    *,
    norm_output: bool = False,
) -> tuple[torch.Tensor, dict, torch.Tensor]:
    """Chunked logits/loss path that never materializes full ``S × V`` tensors.

    Teacher targets are produced on-the-fly via ``target_fn(start, end)``.
    Draft logits are computed per chunk from ``hidden_states``.

    Returns:
        ``(loss, metrics, argmax_ids)`` where ``argmax_ids`` has shape
        ``[1, total_seq_len]`` (built chunk-wise under ``no_grad`` for the
        next TTT input).
    """
    if loss_config is None:
        loss_config = {"kl_div": (kl_div_loss, 1.0)}

    device = hidden_states.device
    total_seq_len = hidden_states.shape[1]
    seq_len = total_seq_len - ttt_step
    if seq_len <= 0:
        zeros = torch.tensor(0.0, device=device)
        empty_ids = torch.argmax(
            lm_head(norm(hidden_states[:, :1])), dim=-1
        ).new_zeros(1, total_seq_len)
        return zeros, {}, empty_ids

    s_loss_mask = (
        loss_mask[:, ttt_step:]
        if loss_mask is not None
        else torch.ones(1, seq_len, device=device, dtype=torch.bool)
    )
    s_prev_correct = (
        prev_correct[:, :-ttt_step]
        if (ttt_step > 0 and prev_correct is not None)
        else prev_correct
    )

    decay_fn = partial(exp_loss_decay, gamma=ttt_step_loss_decay)
    loss_weighted_sum = torch.tensor(0.0, device=device)
    loss_denom_sum = torch.tensor(0.0, device=device)
    full_correct_sum = torch.tensor(0.0, device=device)
    full_total_sum = torch.tensor(0.0, device=device)
    cond_correct_sum = torch.tensor(0.0, device=device)
    cond_total_sum = torch.tensor(0.0, device=device)
    term_sums: dict[str, torch.Tensor] = {}

    # Loss uses logits at [ttt_step + chunk) aligned with targets [ttt_step + chunk)
    # via align_for_step: logits[:, :-ttt] ↔ targets[:, ttt:].
    # Equivalent local view: draft hidden at positions [0, seq_len) when ttt=0,
    # and at [0, seq_len) for logits side when ttt>0 means hidden[:, 0:seq_len]
    # paired with targets from target_fn(ttt_step, ttt_step+seq_len).
    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        # Logits side after align_for_step uses hidden[:, :seq_len] (drop last ttt)
        h_start = chunk_start
        h_end = chunk_end
        c_hidden = hidden_states[:, h_start:h_end]
        if norm_output:
            # Caller already applied norm to hidden_states when norm_output=True
            c_logits = lm_head(c_hidden)
        else:
            c_logits = lm_head(norm(c_hidden))

        c_targets = target_fn(ttt_step + chunk_start, ttt_step + chunk_end)
        c_loss_mask = s_loss_mask[:, chunk_start:chunk_end]
        c_prev = (
            s_prev_correct[:, chunk_start:chunk_end]
            if s_prev_correct is not None
            else None
        )
        pos_idx = torch.full(
            (1, chunk_end - chunk_start),
            ttt_step,
            device=device,
            dtype=torch.long,
        )

        # Weighted compound loss accumulated as sum/count for exact global mean
        for name, (fn, weight) in loss_config.items():
            c_elem = fn(c_logits, c_targets) * c_loss_mask.to(c_logits.dtype)
            if decay_fn is not None:
                c_elem = c_elem * decay_fn(
                    pos_idx.to(c_elem.dtype), elementwise_loss=c_elem
                )
            loss_weighted_sum = loss_weighted_sum + weight * c_elem.sum()
            if len(loss_config) > 1:
                key = f"{name}_loss"
                term_sums[key] = term_sums.get(key, torch.tensor(0.0, device=device)) + (
                    c_elem.detach().sum()
                )

        loss_denom_sum = loss_denom_sum + c_loss_mask.to(torch.float32).sum()

        c_pred = torch.argmax(c_logits, dim=-1)
        c_tgt_ids = torch.argmax(c_targets, dim=-1)
        fc, ft, cc, ct = compute_accuracy_single_step(
            c_pred, c_tgt_ids, c_loss_mask, c_prev
        )
        full_correct_sum = full_correct_sum + fc
        full_total_sum = full_total_sum + ft
        cond_correct_sum = cond_correct_sum + cc
        cond_total_sum = cond_total_sum + ct
        del c_logits, c_targets, c_hidden

    s_loss = loss_weighted_sum / (loss_denom_sum + _EPS)
    ones = torch.tensor(1.0, device=device)
    s_metrics: dict = {
        f"loss_{ttt_step}_sum": s_loss.detach().clone(),
        f"loss_{ttt_step}_total": ones,
        f"full_acc_{ttt_step}_sum": full_correct_sum,
        f"full_acc_{ttt_step}_total": full_total_sum,
        f"cond_acc_{ttt_step}_sum": cond_correct_sum,
        f"cond_acc_{ttt_step}_total": cond_total_sum,
    }
    for term_name, term_val in term_sums.items():
        # Convert accumulated sum into a mean matching compute_metrics
        s_metrics[f"{term_name}_{ttt_step}_sum"] = term_val / (loss_denom_sum + _EPS)
        s_metrics[f"{term_name}_{ttt_step}_total"] = ones.clone()

    # Next-step input ids: full-sequence argmax without keeping logits
    with torch.no_grad():
        argmax_chunks = []
        for cs in range(0, total_seq_len, chunk_size):
            ce = min(cs + chunk_size, total_seq_len)
            h = hidden_states[:, cs:ce]
            if norm_output:
                c_logits = lm_head(h)
            else:
                c_logits = lm_head(norm(h))
            argmax_chunks.append(torch.argmax(c_logits, dim=-1))
            del c_logits
        argmax_ids = torch.cat(argmax_chunks, dim=1)

    return s_loss, s_metrics, argmax_ids
