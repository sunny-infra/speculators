"""Tests for chunked Eagle3 logits/loss path."""

import torch
import torch.nn as nn

from speculators.models.eagle3.metrics import compute_metrics, compute_metrics_chunked
from speculators.models.metrics import kl_div_loss


class _IdentityNorm(nn.Module):
    def forward(self, x):
        return x


def test_chunked_metrics_matches_full():
    torch.manual_seed(0)
    b, s, h, v = 1, 16, 8, 32
    hidden = torch.randn(b, s, h, requires_grad=True)
    verifier_hs = torch.randn(b, s, h)
    lm_head = nn.Linear(h, v, bias=False)
    norm = _IdentityNorm()
    loss_mask = torch.ones(b, s, dtype=torch.bool)
    prev_a = torch.ones(b, s, dtype=torch.bool)
    prev_b = prev_a.clone()

    with torch.no_grad():
        targets = lm_head(verifier_hs)  # reuse shape; not tied to draft

    # Full path
    logits = lm_head(norm(hidden))
    full_loss, full_metrics, _full_denom = compute_metrics(
        logits,
        targets,
        loss_mask,
        prev_a,
        ttt_step=1,
        ttt_step_loss_decay=1.0,
        loss_config={"kl_div": (kl_div_loss, 1.0)},
    )

    def target_fn(start, end):
        with torch.no_grad():
            return targets[:, start:end]

    chunk_loss, chunk_metrics, argmax_ids, _chunk_denom = compute_metrics_chunked(
        hidden,
        norm,
        lm_head,
        target_fn,
        loss_mask,
        prev_b,
        ttt_step=1,
        ttt_step_loss_decay=1.0,
        loss_config={"kl_div": (kl_div_loss, 1.0)},
        chunk_size=5,
        norm_output=True,  # hidden already "normed" via Identity
    )

    assert torch.allclose(full_loss, chunk_loss, atol=1e-4, rtol=1e-4)
    assert argmax_ids.shape == (1, s)
    assert torch.equal(
        argmax_ids, torch.argmax(logits.detach(), dim=-1)
    )
    assert abs(full_metrics["loss_1_sum"].item() - chunk_metrics["loss_1_sum"].item()) < 1e-4
