"""Unit tests for Eagle3 windowed document+SWA attention (no dense O(S²) mask)."""

import math

import pytest
import torch
from torch.nn.attention.flex_attention import create_mask

from speculators.models.eagle3.attention import create_combined_mask_mod
from speculators.models.eagle3.window_attention import (
    Eagle3WindowedCache,
    document_segment_bounds,
    resolve_window_attn_implementation,
    uses_window_attn_kernel,
    windowed_document_attention,
)


def _dense_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bool_mask: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Reference eager attention with a dense boolean mask ``[1,1,S,KV]``."""
    scores = torch.matmul(query, key.transpose(-2, -1)) * scaling
    scores = scores.masked_fill(~bool_mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(probs, value)


def _build_dense_bool_mask(
    document_ids: torch.Tensor,
    total_seq_len: int,
    n_steps: int,
    sliding_window: int | None,
    device: torch.device,
) -> torch.Tensor:
    mask_mod = create_combined_mask_mod(
        document_ids, total_seq_len, sliding_window=sliding_window
    )
    return create_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=total_seq_len,
        KV_LEN=total_seq_len * n_steps,
        device=device,
    )


@pytest.mark.parametrize("sliding_window", [None, 4])
@pytest.mark.parametrize("n_steps", [1, 2, 3])
@pytest.mark.parametrize("use_sdpa", [False, True])
def test_windowed_attention_matches_dense_mask(sliding_window, n_steps, use_sdpa):
    """Window kernel must match create_combined_mask_mod + dense eager attn."""
    torch.manual_seed(0)
    b, h, s, d = 1, 2, 8, 4
    device = torch.device("cpu")
    # Packed docs: lengths [3, 3, 2]
    document_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2], device=device)

    query = torch.randn(b, h, s, d, device=device)
    base_k = torch.randn(b, h, s, d, device=device)
    base_v = torch.randn(b, h, s, d, device=device)
    draft_ks = [torch.randn(b, h, s, d, device=device) for _ in range(n_steps - 1)]
    draft_vs = [torch.randn(b, h, s, d, device=device) for _ in range(n_steps - 1)]

    scaling = 1.0 / math.sqrt(d)
    out = windowed_document_attention(
        query,
        base_k,
        base_v,
        draft_ks,
        draft_vs,
        document_ids=document_ids,
        sliding_window=sliding_window,
        scaling=scaling,
        use_sdpa=use_sdpa,
        chunk_size=3,
        num_key_value_groups=1,
    )

    key = torch.cat([base_k, *draft_ks], dim=-2)
    value = torch.cat([base_v, *draft_vs], dim=-2)
    dense_mask = _build_dense_bool_mask(
        document_ids, s, n_steps, sliding_window, device
    )
    ref = _dense_attn(query, key, value, dense_mask, scaling)

    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_windowed_attention_no_full_s_by_kv_mask_allocation():
    """Local masks stay O(chunk · (W + ttt)), never allocate S×(ttt·S)."""
    torch.manual_seed(1)
    b, h, s, d = 1, 1, 64, 8
    w = 8
    n_draft = 2
    document_ids = torch.zeros(s, dtype=torch.long)
    query = torch.randn(b, h, s, d)
    base_k = torch.randn(b, h, s, d)
    base_v = torch.randn(b, h, s, d)
    draft_ks = [torch.randn(b, h, s, d) for _ in range(n_draft)]
    draft_vs = [torch.randn(b, h, s, d) for _ in range(n_draft)]

    # Monkeypatch local mask builder to record shapes
    from speculators.models.eagle3 import window_attention as wa

    shapes: list[tuple[int, ...]] = []
    orig = wa._build_local_bool_mask

    def _tracking(*args, **kwargs):
        mask = orig(*args, **kwargs)
        shapes.append(tuple(mask.shape))
        return mask

    wa._build_local_bool_mask = _tracking  # type: ignore[assignment]
    try:
        windowed_document_attention(
            query,
            base_k,
            base_v,
            draft_ks,
            draft_vs,
            document_ids=document_ids,
            sliding_window=w,
            scaling=1.0 / math.sqrt(d),
            use_sdpa=False,
            chunk_size=w,
        )
    finally:
        wa._build_local_bool_mask = orig  # type: ignore[assignment]

    assert shapes, "expected at least one local mask"
    dense_kv = s * (1 + n_draft)
    for shape in shapes:
        # [1, 1, C, base_len + n_draft*C] — never a full S × (ttt·S) mask
        assert shape[0] == 1 and shape[1] == 1
        assert shape[2] <= w
        # Windowed base span is at most W+C-1, plus n_draft identity blocks of C
        assert shape[3] <= (w + w - 1) + n_draft * w
        assert shape[2] * shape[3] < s * dense_kv


def test_eagle3_windowed_cache_stores_base_and_drafts_separately():
    cache = Eagle3WindowedCache()
    k0 = torch.randn(1, 2, 4, 8)
    v0 = torch.randn(1, 2, 4, 8)
    rk, rv = cache.update(k0, v0, layer_idx=0)
    assert rk is k0 and rv is v0
    assert cache.ttt_depth(0) == 1
    assert len(cache.draft_keys[0]) == 0

    k1 = torch.randn(1, 2, 4, 8)
    v1 = torch.randn(1, 2, 4, 8)
    cache.update(k1, v1, layer_idx=0)
    assert cache.ttt_depth(0) == 2
    assert len(cache.draft_keys[0]) == 1
    # Base unchanged; no concatenated ttt·S tensor
    assert cache.base_keys[0].shape[-2] == 4
    assert cache.draft_keys[0][0].shape[-2] == 4


def test_document_segment_bounds():
    ids = torch.tensor([0, 0, 1, 1, 1, -1, -1, 2])
    assert document_segment_bounds(ids) == [(0, 2), (2, 5), (5, 7), (7, 8)]


@pytest.mark.parametrize(
    ("impl", "kernel", "expected"),
    [
        ("simple_flex_attention", "auto", "simple_flex_attention"),
        ("sdpa", "auto", "window_sdpa"),
        ("eager", "auto", "window_eager"),
        ("sdpa", "window_sdpa", "window_sdpa"),
        ("sdpa", "dense", "sdpa"),
        ("eager", "dense", "eager"),
    ],
)
def test_resolve_window_attn_implementation(impl, kernel, expected):
    assert resolve_window_attn_implementation(impl, kernel) == expected
    assert uses_window_attn_kernel(expected) == expected.startswith("window_")
