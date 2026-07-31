"""Memory-efficient document + sliding-window attention for Eagle3 (sdpa/eager).

Avoids materializing dense ``O(S × KV)`` attention masks.  Instead, each query
chunk only attends to a local window of base (TTT step-0) keys plus the
per-step draft diagonals required by Eagle3 TTT — matching
``create_combined_mask_mod`` semantics.

Also provides :class:`Eagle3WindowedCache`, which stores base K/V and per-step
draft K/V separately so TTT never concatenates a ``ttt_steps · S`` KV tensor
for the score matrix (attended set size stays ``≈ W + ttt_steps``).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from speculators.models.attention import ALL_ATTENTION_FUNCTIONS


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for GQA (same as transformers.models.llama.modeling_llama)."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def document_segment_bounds(document_ids: torch.Tensor) -> list[tuple[int, int]]:
    """Return half-open ``[start, end)`` spans of contiguous equal document ids.

    Padding tokens (``document_ids == -1``) form their own segments and are
    skipped by callers that check the id value.
    """
    if document_ids.ndim != 1:
        raise ValueError(f"document_ids must be 1-D, got shape {tuple(document_ids.shape)}")
    n = document_ids.numel()
    if n == 0:
        return []
    # Boundaries where the id changes
    ids = document_ids.tolist()
    bounds: list[tuple[int, int]] = []
    start = 0
    for i in range(1, n):
        if ids[i] != ids[i - 1]:
            bounds.append((start, i))
            start = i
    bounds.append((start, n))
    return bounds


class Eagle3WindowedCache:
    """TTT KV cache: base sequence + per-step draft KVs (no ``ttt·S`` concat).

    ``update`` returns only the current-step K/V.  Window attention reconstructs
    the attended set from ``base_*`` and ``draft_*`` via the cache reference
    passed through attention kwargs.
    """

    is_eagle3_windowed_cache: bool = True

    def __init__(self) -> None:
        self.base_keys: dict[int, torch.Tensor] = {}
        self.base_values: dict[int, torch.Tensor] = {}
        self.draft_keys: dict[int, list[torch.Tensor]] = {}
        self.draft_values: dict[int, list[torch.Tensor]] = {}

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,  # noqa: ARG002
        **_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx not in self.base_keys:
            self.base_keys[layer_idx] = key_states
            self.base_values[layer_idx] = value_states
            self.draft_keys[layer_idx] = []
            self.draft_values[layer_idx] = []
        else:
            self.draft_keys[layer_idx].append(key_states)
            self.draft_values[layer_idx].append(value_states)
        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx not in self.base_keys:
            return 0
        return int(self.base_keys[layer_idx].shape[-2])

    def ttt_depth(self, layer_idx: int = 0) -> int:
        """Number of TTT steps stored (1 = base only)."""
        if layer_idx not in self.base_keys:
            return 0
        return 1 + len(self.draft_keys[layer_idx])

    def __len__(self) -> int:
        return len(self.base_keys)


def _build_local_bool_mask(
    q_start: int,
    chunk_len: int,
    kv_start: int,
    base_len: int,
    n_draft: int,
    document_ids: torch.Tensor,
    sliding_window: int | None,
    device: torch.device,
) -> torch.Tensor:
    """Boolean attend mask ``[1, 1, C, base_len + n_draft * C]`` for one query chunk.

    Base block: causal + same-document + optional sliding window.
    Each draft block: identity (diagonal draft tokens from prior/current TTT steps).
    """
    c = chunk_len
    q_pos = torch.arange(q_start, q_start + c, device=device)  # [C]
    base_pos = torch.arange(kv_start, kv_start + base_len, device=device)  # [base_len]

    # Causal within base
    causal = q_pos.unsqueeze(1) >= base_pos.unsqueeze(0)  # [C, base_len]
    if sliding_window is not None:
        causal = causal & (base_pos.unsqueeze(0) > q_pos.unsqueeze(1) - sliding_window)

    q_docs = document_ids[q_pos]
    kv_docs = document_ids[base_pos]
    same_doc = (q_docs.unsqueeze(1) == kv_docs.unsqueeze(0)) & (q_docs.unsqueeze(1) != -1)
    base_mask = causal & same_doc  # [C, base_len]

    if n_draft == 0:
        return base_mask.unsqueeze(0).unsqueeze(0)

    # Draft diagonals: identity per draft step; pad queries still masked
    eye = torch.eye(c, device=device, dtype=torch.bool)
    valid_q = (q_docs != -1).unsqueeze(1)  # [C, 1]
    draft_block = eye & valid_q  # [C, C]
    draft_mask = draft_block.repeat(1, n_draft)  # [C, n_draft * C]

    return torch.cat([base_mask, draft_mask], dim=-1).unsqueeze(0).unsqueeze(0)


def _attn_with_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bool_mask: torch.Tensor,
    scaling: float,
    *,
    use_sdpa: bool,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    """Run attention on a chunk with a small local boolean mask."""
    if use_sdpa:
        # SDPA wants additive float mask (0 / -inf) or bool depending on version;
        # float additive is universally accepted.
        float_mask = torch.zeros(
            bool_mask.shape, dtype=query.dtype, device=query.device
        )
        float_mask.masked_fill_(~bool_mask, torch.finfo(query.dtype).min)
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=float_mask,
            dropout_p=dropout_p if training else 0.0,
            scale=scaling,
        )

    # Eager path
    scores = torch.matmul(query, key.transpose(-2, -1)) * scaling
    scores = scores.masked_fill(~bool_mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    if dropout_p > 0.0 and training:
        probs = F.dropout(probs, p=dropout_p)
    return torch.matmul(probs, value)


def windowed_document_attention(
    query: torch.Tensor,
    base_key: torch.Tensor,
    base_value: torch.Tensor,
    draft_keys: list[torch.Tensor],
    draft_values: list[torch.Tensor],
    document_ids: torch.Tensor,
    sliding_window: int | None,
    scaling: float,
    *,
    use_sdpa: bool = True,
    dropout_p: float = 0.0,
    training: bool = False,
    chunk_size: int | None = None,
    num_key_value_groups: int = 1,
) -> torch.Tensor:
    """Document + SWA (+ TTT diagonal) attention without a dense ``S×KV`` mask.

    Args:
        query: ``[B, H, S, D]``
        base_key / base_value: step-0 K/V ``[B, H_kv, S, D]``
        draft_keys / draft_values: length ``ttt_step`` lists of ``[B, H_kv, S, D]``
            (current TTT step included when ``ttt_step > 0``).
        document_ids: ``[S]`` packed document ids (``-1`` = pad).
        sliding_window: window size ``W``, or ``None`` for full causal within docs.
        scaling: attention scale.
        chunk_size: query chunk length; defaults to ``W`` (or 512 if full attn).

    Returns:
        Attention output ``[B, H, S, D]`` (pre-transpose, same layout as SDPA out).
    """
    base_key = repeat_kv(base_key, num_key_value_groups)
    base_value = repeat_kv(base_value, num_key_value_groups)
    draft_keys = [repeat_kv(k, num_key_value_groups) for k in draft_keys]
    draft_values = [repeat_kv(v, num_key_value_groups) for v in draft_values]

    _b, _h, seq_len, _d = query.shape
    n_draft = len(draft_keys)
    device = query.device

    if chunk_size is None:
        chunk_size = sliding_window if sliding_window is not None else min(512, seq_len)
    chunk_size = max(1, min(chunk_size, seq_len))

    outputs: list[torch.Tensor] = []
    for q_start in range(0, seq_len, chunk_size):
        q_end = min(q_start + chunk_size, seq_len)
        c = q_end - q_start
        q_chunk = query[:, :, q_start:q_end, :]

        if sliding_window is not None:
            kv_start = max(0, q_start - sliding_window + 1)
        else:
            kv_start = 0
        # Causal: last query in chunk sees up to q_end - 1
        kv_end = q_end
        base_len = kv_end - kv_start

        local_k = base_key[:, :, kv_start:kv_end, :]
        local_v = base_value[:, :, kv_start:kv_end, :]
        if n_draft > 0:
            d_ks = [dk[:, :, q_start:q_end, :] for dk in draft_keys]
            d_vs = [dv[:, :, q_start:q_end, :] for dv in draft_values]
            local_k = torch.cat([local_k, *d_ks], dim=-2)
            local_v = torch.cat([local_v, *d_vs], dim=-2)

        bool_mask = _build_local_bool_mask(
            q_start=q_start,
            chunk_len=c,
            kv_start=kv_start,
            base_len=base_len,
            n_draft=n_draft,
            document_ids=document_ids,
            sliding_window=sliding_window,
            device=device,
        )
        out = _attn_with_mask(
            q_chunk,
            local_k,
            local_v,
            bool_mask,
            scaling,
            use_sdpa=use_sdpa,
            dropout_p=dropout_p,
            training=training,
        )
        outputs.append(out)

    return torch.cat(outputs, dim=-2)


def _window_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,  # noqa: ARG001
    scaling: float | None = None,
    *,
    use_sdpa: bool,
    dropout: float = 0.0,
    eagle3_ttt_cache: Eagle3WindowedCache | None = None,
    eagle3_document_ids: torch.Tensor | None = None,
    eagle3_sliding_window: int | None = None,
    eagle3_total_seq_len: int | None = None,  # noqa: ARG001
    eagle3_chunk_size: int | None = None,
    **_kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """AttentionInterface entrypoint for window_sdpa / window_eager."""
    if eagle3_document_ids is None:
        raise ValueError(
            "window_sdpa/window_eager require eagle3_document_ids kwarg "
            "(pass document_ids from Eagle3DraftModel.forward)."
        )
    if scaling is None:
        head_dim = query.shape[-1]
        scaling = 1.0 / math.sqrt(head_dim)

    layer_idx = getattr(module, "layer_idx", 0)
    n_groups = getattr(module, "num_key_value_groups", 1)

    if eagle3_ttt_cache is not None and getattr(
        eagle3_ttt_cache, "is_eagle3_windowed_cache", False
    ):
        base_k = eagle3_ttt_cache.base_keys[layer_idx]
        base_v = eagle3_ttt_cache.base_values[layer_idx]
        draft_ks = eagle3_ttt_cache.draft_keys[layer_idx]
        draft_vs = eagle3_ttt_cache.draft_values[layer_idx]
    else:
        # Fallback: key/value already concatenated as [base | d1 | d2 | ...]
        total_seq_len = eagle3_total_seq_len or query.shape[-2]
        kv_len = key.shape[-2]
        n_steps = kv_len // total_seq_len
        base_k = key[:, :, :total_seq_len, :]
        base_v = value[:, :, :total_seq_len, :]
        draft_ks = [
            key[:, :, t * total_seq_len : (t + 1) * total_seq_len, :]
            for t in range(1, n_steps)
        ]
        draft_vs = [
            value[:, :, t * total_seq_len : (t + 1) * total_seq_len, :]
            for t in range(1, n_steps)
        ]

    doc_ids = eagle3_document_ids
    if doc_ids.ndim > 1:
        doc_ids = doc_ids.reshape(-1)

    attn_out = windowed_document_attention(
        query,
        base_k,
        base_v,
        draft_ks,
        draft_vs,
        document_ids=doc_ids,
        sliding_window=eagle3_sliding_window,
        scaling=scaling,
        use_sdpa=use_sdpa,
        dropout_p=dropout,
        training=bool(getattr(module, "training", False)),
        chunk_size=eagle3_chunk_size,
        num_key_value_groups=n_groups,
    )
    # Match HF attention interface: [B, S, H, D]
    return attn_out.transpose(1, 2).contiguous(), None


def window_sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    return _window_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        use_sdpa=True,
        **kwargs,
    )


def window_eager_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    return _window_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        use_sdpa=False,
        **kwargs,
    )


def resolve_window_attn_implementation(
    draft_attn_impl: str,
    draft_attn_kernel: str = "auto",
) -> str:
    """Map CLI ``draft_attn_impl`` + ``draft_attn_kernel`` to the HF attn name.

    * ``auto`` (default): enable window kernel for ``sdpa`` / ``eager``.
    * ``window_sdpa``: force window kernel (requires sdpa or eager impl).
    * ``dense``: keep stock sdpa/eager with materialized masks.
    """
    if draft_attn_impl == "simple_flex_attention":
        return draft_attn_impl

    use_window = draft_attn_kernel in ("auto", "window_sdpa")
    if draft_attn_kernel == "dense":
        use_window = False
    if not use_window:
        return draft_attn_impl

    if draft_attn_impl == "sdpa":
        return "window_sdpa"
    if draft_attn_impl == "eager":
        return "window_eager"
    return draft_attn_impl


def uses_window_attn_kernel(attn_implementation: str) -> bool:
    return attn_implementation in ("window_sdpa", "window_eager")


# Register into the shared AttentionInterface._global_mapping so Llama/Qwen
# layers resolve these names via transformers.ALL_ATTENTION_FUNCTIONS.
ALL_ATTENTION_FUNCTIONS.register("window_sdpa", window_sdpa_attention_forward)
ALL_ATTENTION_FUNCTIONS.register("window_eager", window_eager_attention_forward)
