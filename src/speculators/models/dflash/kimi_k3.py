"""Kimi-K3 Dense MLA decoder for DFlash / DSpark drafts.

Matches TorchSpec ``K3DSparkModel`` / ``k3_dspark``:

* 64 attention heads and a 14336-wide FFN (not the verifier's 96 / 33792).
* ``v_head_dim=128`` vs ``qk_head_dim=192`` (SDPA pads V).
* YaRN softmax scale folds ``mscale_all_dim`` into ``mscale^2 / sqrt(qk_head)``.
* No MLA output gate / nope path (published checkpoints carry neither).
"""

from __future__ import annotations

import math
from typing import Any

from speculators.models.dflash.mla import (
    MlaDFlashAttention,
    MlaDFlashDecoderLayer,
    MlaDraftConfig,
    register_mla_variant,
)

__all__ = [
    "K3_DEFAULT_ROPE_PARAMETERS",
    "K3_DSPARK_ARCHITECTURE",
    "K3_MLA_MODEL_TYPES",
    "K3_ROTARY_CACHE_CAP",
    "K3Config",
    "K3DFlashDecoderLayer",
    "K3DFlashMLAAttention",
    "assert_k3_yarn_attention_scaling_identity",
    "is_k3_mla_config",
    "yarn_mscale",
]

# TorchSpec / lightseekorg/kimi-k3-dspark export contract.
K3_DSPARK_ARCHITECTURE = "K3DSparkModel"
K3_MLA_MODEL_TYPES = frozenset({"k3_dspark"})

# K3 declares ~1M-token max context; pre-building a cos/sin cache at that
# length is ~0.5 GB. Cap the rotary cache; it grows on demand past this.
K3_ROTARY_CACHE_CAP = 32768

K3_DEFAULT_ROPE_PARAMETERS: dict[str, float | int | str] = {
    "rope_type": "yarn",
    "factor": 32.0,
    "original_max_position_embeddings": 32768,
    "rope_theta": 50000.0,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
}


def is_k3_mla_config(config: Any) -> bool:
    return getattr(config, "model_type", None) in K3_MLA_MODEL_TYPES


def yarn_mscale(factor: float, mscale_all_dim: float) -> float:
    """DeepSeek / TorchSpec YaRN mscale: ``0.1 * mscale * ln(factor) + 1``."""
    if factor <= 1.0:
        return 1.0
    return 0.1 * mscale_all_dim * math.log(factor) + 1.0


def assert_k3_yarn_attention_scaling_identity(rotary_emb: Any) -> None:
    """Refuse configs that would double-apply YaRN mscale to RoPE.

    :class:`~transformers.models.qwen3.modeling_qwen3.Qwen3RotaryEmbedding`
    multiplies cos/sin by ``attention_scaling = mscale / mscale_all_dim``.
    The K3 softmax scale already folds ``mscale_all_dim`` as ``mscale^2``,
    which is complementary only when that ratio is 1.0 (TorchSpec's published
    ``mscale == mscale_all_dim == 1.0``).
    """
    scaling = getattr(rotary_emb, "attention_scaling", 1.0)
    if scaling is None:
        return
    if abs(float(scaling) - 1.0) >= 1e-6:
        raise ValueError(
            "K3 YaRN softmax scale already folds mscale^2; "
            f"Qwen3RotaryEmbedding.attention_scaling={float(scaling)} would "
            "double-scale RoPE. Keep rope mscale == mscale_all_dim == 1.0."
        )


class K3Config(MlaDraftConfig):
    """Kimi-K3 MLA draft decoder.

    Defaults match TorchSpec ``kimi_k3_dspark_mla.json``.
    """

    model_type = "k3_dspark"

    def __init__(
        self,
        q_lora_rank: int | None = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        qk_head_dim: int | None = None,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 64,
        intermediate_size: int = 14336,
        rope_interleave: bool = True,
        mla_use_nope: bool = False,
        mla_use_output_gate: bool = False,
        **kwargs,
    ):
        mla_use_nope = bool(kwargs.pop("mla_use_nope", mla_use_nope))
        mla_use_output_gate = bool(
            kwargs.pop("mla_use_output_gate", mla_use_output_gate)
        )
        if mla_use_output_gate:
            raise NotImplementedError(
                "mla_use_output_gate=True is not supported; the published "
                "Kimi-K3-DSpark checkpoint carries no gate weights."
            )
        if mla_use_nope:
            raise NotImplementedError(
                "mla_use_nope=True is not supported on the K3 DSpark draft."
            )
        kwargs.setdefault("num_attention_heads", num_attention_heads)
        kwargs.setdefault("num_key_value_heads", num_key_value_heads)
        kwargs.setdefault("intermediate_size", intermediate_size)
        kwargs.setdefault("max_position_embeddings", 1048576)
        kwargs.setdefault("rope_parameters", dict(K3_DEFAULT_ROPE_PARAMETERS))
        super().__init__(
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            qk_head_dim=qk_head_dim,
            rope_interleave=rope_interleave,
            **kwargs,
        )
        self.model_type = "k3_dspark"
        self.mla_use_nope = False
        self.mla_use_output_gate = False
        _ensure_yarn_mscale_fields(getattr(self, "rope_parameters", None))


def _ensure_yarn_mscale_fields(rope: Any) -> None:
    """Keep DeepSeek YaRN extras that Qwen3Config may strip on init."""
    if isinstance(rope, dict):
        if rope.get("rope_type") == "yarn" or rope.get("type") == "yarn":
            rope.setdefault("mscale", 1.0)
            rope.setdefault("mscale_all_dim", 1.0)
        return
    if rope is None:
        return
    rope_type = getattr(rope, "rope_type", None) or getattr(rope, "type", None)
    if rope_type != "yarn":
        return
    for name, default in (("mscale", 1.0), ("mscale_all_dim", 1.0)):
        if getattr(rope, name, None) is None:
            try:
                setattr(rope, name, default)
            except (AttributeError, TypeError):
                pass


def _rope_mapping(config: Any) -> dict[str, Any]:
    rope = getattr(config, "rope_parameters", None)
    if rope is None:
        rope = getattr(config, "rope_scaling", None)
    if rope is None:
        return {}
    if isinstance(rope, dict):
        return rope
    if hasattr(rope, "items"):
        return dict(rope)
    return {}


class K3DFlashMLAAttention(MlaDFlashAttention):
    """K3 MLA attention: YaRN mscale^2 folded into the softmax scale."""

    def _compute_scaling(self) -> float:
        rope = _rope_mapping(self.config)
        rope_type = rope.get("rope_type", rope.get("type"))
        if rope_type != "yarn":
            return super()._compute_scaling()
        factor = float(rope.get("factor", 1.0))
        mscale_all_dim = float(rope.get("mscale_all_dim", 0.0))
        mscale = yarn_mscale(factor, mscale_all_dim)
        return (mscale * mscale) / math.sqrt(self.qk_head_dim)


class K3DFlashDecoderLayer(MlaDFlashDecoderLayer):
    attention_class = K3DFlashMLAAttention


register_mla_variant(
    "k3_dspark",
    config_cls=K3Config,
    layer_cls=K3DFlashDecoderLayer,
    export_arch=K3_DSPARK_ARCHITECTURE,
)
