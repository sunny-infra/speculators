"""Dense MLA decoder layers for GLM-5.x DFlash / DSpark drafts.

Thin vendor layer on top of :mod:`speculators.models.dflash.mla`. Q is
projected from the draft block hidden states; K/V are projected from
``cat(target_context, draft)`` through DeepSeek-style MLA (no DSA indexer).
RoPE follows ``rope_interleave``: True uses interleaved pair rotation
``(0,1), (2,3), ...`` (vLLM ``get_rope(..., is_neox_style=False)``); False
uses NeoX split-half rotation. Applied only to the ``qk_rope_head_dim``
side channels.
"""

from __future__ import annotations

from typing import Any

from speculators.models.dflash.mla import (
    MlaDFlashAttention,
    MlaDFlashDecoderLayer,
    MlaDraftConfig,
    apply_interleaved_rotary_pos_emb,
    mla_kwargs_from_verifier,
    register_mla_variant,
)
from speculators.models.dflash.model_definitions import apply_rotary_pos_emb

__all__ = [
    "GLM5_DSPARK_ARCHITECTURE",
    "GLM5_MLA_MODEL_TYPES",
    "Glm5Config",
    "Glm5DFlashDecoderLayer",
    "Glm5DFlashMLAAttention",
    "apply_interleaved_rotary_pos_emb",
    "apply_rotary_pos_emb",
    "is_glm5_mla_config",
    "mla_kwargs_from_verifier",
]

# Serve-side vLLM-Ascend registration name (K3 DSpark uses K3DSparkModel).
GLM5_DSPARK_ARCHITECTURE = "Glm5DSparkForCausalLM"
# Current export name plus the legacy training ``model_type``.
GLM5_MLA_MODEL_TYPES = frozenset({"glm5_dspark", "glm5"})


class Glm5Config(MlaDraftConfig):
    """Qwen3-shaped decoder config plus GLM-5 Dense MLA dimensions.

    ``model_type`` is ``glm5_dspark`` so vLLM-Ascend can dispatch the draft
    independently of the GLM-5.2 verifier (``glm_moe_dsa``). Defaults lock the
    serveable 576-d MLA page spec: ``kv_lora_rank=512`` + ``qk_rope_head_dim=64``.
    """

    model_type = "glm5_dspark"

    def __init__(
        self,
        q_lora_rank: int | None = 2048,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 192,
        qk_rope_head_dim: int = 64,
        v_head_dim: int | None = None,
        qk_head_dim: int | None = None,
        rope_interleave: bool = True,
        **kwargs,
    ):
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
        self.model_type = "glm5_dspark"


def is_glm5_mla_config(config: Any) -> bool:
    return getattr(config, "model_type", None) in GLM5_MLA_MODEL_TYPES


class Glm5DFlashMLAAttention(MlaDFlashAttention):
    """GLM-5 Dense MLA attention. Looks up RoPE helpers on this module so
    existing unit tests can patch ``glm5.apply_interleaved_rotary_pos_emb``.
    """

    def _apply_rope(self, q_rope, k_rope, cos, sin):  # noqa: ANN001
        apply_rope = (
            apply_interleaved_rotary_pos_emb
            if self.rope_interleave
            else apply_rotary_pos_emb
        )
        return apply_rope(q_rope, k_rope, cos, sin)


class Glm5DFlashDecoderLayer(MlaDFlashDecoderLayer):
    attention_class = Glm5DFlashMLAAttention


register_mla_variant(
    "glm5_dspark",
    "glm5",
    config_cls=Glm5Config,
    layer_cls=Glm5DFlashDecoderLayer,
    export_arch=GLM5_DSPARK_ARCHITECTURE,
    autoconfig_model_types=("glm5_dspark",),
)
