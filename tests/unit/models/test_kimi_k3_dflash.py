"""Unit tests for Kimi-K3 Dense MLA DFlash / DSpark draft layers."""

import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import AutoConfig

from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.kimi_k3 import (
    K3_DSPARK_ARCHITECTURE,
    K3Config,
    K3DFlashMLAAttention,
    yarn_mscale,
)
from speculators.models.dspark import DSparkSpeculatorConfig
from speculators.models.dspark.core import DSparkDraftModel
from speculators.train.cli import (
    _apply_mla_lora_rank_overrides,
    create_transformer_layer_config,
    load_draft_transformer_layer_config,
)


TINY_K3_KWARGS = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "max_position_embeddings": 64,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
    "q_lora_rank": 16,
    "kv_lora_rank": 8,
    "qk_nope_head_dim": 8,
    "qk_rope_head_dim": 4,
    "v_head_dim": 4,
    "qk_head_dim": 12,
    "rope_interleave": True,
    "layer_types": ["full_attention", "full_attention"],
    "_attn_implementation": "eager",
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
}


def _tiny_k3_config(**overrides) -> K3Config:
    kwargs = dict(TINY_K3_KWARGS)
    kwargs.update(overrides)
    return K3Config(**kwargs)


def test_autoconfig_registers_k3():
    cfg = AutoConfig.for_model("k3_dspark")
    assert cfg.__class__ is K3Config
    assert cfg.model_type == "k3_dspark"


def test_k3_config_defaults_match_torchspec():
    cfg = K3Config()
    assert cfg.model_type == "k3_dspark"
    assert cfg.q_lora_rank == 1536
    assert cfg.kv_lora_rank == 512
    assert cfg.qk_nope_head_dim == 128
    assert cfg.qk_rope_head_dim == 64
    assert cfg.v_head_dim == 128
    assert cfg.qk_head_dim == 192
    assert cfg.num_attention_heads == 64
    assert cfg.num_key_value_heads == 64
    assert cfg.intermediate_size == 14336
    assert cfg.rope_interleave is True
    assert cfg.mla_use_output_gate is False
    assert cfg.mla_use_nope is False
    assert cfg.max_position_embeddings == 1048576
    rope = cfg.rope_parameters
    if isinstance(rope, dict):
        assert rope["rope_type"] == "yarn"
        assert rope["mscale_all_dim"] == 1.0
    else:
        assert rope.rope_type == "yarn"
        assert float(rope.mscale_all_dim) == 1.0


def test_k3_rejects_output_gate():
    with pytest.raises(NotImplementedError, match="mla_use_output_gate"):
        K3Config(mla_use_output_gate=True)


def test_k3_rejects_nope():
    with pytest.raises(NotImplementedError, match="mla_use_nope"):
        K3Config(mla_use_nope=True)


def test_yarn_softmax_scale():
    yarn = {
        "rope_type": "yarn",
        "factor": 32.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "rope_theta": 50000.0,
    }
    cfg = _tiny_k3_config(
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        v_head_dim=128,
        num_attention_heads=2,
        rope_parameters=yarn,
    )
    attn = K3DFlashMLAAttention(cfg, layer_idx=0)
    mscale = yarn_mscale(32.0, 1.0)
    expected = (mscale * mscale) / math.sqrt(192)
    assert attn.scaling == pytest.approx(expected, rel=1e-5)
    assert attn.scaling == pytest.approx(0.1309, rel=1e-3)


def test_non_yarn_softmax_scale_is_inv_sqrt():
    cfg = _tiny_k3_config()
    attn = K3DFlashMLAAttention(cfg, layer_idx=0)
    assert attn.scaling == pytest.approx(1.0 / math.sqrt(12))


def test_v_head_pad_roundtrip_shape():
    cfg = _tiny_k3_config()
    attn = K3DFlashMLAAttention(cfg, layer_idx=0)
    assert attn.v_head_dim == 4
    assert attn.qk_head_dim == 12
    bsz, q_len, ctx_len = 1, 2, 4
    hidden = torch.randn(bsz, q_len, cfg.hidden_size)
    target = torch.randn(bsz, ctx_len, cfg.hidden_size)
    kv_len = ctx_len + q_len
    cos = torch.randn(bsz, kv_len, cfg.qk_rope_head_dim)
    sin = torch.randn(bsz, kv_len, cfg.qk_rope_head_dim)
    out, _ = attn(hidden, target, (cos, sin), attention_mask=None)
    assert out.shape == (bsz, q_len, cfg.hidden_size)
    assert torch.isfinite(out).all()


def test_create_transformer_layer_config_k3_keeps_draft_geometry():
    verifier = SimpleNamespace(
        vocab_size=128,
        hidden_size=32,
        num_attention_heads=96,
        num_key_value_heads=96,
        hidden_act="silu",
        max_position_embeddings=64,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        head_dim=8,
        intermediate_size=33792,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=4,
        qk_head_dim=12,
    )
    with patch(
        "speculators.train.cli.AutoConfig.from_pretrained", return_value=verifier
    ):
        config = create_transformer_layer_config(
            "dummy",
            num_layers=2,
            draft_arch="k3",
            hidden_act=None,
            sliding_window=2048,
            full_attention_indices=[0, 1],
        )
    assert isinstance(config, K3Config)
    assert config.model_type == "k3_dspark"
    assert config.num_attention_heads == 64
    assert config.num_key_value_heads == 64
    assert config.intermediate_size == 14336
    assert config.q_lora_rank == 16
    assert config.kv_lora_rank == 8
    assert config.qk_nope_head_dim == 8
    assert config.qk_rope_head_dim == 4
    assert config.v_head_dim == 4
    rope = config.rope_parameters
    rope_type = rope["rope_type"] if isinstance(rope, dict) else rope.rope_type
    assert rope_type == "yarn"
    assert config.layer_types == ["full_attention", "full_attention"]


def test_load_draft_config_uses_k3_class(tmp_path):
    cfg_dir = tmp_path / "k3_decoder"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "k3_dspark",
                "hidden_size": 32,
                "vocab_size": 16,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 64,
                "num_key_value_heads": 64,
                "v_head_dim": 128,
                "qk_head_dim": 192,
            }
        )
    )
    verifier = SimpleNamespace(hidden_size=32, vocab_size=64)
    with patch("speculators.train.cli.get_verifier_config", return_value=verifier):
        config = load_draft_transformer_layer_config(str(cfg_dir), "dummy")
    assert isinstance(config, K3Config)
    assert config.model_type == "k3_dspark"
    assert config.num_attention_heads == 64
    assert config.v_head_dim == 128
    assert config.vocab_size == 64


def test_apply_mla_lora_rank_overrides_on_k3():
    config = _tiny_k3_config()
    args = SimpleNamespace(q_lora_rank=32, kv_lora_rank=16)
    _apply_mla_lora_rank_overrides(config, args)
    assert config.q_lora_rank == 32
    assert config.kv_lora_rank == 16


def _make_dflash_k3(*, fc_norm: bool = True) -> DFlashDraftModel:
    tl_config = _tiny_k3_config()
    config = DFlashSpeculatorConfig(
        transformer_layer_config=tl_config,
        draft_vocab_size=64,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        sample_from_anchor=False,
        fc_norm=fc_norm,
    )
    model = DFlashDraftModel(config)
    torch.nn.init.normal_(model.verifier_lm_head.weight)
    torch.nn.init.ones_(model.verifier_norm.weight)
    return model.eval()


def test_k3_dflash_uses_mla_projections_and_fc_norm():
    model = _make_dflash_k3()
    attn = model.layers[0].self_attn
    assert isinstance(attn, K3DFlashMLAAttention)
    assert hasattr(attn, "q_a_proj")
    assert hasattr(attn, "kv_a_proj_with_mqa")
    assert hasattr(model, "context_proj")
    assert hasattr(model, "context_norm")
    assert not hasattr(model, "fc")
    assert model.fc_norm is not None
    assert len(model.fc_norm) == 2
    keys = model.state_dict()
    assert "context_proj.weight" in keys
    assert "fc_norm.0.weight" in keys
    assert "fc_norm.1.weight" in keys
    assert "fc.weight" not in keys
    assert model.rotary_emb.config.head_dim == 4
    scaling = getattr(model.rotary_emb, "attention_scaling", 1.0)
    assert abs(float(scaling) - 1.0) < 1e-6


def test_k3_project_context_finite():
    model = _make_dflash_k3()
    hidden = torch.randn(1, 8, 2 * 32)
    out = model._project_context(hidden)
    assert out.shape == (1, 8, 32)
    assert torch.isfinite(out).all()


def test_k3_dflash_forward_finite_loss():
    torch.manual_seed(0)
    model = _make_dflash_k3()
    seq_len = 16
    hidden_size = 32
    hidden_states = torch.randn(1, seq_len, 2 * hidden_size)
    verifier_last = torch.randn(1, seq_len, hidden_size)
    input_ids = torch.randint(0, 64, (1, seq_len))
    loss_mask = torch.ones(1, seq_len)
    document_ids = torch.zeros(1, seq_len, dtype=torch.long)
    with torch.no_grad():
        _, loss, metrics = model(
            hidden_states=hidden_states,
            input_ids=input_ids,
            loss_mask=loss_mask,
            verifier_last_hidden_states=verifier_last,
            document_ids=document_ids,
            max_anchors=2,
        )
    assert torch.isfinite(loss)
    assert metrics is not None


def test_k3_dspark_config_roundtrip():
    tl_config = _tiny_k3_config()
    config = DSparkSpeculatorConfig(
        transformer_layer_config=tl_config,
        draft_vocab_size=64,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        markov_rank=8,
        fc_norm=True,
    )
    dumped = config.model_dump()
    reloaded = DSparkSpeculatorConfig(**dumped)
    assert reloaded.transformer_layer_config.model_type == "k3_dspark"
    assert reloaded.transformer_layer_config.v_head_dim == 4
    assert reloaded.architectures == [K3_DSPARK_ARCHITECTURE]
    assert reloaded.fc_norm is True


def test_k3_dspark_save_pretrained_export_contract(tmp_path):
    tl_config = _tiny_k3_config()
    config = DSparkSpeculatorConfig(
        transformer_layer_config=tl_config,
        draft_vocab_size=64,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        markov_rank=8,
        enable_confidence_head=False,
        fc_norm=True,
    )
    model = DSparkDraftModel(config)
    assert isinstance(model.layers[0].self_attn, K3DFlashMLAAttention)
    out = tmp_path / "k3_dspark"
    model.save_pretrained(out)
    saved = json.loads((out / "config.json").read_text())
    assert saved["architectures"] == [K3_DSPARK_ARCHITECTURE]
    assert saved["transformer_layer_config"]["model_type"] == "k3_dspark"
    assert saved["transformer_layer_config"]["v_head_dim"] == 4
    assert saved["fc_norm"] is True


def test_k3_dspark_push_to_hub_uploads_after_architecture_rewrite(tmp_path):
    tl_config = _tiny_k3_config()
    config = DSparkSpeculatorConfig(
        transformer_layer_config=tl_config,
        draft_vocab_size=64,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        markov_rank=8,
        enable_confidence_head=False,
    )
    model = DSparkDraftModel(config)
    out = tmp_path / "k3_dspark"
    uploaded: dict[str, object] = {}

    def _upload(save_directory, repo_id, _files_timestamps, **_kwargs):
        uploaded["architectures"] = json.loads(
            (Path(save_directory) / "config.json").read_text()
        )["architectures"]
        uploaded["repo_id"] = repo_id
        return "ok"

    with (
        patch(
            "huggingface_hub.create_repo",
            return_value=SimpleNamespace(repo_id="user/k3-dspark"),
        ),
        patch.object(DSparkDraftModel, "_get_files_timestamps", return_value={}),
        patch.object(DSparkDraftModel, "_upload_modified_files", side_effect=_upload),
    ):
        model.save_pretrained(out, push_to_hub=True, repo_id="user/k3-dspark")

    assert uploaded["architectures"] == [K3_DSPARK_ARCHITECTURE]
    assert uploaded["repo_id"] == "user/k3-dspark"
    saved = json.loads((out / "config.json").read_text())
    assert saved["architectures"] == [K3_DSPARK_ARCHITECTURE]
