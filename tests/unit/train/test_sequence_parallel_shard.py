"""Unit tests for SP batch padding/sharding."""

import torch

from speculators.train.sequence_parallel.shard import (
    pad_seq_to_sp_multiple,
    shard_batch_for_sp,
    validate_sp_head_divisibility,
)


def _make_batch(seq_len: int) -> dict:
    return {
        "input_ids": torch.arange(seq_len).unsqueeze(0),
        "hidden_states": torch.randn(1, seq_len, 8),
        "verifier_last_hidden_states": torch.randn(1, seq_len, 4),
        "loss_mask": torch.ones(1, seq_len, dtype=torch.bool),
        "position_ids": 1 + torch.arange(seq_len).unsqueeze(0),
        "document_ids": torch.zeros(1, seq_len, dtype=torch.long),
        "lengths": torch.tensor([seq_len]),
    }


def test_pad_seq_to_sp_multiple():
    batch = _make_batch(10)
    padded = pad_seq_to_sp_multiple(batch, sp_size=4)
    assert padded["input_ids"].shape[1] == 12
    assert (padded["document_ids"][0, 10:] == -1).all()
    assert not padded["loss_mask"][0, 10:].any()
    # Absolute positions continue
    assert padded["position_ids"][0, 9].item() == 10
    assert padded["position_ids"][0, 10].item() == 11


def test_shard_batch_keeps_absolute_positions():
    batch = pad_seq_to_sp_multiple(_make_batch(8), sp_size=2)
    shard0 = shard_batch_for_sp(batch, sp_rank=0, sp_size=2)
    shard1 = shard_batch_for_sp(batch, sp_rank=1, sp_size=2)
    assert shard0["input_ids"].shape[1] == 4
    assert shard1["input_ids"].shape[1] == 4
    assert shard0["position_ids"][0, 0].item() == 1
    assert shard1["position_ids"][0, 0].item() == 5
    assert shard0["sp_global_start"] == 0
    assert shard1["sp_global_start"] == 4
    assert shard0["sp_global_seq_len"] == 8


def test_validate_sp_head_divisibility():
    validate_sp_head_divisibility(32, 8, 4)
    try:
        validate_sp_head_divisibility(32, 8, 3)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
