import pytest

from inference.checkpoint_io import describe_checkpoint, load_checkpoint


def test_checkpoint_required_keys_and_override(tmp_path):
    import torch

    path = tmp_path / "ckpt.pt"
    torch.save({"model_config": {}}, path)
    with pytest.raises(ValueError, match="rnn_state_dict"):
        load_checkpoint(str(path))
    torch.save(
        {
            "model_config": {"hidden_size": 8},
            "rnn_state_dict": {},
            "output_proj_state_dict": {},
        },
        path,
    )
    ckpt = load_checkpoint(str(path), {"hidden_size": 16})
    assert ckpt["model_config"]["hidden_size"] == 16


def test_describe_checkpoint_includes_architecture_summary():
    text = describe_checkpoint(
        {
            "step": 2,
            "model_config": {
                "controller_type": "cfc",
                "hidden_size": 16,
                "nr_cells": 16,
                "link_matrix_mode": "ablated",
                "moe_enabled": True,
            },
        }
    )
    for value in ("cfc", "hidden=16", "nr_cells=16", "ablated", "moe=True"):
        assert value in text
