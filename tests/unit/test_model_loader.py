import torch

from inference.model_loader import load_model


def test_loader_rebuilds_from_checkpoint_architecture(build_tiny_rnn):
    rnn, _, _, _ = build_tiny_rnn("lstm")
    projection = torch.nn.Linear(16, 8)
    checkpoint = {
        "model_config": {
            "input_size": 8,
            "input_dim": 8,
            "hidden_size": 16,
            "nr_cells": 16,
            "cell_size": 8,
            "read_heads": 2,
            "num_hidden_layers": 1,
            "controller_type": "lstm",
            "link_matrix_mode": "dense",
            "moe_enabled": False,
        },
        "rnn_state_dict": rnn.state_dict(),
        "output_proj_state_dict": projection.state_dict(),
        "step": 3,
        "run_id": "loader-test",
        "beta_target": 0.0,
    }
    loaded = load_model(checkpoint, torch.device("cpu"), deterministic_write=True)
    assert loaded.step == 3
    assert loaded.rnn.memories[0].nr_cells == 16
    for key, value in rnn.state_dict().items():
        torch.testing.assert_close(loaded.rnn.state_dict()[key], value)


def test_legacy_loader_uses_controller_defaults(build_tiny_rnn, monkeypatch):
    import inference.model_loader as model_loader

    monkeypatch.setattr(model_loader.cc, "CFC_BACKBONE_UNITS", 16)
    rnn, _, _, _ = build_tiny_rnn("cfc")
    projection = torch.nn.Linear(16, 8)
    config = {
        "input_size": 8,
        "input_dim": 8,
        "hidden_size": 16,
        "nr_cells": 16,
        "cell_size": 8,
        "read_heads": 2,
        "num_hidden_layers": 1,
        "controller_type": "cfc",
    }
    loaded = model_loader.load_model(
        {
            "model_config": config,
            "rnn_state_dict": rnn.state_dict(),
            "output_proj_state_dict": projection.state_dict(),
            "beta_target": 0.0,
        },
        torch.device("cpu"),
        deterministic_write=True,
    )
    assert loaded.rnn.cfc_backbone_units == 16
