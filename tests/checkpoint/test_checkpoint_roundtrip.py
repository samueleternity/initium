import os
import subprocess
import sys
from pathlib import Path

import torch


def test_model_state_roundtrip_rebuilds_independently(build_tiny_rnn, tmp_checkpoint_dir):
    first, _, _, _ = build_tiny_rnn("lstm")
    projection = torch.nn.Linear(16, 8)
    path = tmp_checkpoint_dir / "checkpoint.pt"
    x = torch.randn(2, 3, 8)
    x_path = tmp_checkpoint_dir / "input.pt"
    y_path = tmp_checkpoint_dir / "output.pt"
    torch.save(x, x_path)
    torch.save(
        {
            "rnn_state_dict": first.state_dict(),
            "output_proj_state_dict": projection.state_dict(),
            "model_config": {
                "input_size": 8,
                "input_dim": 8,
                "hidden_size": 16,
                "nr_cells": 16,
                "cell_size": 8,
                "read_heads": 2,
                "num_hidden_layers": 1,
                "controller_type": "lstm",
            },
            "beta_target": 0.0,
        },
        path,
    )
    first.eval()
    torch.manual_seed(41)
    with torch.no_grad():
        expected, _ = first(x, None)

    src_model = Path(__file__).resolve().parents[2] / "src" / "model"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_model) + os.pathsep + env.get("PYTHONPATH", "")
    script = (
        "import sys, torch; "
        "from inference.model_loader import load_model; "
        "ckpt=torch.load(sys.argv[1], map_location='cpu', weights_only=False); "
        "loaded=load_model(ckpt, torch.device('cpu'), deterministic_write=True); "
        "x=torch.load(sys.argv[2], map_location='cpu', weights_only=True); "
        "torch.manual_seed(41); "
        "y=loaded.rnn(x, None)[0]; torch.save(y, sys.argv[3])"
    )
    subprocess.run(
        [sys.executable, "-c", script, str(path), str(x_path), str(y_path)],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
    )
    actual = torch.load(y_path, map_location="cpu", weights_only=True)
    torch.testing.assert_close(expected, actual)
