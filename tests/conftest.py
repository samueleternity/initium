"""Shared CPU-sized fixtures for the test suite."""

import importlib.util
import random

import numpy as np
import pytest
import torch


def has_mamba_ssm() -> bool:
    return importlib.util.find_spec("mamba_ssm") is not None


requires_mamba_ssm = pytest.mark.skipif(
    not has_mamba_ssm(), reason="mamba-ssm is an optional GPU dependency"
)


@pytest.fixture
def tiny_model_kwargs():
    return {
        "input_size": 8,
        "hidden_size": 16,
        "nr_cells": 16,
        "cell_size": 8,
        "read_heads": 2,
        "num_hidden_layers": 1,
    }


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def gpu_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return torch.device("cuda")


@pytest.fixture
def seeded_rng():
    state = (torch.random.get_rng_state(), random.getstate(), np.random.get_state())
    torch.manual_seed(1729)
    random.seed(1729)
    np.random.seed(1729)
    yield
    torch.random.set_rng_state(state[0])
    random.setstate(state[1])
    np.random.set_state(state[2])


@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    return tmp_path


@pytest.fixture
def build_tiny_rnn(tiny_model_kwargs):
    def build(controller_type="lstm", **overrides):
        from mamba_controller.mamba_controller import MambaDNC
        from memory_manipulation.stochastic_write_head_v2 import install_stochastic_write_heads

        options = {**tiny_model_kwargs, **overrides}
        split_graph = options.pop("split_graph", False)
        if split_graph:
            from mamba_controller.split_graph_dnc import SplitGraphDNC

            model = SplitGraphDNC(
                input_size=options["input_size"],
                hidden_size=options["hidden_size"],
                nr_cells=options["nr_cells"],
                cell_size=options["cell_size"],
                read_heads=options["read_heads"],
                num_backbone_blocks=options["num_hidden_layers"],
                mamba_variant=options.pop("split_graph_variant", "cfc"),
                cfc_kwargs={"backbone_units": 16},
                **{k: v for k, v in options.items() if k not in {"input_size", "hidden_size", "nr_cells", "cell_size", "read_heads", "num_hidden_layers"}},
            )
            output_proj = None
        else:
            model = MambaDNC(
                **options,
                rnn_type=controller_type,
                independent_linears=True,
                share_memory_between_layers=True,
                cfc_backbone_units=16,
                mamba_d_state=4,
                mamba_d_conv=2,
                mamba_expand=1,
            )
            output_proj = getattr(model, "output", None)
        heads = install_stochastic_write_heads(model, sample=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        return model, output_proj, heads, optimizer

    return build


@pytest.fixture
def synthetic_graph_edges(tmp_path):
    path = tmp_path / "tiny-edges.txt"
    path.write_text("0,1,0\n1,2,0\n2,3,0\n3,4,0\n4,0,0\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def synthetic_kv_facts():
    return [(1, 11), (2, 12), (3, 13)]


@pytest.fixture
def configure_tiny_training(monkeypatch, tmp_path):
    import core_training
    import data.graph_traversal.graph_traversal as graph

    monkeypatch.setattr(core_training, "LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(core_training, "CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    monkeypatch.setattr(core_training, "BATCH_SIZE", 2)
    monkeypatch.setattr(core_training, "MODEL_HIDDEN_SIZE", 16)
    monkeypatch.setattr(core_training, "MODEL_NR_CELLS", 16)
    monkeypatch.setattr(core_training, "MODEL_CELL_SIZE", 8)
    monkeypatch.setattr(core_training, "MODEL_READ_HEADS", 2)
    monkeypatch.setattr(core_training, "CFC_BACKBONE_UNITS", 16)
    monkeypatch.setattr(core_training, "SPLIT_GRAPH_NUM_BLOCKS", 1)
    monkeypatch.setattr(core_training, "USE_AMP", False)
    monkeypatch.setattr(core_training, "LOG_EVERY", 1)
    monkeypatch.setattr(core_training, "EVAL_EVERY", 2)
    monkeypatch.setattr(core_training, "PRIOR_SNAPSHOT_EVERY", 1)
    monkeypatch.setattr(core_training, "KL_ANNEAL_STEPS", 1)
    monkeypatch.setattr(core_training, "OOD_EVAL_EPISODES", 1)
    monkeypatch.setattr(core_training, "OOD_EVAL_EPISODES_PERIODIC", 1)
    monkeypatch.setattr(graph, "EVAL_BATCH_SIZE", 1)
    return core_training


@pytest.fixture
def assert_run_artifacts(tmp_path):
    def check(run_id, summary):
        import csv
        import math

        assert {
            "run_id", "beta_target", "id_triple_acc", "id_perfect_frac",
            "ood_triple_acc", "ood_perfect_frac", "total_elapsed_sec",
        } <= summary.keys()
        checkpoint_paths = list((tmp_path / "checkpoints").glob("*.pt"))
        assert checkpoint_paths
        checkpoint = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
        assert checkpoint["model_config"]["nr_cells"] == 16
        assert checkpoint["prior_state"]
        assert checkpoint["scaler_state_dict"] is not None
        assert checkpoint["ood_rng_state"] is not None
        assert "dynamic_n_state" in checkpoint
        suffixes = (
            "",
            "_ood",
            "_memory_dependency",
            "_combiner_stage_dependency",
            "_lesson_advances",
            "_modality_dependency",
            "_hop_breakdown",
            "_prior_snapshots",
            "_field_breakdown",
        )
        log_paths = [tmp_path / "logs" / f"run_{run_id}{suffix}.csv" for suffix in suffixes]
        assert all(path.is_file() for path in log_paths)
        for path in log_paths:
            with path.open(newline="", encoding="utf-8") as f:
                for row in list(csv.reader(f))[1:4]:
                    for value in row:
                        try:
                            number = float(value)
                        except ValueError:
                            continue
                        assert math.isfinite(number), f"non-finite value in {path.name}"

    return check
