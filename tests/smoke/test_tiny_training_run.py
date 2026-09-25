import importlib.util
from pathlib import Path

import pytest

requires_mamba_ssm = pytest.mark.skipif(
    importlib.util.find_spec("mamba_ssm") is None, reason="mamba-ssm is unavailable"
)


@pytest.mark.parametrize("controller", ["lstm", "cfc", "cfc+cfc"])
def test_tiny_training_run(
    configure_tiny_training, assert_run_artifacts, synthetic_graph_edges, controller
):
    run_id = f"tiny-{controller.replace('+', '-')}"
    summary = configure_tiny_training.run(
        beta_target=0.001,
        run_id=run_id,
        seed=17,
        controller=controller,
        total_steps=2,
        checkpoint_every=1,
        dataset_type="graph",
        dataset_link=synthetic_graph_edges,
        test_dataset_link=synthetic_graph_edges,
    )
    assert_run_artifacts(run_id, summary)


@pytest.mark.parametrize(
    "extra",
    [
        {"moe_enabled": True, "moe_num_experts": 4},
        {"dynamic_n_mode": True, "dynamic_n_floor": 16, "dynamic_n_ceiling": 32},
        {"link_matrix_mode": "ablated"},
    ],
)
def test_tiny_training_option_modes_independently(
    configure_tiny_training, assert_run_artifacts, synthetic_graph_edges, extra
):
    run_id = "tiny-option-" + str(len(extra)) + "-" + next(iter(extra))
    summary = configure_tiny_training.run(
        beta_target=0.001,
        run_id=run_id,
        seed=17,
        controller="cfc",
        total_steps=2,
        checkpoint_every=1,
        dataset_type="graph",
        dataset_link=synthetic_graph_edges,
        test_dataset_link=synthetic_graph_edges,
        **extra,
    )
    assert_run_artifacts(run_id, summary)
    if extra.get("dynamic_n_mode"):
        log_path = Path(configure_tiny_training.LOG_DIR) / f"run_{run_id}_dynamic_n_growth.csv"
        assert log_path.is_file()


def test_tiny_split_graph_cfc_run(
    configure_tiny_training, assert_run_artifacts, synthetic_graph_edges
):
    summary = configure_tiny_training.run(
        beta_target=0.001,
        run_id="tiny-split-cfc",
        seed=17,
        controller="lstm",
        total_steps=2,
        checkpoint_every=1,
        dataset_type="graph",
        dataset_link=synthetic_graph_edges,
        test_dataset_link=synthetic_graph_edges,
        split_graph_enabled=True,
        split_graph_variant="cfc",
        split_graph_num_blocks=1,
        split_graph_combiner_mode="linear",
    )
    assert_run_artifacts("tiny-split-cfc", summary)


@requires_mamba_ssm
@pytest.mark.gpu
@pytest.mark.parametrize(
    "controller,split_graph,variant",
    [("mamba", False, "cfc"), ("mamba+cfc", False, "cfc"), ("lstm", True, "mamba1")],
)
def test_gpu_tiny_training_paths(
    configure_tiny_training,
    assert_run_artifacts,
    synthetic_graph_edges,
    controller,
    split_graph,
    variant,
):
    options = {}
    if split_graph:
        options.update(
            split_graph_enabled=True,
            split_graph_variant=variant,
            split_graph_num_blocks=1,
        )
    summary = configure_tiny_training.run(
        beta_target=0.001,
        run_id=f"tiny-gpu-{controller.replace('+', '-')}-{variant}",
        seed=17,
        controller=controller,
        total_steps=2,
        checkpoint_every=1,
        dataset_type="graph",
        dataset_link=synthetic_graph_edges,
        test_dataset_link=synthetic_graph_edges,
        **options,
    )
    assert_run_artifacts(summary["run_id"], summary)
