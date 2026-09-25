import torch


def test_resume_preserves_step_and_curriculum(
    configure_tiny_training, synthetic_graph_edges, tmp_checkpoint_dir
):
    core = configure_tiny_training
    args = {
        "beta_target": 0.001,
        "run_id": "resume-test",
        "seed": 23,
        "controller": "lstm",
        "checkpoint_every": 1,
        "dataset_type": "graph",
        "dataset_link": synthetic_graph_edges,
        "test_dataset_link": synthetic_graph_edges,
    }
    core.run(total_steps=2, **args)
    path = tmp_checkpoint_dir / "checkpoints" / "resume-test_latest.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 2
    lesson_before = checkpoint["curriculum_lesson"]
    summary = core.run(total_steps=4, resume_from=str(path), **args)
    assert summary["run_id"] == "resume-test"
    resumed = torch.load(path, map_location="cpu", weights_only=False)
    assert resumed["step"] == 4
    assert resumed["curriculum_lesson"] >= lesson_before
