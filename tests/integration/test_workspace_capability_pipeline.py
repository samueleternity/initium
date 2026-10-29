import torch
from torch import nn

from initium.data.workspace_capability.dataset import WorkspaceCapabilityDataset
from initium.data.workspace_capability.episode import VOCAB_SIZE
from initium.workspace.broadcast import WorkspaceBroadcast


class TinyWorkspacePipeline(nn.Module):
    """CPU-sized real workspace with the controller call contract used by eval."""

    def __init__(self):
        super().__init__()
        self.workspace = WorkspaceBroadcast(
            d_model=4,
            num_experts=1,
            num_slots=2,
            diagnostics_enabled=True,
        )
        self.workspace_eval_bypass = False
        self.workspace_eval_shuffle_readback = False
        self.input_projection = nn.Linear(VOCAB_SIZE, 4, bias=False)

        # Make writes and readback deterministic enough for the paired
        # intervention to have a visible effect while preserving the actual
        # production WorkspaceBroadcast transition.
        with torch.no_grad():
            self.workspace.query.weight.zero_()
            self.workspace.key.weight.zero_()
            self.workspace.value.weight.copy_(torch.eye(4))
            self.workspace.write_gate.weight.zero_()
            self.workspace.write_gate.bias.fill_(8.0)
            self.workspace.read_gate.fill_(1.0)

    def forward(self, inputs, hidden=None, *, reset_experience=False,
                pass_through_memory=True, combiner_skip_stages=None):
        batch, steps, _ = inputs.shape
        state = None if hidden is None else hidden[2]
        if reset_experience or state is None:
            state = inputs.new_zeros((batch, 2, 4))
        specialists = self.input_projection(inputs).unsqueeze(2)
        routes = torch.zeros(batch, steps, 1, dtype=torch.long, device=inputs.device)
        weights = torch.ones(batch, steps, 1, dtype=inputs.dtype, device=inputs.device)
        output, state = self.workspace(
            specialists,
            routes,
            weights,
            state,
            bypass_broadcast=self.workspace_eval_bypass,
            shuffle_readback_across_batch=self.workspace_eval_shuffle_readback,
            selected_specialists_only=True,
        )
        return output.transpose(0, 1), (None, None, state)


def test_dataset_evaluation_runs_full_bypass_and_shuffled_workspace_paths():
    torch.manual_seed(4)
    dataset = WorkspaceCapabilityDataset("permutation", batch_size_hint=4)
    model = TinyWorkspacePipeline()
    dataset.set_output_proj(nn.Linear(4, VOCAB_SIZE))
    curriculum = dataset.make_curriculum()

    full_accuracy, full_exact, _ = dataset.evaluate_id(
        model, torch.device("cpu"), lesson_idx=1, num_episodes=8
    )
    full_diagnostics = dataset.last_eval_diagnostics
    bypass_accuracy, bypass_exact = dataset.evaluate_workspace_ablated(
        model, torch.device("cpu"), curriculum, lesson_idx=1
    )
    shuffled_accuracy, shuffled_exact = dataset.evaluate_workspace_shuffled(
        model, torch.device("cpu"), curriculum, lesson_idx=1
    )

    assert 0.0 <= full_accuracy <= 100.0
    assert 0.0 <= full_exact <= 100.0
    assert 0.0 <= bypass_accuracy <= 100.0
    assert 0.0 <= bypass_exact <= 100.0
    assert 0.0 <= shuffled_accuracy <= 100.0
    assert 0.0 <= shuffled_exact <= 100.0
    assert full_diagnostics["tracks"]["permutation"]["episodes"] == 8
    assert model.workspace_eval_bypass is False
    assert model.workspace_eval_shuffle_readback is False
    assert model.training is True


def test_permutation_generalization_evaluation_reports_independent_axes():
    torch.manual_seed(21)
    dataset = WorkspaceCapabilityDataset("permutation", batch_size_hint=4)
    model = TinyWorkspacePipeline().train()
    dataset.set_output_proj(nn.Linear(4, VOCAB_SIZE))

    dataset.evaluate_permutation_generalization(
        model,
        torch.device("cpu"),
        lesson_idx=2,
        episodes_per_split=3,
    )
    splits = dataset.last_eval_diagnostics["permutation_generalization"]

    assert set(splits) == {
        "in_distribution", "label_only", "length_only", "combined"
    }
    assert all(item["episodes"] == 3 for item in splits.values())
    assert splits["label_only"]["mean_context_tokens"] == splits["in_distribution"]["mean_context_tokens"]
    assert splits["length_only"]["mean_context_tokens"] > splits["in_distribution"]["mean_context_tokens"]
    assert model.training is True


def test_workspace_readback_shuffle_changes_only_read_path_not_slot_writes():
    torch.manual_seed(12)
    model = TinyWorkspacePipeline().eval()
    inputs = torch.zeros(2, 4, VOCAB_SIZE)
    inputs[0, :, 32] = 1.0
    inputs[1, :, 63] = 1.0

    with torch.no_grad():
        ordinary_output, ordinary_hidden = model(inputs, reset_experience=True)
        model.workspace_eval_shuffle_readback = True
        shuffled_output, shuffled_hidden = model(inputs, reset_experience=True)

    assert not torch.allclose(ordinary_output, shuffled_output)
    torch.testing.assert_close(ordinary_hidden[2], shuffled_hidden[2])


def test_synthetic_batch_trains_through_workspace_and_answer_only_objective():
    torch.manual_seed(19)
    dataset = WorkspaceCapabilityDataset("all", batch_size_hint=3)
    model = TinyWorkspacePipeline().train()
    output_projection = nn.Linear(4, VOCAB_SIZE)
    dataset.set_output_proj(output_projection)
    inputs, targets, mask = dataset.sample_batch(dataset.make_curriculum(), batch_size=3)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(output_projection.parameters()), lr=1e-3
    )

    output, _ = model(inputs, reset_experience=True)
    logits = output_projection(output.transpose(0, 1).contiguous())
    loss = dataset.loss(logits, targets, mask)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

