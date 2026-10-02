"""Utilities for the Q99 online-write order-sensitivity check."""

from __future__ import annotations

import torch


def build_q99_probe_batch(batch_size: int = 8):
    """Build graph episodes with at least two path writes for Q99 pairing."""
    from initium.data.graph_traversal.graph_traversal import (
        build_traversal_episode,
        collate_fn,
    )

    episodes = []
    while len(episodes) < batch_size:
        episode = build_traversal_episode(10, (2, 4), (2, 4))
        if episode is not None:
            episodes.append(episode)
    return collate_fn(episodes)


def permute_early_writes(
    inputs: torch.Tensor,
    *,
    prefix_length: int,
    permutation: tuple[int, ...],
) -> torch.Tensor:
    """Return a copy with only the first ``prefix_length`` events reordered.

    Inputs use the dataset's batch-first ``(batch, time, features)`` layout.
    Queries and later sequence steps remain in their original positions.
    ``permutation`` contains indices relative to the prefix and must be a
    bijection, so callers can provide a deterministic paired episode.
    """
    if inputs.ndim != 3:
        raise ValueError("inputs must have shape (batch, time, features)")
    if not 0 < prefix_length <= inputs.size(1):
        raise ValueError("prefix_length must be within the input sequence")
    if len(permutation) != prefix_length or sorted(permutation) != list(range(prefix_length)):
        raise ValueError("permutation must be a bijection over the write prefix")
    result = inputs.clone()
    indices = torch.tensor(permutation, device=inputs.device)
    result[:, :prefix_length] = inputs[:, :prefix_length].index_select(1, indices)
    return result


def permute_early_graph_query_writes(
    inputs: torch.Tensor, *, label_dim: int = 30
) -> tuple[torch.Tensor, int]:
    """Swap the first two path-query writes in each graph episode.

    Graph facts have a populated destination field; path-query writes have a
    source and/or edge field, no destination, and prediction_required=0. The
    answer prompts and graph-fact write order are left untouched.
    """
    if inputs.ndim != 3 or inputs.size(-1) < 3 * label_dim + 2:
        raise ValueError("expected graph inputs shaped (batch, time, 3*label_dim + 2)")
    result = inputs.clone()
    tested = 0
    for batch_index in range(inputs.size(0)):
        episode = inputs[batch_index]
        source_present = episode[:, :label_dim].abs().sum(dim=-1) > 0
        edge_present = episode[:, label_dim : 2 * label_dim].abs().sum(dim=-1) > 0
        destination_present = episode[:, 2 * label_dim : 3 * label_dim].abs().sum(dim=-1) > 0
        prediction_required = episode[:, -1] > 0.5
        query_rows = torch.nonzero(
            ~destination_present
            & ~prediction_required
            & (source_present | edge_present),
            as_tuple=False,
        ).flatten()
        if query_rows.numel() < 2:
            continue
        first, second = query_rows[:2]
        result[batch_index, first] = inputs[batch_index, second]
        result[batch_index, second] = inputs[batch_index, first]
        tested += 1
    return result, tested


def order_sensitivity_metrics(
    ordered_output: torch.Tensor,
    permuted_output: torch.Tensor,
    *,
    atol: float = 1e-6,
    mask: torch.Tensor | None = None,
) -> dict[str, float | bool]:
    """Summarize whether task outputs respond to a prefix-order change."""
    if ordered_output.shape != permuted_output.shape:
        raise ValueError("paired outputs must have identical shapes")
    if mask is not None:
        if mask.shape != ordered_output.shape[: mask.ndim]:
            raise ValueError("mask dimensions must match leading output dimensions")
        selected = mask.to(device=ordered_output.device, dtype=torch.bool)
        ordered_output = ordered_output[selected]
        permuted_output = permuted_output[selected]
        if ordered_output.numel() == 0:
            raise ValueError("mask selects no output positions")
    difference = (ordered_output - permuted_output).abs()
    if ordered_output.ndim >= 2:
        changed = ordered_output.argmax(dim=-1) != permuted_output.argmax(dim=-1)
        prediction_flip_rate = changed.float().mean().item()
    else:
        prediction_flip_rate = float(
            (ordered_output.round() != permuted_output.round()).float().mean().item()
        )
    return {
        "outputs_change": bool((difference > atol).any().item()),
        "max_absolute_difference": float(difference.max().item()),
        "mean_absolute_difference": float(difference.mean().item()),
        "prediction_flip_rate": prediction_flip_rate,
    }


@torch.no_grad()
def evaluate_q99_order_sensitivity(
    model,
    output_projection,
    inputs: torch.Tensor,
    answer_mask: torch.Tensor,
    *,
    atol: float = 1e-6,
) -> dict[str, float | bool | int | str]:
    """Compare paired-order task outputs with and without external memory."""
    permuted, tested_episodes = permute_early_graph_query_writes(inputs)
    if tested_episodes == 0:
        return {
            "gate": "not_applicable",
            "tested_episodes": 0,
            "outputs_change_with_memory": False,
            "outputs_change_without_memory": False,
            "memory_incremental_order_effect": 0.0,
            "prediction_flip_rate": 0.0,
        }

    was_training = model.training
    projection_was_training = output_projection.training
    model.eval()
    output_projection.eval()

    def forward(sequence: torch.Tensor, pass_through_memory: bool) -> torch.Tensor:
        raw_output, _ = model(
            sequence,
            (None, None, None),
            reset_experience=True,
            pass_through_memory=pass_through_memory,
        )
        return output_projection(raw_output.transpose(0, 1)).view(
            sequence.size(0), sequence.size(1), 9, 10
        )

    try:
        inputs = inputs.to(next(model.parameters()).device)
        permuted = permuted.to(inputs.device)
        answer_mask = answer_mask.to(device=inputs.device, dtype=torch.bool)
        ordered_full = forward(inputs, True)
        permuted_full = forward(permuted, True)
        ordered_ablate = forward(inputs, False)
        permuted_ablate = forward(permuted, False)
    finally:
        model.train(was_training)
        output_projection.train(projection_was_training)

    full_metrics = order_sensitivity_metrics(
        ordered_full, permuted_full, atol=atol, mask=answer_mask
    )
    ablated_metrics = order_sensitivity_metrics(
        ordered_ablate, permuted_ablate, atol=atol, mask=answer_mask
    )
    delta_difference = (
        (ordered_full - permuted_full) - (ordered_ablate - permuted_ablate)
    ).abs()[answer_mask]
    memory_effect = float(delta_difference.mean().item()) if delta_difference.numel() else 0.0
    gate_pass = bool(full_metrics["outputs_change"]) and memory_effect > atol
    return {
        "gate": "pass" if gate_pass else "fail",
        "tested_episodes": tested_episodes,
        "outputs_change_with_memory": bool(full_metrics["outputs_change"]),
        "outputs_change_without_memory": bool(ablated_metrics["outputs_change"]),
        "max_abs_output_delta": float(full_metrics["max_absolute_difference"]),
        "mean_abs_output_delta": float(full_metrics["mean_absolute_difference"]),
        "prediction_flip_rate": float(full_metrics["prediction_flip_rate"]),
        "memory_incremental_order_effect": memory_effect,
    }
