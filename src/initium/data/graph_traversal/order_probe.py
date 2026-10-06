"""Semantic ordering probes for online memory writes (Q99)."""

from __future__ import annotations

import random

import numpy as np
import torch

Q99_PROBE_EPISODES = 128
Q99_MIN_ORDERED_QUERY_ACCURACY = 50.0
Q99_MAX_FACT_PERMUTATION_DROP = 5.0
Q99_MIN_QUERY_ORDER_DROP = 1.0


def build_q99_probe_batch(batch_size: int = Q99_PROBE_EPISODES, seed: int = 991):
    """Build a reproducible paired batch without advancing training RNG state."""
    from initium.data.graph_traversal.graph_traversal import (
        build_traversal_episode,
        collate_fn,
    )

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        episodes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        while len(episodes) < batch_size:
            episode = build_traversal_episode(10, (2, 4), (2, 4))
            if episode is not None:
                episodes.append(episode)
        return collate_fn(episodes)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _fact_row_indices(episode: torch.Tensor, label_dim: int) -> torch.Tensor:
    destination = episode[:, 2 * label_dim : 3 * label_dim].abs().sum(dim=-1) > 0
    prediction_required = episode[:, -1] > 0.5
    return torch.nonzero(destination & ~prediction_required, as_tuple=False).flatten()


def _path_query_indices(episode: torch.Tensor, label_dim: int) -> torch.Tensor:
    source = episode[:, :label_dim].abs().sum(dim=-1) > 0
    edge = episode[:, label_dim : 2 * label_dim].abs().sum(dim=-1) > 0
    destination = episode[:, 2 * label_dim : 3 * label_dim].abs().sum(dim=-1) > 0
    prediction_required = episode[:, -1] > 0.5
    return torch.nonzero(
        ~destination & ~prediction_required & (source | edge), as_tuple=False
    ).flatten()


def permute_graph_fact_writes(
    inputs: torch.Tensor, *, label_dim: int = 30, seed: int = 992
) -> tuple[torch.Tensor, int]:
    """Shuffle graph facts as a set, preserving the fact/query phase marker."""
    if inputs.ndim != 3 or inputs.size(-1) < 3 * label_dim + 2:
        raise ValueError("expected graph inputs shaped (batch, time, 3*label_dim + 2)")
    result = inputs.clone()
    rng = random.Random(seed)
    tested = 0
    for batch_index in range(inputs.size(0)):
        indices = _fact_row_indices(inputs[batch_index], label_dim)
        if indices.numel() < 2:
            continue
        order = list(range(indices.numel()))
        rng.shuffle(order)
        # Avoid treating an unchanged draw as a successful permutation.
        if order == list(range(len(order))):
            order = order[1:] + order[:1]
        source_rows = indices[torch.tensor(order, device=indices.device)]
        result[batch_index, indices] = inputs[batch_index, source_rows]
        # The first-row marker denotes entry to the fact phase, not a property
        # of an individual fact, so it stays at the first fact after shuffling.
        result[batch_index, indices, -2] = 0.0
        result[batch_index, indices[0], -2] = 1.0
        tested += 1
    return result, tested


def swap_first_two_path_query_writes(
    inputs: torch.Tensor, *, label_dim: int = 30
) -> tuple[torch.Tensor, int]:
    """Make an intentionally invalid order control for the dependent path queries."""
    if inputs.ndim != 3 or inputs.size(-1) < 3 * label_dim + 2:
        raise ValueError("expected graph inputs shaped (batch, time, 3*label_dim + 2)")
    result = inputs.clone()
    tested = 0
    for batch_index in range(inputs.size(0)):
        indices = _path_query_indices(inputs[batch_index], label_dim)
        if indices.numel() < 2:
            continue
        first, second = indices[:2]
        result[batch_index, first] = inputs[batch_index, second]
        result[batch_index, second] = inputs[batch_index, first]
        tested += 1
    return result, tested


def _triple_accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    predicted = logits.argmax(dim=-1)
    correct_triples = predicted.eq(targets).all(dim=-1)
    selected = mask.to(device=logits.device, dtype=torch.bool)
    if not selected.any():
        return 0.0
    return float(correct_triples[selected].float().mean().item() * 100.0)


def _answer_prediction_retention(
    first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor
) -> float:
    selected = mask.to(device=first.device, dtype=torch.bool)
    if not selected.any():
        return 0.0
    same = first.argmax(dim=-1).eq(second.argmax(dim=-1)).all(dim=-1)
    return float(same[selected].float().mean().item() * 100.0)


@torch.no_grad()
def evaluate_q99_order_sensitivity(
    model,
    output_projection,
    inputs: torch.Tensor,
    target_digits: torch.Tensor,
    answer_mask: torch.Tensor,
    *,
    atol: float = 1e-6,
) -> dict[str, float | bool | int | str]:
    """Check fact-order invariance and ordered-query correctness separately.

    Graph facts are a set: shuffling them must preserve answer accuracy.
    Path queries form a dependent sequence: swapping the first two query writes
    is an intentionally invalid control and should reduce correctness. The
    latter is compared with memory ablated to isolate memory's incremental role.
    """
    fact_permuted, fact_episodes = permute_graph_fact_writes(inputs)
    query_swapped, query_episodes = swap_first_two_path_query_writes(inputs)
    if fact_episodes == 0 or query_episodes == 0:
        return {
            "gate": "not_applicable",
            "tested_episodes": int(inputs.size(0)),
            "tested_fact_episodes": fact_episodes,
            "tested_query_episodes": query_episodes,
            "ordered_query_accuracy": 0.0,
            "fact_permuted_accuracy": 0.0,
            "fact_accuracy_delta": 0.0,
            "fact_prediction_retention": 0.0,
            "swapped_query_accuracy": 0.0,
            "query_order_accuracy_drop": 0.0,
            "query_prediction_flip_rate": 0.0,
            "query_memory_incremental_order_effect": 0.0,
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
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        fact_permuted = fact_permuted.to(device)
        query_swapped = query_swapped.to(device)
        target_digits = target_digits.to(device=device, dtype=torch.long)
        answer_mask = answer_mask.to(device=device, dtype=torch.bool)

        ordered_full = forward(inputs, True)
        fact_permuted_full = forward(fact_permuted, True)
        query_swapped_full = forward(query_swapped, True)
        ordered_ablate = forward(inputs, False)
        query_swapped_ablate = forward(query_swapped, False)
    finally:
        model.train(was_training)
        output_projection.train(projection_was_training)

    ordered_accuracy = _triple_accuracy(ordered_full, target_digits, answer_mask)
    fact_accuracy = _triple_accuracy(fact_permuted_full, target_digits, answer_mask)
    swapped_accuracy = _triple_accuracy(query_swapped_full, target_digits, answer_mask)
    query_drop = ordered_accuracy - swapped_accuracy
    query_flip = _answer_prediction_retention(ordered_full, query_swapped_full, answer_mask)
    memory_delta = (
        (ordered_full - query_swapped_full) - (ordered_ablate - query_swapped_ablate)
    ).abs()[answer_mask]
    memory_effect = float(memory_delta.mean().item()) if memory_delta.numel() else 0.0
    fact_delta = fact_accuracy - ordered_accuracy

    fact_order_ok = fact_delta >= -Q99_MAX_FACT_PERMUTATION_DROP
    ordered_queries_ok = ordered_accuracy >= Q99_MIN_ORDERED_QUERY_ACCURACY
    query_order_ok = query_drop >= Q99_MIN_QUERY_ORDER_DROP and memory_effect > atol
    gate_pass = fact_order_ok and ordered_queries_ok and query_order_ok
    return {
        "gate": "pass" if gate_pass else "fail",
        "tested_episodes": int(inputs.size(0)),
        "tested_fact_episodes": fact_episodes,
        "tested_query_episodes": query_episodes,
        "ordered_query_accuracy": ordered_accuracy,
        "fact_permuted_accuracy": fact_accuracy,
        "fact_accuracy_delta": fact_delta,
        "fact_prediction_retention": _answer_prediction_retention(
            ordered_full, fact_permuted_full, answer_mask
        ),
        "swapped_query_accuracy": swapped_accuracy,
        "query_order_accuracy_drop": query_drop,
        "query_prediction_flip_rate": 100.0 - query_flip,
        "query_memory_incremental_order_effect": memory_effect,
    }
