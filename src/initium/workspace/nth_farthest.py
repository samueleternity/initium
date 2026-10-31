"""Exact Nth-farthest ranking over decoded workspace candidate vectors."""

from __future__ import annotations

import torch


def select_nth_farthest(
    candidate_vectors: torch.Tensor,
    query_vector: torch.Tensor,
    entity_ids: torch.Tensor,
    rank: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select zero-based descending squared-distance rank with entity-ID ties.

    Args use batch-major candidate tensors ``(B, C, D)`` and matching vectors.
    Invalid candidates are ignored. Ties follow the task oracle: lower entity ID
    wins. Returns selected entity IDs and the computed squared distances.
    """
    if candidate_vectors.ndim != 3:
        raise ValueError("candidate_vectors must have shape (batch, candidates, dimensions)")
    batch, candidates, dimensions = candidate_vectors.shape
    if query_vector.shape != (batch, dimensions):
        raise ValueError("query_vector must have shape (batch, dimensions)")
    if entity_ids.shape != (batch, candidates) or candidate_mask.shape != (batch, candidates):
        raise ValueError("entity_ids and candidate_mask must match the candidate axes")
    if rank.shape != (batch,):
        raise ValueError("rank must have shape (batch,)")
    if candidate_mask.dtype != torch.bool:
        raise TypeError("candidate_mask must be bool")
    distances = (candidate_vectors.float() - query_vector.float().unsqueeze(1)).square().sum(-1)
    # Stable lexicographic ordering: entity ID ascending first, then distance
    # descending while preserving ID order for exact distance ties.
    id_order = entity_ids.argsort(dim=-1, stable=True)
    sorted_distances = distances.gather(1, id_order)
    sorted_mask = candidate_mask.gather(1, id_order)
    sorted_distances = sorted_distances.masked_fill(~sorted_mask, float("-inf"))
    distance_order = sorted_distances.argsort(dim=-1, descending=True, stable=True)
    ordered_indices = id_order.gather(1, distance_order)
    selected_index = ordered_indices.gather(1, rank.unsqueeze(1)).squeeze(1)
    selected_entity = entity_ids.gather(1, selected_index.unsqueeze(1)).squeeze(1)
    return selected_entity, distances
