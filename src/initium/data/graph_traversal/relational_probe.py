"""Deterministic relational retrieval probes for DNC read mechanisms.

The fixtures encode a small knowledge graph with role-separated node/relation
features and deliberately include high-overlap distractors. They are an eval
artifact for comparing cosine and relational read heads; they do not alter the
training curriculum or run automatically during training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RelationalProbeCase:
    """Candidate memory rows, query key, and correct row index for one query."""

    memory: torch.Tensor  # (N, W)
    query: torch.Tensor  # (W,)
    target_index: int
    relation_id: int
    held_out_composition: bool


def build_relational_probe(
    *, cases: int = 8, cell_size: int = 64, seed: int = 17098
) -> list[RelationalProbeCase]:
    """Build fixed episodic fixtures with conjunctive relation cues.

    Each case uses role-separated source, relation, and binding channels. The
    target row stores a pair-binding code (source multiplied by relation),
    while a distractor repeats the query's source and relation as a bag of
    features. Cosine therefore prefers the distractor; a relational scorer
    can learn the pair interaction. Composition ids in the latter half are
    held out so callers can report compositional generalization separately.
    """
    if cases < 2:
        raise ValueError("cases must be at least 2 to reserve held-out compositions")
    # Three role channels are sufficient. Tiny smoke configurations use
    # cell_size=8; requiring four dimensions per channel made the probe abort
    # those runs before training, even though the fixture fits in fewer.
    if cell_size < 6:
        raise ValueError("cell_size must be at least 6")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    width = cell_size // 3
    codebook = torch.randn(3, width, width, generator=generator)
    codebook = torch.nn.functional.normalize(codebook, dim=-1)
    fixtures = []
    for index in range(cases):
        source = codebook[0, index % width]
        relation_id = (index * 3 + 1) % width
        relation = codebook[1, relation_id]
        binding = torch.nn.functional.normalize(source * relation, dim=0)
        padding = torch.zeros(cell_size - 3 * width)
        target = torch.cat((torch.zeros(width), torch.zeros(width), binding, padding))
        wrong_relation = codebook[1, (relation_id + 1) % width]
        feature_lure = torch.cat((source, relation, torch.zeros(width), padding))
        source_only = torch.cat(
            (
                torch.zeros(width),
                torch.zeros(width),
                torch.nn.functional.normalize(source * wrong_relation, dim=0),
                padding,
            )
        )
        # A fourth row is an unrelated distractor with a large norm to ensure
        # the scorer responds to relational features rather than row magnitude.
        noise = torch.randn(cell_size, generator=generator) * 0.1
        memory = torch.stack((feature_lure, source_only, target, noise))
        query = torch.cat((source, relation, torch.zeros(width), padding))
        fixtures.append(
            RelationalProbeCase(
                memory=memory,
                query=query,
                target_index=2,
                relation_id=relation_id,
                held_out_composition=index >= cases // 2,
            )
        )
    return fixtures


def cosine_top1_accuracy(cases: list[RelationalProbeCase]) -> float:
    """Return cosine top-1 accuracy for checking fixture construction."""
    if not cases:
        return 0.0
    correct = 0
    for case in cases:
        scores = torch.nn.functional.cosine_similarity(
            case.memory, case.query.unsqueeze(0), dim=-1
        )
        correct += int(scores.argmax().item() == case.target_index)
    return correct / len(cases)


def score_probe_cases(memory_module, cases: list[RelationalProbeCase]) -> dict[str, float]:
    """Report relational-head top-1 accuracy overall and on held-out pairs."""
    if not cases:
        return {"accuracy": 0.0, "held_out_accuracy": 0.0, "cases": 0.0}
    device = next(memory_module.parameters()).device
    correct, held_out_correct, held_out_count = 0, 0, 0
    with torch.no_grad():
        for case in cases:
            memory = case.memory.to(device).unsqueeze(0)
            query = case.query.to(device).view(1, 1, -1)
            strengths = torch.ones(1, 1, device=device)
            score_fn = getattr(
                memory_module,
                "read_content_weightings",
                memory_module.content_weightings,
            )
            weights = score_fn(memory, query, strengths)
            is_correct = int(weights[0, 0].argmax().item() == case.target_index)
            correct += is_correct
            if case.held_out_composition:
                held_out_correct += is_correct
                held_out_count += 1
    return {
        "accuracy": correct / len(cases),
        "held_out_accuracy": held_out_correct / max(1, held_out_count),
        "cases": float(len(cases)),
    }
