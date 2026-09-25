"""
file: inference/tasks/base_task.py

Interface the inference engine talks to (counterpart of data/base_dataset.py).
Any new type (text, audio, video, ...) implements BaseInferenceTask and
registers itself in task_registry.py; the engine never imports anything
task-specific.

Contract:
    dataset_type : canonical type string matched against the checkpoint's
                   supported_dataset_types (e.g. "graph")
    input_dim    : model input width the task produces (DNC input_size)
    output_dim   : output_proj width the task's scoring expects
    build_episodes(n, rng) -> list[Episode]
    score_episode(output, episode, verbose) -> EpisodeScore
        `output` is the post-output_proj (T, output_dim) CPU tensor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from initium.inference.metrics import EpisodeScore


@dataclass
class Episode:
    input_seq: torch.Tensor  # (T, input_dim)
    target: torch.Tensor  # task-defined, aligned with input_seq's time axis
    mask: torch.Tensor  # (T,) 1 where a step is scored
    meta: dict = field(default_factory=dict)
    # Step indices where the model state may be snapshotted/reused (sorted). Everything
    # before a boundary is a static prefix identical across episodes; no scored step
    # (mask==1) may lie before a boundary. Empty -> nothing cacheable by the prefix cache.
    cache_boundaries: list[int] = field(default_factory=list)


class BaseInferenceTask:
    name: str = "base"
    dataset_type: str = "base"
    input_dim: int | None = None
    output_dim: int | None = None

    def build_episodes(self, n: int, rng, perturbation=None) -> list[Episode]:
        """perturbation: optional dataset-defined robustness probe (see
        data/base_dataset.py's evaluate_robustness for the training-side
        analogue), e.g. {'kind': 'noise', 'severity': 0.1}. None (default)
        builds ordinary, unperturbed episodes -- purely additive, existing
        tasks ignore it unless they opt in."""
        raise NotImplementedError

    def score_episode(
        self, output: torch.Tensor, episode: Episode, verbose: bool = False
    ) -> EpisodeScore:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name
