"""BaseDataset adapter and capability-specific evaluation/logging pipeline."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F

from initium.data.base_dataset import BaseDataset
from initium.data.workspace_capability.curriculum import (
    WorkspaceCapabilityCurriculum,
    difficulty_for,
)
from initium.data.workspace_capability.episode import (
    BOS,
    EOS,
    SEP,
    TRACK_BASE,
    TaskEpisode,
    VOCAB_SIZE,
    validate_episode,
)
from initium.data.workspace_capability.tracks import (
    entity_state,
    nth_farthest,
    permutation,
    relational_join,
    retrieval_aggregation,
)

_GENERATORS = {
    "permutation": permutation.generate,
    "entity_state": entity_state.generate,
    "relational_join": relational_join.generate,
    "nth_farthest": nth_farthest.generate,
    "retrieval_aggregation": retrieval_aggregation.generate,
}
TRACKS = tuple(_GENERATORS)
ALL_TRACK = "all"


class WorkspaceCapabilityDataset(BaseDataset):
    """Synthetic benchmark isolating state, correction, join, and comparison skills."""

    name = "workspace-capability"
    input_dim = VOCAB_SIZE
    output_dim = VOCAB_SIZE
    advance_threshold = 0.82
    old_lesson_mix_rate = 0.0

    def __init__(self, task_track: str = ALL_TRACK, *, batch_size_hint: int = 16):
        task_track = (task_track or ALL_TRACK).lower().replace("-", "_")
        if task_track not in (*TRACKS, ALL_TRACK):
            raise ValueError(
                f"unknown workspace task track {task_track!r}; choose one of {(*TRACKS, ALL_TRACK)}"
            )
        self.task_track = task_track
        self.batch_size_hint = max(1, int(batch_size_hint))
        self._output_proj = None
        self.last_batch_examples: list[TaskEpisode] = []
        self.last_eval_diagnostics: dict[str, Any] = {}
        print(
            f"[workspace-capability] track={self.task_track} | vocabulary={VOCAB_SIZE} "
            "| six-level curriculum | event-oracle generation on CPU"
        )

    def make_curriculum(self):
        return WorkspaceCapabilityCurriculum(self)

    def set_output_proj(self, proj):
        self._output_proj = proj

    def _tracks(self):
        return TRACKS if self.task_track == ALL_TRACK else (self.task_track,)

    def _make_episode(self, track: str, level: int, rng, *, ood=False) -> TaskEpisode:
        episode = validate_episode(
            _GENERATORS[track](difficulty_for(track, level, ood=ood), rng, ood=ood)
        )
        if self.task_track == ALL_TRACK:
            # A typed task marker allows one shared model to condition on the
            # operation grammar without relying on accidental token statistics.
            track_id = TRACKS.index(track)
            episode = TaskEpisode(
                episode.track,
                (BOS, TRACK_BASE + track_id, SEP) + episode.inputs[1:],
                episode.answer,
                {
                    **episode.metadata,
                    "task_id": track_id,
                    "sequence_length": len(episode.inputs) + 2,
                },
            )
        return episode

    def sample_batch(self, curriculum, batch_size):
        if int(batch_size) < 1:
            raise ValueError("batch_size must be >= 1")
        level = int(curriculum.lesson)
        tracks = self._tracks()
        # Balanced assignment per batch prevents small batches from omitting a
        # task entirely and preserves a uniform multi-track objective.
        start_track = random.randrange(len(tracks))
        selected_tracks = [tracks[(start_track + i) % len(tracks)] for i in range(batch_size)]
        random.shuffle(selected_tracks)
        episodes = [self._make_episode(track, level, random) for track in selected_tracks]
        self.last_batch_examples = episodes
        return self._collate(episodes)

    @staticmethod
    def _collate(episodes: list[TaskEpisode]):
        sequences = [episode.sequence for episode in episodes]
        max_len = max(map(len, sequences))
        batch = len(episodes)
        inputs = torch.zeros(batch, max_len, VOCAB_SIZE, dtype=torch.float32)
        targets = torch.zeros(batch, max_len, dtype=torch.long)
        mask = torch.zeros(batch, max_len, dtype=torch.float32)
        for row, (episode, sequence) in enumerate(zip(episodes, sequences)):
            token_ids = torch.tensor(sequence, dtype=torch.long)
            length = token_ids.numel()
            inputs[row, torch.arange(length), token_ids] = 1.0
            targets[row, :length] = torch.tensor(episode.targets, dtype=torch.long)
            mask[row, :length] = torch.tensor(episode.answer_mask, dtype=torch.float32)
        return inputs, targets, mask

    def loss(self, output, target, mask):
        if output.shape[:-1] != target.shape or target.shape != mask.shape:
            raise ValueError("output, target, and mask batch/time dimensions must match")
        valid = mask.bool()
        if not bool(valid.any()):
            return output.new_zeros((), requires_grad=output.requires_grad)
        return F.cross_entropy(output[valid].float(), target[valid])

    def diversity(self, output, target, mask) -> float:
        valid = mask.bool()
        return float(output.argmax(dim=-1)[valid].unique().numel()) if bool(valid.any()) else 0.0

    @staticmethod
    def _bucket(value: int, width: int = 16) -> str:
        lower = max(0, (int(value) // width) * width)
        return f"{lower}-{lower + width - 1}"

    def _record_field(self, field_log, episode, correct_count, answer_count, exact):
        if field_log is None:
            return
        meta = episode.metadata
        key = (
            episode.track,
            str(meta["query_type"]),
            self._bucket(meta["sequence_length"], 64),
            self._bucket(meta["relevant_updates"], 4),
            int(meta["hop_count"]),
            int(meta["source_count"]),
            int(bool(meta["correction_relevant"])),
            int(bool(meta["novelty_distractor"])),
            int(bool(meta.get("equal_priority_conflict", False))),
            self._bucket(meta["distractor_count"], 8),
            self._bucket(meta.get("candidate_count", 1), 8),
        )
        values = field_log.setdefault(key, [0, 0, 0, 0])
        values[0] += int(answer_count)
        values[1] += int(correct_count)
        values[2] += 1
        values[3] += int(exact)

    def _evaluate_episodes(
        self,
        model,
        device,
        episodes: list[TaskEpisode],
        *,
        ablate_memory=False,
        ablate_workspace=False,
        shuffle_workspace_readback=False,
        combiner_skip_stages=None,
        field_log=None,
        verbose_n=0,
    ):
        if self._output_proj is None:
            raise RuntimeError("dataset output projection was not set by the training pipeline")
        was_training = model.training
        prior_workspace_bypass = getattr(model, "workspace_eval_bypass", None)
        prior_workspace_shuffle = getattr(model, "workspace_eval_shuffle_readback", None)
        if ablate_workspace:
            if not hasattr(model, "workspace_eval_bypass") or getattr(model, "workspace", None) is None:
                raise ValueError("workspace ablation requires a model with active workspace")
            model.workspace_eval_bypass = True
        if shuffle_workspace_readback:
            if not hasattr(model, "workspace_eval_shuffle_readback") or getattr(model, "workspace", None) is None:
                raise ValueError("workspace readback shuffle requires a model with active workspace")
            model.workspace_eval_shuffle_readback = True
        totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        metadata_totals: dict[str, list[float]] = defaultdict(lambda: [0.0] * 8)
        correct_total = answer_total = exact_total = 0
        try:
            model.eval()
            with torch.no_grad():
                batch_size = min(self.batch_size_hint, max(1, len(episodes)))
                for offset in range(0, len(episodes), batch_size):
                    group = episodes[offset : offset + batch_size]
                    x, _targets, _mask = self._collate(group)
                    x = x.to(device, non_blocking=True)
                    hidden = (None, None, None)
                    output, _ = model(
                        x,
                        hidden,
                        reset_experience=True,
                        pass_through_memory=not ablate_memory,
                        combiner_skip_stages=combiner_skip_stages,
                    )
                    logits = self._output_proj(output.transpose(0, 1).contiguous())
                    predictions = logits.argmax(dim=-1)
                    for local_idx, episode in enumerate(group):
                        start = len(episode.inputs)
                        pred = predictions[local_idx, start : start + len(episode.answer)].tolist()
                        expected = list(episode.answer)
                        n_correct = sum(int(a == b) for a, b in zip(pred, expected))
                        n_answers = len(expected)
                        exact = n_correct == n_answers
                        correct_total += n_correct
                        answer_total += n_answers
                        exact_total += int(exact)
                        track_counts = totals[episode.track]
                        track_counts[0] += n_correct
                        track_counts[1] += n_answers
                        track_counts[2] += 1
                        track_counts[3] += int(exact)
                        track_meta = metadata_totals[episode.track]
                        track_meta[0] += int(episode.metadata["sequence_length"])
                        track_meta[1] += int(episode.metadata["relevant_updates"])
                        track_meta[2] += int(episode.metadata["distractor_count"])
                        track_meta[3] += int(bool(episode.metadata["correction_relevant"]))
                        track_meta[4] += int(bool(episode.metadata["novelty_distractor"]))
                        track_meta[5] += int(episode.metadata["source_count"])
                        track_meta[6] += float(episode.metadata["candidate_baseline_pct"])
                        track_meta[7] += int(bool(episode.metadata.get("equal_priority_conflict", False)))
                        self._record_field(field_log, episode, n_correct, n_answers, exact)
                        if len(totals) == 1 and totals[episode.track][2] <= verbose_n:
                            print(
                                f"  [{episode.track}/{episode.metadata['query_type']}] "
                                f"pred={pred} target={expected} exact={exact}"
                            )
        finally:
            if ablate_workspace:
                model.workspace_eval_bypass = prior_workspace_bypass
            if shuffle_workspace_readback:
                model.workspace_eval_shuffle_readback = prior_workspace_shuffle
            model.train(was_training)

        accuracy = 100.0 * correct_total / max(answer_total, 1)
        perfect = 100.0 * exact_total / max(len(episodes), 1)
        total_meta = [sum(values[index] for values in metadata_totals.values()) for index in range(8)]
        total_episodes = max(len(episodes), 1)
        # Task-specific strata are emitted through the dataset-owned field log;
        # the shared hop CSV calls its accuracy column "triple_acc" and is
        # intentionally reserved for graph traversal.
        breakdown = {}
        diagnostics = {
            "task_track": self.task_track,
            "accuracy": accuracy,
            "episode_exact_match": perfect,
            "n_episodes": len(episodes),
            "mean_context_tokens": total_meta[0] / total_episodes,
            "mean_relevant_updates": total_meta[1] / total_episodes,
            "mean_distractors": total_meta[2] / total_episodes,
            "correction_episode_fraction": total_meta[3] / total_episodes,
            "novelty_episode_fraction": total_meta[4] / total_episodes,
            "mean_sources": total_meta[5] / total_episodes,
            "mean_candidate_baseline_pct": total_meta[6] / total_episodes,
            "priority_tie_episode_fraction": total_meta[7] / total_episodes,
            "tracks": {
                track: {
                    "accuracy": 100.0 * counts[0] / max(counts[1], 1),
                    "exact_match": 100.0 * counts[3] / max(counts[2], 1),
                    "episodes": counts[2],
                    "answers": counts[1],
                    "mean_context_tokens": metadata_totals[track][0] / max(counts[2], 1),
                    "mean_relevant_updates": metadata_totals[track][1] / max(counts[2], 1),
                    "mean_distractors": metadata_totals[track][2] / max(counts[2], 1),
                    "correction_episode_fraction": metadata_totals[track][3] / max(counts[2], 1),
                    "novelty_episode_fraction": metadata_totals[track][4] / max(counts[2], 1),
                    "mean_sources": metadata_totals[track][5] / max(counts[2], 1),
                    "mean_candidate_baseline_pct": metadata_totals[track][6] / max(counts[2], 1),
                    "priority_tie_episode_fraction": metadata_totals[track][7] / max(counts[2], 1),
                }
                for track, counts in totals.items()
            },
        }
        self.last_eval_diagnostics = diagnostics
        print(
            f"[workspace-capability/{self.task_track}] accuracy {accuracy:.2f}% | "
            f"episode exact {perfect:.2f}% | episodes {len(episodes)}"
        )
        for track, metrics in diagnostics["tracks"].items():
            print(
                f"  [{track}] answer {metrics['accuracy']:.2f}% | "
                f"exact {metrics['exact_match']:.2f}% | "
                f"context={metrics['mean_context_tokens']:.1f} tokens | n={metrics['episodes']}"
            )
        return accuracy, perfect, breakdown

    def evaluate_id(self, model, device, lesson_idx, num_episodes=64, rng=None, **kwargs):
        rng = rng or random.Random(91_000 + int(lesson_idx))
        tracks = self._tracks()
        episodes = [
            self._make_episode(track, lesson_idx, rng)
            for track in (tracks[i % len(tracks)] for i in range(num_episodes))
        ]
        return self._evaluate_episodes(model, device, episodes, **kwargs)

    def evaluate_ood(self, model, device, num_episodes, rng, verbose_n=0, field_log=None):
        rng = rng or random.Random(171_001)
        tracks = self._tracks()
        episodes = [
            self._make_episode(track, 5, rng, ood=True)
            for track in (tracks[i % len(tracks)] for i in range(num_episodes))
        ]
        return self._evaluate_episodes(
            model, device, episodes, field_log=field_log, verbose_n=verbose_n
        )

    def evaluate_ood_ablated(self, model, device, num_episodes, rng):
        tracks = self._tracks()
        rng = rng or random.Random(171_001)
        episodes = [
            self._make_episode(track, 5, rng, ood=True)
            for track in (tracks[i % len(tracks)] for i in range(num_episodes))
        ]
        return self._evaluate_episodes(model, device, episodes, ablate_memory=True)[:2]

    def evaluate_id_ablated(self, model, device, curriculum, lesson_idx):
        rng = random.Random(91_000 + int(lesson_idx))
        tracks = self._tracks()
        episodes = [
            self._make_episode(track, lesson_idx, rng)
            for track in (tracks[i % len(tracks)] for i in range(64))
        ]
        return self._evaluate_episodes(model, device, episodes, ablate_memory=True)[:2]

    def evaluate_workspace_ablated(self, model, device, curriculum, lesson_idx):
        rng = random.Random(91_000 + int(lesson_idx))
        tracks = self._tracks()
        episodes = [
            self._make_episode(track, lesson_idx, rng)
            for track in (tracks[i % len(tracks)] for i in range(64))
        ]
        return self._evaluate_episodes(model, device, episodes, ablate_workspace=True)[:2]

    def evaluate_workspace_shuffled(self, model, device, curriculum, lesson_idx):
        rng = random.Random(91_000 + int(lesson_idx))
        tracks = self._tracks()
        episodes = [
            self._make_episode(track, lesson_idx, rng)
            for track in (tracks[i % len(tracks)] for i in range(64))
        ]
        return self._evaluate_episodes(
            model, device, episodes, shuffle_workspace_readback=True
        )[:2]

    def evaluate_id_combiner_stage_ablated(
        self, model, device, curriculum, lesson_idx, skip_stages
    ):
        rng = random.Random(91_000 + int(lesson_idx))
        tracks = self._tracks()
        episodes = [
            self._make_episode(track, lesson_idx, rng)
            for track in (tracks[i % len(tracks)] for i in range(64))
        ]
        return self._evaluate_episodes(
            model,
            device,
            episodes,
            combiner_skip_stages=set(skip_stages),
        )[:2]

    def field_log_header(self):
        return [
            "track", "query_type", "sequence_length_bucket", "relevant_updates_bucket",
            "hop_count", "source_count", "correction_relevant", "novelty_distractor",
            "equal_priority_conflict", "distractor_count_bucket", "candidate_count_bucket", "answers",
            "correct_answers", "episodes", "exact_episodes",
        ]

    def write_field_log(self, writer, file, step, lesson, eval_type, field_log):
        for key, values in sorted(field_log.items()):
            writer.writerow([step, lesson, eval_type, *key, *values])
        file.flush()
