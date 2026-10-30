"""BaseDataset adapter and capability-specific evaluation/logging pipeline."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
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
    ENTITY_BASE,
    ENTITY_COUNT,
    EV_COMMIT,
    EOS,
    EV_INIT,
    EV_SWAP,
    LOCATION_BASE,
    LOCATION_COUNT,
    QUERY,
    SEP,
    TRACK_BASE,
    TaskEpisode,
    VOCAB_SIZE,
    entity_token,
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

    def _make_episode(
        self,
        track: str,
        level: int,
        rng,
        *,
        ood=False,
        ood_axes=None,
        generalization_axis=None,
    ) -> TaskEpisode:
        difficulty = difficulty_for(track, level, ood=ood, ood_axes=ood_axes)
        if track == "permutation" and generalization_axis is not None:
            difficulty["generalization_axis"] = generalization_axis
            difficulty["report_generalization_axis"] = True
        episode = validate_episode(
            _GENERATORS[track](
                difficulty,
                rng,
                ood=ood,
                **({"ood_axes": ood_axes} if track == "permutation" else {}),
            )
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
                    **(
                        {
                            "state_trace": tuple(
                                {**snapshot, "step": int(snapshot["step"]) + 2}
                                for snapshot in episode.metadata["state_trace"]
                            )
                        }
                        if track == "permutation"
                        else {}
                    ),
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

    def workspace_entity_address_metadata(
        self, input_vectors: torch.Tensor, *, paired_swap_commits: bool = False
    ) -> dict[str, torch.Tensor]:
        """Return causal token-ID keys and state-event write masks for permutation.

        Entity IDs are used only as workspace address/read metadata; they are
        not concatenated into backbone or specialist-value features and do not
        use answer labels. Initialization writes are delayed until the
        entity's location token; swap writes are limited to the two entities in
        each swap event. Both operands are read first; their writes are deferred
        to the commit operands after the complete pair is visible. Query entity
        tokens may read their slot but never write.
        """
        if input_vectors.ndim != 3 or input_vectors.shape[-1] != VOCAB_SIZE:
            raise ValueError(
                "workspace entity metadata expects one-hot inputs shaped (batch, time, vocabulary)"
            )
        token_ids = input_vectors.argmax(dim=-1).detach().cpu().tolist()
        entity_ids = torch.full(
            input_vectors.shape[:2], -1, dtype=torch.long, device=input_vectors.device
        )
        write_mask = torch.zeros(
            input_vectors.shape[:2], dtype=torch.bool, device=input_vectors.device
        )
        pair_swap_ids = torch.full(
            (*input_vectors.shape[:2], 2), -1, dtype=torch.long, device=input_vectors.device
        )
        pair_swap_mask = torch.zeros(
            input_vectors.shape[:2], dtype=torch.bool, device=input_vectors.device
        )
        permutation_track_id = TRACK_BASE + TRACKS.index("permutation")
        for row, tokens in enumerate(token_ids):
            if self.task_track == ALL_TRACK and (
                len(tokens) < 2 or int(tokens[1]) != permutation_track_id
            ):
                continue
            phase = "none"
            pending_entity = -1
            swap_entities_seen = 0
            commit_entities_seen = 0
            pending_commit_entities: list[int] = []
            for position, token_value in enumerate(tokens):
                token = int(token_value)
                if ENTITY_BASE <= token < ENTITY_BASE + ENTITY_COUNT:
                    entity_id = token - ENTITY_BASE
                    entity_ids[row, position] = entity_id
                    if phase == "init_entity":
                        pending_entity = entity_id
                        phase = "init_location"
                    elif phase == "swap" and swap_entities_seen < 2:
                        swap_entities_seen += 1
                        if swap_entities_seen == 2:
                            phase = "swap_done"
                    elif phase == "commit" and commit_entities_seen < 2:
                        if not paired_swap_commits:
                            write_mask[row, position] = True
                        pending_commit_entities.append(entity_id)
                        commit_entities_seen += 1
                        if commit_entities_seen == 2:
                            phase = "commit_done"
                    continue
                if phase == "init_location" and (
                    LOCATION_BASE <= token < LOCATION_BASE + LOCATION_COUNT
                ):
                    entity_ids[row, position] = pending_entity
                    write_mask[row, position] = True
                    phase = "init_done"
                elif token == EV_INIT:
                    phase = "init_entity"
                    pending_entity = -1
                elif token == EV_SWAP:
                    phase = "swap"
                    swap_entities_seen = 0
                    commit_entities_seen = 0
                    pending_commit_entities = []
                elif token == EV_COMMIT and phase == "swap_done":
                    phase = "commit"
                    commit_entities_seen = 0
                elif token == QUERY:
                    phase = "query"
                elif token == SEP:
                    if (
                        paired_swap_commits
                        and phase == "commit_done"
                        and len(pending_commit_entities) == 2
                    ):
                        pair_swap_ids[row, position] = torch.tensor(
                            pending_commit_entities,
                            dtype=torch.long,
                            device=input_vectors.device,
                        )
                        pair_swap_mask[row, position] = True
                    phase = "none"
                    pending_entity = -1
                    swap_entities_seen = 0
                    pending_commit_entities = []
        metadata = {"entity_ids": entity_ids, "entity_write_mask": write_mask}
        if paired_swap_commits:
            metadata["pair_swap_entity_ids"] = pair_swap_ids
            metadata["pair_swap_mask"] = pair_swap_mask
            metadata["pair_swap_steps"] = tuple(
                int(step)
                for step in torch.nonzero(pair_swap_mask.any(dim=0), as_tuple=False)
                .flatten()
                .tolist()
            )
        return metadata

    def workspace_permutation_query_metadata(
        self,
        input_vectors: torch.Tensor,
        episodes: list[TaskEpisode] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Mark the first answer position with its causal permutation query."""
        if input_vectors.ndim != 3 or input_vectors.shape[-1] != VOCAB_SIZE:
            raise ValueError(
                "permutation query metadata expects one-hot inputs shaped "
                "(batch, time, vocabulary)"
            )
        examples = self.last_batch_examples if episodes is None else episodes
        batch, time, _ = input_vectors.shape
        if len(examples) != batch:
            raise RuntimeError("permutation query metadata lost batch/episode alignment")
        query_kind = torch.zeros(
            (batch, time), dtype=torch.long, device=input_vectors.device
        )
        query_id = torch.full_like(query_kind, -1)
        query_steps = set()
        for row, episode in enumerate(examples):
            if episode.track != "permutation":
                continue
            step = len(episode.inputs)
            if not 0 <= step < time:
                raise RuntimeError("permutation answer step exceeds its collated sequence")
            query_type = episode.metadata["query_type"]
            if query_type == "object_to_position":
                query_kind[row, step] = 1
                query_id[row, step] = int(episode.metadata["query_entity_id"])
                query_steps.add(step)
            elif query_type == "position_to_object":
                query_kind[row, step] = 2
                query_id[row, step] = int(episode.metadata["query_location_id"])
                query_steps.add(step)
            else:
                raise ValueError(f"unknown permutation query type {query_type!r}")
        return {
            "permutation_query_kind": query_kind,
            "permutation_query_ids": query_id,
            "permutation_query_steps": tuple(sorted(query_steps)),
        }

    @staticmethod
    def apply_workspace_permutation_query_logits(logits, query_predictions):
        """Replace marked answer logits with workspace-decoded permutation logits."""
        if not query_predictions:
            return logits
        # Keep the task vocabulary mapping here, next to the dataset encoding.
        result = logits.clone()
        for step, position_logits, entity_logits, query_kind, _query_ids in query_predictions:
            if not 0 <= int(step) < logits.shape[1]:
                continue
            position_logits = position_logits.to(device=logits.device, dtype=logits.dtype)
            entity_logits = entity_logits.to(device=logits.device, dtype=logits.dtype)
            query_kind = query_kind.to(device=logits.device)
            position_vocab_logits = result.new_full(
                (result.shape[0], result.shape[-1]), -1e4
            )
            position_vocab_logits[
                :, LOCATION_BASE : LOCATION_BASE + LOCATION_COUNT
            ] = position_logits
            entity_vocab_logits = result.new_full(
                (result.shape[0], result.shape[-1]), -1e4
            )
            entity_vocab_logits[:, ENTITY_BASE : ENTITY_BASE + ENTITY_COUNT] = entity_logits
            original = logits[:, int(step)]
            updated = torch.where(
                query_kind.eq(1).unsqueeze(-1), position_vocab_logits, original
            )
            updated = torch.where(
                query_kind.eq(2).unsqueeze(-1), entity_vocab_logits, updated
            )
            result[:, int(step)] = updated
        return result

    def workspace_state_supervision_metadata(
        self,
        input_vectors: torch.Tensor,
        episodes: list[TaskEpisode] | None = None,
    ) -> dict[str, Any]:
        """Pack oracle entity positions only at causal state-commit boundaries."""
        if input_vectors.ndim != 3 or input_vectors.shape[-1] != VOCAB_SIZE:
            raise ValueError(
                "workspace state supervision expects one-hot inputs shaped (batch, time, vocabulary)"
            )
        batch, time, _ = input_vectors.shape
        examples = self.last_batch_examples if episodes is None else episodes
        if len(examples) != batch:
            raise RuntimeError("workspace state supervision lost batch/episode alignment")
        entity_ids = torch.full(
            (batch, time, 16), -1, dtype=torch.long, device=input_vectors.device
        )
        positions = torch.full_like(entity_ids, -100)
        valid = torch.zeros_like(entity_ids, dtype=torch.bool)
        supervised_steps = set()
        for row, episode in enumerate(examples):
            if episode.track != "permutation":
                continue
            for snapshot in episode.metadata.get("state_trace", ()):
                step = int(snapshot["step"])
                if not 0 <= step < time:
                    raise RuntimeError(
                        f"permutation state snapshot step {step} exceeds its batch sequence"
                    )
                known = [
                    (entity_id, int(position))
                    for entity_id, position in enumerate(snapshot["positions"])
                    if int(position) >= 0
                ]
                if len(known) > entity_ids.shape[-1]:
                    raise RuntimeError("permutation state exceeds the auxiliary head entity width")
                for slot, (entity_id, position) in enumerate(known):
                    entity_ids[row, step, slot] = entity_id
                    positions[row, step, slot] = position
                    valid[row, step, slot] = True
                supervised_steps.add(step)
        return {
            "state_supervision_entity_ids": entity_ids,
            "state_supervision_positions": positions,
            "state_supervision_mask": valid,
            "state_supervision_steps": tuple(sorted(supervised_steps)),
        }

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
        workspace = getattr(model, "workspace", None)
        prior_assignment_observer = getattr(
            workspace, "_initium_address_assignment_observer", None
        )
        collect_entity_bindings = bool(
            workspace is not None
            and getattr(workspace, "entity_keyed_writes", False)
        )
        if ablate_workspace:
            if not hasattr(model, "workspace_eval_bypass") or getattr(model, "workspace", None) is None:
                raise ValueError("workspace ablation requires a model with active workspace")
            model.workspace_eval_bypass = True
        if shuffle_workspace_readback:
            if not hasattr(model, "workspace_eval_shuffle_readback") or getattr(model, "workspace", None) is None:
                raise ValueError("workspace readback shuffle requires a model with active workspace")
            model.workspace_eval_shuffle_readback = True
        totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        split_totals: dict[str, list[float]] = defaultdict(lambda: [0.0] * 7)
        metadata_totals: dict[str, list[float]] = defaultdict(lambda: [0.0] * 8)
        permutation_query_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        permutation_state_depth_totals: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        permutation_pair_commit_attempts = 0
        permutation_pair_commit_successes = 0
        correct_total = answer_total = exact_total = 0
        binding_entity_mentions = binding_assigned_mentions = 0
        binding_repeated_entity_consistency = 0.0
        binding_repeated_entities = 0
        binding_colliding_entity_pairs = binding_entity_pairs = 0
        binding_fragmented_entities = binding_write_route_splits = 0
        binding_write_mentions_with_routes = 0
        binding_unique_slot_fraction = 0.0
        binding_episodes = 0
        try:
            model.eval()
            with torch.no_grad():
                batch_size = min(self.batch_size_hint, max(1, len(episodes)))
                for offset in range(0, len(episodes), batch_size):
                    group = episodes[offset : offset + batch_size]
                    x, _targets, _mask = self._collate(group)
                    workspace_forward_kwargs = {}
                    binding_entity_metadata = None
                    if bool(
                        getattr(model, "workspace_semantic_entity_addressing", False)
                    ):
                        transactional = bool(
                            getattr(model, "workspace_permutation_transactional", False)
                        )
                        binding_entity_metadata = self.workspace_entity_address_metadata(
                            x, paired_swap_commits=transactional
                        )
                        if transactional:
                            binding_entity_metadata.update(
                                self.workspace_permutation_query_metadata(x, group)
                            )
                        workspace_forward_kwargs = {
                            key: value.to(device, non_blocking=True)
                            if isinstance(value, torch.Tensor)
                            else value
                            for key, value in binding_entity_metadata.items()
                        }
                        if transactional:
                            state_metadata = self.workspace_state_supervision_metadata(x, group)
                            workspace_forward_kwargs.update(
                                {
                                    "state_supervision_entity_ids": state_metadata[
                                        "state_supervision_entity_ids"
                                    ].to(device, non_blocking=True),
                                    "state_supervision_mask": state_metadata[
                                        "state_supervision_mask"
                                    ].to(device, non_blocking=True),
                                    "state_supervision_steps": state_metadata[
                                        "state_supervision_steps"
                                    ],
                                }
                            )
                    x = x.to(device, non_blocking=True)
                    hidden = (None, None, None)
                    assignment_trace = []
                    observe_group = collect_entity_bindings and any(
                        episode.track == "permutation" for episode in group
                    )
                    if observe_group:
                        workspace._initium_address_assignment_observer = (
                            lambda step, assignments: assignment_trace.append(
                                (step, assignments.detach())
                            )
                        )
                    try:
                        output, _ = model(
                            x,
                            hidden,
                            reset_experience=True,
                            pass_through_memory=not ablate_memory,
                            combiner_skip_stages=combiner_skip_stages,
                            **workspace_forward_kwargs,
                        )
                    finally:
                        if observe_group:
                            workspace._initium_address_assignment_observer = (
                                prior_assignment_observer
                            )
                    if bool(getattr(model, "workspace_permutation_transactional", False)):
                        permutation_pair_commit_attempts += int(
                            binding_entity_metadata["pair_swap_mask"].sum().item()
                        )
                        committed_events = workspace.pop_semantic_pair_commits()
                        if committed_events:
                            permutation_pair_commit_successes += int(
                                torch.stack([value for _step, value in committed_events])
                                .sum()
                                .item()
                            )
                    if observe_group and assignment_trace:
                        # Copy the full trace once per batch; the observer is
                        # evaluation-only and does not add per-token GPU syncs.
                        trace_steps = [entry[0] for entry in assignment_trace]
                        trace = torch.stack(
                            [entry[1] for entry in assignment_trace], dim=0
                        ).cpu()
                        trace_by_step = {
                            step: trace[index]
                            for index, step in enumerate(trace_steps)
                        }
                        for row, episode in enumerate(group):
                            if episode.track != "permutation":
                                continue
                            entity_slots: dict[int, list[int]] = defaultdict(list)
                            for step, token in enumerate(episode.inputs):
                                if binding_entity_metadata is not None:
                                    if not bool(
                                        binding_entity_metadata["entity_write_mask"][row, step]
                                    ):
                                        continue
                                    entity = int(
                                        binding_entity_metadata["entity_ids"][row, step]
                                    )
                                    if entity < 0:
                                        continue
                                else:
                                    if not ENTITY_BASE <= int(token) < ENTITY_BASE + ENTITY_COUNT:
                                        continue
                                    entity = int(token) - ENTITY_BASE
                                binding_entity_mentions += 1
                                step_assignments = trace_by_step.get(step)
                                if step_assignments is None:
                                    continue
                                valid_slots = step_assignments[row]
                                valid_slots = valid_slots[valid_slots >= 0]
                                if valid_slots.numel() == 0:
                                    continue
                                # Count every accepted MoE route. Reducing the
                                # mention to its first route concealed within-token
                                # route splits and could make collision telemetry
                                # depend on expert ordering.
                                unique_route_slots = torch.unique(valid_slots)
                                binding_write_mentions_with_routes += 1
                                binding_write_route_splits += int(
                                    unique_route_slots.numel() > 1
                                )
                                route_counts = Counter(valid_slots.tolist())
                                entity_slots[entity].append(
                                    route_counts.most_common(1)[0][0]
                                )
                                binding_assigned_mentions += 1
                            if not entity_slots:
                                continue
                            binding_episodes += 1
                            dominant_slots = {}
                            for entity, slots in entity_slots.items():
                                counts = Counter(slots)
                                dominant_slots[entity] = counts.most_common(1)[0][0]
                                binding_fragmented_entities += int(len(counts) > 1)
                                if len(slots) >= 2:
                                    binding_repeated_entity_consistency += (
                                        counts.most_common(1)[0][1] / len(slots)
                                    )
                                    binding_repeated_entities += 1
                            unique_slots = len(set(dominant_slots.values()))
                            binding_unique_slot_fraction += (
                                unique_slots / len(dominant_slots)
                            )
                            slots = list(dominant_slots.values())
                            for left in range(len(slots)):
                                for right in range(left + 1, len(slots)):
                                    binding_entity_pairs += 1
                                    binding_colliding_entity_pairs += int(
                                        slots[left] == slots[right]
                                    )
                    logits = self._output_proj(output.transpose(0, 1).contiguous())
                    if bool(getattr(model, "workspace_permutation_transactional", False)):
                        query_predictions = workspace.pop_semantic_query_predictions()
                        logits = self.apply_workspace_permutation_query_logits(
                            logits, query_predictions
                        )
                        event_logits = dict(workspace.pop_semantic_state_predictions())
                        event_steps = sorted(event_logits)
                        if event_steps:
                            event_predictions_cpu = torch.stack(
                                [event_logits[step].argmax(dim=-1) for step in event_steps]
                            ).cpu()
                            event_predictions_by_step = dict(
                                zip(event_steps, event_predictions_cpu)
                            )
                        else:
                            event_predictions_by_step = {}
                        for row, episode in enumerate(group):
                            if episode.track != "permutation":
                                continue
                            swap_depth = 0
                            for snapshot in episode.metadata.get("state_trace", ()):
                                if snapshot["event"] in ("swap", "distractor"):
                                    swap_depth += 1
                                state_predictions = event_predictions_by_step.get(
                                    int(snapshot["step"])
                                )
                                if state_predictions is None:
                                    continue
                                known_states = [
                                    int(position)
                                    for position in snapshot["positions"]
                                    if int(position) >= 0
                                ]
                                for state_index, target_position in enumerate(known_states):
                                    if int(target_position) < 0:
                                        continue
                                    counts = permutation_state_depth_totals[swap_depth]
                                    counts[0] += int(
                                        state_predictions[row, state_index].item()
                                        == int(target_position)
                                    )
                                    counts[1] += 1
                    predictions = logits.argmax(dim=-1)
                    for local_idx, episode in enumerate(group):
                        start = len(episode.inputs)
                        pred = predictions[local_idx, start : start + len(episode.answer)].tolist()
                        expected = list(episode.answer)
                        n_correct = sum(int(a == b) for a, b in zip(pred, expected))
                        n_answers = len(expected)
                        exact = n_correct == n_answers
                        if episode.track == "permutation":
                            query_counts = permutation_query_totals[
                                str(episode.metadata.get("query_type", "unknown"))
                            ]
                            query_counts[0] += int(exact)
                            query_counts[1] += 1
                        correct_total += n_correct
                        answer_total += n_answers
                        exact_total += int(exact)
                        track_counts = totals[episode.track]
                        track_counts[0] += n_correct
                        track_counts[1] += n_answers
                        track_counts[2] += 1
                        track_counts[3] += int(exact)
                        split = (
                            episode.metadata.get("generalization_axis")
                            if episode.metadata.get("report_generalization_axis")
                            else None
                        )
                        if split is not None:
                            split_counts = split_totals[str(split)]
                            split_counts[0] += n_correct
                            split_counts[1] += n_answers
                            split_counts[2] += 1
                            split_counts[3] += int(exact)
                            split_counts[4] += int(episode.metadata["sequence_length"])
                            split_counts[5] += int(episode.metadata["relevant_updates"])
                            split_counts[6] += int(episode.metadata["distractor_count"])
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
            if collect_entity_bindings:
                workspace._initium_address_assignment_observer = (
                    prior_assignment_observer
                )
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
            "permutation_query_directions": {
                query_type: {
                    "episodes": int(counts[1]),
                    "exact_accuracy": float(counts[0] / counts[1]) if counts[1] else 0.0,
                }
                for query_type, counts in sorted(permutation_query_totals.items())
            },
            "permutation_state_accuracy_by_swap_depth": {
                str(depth): {
                    "entities": int(counts[1]),
                    "accuracy": float(counts[0] / counts[1]) if counts[1] else 0.0,
                }
                for depth, counts in sorted(permutation_state_depth_totals.items())
            },
            "permutation_pair_commit_success_rate": (
                permutation_pair_commit_successes / permutation_pair_commit_attempts
                if permutation_pair_commit_attempts
                else 0.0
            ),
            "permutation_pair_commit_attempts": permutation_pair_commit_attempts,
            "permutation_pair_commit_successes": permutation_pair_commit_successes,
            "tracks": {
                track: {
                    "accuracy": 100.0 * counts[0] / max(counts[1], 1),
                    "exact_match": 100.0 * counts[3] / max(counts[2], 1),
                    "episodes": int(counts[2]),
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
        if binding_entity_mentions:
            binding_metrics = {
                "entity_write_coverage": (
                    binding_assigned_mentions / binding_entity_mentions
                ),
                "same_entity_slot_consistency": (
                    binding_repeated_entity_consistency
                    / max(binding_repeated_entities, 1)
                ),
                "repeated_entities": binding_repeated_entities,
                "distinct_entity_slot_collision": (
                    binding_colliding_entity_pairs / max(binding_entity_pairs, 1)
                ),
                "distinct_entity_pairs": binding_entity_pairs,
                "fragmented_entities": binding_fragmented_entities,
                "write_route_split_fraction": (
                    binding_write_route_splits
                    / max(binding_write_mentions_with_routes, 1)
                ),
                "write_mentions_with_routes": binding_write_mentions_with_routes,
                "unique_entity_slots_fraction": (
                    binding_unique_slot_fraction / max(binding_episodes, 1)
                ),
                "episodes": binding_episodes,
                "entity_mentions": binding_entity_mentions,
            }
            diagnostics["permutation_workspace_binding"] = binding_metrics
        if split_totals:
            diagnostics["permutation_generalization"] = {
                split: {
                    "accuracy": 100.0 * counts[0] / max(counts[1], 1),
                    "exact_match": 100.0 * counts[3] / max(counts[2], 1),
                    "episodes": int(counts[2]),
                    "answers": counts[1],
                    "mean_context_tokens": counts[4] / max(counts[2], 1),
                    "mean_relevant_updates": counts[5] / max(counts[2], 1),
                    "mean_distractors": counts[6] / max(counts[2], 1),
                }
                for split, counts in split_totals.items()
            }
        self.last_eval_diagnostics = diagnostics
        print(
            f"[workspace-capability/{self.task_track}] accuracy {accuracy:.2f}% | "
            f"episode exact {perfect:.2f}% | episodes {len(episodes)}"
        )
        if binding_entity_mentions:
            print(
                "  [workspace-binding/permutation] "
                f"within-episode same-entity slot consistency="
                f"{100.0 * binding_metrics['same_entity_slot_consistency']:.1f}% "
                f"(n={binding_metrics['repeated_entities']} repeated entities) | "
                f"different-entity collision="
                f"{100.0 * binding_metrics['distinct_entity_slot_collision']:.1f}% "
                f"(n={binding_metrics['distinct_entity_pairs']} pairs) | "
                f"write coverage={100.0 * binding_metrics['entity_write_coverage']:.1f}% "
                f"| unique slots/entity="
                f"{binding_metrics['unique_entity_slots_fraction']:.2f} | "
                f"fragmented entities={binding_metrics['fragmented_entities']} | "
                f"multi-slot write routes="
                f"{100.0 * binding_metrics['write_route_split_fraction']:.1f}% "
                f"(n={binding_metrics['write_mentions_with_routes']}) | "
                f"episodes={binding_metrics['episodes']}"
            )
        for track, metrics in diagnostics["tracks"].items():
            print(
                f"  [{track}] answer {metrics['accuracy']:.2f}% | "
                f"exact {metrics['exact_match']:.2f}% | "
                f"context={metrics['mean_context_tokens']:.1f} tokens | n={metrics['episodes']}"
            )
        for split, metrics in diagnostics.get("permutation_generalization", {}).items():
            print(
                f"  [permutation/{split}] answer {metrics['accuracy']:.2f}% | "
                f"exact {metrics['exact_match']:.2f}% | n={metrics['episodes']}"
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

    def evaluate_permutation_generalization(
        self,
        model,
        device,
        lesson_idx,
        episodes_per_split=64,
        rng=None,
        field_log=None,
        ablate_memory=False,
    ):
        """Compare label shift and sequence-length shift independently.

        All conditions use the requested lesson's entity count. The
        label-only condition exactly matches ID swap/distractor difficulty;
        length-only retains ID labels and entity count while increasing the
        number of swaps/distractors. The combined condition measures their
        interaction. Each group contains ``episodes_per_split`` samples.
        """
        if "permutation" not in self._tracks():
            raise ValueError(
                "permutation generalization requires permutation in the selected tracks"
            )
        if int(episodes_per_split) < 1:
            raise ValueError("episodes_per_split must be >= 1")
        rng = rng or random.Random(171_101 + int(lesson_idx))
        axes = (
            ("in_distribution", ()),
            ("label_only", ("labels",)),
            ("length_only", ("length",)),
            ("combined", ("labels", "length")),
        )
        episodes = []
        for _ in range(int(episodes_per_split)):
            paired_seed = rng.getrandbits(64)
            for axis_name, ood_axes in axes:
                episodes.append(
                    self._make_episode(
                        "permutation",
                        lesson_idx,
                        random.Random(paired_seed),
                        ood_axes=ood_axes,
                        generalization_axis=axis_name,
                    )
                )
        return self._evaluate_episodes(
            model,
            device,
            episodes,
            field_log=field_log,
            ablate_memory=ablate_memory,
        )

    def evaluate_permutation_state_diagnostics(
        self,
        model,
        device,
        lesson_idx,
        *,
        num_episodes=1,
        max_events_per_episode=16,
        rng=None,
    ):
        """Probe entity-to-position readback after causal initialization/swap events.

        Each probe replays an episode prefix through one state transition, then
        asks the model for every entity's current position. This tests what the
        workspace actually makes available to the answer path, without giving
        it oracle state or adding a training objective.
        """
        if not bool(getattr(model, "workspace_semantic_entity_addressing", False)):
            raise ValueError(
                "permutation state diagnostics require semantic entity addressing"
            )
        if getattr(model, "workspace", None) is None:
            raise ValueError("permutation state diagnostics require an active workspace")
        if int(num_episodes) < 1 or int(max_events_per_episode) < 0:
            raise ValueError("state diagnostic episode/event limits are invalid")
        rng = rng or random.Random(190_001 + int(lesson_idx))
        source_episodes = [
            self._make_episode("permutation", lesson_idx, rng)
            for _ in range(int(num_episodes))
        ]
        probes = []
        for episode_index, episode in enumerate(source_episodes):
            snapshots = list(episode.metadata.get("state_trace", ()))
            event_limit = int(max_events_per_episode)
            if event_limit and len(snapshots) > event_limit:
                if event_limit == 1:
                    selected_snapshot_indices = [len(snapshots) - 1]
                else:
                    selected_snapshot_indices = sorted(
                        {
                            round(index * (len(snapshots) - 1) / (event_limit - 1))
                            for index in range(event_limit)
                        }
                    )
                snapshots = [snapshots[index] for index in selected_snapshot_indices]
            for snapshot in snapshots:
                positions = snapshot["positions"]
                prefix_end = int(snapshot["step"]) + 1
                if prefix_end > len(episode.inputs):
                    raise RuntimeError("permutation oracle trace points beyond its input stream")
                prefix = episode.inputs[:prefix_end]
                for entity_id, position in enumerate(positions):
                    if int(position) < 0:
                        continue
                    query_inputs = prefix + (QUERY, entity_token(entity_id), SEP)
                    probe = TaskEpisode(
                        "permutation",
                        query_inputs,
                        (LOCATION_BASE + int(position), EOS),
                        {
                            "query_type": "object_to_position",
                            "query_entity_id": int(entity_id),
                            "query_location_id": int(position),
                            "state_probe": True,
                            "sequence_length": len(query_inputs),
                        },
                    )
                    probes.append(
                        {
                            "episode": episode_index,
                            "event": str(snapshot["event"]),
                            "event_index": int(snapshot["event_index"]),
                            "event_step": int(snapshot["step"]),
                            "entity_id": entity_id,
                            "oracle_position": int(position),
                            "probe": probe,
                        }
                    )
        if not probes:
            raise RuntimeError("permutation state diagnostic generated no valid state probes")

        workspace = model.workspace
        prior_assignment_observer = getattr(
            workspace, "_initium_address_assignment_observer", None
        )
        prior_state_observer = getattr(
            workspace, "_initium_recurrent_state_observer", None
        )
        prior_workspace_bypass = getattr(model, "workspace_eval_bypass", None)
        was_training = model.training
        rows = []
        try:
            model.eval()
            with torch.no_grad():
                batch_size = min(self.batch_size_hint, len(probes))
                for offset in range(0, len(probes), batch_size):
                    group = probes[offset : offset + batch_size]
                    episodes = [item["probe"] for item in group]
                    x, _targets, _mask = self._collate(episodes)
                    transactional = bool(
                        getattr(model, "workspace_permutation_transactional", False)
                    )
                    entity_metadata = self.workspace_entity_address_metadata(
                        x, paired_swap_commits=transactional
                    )
                    if transactional:
                        entity_metadata.update(
                            self.workspace_permutation_query_metadata(x, episodes)
                        )
                    forward_metadata = {
                        key: value.to(device, non_blocking=True)
                        if isinstance(value, torch.Tensor)
                        else value
                        for key, value in entity_metadata.items()
                    }
                    # Keep CPU copies for the detailed report below; the
                    # forward metadata itself is moved to the model device.
                    entity_ids = entity_metadata["entity_ids"]
                    entity_write_mask = entity_metadata["entity_write_mask"]
                    assignment_trace = []
                    recurrent_state_trace = []
                    event_steps_needed = {item["event_step"] for item in group}
                    workspace._initium_address_assignment_observer = (
                        lambda step, assignments: assignment_trace.append(
                            (step, assignments.detach())
                        )
                    )

                    def capture_event_state(state, call_index):
                        if call_index in event_steps_needed:
                            recurrent_state_trace.append(
                                (call_index, state.detach().cpu())
                            )

                    workspace._initium_recurrent_state_observer = capture_event_state
                    output, _ = model(
                        x.to(device, non_blocking=True),
                        (None, None, None),
                        reset_experience=True,
                        pass_through_memory=True,
                        **forward_metadata,
                    )
                    logits = self._output_proj(output.transpose(0, 1).contiguous())
                    if transactional:
                        logits = self.apply_workspace_permutation_query_logits(
                            logits, workspace.pop_semantic_query_predictions()
                        )
                    predictions = logits.argmax(dim=-1)
                    workspace._initium_address_assignment_observer = None
                    workspace._initium_recurrent_state_observer = None
                    if prior_workspace_bypass is not None:
                        model.workspace_eval_bypass = True
                    bypass_output, _ = model(
                        x.to(device, non_blocking=True),
                        (None, None, None),
                        reset_experience=True,
                        pass_through_memory=True,
                        **forward_metadata,
                    )
                    bypass_logits = self._output_proj(
                        bypass_output.transpose(0, 1).contiguous()
                    )
                    if transactional:
                        bypass_logits = self.apply_workspace_permutation_query_logits(
                            bypass_logits, workspace.pop_semantic_query_predictions()
                        )
                    bypass_predictions = bypass_logits.argmax(dim=-1)
                    workspace._initium_address_assignment_observer = (
                        lambda step, assignments: assignment_trace.append(
                            (step, assignments.detach())
                        )
                    )

                    def capture_event_state(state, call_index):
                        if call_index in event_steps_needed:
                            recurrent_state_trace.append(
                                (call_index, state.detach().cpu())
                            )

                    workspace._initium_recurrent_state_observer = capture_event_state
                    if prior_workspace_bypass is not None:
                        model.workspace_eval_bypass = prior_workspace_bypass
                    assignments_by_step = {
                        step: value.cpu()
                        for step, value in assignment_trace
                    }
                    states_by_step = {
                        step: value
                        for step, value in recurrent_state_trace
                    }
                    for row_index, item in enumerate(group):
                        probe = item["probe"]
                        answer_index = len(probe.inputs)
                        predicted_token = int(predictions[row_index, answer_index])
                        bypass_predicted_token = int(
                            bypass_predictions[row_index, answer_index]
                        )
                        predicted_position = (
                            predicted_token - LOCATION_BASE
                            if LOCATION_BASE <= predicted_token < LOCATION_BASE + LOCATION_COUNT
                            else -1
                        )
                        bypass_position = (
                            bypass_predicted_token - LOCATION_BASE
                            if LOCATION_BASE
                            <= bypass_predicted_token
                            < LOCATION_BASE + LOCATION_COUNT
                            else -1
                        )
                        write_steps = torch.nonzero(
                            entity_write_mask[row_index]
                            & entity_ids[row_index].eq(item["entity_id"]),
                            as_tuple=False,
                        ).flatten().tolist()
                        last_write_step = write_steps[-1] if write_steps else None
                        accepted_slots = []
                        if last_write_step is not None:
                            step_assignments = assignments_by_step.get(last_write_step)
                            if step_assignments is not None:
                                accepted_slots = [
                                    int(slot)
                                    for slot in step_assignments[row_index].tolist()
                                    if int(slot) >= 0
                                ]
                        slot_counts = Counter(accepted_slots)
                        assigned_slot = (
                            slot_counts.most_common(1)[0][0] if slot_counts else -1
                        )
                        slot_route_split = len(slot_counts) > 1
                        slot_rms = None
                        state = states_by_step.get(item["event_step"])
                        if state is not None and assigned_slot >= 0:
                            state_row = state[row_index].float()
                            if getattr(workspace, "entity_keyed_writes", False):
                                state_row = state_row.split(workspace.d_model, dim=-1)[1]
                            slot_rms = float(
                                state_row[assigned_slot].square().mean().sqrt().cpu()
                            )
                        rows.append(
                            {
                                "episode": item["episode"],
                                "event": item["event"],
                                "event_index": item["event_index"],
                                "event_step": item["event_step"],
                                "entity_id": item["entity_id"],
                                "slot": assigned_slot,
                                "accepted": bool(accepted_slots),
                                "route_slots": sorted(slot_counts),
                                "route_split": slot_route_split,
                                "oracle_position": item["oracle_position"],
                                "predicted_position": predicted_position,
                                "correct": predicted_position == item["oracle_position"],
                                "bypass_position": bypass_position,
                                "bypass_correct": bypass_position == item["oracle_position"],
                                "slot_value_rms": slot_rms,
                            }
                        )
        finally:
            workspace._initium_address_assignment_observer = prior_assignment_observer
            workspace._initium_recurrent_state_observer = prior_state_observer
            if prior_workspace_bypass is not None:
                model.workspace_eval_bypass = prior_workspace_bypass
            model.train(was_training)

        event_groups: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            event_groups[(row["episode"], row["event"], row["event_index"])].append(row)
        correct = sum(int(row["correct"]) for row in rows)
        bypass_correct = sum(int(row["bypass_correct"]) for row in rows)
        accepted = sum(int(row["accepted"]) for row in rows)
        route_splits = sum(int(row["route_split"]) for row in rows)
        event_summaries = []
        print(
            f"[workspace-state/permutation] readback={correct}/{len(rows)} "
            f"({100.0 * correct / max(len(rows), 1):.1f}%) | "
            f"workspace-bypassed={bypass_correct}/{len(rows)} "
            f"({100.0 * bypass_correct / max(len(rows), 1):.1f}%) | "
            f"accepted={accepted}/{len(rows)} "
            f"({100.0 * accepted / max(len(rows), 1):.1f}%) | "
            f"multi-slot write routes={route_splits}/{len(rows)} | "
            f"episodes={int(num_episodes)} transitions={len(event_groups)}"
        )
        for (episode_index, event, event_index), event_rows in sorted(event_groups.items()):
            event_correct = sum(int(row["correct"]) for row in event_rows)
            event_bypass_correct = sum(int(row["bypass_correct"]) for row in event_rows)
            event_accepted = sum(int(row["accepted"]) for row in event_rows)
            event_summaries.append(
                {
                    "name": f"episode-{episode_index}/{event}-{event_index}",
                    "accuracy": 100.0 * event_correct / max(len(event_rows), 1),
                    "bypass_accuracy": (
                        100.0 * event_bypass_correct / max(len(event_rows), 1)
                    ),
                    "accepted_fraction": event_accepted / max(len(event_rows), 1),
                    "queries": len(event_rows),
                }
            )
            print(
                f"  [workspace-state/permutation] episode={episode_index} "
                f"event={event}#{event_index} step={event_rows[0]['event_step']} "
                f"readback={event_correct}/{len(event_rows)} "
                f"bypassed={event_bypass_correct}/{len(event_rows)} "
                f"accepted={event_accepted}/{len(event_rows)}"
            )
            for row in sorted(event_rows, key=lambda value: value["entity_id"]):
                slot_text = str(row["slot"]) if row["slot"] >= 0 else "none"
                predicted_text = (
                    str(row["predicted_position"])
                    if row["predicted_position"] >= 0
                    else "non-location"
                )
                print(
                    f"    entity={row['entity_id']} slot={slot_text} "
                    f"accepted={int(row['accepted'])} "
                    f"oracle_pos={row['oracle_position']} predicted_pos={predicted_text} "
                    f"correct={int(row['correct'])} "
                    f"bypassed_correct={int(row['bypass_correct'])} "
                    f"route_split={int(row['route_split'])} "
                    f"slot_rms={row['slot_value_rms']}"
                )
        return {
            "readback_accuracy": 100.0 * correct / max(len(rows), 1),
            "workspace_bypassed_accuracy": 100.0 * bypass_correct / max(len(rows), 1),
            "accepted_write_fraction": accepted / max(len(rows), 1),
            "multi_slot_route_fraction": route_splits / max(len(rows), 1),
            "queries": len(rows),
            "transitions": len(event_groups),
            "episodes": int(num_episodes),
            "events": event_summaries,
            "rows": rows,
        }

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
