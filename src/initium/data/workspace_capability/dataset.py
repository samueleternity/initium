"""BaseDataset adapter and capability-specific evaluation/logging pipeline."""

from __future__ import annotations

import math
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
    BOOL_FALSE,
    BOOL_TRUE,
    BOS,
    COUNT_BASE,
    ENTITY_BASE,
    ENTITY_COUNT,
    EOS,
    EV_ASSERT,
    EV_COMMIT,
    EV_COUNT_QUERY,
    EV_FACT,
    EV_INIT,
    EV_KV,
    EV_MOVE,
    EV_NOISE,
    EV_PARITY_QUERY,
    EV_PUT,
    EV_REMOVE,
    EV_SUM_QUERY,
    EV_SUPERSEDE,
    EV_SWAP,
    EV_UPDATE,
    EV_VECTOR,
    LOCATION_BASE,
    LOCATION_COUNT,
    QUERY,
    RELATION_BASE,
    RELATION_COUNT,
    SCALAR_BASE,
    SCALAR_COUNT,
    SOURCE_BASE,
    SOURCE_COUNT,
    SEP,
    TRACK_TOKEN_IDS,
    track_marker_info,
    track_token,
    VOCAB_SIZE,
    TaskEpisode,
    entity_token,
    scalar_token,
    source_token,
    validate_episode,
)
from initium.data.workspace_capability.tracks import (
    attentional_blink,
    belief_tracking,
    bistable_perception,
    capacity_bottleneck,
    entity_state,
    ignition,
    masked_priming,
    mixed_skill,
    no_report_confidence,
    nth_farthest,
    permutation,
    psychological_refractory,
    relational_join,
    retrieval_aggregation,
    rule_switching,
    sustained_maintenance,
    trace_conditioning,
    variable_binding,
)
from initium.data.workspace_capability.gnw_metrics import (
    binary_brier,
    expected_calibration_error,
    psychometric_logistic,
    type2_auc,
    wilson_interval,
)

_GENERATORS = {
    "permutation": permutation.generate,
    "entity_state": entity_state.generate,
    "relational_join": relational_join.generate,
    "nth_farthest": nth_farthest.generate,
    "retrieval_aggregation": retrieval_aggregation.generate,
    "capacity_bottleneck": capacity_bottleneck.generate,
    "belief_tracking": belief_tracking.generate,
    "mixed_skill": mixed_skill.generate,
    "rule_switching": rule_switching.generate,
    "attentional_blink": attentional_blink.generate,
    "variable_binding": variable_binding.generate,
    "ignition": ignition.generate,
    "psychological_refractory": psychological_refractory.generate,
    "masked_priming": masked_priming.generate,
    "sustained_maintenance": sustained_maintenance.generate,
    "no_report_confidence": no_report_confidence.generate,
    "bistable_perception": bistable_perception.generate,
    "trace_conditioning": trace_conditioning.generate,
}
TRACKS = tuple(_GENERATORS)
ALL_TRACK = "all"
NEW_TRACKS = (
    "capacity_bottleneck",
    "belief_tracking",
    "mixed_skill",
    "rule_switching",
    "attentional_blink",
    "variable_binding",
)
NEW_ALL_TRACK = "new_all"
GNW_TRACKS = (
    "ignition", "attentional_blink", "psychological_refractory", "masked_priming",
    "sustained_maintenance", "no_report_confidence", "bistable_perception", "trace_conditioning",
)
GNW_ALL_TRACK = "gnw_all"
MIXED_BINDING_TRACK = "mixed_binding"
GNW_TASK_DESIGN_VERSION = 2
MULTI_TRACKS = {ALL_TRACK, NEW_ALL_TRACK, GNW_ALL_TRACK, MIXED_BINDING_TRACK}
# Optional task-specific pathways whose causal contribution can be tested at
# evaluation time. Attribute names are deliberately model-facing so the
# evaluator only toggles mechanisms that are present and enabled in a run.
_FUNCTIONAL_FEATURE_ABLATIONS = (
    (
        "no_report_confidence_readout",
        "workspace_no_report_confidence_readout_enabled",
    ),
    ("masked_prime_trace", "workspace_masked_prime_trace_enabled"),
    ("rule_feedback_update", "workspace_rule_feedback_update_enabled"),
    ("attentional_item_memory", "workspace_attentional_item_memory_enabled"),
    ("attentional_cue_readout", "workspace_attentional_cue_readout_enabled"),
    ("nth_farthest_exact_readout", "workspace_nth_farthest_exact_readout"),
    ("retrieval_exact_readout", "workspace_retrieval_exact_readout"),
    ("operation_executor", "workspace_operation_executor_enabled"),
    ("exact_value_comparison", "workspace_exact_value_comparison_enabled"),
    ("exact_selection", "workspace_exact_selection_enabled"),
)
_FUNCTIONAL_FEATURE_TRACKS = {
    "no_report_confidence_readout": {"no_report_confidence"},
    "masked_prime_trace": {"masked_priming"},
    "attentional_item_memory": {"attentional_blink"},
    "attentional_cue_readout": {"attentional_blink"},
}
_WORKSPACE_MODULE_FEATURE_ABLATIONS = {
    "attentional_cue_readout": "attentional_cue_readout_enabled",
    "nth_farthest_exact_readout": "nth_farthest_exact_readout",
    "retrieval_exact_readout": "retrieval_exact_readout",
    "operation_executor": "operation_executor",
}
KEY_VALUE_OPERAND_CAPACITY = 16
_EVAL_KV_ATTENTION_BUDGET_BYTES = 512 * 1024 * 1024
_SDPA_MATH_WORKSPACE_MULTIPLIER = 2
_WORKSPACE_OPERATION_CODES = {
    "lookup": 1,
    "copy": 2,
    "add": 3,
    "sort": 4,
    "compare": 5,
    "select": 6,
    "rank": 7,
    "swap": 8,
    "reduce": 9,
}
_WORKSPACE_OPERATION_NAMES = {
    1: "lookup",
    2: "copy",
    3: "modular_add",
    4: "sort",
    5: "compare",
    6: "select",
    7: "rank",
    8: "swap",
    9: "reduce",
}
_WORKSPACE_OPERAND_KEY = 1
_WORKSPACE_OPERAND_LITERAL = 2
_REDUCE_RESULT_KEY = 63
_ENTITY_ADDRESS_EVENT_TOKENS = {
    "capacity_bottleneck": {EV_INIT, EV_MOVE, EV_UPDATE},
    "rule_switching": {EV_ASSERT, EV_VECTOR},
    "variable_binding": {EV_INIT, EV_KV, EV_UPDATE, EV_VECTOR},
    "entity_state": {EV_PUT, EV_ASSERT, EV_MOVE, EV_REMOVE, EV_SUPERSEDE},
    "relational_join": {EV_FACT},
    "nth_farthest": {EV_VECTOR},
    "retrieval_aggregation": {EV_KV, EV_UPDATE},
    "ignition": {EV_INIT, EV_UPDATE},
    "psychological_refractory": {EV_INIT, EV_UPDATE},
    "masked_priming": {EV_INIT, EV_UPDATE},
    "sustained_maintenance": {EV_INIT, EV_UPDATE},
    "no_report_confidence": {EV_INIT, EV_UPDATE},
    "bistable_perception": {EV_INIT, EV_UPDATE},
    "trace_conditioning": {EV_INIT, EV_UPDATE},
}


class WorkspaceCapabilityDataset(BaseDataset):
    """Synthetic benchmark isolating state, belief, capacity, and coordination skills."""

    name = "workspace-capability"
    input_dim = VOCAB_SIZE
    output_dim = VOCAB_SIZE
    advance_threshold = 0.80
    old_lesson_mix_rate = 0.0

    def __init__(self, task_track: str = ALL_TRACK, *, batch_size_hint: int = 16):
        task_track = (task_track or ALL_TRACK).lower().replace("-", "_")
        if task_track not in (*TRACKS, ALL_TRACK, NEW_ALL_TRACK, GNW_ALL_TRACK, MIXED_BINDING_TRACK):
            raise ValueError(
                "unknown workspace task track "
                f"{task_track!r}; choose one of "
                f"{(*TRACKS, ALL_TRACK, NEW_ALL_TRACK, GNW_ALL_TRACK, MIXED_BINDING_TRACK)}"
            )
        self.task_track = task_track
        self.gnw_task_design_version = GNW_TASK_DESIGN_VERSION
        self.batch_size_hint = max(1, int(batch_size_hint))
        # Promotion checks remain inexpensive by default. Runs that are
        # diagnosing multi-track stalls can raise this to 128+ so each
        # required progression stratum has a stable estimate.
        self.curriculum_eval_episodes_per_group = 16
        self.track_homogeneous_batches = False
        self.training_step = 0
        self.last_batch_padding_utilization = 1.0
        self.last_batch_sequence_mean = 0.0
        self.last_batch_sequence_max = 0
        self._output_proj = None
        self.token_id_input = False
        self.last_batch_examples: list[TaskEpisode] = []
        self.last_eval_diagnostics: dict[str, Any] = {}
        self._last_id_eval_lesson: int | None = None
        self._last_id_eval_episodes: list[TaskEpisode] = []
        print(
            f"[workspace-capability] track={self.task_track} | vocabulary={VOCAB_SIZE} "
            "| six-level curriculum | event-oracle generation on CPU"
        )

    def make_curriculum(self):
        return WorkspaceCapabilityCurriculum(self)

    def set_output_proj(self, proj):
        self._output_proj = proj

    def _tracks(self):
        if self.task_track == ALL_TRACK:
            return TRACKS
        if self.task_track == NEW_ALL_TRACK:
            return NEW_TRACKS
        if self.task_track == GNW_ALL_TRACK:
            return GNW_TRACKS
        if self.task_track == MIXED_BINDING_TRACK:
            return ("mixed_skill", "variable_binding")
        return (self.task_track,)

    @staticmethod
    def _curriculum_groups(track: str, lesson: int) -> tuple[str, ...]:
        """Return challenge strata that must pass before this track advances."""
        difficulty = difficulty_for(track, lesson)
        if track == "capacity_bottleneck":
            return ("change_detection", "multiple_object_tracking", "n_back")
        if track == "belief_tracking":
            return ("false_belief", "true_belief", "reality")
        if track == "mixed_skill":
            return (f"dependent_ops_{int(difficulty['operations'])}",)
        if track == "rule_switching":
            switches = int(difficulty["switches"])
            if not switches:
                return ("stable_rule",)
            groups = ["cued_switch", "feedback_inferred_switch"]
            if switches >= 2:
                groups.append("mixed_cue_switch")
            return tuple(groups)
        if track == "attentional_blink":
            maximum_lag = int(difficulty["maximum_lag"])
            return tuple(
                f"lag_{lag}_{mask}"
                for lag in range(1, maximum_lag + 1)
                for mask in ("unmasked", "masked")
            )
        if track == "variable_binding":
            chain = "long" if int(difficulty["assignments"]) >= 12 else "short"
            return (f"computed_chain_{chain}",)
        if track == "ignition":
            return ("all",)
        if track in {
            "psychological_refractory", "masked_priming", "sustained_maintenance",
            "no_report_confidence", "bistable_perception", "trace_conditioning",
        }:
            return ("all",)
        return ("all",)

    def _make_curriculum_eval_episodes(
        self,
        lesson_idx: int,
        rng: random.Random,
        episodes_per_group: int | None = None,
    ) -> list[TaskEpisode]:
        """Build a balanced, bounded evaluation set for promotion and ablations."""
        if episodes_per_group is None:
            episodes_per_group = self.curriculum_eval_episodes_per_group
        episodes_per_group = max(1, int(episodes_per_group))
        episodes: list[TaskEpisode] = []
        for track in self._tracks():
            required = self._curriculum_groups(track, lesson_idx)
            counts = {group: 0 for group in required}
            attempts = 0
            max_attempts = max(256, len(required) * episodes_per_group * 64)
            while any(counts[group] < episodes_per_group for group in required):
                if attempts >= max_attempts:
                    missing = {
                        group: episodes_per_group - count
                        for group, count in counts.items()
                        if count < episodes_per_group
                    }
                    raise RuntimeError(
                        f"could not balance {track} curriculum evaluation groups: {missing}"
                    )
                episode = self._make_episode(track, lesson_idx, rng)
                group = str(episode.metadata.get("progression_group", "all"))
                attempts += 1
                if group in counts and counts[group] < episodes_per_group:
                    episodes.append(episode)
                    counts[group] += 1
        return episodes

    def evaluate_curriculum_id(self, model, device, lesson_idx, *, field_log=None):
        rng = random.Random(91_000 + int(lesson_idx))
        episodes = self._make_curriculum_eval_episodes(int(lesson_idx), rng)
        self._last_id_eval_lesson = int(lesson_idx)
        self._last_id_eval_episodes = episodes
        result = self._evaluate_episodes(model, device, episodes, field_log=field_log)
        _t = (self.last_eval_diagnostics or {}).get("tracks", {})
        # self._track_weights = {t: 0.1 + max(0.0, 1.0 - m["exact_match"] / 100.0) for t, m in _t.items()}
        self._track_weights = {t: 0.05 for t in _t}
        _cand = [t for t in _t if t not in ("ignition", "masked_priming")]  # noise-ceiling tracks
        if _cand:
            _worst = min(_cand, key=lambda t: _t[t]["exact_match"])
            self._track_weights[_worst] = 0.05 * (len(_t) - 1)  # ~50% of steps
        for _t_name in ("ignition", "masked_priming"):
            if _t_name in self._track_weights:
                self._track_weights[_t_name] = 0.02
        return result

    def _paired_id_episodes(self, lesson_idx: int) -> list[TaskEpisode]:
        if self._last_id_eval_lesson == int(lesson_idx) and self._last_id_eval_episodes:
            return self._last_id_eval_episodes
        rng = random.Random(91_000 + int(lesson_idx))
        episodes = self._make_curriculum_eval_episodes(int(lesson_idx), rng)
        self._last_id_eval_lesson = int(lesson_idx)
        self._last_id_eval_episodes = episodes
        return episodes

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
        if self.task_track in MULTI_TRACKS:
            # A typed task marker allows one shared model to condition on the
            # operation grammar without relying on accidental token statistics.
            track_id = TRACKS.index(track)
            marker_prefix = (
                (
                    BOS,
                    track_token(0),
                    track_token(len(TRACK_TOKEN_IDS) - 1),
                    source_token(track_id - len(TRACK_TOKEN_IDS)),
                    SEP,
                )
                if track_id >= len(TRACK_TOKEN_IDS)
                else (BOS, track_token(track_id), SEP)
            )
            prefix_shift = len(marker_prefix) - 1
            episode = TaskEpisode(
                episode.track,
                marker_prefix + episode.inputs[1:],
                episode.answer,
                {
                    **episode.metadata,
                    "task_id": track_id,
                    "sequence_length": len(episode.inputs) + prefix_shift,
                    **(
                        {
                            "state_trace": tuple(
                                {**snapshot, "step": int(snapshot["step"]) + prefix_shift}
                                for snapshot in episode.metadata["state_trace"]
                            )
                        }
                        if track == "permutation"
                        else {}
                    ),
                    **(
                        {
                            "workspace_query_reads": tuple(
                                (int(step) + prefix_shift, int(key), int(value))
                                for step, key, value in episode.metadata[
                                    "workspace_query_reads"
                                ]
                            )
                        }
                        if episode.metadata.get("workspace_query_reads")
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
        if self.track_homogeneous_batches and len(tracks) > 1:
            # Keep all examples in an optimizer batch on one track. The track
            # cycles by global optimizer step, so a resumed run preserves the
            # schedule. This avoids padding every task to the longest episode
            # in a mixed batch while keeping long-run track exposure balanced.
            _w = getattr(self, "_track_weights", None)
            selected_track = (
                random.choices(tracks, [_w.get(t, 1.0) for t in tracks])[0]
                if _w
                else tracks[int(self.training_step) % len(tracks)]
            )
            selected_tracks = [selected_track] * batch_size
        else:
            # Cycle through tracks so their long-run share remains balanced;
            # every track appears in each batch when batch_size >= track count.
            start_track = random.randrange(len(tracks))
            selected_tracks = [tracks[(start_track + i) % len(tracks)] for i in range(batch_size)]
            random.shuffle(selected_tracks)
        episodes = [self._make_episode(track, level, random) for track in selected_tracks]
        self.last_batch_examples = episodes
        lengths = [len(episode.sequence) for episode in episodes]
        self.last_batch_sequence_mean = sum(lengths) / len(lengths)
        self.last_batch_sequence_max = max(lengths)
        valid_tokens = sum(lengths)
        padded_tokens = batch_size * self.last_batch_sequence_max
        self.last_batch_padding_utilization = valid_tokens / max(padded_tokens, 1)
        return self._collate(episodes, token_ids_only=self.token_id_input)

    @staticmethod
    def _collate(episodes: list[TaskEpisode], *, token_ids_only: bool = False):
        sequences = [episode.sequence for episode in episodes]
        max_len = max(map(len, sequences))
        batch = len(episodes)
        inputs = torch.zeros(
            (batch, max_len) if token_ids_only else (batch, max_len, VOCAB_SIZE),
            dtype=torch.long if token_ids_only else torch.float32,
        )
        targets = torch.zeros(batch, max_len, dtype=torch.long)
        mask = torch.zeros(batch, max_len, dtype=torch.float32)
        for row, (episode, sequence) in enumerate(zip(episodes, sequences)):
            row_token_ids = torch.tensor(sequence, dtype=torch.long)
            length = row_token_ids.numel()
            if token_ids_only:
                inputs[row, :length] = row_token_ids
            else:
                inputs[row, torch.arange(length), row_token_ids] = 1.0
            targets[row, :length] = torch.tensor(episode.targets, dtype=torch.long)
            mask[row, :length] = torch.tensor(episode.answer_mask, dtype=torch.float32)
        return inputs, targets, mask

    @staticmethod
    def _input_token_ids(input_vectors: torch.Tensor) -> torch.Tensor:
        """Accept either the standard one-hot batch or the opt-in compact IDs."""
        if input_vectors.ndim == 2 and input_vectors.dtype in (torch.int32, torch.int64):
            return input_vectors.to(dtype=torch.long)
        if input_vectors.ndim == 3 and input_vectors.shape[-1] == VOCAB_SIZE:
            return input_vectors.argmax(dim=-1)
        raise ValueError(
            "workspace-capability inputs must be token IDs shaped (batch,time) "
            "or one-hot vectors shaped (batch,time,vocabulary)"
        )

    @classmethod
    def workspace_input_token_rows(cls, input_vectors: torch.Tensor) -> list[list[int]]:
        """Decode the batch once for CPU-side causal metadata construction."""
        return cls._input_token_ids(input_vectors).detach().cpu().tolist()

    @staticmethod
    def _evaluation_groups(episodes, max_batch_size, *, kv_attention_heads=0):
        """Microbatch long KV evaluation streams within a conservative budget.

        The unfused SDPA fallback can materialize both scores and probabilities
        with shape ``[batch, heads, query, key]``. Estimate that workspace in
        fp32, sort by length to avoid padding short tasks to a long episode, and
        preserve the ordinary evaluation batch size for non-KV models.
        """
        if max_batch_size < 1:
            raise ValueError("evaluation batch size must be positive")
        if kv_attention_heads <= 0:
            for offset in range(0, len(episodes), max_batch_size):
                yield episodes[offset : offset + max_batch_size]
            return

        ordered = sorted(episodes, key=lambda episode: len(episode.sequence))
        offset = 0
        while offset < len(ordered):
            group = []
            maximum_length = 0
            while offset < len(ordered) and len(group) < max_batch_size:
                episode = ordered[offset]
                candidate_length = max(maximum_length, len(episode.sequence))
                estimated_bytes = (
                    (len(group) + 1)
                    * kv_attention_heads
                    * candidate_length
                    * candidate_length
                    * 4
                    * _SDPA_MATH_WORKSPACE_MULTIPLIER
                )
                if group and estimated_bytes > _EVAL_KV_ATTENTION_BUDGET_BYTES:
                    break
                group.append(episode)
                maximum_length = candidate_length
                offset += 1
            yield group

    def workspace_entity_address_metadata(
        self,
        input_vectors: torch.Tensor,
        *,
        paired_swap_commits: bool = False,
        attentional_item_memory: bool = False,
        attentional_cue_readout: bool = False,
        masked_prime_trace: bool = False,
        episodes: list[TaskEpisode] | None = None,
        token_rows: list[list[int]] | None = None,
    ) -> dict[str, Any]:
        """Build causal entity keys and event-boundary writes for each task track.

        Entity IDs are used only as workspace address/read metadata; they are
        not concatenated into backbone or specialist-value features and do not
        use answer labels. Permutation keeps its specialized initialization and
        paired-swap schedule. Other tracks write a selected entity address at a
        complete input-event boundary, after the event payload is visible.
        Query entity tokens never write.
        """
        token_ids = (
            token_rows
            if token_rows is not None
            else self._input_token_ids(input_vectors).detach().cpu().tolist()
        )
        examples = self.last_batch_examples if episodes is None else episodes
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
        for row, tokens in enumerate(token_ids):
            if self.task_track in MULTI_TRACKS:
                decoded_track_index, _prefix_length = track_marker_info(tokens)
                if decoded_track_index is None or decoded_track_index >= len(TRACKS):
                    continue
                track = TRACKS[decoded_track_index]
            else:
                track = self.task_track
            if track != "permutation":
                if track == "attentional_blink" and attentional_item_memory:
                    self._add_attentional_item_address_writes(
                        row, tokens, entity_ids, write_mask
                    )
                    continue
                if track == "rule_switching":
                    self._add_rule_switching_address_writes(row, tokens, entity_ids, write_mask)
                    continue
                if track == "capacity_bottleneck" and row < len(examples):
                    query_type = str(examples[row].metadata.get("query_type", ""))
                    if query_type.startswith("change_detection"):
                        self._add_capacity_change_address_writes(
                            row, tokens, entity_ids, write_mask
                        )
                        continue
                    if query_type.startswith("n_back"):
                        self._add_capacity_n_back_address_writes(
                            row,
                            tokens,
                            int(examples[row].metadata.get("n_back", 1)),
                            entity_ids,
                            write_mask,
                        )
                        continue
                self._add_event_boundary_entity_writes(row, tokens, track, entity_ids, write_mask)
                if (
                    track == "masked_priming"
                    and masked_prime_trace
                    and row < len(examples)
                    and bool(examples[row].metadata.get("mask_condition", False))
                ):
                    self._add_masked_prime_trace_write(row, tokens, entity_ids, write_mask)
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
                        if not (paired_swap_commits or getattr(self, "v3_swap_operator", False)):
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
        metadata: dict[str, Any] = {
            "entity_ids": entity_ids,
            "entity_write_mask": write_mask,
        }
        if paired_swap_commits:
            metadata["pair_swap_entity_ids"] = pair_swap_ids
            metadata["pair_swap_mask"] = pair_swap_mask
            metadata["pair_swap_steps"] = tuple(
                int(step)
                for step in torch.nonzero(pair_swap_mask.any(dim=0), as_tuple=False)
                .flatten()
                .tolist()
            )
        if attentional_cue_readout:
            cue_metadata = self.workspace_attentional_cue_metadata(
                input_vectors, episodes=examples
            )
            metadata.update(cue_metadata)
        return metadata

    @staticmethod
    def _add_event_boundary_entity_writes(
        row: int,
        tokens: list[int],
        track: str,
        entity_ids: torch.Tensor,
        write_mask: torch.Tensor,
    ) -> None:
        """Address the entity represented by each complete, supported event.

        One entity slot is used per task entity. Relational facts write the
        subject address when the object token completes the edge, then write
        the object address at the separator. This exposes both the forward
        edge and its contextual reverse without adding oracle targets.
        """
        if track == "belief_tracking":
            WorkspaceCapabilityDataset._add_belief_address_writes(
                row, tokens, entity_ids, write_mask
            )
            return
        if track == "mixed_skill":
            WorkspaceCapabilityDataset._add_mixed_skill_address_writes(
                row, tokens, entity_ids, write_mask
            )
            return
        event_tokens = _ENTITY_ADDRESS_EVENT_TOKENS.get(track, set())
        if not event_tokens:
            return

        active_event = None
        event_entities: list[int] = []
        has_location = False
        has_scalar = False
        has_relation = False
        has_boolean = False
        for position, token_value in enumerate(tokens):
            token = int(token_value)
            if ENTITY_BASE <= token < ENTITY_BASE + ENTITY_COUNT:
                # Semantic IDs select reads everywhere in the stream, while
                # the separate write mask stays limited to valid event commits.
                entity_ids[row, position] = token - ENTITY_BASE
            if token == QUERY:
                active_event = None
                event_entities.clear()
                continue
            if active_event is None:
                if token in event_tokens:
                    active_event = token
                    event_entities.clear()
                    has_location = False
                    has_scalar = False
                    has_relation = False
                    has_boolean = False
                continue
            if token == SEP:
                if event_entities:
                    if track == "entity_state":
                        valid_event = has_location
                    elif track == "relational_join":
                        valid_event = len(event_entities) >= 2 and has_relation
                    elif track == "nth_farthest":
                        valid_event = has_scalar
                    elif track == "capacity_bottleneck":
                        valid_event = has_location
                    elif track == "rule_switching":
                        valid_event = has_scalar or has_boolean
                    elif track == "variable_binding":
                        valid_event = has_scalar or len(event_entities) >= 2
                    elif track == "trace_conditioning":
                        valid_event = has_scalar or has_boolean
                    else:
                        valid_event = has_scalar
                    if valid_event:
                        entity_ids[row, position] = (
                            event_entities[-1] if track == "relational_join" else event_entities[0]
                        )
                        write_mask[row, position] = True
                active_event = None
                event_entities.clear()
                continue
            if ENTITY_BASE <= token < ENTITY_BASE + ENTITY_COUNT:
                event_entities.append(token - ENTITY_BASE)
                if track == "relational_join" and len(event_entities) == 2:
                    # The causal representation now contains the complete edge.
                    entity_ids[row, position] = event_entities[0]
                    write_mask[row, position] = True
            elif LOCATION_BASE <= token < LOCATION_BASE + LOCATION_COUNT:
                has_location = True
            elif SCALAR_BASE <= token < SCALAR_BASE + SCALAR_COUNT:
                has_scalar = True
            elif RELATION_BASE <= token < RELATION_BASE + RELATION_COUNT:
                has_relation = True
            elif token in (BOOL_FALSE, BOOL_TRUE):
                has_boolean = True

    @staticmethod
    def _add_masked_prime_trace_write(row, tokens, entity_ids, write_mask) -> None:
        """Store the masked prime in its own slot when the opt-in trace is enabled."""
        in_prime = False
        for position, token_value in enumerate(tokens):
            token = int(token_value)
            if token == EV_NOISE and not in_prime:
                in_prime = True
            elif in_prime and token == SEP:
                # Entity zero is reserved for the prime; the visible target
                # occupies entity one in this task.
                entity_ids[row, position] = 0
                write_mask[row, position] = True
                return
            elif in_prime and token == QUERY:
                return

    @staticmethod
    def _add_belief_address_writes(row, tokens, entity_ids, write_mask) -> None:
        """Address world state separately from each agent/object belief pair."""
        active_event = None
        event_agent = None
        event_object = None
        has_location = False
        query_agent = None
        querying = False
        for position, token_value in enumerate(tokens):
            token = int(token_value)
            if token == QUERY:
                active_event = None
                event_agent = None
                event_object = None
                has_location = False
                querying = True
                query_agent = None
            if token in (EV_INIT, EV_MOVE, EV_ASSERT):
                active_event = token
                event_agent = None
                event_object = None
                has_location = False
                querying = False
                continue
            if querying and SOURCE_BASE <= token < SOURCE_BASE + SOURCE_COUNT:
                query_agent = token - SOURCE_BASE
                continue
            if active_event == EV_ASSERT and SOURCE_BASE <= token < SOURCE_BASE + SOURCE_COUNT:
                event_agent = token - SOURCE_BASE
                continue
            if ENTITY_BASE <= token < ENTITY_BASE + ENTITY_COUNT:
                object_id = token - ENTITY_BASE
                if querying:
                    key = 8 + query_agent * 8 + object_id if query_agent is not None else object_id
                elif active_event == EV_ASSERT and event_agent is not None:
                    key = 8 + event_agent * 8 + object_id
                    event_object = object_id
                else:
                    key = object_id
                    event_object = object_id
                if 0 <= key < ENTITY_COUNT:
                    entity_ids[row, position] = key
            elif LOCATION_BASE <= token < LOCATION_BASE + LOCATION_COUNT:
                has_location = True
            elif token == SEP:
                if active_event in (EV_INIT, EV_MOVE) and event_object is not None and has_location:
                    entity_ids[row, position] = event_object
                    write_mask[row, position] = True
                elif (
                    active_event == EV_ASSERT
                    and event_agent is not None
                    and event_object is not None
                    and has_location
                ):
                    key = 8 + event_agent * 8 + event_object
                    entity_ids[row, position] = key
                    write_mask[row, position] = True
                active_event = None
                event_agent = None
                event_object = None
                has_location = False
                querying = False
                query_agent = None

    @staticmethod
    def _add_rule_switching_address_writes(row, tokens, entity_ids, write_mask) -> None:
        """Keep the inferred task-set under one stable semantic address."""
        active_event = None
        event_entities: list[int] = []
        scalar_count = 0
        for position, token_value in enumerate(tokens):
            token = int(token_value)
            if ENTITY_BASE <= token < ENTITY_BASE + ENTITY_COUNT:
                entity_ids[row, position] = token - ENTITY_BASE
                if active_event is not None:
                    event_entities.append(token - ENTITY_BASE)
            if token in (EV_UPDATE, EV_ASSERT, EV_VECTOR):
                active_event = token
                event_entities = []
                scalar_count = 0
                continue
            if SCALAR_BASE <= token < SCALAR_BASE + SCALAR_COUNT:
                if active_event == EV_VECTOR and scalar_count < 3:
                    entity_ids[row, position] = scalar_count
                    write_mask[row, position] = True
                scalar_count += 1
            if token == SEP:
                if active_event in (EV_UPDATE, EV_ASSERT):
                    entity_ids[row, position] = 63
                    write_mask[row, position] = True
                active_event = None
                event_entities = []
                scalar_count = 0

    @staticmethod
    def _add_capacity_change_address_writes(row, tokens, entity_ids, write_mask) -> None:
        """Preserve initial probe values beside the current entity value."""
        active_event = None
        pending_entity = -1
        has_location = False
        for position, token_value in enumerate(tokens):
            token = int(token_value)
            if token in (EV_INIT, EV_UPDATE):
                active_event = token
                pending_entity = -1
                has_location = False
            elif ENTITY_BASE <= token < ENTITY_BASE + ENTITY_COUNT:
                pending_entity = token - ENTITY_BASE
                entity_ids[row, position] = pending_entity
            elif LOCATION_BASE <= token < LOCATION_BASE + LOCATION_COUNT:
                has_location = True
            elif token == SEP:
                if active_event in (EV_INIT, EV_UPDATE) and pending_entity >= 0 and has_location:
                    key = pending_entity + 32 if active_event == EV_INIT else pending_entity
                    entity_ids[row, position] = key
                    write_mask[row, position] = True
                active_event = None
                pending_entity = -1
                has_location = False

    @staticmethod
    def _add_capacity_n_back_address_writes(row, tokens, n_back, entity_ids, write_mask) -> None:
        """Keep only the two symbols the n-back decision must compare."""
        query_index = next(
            (index for index, token in enumerate(tokens) if int(token) == QUERY), len(tokens)
        )
        value_positions = [
            index
            for index, token in enumerate(tokens[:query_index])
            if LOCATION_BASE <= int(token) < LOCATION_BASE + LOCATION_COUNT
        ]
        if len(value_positions) <= n_back:
            return
        positions = (value_positions[-1 - n_back], value_positions[-1])
        for key, position in zip((60, 61), positions):
            entity_ids[row, position] = key
            write_mask[row, position] = True

    @staticmethod
    def _add_mixed_skill_address_writes(row, tokens, entity_ids, write_mask) -> None:
        """Store initial table values and each operation result by stage ID."""
        active_event = None
        event_entities = []
        has_scalar = False
        has_source = False
        for position, token_value in enumerate(tokens):
            token = int(token_value)
            if ENTITY_BASE <= token < ENTITY_BASE + ENTITY_COUNT:
                entity_id = token - ENTITY_BASE
                entity_ids[row, position] = entity_id
                if active_event is not None:
                    event_entities.append(entity_id)
            if token == QUERY:
                active_event = None
                event_entities.clear()
                continue
            if active_event is None:
                if token in (EV_KV, EV_VECTOR):
                    active_event = token
                    event_entities.clear()
                    has_scalar = False
                    has_source = False
                continue
            if SOURCE_BASE <= token < SOURCE_BASE + SOURCE_COUNT:
                has_source = True
            elif SCALAR_BASE <= token < SCALAR_BASE + SCALAR_COUNT:
                has_scalar = True
            elif token == SEP:
                if event_entities:
                    if active_event == EV_KV and has_scalar:
                        address = event_entities[0]
                        entity_ids[row, position] = address
                        write_mask[row, position] = True
                    elif active_event == EV_VECTOR and has_source:
                        address = event_entities[0]
                        entity_ids[row, position] = address
                        write_mask[row, position] = True
                active_event = None
                event_entities.clear()
                has_scalar = False
                has_source = False

    def workspace_permutation_query_metadata(
        self,
        input_vectors: torch.Tensor,
        episodes: list[TaskEpisode] | None = None,
    ) -> dict[str, Any]:
        """Mark the first answer position with its causal permutation query."""
        examples = self.last_batch_examples if episodes is None else episodes
        batch, time = input_vectors.shape[:2]
        if len(examples) != batch:
            raise RuntimeError("permutation query metadata lost batch/episode alignment")
        query_kind = torch.zeros((batch, time), dtype=torch.long, device=input_vectors.device)
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
    def _attentional_item_frame_records(tokens: list[int]):
        """Parse cue-independent RSVP frame fields and their ordinal addresses."""
        cue_tokens: list[int] = []
        cue_complete = False
        frame_sources: list[int] = []
        frame_value = -1
        ignore_mask_frame = False
        frame_index = 0
        records: list[tuple[int, int, int, int, int]] = []
        for position, token_value in enumerate(tokens):
            token = int(token_value)
            if not cue_complete:
                if SOURCE_BASE <= token < SOURCE_BASE + SOURCE_COUNT:
                    cue_tokens.append(token)
                elif token == SEP and len(cue_tokens) >= 2:
                    cue_complete = True
                continue
            if token == QUERY:
                break
            if token == EV_NOISE:
                ignore_mask_frame = True
                continue
            if SOURCE_BASE <= token < SOURCE_BASE + SOURCE_COUNT:
                frame_sources.append(token)
            elif SCALAR_BASE <= token < SCALAR_BASE + SCALAR_COUNT:
                frame_value = token
            elif token == SEP:
                if not ignore_mask_frame and len(frame_sources) == 2 and frame_value >= 0:
                    records.append(
                        (position, frame_index, frame_sources[0], frame_sources[1], frame_value)
                    )
                frame_index += 1
                frame_sources.clear()
                frame_value = -1
                ignore_mask_frame = False
        return records

    @staticmethod
    def _add_attentional_item_address_writes(row, tokens, entity_ids, write_mask) -> None:
        """Address every RSVP item by stream order, without target-position labels."""
        for position, frame_index, _color, _shape, _value in (
            WorkspaceCapabilityDataset._attentional_item_frame_records(tokens)
        ):
            if frame_index >= ENTITY_COUNT:
                continue
            entity_ids[row, position] = frame_index
            write_mask[row, position] = True

    def workspace_attentional_cue_metadata(
        self,
        input_vectors: torch.Tensor,
        *,
        episodes: list[TaskEpisode] | None = None,
    ) -> dict[str, Any]:
        """Build cue reads from visible cue tokens, never from answer labels."""
        examples = self.last_batch_examples if episodes is None else episodes
        if len(examples) != input_vectors.shape[0]:
            raise RuntimeError("attentional cue metadata lost batch/episode alignment")
        token_rows = self._input_token_ids(input_vectors).detach().cpu().tolist()
        color_ids = torch.full((len(examples),), -1, dtype=torch.long)
        shape_ids = torch.full((len(examples),), -1, dtype=torch.long)
        query_step_ids = torch.full((len(examples),), -1, dtype=torch.long)
        frame_entity_ids = torch.full(input_vectors.shape[:2], -1, dtype=torch.long)
        frame_mask = torch.zeros(input_vectors.shape[:2], dtype=torch.bool)
        query_steps: set[int] = set()
        frame_steps: set[int] = set()
        for row, (episode, tokens) in enumerate(zip(examples, token_rows)):
            if episode.track != "attentional_blink":
                continue
            for position, frame_index, _color, _shape, _value in (
                self._attentional_item_frame_records(tokens)
            ):
                if frame_index < ENTITY_COUNT and position < input_vectors.shape[1]:
                    frame_entity_ids[row, position] = frame_index
                    frame_mask[row, position] = True
                    frame_steps.add(position)
            cue = []
            _track_id, prefix_length = track_marker_info(tokens)
            cue_tokens = tokens[prefix_length:] if prefix_length else tokens
            for token_value in cue_tokens:
                token = int(token_value)
                # Begin after the explicit multi-track marker, if present.
                if token == SEP and len(cue) >= 2:
                    break
                if SOURCE_BASE <= token < SOURCE_BASE + SOURCE_COUNT:
                    cue.append(token)
                    if len(cue) == 2:
                        break
            if len(cue) != 2:
                raise RuntimeError("attentional-blink input is missing its two-token cue")
            color_ids[row], shape_ids[row] = cue
            query_step = len(episode.inputs)
            query_step_ids[row] = query_step
            query_steps.add(query_step)
        return {
            "attentional_cue_color_ids": color_ids,
            "attentional_cue_shape_ids": shape_ids,
            "attentional_cue_query_step_ids": query_step_ids,
            "attentional_cue_readout_steps": tuple(sorted(query_steps)),
            "attentional_frame_entity_ids": frame_entity_ids,
            "attentional_frame_mask": frame_mask,
            "attentional_frame_steps": tuple(sorted(frame_steps)),
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
            position_vocab_logits = result.new_full((result.shape[0], result.shape[-1]), -1e4)
            position_vocab_logits[:, LOCATION_BASE : LOCATION_BASE + LOCATION_COUNT] = (
                position_logits
            )
            entity_vocab_logits = result.new_full((result.shape[0], result.shape[-1]), -1e4)
            entity_vocab_logits[:, ENTITY_BASE : ENTITY_BASE + ENTITY_COUNT] = entity_logits
            original = logits[:, int(step)]
            updated = torch.where(query_kind.eq(1).unsqueeze(-1), position_vocab_logits, original)
            updated = torch.where(query_kind.eq(2).unsqueeze(-1), entity_vocab_logits, updated)
            result[:, int(step)] = updated
        return result

    @staticmethod
    def apply_workspace_attentional_cue_readout_logits(logits, predictions):
        """Decode T1/T2 from cue-matched, workspace-stored RSVP item slots."""
        if not predictions:
            return logits
        result = logits.clone()
        for step, first_logits, second_logits, active in predictions:
            active = active.to(device=logits.device, dtype=torch.bool)
            for offset, value_logits in enumerate((first_logits, second_logits)):
                output_step = int(step) + offset
                if not 0 <= output_step < logits.shape[1]:
                    continue
                value_logits = value_logits.to(device=logits.device, dtype=logits.dtype)
                selected_logits = result.new_full(
                    (result.shape[0], result.shape[-1]), -1e4
                )
                selected_logits[:, SCALAR_BASE : SCALAR_BASE + SCALAR_COUNT] = value_logits
                result[:, output_step] = torch.where(
                    active.unsqueeze(-1), selected_logits, result[:, output_step]
                )
        return result

    @staticmethod
    def apply_workspace_retrieval_query_logits(
        logits,
        query_predictions,
        metadata: dict[str, torch.Tensor],
        episodes: list[TaskEpisode],
    ):
        """Apply exact aggregation to values decoded from addressed workspace slots."""
        totals: dict[str, Any] = {
            "write_value_correct": 0,
            "write_value_count": 0,
            "query_values_correct": 0,
            "query_values_count": 0,
            "query_slots_found": 0,
            "query_slots_total": 0,
            "updated_query_values_correct": 0,
            "updated_query_values_count": 0,
            "untouched_query_values_correct": 0,
            "untouched_query_values_count": 0,
            "readout_exact": 0,
            "readout_count": 0,
            "invalid_readouts": 0,
            "readout_by_query_type": {},
        }
        if not query_predictions:
            return logits, totals
        result = logits.clone()
        targets = metadata["retrieval_query_targets"]
        was_updated = metadata["retrieval_query_was_updated"]
        operations = metadata["retrieval_query_operation"]
        for step, value_logits, found, active_mask in query_predictions:
            value_ids = value_logits.argmax(dim=-1)
            found = found.bool() & active_mask.bool()
            active_mask = active_mask.bool()
            expected_values = targets[:, int(step)].to(value_ids.device)
            compare = active_mask & expected_values.ge(0)
            totals["query_values_correct"] += int(
                (value_ids.eq(expected_values) & compare).sum().item()
            )
            totals["query_values_count"] += int(compare.sum().item())
            updated = was_updated[:, int(step)].to(value_ids.device).bool() & compare
            untouched = (~was_updated[:, int(step)].to(value_ids.device).bool()) & compare
            totals["updated_query_values_correct"] += int(
                (value_ids.eq(expected_values) & updated).sum().item()
            )
            totals["updated_query_values_count"] += int(updated.sum().item())
            totals["untouched_query_values_correct"] += int(
                (value_ids.eq(expected_values) & untouched).sum().item()
            )
            totals["untouched_query_values_count"] += int(untouched.sum().item())
            totals["query_slots_found"] += int(found.sum().item())
            totals["query_slots_total"] += int(active_mask.sum().item())
            valid_rows = active_mask.any(dim=-1) & (~active_mask | found).all(dim=-1)
            operation_ids = operations[:, int(step)].tolist()
            values_cpu = value_ids.tolist()
            active_cpu = active_mask.tolist()
            valid_cpu = valid_rows.tolist()
            for row, episode in enumerate(episodes):
                if episode.track != "retrieval_aggregation" or not active_cpu[row]:
                    continue
                totals["readout_count"] += 1
                query_type = str(episode.metadata.get("query_type", "unknown"))
                stratum = totals["readout_by_query_type"].setdefault(
                    query_type, {"correct": 0, "episodes": 0}
                )
                stratum["episodes"] += 1
                if not valid_cpu[row]:
                    totals["invalid_readouts"] += 1
                    continue
                values = [
                    int(values_cpu[row][index])
                    for index, is_active in enumerate(active_cpu[row])
                    if is_active
                ]
                operation = int(operation_ids[row])
                if operation == 1:
                    output_tokens = [SCALAR_BASE + value for value in values] + [EOS]
                elif operation == 2:
                    output_tokens = [SCALAR_BASE + (sum(values) % SCALAR_COUNT), EOS]
                elif operation == 3:
                    output_tokens = [COUNT_BASE + sum(value % 2 == 0 for value in values), EOS]
                elif operation == 4:
                    output_tokens = [BOOL_TRUE if sum(values) % 2 else BOOL_FALSE, EOS]
                else:
                    raise ValueError(f"unknown retrieval query operation ID {operation}")
                answer_start = int(step)
                answer_end = answer_start + len(output_tokens)
                if answer_end > result.shape[1]:
                    raise RuntimeError("retrieval readout answer exceeds the output sequence")
                for offset, token in enumerate(output_tokens):
                    result[row, answer_start + offset].fill_(-1e4)
                    result[row, answer_start + offset, token] = 1e4
                correct_readout = int(tuple(output_tokens) == tuple(episode.answer))
                totals["readout_exact"] += correct_readout
                stratum["correct"] += correct_readout
        return result, totals

    def workspace_state_supervision_metadata(
        self,
        input_vectors: torch.Tensor,
        episodes: list[TaskEpisode] | None = None,
    ) -> dict[str, Any]:
        """Pack oracle entity positions only at causal state-commit boundaries."""
        batch, time = input_vectors.shape[:2]
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

    @staticmethod
    def _workspace_operation_events(episode: TaskEpisode, tokens: list[int]):
        """Parse supported input programs and simulate labels for sparse loss only."""
        if episode.track == "retrieval_aggregation":
            return WorkspaceCapabilityDataset._retrieval_reduce_records(episode, tokens)
        if episode.track not in ("mixed_skill", "variable_binding", "permutation"):
            return ()
        current_values: dict[int, int] = {}
        records = []
        index = 0
        while index < len(episode.inputs):
            event = int(tokens[index])
            if event not in (EV_INIT, EV_UPDATE, EV_KV, EV_VECTOR, EV_SWAP):
                index += 1
                continue
            end = index + 1
            while end < len(episode.inputs) and int(tokens[end]) != SEP:
                end += 1
            if end >= len(episode.inputs):
                break
            fields = [int(value) for value in tokens[index + 1 : end]]
            entity_fields = [
                value - ENTITY_BASE
                for value in fields
                if ENTITY_BASE <= value < ENTITY_BASE + ENTITY_COUNT
            ]
            scalar_fields = [
                value
                for value in fields
                if SCALAR_BASE <= value < SCALAR_BASE + SCALAR_COUNT
            ]
            step = end
            if episode.track == "permutation":
                if event == EV_INIT and entity_fields:
                    loc = next((v for v in fields if LOCATION_BASE <= v < LOCATION_BASE + LOCATION_COUNT), None)
                    if loc is not None:
                        current_values[entity_fields[0]] = loc
                elif event == EV_SWAP and len(entity_fields) >= 2:
                    left, right = entity_fields[:2]
                    lv, rv = current_values[left], current_values[right]
                    records.append(
                        (step, "swap", left, right, (_WORKSPACE_OPERAND_KEY, left),
                         (_WORKSPACE_OPERAND_KEY, right), (rv, lv), (lv, rv))
                    )
                    current_values[left], current_values[right] = rv, lv
                index = end + 1
                continue
            if episode.track == "mixed_skill":
                if event == EV_KV and entity_fields and scalar_fields:
                    current_values[entity_fields[0]] = scalar_fields[0]
                elif event == EV_VECTOR and len(entity_fields) >= 2:
                    if len(fields) < 2 or not SOURCE_BASE <= fields[1] < SOURCE_BASE + SOURCE_COUNT:
                        raise ValueError("mixed-skill operation is missing its skill code")
                    skill = fields[1] - SOURCE_BASE
                    destination = entity_fields[0]
                    operands = entity_fields[1:]
                    literal = scalar_fields[0] if scalar_fields else -1
                    if skill == 0 and operands:
                        operation = "lookup"
                        operand0 = (_WORKSPACE_OPERAND_KEY, operands[0])
                        operand1 = (0, -1)
                        outputs = (current_values[operands[0]],)
                        expected_operands = (current_values[operands[0]], -100)
                    elif skill == 1 and operands and literal >= 0:
                        operation = "add"
                        operand0 = (_WORKSPACE_OPERAND_KEY, operands[0])
                        operand1 = (_WORKSPACE_OPERAND_LITERAL, literal)
                        outputs = (
                            SCALAR_BASE
                            + ((current_values[operands[0]] - SCALAR_BASE + literal - SCALAR_BASE)
                               % SCALAR_COUNT),
                        )
                        expected_operands = (current_values[operands[0]], literal)
                    elif skill == 2 and operands:
                        operation = "copy"
                        operand0 = (_WORKSPACE_OPERAND_KEY, operands[0])
                        operand1 = (0, -1)
                        outputs = (current_values[operands[0]],)
                        expected_operands = (current_values[operands[0]], -100)
                    elif skill == 3 and operands and literal >= 0:
                        operation = "sort"
                        operand0 = (_WORKSPACE_OPERAND_KEY, operands[0])
                        operand1 = (_WORKSPACE_OPERAND_LITERAL, literal)
                        outputs = tuple(
                            sorted((current_values[operands[0]], literal))
                        )
                        expected_operands = (current_values[operands[0]], literal)
                    elif skill == 4 and operands and literal >= 0:
                        operation = "compare"
                        operand0 = (_WORKSPACE_OPERAND_KEY, operands[0])
                        operand1 = (_WORKSPACE_OPERAND_LITERAL, literal)
                        outputs = (
                            SCALAR_BASE
                            + int(current_values[operands[0]] > literal),
                        )
                        expected_operands = (current_values[operands[0]], literal)
                    elif skill == 5 and operands and literal >= 0:
                        operation = "select"
                        operand0 = (_WORKSPACE_OPERAND_KEY, operands[0])
                        operand1 = (_WORKSPACE_OPERAND_LITERAL, literal)
                        outputs = (max(current_values[operands[0]], literal),)
                        expected_operands = (current_values[operands[0]], literal)
                    elif skill == 6 and operands and literal >= 0:
                        operation = "rank"
                        operand0 = (_WORKSPACE_OPERAND_KEY, operands[0])
                        operand1 = (_WORKSPACE_OPERAND_LITERAL, literal)
                        outputs = (
                            SCALAR_BASE
                            + int(current_values[operands[0]] < literal),
                        )
                        expected_operands = (current_values[operands[0]], literal)
                    else:
                        raise ValueError(f"invalid mixed-skill operation code {skill}")
                    destination1 = destination + 1 if operation == "sort" else -1
                    records.append(
                        (
                            step,
                            operation,
                            destination,
                            destination1,
                            operand0,
                            operand1,
                            outputs,
                            expected_operands,
                        )
                    )
                    current_values[destination] = outputs[0]
                    if destination1 >= 0:
                        current_values[destination1] = outputs[1]
            else:
                if event in (EV_INIT, EV_UPDATE) and entity_fields and scalar_fields:
                    current_values[entity_fields[0]] = scalar_fields[0]
                elif event == EV_KV and len(entity_fields) >= 2:
                    destination, source = entity_fields[:2]
                    value = current_values[source]
                    records.append(
                        (
                            step,
                            "copy",
                            destination,
                            -1,
                            (_WORKSPACE_OPERAND_KEY, source),
                            (0, -1),
                            (value,),
                            (value, -100),
                        )
                    )
                    current_values[destination] = value
                elif event == EV_VECTOR and len(entity_fields) >= 3:
                    destination, left, right = entity_fields[:3]
                    if not any(
                        SOURCE_BASE <= value < SOURCE_BASE + SOURCE_COUNT
                        and value - SOURCE_BASE == 4
                        for value in fields
                    ):
                        raise ValueError("variable-binding compute operation is not modular add")
                    left_value = current_values[left] - SCALAR_BASE
                    right_value = current_values[right] - SCALAR_BASE
                    result = SCALAR_BASE + ((left_value + right_value) % SCALAR_COUNT)
                    records.append(
                        (
                            step,
                            "add",
                            destination,
                            -1,
                            (_WORKSPACE_OPERAND_KEY, left),
                            (_WORKSPACE_OPERAND_KEY, right),
                            (result,),
                            (current_values[left], current_values[right]),
                        )
                    )
                    current_values[destination] = result
            index = end + 1
        return tuple(records)


    @staticmethod
    def _retrieval_reduce_records(episode, tokens):
        """Reduce teacher record for retrieval aggregate queries (sum/count_even/parity)."""
        n = len(episode.inputs)
        q = next((i for i in range(n) if int(tokens[i]) == QUERY), None)
        kinds = {EV_SUM_QUERY: 0, EV_COUNT_QUERY: 1, EV_PARITY_QUERY: 2}
        if q is None or q + 1 >= n or int(tokens[q + 1]) not in kinds:
            return ()
        return (
            (
                n - 1, "reduce", _REDUCE_RESULT_KEY, -1, (0, -1), (0, -1),
                (int(episode.answer[0]),), (-100, kinds[int(tokens[q + 1])]),
            ),
        )

    @staticmethod
    def _rule_feedback_targets(episode: TaskEpisode, tokens: list[int]) -> tuple[int, ...]:
        """Infer a current rule from visible cues and noisy choice feedback only."""
        import math

        log_scores = [0.0, 0.0, 0.0]
        feedback_targets: list[int] = []
        index = 0
        while index < len(episode.inputs):
            event = int(tokens[index])
            end = index + 1
            while end < len(episode.inputs) and int(tokens[end]) != SEP:
                end += 1
            fields = [int(value) for value in tokens[index + 1 : end]]
            if event == EV_UPDATE:
                cue = next(
                    (
                        value - SOURCE_BASE
                        for value in fields
                        if SOURCE_BASE <= value < SOURCE_BASE + 3
                    ),
                    None,
                )
                if cue is not None:
                    # A visible cue is an explicit task-set update. Give it a
                    # finite prior so subsequent feedback can still correct it.
                    log_scores = [-3.0, -3.0, -3.0]
                    log_scores[cue] = 0.0
            elif event == EV_ASSERT:
                choice = next(
                    (
                        value - SOURCE_BASE
                        for value in fields
                        if SOURCE_BASE <= value < SOURCE_BASE + 3
                    ),
                    None,
                )
                feedback = next(
                    (value == BOOL_TRUE for value in fields if value in (BOOL_FALSE, BOOL_TRUE)),
                    None,
                )
                if choice is not None and feedback is not None:
                    # The task's feedback noise is variable and not provided to
                    # the updater; a conservative fixed likelihood prevents a
                    # single noisy observation from dominating. Exponential
                    # decay lets later evidence recover after unannounced shifts.
                    log_scores = [0.72 * value for value in log_scores]
                    for hypothesis in range(3):
                        predicts_success = hypothesis == choice
                        log_scores[hypothesis] += math.log(
                            0.8 if feedback == predicts_success else 0.2
                        )
                    maximum = max(log_scores)
                    winners = [i for i, value in enumerate(log_scores) if value == maximum]
                    feedback_targets.append(winners[0])
            index = end + 1
        return tuple(feedback_targets)

    def workspace_key_value_metadata(
        self,
        input_vectors: torch.Tensor,
        episodes: list[TaskEpisode] | None = None,
        *,
        include_operations: bool = False,
        include_adaptive_v3: bool = False,
        adaptive_v3_distillation: bool = True,
        adaptive_v3_feature_off_probability: float = 0.0,
        adaptive_v3_feature_off_tracks: tuple[str, ...] = ("mixed_skill", "variable_binding"),
        adaptive_v3_always_feature_off_tracks: tuple[str, ...] = (),
        adaptive_v3_holdout_tracks: tuple[str, ...] = (),
        adaptive_v3_step: int = 0,
        adaptive_v3_seed: int = 0,
        include_operation_diagnostics: bool = False,
        rule_feedback_update: bool = False,
        attentional_item_memory: bool = False,
        no_report_confidence_readout: bool = False,
        masked_prime_trace: bool = False,
        entity_metadata: dict[str, Any] | None = None,
        token_rows: list[list[int]] | None = None,
    ) -> dict[str, Any]:
        """Pack sparse semantic value reads for causal updates and answer queries.

        Episodes may provide ``workspace_output_keys`` aligned to their answer
        tokens. Simple event updates with an explicit categorical payload are
        also supervised at their commit boundary. Read width is statically
        bounded; task aggregators can use a wider answer row without allocating
        sequence-length-sized intermediates.
        """
        examples = self.last_batch_examples if episodes is None else episodes
        batch, time = input_vectors.shape[:2]
        if len(examples) != batch:
            raise RuntimeError("workspace key-value metadata lost batch/episode alignment")
        operand_capacity = (
            KEY_VALUE_OPERAND_CAPACITY
        )
        entity_meta = entity_metadata or self.workspace_entity_address_metadata(
            input_vectors,
            episodes=examples,
            attentional_item_memory=attentional_item_memory,
            masked_prime_trace=masked_prime_trace,
            token_rows=token_rows,
        )
        token_rows = (
            token_rows
            if token_rows is not None
            else self._input_token_ids(input_vectors).detach().cpu().tolist()
        )
        entity_rows = entity_meta["entity_ids"].detach().cpu().tolist()
        write_rows = entity_meta["entity_write_mask"].detach().cpu().tolist()
        key_ids = torch.full((batch, time, operand_capacity), -1, dtype=torch.long)
        targets = torch.full((batch, time, operand_capacity), -100, dtype=torch.long)
        masks = torch.zeros((batch, time, operand_capacity), dtype=torch.bool)
        attentional_color_targets = torch.full((batch, time), -100, dtype=torch.long)
        attentional_shape_targets = torch.full((batch, time), -100, dtype=torch.long)
        operation_metadata = (
            torch.full((batch, time, 7), -1, dtype=torch.long)
            if include_operations or include_adaptive_v3
            else None
        )
        adaptive_teacher_targets = (
            torch.full((batch, time, 12), -100, dtype=torch.long)
            if include_adaptive_v3 and adaptive_v3_distillation
            else None
        )
        adaptive_candidate_mask = (
            torch.zeros((batch, time), dtype=torch.bool) if include_adaptive_v3 else None
        )
        adaptive_execution_mask = (
            torch.ones((batch, time), dtype=torch.bool) if include_adaptive_v3 else None
        )
        adaptive_steps: set[int] = set()
        adaptive_teacher_steps: set[int] = set()
        adaptive_v3_batch_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "episodes": 0,
                "manual_off": 0,
                "lookup": 0,
                "copy": 0,
                "add": 0,
                "sort": 0,
                "compare": 0,
                "select": 0,
                "rank": 0,
                "swap": 0,
                "reduce": 0,
            }
        )
        operation_diagnostic_targets = (
            torch.full((batch, time, 4), -100, dtype=torch.long)
            if include_operation_diagnostics
            else None
        )
        operation_steps: set[int] = set()
        read_steps: set[int] = set()
        output_program_specs: list[tuple[int, int, str, int]] = []
        argmatch_locations: dict[tuple[int, int], int] = {}

        for row, (episode, tokens, event_keys, event_writes) in enumerate(
            zip(examples, token_rows, entity_rows, write_rows)
        ):
            per_step: dict[int, list[tuple[int, int]]] = {}
            operation_records = (
                self._workspace_operation_events(episode, tokens)
                if include_operations
                or include_adaptive_v3
                else ()
            )
            feature_off = bool(
                include_adaptive_v3 and episode.track in adaptive_v3_holdout_tracks
            )
            if include_adaptive_v3 and (
                adaptive_v3_feature_off_probability > 0
                or adaptive_v3_always_feature_off_tracks
            ):
                is_holdout = episode.track in adaptive_v3_holdout_tracks
                always_off = episode.track in adaptive_v3_always_feature_off_tracks
                rng = random.Random(int(adaptive_v3_seed) + int(adaptive_v3_step) * 1_000_003 + row * 97)
                sampled_off = (
                    episode.track in adaptive_v3_feature_off_tracks
                    and rng.random() < adaptive_v3_feature_off_probability
                )
                feature_off = is_holdout or always_off or sampled_off
            if adaptive_candidate_mask is not None:
                stats = adaptive_v3_batch_stats[episode.track]
                stats["episodes"] += 1
                stats["manual_off"] += int(feature_off)
                pending_operator_event = False
                for candidate_step, token_value in enumerate(tokens[: min(time, len(episode.inputs))]):
                    token = int(token_value)
                    if token in (EV_INIT, EV_MOVE, EV_UPDATE, EV_ASSERT, EV_KV, EV_VECTOR, EV_SWAP, QUERY):
                        pending_operator_event = True
                    elif token == SEP and pending_operator_event:
                        adaptive_candidate_mask[row, candidate_step] = True
                        adaptive_steps.add(candidate_step)
                        pending_operator_event = False
                if (
                    adaptive_teacher_targets is not None
                    and episode.track not in adaptive_v3_holdout_tracks
                ):
                    adaptive_teacher_targets[row, adaptive_candidate_mask[row], 0] = 0
                    adaptive_teacher_targets[row, adaptive_candidate_mask[row], 11] = int(
                        feature_off
                    )
                    adaptive_teacher_steps.update(
                        adaptive_candidate_mask[row].nonzero(as_tuple=False).flatten().tolist()
                    )
                if feature_off and adaptive_execution_mask is not None:
                    adaptive_execution_mask[row] = True
            row_operation_steps = set()
            for (
                step,
                operation,
                destination0,
                destination1,
                operand0,
                operand1,
                output_values,
                expected_operands,
            ) in operation_records:
                if episode.track in adaptive_v3_batch_stats:
                    adaptive_v3_batch_stats[episode.track][operation] += 1
                if operation_metadata is None:
                    raise RuntimeError("operation metadata storage was not initialized")
                if not 0 <= step < min(time, len(episode.inputs)):
                    continue
                operation_metadata[row, step] = torch.tensor(
                    (
                        _WORKSPACE_OPERATION_CODES[operation],
                        destination0,
                        destination1,
                        operand0[0],
                        operand0[1],
                        operand1[0],
                        operand1[1],
                    ),
                    dtype=torch.long,
                )
                if feature_off or operation == "reduce":
                    operation_metadata[row, step, 0] = 0
                elif include_operations and adaptive_execution_mask is not None:
                    adaptive_execution_mask[row, step] = False
                if (
                    adaptive_teacher_targets is not None
                    and episode.track in ("mixed_skill", "variable_binding", "permutation", "retrieval_aggregation")
                    and episode.track not in adaptive_v3_holdout_tracks
                ):
                    op_code = _WORKSPACE_OPERATION_CODES[operation]
                    adaptive_teacher_targets[row, step] = torch.tensor(
                        (
                            op_code,
                            destination0,
                            destination1,
                            operand0[0],
                            operand0[1],
                            operand1[0],
                            operand1[1],
                            expected_operands[0],
                            expected_operands[1],
                            output_values[0],
                            output_values[1] if len(output_values) > 1 else -100,
                            int(feature_off),
                        ),
                        dtype=torch.long,
                    )
                if operation_diagnostic_targets is not None:
                    operation_diagnostic_targets[row, step] = torch.tensor(
                        (
                            expected_operands[0],
                            expected_operands[1],
                            output_values[0],
                            output_values[1] if len(output_values) > 1 else -100,
                        ),
                        dtype=torch.long,
                    )
                row_operation_steps.add(step)
                operation_steps.add(step)
                destinations = (destination0, destination1)
                for column, (destination, target_value) in enumerate(
                    zip(destinations, output_values)
                ):
                    if destination < 0:
                        continue
                    key_ids[row, step, column] = destination
                    targets[row, step, column] = target_value
                    masks[row, step, column] = True
                read_steps.add(step)
            active_event = -1
            payload = -100
            source_payload = -1
            rule_trial = 0
            for _r in operation_records:
                _a = len(episode.inputs)
                if _r[1] == "reduce" and _a < time:
                    key_ids[row, _a, 0] = _REDUCE_RESULT_KEY
                    targets[row, _a, 0] = int(_r[6][0])
                    masks[row, _a, 0] = True
                    read_steps.add(_a)
            rule_feedback_targets = (
                self._rule_feedback_targets(episode, tokens)
                if rule_feedback_update and episode.track == "rule_switching"
                else ()
            )
            if attentional_item_memory and episode.track == "attentional_blink":
                for frame_step, frame_id, color_token, shape_token, value_token in (
                    self._attentional_item_frame_records(tokens)
                ):
                    if frame_step >= min(time, len(episode.inputs)) or frame_id >= ENTITY_COUNT:
                        continue
                    attentional_color_targets[row, frame_step] = color_token - SOURCE_BASE
                    attentional_shape_targets[row, frame_step] = shape_token - SOURCE_BASE
                    # The generic vocabulary head is categorical. Train it on
                    # the frame's scalar value only; color and shape use
                    # separate typed heads to avoid contradictory labels for
                    # identical semantic-address reads.
                    per_step.setdefault(frame_step, []).append((frame_id, value_token))
            for position, token_value in enumerate(tokens[: len(episode.inputs)]):
                token = int(token_value)
                if token in (EV_INIT, EV_MOVE, EV_UPDATE, EV_ASSERT, EV_KV) or (
                    masked_prime_trace
                    and episode.track == "masked_priming"
                    and token == EV_NOISE
                ):
                    active_event, payload, source_payload = token, -100, -1
                    if episode.track == "rule_switching" and token == EV_ASSERT:
                        if rule_feedback_update:
                            if rule_trial < len(rule_feedback_targets):
                                source_payload = SOURCE_BASE + int(
                                    rule_feedback_targets[rule_trial]
                                )
                        rule_trial += 1
                elif (
                    include_operations
                    and episode.track in ("mixed_skill", "variable_binding")
                    and token == EV_VECTOR
                ):
                    active_event, payload, source_payload = token, -100, -1
                elif episode.track == "rule_switching" and token == EV_VECTOR:
                    active_event, payload = token, -100
                elif (
                    episode.track == "rule_switching"
                    and active_event == EV_UPDATE
                    and SOURCE_BASE <= token < SOURCE_BASE + SOURCE_COUNT
                ):
                    source_payload = token
                elif LOCATION_BASE <= token < LOCATION_BASE + LOCATION_COUNT:
                    payload = token
                    if episode.track == "permutation" and event_writes[position]:
                        per_step.setdefault(position, []).append((int(event_keys[position]), token))
                elif SCALAR_BASE <= token < SCALAR_BASE + SCALAR_COUNT:
                    payload = token
                    if (
                        episode.track == "rule_switching"
                        and active_event == EV_VECTOR
                        and event_writes[position]
                    ):
                        key = int(event_keys[position])
                        per_step.setdefault(position, []).append((key, token))
                elif token in (BOOL_FALSE, BOOL_TRUE):
                    payload = token
                if (
                    token == SEP
                    and event_writes[position]
                    and payload >= 0
                    and position not in row_operation_steps
                    and not (
                        attentional_item_memory and episode.track == "attentional_blink"
                    )
                    and not (
                        episode.track == "rule_switching"
                        and active_event in (EV_UPDATE, EV_ASSERT)
                    )
                ):
                    key = int(event_keys[position])
                    if key >= 0:
                        per_step.setdefault(position, []).append((key, payload))
                if (
                    episode.track == "rule_switching"
                    and token == SEP
                    and event_writes[position]
                    and source_payload >= 0
                ):
                    per_step.setdefault(position, []).append((63, source_payload))
                if token == SEP:
                    active_event, payload, source_payload = -1, -100, -1

            # Updates are supervised only where a token stream explicitly
            # carries the stored value. Copy/compute events receive pressure
            # from a later query instead of a fabricated scalar target.
            for step, values in per_step.items():
                packed = values[:operand_capacity]
                key_ids[row, step, : len(packed)] = torch.tensor([v[0] for v in packed])
                targets[row, step, : len(packed)] = torch.tensor([v[1] for v in packed])
                masks[row, step, : len(packed)] = True
                read_steps.add(step)

            output_key_field = (
                "workspace_operation_output_keys"
                if include_operations and episode.track == "mixed_skill"
                else "workspace_output_keys"
            )
            output_keys = tuple(episode.metadata.get(output_key_field, ()))
            if output_keys and len(output_keys) != len(episode.answer):
                raise ValueError(
                    f"{episode.track} workspace_output_keys must align with all answer tokens"
                )
            for offset, key_value in enumerate(output_keys):
                key = int(key_value)
                step = len(episode.inputs) + offset
                if key < -1 or key >= ENTITY_COUNT:
                    raise ValueError(f"{episode.track} workspace output key must be -1 or [0, 64)")
                if key < 0 or step >= time or int(episode.answer[offset]) == EOS:
                    continue
                key_ids[row, step, 0] = key
                targets[row, step, 0] = int(episode.answer[offset])
                masks[row, step, 0] = True
                read_steps.add(step)
            query_reads = list(episode.metadata.get("workspace_query_reads", ()))
            if episode.track == "rule_switching":
                if rule_feedback_update and rule_feedback_targets:
                    feedback_rule = SOURCE_BASE + int(rule_feedback_targets[-1])
                    query_reads = [
                        (int(step), int(key), feedback_rule if int(key) == 63 else int(value))
                        for step, key, value in query_reads
                    ]
                else:
                    query_reads = [query for query in query_reads if int(query[1]) != 63]
            if masked_prime_trace and episode.track == "masked_priming":
                query_step = int(episode.inputs.index(QUERY))
                query_reads.append(
                    (query_step, 0, scalar_token(int(episode.metadata["prime_value"])))
                )
            for query in query_reads:
                step, key_value, target_value = (int(value) for value in query)
                key = key_value
                if key < 0 or key >= ENTITY_COUNT:
                    raise ValueError(f"{episode.track} workspace query key is outside [0, 64)")
                if target_value < 0 or target_value >= VOCAB_SIZE:
                    raise ValueError(f"{episode.track} workspace query target is outside the vocabulary")
                if not 0 <= step < min(time, len(episode.inputs)):
                    continue
                free = torch.nonzero(~masks[row, step], as_tuple=False).flatten()
                if free.numel() == 0:
                    continue
                column = int(free[0])
                key_ids[row, step, column] = key
                targets[row, step, column] = target_value
                masks[row, step, column] = True
                read_steps.add(step)
            output_programs = tuple(episode.metadata.get("workspace_output_programs", ()))
            if episode.track == "rule_switching" and not rule_feedback_update:
                # Without the feedback-tracker feature, do not train the
                # workspace against the generator's hidden rule labels or
                # provide an exact selector program.
                output_programs = ()
            if no_report_confidence_readout and episode.track == "no_report_confidence":
                evidence_keys = tuple(range(1, len(episode.metadata["observations"]) + 1))
                evidence_targets = tuple(
                    scalar_token(int(value)) for value in episode.metadata["observations"]
                )
                output_programs += ((0, "vote", evidence_keys, evidence_targets),)
                if bool(episode.metadata["explicit_report"]):
                    output_programs += ((1, "vote_confidence", evidence_keys, evidence_targets),)
            for program in output_programs:
                offset, operation, keys, expected = program
                offset = int(offset)
                if not 0 <= offset < len(episode.answer):
                    raise ValueError(f"{episode.track} workspace output offset is out of range")
                step = len(episode.inputs) + offset
                answer_keys = tuple(int(key) for key in keys)
                if step >= time:
                    raise RuntimeError(f"{episode.track} workspace output exceeds its batch sequence")
                if not answer_keys:
                    continue
                expected_values = tuple(int(value) for value in expected)
                if (
                    rule_feedback_update
                    and episode.track == "rule_switching"
                    and operation == "select"
                    and rule_feedback_targets
                ):
                    # The selector's rule operand must follow the feedback-only
                    # tracker too; otherwise the final query would reintroduce
                    # the hidden trial_rules label as auxiliary supervision.
                    expected_values = (
                        SOURCE_BASE + int(rule_feedback_targets[-1]),
                        *expected_values[1:],
                    )
                if (
                    len(answer_keys) != len(expected_values)
                    or len(answer_keys) > operand_capacity
                ):
                    raise ValueError(f"{episode.track} has invalid workspace output program operands")
                if any(key < 0 or key >= ENTITY_COUNT for key in answer_keys):
                    raise ValueError(f"{episode.track} workspace output key is outside [0, 64)")
                if any(value < 0 or value >= VOCAB_SIZE for value in expected_values):
                    raise ValueError(f"{episode.track} workspace output target is outside the vocabulary")
                if operation in {"equal", "not_equal"} and len(answer_keys) != 2:
                    raise ValueError("workspace comparison output requires exactly two values")
                if operation == "select" and len(answer_keys) != 4:
                    raise ValueError("workspace selector output requires a rule and three values")
                key_ids[row, step, : len(answer_keys)] = torch.tensor(answer_keys)
                targets[row, step, : len(expected_values)] = torch.tensor(expected_values)
                masks[row, step, : len(answer_keys)] = True
                output_program_specs.append((row, step, str(operation), len(answer_keys)))
                if operation == "argmatch":
                    argmatch_locations[(row, step)] = int(tokens[len(episode.inputs) - 2])
                read_steps.add(step)

        result = {
            "key_value_entity_ids": key_ids,
            "key_value_targets": targets,
            "key_value_mask": masks,
            "key_value_steps": tuple(sorted(read_steps)),
            "key_value_output_programs": tuple(output_program_specs),
            "key_value_argmatch_locations": argmatch_locations,
            "attentional_frame_color_targets": attentional_color_targets,
            "attentional_frame_shape_targets": attentional_shape_targets,
        }
        if include_operations:
            if operation_metadata is None:
                raise RuntimeError("workspace operation metadata was not initialized")
            result["workspace_operation_metadata"] = operation_metadata
            result["workspace_operation_steps"] = tuple(sorted(operation_steps))
        if include_adaptive_v3:
            if adaptive_candidate_mask is None or adaptive_execution_mask is None:
                raise RuntimeError("adaptive V3 metadata was not initialized")
            result["adaptive_operator_candidate_mask"] = adaptive_candidate_mask
            result["adaptive_operator_execution_mask"] = adaptive_execution_mask
            result["adaptive_operator_steps"] = tuple(sorted(adaptive_steps))
            result["adaptive_operator_teacher_steps"] = tuple(
                sorted(adaptive_teacher_steps)
            )
            self.last_adaptive_v3_batch_stats = {
                track: dict(stats) for track, stats in adaptive_v3_batch_stats.items()
            }
            # These CPU-side metadata counts expose V3's event workload without
            # adding reductions or synchronization to the recurrent CUDA path.
            self.last_adaptive_v3_batch_stats["__totals__"] = {
                "candidate_tokens": int(adaptive_candidate_mask.sum()),
                "executed_tokens": int(
                    (adaptive_candidate_mask & adaptive_execution_mask).sum()
                ),
                "candidate_steps": len(adaptive_steps),
            }
            if adaptive_teacher_targets is not None:
                result["adaptive_operator_teacher_targets"] = adaptive_teacher_targets
        if include_operation_diagnostics:
            if not include_operations or operation_diagnostic_targets is None:
                raise ValueError(
                    "operation diagnostic targets require operation metadata"
                )
            result["workspace_operation_diagnostic_targets"] = operation_diagnostic_targets
        return result

    @staticmethod
    def apply_workspace_key_value_readout_logits(
        logits,
        predictions,
        metadata,
        answer_mask,
        *,
        exact_value_comparison: bool = False,
        exact_selection: bool = False,
        exact_no_report_readout: bool = False,
    ):
        """Route answer logits through addressed workspace values when keyed."""
        if not predictions:
            return logits
        result = logits.clone()
        masks = metadata["key_value_mask"].to(device=logits.device)
        answer_mask = answer_mask.to(device=logits.device).bool()
        for step, value_logits in predictions:
            if not 0 <= int(step) < logits.shape[1]:
                continue
            active = masks[:, int(step), 0] & answer_mask[:, int(step)]
            routed = value_logits[:, 0].to(device=logits.device, dtype=logits.dtype)
            result[:, int(step)] = torch.where(active.unsqueeze(-1), routed, result[:, int(step)])
        predictions_by_step = {int(step): value for step, value in predictions}
        for row, step, operation, operand_count in metadata.get(
            "key_value_output_programs", ()
        ):
            value_logits = predictions_by_step.get(int(step))
            if value_logits is None:
                continue
            # Compose a probability distribution rather than emitting a
            # +/-1e4 one-hot logit vector. The latter makes a wrong hard
            # operation contribute ~20,000 to task CE and severs the task-loss
            # gradient at argmax. Marginalizing the operation over the value
            # head's categorical predictions gives a calibrated output
            # distribution and lets the task loss train the addressed values.
            operands = value_logits[int(row), : int(operand_count)].float()
            hard_program_probs = None
            if operation in {"equal", "not_equal"}:
                if int(operand_count) != 2:
                    raise ValueError("workspace comparison output requires exactly two values")
                # Values on these tasks are location tokens. Restricting the
                # comparator to that typed range prevents probability mass on
                # unrelated vocabulary items from acting like a false value.
                value_logits = operands[:, LOCATION_BASE : LOCATION_BASE + LOCATION_COUNT]
                operand_probs = value_logits.softmax(dim=-1)
                p_equal = (operand_probs[0] * operand_probs[1]).sum()
                p_true = p_equal if operation == "equal" else 1.0 - p_equal
                bool_probs = torch.stack((1.0 - p_true, p_true))
                if exact_value_comparison:
                    hard_values = value_logits.argmax(dim=-1)
                    hard_equal = hard_values[0].eq(hard_values[1])
                    hard_true = hard_equal if operation == "equal" else ~hard_equal
                    hard_probs = torch.stack((~hard_true, hard_true)).to(bool_probs.dtype)
                bool_tokens = torch.tensor(
                    (BOOL_FALSE, BOOL_TRUE), device=logits.device, dtype=torch.long
                )
                operation_probs = torch.zeros(
                    result.shape[-1], device=logits.device, dtype=torch.float32
                ).scatter(0, bool_tokens, bool_probs)
                if exact_value_comparison:
                    hard_program_probs = torch.zeros_like(operation_probs).scatter(
                        0, bool_tokens, hard_probs
                    )
            elif operation == "select":
                if int(operand_count) != 4:
                    raise ValueError("workspace selector output requires four values")
                selector_probs = operands[0].softmax(dim=-1)
                # Match the existing selector rule exactly: clamp(value -
                # SOURCE_BASE, 0, 2), which maps all vocabulary values into
                # one of the three value operands.
                selector_weights = torch.stack(
                    (
                        selector_probs[: SOURCE_BASE + 1].sum(),
                        selector_probs[SOURCE_BASE + 1],
                        selector_probs[SOURCE_BASE + 2 :].sum(),
                    )
                )
                value_probs = operands[1:].softmax(dim=-1)
                operation_probs = selector_weights @ value_probs
                if exact_selection:
                    hard_selector = operands[0, SOURCE_BASE : SOURCE_BASE + 3].argmax()
                    hard_value_token = value_probs[hard_selector].argmax()
                    hard_program_probs = F.one_hot(
                        hard_value_token, num_classes=result.shape[-1]
                    ).to(operation_probs.dtype)
            elif operation in {"vote", "vote_confidence"}:
                if int(operand_count) < 1:
                    raise ValueError("workspace vote readout requires at least one evidence value")
                binary_logits = operands[
                    :, SCALAR_BASE : SCALAR_BASE + 2
                ]
                binary_probs = binary_logits.softmax(dim=-1)[:, 1]
                count_distribution = binary_probs.new_ones((1,))
                for positive_probability in binary_probs:
                    zero = positive_probability.new_zeros((1,))
                    count_distribution = torch.cat(
                        (count_distribution * (1.0 - positive_probability), zero)
                    ) + torch.cat((zero, count_distribution * positive_probability))
                evidence_count = int(operand_count)
                positive_counts = torch.arange(
                    evidence_count + 1, device=logits.device
                )
                hard_values = binary_logits.argmax(dim=-1)
                hard_positive_count = hard_values.sum()
                hard_margin = (2 * hard_positive_count - evidence_count).abs()
                if operation == "vote":
                    threshold = (evidence_count + 1) // 2
                    positive_probability = count_distribution[threshold:].sum()
                    operation_probs = torch.zeros(
                        result.shape[-1], device=logits.device, dtype=torch.float32
                    ).scatter(
                        0,
                        torch.tensor((BOOL_FALSE, BOOL_TRUE), device=logits.device),
                        torch.stack((1.0 - positive_probability, positive_probability)),
                    )
                    hard_token = torch.where(
                        hard_positive_count * 2 >= evidence_count,
                        torch.tensor(BOOL_TRUE, device=logits.device),
                        torch.tensor(BOOL_FALSE, device=logits.device),
                    )
                else:
                    confidence_ids = (
                        (2 * positive_counts - evidence_count).abs() * 4 // evidence_count
                    ).clamp(max=3)
                    confidence_probs = torch.stack(
                        [
                            count_distribution[confidence_ids.eq(confidence_bin)].sum()
                            for confidence_bin in range(4)
                        ]
                    )
                    operation_probs = torch.zeros(
                        result.shape[-1], device=logits.device, dtype=torch.float32
                    ).scatter(
                        0,
                        torch.arange(SCALAR_BASE, SCALAR_BASE + 4, device=logits.device),
                        confidence_probs,
                    )
                    hard_bin = (hard_margin * 4 // evidence_count).clamp(max=3)
                    hard_token = SCALAR_BASE + hard_bin
                if exact_no_report_readout:
                    hard_program_probs = F.one_hot(
                        hard_token,
                        num_classes=result.shape[-1],
                    ).to(operation_probs.dtype)
            elif operation == "argmatch":
                query_location = int(metadata["key_value_argmatch_locations"][(int(row), int(step))]) - LOCATION_BASE
                match = operands[:, LOCATION_BASE : LOCATION_BASE + LOCATION_COUNT].softmax(dim=-1)[:, query_location]
                entity_tokens = (
                    metadata["key_value_entity_ids"][int(row), int(step), : int(operand_count)].long() + ENTITY_BASE
                ).to(logits.device)
                operation_probs = torch.zeros(
                    result.shape[-1], device=logits.device, dtype=torch.float32
                ).scatter(0, entity_tokens, match / match.sum().clamp_min(1e-6))
            else:
                raise ValueError(f"unknown workspace key-value output operation {operation!r}")
            # Keep probabilities normalized and cap the penalty for an
            # impossible/very-low-probability class at a meaningful finite
            # value. These are log-probabilities, so cross-entropy is a proper
            # NLL and has scale independent of vocabulary size or logit units.
            operation_probs = operation_probs.clamp_min(1e-7)
            operation_log_probs = operation_probs.log()
            operation_log_probs = operation_log_probs - torch.logsumexp(
                operation_log_probs, dim=-1
            )
            if hard_program_probs is not None:
                # The forward pass uses an exact discrete comparison. Keep a
                # finite hard decision for NLL while using the soft categorical
                # comparison only as a straight-through training gradient.
                hard_program_probs = hard_program_probs.clamp_min(1e-7)
                hard_log_probs = hard_program_probs.log()
                hard_log_probs = hard_log_probs - torch.logsumexp(
                    hard_log_probs, dim=-1
                )
                operation_log_probs = (
                    hard_log_probs
                    + operation_log_probs
                    - operation_log_probs.detach()
                )
            is_answer = answer_mask[int(row), int(step)].to(device=logits.device).bool()
            result[int(row), int(step)] = torch.where(
                is_answer,
                operation_log_probs.to(dtype=result.dtype),
                result[int(row), int(step)],
            )
        return result

    @staticmethod
    def workspace_key_value_loss(predictions, metadata, *, device, dtype):
        """Sparse CE for event-boundary state and query-time value retrieval."""
        if not predictions:
            return torch.zeros((), device=device, dtype=dtype)
        steps = torch.tensor([step for step, _ in predictions], device=device, dtype=torch.long)
        logits = torch.stack([value for _, value in predictions], dim=1).float()
        targets = metadata["key_value_targets"].to(device=device).index_select(1, steps)
        masks = metadata["key_value_mask"].to(device=device).index_select(1, steps)
        valid = masks & targets.ge(0)
        if not bool(valid.any()):
            return logits.new_zeros(())
        return F.cross_entropy(logits[valid], targets[valid])

    @staticmethod
    def workspace_attentional_frame_loss(predictions, metadata, *, device, dtype):
        """Supervise separate cue features from their addressed RSVP frame slots."""
        if not predictions:
            return torch.zeros((), device=device, dtype=dtype)
        steps = torch.tensor([step for step, *_ in predictions], device=device, dtype=torch.long)
        color_logits = torch.stack([color for _, color, _, _ in predictions], dim=1).float()
        shape_logits = torch.stack([shape for _, _, shape, _ in predictions], dim=1).float()
        found = torch.stack([valid for _, _, _, valid in predictions], dim=1).bool()
        color_targets = metadata["attentional_frame_color_targets"].to(device=device).index_select(
            1, steps
        )
        shape_targets = metadata["attentional_frame_shape_targets"].to(device=device).index_select(
            1, steps
        )
        color_valid = color_targets.ge(0) & found
        shape_valid = shape_targets.ge(0) & found
        losses = []
        for logits, targets, valid in (
            (color_logits, color_targets, color_valid),
            (shape_logits, shape_targets, shape_valid),
        ):
            if bool(valid.any()):
                losses.append(F.cross_entropy(logits[valid], targets[valid]))
        if not losses:
            return color_logits.new_zeros(())
        return torch.stack(losses).mean()

    def workspace_nth_farthest_metadata(
        self,
        input_vectors: torch.Tensor,
        episodes: list[TaskEpisode] | None = None,
    ) -> dict[str, Any]:
        """Build input-only vector targets and causal query metadata.

        Targets are attached to completed EV_VECTOR separators. Candidate IDs,
        query coordinates, and zero-based rank come only from input tokens;
        answer labels are never inspected.
        """
        token_ids = self._input_token_ids(input_vectors)
        examples = self.last_batch_examples if episodes is None else episodes
        batch, time = input_vectors.shape[:2]
        if len(examples) != batch:
            raise RuntimeError("Nth-farthest metadata lost batch/episode alignment")
        tokens_by_row = token_ids.detach().cpu().tolist()
        event_ids = torch.full((batch, time, 16), -1, dtype=torch.long, device=input_vectors.device)
        event_targets = torch.full(
            (batch, time, 16, 6), -100, dtype=torch.long, device=input_vectors.device
        )
        event_mask = torch.zeros((batch, time, 16), dtype=torch.bool, device=input_vectors.device)
        query_ids = torch.full_like(event_ids, -1)
        query_mask = torch.zeros_like(event_mask)
        query_vectors = torch.full(
            (batch, time, 6), -1, dtype=torch.long, device=input_vectors.device
        )
        query_ranks = torch.full((batch, time), -1, dtype=torch.long, device=input_vectors.device)
        vector_steps: set[int] = set()
        query_steps: set[int] = set()
        for row, episode in enumerate(examples):
            if episode.track != "nth_farthest":
                continue
            tokens = [int(token) for token in tokens_by_row[row][: len(episode.inputs)]]
            candidates: list[int] = []
            index = 0
            while index < len(tokens):
                if tokens[index] != EV_VECTOR:
                    index += 1
                    continue
                if (
                    index + 1 >= len(tokens)
                    or not ENTITY_BASE <= tokens[index + 1] < ENTITY_BASE + ENTITY_COUNT
                ):
                    raise ValueError("malformed Nth-farthest vector event: missing entity ID")
                entity_id = tokens[index + 1] - ENTITY_BASE
                end = index + 2
                values = []
                while end < len(tokens) and tokens[end] != SEP:
                    if SCALAR_BASE <= tokens[end] < SCALAR_BASE + SCALAR_COUNT:
                        values.append(tokens[end] - SCALAR_BASE)
                    end += 1
                if end >= len(tokens) or not 1 <= len(values) <= 6 or len(candidates) >= 16:
                    raise ValueError("malformed or oversized Nth-farthest vector event")
                candidates.append(entity_id)
                event_ids[row, end, 0] = entity_id
                event_targets[row, end, 0, : len(values)] = torch.tensor(
                    values, dtype=torch.long, device=input_vectors.device
                )
                event_mask[row, end, 0] = True
                vector_steps.add(end)
                index = end + 1

            query_index = next((i for i, token in enumerate(tokens) if token == QUERY), None)
            if query_index is None:
                raise ValueError("Nth-farthest episode is missing its query")
            query_end = tokens.index(SEP, query_index + 1)
            query_tokens = tokens[query_index + 1 : query_end]
            if not query_tokens or not candidates:
                raise ValueError("Nth-farthest query has no rank or candidate vectors")
            rank_token = query_tokens[0]
            if not SCALAR_BASE <= rank_token < SCALAR_BASE + SCALAR_COUNT:
                raise ValueError("Nth-farthest query rank is not a scalar token")
            coords = [
                token - SCALAR_BASE
                for token in query_tokens[1:]
                if SCALAR_BASE <= token < SCALAR_BASE + SCALAR_COUNT
            ]
            if not 1 <= len(coords) <= 6:
                raise ValueError("Nth-farthest query must contain 1..6 coordinates")
            answer_step = len(episode.inputs)
            if not 0 <= answer_step < time:
                raise RuntimeError("Nth-farthest answer step exceeds collated sequence")
            query_ids[row, answer_step, : len(candidates)] = torch.tensor(
                candidates, dtype=torch.long, device=input_vectors.device
            )
            query_mask[row, answer_step, : len(candidates)] = True
            query_vectors[row, answer_step, : len(coords)] = torch.tensor(
                coords, dtype=torch.long, device=input_vectors.device
            )
            query_ranks[row, answer_step] = rank_token - SCALAR_BASE
            query_steps.add(answer_step)
        return {
            "nth_farthest_vector_entity_ids": event_ids,
            "nth_farthest_vector_targets": event_targets,
            "nth_farthest_vector_mask": event_mask,
            "nth_farthest_vector_steps": tuple(sorted(vector_steps)),
            "nth_farthest_query_candidate_ids": query_ids,
            "nth_farthest_query_candidate_mask": query_mask,
            "nth_farthest_query_vectors": query_vectors,
            "nth_farthest_query_ranks": query_ranks,
            "nth_farthest_query_steps": tuple(sorted(query_steps)),
        }

    def workspace_retrieval_metadata(
        self,
        input_vectors: torch.Tensor,
        episodes: list[TaskEpisode] | None = None,
    ) -> dict[str, Any]:
        """Build input-only value-supervision and keyed-query metadata."""
        token_ids = self._input_token_ids(input_vectors)
        examples = self.last_batch_examples if episodes is None else episodes
        batch, time = input_vectors.shape[:2]
        if len(examples) != batch:
            raise RuntimeError("retrieval metadata lost batch/episode alignment")
        tokens_by_row = token_ids.detach().cpu().tolist()
        value_entity_ids = torch.full(
            (batch, time), -1, dtype=torch.long, device=input_vectors.device
        )
        value_targets = torch.full(
            (batch, time), -100, dtype=torch.long, device=input_vectors.device
        )
        value_mask = torch.zeros((batch, time), dtype=torch.bool, device=input_vectors.device)
        query_entity_ids = torch.full(
            (batch, time, 8), -1, dtype=torch.long, device=input_vectors.device
        )
        query_mask = torch.zeros((batch, time, 8), dtype=torch.bool, device=input_vectors.device)
        query_targets = torch.full_like(query_entity_ids, -100)
        value_is_update = torch.zeros((batch, time), dtype=torch.bool, device=input_vectors.device)
        query_was_updated = torch.zeros_like(query_mask)
        query_operation = torch.zeros((batch, time), dtype=torch.long, device=input_vectors.device)
        value_steps: set[int] = set()
        query_steps: set[int] = set()
        operation_by_token = {
            None: 1,  # ordered recall
            EV_SUM_QUERY: 2,
            EV_COUNT_QUERY: 3,
            EV_PARITY_QUERY: 4,
        }
        for row, episode in enumerate(examples):
            if episode.track != "retrieval_aggregation":
                continue
            tokens = [int(token) for token in tokens_by_row[row][: len(episode.inputs)]]
            current_values: dict[int, int] = {}
            updated_entities: set[int] = set()
            index = 0
            while index < len(tokens):
                token = tokens[index]
                if token in (EV_KV, EV_UPDATE):
                    if index + 2 >= len(tokens):
                        raise ValueError("malformed retrieval write event")
                    entity_token_value, scalar_token_value = tokens[index + 1 : index + 3]
                    if not ENTITY_BASE <= entity_token_value < ENTITY_BASE + ENTITY_COUNT:
                        raise ValueError("retrieval write event has an invalid entity ID")
                    if not SCALAR_BASE <= scalar_token_value < SCALAR_BASE + SCALAR_COUNT:
                        raise ValueError("retrieval write event has an invalid scalar value")
                    end = index + 3
                    if end >= len(tokens) or tokens[end] != SEP:
                        raise ValueError("retrieval write event is missing its separator")
                    entity_id = entity_token_value - ENTITY_BASE
                    value = scalar_token_value - SCALAR_BASE
                    current_values[entity_id] = value
                    is_update = token == EV_UPDATE
                    if is_update:
                        updated_entities.add(entity_id)
                    value_entity_ids[row, end] = entity_id
                    value_targets[row, end] = value
                    value_mask[row, end] = True
                    value_is_update[row, end] = is_update
                    value_steps.add(end)
                    index = end + 1
                    continue
                if token == QUERY:
                    query_start = index + 1
                    query_end = tokens.index(SEP, query_start)
                    cursor = query_start
                    operation_token = None
                    if cursor < query_end and tokens[cursor] in (
                        EV_SUM_QUERY,
                        EV_COUNT_QUERY,
                        EV_PARITY_QUERY,
                    ):
                        operation_token = tokens[cursor]
                        cursor += 1
                    operation = operation_by_token[operation_token]
                    answer_step = len(episode.inputs)
                    if not 0 <= answer_step < time:
                        raise RuntimeError("retrieval answer step exceeds its collated sequence")
                    ids = []
                    while cursor < query_end:
                        entity_token_value = tokens[cursor]
                        if not ENTITY_BASE <= entity_token_value < ENTITY_BASE + ENTITY_COUNT:
                            raise ValueError("retrieval query contains a malformed entity ID")
                        entity_id = entity_token_value - ENTITY_BASE
                        if entity_id not in current_values:
                            raise ValueError("retrieval query references an entity without a value")
                        ids.append(entity_id)
                        cursor += 1
                    if not ids or len(ids) > 8:
                        raise ValueError("retrieval query must contain 1..8 entity IDs")
                    query_entity_ids[row, answer_step, : len(ids)] = torch.tensor(
                        ids, dtype=torch.long, device=input_vectors.device
                    )
                    query_mask[row, answer_step, : len(ids)] = True
                    query_targets[row, answer_step, : len(ids)] = torch.tensor(
                        [current_values[entity_id] for entity_id in ids],
                        dtype=torch.long,
                        device=input_vectors.device,
                    )
                    query_was_updated[row, answer_step, : len(ids)] = torch.tensor(
                        [entity_id in updated_entities for entity_id in ids],
                        dtype=torch.bool,
                        device=input_vectors.device,
                    )
                    query_operation[row, answer_step] = operation
                    query_steps.add(answer_step)
                    break
                index += 1
        return {
            "retrieval_value_entity_ids": value_entity_ids,
            "retrieval_value_targets": value_targets,
            "retrieval_value_mask": value_mask,
            "retrieval_value_is_update": value_is_update,
            "retrieval_value_steps": tuple(sorted(value_steps)),
            "retrieval_query_entity_ids": query_entity_ids,
            "retrieval_query_mask": query_mask,
            "retrieval_query_targets": query_targets,
            "retrieval_query_was_updated": query_was_updated,
            "retrieval_query_operation": query_operation,
            "retrieval_query_steps": tuple(sorted(query_steps)),
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
        ablate_workspace_adaptive=False,
        ablate_workspace_operation_executor=False,
        shuffle_workspace_readback=False,
        ablate_workspace_feature: str | None = None,
        combiner_skip_stages=None,
        field_log=None,
        verbose_n=0,
    ):
        if self._output_proj is None:
            raise RuntimeError("dataset output projection was not set by the training pipeline")
        was_training = model.training
        prior_workspace_bypass = getattr(model, "workspace_eval_bypass", None)
        prior_workspace_shuffle = getattr(model, "workspace_eval_shuffle_readback", None)
        prior_adaptive_bypass = getattr(model, "workspace_adaptive_eval_bypass", None)
        prior_adaptive_v3_bypass = getattr(model, "workspace_adaptive_v3_eval_bypass", None)
        adaptive_eval_bypass = bool(prior_adaptive_bypass)
        adaptive_module = getattr(model, "workspace_adaptive", None)
        adaptive_v3_enabled = bool(getattr(model, "workspace_adaptive_v3_enabled", False))
        workspace = getattr(model, "workspace", None)
        prior_operation_executor = getattr(model, "workspace_operation_executor_enabled", None)
        prior_module_operation_executor = getattr(workspace, "operation_executor", None)
        v3_answer_logits: dict[str, list[torch.Tensor]] = defaultdict(list)
        episode_results_by_track: dict[str, list[tuple[float, bool]]] = defaultdict(list)
        feature_ablation_attribute = None
        prior_feature_ablation_value = None
        module_feature_ablation_attribute = None
        prior_module_feature_ablation_value = None
        operation_diagnostics_enabled = False
        prior_assignment_observer = getattr(workspace, "_initium_address_assignment_observer", None)
        collect_entity_bindings = bool(
            workspace is not None and getattr(workspace, "entity_keyed_writes", False)
        )
        if ablate_workspace:
            if (
                not hasattr(model, "workspace_eval_bypass")
                or getattr(model, "workspace", None) is None
            ):
                raise ValueError("workspace ablation requires a model with active workspace")
            model.workspace_eval_bypass = True
        if shuffle_workspace_readback:
            if (
                not hasattr(model, "workspace_eval_shuffle_readback")
                or getattr(model, "workspace", None) is None
            ):
                raise ValueError(
                    "workspace readback shuffle requires a model with active workspace"
                )
            model.workspace_eval_shuffle_readback = True
        if ablate_workspace_adaptive:
            if (adaptive_module is None and not adaptive_v3_enabled) or prior_adaptive_bypass is None:
                raise ValueError("adaptive workspace ablation requires an adaptive workspace")
            if adaptive_module is not None:
                model.workspace_adaptive_eval_bypass = True
            if adaptive_v3_enabled:
                model.workspace_adaptive_v3_eval_bypass = True
        if ablate_workspace_operation_executor:
            if prior_operation_executor is None or workspace is None:
                raise ValueError("manual operation ablation requires an active workspace")
            model.workspace_operation_executor_enabled = False
            workspace.operation_executor = False
        if ablate_workspace_feature is not None:
            feature_attribute = dict(_FUNCTIONAL_FEATURE_ABLATIONS).get(
                str(ablate_workspace_feature)
            )
            if feature_attribute is None:
                raise ValueError(
                    f"unsupported workspace functional feature {ablate_workspace_feature!r}"
                )
            if not bool(getattr(model, feature_attribute, False)):
                raise ValueError(
                    f"workspace functional feature {ablate_workspace_feature!r} "
                    "must be enabled before it can be ablated"
                )
            feature_ablation_attribute = feature_attribute
            prior_feature_ablation_value = getattr(model, feature_attribute)
            setattr(model, feature_attribute, False)
            module_feature_ablation_attribute = _WORKSPACE_MODULE_FEATURE_ABLATIONS.get(
                str(ablate_workspace_feature)
            )
            if module_feature_ablation_attribute is not None and workspace is not None:
                prior_module_feature_ablation_value = getattr(
                    workspace, module_feature_ablation_attribute
                )
                setattr(workspace, module_feature_ablation_attribute, False)
        # Feature ablations mutate model-facing switches for the duration of
        # this evaluation. Operation diagnostics require the operation
        # executor to remain active because their targets are packed alongside
        # operation metadata. Recompute after applying the ablation so the
        # operation_executor-off pass skips those diagnostics instead of
        # requesting metadata for a disabled pathway.
        operation_diagnostics_enabled = bool(
            getattr(model, "workspace_operation_diagnostics_enabled", False)
            and getattr(model, "workspace_operation_executor_enabled", False)
        )
        totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        progression_totals: dict[tuple[str, str], list[int]] = defaultdict(
            lambda: [0, 0, 0, 0]
        )
        query_type_totals: dict[tuple[str, str], list[int]] = defaultdict(
            lambda: [0, 0, 0, 0]
        )
        gnw_process_totals: dict[tuple[str, str], list[int]] = defaultdict(
            lambda: [0, 0, 0, 0, 0, 0]
        )
        # hit, present, false alarm, absent, correct, total
        ignition_access_totals: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0, 0])
        no_report_confidence_scores: list[float] = []
        no_report_confidence_correct: list[int] = []
        masked_prime_condition_totals: dict[tuple[str, str], list[int]] = defaultdict(
            lambda: [0, 0, 0, 0, 0]
        )
        blink_t2_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        split_totals: dict[str, list[float]] = defaultdict(lambda: [0.0] * 7)
        metadata_totals: dict[str, list[float]] = defaultdict(lambda: [0.0] * 8)
        permutation_query_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        permutation_state_depth_totals: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        nth_vector_coordinate_correct = nth_vector_coordinate_count = 0
        nth_vector_exact_correct = nth_vector_exact_count = 0
        nth_readout_correct = nth_readout_count = 0
        nth_invalid_readout_count = 0
        nth_readout_by_stratum: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        retrieval_totals = {
            "write_value_correct": 0,
            "write_value_count": 0,
            "query_values_correct": 0,
            "query_values_count": 0,
            "query_slots_found": 0,
            "query_slots_total": 0,
            "updated_query_values_correct": 0,
            "updated_query_values_count": 0,
            "untouched_query_values_correct": 0,
            "untouched_query_values_count": 0,
            "readout_exact": 0,
            "readout_count": 0,
            "invalid_readouts": 0,
            "readout_by_query_type": {},
        }
        operation_diagnostic_totals: dict[str, list[int]] = defaultdict(lambda: [0] * 6)
        adaptive_v3_route_totals: dict[str, list[int]] = defaultdict(lambda: [0] * 9)
        variable_binding_stage_totals: dict[tuple[str, str], list[int]] = defaultdict(
            lambda: [0] * 6
        )
        variable_binding_query_read_correct = 0
        variable_binding_query_read_count = 0
        rule_feedback_tracker_correct = 0
        rule_feedback_tracker_count = 0
        value_program_totals: dict[str, list[int]] = defaultdict(lambda: [0] * 4)
        masked_prime_trace_reads: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        adaptive_route_totals: dict[str, list[int]] = defaultdict(
            lambda: [0] * int(getattr(adaptive_module, "num_skills", 0))
        )
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
                sequence_memory = getattr(model, "sequence_memory", None)
                kv_attention_heads = (
                    int(getattr(sequence_memory, "heads", 0))
                    if getattr(model, "memory_type", None) == "kv"
                    else 0
                )
                for group in self._evaluation_groups(
                    episodes,
                    batch_size,
                    kv_attention_heads=kv_attention_heads,
                ):
                    x, _targets, _mask = self._collate(
                        group,
                        token_ids_only=bool(getattr(model, "workspace_token_id_input", False)),
                    )
                    sequence_valid_mask = None
                    if adaptive_module is not None:
                        sequence_valid_mask = torch.zeros(
                            (len(group), x.shape[1]), dtype=torch.bool
                        )
                        for row, episode in enumerate(group):
                            sequence_valid_mask[row, : len(episode.sequence)] = True
                        sequence_valid_mask = sequence_valid_mask.to(device, non_blocking=True)
                    token_rows = (
                        self.workspace_input_token_rows(x)
                        if bool(getattr(model, "workspace_semantic_entity_addressing", False))
                        else None
                    )
                    workspace_forward_kwargs = {}
                    if sequence_valid_mask is not None:
                        workspace_forward_kwargs["sequence_valid_mask"] = sequence_valid_mask
                    binding_entity_metadata = None
                    key_value_metadata = None
                    if bool(getattr(model, "workspace_semantic_entity_addressing", False)):
                        transactional = bool(
                            getattr(model, "workspace_permutation_transactional", False)
                        )
                        binding_entity_metadata = self.workspace_entity_address_metadata(
                            x,
                            paired_swap_commits=transactional,
                            attentional_item_memory=bool(
                                getattr(model, "workspace_attentional_item_memory_enabled", False)
                            ),
                            attentional_cue_readout=bool(
                                getattr(model, "workspace_attentional_cue_readout_enabled", False)
                            ),
                            masked_prime_trace=bool(
                                getattr(model, "workspace_masked_prime_trace_enabled", False)
                            ),
                            episodes=group,
                            token_rows=token_rows,
                        )
                        if transactional:
                            binding_entity_metadata.update(
                                self.workspace_permutation_query_metadata(x, group)
                            )
                        nth_enabled = bool(
                            getattr(model, "workspace_nth_farthest_exact_readout", False)
                        )
                        if nth_enabled:
                            nth_metadata = self.workspace_nth_farthest_metadata(x, group)
                            binding_entity_metadata.update(
                                {
                                    key: value
                                    for key, value in nth_metadata.items()
                                    if key != "nth_farthest_vector_targets"
                                }
                            )
                        retrieval_enabled = bool(
                            getattr(model, "workspace_retrieval_exact_readout", False)
                        )
                        retrieval_metadata = None
                        if retrieval_enabled:
                            retrieval_metadata = self.workspace_retrieval_metadata(x, group)
                            binding_entity_metadata.update(
                                {
                                    key: value
                                    for key, value in retrieval_metadata.items()
                                    if key
                                    not in (
                                        "retrieval_value_targets",
                                        "retrieval_query_targets",
                                        "retrieval_value_is_update",
                                        "retrieval_query_was_updated",
                                    )
                                }
                            )
                        if bool(getattr(model, "workspace_key_value_readout_enabled", False)):
                            operation_executor = bool(
                                getattr(model, "workspace_operation_executor_enabled", False)
                            )
                            key_value_metadata = {
                                key: value.to(device, non_blocking=True)
                                if isinstance(value, torch.Tensor)
                                and key != "workspace_operation_diagnostic_targets"
                                else value
                                for key, value in self.workspace_key_value_metadata(
                                    x,
                                    group,
                                    include_operations=operation_executor,
                                    include_adaptive_v3=bool(
                                        getattr(model, "workspace_adaptive_v3_enabled", False)
                                    ),
                                    adaptive_v3_distillation=False,
                                    adaptive_v3_holdout_tracks=tuple(
                                        getattr(model, "workspace_adaptive_v3_holdout_tracks", ())
                                    ),
                                    include_operation_diagnostics=(
                                        operation_diagnostics_enabled
                                    ),
                                    rule_feedback_update=bool(
                                        getattr(
                                            model,
                                            "workspace_rule_feedback_update_enabled",
                                            False,
                                        )
                                    ),
                                    attentional_item_memory=bool(
                                        getattr(
                                            model,
                                            "workspace_attentional_item_memory_enabled",
                                            False,
                                        )
                                    ),
                                    no_report_confidence_readout=bool(
                                        getattr(
                                            model,
                                            "workspace_no_report_confidence_readout_enabled",
                                            False,
                                        )
                                    ),
                                    masked_prime_trace=bool(
                                        getattr(model, "workspace_masked_prime_trace_enabled", False)
                                    ),
                                    entity_metadata=binding_entity_metadata,
                                    token_rows=token_rows,
                                ).items()
                            }
                            binding_entity_metadata.update(
                                {
                                    "key_value_entity_ids": key_value_metadata[
                                        "key_value_entity_ids"
                                    ],
                                    "key_value_mask": key_value_metadata["key_value_mask"],
                                    "key_value_steps": key_value_metadata["key_value_steps"],
                                }
                            )
                            if operation_executor:
                                binding_entity_metadata.update(
                                    {
                                        "operation_metadata": key_value_metadata[
                                            "workspace_operation_metadata"
                                        ],
                                        "operation_steps": key_value_metadata[
                                            "workspace_operation_steps"
                                        ],
                                    }
                                )
                            if bool(getattr(model, "workspace_adaptive_v3_enabled", False)):
                                binding_entity_metadata.update(
                                    {
                                        "adaptive_operator_candidate_mask": key_value_metadata[
                                            "adaptive_operator_candidate_mask"
                                        ],
                                        "adaptive_operator_execution_mask": key_value_metadata[
                                            "adaptive_operator_execution_mask"
                                        ],
                                        "adaptive_operator_steps": key_value_metadata[
                                            "adaptive_operator_steps"
                                        ],
                                    }
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
                        # fmt: off
                        workspace._initium_address_assignment_observer = (
                            lambda step,
                            assignments,
                            assignment_trace=assignment_trace: assignment_trace.append(
                                (step, assignments.detach())
                            )
                        )
                        # fmt: on
                    try:
                        output, _ = model(
                            x,
                            hidden,
                            reset_experience=True,
                            pass_through_memory=not ablate_memory,
                            combiner_skip_stages=combiner_skip_stages,
                            **workspace_forward_kwargs,
                        )
                        if (
                            adaptive_module is not None
                            and not ablate_workspace_adaptive
                            and not adaptive_eval_bypass
                        ):
                            assignments = adaptive_module.last_route_assignments()
                            if assignments is not None:
                                assignments_cpu = assignments.cpu()
                                for row, episode in enumerate(group):
                                    token_count = min(
                                        len(episode.sequence), assignments_cpu.shape[1]
                                    )
                                    route_counts = torch.bincount(
                                        assignments_cpu[row, :token_count].long(),
                                        minlength=adaptive_module.num_skills,
                                    ).tolist()
                                    track_counts = adaptive_route_totals[episode.track]
                                    for route_index, count in enumerate(route_counts):
                                        track_counts[route_index] += int(count)
                        if adaptive_v3_enabled and not bool(
                            getattr(model, "workspace_adaptive_v3_eval_bypass", False)
                        ):
                            for _route_step, routes in workspace.adaptive_v3_route_steps():
                                route_ids = routes.detach().cpu().tolist()
                                for episode, route_id in zip(group, route_ids):
                                    adaptive_v3_route_totals[episode.track][int(route_id)] += 1
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
                        trace = torch.stack([entry[1] for entry in assignment_trace], dim=0).cpu()
                        trace_by_step = {
                            step: trace[index] for index, step in enumerate(trace_steps)
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
                                    entity = int(binding_entity_metadata["entity_ids"][row, step])
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
                                binding_write_route_splits += int(unique_route_slots.numel() > 1)
                                route_counts = Counter(valid_slots.tolist())
                                entity_slots[entity].append(route_counts.most_common(1)[0][0])
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
                                    binding_repeated_entity_consistency += counts.most_common(1)[0][
                                        1
                                    ] / len(slots)
                                    binding_repeated_entities += 1
                            unique_slots = len(set(dominant_slots.values()))
                            binding_unique_slot_fraction += unique_slots / len(dominant_slots)
                            slots = list(dominant_slots.values())
                            for left in range(len(slots)):
                                for right in range(left + 1, len(slots)):
                                    binding_entity_pairs += 1
                                    binding_colliding_entity_pairs += int(
                                        slots[left] == slots[right]
                                    )
                    logits = self._output_proj(output.transpose(0, 1).contiguous())
                    if key_value_metadata is not None:
                        key_value_predictions = workspace.pop_key_value_predictions()
                        if bool(
                            getattr(model, "workspace_masked_prime_trace_enabled", False)
                        ):
                            targets_by_step = key_value_metadata["key_value_targets"]
                            masks_by_step = key_value_metadata["key_value_mask"]
                            for read_step, read_logits in key_value_predictions:
                                for row, episode in enumerate(group):
                                    if (
                                        episode.track == "variable_binding"
                                        and int(read_step) == len(episode.inputs)
                                        and bool(key_value_metadata["key_value_mask"][row, read_step, 0])
                                    ):
                                        target_value = int(
                                            key_value_metadata["key_value_targets"][row, read_step, 0]
                                        )
                                        variable_binding_query_read_correct += int(
                                            int(read_logits[row, 0].argmax().item()) == target_value
                                        )
                                        variable_binding_query_read_count += 1
                                    if episode.track != "masked_priming":
                                        continue
                                    prime_query_step = int(episode.inputs.index(QUERY))
                                    if int(read_step) != prime_query_step:
                                        continue
                                    read_mask = masks_by_step[row, read_step]
                                    if not bool(read_mask.any()):
                                        continue
                                    target_values = targets_by_step[row, read_step][read_mask]
                                    predicted_values = (
                                        read_logits[row, read_mask].argmax(dim=-1)
                                    )
                                    trace_counts = masked_prime_trace_reads[
                                        str(episode.metadata["query_type"])
                                    ]
                                    trace_counts[0] += int(
                                        predicted_values.eq(target_values).sum().item()
                                    )
                                    trace_counts[1] += int(target_values.numel())
                        exact_comparison = bool(
                            getattr(model, "workspace_exact_value_comparison_enabled", False)
                        )
                        exact_selection = bool(
                            getattr(model, "workspace_exact_selection_enabled", False)
                        )
                        exact_no_report = bool(
                            getattr(
                                model,
                                "workspace_no_report_confidence_readout_enabled",
                                False,
                            )
                        )
                        if exact_comparison or exact_selection or exact_no_report:
                            prediction_steps = {
                                int(step): values for step, values in key_value_predictions
                            }
                            for row, step, operation, operand_count in key_value_metadata.get(
                                "key_value_output_programs", ()
                            ):
                                values = prediction_steps.get(int(step))
                                if values is None:
                                    continue
                                operand_count = int(operand_count)
                                targets = key_value_metadata["key_value_targets"][
                                    int(row), int(step), :operand_count
                                ]
                                predictions_for_row = values[int(row), :operand_count].argmax(
                                    dim=-1
                                )
                                if operation in {"equal", "not_equal"} and exact_comparison:
                                    if operand_count != 2 or not bool(
                                        targets.ge(LOCATION_BASE).logical_and(
                                            targets.lt(LOCATION_BASE + LOCATION_COUNT)
                                        ).all()
                                    ):
                                        continue
                                    predicted_values = values[
                                        int(row), :2,
                                        LOCATION_BASE : LOCATION_BASE + LOCATION_COUNT,
                                    ].argmax(dim=-1) + LOCATION_BASE
                                    predictions_for_row = predicted_values
                                    counts = value_program_totals[str(operation)]
                                    expected_equal = targets[0].eq(targets[1])
                                    expected_true = (
                                        expected_equal
                                        if operation == "equal"
                                        else ~expected_equal
                                    )
                                    predicted_true = (
                                        predicted_values[0].eq(predicted_values[1])
                                        if operation == "equal"
                                        else predicted_values[0].ne(predicted_values[1])
                                    )
                                    program_correct = predicted_true.eq(expected_true)
                                elif operation == "select" and exact_selection:
                                    if operand_count != 4:
                                        continue
                                    counts = value_program_totals["select"]
                                    expected_index = int(targets[0]) - SOURCE_BASE
                                    if not 0 <= expected_index < 3:
                                        continue
                                    predicted_index = int(
                                        values[
                                            int(row), 0, SOURCE_BASE : SOURCE_BASE + 3
                                        ].argmax()
                                    )
                                    predicted_output = int(
                                        values[int(row), predicted_index + 1].argmax()
                                    )
                                    program_correct = predicted_output == int(
                                        targets[expected_index + 1]
                                    )
                                elif operation in {"vote", "vote_confidence"} and exact_no_report:
                                    counts = value_program_totals[operation]
                                    predicted_values = values[
                                        int(row), :operand_count,
                                        SCALAR_BASE : SCALAR_BASE + 2,
                                    ].argmax(dim=-1)
                                    predicted_positive = int(predicted_values.sum().item())
                                    margin = abs(2 * predicted_positive - operand_count)
                                    output_offset = int(step) - len(group[int(row)].inputs)
                                    if operation == "vote":
                                        predicted_output = (
                                            BOOL_TRUE
                                            if predicted_positive * 2 >= operand_count
                                            else BOOL_FALSE
                                        )
                                    else:
                                        predicted_output = SCALAR_BASE + min(
                                            3, margin * 4 // operand_count
                                        )
                                    target_output = int(group[int(row)].answer[output_offset])
                                    predictions_for_row = predicted_values + SCALAR_BASE
                                    program_correct = predicted_output == target_output
                                else:
                                    continue
                                counts[0] += int((predictions_for_row == targets).sum())
                                counts[1] += operand_count
                                counts[2] += int(program_correct)
                                counts[3] += 1
                        logits = self.apply_workspace_key_value_readout_logits(
                            logits,
                            key_value_predictions,
                            key_value_metadata,
                            _mask,
                            exact_value_comparison=exact_comparison,
                            exact_selection=exact_selection,
                            exact_no_report_readout=bool(
                                getattr(
                                    model,
                                    "workspace_no_report_confidence_readout_enabled",
                                    False,
                                )
                            ),
                        )
                    if bool(getattr(model, "workspace_attentional_cue_readout_enabled", False)):
                        logits = self.apply_workspace_attentional_cue_readout_logits(
                            logits, workspace.pop_attentional_cue_predictions()
                        )
                    if adaptive_v3_enabled and bool(
                        getattr(model, "workspace_adaptive_v3_logit_diagnostics_enabled", False)
                    ):
                        for row, episode in enumerate(group):
                            answer_start = len(episode.inputs)
                            answer_end = min(
                                answer_start + len(episode.answer), logits.shape[1]
                            )
                            if answer_end > answer_start:
                                v3_answer_logits[episode.track].append(
                                    logits[row, answer_start:answer_end]
                                    .detach()
                                    .to(device="cpu", dtype=torch.float32)
                                )
                    if operation_diagnostics_enabled and key_value_metadata is not None:
                        operation_traces = workspace.pop_operation_diagnostics()
                        diagnostic_targets = key_value_metadata[
                            "workspace_operation_diagnostic_targets"
                        ]
                        for (
                            operation_step,
                            operation_codes,
                            source0_predictions,
                            source1_predictions,
                            destination0_committed,
                            destination1_committed,
                            destination0_values,
                            destination1_values,
                        ) in operation_traces:
                            step_codes = operation_codes.detach().cpu().tolist()
                            source0 = source0_predictions.detach().cpu().tolist()
                            source1 = source1_predictions.detach().cpu().tolist()
                            commit0 = destination0_committed.detach().cpu().tolist()
                            commit1 = destination1_committed.detach().cpu().tolist()
                            value0 = destination0_values.detach().cpu().tolist()
                            value1 = destination1_values.detach().cpu().tolist()
                            expected = diagnostic_targets[:, operation_step].tolist()
                            operand_kinds = (
                                key_value_metadata["workspace_operation_metadata"][:, operation_step][
                                    :, [3, 5]
                                ]
                                .detach()
                                .cpu()
                                .tolist()
                            )
                            for row, operation_code in enumerate(step_codes):
                                operation_name = _WORKSPACE_OPERATION_NAMES.get(
                                    int(operation_code)
                                )
                                if operation_name is None:
                                    continue
                                episode = group[row]
                                variable_binding_counts = None
                                if episode.track == "variable_binding":
                                    assignment_count = int(
                                        episode.metadata.get("assignment_count", 0)
                                    )
                                    complexity = "long" if assignment_count >= 12 else "short"
                                    variable_binding_counts = variable_binding_stage_totals[
                                        (f"assignments_{complexity}", operation_name)
                                    ]
                                counts = operation_diagnostic_totals[operation_name]
                                for prediction, target, operand_kind in zip(
                                    (source0[row], source1[row]),
                                    expected[row][:2],
                                    operand_kinds[row],
                                ):
                                    if int(target) >= 0 and int(operand_kind) == 1:
                                        correct_read = int(int(prediction) == int(target))
                                        counts[0] += correct_read
                                        counts[1] += 1
                                        if variable_binding_counts is not None:
                                            variable_binding_counts[0] += correct_read
                                            variable_binding_counts[1] += 1
                                for committed, prediction, target in zip(
                                    (commit0[row], commit1[row]),
                                    (value0[row], value1[row]),
                                    expected[row][2:],
                                ):
                                    if int(target) >= 0:
                                        counts[2] += int(bool(committed))
                                        counts[3] += 1
                                        if variable_binding_counts is not None:
                                            variable_binding_counts[2] += int(bool(committed))
                                            variable_binding_counts[3] += 1
                                        if bool(committed):
                                            correct_output = int(int(prediction) == int(target))
                                            counts[4] += correct_output
                                            counts[5] += 1
                                            if variable_binding_counts is not None:
                                                variable_binding_counts[4] += correct_output
                                                variable_binding_counts[5] += 1
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
                    nth_predictions_by_step = {}
                    if bool(getattr(model, "workspace_nth_farthest_exact_readout", False)):
                        query_predictions = workspace.pop_nth_farthest_query_predictions()
                        vector_predictions = dict(workspace.pop_nth_farthest_vector_predictions())
                        nth_predictions_by_step = {
                            step: (selected, active)
                            for step, _query_logits, selected, active in query_predictions
                        }
                        if query_predictions:
                            logits = logits.clone()
                            for query_step, query_logits, selected, active in query_predictions:
                                if not 0 <= int(query_step) < logits.shape[1]:
                                    continue
                                entity_logits = logits.new_full(
                                    (logits.shape[0], logits.shape[-1]), -1e4
                                )
                                entity_logits[:, ENTITY_BASE : ENTITY_BASE + ENTITY_COUNT] = (
                                    query_logits.to(device=logits.device, dtype=logits.dtype)
                                )
                                invalid_readout = active.to(device=logits.device) & selected.to(
                                    device=logits.device
                                ).lt(0)
                                entity_logits = torch.where(
                                    invalid_readout.unsqueeze(-1),
                                    torch.full_like(entity_logits, -1e4),
                                    entity_logits,
                                )
                                logits[:, int(query_step)] = torch.where(
                                    active.to(device=logits.device).unsqueeze(-1),
                                    entity_logits,
                                    logits[:, int(query_step)],
                                )
                        for vector_step, vector_logits in vector_predictions.items():
                            target = nth_metadata["nth_farthest_vector_targets"][:, vector_step].to(
                                device=vector_logits.device
                            )
                            valid_coordinates = target.ne(-100)
                            prediction = vector_logits.argmax(dim=-1)
                            equal = prediction.eq(target) & valid_coordinates
                            nth_vector_coordinate_correct += int(equal.sum().item())
                            nth_vector_coordinate_count += int(valid_coordinates.sum().item())
                            valid_vectors = valid_coordinates.any(dim=-1)
                            vector_exact = (prediction.eq(target) | ~valid_coordinates).all(dim=-1)
                            nth_vector_exact_correct += int(
                                (vector_exact & valid_vectors).sum().item()
                            )
                            nth_vector_exact_count += int(valid_vectors.sum().item())
                        for row, episode in enumerate(group):
                            if episode.track != "nth_farthest":
                                continue
                            answer_step = len(episode.inputs)
                            selected_active = nth_predictions_by_step.get(answer_step)
                            if selected_active is None or not bool(selected_active[1][row].item()):
                                continue
                            selected_entity = int(selected_active[0][row].item())
                            expected_entity = int(episode.answer[0]) - ENTITY_BASE
                            nth_invalid_readout_count += int(selected_entity < 0)
                            correct_readout = int(selected_entity == expected_entity)
                            nth_readout_correct += correct_readout
                            nth_readout_count += 1
                            zero_based_rank = int(
                                nth_metadata["nth_farthest_query_ranks"][row, answer_step].item()
                            )
                            stratum = (
                                f"{episode.metadata.get('query_type', 'unknown')}|"
                                f"n{episode.metadata.get('candidate_count', 0)}|"
                                f"d{int(nth_metadata['nth_farthest_query_vectors'][row, answer_step].ne(-1).sum().item())}|"
                                f"r{zero_based_rank}"
                            )
                            nth_readout_by_stratum[stratum][0] += correct_readout
                            nth_readout_by_stratum[stratum][1] += 1
                    if bool(getattr(model, "workspace_retrieval_exact_readout", False)):
                        assert binding_entity_metadata is not None
                        retrieval_predictions = workspace.pop_retrieval_query_predictions()
                        value_predictions = workspace.pop_retrieval_value_predictions()
                        if retrieval_metadata is None:
                            raise RuntimeError("retrieval supervision metadata was not constructed")
                        for value_step, value_logits, found in value_predictions:
                            targets = retrieval_metadata["retrieval_value_targets"][
                                :, value_step
                            ].to(device=value_logits.device)
                            mask = retrieval_metadata["retrieval_value_mask"][:, value_step].to(
                                device=value_logits.device
                            )
                            correct = value_logits.argmax(dim=-1).eq(targets) & mask & found
                            retrieval_totals["write_value_correct"] += int(correct.sum().item())
                            retrieval_totals["write_value_count"] += int(mask.sum().item())
                            is_update = retrieval_metadata["retrieval_value_is_update"][
                                :, value_step
                            ].to(device=value_logits.device)
                            for label, selection in (
                                ("update", is_update),
                                ("initial", ~is_update),
                            ):
                                count_key = f"{label}_value_count"
                                correct_key = f"{label}_value_correct"
                                retrieval_totals.setdefault(count_key, 0)
                                retrieval_totals.setdefault(correct_key, 0)
                                selected = mask & selection
                                retrieval_totals[correct_key] += int(
                                    (value_logits.argmax(dim=-1).eq(targets) & selected & found)
                                    .sum()
                                    .item()
                                )
                                retrieval_totals[count_key] += int(selected.sum().item())
                        logits, query_totals = self.apply_workspace_retrieval_query_logits(
                            logits, retrieval_predictions, retrieval_metadata, group
                        )
                        for metric, value in query_totals.items():
                            if metric == "readout_by_query_type":
                                for query_type, counts in value.items():
                                    total_counts = retrieval_totals[metric].setdefault(
                                        query_type, {"correct": 0, "episodes": 0}
                                    )
                                    total_counts["correct"] += counts["correct"]
                                    total_counts["episodes"] += counts["episodes"]
                            else:
                                retrieval_totals[metric] += value
                    predictions = logits.argmax(dim=-1)
                    for local_idx, episode in enumerate(group):
                        start = len(episode.inputs)
                        pred = predictions[local_idx, start : start + len(episode.answer)].tolist()
                        expected = list(episode.answer)
                        n_correct = sum(int(a == b) for a, b in zip(pred, expected))
                        n_answers = len(expected)
                        exact = n_correct == n_answers
                        episode_results_by_track[episode.track].append(
                            (n_correct / max(n_answers, 1), bool(exact))
                        )
                        progression_group = str(
                            episode.metadata.get("progression_group", "all")
                        )
                        if (
                            episode.track == "rule_switching"
                            and bool(
                                getattr(model, "workspace_rule_feedback_update_enabled", False)
                            )
                        ):
                            tokens = token_rows[local_idx]
                            inferred_rules = self._rule_feedback_targets(episode, tokens)
                            true_rules = tuple(episode.metadata.get("trial_rules", ()))
                            matched = min(len(inferred_rules), len(true_rules))
                            rule_feedback_tracker_correct += sum(
                                int(inferred_rules[index] == int(true_rules[index]))
                                for index in range(matched)
                            )
                            rule_feedback_tracker_count += matched
                        progression_counts = progression_totals[
                            (episode.track, progression_group)
                        ]
                        progression_counts[0] += n_correct
                        progression_counts[1] += n_answers
                        progression_counts[2] += 1
                        progression_counts[3] += int(exact)
                        query_type = str(episode.metadata.get("query_type", "unknown"))
                        query_counts = query_type_totals[(episode.track, query_type)]
                        query_counts[0] += n_correct
                        query_counts[1] += n_answers
                        query_counts[2] += 1
                        query_counts[3] += int(exact)
                        if episode.track == "attentional_blink" and len(pred) >= 2:
                            blink_counts = blink_t2_totals[query_type]
                            blink_counts[0] += int(pred[0] == expected[0])
                            blink_counts[1] += int(pred[1] == expected[1])
                            blink_counts[2] += 1
                            blink_counts[3] += int(exact)
                        if episode.track == "no_report_confidence":
                            process = gnw_process_totals[
                                (episode.track, str(episode.metadata["query_type"]))
                            ]
                            action_correct = int(
                                pred[0]
                                == (
                                    BOOL_TRUE
                                    if bool(episode.metadata["majority_action"])
                                    else BOOL_FALSE
                                )
                            )
                            latent_correct = int(
                                (pred[0] == BOOL_TRUE)
                                == bool(episode.metadata["latent_truth"])
                            )
                            process[0] += action_correct
                            process[1] += 1
                            process[2] += latent_correct
                            process[3] += 1
                            reported = bool(episode.metadata["explicit_report"])
                            if reported:
                                confidence_logits = logits[
                                    local_idx, start + 1,
                                    SCALAR_BASE : SCALAR_BASE + 4,
                                ].float().softmax(dim=-1)
                                confidence_score = float(
                                    (confidence_logits * confidence_logits.new_tensor(
                                        (0.125, 0.375, 0.625, 0.875)
                                    )).sum().item()
                                )
                                no_report_confidence_scores.append(confidence_score)
                                no_report_confidence_correct.append(latent_correct)
                                process[4] += int(
                                    len(pred) > 1
                                    and pred[1]
                                    == scalar_token(int(episode.metadata["confidence_bin"]))
                                )
                                process[5] += 1
                        elif episode.track == "ignition":
                            strength = int(episode.metadata["stimulus_strength"])
                            response_present = int(pred[0] == BOOL_TRUE)
                            target_present = bool(episode.metadata["target_present"])
                            curve_counts = ignition_access_totals[strength]
                            curve_counts[0] += int(response_present and target_present)
                            curve_counts[1] += int(target_present)
                            curve_counts[2] += int(response_present and not target_present)
                            curve_counts[3] += int(not target_present)
                            curve_counts[4] += int(response_present == int(target_present))
                            curve_counts[5] += 1
                        elif episode.track == "masked_priming":
                            process = gnw_process_totals[
                                (episode.track, query_type)
                            ]
                            predicted_response = (
                                int(pred[0]) - SCALAR_BASE
                                if SCALAR_BASE <= int(pred[0]) < SCALAR_BASE + 2
                                else -1
                            )
                            prime = int(episode.metadata["prime_value"])
                            response = int(episode.metadata["target_value"])
                            process[0] += int(predicted_response == response)
                            process[1] += 1
                            process[2] += int(
                                prime != response and predicted_response == prime
                            )
                            process[3] += int(prime != response)
                            process[4] += int(
                                predicted_response == int(episode.metadata["latent_target"])
                            )
                            process[5] += 1
                            condition = (
                                "masked" if bool(episode.metadata["mask_condition"]) else "visible",
                                "congruent" if bool(episode.metadata["prime_congruent"]) else "incongruent",
                            )
                            condition_counts = masked_prime_condition_totals[condition]
                            condition_counts[0] += int(predicted_response == response)
                            condition_counts[1] += int(
                                prime != response and predicted_response == prime
                            )
                            condition_counts[2] += int(prime != response)
                            condition_counts[3] += int(
                                predicted_response == int(episode.metadata["latent_target"])
                            )
                            condition_counts[4] += 1
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
                        track_meta[7] += int(
                            bool(episode.metadata.get("equal_priority_conflict", False))
                        )
                        self._record_field(field_log, episode, n_correct, n_answers, exact)
                        if len(totals) == 1 and totals[episode.track][2] <= verbose_n:
                            print(
                                f"  [{episode.track}/{episode.metadata['query_type']}] "
                                f"pred={pred} target={expected} exact={exact}"
                            )
        finally:
            if collect_entity_bindings:
                workspace._initium_address_assignment_observer = prior_assignment_observer
            if ablate_workspace:
                model.workspace_eval_bypass = prior_workspace_bypass
            if shuffle_workspace_readback:
                model.workspace_eval_shuffle_readback = prior_workspace_shuffle
            if ablate_workspace_adaptive:
                model.workspace_adaptive_eval_bypass = prior_adaptive_bypass
                if adaptive_v3_enabled:
                    model.workspace_adaptive_v3_eval_bypass = prior_adaptive_v3_bypass
            if ablate_workspace_operation_executor:
                model.workspace_operation_executor_enabled = prior_operation_executor
                workspace.operation_executor = prior_module_operation_executor
            if feature_ablation_attribute is not None:
                setattr(
                    model,
                    feature_ablation_attribute,
                    prior_feature_ablation_value,
                )
            if module_feature_ablation_attribute is not None and workspace is not None:
                setattr(
                    workspace,
                    module_feature_ablation_attribute,
                    prior_module_feature_ablation_value,
                )
            model.train(was_training)

        accuracy = 100.0 * correct_total / max(answer_total, 1)
        self._last_eval_episode_results = dict(episode_results_by_track)
        self._last_eval_v3_answer_logits = {
            track: torch.cat(rows, dim=0) for track, rows in v3_answer_logits.items() if rows
        }
        perfect = 100.0 * exact_total / max(len(episodes), 1)
        total_meta = [
            sum(values[index] for values in metadata_totals.values()) for index in range(8)
        ]
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
            "adaptive_workspace_routes_by_track": {
                track: {
                    "token_count": int(sum(counts)),
                    "route_token_counts": [int(count) for count in counts],
                    "route_token_fractions": [
                        float(count / max(sum(counts), 1)) for count in counts
                    ],
                }
                for track, counts in sorted(adaptive_route_totals.items())
            },
            "adaptive_v3_operator_routes_by_track": {
                track: {
                    "candidate_count": int(sum(counts)),
                    "operator_names": ["noop", "copy", "add_mod", "compare", "select", "rank", "sort", "swap", "reduce"],
                    "operator_counts": [int(count) for count in counts],
                    "operator_fractions": [
                        float(count / max(sum(counts), 1)) for count in counts
                    ],
                }
                for track, counts in sorted(adaptive_v3_route_totals.items())
            },
            "progression_groups": {
                track: {
                    group: {
                        "accuracy": 100.0 * counts[0] / max(counts[1], 1),
                        "exact_match": 100.0 * counts[3] / max(counts[2], 1),
                        "episodes": int(counts[2]),
                        "answers": int(counts[1]),
                    }
                    for (group_track, group), counts in sorted(progression_totals.items())
                    if group_track == track
                }
                for track in sorted({key[0] for key in progression_totals})
            },
            "query_types": {
                track: {
                    query_type: {
                        "accuracy": 100.0 * counts[0] / max(counts[1], 1),
                        "exact_match": 100.0 * counts[3] / max(counts[2], 1),
                        "episodes": int(counts[2]),
                        "answers": int(counts[1]),
                    }
                    for (query_track, query_type), counts in sorted(query_type_totals.items())
                    if query_track == track
                }
                for track in sorted({key[0] for key in query_type_totals})
            },
            "attentional_blink_by_lag_mask": {
                query_type: {
                    "t1_accuracy": 100.0 * counts[0] / max(counts[2], 1),
                    "t2_accuracy": 100.0 * counts[1] / max(counts[2], 1),
                    "episode_exact_match": 100.0 * counts[3] / max(counts[2], 1),
                    "episodes": int(counts[2]),
                }
                for query_type, counts in sorted(blink_t2_totals.items())
            },
        }
        if gnw_process_totals:
            signatures = {}
            for (track, condition), counts in sorted(gnw_process_totals.items()):
                if track == "no_report_confidence":
                    lower, upper = wilson_interval(counts[0], counts[1])
                    metrics = {
                        "evidence_action_accuracy": 100.0 * counts[0] / max(counts[1], 1),
                        "evidence_action_accuracy_wilson95_pct": [100.0 * lower, 100.0 * upper],
                        "latent_truth_alignment": 100.0 * counts[2] / max(counts[3], 1),
                        "reported_confidence_bin_accuracy": (
                            100.0 * counts[4] / counts[5] if counts[5] else math.nan
                        ),
                        "episodes": counts[1],
                        "reported_episodes": counts[5],
                        "small_sample_warning": counts[1] < 20,
                    }
                else:
                    lower, upper = wilson_interval(counts[0], counts[1])
                    metrics = {
                        "percept_response_accuracy": 100.0 * counts[0] / max(counts[1], 1),
                        "percept_response_accuracy_wilson95_pct": [100.0 * lower, 100.0 * upper],
                        "prime_intrusion_rate_on_conflict": 100.0 * counts[2] / max(counts[3], 1),
                        "latent_target_alignment": 100.0 * counts[4] / max(counts[5], 1),
                        "episodes": counts[1],
                        "conflict_episodes": counts[3],
                        "small_sample_warning": counts[1] < 20 or counts[3] < 20,
                    }
                signatures.setdefault(track, {})[condition] = metrics
            diagnostics["gnw_process_signatures"] = signatures
            if no_report_confidence_scores:
                diagnostics["gnw_no_report_confidence_calibration"] = {
                    "n_reported": len(no_report_confidence_scores),
                    "brier_score": binary_brier(
                        no_report_confidence_scores, no_report_confidence_correct
                    ),
                    "ece_10_bins": expected_calibration_error(
                        no_report_confidence_scores, no_report_confidence_correct, bins=10
                    ),
                    "type2_roc_auc": type2_auc(
                        no_report_confidence_scores, no_report_confidence_correct
                    ),
                    "confidence_score_definition": "expected normalized confidence-bin probability (bin centers)",
                    "small_sample_warning": len(no_report_confidence_scores) < 20,
                }
            if masked_prime_condition_totals:
                conditions = {}
                for (visibility, congruency), counts in sorted(masked_prime_condition_totals.items()):
                    low, high = wilson_interval(counts[0], counts[4])
                    intrusion_low, intrusion_high = wilson_interval(counts[1], counts[2])
                    conditions[f"{visibility}_{congruency}"] = {
                        "target_response_accuracy_pct": 100.0 * counts[0] / max(counts[4], 1),
                        "target_response_accuracy_wilson95_pct": [100.0 * low, 100.0 * high],
                        "prime_intrusion_on_conflict_pct": 100.0 * counts[1] / max(counts[2], 1),
                        "prime_intrusion_wilson95_pct": [
                            100.0 * intrusion_low, 100.0 * intrusion_high
                        ],
                        "conflict_episodes": counts[2],
                        "latent_target_alignment_pct": 100.0 * counts[3] / max(counts[4], 1),
                        "episodes": counts[4],
                        "small_sample_warning": counts[4] < 20 or counts[2] < 20,
                    }
                conflict_contrast = math.nan
                conflict_contrast_ci = [math.nan, math.nan]
                visible_conflict = masked_prime_condition_totals.get(
                    ("visible", "incongruent"), [0, 0, 0, 0, 0]
                )
                masked_conflict = masked_prime_condition_totals.get(
                    ("masked", "incongruent"), [0, 0, 0, 0, 0]
                )
                if visible_conflict[2] > 0 and masked_conflict[2] > 0:
                    visible_rate = visible_conflict[1] / visible_conflict[2]
                    masked_rate = masked_conflict[1] / masked_conflict[2]
                    conflict_contrast = masked_rate - visible_rate
                    # Bonferroni-adjusted Wilson intervals for the two
                    # independent conflict conditions. Congruent cells have
                    # no conflicts by construction and cannot define this
                    # contrast or a visibility×congruency interaction.
                    simultaneous_z = 2.241
                    visible_interval = wilson_interval(
                        visible_conflict[1], visible_conflict[2], z=simultaneous_z
                    )
                    masked_interval = wilson_interval(
                        masked_conflict[1], masked_conflict[2], z=simultaneous_z
                    )
                    conflict_contrast_ci = [
                        masked_interval[0] - visible_interval[1],
                        masked_interval[1] - visible_interval[0],
                    ]
                diagnostics["gnw_masked_priming_conditions"] = {
                    "conditions": conditions,
                    "masked_minus_visible_prime_intrusion_on_conflict": conflict_contrast,
                    "conflict_intrusion_difference_simultaneous_wilson95": conflict_contrast_ci,
                    "conflict_intrusion_difference_definition": (
                        "masked minus visible prime-intrusion rate on incongruent trials; "
                        "congruent trials have no conflict by construction"
                    ),
                    "visible_conflict_episodes": visible_conflict[2],
                    "masked_conflict_episodes": masked_conflict[2],
                }
            diagnostics["gnw_process_context"] = {
                "workspace_enabled": bool(getattr(model, "workspace_enabled", False)),
                "workspace_readback_bypassed": bool(ablate_workspace),
                "workspace_readback_shuffled": bool(shuffle_workspace_readback),
                "no_report_exact_aggregation": bool(
                    getattr(model, "workspace_no_report_confidence_readout_enabled", False)
                ),
                "masked_prime_trace_enabled": bool(
                    getattr(model, "workspace_masked_prime_trace_enabled", False)
                ),
                "masked_prime_trace_read_accuracy_by_condition": {
                    condition: 100.0 * counts[0] / max(counts[1], 1)
                    for condition, counts in sorted(masked_prime_trace_reads.items())
                },
                "masked_prime_trace_reads_by_condition": {
                    condition: counts[1]
                    for condition, counts in sorted(masked_prime_trace_reads.items())
                },
            }
        if ignition_access_totals:
            diagnostics["gnw_ignition_access_curve"] = {
                str(strength): {
                    "hit_rate": 100.0 * counts[0] / max(counts[1], 1),
                    "predicted_access_wilson95_pct": [
                        100.0 * wilson_interval(counts[0], counts[1])[0],
                        100.0 * wilson_interval(counts[0], counts[1])[1],
                    ],
                    "false_alarm_rate": 100.0 * counts[2] / max(counts[3], 1),
                    "balanced_accuracy": 100.0
                    * (counts[0] + counts[3] - counts[2])
                    / max(counts[1] + counts[3], 1),
                    "balanced_accuracy_wilson95_pct": [
                        100.0
                        * wilson_interval(
                            counts[0] + counts[3] - counts[2], counts[1] + counts[3]
                        )[0],
                        100.0
                        * wilson_interval(
                            counts[0] + counts[3] - counts[2], counts[1] + counts[3]
                        )[1],
                    ],
                    "response_accuracy": 100.0 * counts[4] / max(counts[5], 1),
                    "present_trials": counts[1],
                    "absent_trials": counts[3],
                    "episodes": counts[5],
                    "small_sample_warning": min(counts[1], counts[3]) < 20,
                }
                for strength, counts in sorted(ignition_access_totals.items())
            }
            balanced_accuracy_counts = {}
            for strength, counts in ignition_access_totals.items():
                positive_n, negative_n = counts[1], counts[3]
                if positive_n <= 0 or negative_n <= 0:
                    continue
                balanced_accuracy = 0.5 * (
                    counts[0] / positive_n + (negative_n - counts[2]) / negative_n
                )
                # Effective sample size for the equally weighted mean of
                # independent hit and correct-rejection rates. It equals
                # 2*n when both classes have n examples.
                effective_n = 4.0 / (1.0 / positive_n + 1.0 / negative_n)
                balanced_accuracy_counts[strength] = (
                    balanced_accuracy * effective_n,
                    effective_n,
                )
            diagnostics["gnw_ignition_psychometric_fit"] = psychometric_logistic(
                balanced_accuracy_counts,
                threshold_probability=0.75,
            )
        if bool(getattr(model, "workspace_nth_farthest_exact_readout", False)):
            diagnostics["nth_farthest_workspace"] = {
                "vector_coordinate_accuracy": 100.0
                * nth_vector_coordinate_correct
                / max(nth_vector_coordinate_count, 1),
                "vector_coordinates_correct": nth_vector_coordinate_correct,
                "vector_coordinates_total": nth_vector_coordinate_count,
                "vector_exact_accuracy": 100.0
                * nth_vector_exact_correct
                / max(nth_vector_exact_count, 1),
                "vectors_exact": nth_vector_exact_correct,
                "vectors_total": nth_vector_exact_count,
                "exact_readout_accuracy": 100.0 * nth_readout_correct / max(nth_readout_count, 1),
                "readouts_correct": nth_readout_correct,
                "readouts_total": nth_readout_count,
                "invalid_readouts": nth_invalid_readout_count,
                "readout_by_query_stratum": {
                    key: {
                        "accuracy": 100.0 * counts[0] / max(counts[1], 1),
                        "correct": counts[0],
                        "episodes": counts[1],
                    }
                    for key, counts in nth_readout_by_stratum.items()
                },
            }
        if bool(getattr(model, "workspace_retrieval_exact_readout", False)):
            diagnostics["retrieval_workspace"] = {
                **retrieval_totals,
                "write_value_accuracy": 100.0
                * retrieval_totals["write_value_correct"]
                / max(retrieval_totals["write_value_count"], 1),
                "queried_value_accuracy": 100.0
                * retrieval_totals["query_values_correct"]
                / max(retrieval_totals["query_values_count"], 1),
                "query_slot_lookup_rate": 100.0
                * retrieval_totals["query_slots_found"]
                / max(retrieval_totals["query_slots_total"], 1),
                "updated_query_value_accuracy": 100.0
                * retrieval_totals["updated_query_values_correct"]
                / max(retrieval_totals["updated_query_values_count"], 1),
                "untouched_query_value_accuracy": 100.0
                * retrieval_totals["untouched_query_values_correct"]
                / max(retrieval_totals["untouched_query_values_count"], 1),
                "initial_write_value_accuracy": 100.0
                * retrieval_totals.get("initial_value_correct", 0)
                / max(retrieval_totals.get("initial_value_count", 0), 1),
                "update_write_value_accuracy": 100.0
                * retrieval_totals.get("update_value_correct", 0)
                / max(retrieval_totals.get("update_value_count", 0), 1),
                "exact_readout_accuracy": 100.0
                * retrieval_totals["readout_exact"]
                / max(retrieval_totals["readout_count"], 1),
            }
        if binding_entity_mentions:
            binding_metrics = {
                "entity_write_coverage": (binding_assigned_mentions / binding_entity_mentions),
                "same_entity_slot_consistency": (
                    binding_repeated_entity_consistency / max(binding_repeated_entities, 1)
                ),
                "repeated_entities": binding_repeated_entities,
                "distinct_entity_slot_collision": (
                    binding_colliding_entity_pairs / max(binding_entity_pairs, 1)
                ),
                "distinct_entity_pairs": binding_entity_pairs,
                "fragmented_entities": binding_fragmented_entities,
                "write_route_split_fraction": (
                    binding_write_route_splits / max(binding_write_mentions_with_routes, 1)
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
        if operation_diagnostics_enabled:
            diagnostics["workspace_operation_diagnostics"] = {
                "by_operation": {
                    operation_name: {
                        "source_read_accuracy": 100.0 * counts[0] / max(counts[1], 1),
                        "source_reads_correct": counts[0],
                        "source_reads_total": counts[1],
                        "destination_commit_rate": 100.0 * counts[2] / max(counts[3], 1),
                        "destination_commits": counts[2],
                        "destination_attempts": counts[3],
                        "committed_value_accuracy": 100.0 * counts[4] / max(counts[5], 1),
                        "committed_values_correct": counts[4],
                        "committed_values_total": counts[5],
                    }
                    for operation_name, counts in sorted(operation_diagnostic_totals.items())
                },
                "final_query_by_track": {
                    track: {
                        "answer_accuracy": diagnostics["tracks"][track]["accuracy"],
                        "episode_exact_match": diagnostics["tracks"][track]["exact_match"],
                        "episodes": diagnostics["tracks"][track]["episodes"],
                        "answers": diagnostics["tracks"][track]["answers"],
                    }
                    for track in ("mixed_skill", "variable_binding")
                    if track in diagnostics["tracks"]
                },
            }
            diagnostics["variable_binding_stage_diagnostics"] = {
                "query_slot_read_accuracy": (
                    100.0 * variable_binding_query_read_correct
                    / max(variable_binding_query_read_count, 1)
                ),
                "query_slot_reads_correct": variable_binding_query_read_correct,
                "query_slot_reads_total": variable_binding_query_read_count,
                "by_assignment_length_and_operation": {
                    f"{complexity}/{operation}": {
                        "operand_read_accuracy": 100.0 * counts[0] / max(counts[1], 1),
                        "operand_values_correct": counts[0],
                        "operand_values_total": counts[1],
                        "destination_commit_rate": 100.0 * counts[2] / max(counts[3], 1),
                        "destination_commits": counts[2],
                        "destination_attempts": counts[3],
                        "committed_value_accuracy": 100.0 * counts[4] / max(counts[5], 1),
                        "committed_values_correct": counts[4],
                        "committed_values_total": counts[5],
                    }
                    for (complexity, operation), counts in sorted(
                        variable_binding_stage_totals.items()
                    )
                },
            }
        if bool(getattr(model, "workspace_rule_feedback_update_enabled", False)):
            diagnostics["workspace_rule_feedback_tracker"] = {
                "feedback_only_target_accuracy_pct": 100.0 * rule_feedback_tracker_correct
                / max(rule_feedback_tracker_count, 1),
                "correct_updates": rule_feedback_tracker_correct,
                "updates": rule_feedback_tracker_count,
            }
        if value_program_totals:
            diagnostics["workspace_value_programs"] = {
                "by_operation": {
                    operation: {
                        "operand_value_accuracy": 100.0 * counts[0] / max(counts[1], 1),
                        "operand_values_correct": counts[0],
                        "operand_values_total": counts[1],
                        "output_accuracy": 100.0 * counts[2] / max(counts[3], 1),
                        "outputs_correct": counts[2],
                        "outputs_total": counts[3],
                    }
                    for operation, counts in sorted(value_program_totals.items())
                }
            }
        self.last_eval_diagnostics = diagnostics
        self._last_eval_v3_answer_logits = {
            track: torch.cat(rows, dim=0) for track, rows in v3_answer_logits.items() if rows
        }
        for track, counts in sorted(adaptive_v3_route_totals.items()):
            total = max(sum(counts), 1)
            route_text = ",".join(
                f"{name}:{count / total:.1%}"
                for name, count in zip(
                    ("noop", "copy", "add_mod", "compare", "select", "rank", "sort", "swap", "reduce"), counts
                )
            )
            print(
                f"  [adaptive-v3/routes/{track}] candidates={sum(counts)} "
                f"{route_text}"
            )
        print(
            f"[workspace-capability/{self.task_track}] accuracy {accuracy:.2f}% | "
            f"episode exact {perfect:.2f}% | episodes {len(episodes)}"
        )
        for track, conditions in diagnostics.get("gnw_process_signatures", {}).items():
            for condition, metrics in conditions.items():
                if track == "no_report_confidence":
                    confidence_bin_accuracy = metrics["reported_confidence_bin_accuracy"]
                    confidence_bin_text = (
                        f"{confidence_bin_accuracy:.1f}%"
                        if math.isfinite(confidence_bin_accuracy)
                        else "n/a"
                    )
                    print(
                        f"  [GNW-process/{track}/{condition}] evidence-action="
                        f"{metrics['evidence_action_accuracy']:.1f}% | latent-truth alignment="
                        f"{metrics['latent_truth_alignment']:.1f}% | confidence-bin="
                        f"{confidence_bin_text} "
                        f"({metrics['episodes']} episodes; {metrics['reported_episodes']} reports)"
                    )
                else:
                    print(
                        f"  [GNW-process/{track}/{condition}] percept-response="
                        f"{metrics['percept_response_accuracy']:.1f}% | prime intrusion on "
                        f"conflict={metrics['prime_intrusion_rate_on_conflict']:.1f}% | "
                        f"latent-target alignment={metrics['latent_target_alignment']:.1f}% "
                        f"({metrics['episodes']} episodes)"
                    )
        for condition, accuracy_value in diagnostics.get("gnw_process_context", {}).get(
            "masked_prime_trace_read_accuracy_by_condition", {}
        ).items():
            print(
                f"  [GNW-process/masked-prime-trace/{condition}] workspace prime "
                f"read accuracy={accuracy_value:.1f}%"
            )
        for strength, values in diagnostics.get("gnw_ignition_access_curve", {}).items():
            print(
                f"  [GNW-ignition/strength_{strength}] hit rate="
                f"{values['hit_rate']:.1f}% | false alarm="
                f"{values['false_alarm_rate']:.1f}% | balanced accuracy="
                f"{values['balanced_accuracy']:.1f}% | response accuracy="
                f"{values['response_accuracy']:.1f}% (n={values['episodes']}; "
                f"present/absent={values['present_trials']}/{values['absent_trials']})"
            )
        rule_tracker = diagnostics.get("workspace_rule_feedback_tracker")
        if rule_tracker:
            print(
                "  [workspace-rule-feedback/tracker] "
                f"feedback-only target estimate={rule_tracker['feedback_only_target_accuracy_pct']:.1f}% "
                f"({rule_tracker['correct_updates']}/{rule_tracker['updates']} visible-feedback updates)"
            )
        ignition_fit = diagnostics.get("gnw_ignition_psychometric_fit")
        if ignition_fit:
            print(
                "  [GNW-ignition/logistic-fit] "
                f"75% balanced-accuracy threshold="
                f"{ignition_fit['threshold_criterion']:.3g} | "
                f"95% CI=[{ignition_fit['threshold_criterion_wald95'][0]:.3g},"
                f"{ignition_fit['threshold_criterion_wald95'][1]:.3g}] | "
                f"in tested range={ignition_fit['threshold_criterion_in_observed_range']} | "
                f"slope={ignition_fit['slope']:.3g} "
                f"95% CI=[{ignition_fit['slope_wald95'][0]:.3g},"
                f"{ignition_fit['slope_wald95'][1]:.3g}] | n={ignition_fit['n']} "
                f"({ignition_fit['strength_levels']} strengths; "
                f"small_n={ignition_fit['small_sample_warning']})"
            )
        confidence_metrics = diagnostics.get("gnw_no_report_confidence_calibration")
        if confidence_metrics:
            print(
                "  [GNW-no-report/confidence-calibration] "
                f"Brier={confidence_metrics['brier_score']:.4f} | "
                f"ECE={confidence_metrics['ece_10_bins']:.4f} | "
                f"type-2 AUC={confidence_metrics['type2_roc_auc']:.4f} | "
                f"reported n={confidence_metrics['n_reported']}"
            )
        priming_metrics = diagnostics.get("gnw_masked_priming_conditions")
        if priming_metrics:
            for condition, metrics in priming_metrics["conditions"].items():
                print(
                    f"  [GNW-masked-priming/{condition}] "
                    f"target accuracy={metrics['target_response_accuracy_pct']:.1f}% "
                    f"CI95=[{metrics['target_response_accuracy_wilson95_pct'][0]:.1f},"
                    f"{metrics['target_response_accuracy_wilson95_pct'][1]:.1f}] | "
                    f"prime intrusion={metrics['prime_intrusion_on_conflict_pct']:.1f}% "
                    f"({metrics['conflict_episodes']} conflicts) "
                    f"n={metrics['episodes']}"
                )
            print(
                "  [GNW-masked-priming] masked-visible conflict intrusion difference="
                f"{priming_metrics['masked_minus_visible_prime_intrusion_on_conflict']:.4f} "
                f"95% simultaneous CI=["
                f"{priming_metrics['conflict_intrusion_difference_simultaneous_wilson95'][0]:.4f},"
                f"{priming_metrics['conflict_intrusion_difference_simultaneous_wilson95'][1]:.4f}] "
                f"(masked/visible n={priming_metrics['masked_conflict_episodes']}/"
                f"{priming_metrics['visible_conflict_episodes']})"
            )
        nth_metrics = diagnostics.get("nth_farthest_workspace")
        if nth_metrics:
            print(
                "  [workspace-nth-farthest] "
                f"vector coordinate accuracy={nth_metrics['vector_coordinate_accuracy']:.1f}% "
                f"({nth_metrics['vector_coordinates_correct']}/{nth_metrics['vector_coordinates_total']}) | "
                f"exact candidate vectors={nth_metrics['vector_exact_accuracy']:.1f}% "
                f"({nth_metrics['vectors_exact']}/{nth_metrics['vectors_total']}) | "
                f"direct exact readout={nth_metrics['exact_readout_accuracy']:.1f}% "
                f"({nth_metrics['readouts_correct']}/{nth_metrics['readouts_total']}) | "
                f"missing/invalid candidate state={nth_metrics['invalid_readouts']}"
            )
            for stratum, values in sorted(nth_metrics["readout_by_query_stratum"].items()):
                print(
                    f"    [workspace-nth-farthest/{stratum}] "
                    f"exact readout={values['accuracy']:.1f}% "
                    f"({values['correct']}/{values['episodes']})"
                )
        retrieval_metrics = diagnostics.get("retrieval_workspace")
        if retrieval_metrics:
            print(
                "  [workspace-retrieval] "
                f"write value accuracy={retrieval_metrics['write_value_accuracy']:.1f}% "
                f"({retrieval_metrics['write_value_correct']}/{retrieval_metrics['write_value_count']}) | "
                f"query slot lookup={retrieval_metrics['query_slot_lookup_rate']:.1f}% "
                f"({retrieval_metrics['query_slots_found']}/{retrieval_metrics['query_slots_total']}) | "
                f"queried value accuracy={retrieval_metrics['queried_value_accuracy']:.1f}% "
                f"({retrieval_metrics['query_values_correct']}/{retrieval_metrics['query_values_count']}) | "
                f"updated/untouched query value accuracy="
                f"{retrieval_metrics['updated_query_value_accuracy']:.1f}%/"
                f"{retrieval_metrics['untouched_query_value_accuracy']:.1f}% | "
                f"initial/update write accuracy="
                f"{retrieval_metrics['initial_write_value_accuracy']:.1f}%/"
                f"{retrieval_metrics['update_write_value_accuracy']:.1f}% | "
                f"exact readout={retrieval_metrics['exact_readout_accuracy']:.1f}% "
                f"({retrieval_metrics['readout_exact']}/{retrieval_metrics['readout_count']}) | "
                f"invalid={retrieval_metrics['invalid_readouts']}"
            )
            for query_type, values in sorted(retrieval_metrics["readout_by_query_type"].items()):
                print(
                    f"    [workspace-retrieval/{query_type}] exact readout="
                    f"{100.0 * values['correct'] / max(values['episodes'], 1):.1f}% "
                    f"({values['correct']}/{values['episodes']})"
                )
        operation_metrics = diagnostics.get("workspace_operation_diagnostics")
        if operation_metrics:
            for operation_name, metrics in sorted(operation_metrics["by_operation"].items()):
                print(
                    f"  [workspace-operation/{operation_name}] "
                    f"source read={metrics['source_read_accuracy']:.1f}% "
                    f"({metrics['source_reads_correct']}/{metrics['source_reads_total']}) | "
                    f"destination commit={metrics['destination_commit_rate']:.1f}% "
                    f"({metrics['destination_commits']}/{metrics['destination_attempts']}) | "
                    f"committed value={metrics['committed_value_accuracy']:.1f}% "
                    f"({metrics['committed_values_correct']}/"
                    f"{metrics['committed_values_total']})"
                )
            for track, metrics in sorted(operation_metrics["final_query_by_track"].items()):
                print(
                    f"  [workspace-operation/final-query/{track}] "
                    f"answer={metrics['answer_accuracy']:.1f}% | "
                    f"episode exact={metrics['episode_exact_match']:.1f}% "
                    f"n={metrics['episodes']}"
                )
            variable_metrics = diagnostics.get("variable_binding_stage_diagnostics")
            if variable_metrics and variable_metrics["query_slot_reads_total"]:
                print(
                    "  [workspace-variable-binding/query-read] "
                    f"accuracy={variable_metrics['query_slot_read_accuracy']:.1f}% "
                    f"({variable_metrics['query_slot_reads_correct']}/"
                    f"{variable_metrics['query_slot_reads_total']})"
                )
            if variable_metrics:
                for stage, values in sorted(
                    variable_metrics["by_assignment_length_and_operation"].items()
                ):
                    print(
                        f"  [workspace-variable-binding/{stage}] "
                        f"operand reads={values['operand_read_accuracy']:.1f}% "
                        f"({values['operand_values_correct']}/{values['operand_values_total']}) | "
                        f"commit={values['destination_commit_rate']:.1f}% "
                        f"({values['destination_commits']}/{values['destination_attempts']}) | "
                        f"committed values={values['committed_value_accuracy']:.1f}% "
                        f"({values['committed_values_correct']}/"
                        f"{values['committed_values_total']})"
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
        for operation, metrics in sorted(
            diagnostics.get("workspace_value_programs", {}).get("by_operation", {}).items()
        ):
            print(
                f"  [workspace-value-program/{operation}] "
                f"operand accuracy={metrics['operand_value_accuracy']:.1f}% "
                f"({metrics['operand_values_correct']}/{metrics['operand_values_total']}) | "
                f"program output={metrics['output_accuracy']:.1f}% "
                f"({metrics['outputs_correct']}/{metrics['outputs_total']})"
            )
        for track, metrics in diagnostics["tracks"].items():
            print(
                f"  [{track}] answer {metrics['accuracy']:.2f}% | "
                f"exact {metrics['exact_match']:.2f}% | "
                f"context={metrics['mean_context_tokens']:.1f} tokens | n={metrics['episodes']}"
            )
        for track, groups in diagnostics.get("progression_groups", {}).items():
            for group, metrics in sorted(groups.items()):
                print(
                    f"    [{track}/{group}] exact={metrics['exact_match']:.1f}% "
                    f"answer={metrics['accuracy']:.1f}% n={metrics['episodes']}"
                )
        for query_type, metrics in sorted(
            diagnostics.get("attentional_blink_by_lag_mask", {}).items()
        ):
            print(
                f"    [attentional_blink/{query_type}] T1={metrics['t1_accuracy']:.1f}% "
                f"T2={metrics['t2_accuracy']:.1f}% exact="
                f"{metrics['episode_exact_match']:.1f}% n={metrics['episodes']}"
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

    def evaluate_gnw_balanced_probe(
        self,
        model,
        device,
        episodes_per_cell: int = 64,
        *,
        seed: int = 91_404,
        field_log=None,
        feature_ablations: bool = False,
    ):
        """Evaluate GNW probes on a deterministic, condition-balanced set.

        The selected strata prevent random sampling from omitting conflict,
        confidence, lag, or strength cells. Ignition is balanced by presence
        at each intensity; masked priming by visibility and prime congruency;
        and confidence by report condition and whether the evidence-supported
        action matches latent truth. This is a behavioral signature probe,
        not a consciousness test.
        """
        if not set(GNW_TRACKS).issubset(set(self._tracks())):
            raise ValueError(
                "balanced GNW evaluation requires --workspace-task-track gnw_all or all"
            )
        episodes_per_cell = int(episodes_per_cell)
        if episodes_per_cell < 1:
            raise ValueError("episodes_per_cell must be >= 1")

        rng = random.Random(int(seed))
        episodes: list[TaskEpisode] = []

        def collect(track: str, cells, key_fn):
            selected = {cell: [] for cell in cells}
            attempts = 0
            max_attempts = max(10_000, len(cells) * episodes_per_cell * 200)
            while any(len(values) < episodes_per_cell for values in selected.values()):
                if attempts >= max_attempts:
                    short = {
                        str(cell): episodes_per_cell - len(values)
                        for cell, values in selected.items()
                        if len(values) < episodes_per_cell
                    }
                    raise RuntimeError(
                        f"could not balance GNW probe {track} cells: {short}"
                    )
                episode = self._make_episode(track, 5, rng)
                cell = key_fn(episode)
                attempts += 1
                if cell in selected and len(selected[cell]) < episodes_per_cell:
                    selected[cell].append(episode)
            for values in selected.values():
                episodes.extend(values)

        collect(
            "ignition",
            tuple((strength, present) for strength in range(9) for present in (False, True)),
            lambda episode: (
                int(episode.metadata["stimulus_strength"]),
                bool(episode.metadata["target_present"]),
            ),
        )
        collect(
            "attentional_blink",
            tuple((lag, masked) for lag in range(1, 9) for masked in (False, True)),
            lambda episode: (
                int(episode.metadata["target_lag"]),
                bool(episode.metadata["mask_condition"]),
            ),
        )
        collect(
            "masked_priming",
            tuple((masked, congruent) for masked in (False, True) for congruent in (False, True)),
            lambda episode: (
                bool(episode.metadata["mask_condition"]),
                bool(episode.metadata["prime_congruent"]),
            ),
        )
        collect(
            "no_report_confidence",
            tuple((report, correct) for report in (False, True) for correct in (False, True)),
            lambda episode: (
                bool(episode.metadata["explicit_report"]),
                bool(episode.metadata["action_correct"]),
            ),
        )

        print(
            f"[workspace-capability/GNW-balanced-probe] seed={seed} "
            f"episodes_per_cell={episodes_per_cell} total_episodes={len(episodes)} "
            "(ignition strength×presence; blink lag×mask; priming visibility×congruency; "
            "confidence report×action-correctness)"
        )
        result = self._evaluate_episodes(model, device, episodes, field_log=field_log)
        full_diagnostics = self.last_eval_diagnostics
        workspace_available = bool(
            getattr(model, "workspace", None) is not None
            and getattr(model, "workspace_enabled", False)
        )
        if workspace_available:
            bypass_result = self._evaluate_episodes(
                model, device, episodes, ablate_workspace=True
            )
            bypass_diagnostics = self.last_eval_diagnostics
            shuffled_result = self._evaluate_episodes(
                model, device, episodes, shuffle_workspace_readback=True
            )
            shuffled_diagnostics = self.last_eval_diagnostics
        else:
            bypass_result = shuffled_result = None
            bypass_diagnostics = shuffled_diagnostics = {}
        adaptive_module = getattr(model, "workspace_adaptive", None)
        adaptive_eval_bypass = bool(getattr(model, "workspace_adaptive_eval_bypass", False))
        if adaptive_module is not None and not adaptive_eval_bypass:
            adaptive_bypass_result = self._evaluate_episodes(
                model, device, episodes, ablate_workspace_adaptive=True
            )
            adaptive_bypass_diagnostics = self.last_eval_diagnostics
        else:
            adaptive_bypass_result = None
            adaptive_bypass_diagnostics = {}
        feature_results = {}
        if feature_ablations:
            for feature_name, model_attribute in _FUNCTIONAL_FEATURE_ABLATIONS:
                if not bool(getattr(model, model_attribute, False)):
                    continue
                relevant_tracks = _FUNCTIONAL_FEATURE_TRACKS.get(feature_name, set())
                feature_episodes = [
                    episode for episode in episodes if episode.track in relevant_tracks
                ]
                if not feature_episodes:
                    continue
                feature_baseline = self._evaluate_episodes(
                    model, device, feature_episodes
                )
                feature_result = self._evaluate_episodes(
                    model,
                    device,
                    feature_episodes,
                    ablate_workspace_feature=feature_name,
                )
                feature_results[feature_name] = {
                    "track": next(iter(relevant_tracks)),
                    "baseline_accuracy_pct": float(feature_baseline[0]),
                    "accuracy_pct": float(feature_result[0]),
                    "episode_exact_pct": float(feature_result[1]),
                    "full_minus_feature_off_pp": float(
                        feature_baseline[0] - feature_result[0]
                    ),
                    "episodes": len(feature_episodes),
                }
                print(
                    f"  [GNW-balanced/feature-off/{feature_name}] accuracy="
                    f"{feature_result[0]:.2f}% (baseline {feature_baseline[0]:.2f}%) | drop="
                    f"{feature_baseline[0] - feature_result[0]:+.2f} pp | "
                    f"n={len(feature_episodes)}"
                )
        natural_report_n = max(256, episodes_per_cell * 4)
        natural_report_episodes: list[TaskEpisode] = []
        attempts = 0
        while len(natural_report_episodes) < natural_report_n:
            if attempts >= natural_report_n * 20:
                raise RuntimeError(
                    "could not build natural-prevalence reported-confidence sample"
                )
            episode = self._make_episode("no_report_confidence", 5, rng)
            attempts += 1
            if bool(episode.metadata["explicit_report"]):
                natural_report_episodes.append(episode)
        self._evaluate_episodes(model, device, natural_report_episodes)
        natural_confidence_metrics = self.last_eval_diagnostics.get(
            "gnw_no_report_confidence_calibration", {}
        )
        full_diagnostics["gnw_balanced_probe"] = {
            "seed": int(seed),
            "episodes_per_cell": episodes_per_cell,
            "episode_count": len(episodes),
            "confidence_outcome_prevalence": (
                "balanced 50/50 correct/incorrect; calibration scores here are "
                "balanced-prevalence diagnostics"
            ),
            "natural_report_calibration_episodes": natural_report_n,
            "balanced_cells": {
                "ignition": 18,
                "attentional_blink": 16,
                "masked_priming": 4,
                "no_report_confidence": 4,
            },
            "interpretation": "behavioral functional signatures; not a consciousness measure",
        }
        full_diagnostics["gnw_balanced_workspace_dependency"] = {
            "full_accuracy_pct": float(result[0]),
            "workspace_bypassed_accuracy_pct": (
                float(bypass_result[0]) if bypass_result is not None else math.nan
            ),
            "readback_shuffled_accuracy_pct": (
                float(shuffled_result[0]) if shuffled_result is not None else math.nan
            ),
            "full_minus_bypassed_pp": (
                float(result[0] - bypass_result[0]) if bypass_result is not None else math.nan
            ),
            "full_minus_shuffled_pp": (
                float(result[0] - shuffled_result[0]) if shuffled_result is not None else math.nan
            ),
            "episodes": len(episodes),
        }
        full_diagnostics["gnw_balanced_workspace_dependency_by_track"] = {
            track: {
                "full_exact_match_pct": float(metrics["exact_match"]),
                "workspace_bypassed_exact_match_pct": float(
                    bypass_diagnostics.get("tracks", {}).get(track, {}).get(
                        "exact_match", math.nan
                    )
                ),
                "readback_shuffled_exact_match_pct": float(
                    shuffled_diagnostics.get("tracks", {}).get(track, {}).get(
                        "exact_match", math.nan
                    )
                ),
                "full_minus_bypassed_pp": float(
                    metrics["exact_match"]
                    - bypass_diagnostics.get("tracks", {}).get(track, {}).get(
                        "exact_match", math.nan
                    )
                ),
                "full_minus_shuffled_pp": float(
                    metrics["exact_match"]
                    - shuffled_diagnostics.get("tracks", {}).get(track, {}).get(
                        "exact_match", math.nan
                    )
                ),
                "episodes": int(metrics["episodes"]),
            }
            for track, metrics in full_diagnostics.get("tracks", {}).items()
            if track in {"ignition", "attentional_blink", "masked_priming", "no_report_confidence"}
        }
        full_diagnostics["gnw_balanced_adaptive_dependency"] = {
            "full_accuracy_pct": float(result[0]),
            "full_episode_exact_match_pct": float(result[1]),
            "adaptive_pathway_bypassed_accuracy_pct": (
                float(adaptive_bypass_result[0])
                if adaptive_bypass_result is not None
                else math.nan
            ),
            "adaptive_pathway_bypassed_episode_exact_match_pct": (
                float(adaptive_bypass_result[1])
                if adaptive_bypass_result is not None
                else math.nan
            ),
            "full_minus_adaptive_bypassed_pp": (
                float(result[0] - adaptive_bypass_result[0])
                if adaptive_bypass_result is not None
                else math.nan
            ),
            "episodes": len(episodes),
            "evaluation_forced_bypass": adaptive_eval_bypass,
            "by_track": {
                track: {
                    "full_accuracy_pct": float(metrics["accuracy"]),
                    "adaptive_pathway_bypassed_accuracy_pct": float(
                        adaptive_bypass_diagnostics.get("tracks", {}).get(track, {}).get(
                            "accuracy", math.nan
                        )
                    ),
                    "full_exact_match_pct": float(metrics["exact_match"]),
                    "adaptive_pathway_bypassed_exact_match_pct": float(
                        adaptive_bypass_diagnostics.get("tracks", {}).get(track, {}).get(
                            "exact_match", math.nan
                        )
                    ),
                    "full_minus_adaptive_bypassed_pp": float(
                        metrics["exact_match"]
                        - adaptive_bypass_diagnostics.get("tracks", {}).get(track, {}).get(
                            "exact_match", math.nan
                        )
                    ),
                    "episodes": int(metrics["episodes"]),
                }
                for track, metrics in full_diagnostics.get("tracks", {}).items()
                if track in {
                    "ignition",
                    "attentional_blink",
                    "masked_priming",
                    "no_report_confidence",
                }
            },
        }
        if feature_ablations:
            full_diagnostics["gnw_balanced_feature_ablations"] = feature_results
        full_diagnostics["gnw_natural_report_confidence_calibration"] = {
            **natural_confidence_metrics,
            "episodes": len(natural_report_episodes),
        }
        self.last_eval_diagnostics = full_diagnostics
        dependency = full_diagnostics["gnw_balanced_workspace_dependency"]
        if workspace_available:
            print(
                "  [GNW-balanced/workspace-dependence] full/bypass/shuffled="
                f"{dependency['full_accuracy_pct']:.2f}%/"
                f"{dependency['workspace_bypassed_accuracy_pct']:.2f}%/"
                f"{dependency['readback_shuffled_accuracy_pct']:.2f}% | "
                f"drops={dependency['full_minus_bypassed_pp']:+.2f}/"
                f"{dependency['full_minus_shuffled_pp']:+.2f} pp"
            )
        if adaptive_bypass_result is not None:
            adaptive_dependency = full_diagnostics["gnw_balanced_adaptive_dependency"]
            print(
                "  [GNW-balanced/adaptive-workspace-dependence] full/bypassed="
                f"{result[0]:.2f}%/{adaptive_bypass_result[0]:.2f}% | "
                f"drop={adaptive_dependency['full_minus_adaptive_bypassed_pp']:+.2f} pp"
            )
            for track, route_metrics in full_diagnostics.get(
                "adaptive_workspace_routes_by_track", {}
            ).items():
                fractions = ",".join(
                    f"{value:.2f}" for value in route_metrics["route_token_fractions"]
                )
                print(
                    f"  [workspace-adaptive/routes/{track}] "
                    f"fraction={fractions} tokens={route_metrics['token_count']}"
                )
            for track, metrics in sorted(adaptive_dependency["by_track"].items()):
                print(
                    f"    [GNW-balanced/adaptive-workspace-dependence/{track}] "
                    f"accuracy full/bypassed={metrics['full_accuracy_pct']:.2f}%/"
                    f"{metrics['adaptive_pathway_bypassed_accuracy_pct']:.2f}% | "
                    f"exact drop={metrics['full_minus_adaptive_bypassed_pp']:+.2f} pp "
                    f"n={metrics['episodes']}"
                )
            for track, metrics in sorted(
                full_diagnostics["gnw_balanced_workspace_dependency_by_track"].items()
            ):
                print(
                    f"    [GNW-balanced/workspace-dependence/{track}] exact full/bypass/shuffled="
                    f"{metrics['full_exact_match_pct']:.2f}%/"
                    f"{metrics['workspace_bypassed_exact_match_pct']:.2f}%/"
                    f"{metrics['readback_shuffled_exact_match_pct']:.2f}% | "
                    f"drops={metrics['full_minus_bypassed_pp']:+.2f}/"
                    f"{metrics['full_minus_shuffled_pp']:+.2f} pp "
                    f"n={metrics['episodes']}"
                )
        print(
            "  [GNW-balanced/natural-report-calibration] "
            f"Brier={natural_confidence_metrics.get('brier_score', math.nan):.4f} | "
            f"ECE={natural_confidence_metrics.get('ece_10_bins', math.nan):.4f} | "
            f"type-2 AUC={natural_confidence_metrics.get('type2_roc_auc', math.nan):.4f} | "
            f"reported n={natural_confidence_metrics.get('n_reported', 0)}"
        )
        return result

    def _evaluate_paired_id(self, model, device, lesson_idx: int, **kwargs):
        episodes = self._paired_id_episodes(lesson_idx)
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
            raise ValueError("permutation state diagnostics require semantic entity addressing")
        if getattr(model, "workspace", None) is None:
            raise ValueError("permutation state diagnostics require an active workspace")
        if int(num_episodes) < 1 or int(max_events_per_episode) < 0:
            raise ValueError("state diagnostic episode/event limits are invalid")
        rng = rng or random.Random(190_001 + int(lesson_idx))
        source_episodes = [
            self._make_episode("permutation", lesson_idx, rng) for _ in range(int(num_episodes))
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
        prior_assignment_observer = getattr(workspace, "_initium_address_assignment_observer", None)
        prior_state_observer = getattr(workspace, "_initium_recurrent_state_observer", None)
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
                    x, _targets, _mask = self._collate(
                        episodes,
                        token_ids_only=bool(getattr(model, "workspace_token_id_input", False)),
                    )
                    transactional = bool(
                        getattr(model, "workspace_permutation_transactional", False)
                    )
                    entity_metadata = self.workspace_entity_address_metadata(
                        x,
                        paired_swap_commits=transactional,
                        attentional_item_memory=bool(
                            getattr(model, "workspace_attentional_item_memory_enabled", False)
                        ),
                        attentional_cue_readout=bool(
                            getattr(model, "workspace_attentional_cue_readout_enabled", False)
                        ),
                        episodes=episodes,
                    )
                    if transactional:
                        entity_metadata.update(
                            self.workspace_permutation_query_metadata(x, episodes)
                        )
                    if bool(getattr(model, "workspace_nth_farthest_exact_readout", False)):
                        nth_metadata = self.workspace_nth_farthest_metadata(x, episodes)
                        entity_metadata.update(
                            {
                                key: value
                                for key, value in nth_metadata.items()
                                if key != "nth_farthest_vector_targets"
                            }
                        )
                    if bool(getattr(model, "workspace_retrieval_exact_readout", False)):
                        retrieval_metadata = self.workspace_retrieval_metadata(x, episodes)
                        entity_metadata.update(
                            {
                                key: value
                                for key, value in retrieval_metadata.items()
                                if key
                                not in (
                                    "retrieval_value_targets",
                                    "retrieval_query_targets",
                                    "retrieval_query_was_updated",
                                    "retrieval_value_is_update",
                                )
                            }
                        )
                    operation_executor_enabled = bool(
                        getattr(model, "workspace_operation_executor_enabled", False)
                    )
                    adaptive_v3_enabled = bool(
                        getattr(model, "workspace_adaptive_v3_enabled", False)
                    )
                    if operation_executor_enabled or adaptive_v3_enabled:
                        operation_metadata = self.workspace_key_value_metadata(
                            x,
                            episodes,
                            include_operations=operation_executor_enabled,
                            include_adaptive_v3=adaptive_v3_enabled,
                            adaptive_v3_distillation=False,
                            adaptive_v3_holdout_tracks=tuple(
                                getattr(model, "workspace_adaptive_v3_holdout_tracks", ())
                            ),
                            rule_feedback_update=bool(
                                getattr(
                                    model,
                                    "workspace_rule_feedback_update_enabled",
                                    False,
                                )
                            ),
                            attentional_item_memory=bool(
                                getattr(model, "workspace_attentional_item_memory_enabled", False)
                            ),
                            no_report_confidence_readout=bool(
                                getattr(
                                    model,
                                    "workspace_no_report_confidence_readout_enabled",
                                    False,
                                )
                            ),
                            masked_prime_trace=bool(
                                getattr(model, "workspace_masked_prime_trace_enabled", False)
                            ),
                            entity_metadata=entity_metadata,
                        )
                        if operation_executor_enabled:
                            entity_metadata.update(
                                {
                                    "operation_metadata": operation_metadata[
                                        "workspace_operation_metadata"
                                    ],
                                    "operation_steps": operation_metadata[
                                        "workspace_operation_steps"
                                    ],
                                }
                            )
                        if adaptive_v3_enabled:
                            entity_metadata.update(
                                {
                                    "adaptive_operator_candidate_mask": operation_metadata[
                                        "adaptive_operator_candidate_mask"
                                    ],
                                    "adaptive_operator_execution_mask": operation_metadata[
                                        "adaptive_operator_execution_mask"
                                    ],
                                    "adaptive_operator_steps": operation_metadata[
                                        "adaptive_operator_steps"
                                    ],
                                }
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
                    # fmt: off
                    workspace._initium_address_assignment_observer = (
                        lambda step,
                        assignments,
                        assignment_trace=assignment_trace: assignment_trace.append(
                            (step, assignments.detach())
                        )
                    )
                    # fmt: on

                    def capture_event_state(
                        state,
                        call_index,
                        event_steps_needed=event_steps_needed,
                        recurrent_state_trace=recurrent_state_trace,
                    ):
                        if call_index in event_steps_needed:
                            recurrent_state_trace.append((call_index, state.detach().cpu()))

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
                    if bool(getattr(model, "workspace_attentional_cue_readout_enabled", False)):
                        logits = self.apply_workspace_attentional_cue_readout_logits(
                            logits, workspace.pop_attentional_cue_predictions()
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
                    bypass_logits = self._output_proj(bypass_output.transpose(0, 1).contiguous())
                    if transactional:
                        bypass_logits = self.apply_workspace_permutation_query_logits(
                            bypass_logits, workspace.pop_semantic_query_predictions()
                        )
                    if bool(getattr(model, "workspace_attentional_cue_readout_enabled", False)):
                        bypass_logits = self.apply_workspace_attentional_cue_readout_logits(
                            bypass_logits, workspace.pop_attentional_cue_predictions()
                        )
                    bypass_predictions = bypass_logits.argmax(dim=-1)
                    # fmt: off
                    workspace._initium_address_assignment_observer = (
                        lambda step,
                        assignments,
                        assignment_trace=assignment_trace: assignment_trace.append(
                            (step, assignments.detach())
                        )
                    )
                    # fmt: on

                    def capture_event_state(
                        state,
                        call_index,
                        event_steps_needed=event_steps_needed,
                        recurrent_state_trace=recurrent_state_trace,
                    ):
                        if call_index in event_steps_needed:
                            recurrent_state_trace.append((call_index, state.detach().cpu()))

                    workspace._initium_recurrent_state_observer = capture_event_state
                    if prior_workspace_bypass is not None:
                        model.workspace_eval_bypass = prior_workspace_bypass
                    assignments_by_step = {step: value.cpu() for step, value in assignment_trace}
                    states_by_step = {step: value for step, value in recurrent_state_trace}
                    for row_index, item in enumerate(group):
                        probe = item["probe"]
                        answer_index = len(probe.inputs)
                        predicted_token = int(predictions[row_index, answer_index])
                        bypass_predicted_token = int(bypass_predictions[row_index, answer_index])
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
                        write_steps = (
                            torch.nonzero(
                                entity_write_mask[row_index]
                                & entity_ids[row_index].eq(item["entity_id"]),
                                as_tuple=False,
                            )
                            .flatten()
                            .tolist()
                        )
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
                        assigned_slot = slot_counts.most_common(1)[0][0] if slot_counts else -1
                        slot_route_split = len(slot_counts) > 1
                        slot_rms = None
                        state = states_by_step.get(item["event_step"])
                        if state is not None and assigned_slot >= 0:
                            state_row = state[row_index].float()
                            if getattr(workspace, "entity_keyed_writes", False):
                                state_row = torch.split(
                                    state_row, (workspace._address_dim, workspace.d_model), dim=-1
                                )[1]
                            slot_rms = float(state_row[assigned_slot].square().mean().sqrt().cpu())
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
                    "bypass_accuracy": (100.0 * event_bypass_correct / max(len(event_rows), 1)),
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
        episodes = self._paired_id_episodes(lesson_idx)
        return self._evaluate_episodes(model, device, episodes, ablate_memory=True)[:2]

    def evaluate_workspace_ablated(self, model, device, curriculum, lesson_idx):
        episodes = self._paired_id_episodes(lesson_idx)
        return self._evaluate_episodes(model, device, episodes, ablate_workspace=True)[:2]

    def evaluate_workspace_adaptive_ablated(self, model, device, curriculum, lesson_idx):
        """Evaluate paired ID episodes with adaptive workspace mechanisms bypassed."""
        if bool(getattr(model, "workspace_adaptive_v3_enabled", False)):
            count = int(getattr(model, "workspace_adaptive_v3_eval_episodes_per_track", 128))
            tracks = tuple(
                getattr(model, "workspace_adaptive_v3_feature_off_tracks", ())
            ) or ("mixed_skill", "variable_binding")
            rng = random.Random(1_910_003 + int(lesson_idx))
            episodes = [
                self._make_episode(track, int(lesson_idx), rng)
                for track in tracks
                for _ in range(count)
            ]
            full = self._evaluate_episodes(
                model,
                device,
                episodes,
                ablate_workspace_operation_executor=True,
            )
            full_logits = getattr(self, "_last_eval_v3_answer_logits", {})
            full_episode_results = getattr(self, "_last_eval_episode_results", {})
            bypass = self._evaluate_episodes(
                model,
                device,
                episodes,
                ablate_workspace_adaptive=True,
                ablate_workspace_operation_executor=True,
            )
            bypass_logits = getattr(self, "_last_eval_v3_answer_logits", {})
            bypass_episode_results = getattr(self, "_last_eval_episode_results", {})
            paired_metrics = {}
            for track in tracks:
                enabled_rows = full_episode_results.get(track, ())
                bypass_rows = bypass_episode_results.get(track, ())
                paired_count = min(len(enabled_rows), len(bypass_rows))
                if paired_count:
                    enabled_acc = torch.tensor(
                        [row[0] for row in enabled_rows[:paired_count]], dtype=torch.float64
                    )
                    bypass_acc = torch.tensor(
                        [row[0] for row in bypass_rows[:paired_count]], dtype=torch.float64
                    )
                    enabled_exact = torch.tensor(
                        [row[1] for row in enabled_rows[:paired_count]], dtype=torch.float64
                    )
                    bypass_exact = torch.tensor(
                        [row[1] for row in bypass_rows[:paired_count]], dtype=torch.float64
                    )
                    accuracy_delta = enabled_acc - bypass_acc
                    exact_delta = enabled_exact - bypass_exact
                    generator = torch.Generator().manual_seed(1_910_003 + paired_count)
                    indices = torch.randint(
                        paired_count, (4096, paired_count), generator=generator
                    )
                    accuracy_bootstrap = accuracy_delta[indices].mean(dim=1) * 100.0
                    exact_bootstrap = exact_delta[indices].mean(dim=1) * 100.0
                    accuracy_ci = torch.quantile(
                        accuracy_bootstrap, torch.tensor([0.025, 0.975], dtype=torch.float64)
                    ).tolist()
                    exact_ci = torch.quantile(
                        exact_bootstrap, torch.tensor([0.025, 0.975], dtype=torch.float64)
                    ).tolist()
                    paired_metrics[track] = {
                        "episodes": paired_count,
                        "v3_accuracy_pct": float(enabled_acc.mean().item() * 100.0),
                        "bypass_accuracy_pct": float(bypass_acc.mean().item() * 100.0),
                        "accuracy_delta_pp": float(accuracy_delta.mean().item() * 100.0),
                        "accuracy_delta_bootstrap95_pp": accuracy_ci,
                        "v3_exact_pct": float(enabled_exact.mean().item() * 100.0),
                        "bypass_exact_pct": float(bypass_exact.mean().item() * 100.0),
                        "exact_delta_pp": float(exact_delta.mean().item() * 100.0),
                        "exact_delta_bootstrap95_pp": exact_ci,
                    }
                    print(
                        f"  [adaptive-v3/paired/{track}] n={paired_count} "
                        f"accuracy {enabled_acc.mean().item() * 100:.2f}% vs "
                        f"{bypass_acc.mean().item() * 100:.2f}% "
                        f"delta={accuracy_delta.mean().item() * 100:+.2f}pp "
                        f"bootstrap95=[{accuracy_ci[0]:+.2f},{accuracy_ci[1]:+.2f}] "
                        f"exact {enabled_exact.mean().item() * 100:.2f}% vs "
                        f"{bypass_exact.mean().item() * 100:.2f}% "
                        f"delta={exact_delta.mean().item() * 100:+.2f}pp "
                        f"bootstrap95=[{exact_ci[0]:+.2f},{exact_ci[1]:+.2f}]"
                    )
            if bool(getattr(model, "workspace_adaptive_v3_logit_diagnostics_enabled", False)):
                by_track = {}
                for track in sorted(set(full_logits) & set(bypass_logits)):
                    enabled_logits = full_logits[track].float()
                    bypassed_logits = bypass_logits[track].float()
                    count = min(enabled_logits.shape[0], bypassed_logits.shape[0])
                    enabled_logits = enabled_logits[:count]
                    bypassed_logits = bypassed_logits[:count]
                    enabled_logits64 = enabled_logits.double()
                    bypassed_logits64 = bypassed_logits.double()
                    enabled_logp = enabled_logits64.log_softmax(dim=-1)
                    bypass_logp = bypassed_logits64.log_softmax(dim=-1)
                    enabled_prob = enabled_logp.exp()
                    bypass_prob = bypass_logp.exp()
                    midpoint = 0.5 * (enabled_prob + bypass_prob)
                    midpoint_log = midpoint.clamp_min(1e-8).log()
                    js = 0.5 * (
                        enabled_prob * (enabled_logp - midpoint_log)
                    ).sum(dim=-1) + 0.5 * (
                        bypass_prob * (bypass_logp - midpoint_log)
                    ).sum(dim=-1)
                    js = js.clamp_min(0.0)
                    mean_abs_logit_shift = (
                        enabled_logits.double() - bypassed_logits.double()
                    ).abs().mean()
                    flips = enabled_logits.argmax(dim=-1).ne(
                        bypassed_logits.argmax(dim=-1)
                    )
                    by_track[track] = {
                        "answer_tokens": int(count),
                        "mean_js_divergence_nats": float(js.mean().item()) if count else 0.0,
                        "mean_abs_logit_shift": float(mean_abs_logit_shift.item()) if count else 0.0,
                        "prediction_flip_fraction": float(flips.float().mean().item()) if count else 0.0,
                    }
                    print(
                        f"  [adaptive-v3/logits/{track}] tokens={count} "
                        f"JS={by_track[track]['mean_js_divergence_nats']:.6g} nats "
                        f"mean_abs_logit_shift={by_track[track]['mean_abs_logit_shift']:.6g} "
                        f"prediction_flips={by_track[track]['prediction_flip_fraction']:.2%}"
                    )
                self.last_eval_diagnostics["adaptive_v3_logit_effect"] = by_track
            self.last_eval_diagnostics["adaptive_v3_paired_metrics"] = paired_metrics
            return bypass[:2]
        episodes = self._paired_id_episodes(lesson_idx)
        return self._evaluate_episodes(
            model, device, episodes, ablate_workspace_adaptive=True
        )[:2]

    def evaluate_workspace_shuffled(self, model, device, curriculum, lesson_idx):
        episodes = self._paired_id_episodes(lesson_idx)
        return self._evaluate_episodes(model, device, episodes, shuffle_workspace_readback=True)[:2]

    def evaluate_workspace_feature_ablated(
        self, model, device, curriculum, lesson_idx, feature_name: str
    ):
        """Run a paired evaluation with one enabled task-specific pathway disabled."""
        episodes = self._paired_id_episodes(lesson_idx)
        return self._evaluate_episodes(
            model,
            device,
            episodes,
            ablate_workspace_feature=feature_name,
        )[:2]

    @staticmethod
    def enabled_workspace_functional_features(model) -> tuple[str, ...]:
        """List active task-specific workspace mechanisms eligible for ablation."""
        return tuple(
            name
            for name, attribute in _FUNCTIONAL_FEATURE_ABLATIONS
            if bool(getattr(model, attribute, False))
        )

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
            "track",
            "query_type",
            "sequence_length_bucket",
            "relevant_updates_bucket",
            "hop_count",
            "source_count",
            "correction_relevant",
            "novelty_distractor",
            "equal_priority_conflict",
            "distractor_count_bucket",
            "candidate_count_bucket",
            "answers",
            "correct_answers",
            "episodes",
            "exact_episodes",
        ]

    def write_field_log(self, writer, file, step, lesson, eval_type, field_log):
        for key, values in sorted(field_log.items()):
            writer.writerow([step, lesson, eval_type, *key, *values])
        file.flush()
