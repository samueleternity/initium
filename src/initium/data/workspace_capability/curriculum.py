"""Track-specific difficulty schedules and advancement policy."""

from __future__ import annotations

import random
from collections.abc import Iterable

_LEVELS = 6


def difficulty_for(
    track: str,
    level: int,
    *,
    ood: bool = False,
    ood_axes: Iterable[str] | None = None,
) -> dict[str, int | float | str | bool | tuple[str, ...]]:
    """Return a fresh serializable configuration for a track/lesson."""
    level = max(0, min(int(level), _LEVELS - 1))
    if ood_axes is None:
        if ood and track == "permutation":
            axes = {"labels", "length"}
        else:
            axes = {"legacy"} if ood else set()
    else:
        axes = {str(axis).strip().lower() for axis in ood_axes}
        unknown_axes = axes - {"labels", "length"}
        if unknown_axes:
            raise ValueError(f"unknown OOD axes for {track!r}: {sorted(unknown_axes)}")
        if track != "permutation":
            raise ValueError("explicit OOD axes are currently supported only for permutation")
    config: dict[str, int | float | str | bool | tuple[str, ...]]
    if track == "permutation":
        config = {
            "entities": (4, 6, 8, 10, 12, 16)[level],
            "swaps": (8, 16, 32, 64, 128, 256)[level],
            "distractors": (0, 2, 4, 8, 16, 32)[level],
        }
    elif track == "entity_state":
        config = {
            "entities": (4, 6, 8, 10, 12, 16)[level],
            "sources": (2, 3, 4, 5, 6, 8)[level],
            "updates": (0, 2, 4, 8, 16, 24)[level],
            "distractors": (0, 2, 4, 8, 12, 16)[level],
            "correction": level >= 1,
            "novelty": level >= 1,
        }
    elif track == "relational_join":
        config = {
            "hops": (1, 1, 2, 2, 3, 3)[level],
            "distractors": (0, 2, 4, 8, 16, 32)[level],
        }
    elif track == "nth_farthest":
        candidates = (4, 5, 6, 8, 12, 16)[level]
        config = {
            "candidates": candidates,
            "dimensions": (2, 2, 3, 4, 4, 6)[level],
            "rank": min((1, 2, 3, 4, 6, 8)[level], candidates),
            "control_probability": 0.2,
            "distractors": (0, 1, 2, 4, 8, 16)[level],
        }
    elif track == "retrieval_aggregation":
        config = {
            "keys": (4, 8, 12, 16, 24, 32)[level],
            "updates": (0, 2, 4, 8, 12, 16)[level],
            "distractors": (0, 2, 4, 8, 16, 32)[level],
            "queries": (2, 2, 2, 3, 4, 8)[level],
            "aggregate": (
                ("recall", "sum_mod16", "count_even", "parity"),
                ("recall", "sum_mod16", "count_even", "parity"),
                "sum_mod16", "count_even", "parity", "recall",
            )[level],
        }
    elif track == "capacity_bottleneck":
        config = {
            "set_size": (2, 3, 4, 5, 6, 8)[level],
            "target_count": (2, 3, 4, 5, 6, 8)[level],
            "tracking_frames": (2, 3, 4, 4, 5, 6)[level],
            "n_back": (1, 1, 2, 2, 3, 4)[level],
            "delay_events": (0, 1, 2, 4, 6, 8)[level],
            "stream_length": (6, 8, 10, 12, 14, 16)[level],
        }
    elif track == "belief_tracking":
        config = {
            "agents": (2, 2, 3, 4, 5, 5)[level],
            "objects": (1, 2, 3, 4, 6, 8)[level],
            "moves": (2, 3, 4, 5, 6, 8)[level],
        }
    elif track == "mixed_skill":
        config = {
            "operations": (4, 5, 6, 7, 8, 8)[level],
            # "operations": (8, 8, 8, 8, 8, 8)[level], # Revert it afterwards. This tests whether the operators can be learned and routed at all, independent of promotion.
            "keys": (4, 5, 6, 7, 8, 10)[level],
        }
    elif track == "rule_switching":
        config = {
            "evidence_trials": (4, 6, 8, 10, 12, 14)[level],
            "switches": (0, 0, 1, 2, 3, 4)[level],
            "feedback_noise": (0.20, 0.20, 0.22, 0.25, 0.28, 0.30)[level],
        }
    elif track == "attentional_blink":
        config = {
            "maximum_lag": (2, 2, 3, 4, 6, 8)[level],
            "distractors": (4, 6, 8, 10, 12, 14)[level],
            "lure_probability": (0.40, 0.50, 0.60, 0.70, 0.80, 0.85)[level],
        }
    elif track == "variable_binding":
        config = {
            "variables": (5, 5, 6, 7, 8, 10)[level],
            "assignments": (7, 9, 11, 13, 17, 21)[level],
        }
    elif track == "ignition":
        config = {"maximum_strength": (3, 4, 5, 6, 7, 8)[level]}
    elif track == "psychological_refractory":
        config = {"maximum_soa": (2, 3, 4, 5, 6, 7)[level]}
    elif track == "masked_priming":
        config = {"masked_probability": (0.35, 0.45, 0.55, 0.65, 0.75, 0.8)[level]}
    elif track == "sustained_maintenance":
        config = {"maximum_delay": (3, 5, 7, 9, 11, 12)[level]}
    elif track == "no_report_confidence":
        config = {"maximum_evidence": (4, 5, 6, 7, 8, 8)[level]}
    elif track == "bistable_perception":
        config = {"maximum_frames": (4, 5, 6, 7, 8, 10)[level]}
    elif track == "trace_conditioning":
        config = {"maximum_gap": (1, 2, 3, 4, 5, 6)[level]}
    elif track == "conditional_ops":
        config = {
            "operations": (2, 3, 4, 5, 6, 8)[level],
            "keys": (4, 5, 6, 7, 8, 10)[level],
        }
    else:
        raise ValueError(f"unknown workspace capability track {track!r}")
    if ood and track != "permutation":
        # OOD increases task-specific load beyond the ID lesson while keeping
        # each generator's token and sequence bounds valid.
        if track == "entity_state":
            updates = config["updates"]
            distractors = config["distractors"]
            assert isinstance(updates, int) and isinstance(distractors, int)
            config["updates"] = min(updates + 8, 32)
            config["distractors"] = min(distractors * 2 + 2, 24)
        elif track == "relational_join":
            distractors = config["distractors"]
            assert isinstance(distractors, int)
            config["distractors"] = min(distractors * 2 + 4, 64)
        elif track == "nth_farthest":
            distractors = config["distractors"]
            assert isinstance(distractors, int)
            config["distractors"] = min(distractors * 2 + 2, 24)
        elif track == "capacity_bottleneck":
            config["set_size"] = min(int(config["set_size"]) + 1, 9)
            config["target_count"] = min(int(config["target_count"]) + 1, 9)
            config["n_back"] = min(int(config["n_back"]) + 1, 5)
            config["tracking_frames"] = min(int(config["tracking_frames"]) + 1, 6)
            config["delay_events"] = min(int(config["delay_events"]) + 2, 10)
            config["stream_length"] = min(int(config["stream_length"]) + 2, 16)
        elif track == "belief_tracking":
            config["moves"] = min(int(config["moves"]) + 2, 10)
        elif track == "mixed_skill":
            config["keys"] = min(int(config["keys"]) + 2, 10)
            config["operations"] = min(int(config["operations"]) + 1, 8)
        elif track == "rule_switching":
            config["evidence_trials"] = min(int(config["evidence_trials"]) + 2, 16)
            config["switches"] = min(int(config["switches"]) + 1, 5)
            config["feedback_noise"] = min(float(config["feedback_noise"]) + 0.05, 0.35)
        elif track == "attentional_blink":
            config["maximum_lag"] = min(int(config["maximum_lag"]) + 1, 9)
            config["distractors"] = min(int(config["distractors"]) + 2, 16)
            config["lure_probability"] = min(
                float(config["lure_probability"]) + 0.05, 0.90
            )
        elif track == "variable_binding":
            config["assignments"] = min(int(config["assignments"]) + 2, 24)
        elif track == "ignition":
            config["maximum_strength"] = min(int(config["maximum_strength"]) + 1, 8)
        elif track == "psychological_refractory":
            config["maximum_soa"] = min(int(config["maximum_soa"]) + 1, 8)
        elif track == "masked_priming":
            config["masked_probability"] = min(float(config["masked_probability"]) + 0.1, 0.9)
        elif track == "sustained_maintenance":
            config["maximum_delay"] = min(int(config["maximum_delay"]) + 2, 12)
        elif track == "no_report_confidence":
            config["maximum_evidence"] = min(int(config["maximum_evidence"]) + 1, 8)
        elif track == "bistable_perception":
            config["maximum_frames"] = min(int(config["maximum_frames"]) + 1, 10)
        elif track == "trace_conditioning":
            config["maximum_gap"] = min(int(config["maximum_gap"]) + 1, 6)
        elif track == "conditional_ops":
            config["keys"] = min(int(config["keys"]) + 2, 12)
            config["operations"] = min(int(config["operations"]) + 1, 8)
        else:
            updates = config["updates"]
            distractors = config["distractors"]
            assert isinstance(updates, int) and isinstance(distractors, int)
            config["updates"] = min(updates + 8, 24)
            config["distractors"] = min(distractors * 2 + 4, 64)
    if track == "permutation" and "length" in axes:
        swaps = config["swaps"]
        distractors = config["distractors"]
        assert isinstance(swaps, int) and isinstance(distractors, int)
        config["swaps"] = min(swaps * 2, 512)
        config["distractors"] = min(distractors * 2 + 4, 64)
    return config


class WorkspaceCapabilityCurriculum:
    """Fixed track curriculum advanced by exact-answer lesson accuracy."""

    def __init__(self, dataset):
        self.dataset = dataset
        # Store every track's level in the shared curriculum. The runner only
        # depends on ``len(table)``; keeping the per-track schedule here makes
        # the all-track curriculum inspectable/checkpoint-friendly without
        # pretending that ``all`` is itself a generator track.
        self.table = [
            {track: difficulty_for(track, level) for track in dataset._tracks()}
            for level in range(_LEVELS)
        ]
        self.lesson = 0
        # Split-graph sequence memories do not use DNC N, but retain this
        # interface for generic training/checkpoint code.
        self.lesson_nr_cells = [max(128, 32 * (i + 1)) for i in range(_LEVELS)]

    def maybe_advance(self, model, device, step=None, optimizer=None, *, allow_advance=True):
        field_log = {}
        evaluate_curriculum = getattr(self.dataset, "evaluate_curriculum_id", None)
        if callable(evaluate_curriculum):
            accuracy, perfect, _ = evaluate_curriculum(
                model, device, self.lesson, field_log=field_log
            )
        else:
            accuracy, perfect, _ = self.dataset.evaluate_id(
                model,
                device,
                self.lesson,
                num_episodes=64,
                rng=random.Random(91_000 + self.lesson),
                field_log=field_log,
            )
        writer = getattr(self, "field_log_writer", None)
        file = getattr(self, "field_log_file", None)
        if writer is not None and file is not None:
            self.dataset.write_field_log(writer, file, step, self.lesson + 1, "id", field_log)
        per_track = (self.dataset.last_eval_diagnostics or {}).get("tracks", {})
        progression_groups = (self.dataset.last_eval_diagnostics or {}).get(
            "progression_groups", {}
        )
        blockers = []
        groups_ready = True
        for track in self.dataset._tracks():
            if track in ("ignition", "nth_farthest"):
                continue
            required = self.dataset._curriculum_groups(track, self.lesson)
            actual = progression_groups.get(track, {})
            for group in required:
                metrics = actual.get(group)
                if metrics is None:
                    blockers.append(f"{track}/{group}=missing")
                elif metrics["episodes"] < self.dataset.curriculum_eval_episodes_per_group:
                    blockers.append(
                        f"{track}/{group}={metrics['exact_match']:.1f}% "
                        f"n={metrics['episodes']} "
                        f"(needs n>={self.dataset.curriculum_eval_episodes_per_group})"
                    )
                elif metrics["exact_match"] / 100.0 < self.dataset.advance_threshold:
                    blockers.append(
                        f"{track}/{group}={metrics['exact_match']:.1f}% "
                        f"n={metrics['episodes']}"
                    )
            if any(
                group not in actual
                or actual[group]["episodes"] < self.dataset.curriculum_eval_episodes_per_group
                or actual[group]["exact_match"] / 100.0 < self.dataset.advance_threshold
                for group in required
            ):
                groups_ready = False
        multi_track_run = len(self.dataset._tracks()) > 1
        all_tracks_ready = not multi_track_run or (
            len(per_track) == len(self.dataset._tracks())
            and all(
                item["exact_match"] / 100.0 >= self.dataset.advance_threshold
                for _t, item in per_track.items() if _t not in ("ignition", "nth_farthest")
            )
        )
        if multi_track_run:
            for track in self.dataset._tracks():
                if track in ("ignition", "nth_farthest"):
                    continue
                metrics = per_track.get(track)
                if metrics is None:
                    blockers.append(f"{track}/overall=missing")
                elif metrics["exact_match"] / 100.0 < self.dataset.advance_threshold:
                    blockers.append(
                        f"{track}/overall={metrics['exact_match']:.1f}% "
                        f"n={metrics['episodes']}"
                    )
        if perfect / 100.0 < self.dataset.advance_threshold:
            blockers.insert(
                0,
                f"overall={perfect:.1f}% "
                f"n={(self.dataset.last_eval_diagnostics or {}).get('n_episodes', 0)}",
            )
        if allow_advance and self.lesson < len(self.table) - 1:
            status = "ready" if not blockers else "blocked"
            print(
                f"[workspace-curriculum] lesson {self.lesson + 1} promotion {status}; "
                f"threshold={self.dataset.advance_threshold * 100:.1f}% exact, "
                f"episodes/group={self.dataset.curriculum_eval_episodes_per_group}"
            )
            for start in range(0, len(blockers), 6):
                print(
                    "[workspace-curriculum] blockers: "
                    + " | ".join(blockers[start : start + 6])
                )
        if (
            allow_advance
            and perfect / 100.0 >= self.dataset.advance_threshold
            and all_tracks_ready
            and groups_ready
            and self.lesson < len(self.table) - 1
        ):
            self.lesson += 1
            writer = getattr(self, "advance_log_writer", None)
            if writer is not None:
                writer.writerow([step, self.lesson + 1, len(self.table)])
                self.advance_log_file.flush()
            print(
                f">>> Workspace-capability curriculum advanced to "
                f"lesson {self.lesson + 1}/{len(self.table)} "
                f"({self.dataset.task_track}; episode exact {perfect:.2f}%)"
            )
        return self.lesson, accuracy, perfect
