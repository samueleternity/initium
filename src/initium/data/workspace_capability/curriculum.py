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
) -> dict:
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
            "queries": (1, 1, 2, 3, 4, 8)[level],
            "aggregate": ("recall", "recall", "sum_mod16", "count_even", "parity", "recall")[level],
        }
    else:
        raise ValueError(f"unknown workspace capability track {track!r}")
    if ood and track != "permutation":
        # OOD increases temporal interference while keeping each generator's
        # schema bounds valid and uses its held-out identifier band where able.
        if track == "entity_state":
            config["updates"] = min(config["updates"] + 8, 32)
            config["distractors"] = min(config["distractors"] * 2 + 2, 24)
        elif track == "relational_join":
            config["distractors"] = min(config["distractors"] * 2 + 4, 64)
        elif track == "nth_farthest":
            config["distractors"] = min(config["distractors"] * 2 + 2, 24)
        else:
            config["updates"] = min(config["updates"] + 8, 24)
            config["distractors"] = min(config["distractors"] * 2 + 4, 64)
    if track == "permutation" and "length" in axes:
        config["swaps"] = min(config["swaps"] * 2, 512)
        config["distractors"] = min(config["distractors"] * 2 + 4, 64)
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
            self.dataset.write_field_log(
                writer, file, step, self.lesson + 1, "id", field_log
            )
        per_track = (self.dataset.last_eval_diagnostics or {}).get("tracks", {})
        all_tracks_ready = (
            self.dataset.task_track != "all"
            or (len(per_track) == len(self.dataset._tracks())
                and all(item["accuracy"] / 100.0 >= self.dataset.advance_threshold
                        for item in per_track.values()))
        )
        if (
            allow_advance
            and accuracy / 100.0 >= self.dataset.advance_threshold
            and all_tracks_ready
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
                f"({self.dataset.task_track}; answer accuracy {accuracy:.2f}%)"
            )
        return self.lesson, accuracy, perfect
