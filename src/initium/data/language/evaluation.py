"""Capability-separated metrics and explicit scale-readiness gates."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path


CAPABILITY_SUITES = (
    "language", "meaning", "world_model", "memory", "conversation", "generalization",
    "data_integrity",
)


@dataclass(frozen=True)
class ScaleGateReport:
    repeatable_resume: bool
    heldout_language_non_regression: bool
    jepa_beats_matched_control: bool
    transfer_to_unseen_combinations: bool
    runtime_training_policy_match: bool
    cost_curves_measured: bool

    @property
    def ready(self):
        return all(asdict(self).values())

    def save(self, path: str | Path):
        payload = asdict(self) | {"ready": self.ready}
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def validate_suite_metrics(metrics: dict):
    missing = sorted(set(CAPABILITY_SUITES) - set(metrics))
    if missing:
        raise ValueError(f"evaluation report is missing capability suites: {missing}")
    return {name: metrics[name] for name in CAPABILITY_SUITES}


def contamination_report(document_ids: list[str], heldout_ids: list[str]):
    train = set(document_ids)
    heldout = set(heldout_ids)
    overlap = sorted(train & heldout)
    return {"train_documents": len(train), "heldout_documents": len(heldout),
            "overlap_count": len(overlap), "clean": not overlap, "overlap_ids": overlap[:100]}
