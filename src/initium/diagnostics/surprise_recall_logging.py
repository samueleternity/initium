"""Low-overhead, SGR-only run diagnostics.

The logger writes one sampled training batch at the normal training-log
cadence, plus token-level truth traces for that batch and per-event evaluation
rows. It is constructed only for the surprise-recall dataset.
"""

from __future__ import annotations

import csv
import os
from collections import Counter

from initium.data.surprise_recall.surprise_recall import (
    ALERT_TOKENS,
    SGRExample,
)

ANOMALY_TYPES = ("A1", "A2", "A3", "A4", "A5", "A6")


class SurpriseRecallLogger:
    """Writes compact training summaries, stream labels, and eval diagnostics."""

    def __init__(
        self, log_dir: str, run_id: str, append: bool = False, *, phase1_enabled: bool = False
    ):
        mode = "a" if append else "w"
        self.run_id = run_id
        self._files = []
        prefix = os.path.join(log_dir, f"run_{run_id}_sgr")

        self.train_file = open(f"{prefix}_train.csv", mode, newline="", encoding="utf-8")
        self.train_writer = csv.writer(self.train_file)
        self._files.append(self.train_file)
        if os.path.getsize(f"{prefix}_train.csv") == 0:
            self.train_writer.writerow(
                [
                    "step",
                    "lesson",
                    "episodes",
                    "mean_period",
                    "mean_stream_tokens",
                    "mean_answer_tokens",
                    "task_loss_window_mean",
                    "prediction_loss_window_mean",
                    "mean_event_gap_to_query",
                    "a2_same_pattern_fraction",
                    "a2_alert_token_fraction",
                    "a3_alert_fraction",
                    "a6_jitter_positions",
                    *[f"episodes_{kind}" for kind in ANOMALY_TYPES],
                    *[f"events_{kind}" for kind in ANOMALY_TYPES],
                ]
            )

        self.position_file = open(f"{prefix}_positions.csv", mode, newline="", encoding="utf-8")
        self.position_writer = csv.writer(self.position_file)
        self._files.append(self.position_file)
        if os.path.getsize(f"{prefix}_positions.csv") == 0:
            self.position_writer.writerow(
                [
                    "step",
                    "lesson",
                    "batch_episode",
                    "position",
                    "token_id",
                    "pattern_expected_token_id",
                    "event_type",
                    "is_recalled_anomaly",
                    "is_a3_decoy",
                    "is_a6_benign_jitter",
                    "alert_rule_flag",
                    "tokens_until_query",
                ]
            )

        self.eval_file = open(f"{prefix}_eval.csv", mode, newline="", encoding="utf-8")
        self.eval_writer = csv.writer(self.eval_file)
        self._files.append(self.eval_file)
        if os.path.getsize(f"{prefix}_eval.csv") == 0:
            self.eval_writer.writerow(
                [
                    "step",
                    "lesson",
                    "eval_type",
                    "metric",
                    "event_type",
                    "count",
                    "correct",
                    "accuracy_pct",
                    "exact_episode_pct",
                    "detector_tp",
                    "detector_fp",
                    "detector_fn",
                    "detector_tn",
                ]
            )
        self.phase1_file = None
        self.phase1_writer = None
        if phase1_enabled:
            self.phase1_file = open(f"{prefix}_phase1.csv", mode, newline="", encoding="utf-8")
            self.phase1_writer = csv.writer(self.phase1_file)
            self._files.append(self.phase1_file)
            if os.path.getsize(f"{prefix}_phase1.csv") == 0:
                self.phase1_writer.writerow(
                    [
                        "step",
                        "eval_type",
                        "lesson_scope",
                        "metric",
                        "event_type",
                        "count",
                        "raw_auroc",
                        "excess_auroc",
                        "mean_raw_surprisal",
                        "mean_excess_surprise",
                        "value",
                        "notes",
                    ]
                )

    def write_training_batch(
        self,
        step: int,
        lesson: int,
        examples: list[SGRExample],
        *,
        task_loss: float | None = None,
        prediction_loss: float | None = None,
    ) -> None:
        if not examples:
            return
        event_counts: Counter[str] = Counter()
        a2_same_pattern = a2_total = a2_alert = a3_total = a3_alert = jitter_total = 0
        periods = [len(example.pattern) for example in examples]
        stream_lengths = [example.answer_start - 1 for example in examples]
        answer_lengths = [len(example.targets) - example.answer_start for example in examples]
        event_gaps = []
        for batch_episode, example in enumerate(examples):
            event_map = dict(zip(example.anomaly_positions, example.anomaly_types))
            event_counts.update(example.anomaly_types)
            jitter_total += len(example.noise_positions)
            event_gaps.extend(
                (example.answer_start - 1) - position
                for position, kind in zip(example.anomaly_positions, example.anomaly_types)
                if kind != "A3"
            )
            for kind, expected, observed in zip(
                example.anomaly_types, example.expected_tokens, example.observed_tokens
            ):
                if kind == "A2":
                    a2_total += 1
                    a2_same_pattern += int(
                        expected in example.pattern and observed in example.pattern
                    )
                    a2_alert += int(observed in ALERT_TOKENS)
                elif kind == "A3":
                    a3_total += 1
                    a3_alert += int(observed in ALERT_TOKENS)

            jitter_positions = set(example.noise_positions)
            for position, token in enumerate(example.inputs[: example.answer_start - 1]):
                kind = event_map.get(
                    position, "A6" if position in example.noise_positions else "normal"
                )
                self.position_writer.writerow(
                    [
                        step,
                        lesson,
                        batch_episode,
                        position,
                        token,
                        example.pattern[position % len(example.pattern)],
                        kind,
                        int(kind in {"A1", "A2", "A4", "A5"}),
                        int(kind == "A3"),
                        int(position in jitter_positions),
                        int(token in ALERT_TOKENS),
                        (example.answer_start - 1) - position,
                    ]
                )

        episode_type_counts = [
            sum(
                (kind in example.anomaly_types) or (kind == "A6" and bool(example.noise_positions))
                for example in examples
            )
            for kind in ANOMALY_TYPES
        ]
        event_type_counts = [
            len(example.noise_positions) if kind == "A6" else event_counts[kind]
            for kind in ANOMALY_TYPES
        ]
        self.train_writer.writerow(
            [
                step,
                lesson,
                len(examples),
                sum(periods) / len(periods),
                sum(stream_lengths) / len(stream_lengths),
                sum(answer_lengths) / len(answer_lengths),
                task_loss,
                prediction_loss,
                sum(event_gaps) / len(event_gaps) if event_gaps else None,
                a2_same_pattern / a2_total if a2_total else None,
                a2_alert / a2_total if a2_total else None,
                a3_alert / a3_total if a3_total else None,
                jitter_total,
                *episode_type_counts,
                *event_type_counts,
            ]
        )
        fmt = self._format_number
        event_summary = ",".join(
            f"{kind}:{episode_type_counts[index]}ep/{event_type_counts[index]}evt"
            for index, kind in enumerate(ANOMALY_TYPES)
        )
        print(
            f"[{self.run_id}] [SGR train] step={step} lesson={lesson} episodes={len(examples)} "
            f"period={fmt(sum(periods) / len(periods), 2)} "
            f"stream_tokens={fmt(sum(stream_lengths) / len(stream_lengths), 2)} "
            f"answer_tokens={fmt(sum(answer_lengths) / len(answer_lengths), 2)} "
            f"task_loss={fmt(task_loss, 5)} prediction_loss={fmt(prediction_loss, 5)} "
            f"event_gap_to_query={fmt(sum(event_gaps) / len(event_gaps) if event_gaps else None, 2)} "
            f"A2_same_pattern={fmt(a2_same_pattern / a2_total if a2_total else None, 3)} "
            f"A2_alert={fmt(a2_alert / a2_total if a2_total else None, 3)} "
            f"A3_alert={fmt(a3_alert / a3_total if a3_total else None, 3)} "
            f"A6_jitter_positions={jitter_total} | {event_summary}",
            flush=True,
        )
        # Mirror the sampled position CSV in a compact, episode-grouped
        # console trace. One row per token is too noisy for a live terminal;
        # this preserves every position field while keeping one line per run
        # logging interval.
        position_trace = []
        for batch_episode, example in enumerate(examples):
            event_map = dict(zip(example.anomaly_positions, example.anomaly_types))
            jitter_positions = set(example.noise_positions)
            position_descriptors = []
            for position, token in enumerate(example.inputs[: example.answer_start - 1]):
                kind = event_map.get(position, "A6" if position in jitter_positions else "normal")
                pattern_token = example.pattern[position % len(example.pattern)]
                position_descriptors.append(
                    f"p{position}:t{token}/pat{pattern_token}/{kind}/"
                    f"r{int(kind in ('A1', 'A2', 'A4', 'A5'))}"
                    f"d{int(kind == 'A3')}j{int(position in jitter_positions)}"
                    f"a{int(token in ALERT_TOKENS)}"
                    f"q{(example.answer_start - 1) - position}"
                )
            position_trace.append(f"ep{batch_episode}[" + ";".join(position_descriptors) + "]")
        print(
            f"[{self.run_id}] [SGR train positions] step={step} " + " | ".join(position_trace),
            flush=True,
        )
        self._flush()

    def write_phase1_evaluation(self, step: int, eval_type: str, summary: dict | None) -> None:
        if not summary or self.phase1_writer is None:
            return
        lesson_scope = ",".join(str(index + 1) for index in summary.get("lessons", []))
        for kind, values in summary.get("per_type", {}).items():
            self.phase1_writer.writerow(
                [
                    step,
                    eval_type,
                    lesson_scope,
                    "type_vs_clean_normal_auroc",
                    kind,
                    values.get("count", 0),
                    values.get("raw_auroc_vs_clean_normal"),
                    values.get("excess_auroc_vs_clean_normal"),
                    values.get("mean_raw_surprisal"),
                    values.get("mean_excess_surprise"),
                    None,
                    None,
                ]
            )
        a6 = summary.get("a6_vs_true_anomalies", {})
        self.phase1_writer.writerow(
            [
                step,
                eval_type,
                lesson_scope,
                "true_anomaly_vs_a6_jitter_auroc",
                "A1+A2+A4+A5 vs A6",
                a6.get("a6_jitter_count", 0) + a6.get("true_anomaly_count", 0),
                a6.get("raw_auroc"),
                a6.get("excess_auroc"),
                a6.get("mean_true_anomaly_raw_surprisal"),
                a6.get("mean_true_anomaly_excess_surprise"),
                a6.get("mean_a6_raw_surprisal"),
                f"A6 mean excess={a6.get('mean_a6_excess_surprise')}",
            ]
        )
        calibration = summary.get("calibration", {})
        self.phase1_writer.writerow(
            [
                step,
                eval_type,
                lesson_scope,
                "next_token_calibration",
                "ALL",
                calibration.get("count", 0),
                None,
                None,
                None,
                None,
                calibration.get("ece"),
                f"accuracy_pct={calibration.get('accuracy_pct')}; brier={calibration.get('brier')}",
            ]
        )
        recovery = summary.get("a5_recovery", {})
        self.phase1_writer.writerow(
            [
                step,
                eval_type,
                lesson_scope,
                "a5_recovery_tokens",
                "A5",
                recovery.get("recovered_episodes", 0) + recovery.get("censored_episodes", 0),
                None,
                None,
                None,
                None,
                recovery.get("mean_tokens_to_recovery"),
                f"recovered={recovery.get('recovered_episodes')}; "
                f"censored={recovery.get('censored_episodes')}; {recovery.get('window_rule')}",
            ]
        )
        fmt = self._format_number
        lesson_scope_text = lesson_scope or "none"
        print(
            f"[{self.run_id}] [SGR phase1] step={step} eval={eval_type} "
            f"lessons={lesson_scope_text} episodes_per_lesson={summary.get('episodes_per_lesson')} "
            f"normal_positions={summary.get('normal_positions', 0)} "
            f"mean_raw_surprisal={fmt(summary.get('mean_raw_surprisal'), 5)} "
            f"mean_predictive_entropy={fmt(summary.get('mean_predictive_entropy'), 5)}",
            flush=True,
        )
        for kind in ANOMALY_TYPES:
            values = summary.get("per_type", {}).get(kind, {})
            print(
                f"[{self.run_id}] [SGR phase1 type] step={step} eval={eval_type} {kind} "
                f"n={values.get('count', 0)} "
                f"raw_AUROC={fmt(values.get('raw_auroc_vs_clean_normal'), 4)} "
                f"excess_AUROC={fmt(values.get('excess_auroc_vs_clean_normal'), 4)} "
                f"mean_raw={fmt(values.get('mean_raw_surprisal'), 5)} "
                f"mean_excess={fmt(values.get('mean_excess_surprise'), 5)}",
                flush=True,
            )
        a6_counts = summary.get("a6_vs_true_anomalies", {})
        print(
            f"[{self.run_id}] [SGR phase1 A6-control] step={step} eval={eval_type} "
            f"true_anomaly_n={a6_counts.get('true_anomaly_count', 0)} "
            f"jitter_n={a6_counts.get('a6_jitter_count', 0)} "
            f"raw_AUROC={fmt(a6_counts.get('raw_auroc'), 4)} "
            f"excess_AUROC={fmt(a6_counts.get('excess_auroc'), 4)} "
            f"true_mean_raw={fmt(a6_counts.get('mean_true_anomaly_raw_surprisal'), 5)} "
            f"true_mean_excess={fmt(a6_counts.get('mean_true_anomaly_excess_surprise'), 5)} "
            f"A6_mean_raw={fmt(a6_counts.get('mean_a6_raw_surprisal'), 5)} "
            f"A6_mean_excess={fmt(a6_counts.get('mean_a6_excess_surprise'), 5)}",
            flush=True,
        )
        calibration = summary.get("calibration", {})
        print(
            f"[{self.run_id}] [SGR phase1 calibration] step={step} eval={eval_type} "
            f"n={calibration.get('count', 0)} ECE={fmt(calibration.get('ece'), 5)} "
            f"accuracy={fmt(calibration.get('accuracy_pct'), 3)}% "
            f"Brier={fmt(calibration.get('brier'), 5)}",
            flush=True,
        )
        print(
            f"[{self.run_id}] [SGR phase1 A5-recovery] step={step} eval={eval_type} "
            f"recovered={recovery.get('recovered_episodes', 0)} "
            f"censored={recovery.get('censored_episodes', 0)} "
            f"mean_tokens={fmt(recovery.get('mean_tokens_to_recovery'), 3)} "
            f"rule={recovery.get('window_rule', 'n/a')}",
            flush=True,
        )
        self._flush()

    def write_evaluation(
        self, step: int, lesson: int, eval_type: str, summary: dict | None
    ) -> None:
        if not summary:
            return
        detector = summary.get("content_detector", {})
        common = [
            detector.get("tp", 0),
            detector.get("fp", 0),
            detector.get("fn", 0),
            detector.get("tn", 0),
        ]
        total = summary.get("task", {})
        self.eval_writer.writerow(
            [
                step,
                lesson,
                eval_type,
                "task_token_accuracy",
                "ALL",
                total.get("count", 0),
                total.get("correct", 0),
                total.get("accuracy_pct", 0.0),
                total.get("exact_episode_pct", 0.0),
                *common,
            ]
        )
        for kind, values in summary.get("by_type", {}).items():
            if not values.get("count", 0):
                continue
            self.eval_writer.writerow(
                [
                    step,
                    lesson,
                    eval_type,
                    "recalled_event_accuracy",
                    kind,
                    values.get("count", 0),
                    values.get("correct", 0),
                    values.get("accuracy_pct", 0.0),
                    values.get("exact_episode_pct"),
                    *common,
                ]
            )
        for kind, count in summary.get("event_counts", {}).items():
            self.eval_writer.writerow(
                [
                    step,
                    lesson,
                    eval_type,
                    "event_prevalence",
                    kind,
                    count,
                    None,
                    None,
                    None,
                    *common,
                ]
            )
        for kind, count in summary.get("event_episode_counts", {}).items():
            self.eval_writer.writerow(
                [
                    step,
                    lesson,
                    eval_type,
                    "episodes_with_event_type",
                    kind,
                    count,
                    None,
                    None,
                    None,
                    *common,
                ]
            )
        for kind, values in summary.get("event_detector_counts", {}).items():
            count = values.get("count", 0)
            for detector_key, flagged in (
                ("alert", values.get("alert_flagged", 0)),
                ("pattern_mismatch", values.get("pattern_mismatch_flagged", 0)),
            ):
                self.eval_writer.writerow(
                    [
                        step,
                        lesson,
                        eval_type,
                        f"{detector_key}_flag_rate_at_event",
                        kind,
                        count,
                        flagged,
                        100.0 * flagged / max(count, 1),
                        None,
                        None,
                        None,
                        None,
                        None,
                    ]
                )
        for detector_name, values in summary.get("detectors", {}).items():
            if detector_name == "a6_jitter":
                for detector_key in ("pattern_mismatch", "alert"):
                    false_alarms = values.get(detector_key, 0)
                    count = values.get("count", 0)
                    self.eval_writer.writerow(
                        [
                            step,
                            lesson,
                            eval_type,
                            f"a6_{detector_key}_false_alarm_rate",
                            "A6",
                            count,
                            false_alarms,
                            100.0 * false_alarms / max(count, 1),
                            None,
                            None,
                            None,
                            None,
                            None,
                        ]
                    )
                continue
            self.eval_writer.writerow(
                [
                    step,
                    lesson,
                    eval_type,
                    f"{detector_name}_confusion",
                    "ALL",
                    sum(values.values()),
                    values.get("tp", 0),
                    100.0
                    * (values.get("tp", 0) + values.get("tn", 0))
                    / max(sum(values.values()), 1),
                    None,
                    values.get("tp", 0),
                    values.get("fp", 0),
                    values.get("fn", 0),
                    values.get("tn", 0),
                ]
            )
        self.eval_writer.writerow(
            [
                step,
                lesson,
                eval_type,
                "content_alert_detector",
                "ALL",
                sum(common),
                detector.get("tp", 0),
                detector.get("accuracy_pct", 0.0),
                None,
                *common,
            ]
        )
        fmt = self._format_number
        print(
            f"[{self.run_id}] [SGR eval] step={step} lesson={lesson} eval={eval_type} "
            f"task_accuracy={fmt(total.get('accuracy_pct'), 3)}% "
            f"exact_episode={fmt(total.get('exact_episode_pct'), 3)}% "
            f"tokens={total.get('correct', 0)}/{total.get('count', 0)} "
            f"alert_detector[tp={common[0]} fp={common[1]} fn={common[2]} tn={common[3]} "
            f"accuracy={fmt(detector.get('accuracy_pct'), 3)}%]",
            flush=True,
        )
        recall_parts = []
        for kind in ANOMALY_TYPES:
            values = summary.get("by_type", {}).get(kind, {})
            if values.get("count", 0):
                recall_parts.append(
                    f"{kind}={values.get('correct', 0)}/{values.get('count', 0)}"
                    f"({fmt(values.get('accuracy_pct'), 2)}%;"
                    f"exact_ep={fmt(values.get('exact_episode_pct'), 2)}%)"
                )
            else:
                recall_parts.append(f"{kind}=n/a(0)")
        event_parts = [
            f"{kind}:{summary.get('event_counts', {}).get(kind, 0)}events/"
            f"{summary.get('event_episode_counts', {}).get(kind, 0)}episodes"
            for kind in ANOMALY_TYPES
        ]
        print(
            f"[{self.run_id}] [SGR eval recall] step={step} lesson={lesson} eval={eval_type} "
            + " ".join(recall_parts)
            + " | prevalence "
            + " ".join(event_parts),
            flush=True,
        )
        event_detector_parts = []
        for kind in ANOMALY_TYPES:
            values = summary.get("event_detector_counts", {}).get(kind, {})
            count = values.get("count", 0)
            alert_count = values.get("alert_flagged", 0)
            mismatch_count = values.get("pattern_mismatch_flagged", 0)
            event_detector_parts.append(
                f"{kind}[alert={alert_count}/{count}({fmt(100.0 * alert_count / max(count, 1), 1)}%) "
                f"mismatch={mismatch_count}/{count}"
                f"({fmt(100.0 * mismatch_count / max(count, 1), 1)}%)]"
            )
        detector_parts = []
        for name, values in summary.get("detectors", {}).items():
            if name == "a6_jitter":
                n = values.get("count", 0)
                detector_parts.append(
                    f"A6_false_alarms[n={n} "
                    f"mismatch={values.get('pattern_mismatch', 0)}/"
                    f"{n}({fmt(100.0 * values.get('pattern_mismatch', 0) / max(n, 1), 1)}%) "
                    f"alert={values.get('alert', 0)}/{n}"
                    f"({fmt(100.0 * values.get('alert', 0) / max(n, 1), 1)}%)]"
                )
            else:
                detector_total = sum(values.values())
                detector_accuracy = (
                    100.0 * (values.get("tp", 0) + values.get("tn", 0)) / max(detector_total, 1)
                )
                detector_parts.append(
                    f"{name}[tp={values.get('tp', 0)} fp={values.get('fp', 0)} "
                    f"fn={values.get('fn', 0)} tn={values.get('tn', 0)} "
                    f"accuracy={fmt(detector_accuracy, 2)}%]"
                )
        print(
            f"[{self.run_id}] [SGR eval event-detectors] step={step} lesson={lesson} "
            f"eval={eval_type} " + " ".join(event_detector_parts + detector_parts),
            flush=True,
        )
        self._flush()

    @staticmethod
    def _format_number(value, precision: int = 4) -> str:
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.{precision}f}"
        except (TypeError, ValueError):
            return "n/a"

    def _flush(self) -> None:
        for file in self._files:
            file.flush()

    def close(self) -> None:
        for file in self._files:
            file.close()
