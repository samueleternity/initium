"""Surprise-Gated Recall synthetic task and BaseDataset adapter.

Phase 0 deliberately contains no model-side prediction-error machinery. It
defines episodes, labels, simple detector baselines, and the training adapter
needed to exercise the task with the existing model/training loop.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # generator/oracle checks are intentionally dependency-light
    torch = None
    F = None

from initium.data.base_dataset import BaseDataset

PATTERN_VOCAB_SIZE = 16
ORDINARY_VOCAB_SIZE = 16
ALERT_TOKENS = frozenset(range(16, 20))
QUERY_TOKEN = 20
ANSWER_TOKEN = 21
EOS_TOKEN = 22
VOCAB_SIZE = 23
MAX_ANSWER_TOKENS = 7  # six anomalies plus EOS
ADVANCE_THRESHOLD = 0.80
OLD_LESSON_MIX_RATE = 0.10
EVAL_BATCH_SIZE = 128


@dataclass(frozen=True)
class Lesson:
    name: str
    stream_length: int
    anomaly_types: tuple[str, ...]
    anomaly_count: tuple[int, int] = (1, 1)
    gap: int = 2


LESSONS = (
    Lesson("continuation", 18, ()),
    Lesson("marked", 24, ("A1",)),
    Lesson("silent-point", 24, ("A2",)),
    Lesson("silent-plus-decoy", 32, ("A2", "A3"), (1, 3)),
    Lesson("phase-slip", 40, ("A2", "A3", "A4"), (1, 3)),
    Lesson("long-gap", 64, ("A2", "A3", "A4", "A5"), (1, 3), gap=12),
    Lesson("noisy-channel", 64, ("A2", "A3", "A4", "A5", "A6"), (1, 3), gap=12),
)


@dataclass(frozen=True)
class SGRExample:
    inputs: tuple[int, ...]
    targets: tuple[int, ...]
    answer_start: int
    pattern: tuple[int, ...]
    anomaly_positions: tuple[int, ...]
    anomaly_types: tuple[str, ...]
    expected_tokens: tuple[int, ...]
    observed_tokens: tuple[int, ...]
    noise_positions: tuple[int, ...] = ()


def build_episode(
    lesson: Lesson | int,
    rng: random.Random | None = None,
    *,
    period_range: tuple[int, int] = (3, 6),
    stream_multiplier: int = 1,
) -> SGRExample:
    """Create one causal sequence episode with oracle-verified labels."""
    rng = rng or random
    lesson = LESSONS[lesson] if isinstance(lesson, int) else lesson
    period = rng.randint(*period_range)
    pattern = list(rng.sample(range(PATTERN_VOCAB_SIZE), period))
    # A3 requires salient alert tokens to be predictable members of the pattern.
    if "A3" in lesson.anomaly_types:
        pattern[0] = rng.choice(tuple(ALERT_TOKENS))
        pattern = list(dict.fromkeys(pattern))
        while len(pattern) < period:
            candidate = rng.randrange(PATTERN_VOCAB_SIZE)
            if candidate not in pattern:
                pattern.append(candidate)

    length = max(lesson.stream_length * stream_multiplier, 2 * period + 1)
    clean = [pattern[i % period] for i in range(length)]
    stream = clean[:]
    noise_positions: list[int] = []
    if "A6" in lesson.anomaly_types:
        # Jitter affects one randomly selected interleaved channel, but these
        # tokens are intentionally never added to the anomaly labels.
        noisy_parity = rng.randrange(2)
        for j in range(2 * period, length):
            if j % 2 == noisy_parity and rng.random() < 0.30:
                alternatives = [x for x in range(ORDINARY_VOCAB_SIZE) if x != stream[j]]
                stream[j] = rng.choice(alternatives)
                noise_positions.append(j)
    candidates = list(range(2 * period, max(2 * period + 1, length - lesson.gap)))
    rng.shuffle(candidates)
    n_anomalies = rng.randint(*lesson.anomaly_count) if lesson.anomaly_types else 0
    event_types: list[str] = []
    if "A2" in lesson.anomaly_types:
        event_types.extend(["A2"] * n_anomalies)
    else:
        event_types.extend(
            kind for kind in lesson.anomaly_types if kind not in {"A3", "A6"}
        )
    if "A4" in lesson.anomaly_types and n_anomalies >= 2:
        event_types.append("A4")
    if "A5" in lesson.anomaly_types and n_anomalies >= 3:
        event_types.append("A5")
    if "A3" in lesson.anomaly_types:
        event_types.append("A3")
    positions: list[int] = []
    types: list[str] = []
    expected: list[int] = []
    observed: list[int] = []

    # Keep events separated so each anomaly has an interpretable local label.
    for anomaly_type in event_types:
        if not candidates:
            break
        position = next(
            (p for p in candidates if all(abs(p - q) > 1 for q in positions)), None
        )
        if position is None:
            break
        candidates.remove(position)
        expected_token = clean[position]
        if anomaly_type == "A1":
            observed_token = rng.choice(tuple(ALERT_TOKENS - {expected_token}))
        elif anomaly_type == "A3":
            # Decoy alert tokens are ordinary predictable pattern members.
            decoys = [token for token in ALERT_TOKENS if token in pattern]
            if not decoys:
                continue
            position = next((p for p in candidates if clean[p] in decoys), position)
            expected_token = clean[position]
            observed_token = expected_token
        elif anomaly_type == "A4":
            # Shift the phase from this point onward. Only the first position is
            # labelled as the phase-slip event; following tokens remain legal.
            observed_token = clean[(position + 1) % length]
            for j in range(position, length):
                stream[j] = pattern[(j + 1) % period]
        elif anomaly_type == "A5":
            # Change the latent regime after the change point; the change point
            # is the sole event recalled in this Phase 0 task.
            new_pattern = rng.sample(range(PATTERN_VOCAB_SIZE), period)
            while (
                new_pattern == pattern
                or new_pattern[position % period] == expected_token
            ):
                new_pattern = rng.sample(range(PATTERN_VOCAB_SIZE), period)
            observed_token = new_pattern[position % period]
            for j in range(position, length):
                stream[j] = new_pattern[j % period]
        elif anomaly_type == "A6":
            continue
        else:  # A2: silent replacement with a different ordinary token.
            alternatives = [
                token
                for token in pattern
                if token not in ALERT_TOKENS and token != expected_token
            ]
            observed_token = rng.choice(alternatives)
        if anomaly_type != "A6":
            stream[position] = observed_token
            positions.append(position)
            types.append(anomaly_type)
            expected.append(expected_token)
            observed.append(observed_token)

    # A3 entries are distractors, never positive recall targets.
    event_order = sorted(zip(positions, types, expected, observed))
    positions = [item[0] for item in event_order]
    types = [item[1] for item in event_order]
    expected = [item[2] for item in event_order]
    observed = [item[3] for item in event_order]
    regime_change_positions = [
        position for position, kind in zip(positions, types) if kind in {"A4", "A5"}
    ]
    noise_positions = [
        position
        for position in noise_positions
        if position not in positions
        and not any(change_position <= position for change_position in regime_change_positions)
    ]
    answer = [token for token, kind in zip(observed, types) if kind != "A3"]
    if not answer:  # clean continuation lessons predict the next pattern tokens
        answer = [pattern[(length + j) % period] for j in range(3)]
    answer = answer[: MAX_ANSWER_TOKENS - 1] + [EOS_TOKEN]

    inputs = stream + [QUERY_TOKEN] + [ANSWER_TOKEN] * len(answer)
    targets = [0] * (length + 1) + answer
    return SGRExample(
        tuple(inputs),
        tuple(targets),
        length + 1,
        tuple(pattern),
        tuple(positions),
        tuple(types),
        tuple(expected),
        tuple(observed),
        tuple(noise_positions),
    )


def oracle_answer(example: SGRExample) -> tuple[int, ...]:
    """Return the known-correct answer tokens including EOS."""
    if not example.anomaly_positions:
        # The pattern and clean continuation are available to this task oracle.
        period = len(example.pattern)
        stream_length = example.answer_start - 1
        answer = [example.pattern[(stream_length + i) % period] for i in range(3)]
    else:
        answer = [
            token
            for token, kind in zip(example.observed_tokens, example.anomaly_types)
            if kind != "A3"
        ]
    return tuple(answer[: MAX_ANSWER_TOKENS - 1] + [EOS_TOKEN])


def verify_labels(example: SGRExample) -> bool:
    """Check event labels against the episode's known clean generator."""
    if len(example.anomaly_positions) != len(example.anomaly_types):
        return False
    for p, kind, expected, observed in zip(
        example.anomaly_positions,
        example.anomaly_types,
        example.expected_tokens,
        example.observed_tokens,
    ):
        if kind == "A3":
            if expected != observed or observed not in ALERT_TOKENS:
                return False
        elif kind == "A2":
            if expected == observed or expected not in example.pattern:
                return False
            if observed not in example.pattern or observed in ALERT_TOKENS:
                return False
        elif expected == observed:
            return False
        if p < 0 or p >= example.answer_start - 1:
            return False
    return True


class SGRCurriculum:
    def __init__(self, dataset: "SurpriseRecallDataset"):
        self.dataset = dataset
        self.table = list(LESSONS)
        self.lesson = 0
        self.lesson_nr_cells = [128] * len(self.table)

    def maybe_advance(self, model, device, step=None, optimizer=None, *, allow_advance=True):
        accuracy, perfect, _ = self.dataset.evaluate_episodes(
            model, device, EVAL_BATCH_SIZE, lesson_idx=self.lesson, rng=random.Random(31_000 + self.lesson)
        )
        if (
            allow_advance
            and accuracy / 100 >= self.dataset.advance_threshold
            and self.lesson < len(self.table) - 1
        ):
            self.lesson += 1
            writer = getattr(self, "advance_log_writer", None)
            if writer is not None:
                writer.writerow([step, self.lesson + 1, len(self.table)])
                self.advance_log_file.flush()
            print(f">>> Curriculum advanced to lesson {self.lesson + 1}/{len(self.table)}")
        return self.lesson, accuracy, perfect


class SurpriseRecallDataset(BaseDataset):
    name = "surprise-recall"
    input_dim = VOCAB_SIZE
    output_dim = VOCAB_SIZE
    advance_threshold = ADVANCE_THRESHOLD
    old_lesson_mix_rate = OLD_LESSON_MIX_RATE

    def make_curriculum(self):
        return SGRCurriculum(self)

    def build_episode(self, lesson: Lesson | int, rng: random.Random | None = None):
        return build_episode(lesson, rng)

    def sample_batch(self, curriculum, batch_size):
        episodes = []
        for _ in range(batch_size):
            lesson = curriculum.lesson
            if lesson and random.random() < self.old_lesson_mix_rate:
                lesson = random.randrange(lesson)
            episodes.append(build_episode(lesson))
        # Retain CPU-side oracle episodes for the optional SGR diagnostics.
        # The training path does not inspect these unless that logger is enabled.
        self.last_batch_examples = episodes
        self.last_stream_lengths = [episode.answer_start - 1 for episode in episodes]
        max_len = max(len(ep.inputs) for ep in episodes)
        inputs = torch.zeros(batch_size, max_len, self.input_dim)
        targets = torch.zeros(batch_size, max_len, dtype=torch.long)
        mask = torch.zeros(batch_size, max_len)
        for i, ep in enumerate(episodes):
            n = len(ep.inputs)
            inputs[i, torch.arange(n), torch.tensor(ep.inputs)] = 1.0
            targets[i, :n] = torch.tensor(ep.targets)
            mask[i, ep.answer_start:n] = 1.0
        return inputs, targets, mask

    def loss(self, output, target, mask):
        losses = F.cross_entropy(
            output.reshape(-1, self.output_dim),
            target.reshape(-1),
            reduction="none",
        )
        weights = mask.reshape(-1).to(losses.dtype)
        return (losses * weights).sum() / weights.sum().clamp_min(1.0)

    def diversity(self, output, target, mask):
        valid = mask.bool()
        return float(output.argmax(-1)[valid].unique().numel()) if valid.any() else 0.0

    def set_output_proj(self, proj):
        self._output_proj = proj

    def score(self, output: torch.Tensor, example: SGRExample) -> tuple[int, int]:
        pred = output.argmax(dim=-1).tolist()
        expected = list(oracle_answer(example))
        got = pred[example.answer_start : example.answer_start + len(expected)]
        correct = sum(a == b for a, b in zip(got, expected))
        return correct, len(expected)

    def evaluate_episodes(
        self,
        model,
        device,
        num_episodes,
        lesson_idx=0,
        rng=None,
        episode_builder=None,
        ablate_memory=False,
    ):
        rng = rng or random
        model.eval()
        correct = total = perfect = 0
        by_type = {
            kind: {"count": 0, "correct": 0, "episodes": 0, "exact_episodes": 0}
            for kind in ("A1", "A2", "A3", "A4", "A5", "A6")
        }
        event_counts = {kind: 0 for kind in ("A1", "A2", "A3", "A4", "A5", "A6")}
        event_episode_counts = {kind: 0 for kind in event_counts}
        event_detector_counts = {
            kind: {"count": 0, "alert_flagged": 0, "pattern_mismatch_flagged": 0}
            for kind in event_counts
        }
        detector = {"alert": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
                    "pattern_mismatch": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}}
        event_positive = {"A1", "A2", "A4", "A5"}
        with torch.no_grad():
            for _ in range(num_episodes):
                ep = (episode_builder or build_episode)(lesson_idx, rng)
                for kind in ep.anomaly_types:
                    event_counts[kind] += 1
                for kind in set(ep.anomaly_types):
                    event_episode_counts[kind] += 1
                event_counts["A6"] += len(ep.noise_positions)
                event_episode_counts["A6"] += int(bool(ep.noise_positions))
                x = torch.zeros(1, len(ep.inputs), self.input_dim, device=device)
                x[
                    0,
                    torch.arange(len(ep.inputs), device=device),
                    torch.tensor(ep.inputs, device=device),
                ] = 1
                output, _ = model(
                    x,
                    (None, None, None),
                    reset_experience=True,
                    pass_through_memory=not ablate_memory,
                )
                logits = self._output_proj(output.transpose(0, 1).contiguous().squeeze(0))
                n_ok, n = self.score(logits, ep)
                correct += n_ok
                total += n
                perfect += int(n_ok == n)
                predicted = logits.argmax(dim=-1).tolist()
                expected_answer = oracle_answer(ep)
                prediction = predicted[ep.answer_start : ep.answer_start + len(expected_answer)]
                recalled = {kind for kind in ep.anomaly_types if kind != "A3"}
                for kind in recalled:
                    by_type[kind]["episodes"] += 1
                # Answer events are emitted in sorted stream order; EOS and
                # clean-continuation tokens are excluded from anomaly recall.
                answer_events = [
                    (kind, token) for kind, token in zip(ep.anomaly_types, ep.observed_tokens)
                    if kind != "A3"
                ]
                episode_type_correct = {kind: True for kind, _ in answer_events}
                for event_index, (kind, token) in enumerate(answer_events):
                    is_correct = event_index < len(prediction) and prediction[event_index] == token
                    by_type[kind]["count"] += 1
                    by_type[kind]["correct"] += int(is_correct)
                    episode_type_correct[kind] &= is_correct
                for kind, is_correct in episode_type_correct.items():
                    by_type[kind]["exact_episodes"] += int(is_correct)
                event_map = dict(zip(ep.anomaly_positions, ep.anomaly_types))
                jitter_positions = set(ep.noise_positions)
                for position, token in enumerate(ep.inputs[: ep.answer_start - 1]):
                    kind = event_map.get(position)
                    truth = kind in event_positive
                    if kind is None and position in jitter_positions:
                        kind = "A6"
                    if kind is not None:
                        event_detector_counts[kind]["count"] += 1
                        event_detector_counts[kind]["alert_flagged"] += int(token in ALERT_TOKENS)
                        event_detector_counts[kind]["pattern_mismatch_flagged"] += int(
                            token != ep.pattern[position % len(ep.pattern)]
                        )
                    for detector_name, detected in (
                        ("alert", token in ALERT_TOKENS),
                        ("pattern_mismatch", token != ep.pattern[position % len(ep.pattern)]),
                    ):
                        bucket = "tp" if truth and detected else "fn" if truth else "fp" if detected else "tn"
                        detector[detector_name][bucket] += 1
                    if position in jitter_positions:
                        # Track benign jitter explicitly: it is a key negative
                        # control for pattern-mismatch detectors.
                        detector.setdefault("a6_jitter", {"count": 0, "pattern_mismatch": 0, "alert": 0})
                        detector["a6_jitter"]["count"] += 1
                        detector["a6_jitter"]["pattern_mismatch"] += int(
                            token != ep.pattern[position % len(ep.pattern)]
                        )
                        detector["a6_jitter"]["alert"] += int(token in ALERT_TOKENS)
        model.train()
        task_accuracy = 100.0 * correct / max(total, 1)
        self.last_eval_diagnostics = {
            "task": {
                "count": total,
                "correct": correct,
                "accuracy_pct": task_accuracy,
                "exact_episode_pct": 100.0 * perfect / max(num_episodes, 1),
            },
            "by_type": {
                kind: {
                    **values,
                    "accuracy_pct": 100.0 * values["correct"] / max(values["count"], 1),
                    "exact_episode_pct": 100.0 * values["exact_episodes"] / max(values["episodes"], 1),
                }
                for kind, values in by_type.items()
            },
            "event_counts": event_counts,
            "event_episode_counts": event_episode_counts,
            "event_detector_counts": event_detector_counts,
            "content_detector": {
                **detector["alert"],
                "accuracy_pct": 100.0 * (detector["alert"]["tp"] + detector["alert"]["tn"])
                / max(sum(detector["alert"].values()), 1),
            },
            "detectors": detector,
        }
        # Preserve BaseDataset's third-return contract (task-specific breakdown,
        # e.g. hop counts); SGR diagnostics are exposed separately as state.
        return task_accuracy, 100.0 * perfect / max(num_episodes, 1), {}

    def evaluate_ood(self, model, device, num_episodes, rng, verbose_n=0, field_log=None):
        suite_index = 0

        def make_ood(lesson_idx, episode_rng):
            nonlocal suite_index
            suite = suite_index % 5
            suite_index += 1
            if suite == 0:  # four times the longest training stream
                lesson = LESSONS[5]
                return build_episode(lesson, episode_rng, stream_multiplier=4)
            if suite == 1:  # unseen pattern periods
                return build_episode(LESSONS[3], episode_rng, period_range=(7, 8))
            if suite == 2:  # anomaly count beyond the curriculum, capped at six
                lesson = Lesson("ood-more-anomalies", 96, ("A2",) * 6, (6, 6), gap=12)
                return build_episode(lesson, episode_rng)
            if suite == 3:  # vocabulary remap; token identities carry no label meaning
                ep = build_episode(LESSONS[3], episode_rng)
                def remap(token):
                    return (7 * token + 3) % PATTERN_VOCAB_SIZE

                inputs = tuple(remap(x) if x < PATTERN_VOCAB_SIZE else x for x in ep.inputs)
                targets = tuple(remap(x) if x < PATTERN_VOCAB_SIZE else x for x in ep.targets)
                return SGRExample(
                    inputs, targets, ep.answer_start,
                    tuple(remap(x) for x in ep.pattern), ep.anomaly_positions,
                    ep.anomaly_types, tuple(remap(x) for x in ep.expected_tokens),
                    tuple(remap(x) for x in ep.observed_tokens), ep.noise_positions,
                )
            # A novel interaction: predictable alert decoy followed by a phase slip.
            lesson = Lesson("ood-slip-plus-decoy", 80, ("A3", "A4"), (2, 2), gap=12)
            return build_episode(lesson, episode_rng)

        return self.evaluate_episodes(
            model, device, num_episodes, lesson_idx=5, rng=rng, episode_builder=make_ood
        )

    def evaluate_id_ablated(self, model, device, curriculum, lesson_idx):
        return self.evaluate_episodes(
            model,
            device,
            EVAL_BATCH_SIZE,
            lesson_idx,
            random.Random(55_000 + lesson_idx),
            ablate_memory=True,
        )[:2]

    def evaluate_id_combiner_stage_ablated(
        self, model, device, curriculum, lesson_idx, skip_stages
    ):
        raise NotImplementedError("SGR Phase 0 does not define combiner-stage ablations")

    def field_log_header(self):
        return ["anomaly_type", "n", "token_accuracy"]

    def write_field_log(self, writer, file, step, lesson, eval_type, field_log):
        for anomaly_type, (n, correct) in sorted(field_log.items()):
            writer.writerow([step, lesson, eval_type, anomaly_type, n, correct / max(n, 1)])
        file.flush()

