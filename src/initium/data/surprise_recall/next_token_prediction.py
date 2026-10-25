"""Phase 1 next-token prediction head and offline signal evaluation."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from initium.data.surprise_recall.surprise_recall import (
    LESSONS,
    VOCAB_SIZE,
    build_episode,
)

TRUE_ANOMALIES = ("A1", "A2", "A4", "A5")
EVENT_TYPES = ("A1", "A2", "A3", "A4", "A5", "A6")


class SurprisePredictionHead(nn.Module):
    """Small token decoder over causal per-token backbone representations."""

    def __init__(self, hidden_size: int, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.projection = nn.Linear(hidden_size, vocab_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Decoder inputs are block-0 residual features and can be very large
        # even when the final backbone output is normalized.  Casting the
        # projection result to float32 at the call site is too late if CUDA
        # autocast already overflowed its fp16 output.
        with torch.autocast(device_type=features.device.type, enabled=False):
            return F.linear(
                features.float(),
                self.projection.weight.float(),
                (self.projection.bias.float() if self.projection.bias is not None else None),
            )


def prediction_error_signal(
    features: torch.Tensor,
    input_one_hot: torch.Tensor,
    decoder: SurprisePredictionHead,
    *,
    signal_mode: str,
    stream_lengths: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached, bounded per-token error and next-token logits.

    Signal at position ``t`` is computed only from the block-0 state at
    ``t-1`` and observed token ``x_t``. Query, answer, and padding positions
    are zeroed so consumers cannot use target-side information.
    """
    if signal_mode not in {"surprisal", "excess-surprise"}:
        raise ValueError("signal_mode must be surprisal or excess-surprise")
    if features.ndim != 3 or input_one_hot.ndim != 3:
        raise ValueError("features and input_one_hot must be rank-three tensors")
    if features.shape[:2] != input_one_hot.shape[:2]:
        raise ValueError("features and input sequence must have matching batch/time dimensions")
    logits = decoder(features[:, :-1]).float()
    log_probs = F.log_softmax(logits, dim=-1)
    targets = input_one_hot[:, 1:].argmax(dim=-1)
    raw = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    if signal_mode == "excess-surprise":
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        scores = raw - entropy
    else:
        scores = raw

    batch, sequence = input_one_hot.shape[:2]
    signal = torch.zeros(batch, sequence, dtype=torch.float32, device=input_one_hot.device)
    signal[:, 1:] = torch.tanh(scores.detach())
    token_ids = input_one_hot.argmax(dim=-1)
    if stream_lengths is None:
        positions = torch.arange(sequence, device=input_one_hot.device).unsqueeze(0)
        query_positions = torch.where(token_ids.eq(20), positions, sequence)
        lengths = query_positions.amin(dim=1)
    else:
        if len(stream_lengths) != batch:
            raise ValueError("stream_lengths must contain one value per batch row")
        lengths = torch.as_tensor(stream_lengths, device=input_one_hot.device)
    stream_mask = torch.arange(sequence, device=input_one_hot.device).unsqueeze(
        0
    ) < lengths.unsqueeze(1)
    # Multiplication does not safely mask NaNs: NaN * 0 remains NaN. Keep
    # excluded query/answer/padding positions exactly zero.
    signal = signal.masked_fill(~stream_mask, 0.0)
    return signal, logits


def masked_next_token_loss(
    logits: torch.Tensor,
    input_one_hot: torch.Tensor,
    stream_lengths: list[int],
) -> torch.Tensor:
    """Mean CE predicting x[t+1] from the backbone state at t.

    The stream-only mask excludes the first token (no prior state), padding,
    query, and answer positions. ``logits`` has shape (B, T-1, vocab).
    """
    if logits.ndim != 3 or input_one_hot.ndim != 3:
        raise ValueError("prediction logits and inputs must be rank-three tensors")
    batch_size, steps, _ = logits.shape
    if input_one_hot.shape[0] != batch_size or input_one_hot.shape[1] != steps + 1:
        raise ValueError("prediction logits must align with input positions 1..T-1")
    if len(stream_lengths) != batch_size:
        raise ValueError("one stream length is required per batch episode")

    labels = input_one_hot[:, 1:].argmax(dim=-1)
    positions = torch.arange(1, steps + 1, device=logits.device).unsqueeze(0)
    lengths = torch.as_tensor(stream_lengths, device=logits.device).unsqueeze(1)
    valid = positions < lengths
    # Evaluate CE only where the target belongs to the stream. Computing CE
    # on excluded query/answer/padding positions and multiplying by zero
    # afterward lets a single NaN there poison the whole loss (NaN * 0 = NaN).
    valid_logits = logits[valid].float()
    valid_labels = labels[valid]
    if valid_logits.shape[0] == 0:
        # Summing the empty selection yields a differentiable exact zero,
        # without touching excluded NaN logits.
        return valid_logits.sum()
    loss_sum = F.cross_entropy(valid_logits, valid_labels, reduction="sum")
    return loss_sum / valid.sum().clamp_min(1)


def _auroc(positive: list[float], negative: list[float]) -> float | None:
    """Exact rank AUROC with average ranks for ties; no sklearn dependency."""
    if not positive or not negative:
        return None
    values = np.asarray(positive + negative, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_count = len(positive)
    rank_sum = ranks[:positive_count].sum()
    return float(
        (rank_sum - positive_count * (positive_count + 1) / 2) / (positive_count * len(negative))
    )


def _calibration_summary(
    probabilities: list[np.ndarray], labels: list[int], bin_count: int = 15
) -> dict[str, float | int | None]:
    if not labels:
        return {"count": 0, "accuracy_pct": None, "ece": None, "brier": None}
    probs = np.concatenate(probabilities, axis=0).astype(np.float64, copy=False)
    targets = np.asarray(labels, dtype=np.int64)
    predictions = probs.argmax(axis=-1)
    confidence = probs.max(axis=-1)
    correct = predictions == targets
    ece = 0.0
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        in_bin = (confidence >= lower) & (
            confidence <= upper if index == bin_count - 1 else confidence < upper
        )
        if in_bin.any():
            ece += float(in_bin.mean()) * abs(
                float(correct[in_bin].mean()) - float(confidence[in_bin].mean())
            )
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(targets.size), targets] = 1.0
    return {
        "count": int(targets.size),
        "accuracy_pct": float(100.0 * correct.mean()),
        "ece": ece,
        "brier": float(np.square(probs - one_hot).sum(axis=-1).mean()),
    }


@torch.no_grad()
def evaluate_prediction_signals(
    backbone: nn.Module,
    decoder: SurprisePredictionHead,
    device: torch.device,
    *,
    episodes_per_lesson: int = 128,
    batch_size: int = 32,
    lesson_indices: tuple[int, ...] | None = None,
    seed: int = 730_201,
) -> dict[str, Any]:
    """Evaluate causal next-token surprise and calibration on fixed probes.

    Event scores use the prediction made at p-1 for the token at p. Clean
    normal positions exclude all labeled events and A6 jitter. A3 is retained
    as a decoy-specific diagnostic, not pooled as a true anomaly.
    """
    if episodes_per_lesson < 1 or batch_size < 1:
        raise ValueError("episodes_per_lesson and batch_size must be positive")
    lesson_indices = lesson_indices or tuple(range(len(LESSONS)))
    previous_backbone_mode = backbone.training
    previous_decoder_mode = decoder.training
    backbone.eval()
    decoder.eval()

    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"raw": [], "excess": []})
    normal_scores: dict[str, list[float]] = {"raw": [], "excess": []}
    true_anomaly_scores: dict[str, list[float]] = {"raw": [], "excess": []}
    jitter_scores: dict[str, list[float]] = {"raw": [], "excess": []}
    calibration_probabilities: list[np.ndarray] = []
    calibration_labels: list[int] = []
    entropy_values: list[float] = []
    surprisal_values: list[float] = []
    a5_recovery_steps: list[int] = []
    a5_recovery_censored = 0
    event_counts = {kind: 0 for kind in EVENT_TYPES}
    normal_count = 0

    try:
        for lesson_index in lesson_indices:
            if not 0 <= lesson_index < len(LESSONS):
                raise ValueError(f"invalid SGR lesson index {lesson_index}")
            rng = random.Random(seed + lesson_index * 10_007)
            remaining = episodes_per_lesson
            while remaining:
                count = min(batch_size, remaining)
                examples = [build_episode(lesson_index, rng) for _ in range(count)]
                streams = [example.inputs[: example.answer_start - 1] for example in examples]
                max_length = max(map(len, streams))
                token_ids = torch.zeros(count, max_length, dtype=torch.long, device=device)
                stream_lengths = []
                for row, stream in enumerate(streams):
                    token_ids[row, : len(stream)] = torch.as_tensor(stream, device=device)
                    stream_lengths.append(len(stream))
                one_hot = F.one_hot(token_ids, num_classes=VOCAB_SIZE).to(dtype=torch.float32)
                # Match training and the roadmap's inter-layer plan: measure
                # predictions from block 0, before any later backbone block.
                forward_with_intermediate = getattr(backbone, "forward_with_intermediate", None)
                if callable(forward_with_intermediate):
                    feature_holder: dict[str, torch.Tensor] = {}
                    forward_with_intermediate(
                        one_hot,
                        intermediate_observer=lambda value, holder=feature_holder: (
                            holder.setdefault("features", value)
                        ),
                    )
                    features = feature_holder.get("features")
                    if features is None:
                        raise RuntimeError(
                            "backbone forward_with_intermediate did not expose block-0 features"
                        )
                elif isinstance(backbone, nn.Sequential) and len(backbone) > 1:
                    features = backbone[0](one_hot)
                else:
                    features = backbone(one_hot)
                logits = decoder(features[:, :-1]).float()
                log_probs = F.log_softmax(logits, dim=-1)
                probs = log_probs.exp()
                targets = token_ids[:, 1:]
                raw_surprisal = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                entropy = -(probs * log_probs).sum(dim=-1)
                excess_surprise = raw_surprisal - entropy
                predictions = probs.argmax(dim=-1)
                correct_next = predictions.eq(targets)
                raw_cpu = raw_surprisal.cpu().numpy()
                excess_cpu = excess_surprise.cpu().numpy()
                entropy_cpu = entropy.cpu().numpy()
                probs_cpu = probs.cpu().numpy()
                correct_cpu = correct_next.cpu().numpy()

                for row, (example, stream_length) in enumerate(zip(examples, stream_lengths)):
                    event_map = dict(zip(example.anomaly_positions, example.anomaly_types))
                    jitter = set(example.noise_positions)
                    event_positions = {
                        kind: position
                        for position, kind in zip(example.anomaly_positions, example.anomaly_types)
                    }
                    if "A5" in event_positions:
                        change_position = event_positions["A5"]
                        recovery_start = change_position + 1
                        window_size = len(example.pattern)
                        found_recovery = False
                        for window_start in range(recovery_start, stream_length - window_size + 1):
                            window = correct_cpu[
                                row, window_start - 1 : window_start - 1 + window_size
                            ]
                            if float(window.mean()) >= 0.9:
                                a5_recovery_steps.append(window_start - change_position)
                                found_recovery = True
                                break
                        if not found_recovery:
                            a5_recovery_censored += 1
                    calibration_probabilities.append(probs_cpu[row, : stream_length - 1])
                    calibration_labels.extend(streams[row][1:])
                    for position in range(1, stream_length):
                        score_index = position - 1
                        raw = float(raw_cpu[row, score_index])
                        excess = float(excess_cpu[row, score_index])
                        entropy_value = float(entropy_cpu[row, score_index])
                        kind = event_map.get(position)
                        if kind is not None:
                            event_counts[kind] += 1
                            scores[kind]["raw"].append(raw)
                            scores[kind]["excess"].append(excess)
                            if kind in TRUE_ANOMALIES:
                                true_anomaly_scores["raw"].append(raw)
                                true_anomaly_scores["excess"].append(excess)
                        elif position in jitter:
                            event_counts["A6"] += 1
                            jitter_scores["raw"].append(raw)
                            jitter_scores["excess"].append(excess)
                        else:
                            normal_count += 1
                            normal_scores["raw"].append(raw)
                            normal_scores["excess"].append(excess)
                        surprisal_values.append(raw)
                        entropy_values.append(entropy_value)
                remaining -= count
    finally:
        backbone.train(previous_backbone_mode)
        decoder.train(previous_decoder_mode)

    per_type = {}
    for kind in EVENT_TYPES:
        type_scores = jitter_scores if kind == "A6" else scores[kind]
        per_type[kind] = {
            "count": event_counts[kind],
            "raw_auroc_vs_clean_normal": _auroc(type_scores["raw"], normal_scores["raw"]),
            "excess_auroc_vs_clean_normal": _auroc(type_scores["excess"], normal_scores["excess"]),
            "mean_raw_surprisal": (
                float(np.mean(type_scores["raw"])) if type_scores["raw"] else None
            ),
            "mean_excess_surprise": (
                float(np.mean(type_scores["excess"])) if type_scores["excess"] else None
            ),
        }

    return {
        "lessons": list(lesson_indices),
        "episodes_per_lesson": episodes_per_lesson,
        "normal_positions": normal_count,
        "per_type": per_type,
        "a6_vs_true_anomalies": {
            "a6_jitter_count": len(jitter_scores["raw"]),
            "true_anomaly_count": len(true_anomaly_scores["raw"]),
            "raw_auroc": _auroc(true_anomaly_scores["raw"], jitter_scores["raw"]),
            "excess_auroc": _auroc(true_anomaly_scores["excess"], jitter_scores["excess"]),
            "mean_a6_raw_surprisal": (
                float(np.mean(jitter_scores["raw"])) if jitter_scores["raw"] else None
            ),
            "mean_a6_excess_surprise": (
                float(np.mean(jitter_scores["excess"])) if jitter_scores["excess"] else None
            ),
            "mean_true_anomaly_raw_surprisal": (
                float(np.mean(true_anomaly_scores["raw"])) if true_anomaly_scores["raw"] else None
            ),
            "mean_true_anomaly_excess_surprise": (
                float(np.mean(true_anomaly_scores["excess"]))
                if true_anomaly_scores["excess"]
                else None
            ),
        },
        "calibration": _calibration_summary(calibration_probabilities, calibration_labels),
        "a5_recovery": {
            "recovered_episodes": len(a5_recovery_steps),
            "censored_episodes": a5_recovery_censored,
            "mean_tokens_to_recovery": (
                float(np.mean(a5_recovery_steps)) if a5_recovery_steps else None
            ),
            "window_rule": "first period-length window after A5 change with >=90% next-token accuracy",
        },
        "mean_raw_surprisal": float(np.mean(surprisal_values)) if surprisal_values else None,
        "mean_predictive_entropy": float(np.mean(entropy_values)) if entropy_values else None,
    }
