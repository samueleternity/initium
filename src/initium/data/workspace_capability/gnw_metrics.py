"""Small, dependency-free summary statistics for GNW functional probes.

These metrics describe task behavior only. They are not consciousness measures.
"""

from __future__ import annotations

import math
from collections.abc import Iterable


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a 95% Wilson interval for a binomial proportion, in [0, 1]."""
    if total <= 0:
        return (math.nan, math.nan)
    p = min(max(successes / total, 0.0), 1.0)
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total) / denominator
    return (max(0.0, center - radius), min(1.0, center + radius))


def binary_brier(scores: Iterable[float], outcomes: Iterable[int]) -> float:
    pairs = [(min(max(float(s), 0.0), 1.0), int(y)) for s, y in zip(scores, outcomes)]
    return sum((s - y) ** 2 for s, y in pairs) / len(pairs) if pairs else math.nan


def expected_calibration_error(
    scores: Iterable[float], outcomes: Iterable[int], bins: int = 10
) -> float:
    pairs = [(min(max(float(s), 0.0), 1.0), int(y)) for s, y in zip(scores, outcomes)]
    if not pairs:
        return math.nan
    counts = [0] * bins
    confidence = [0.0] * bins
    accuracy = [0.0] * bins
    for score, outcome in pairs:
        index = min(int(score * bins), bins - 1)
        counts[index] += 1
        confidence[index] += score
        accuracy[index] += outcome
    n = len(pairs)
    return sum(
        counts[i] / n * abs(confidence[i] / counts[i] - accuracy[i] / counts[i])
        for i in range(bins)
        if counts[i]
    )


def type2_auc(scores: Iterable[float], outcomes: Iterable[int]) -> float:
    """Mann–Whitney type-2 ROC AUC with average ranks for tied scores."""
    pairs = sorted((float(s), int(y)) for s, y in zip(scores, outcomes))
    positives = sum(y == 1 for _, y in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return math.nan
    positive_rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(pairs[j][1] == 1 for j in range(index, end))
        index = end
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def psychometric_logistic(strength_counts: dict[int, tuple[int, int]]) -> dict[str, float | int]:
    """Fit P(access)=sigmoid(intercept+slope*strength) by bounded Newton steps."""
    samples = [(float(x), int(k), int(n)) for x, (k, n) in strength_counts.items() if n > 0]
    total = sum(n for _, _, n in samples)
    if len(samples) < 2 or total == 0:
        return {"threshold_50": math.nan, "slope": math.nan, "n": total, "strength_levels": len(samples)}
    mean_x = sum(x * n for x, _, n in samples) / total
    a, b = 0.0, 0.0
    for _ in range(40):
        ga = gb = haa = hab = hbb = 0.0
        for x, k, n in samples:
            z = max(-30.0, min(30.0, a + b * x))
            p = 1.0 / (1.0 + math.exp(-z))
            w = max(p * (1.0 - p) * n, 1e-8)
            residual = k - n * p
            ga += residual
            gb += residual * x
            haa += w
            hab += w * x
            hbb += w * x * x
        haa += 1e-6
        hbb += 1e-6
        determinant = haa * hbb - hab * hab
        if determinant <= 1e-12:
            break
        da = (hbb * ga - hab * gb) / determinant
        db = (haa * gb - hab * ga) / determinant
        scale = min(1.0, 4.0 / max(abs(da), abs(db), 1e-12))
        a += scale * da
        b += scale * db
        if max(abs(scale * da), abs(scale * db)) < 1e-6:
            break
    threshold = -a / b if abs(b) > 1e-8 else math.nan
    haa = hab = hbb = 1e-6
    for x, _, n in samples:
        z = max(-30.0, min(30.0, a + b * x))
        p = 1.0 / (1.0 + math.exp(-z))
        weight = max(p * (1.0 - p) * n, 1e-8)
        haa += weight
        hab += weight * x
        hbb += weight * x * x
    determinant = haa * hbb - hab * hab
    slope_se = math.sqrt(max(haa / determinant, 0.0)) if determinant > 1e-12 else math.nan
    if abs(b) > 1e-8 and determinant > 1e-12:
        gradient_a = -1.0 / b
        gradient_b = a / (b * b)
        threshold_variance = (
            gradient_a * gradient_a * hbb
            - 2.0 * gradient_a * gradient_b * hab
            + gradient_b * gradient_b * haa
        ) / determinant
        threshold_se = math.sqrt(max(threshold_variance, 0.0))
    else:
        threshold_se = math.nan
    return {
        "threshold_50": threshold,
        "threshold_50_wald95": [threshold - 1.959963984540054 * threshold_se,
                                threshold + 1.959963984540054 * threshold_se],
        "slope": b,
        "slope_wald95": [b - 1.959963984540054 * slope_se,
                         b + 1.959963984540054 * slope_se],
        "n": total,
        "strength_levels": len(samples),
        "small_sample_warning": total < 100 or len(samples) < 5,
    }
