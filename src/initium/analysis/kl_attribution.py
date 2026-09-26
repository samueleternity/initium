"""Matched-pair OGS/KL attribution report generator.

Manifest example::

    {"pairs": [{"seed": 1, "beta0_ogs": "...csv", "beta_ogs": "...csv",
                "beta0_train": "...csv", "beta_train": "...csv"}]}

Select the evaluation window before examining outcomes with --step-start and
--step-end. Each run is averaged over that same inclusive step interval before
the within-seed beta-minus-zero difference is computed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

METRICS = (
    "ogs",
    "ood_acc",
    "offset_normalized",
    "memory_attributable_ood",
    "ood_perfect_frac",
    "classic_probe_ood_delta",
)


def _read_window(path, step_start, step_end, columns):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            step = int(float(row["step"]))
            if step_start <= step <= step_end:
                rows.append({name: float(row[name]) for name in columns if row.get(name) not in (None, "")})
    if not rows:
        raise ValueError(f"{path}: no rows in preregistered step window {step_start}..{step_end}")
    return {name: sum(row[name] for row in rows if name in row) / sum(name in row for row in rows)
            for name in columns if any(name in row for row in rows)}


def _mean(values):
    return sum(values) / len(values)


def _sd(values):
    return math.sqrt(sum((x - _mean(values)) ** 2 for x in values) / (len(values) - 1)) if len(values) > 1 else 0.0


def _betacf(a, b, x):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1e-300 if abs(d) < 1e-300 else 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1.0 + aa / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1.0 + aa / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return h


def _ibeta(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    factor = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                      + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * _betacf(a, b, x) / a
    return 1.0 - factor * _betacf(b, a, 1.0 - x) / b


def _t_cdf(t, degrees):
    x = degrees / (degrees + t * t)
    tail = 0.5 * _ibeta(degrees / 2.0, 0.5, x)
    return 1.0 - tail if t >= 0 else tail


def _t_critical(degrees, confidence=0.95):
    target = (1.0 + confidence) / 2.0
    lo, hi = 0.0, 100.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _t_cdf(mid, degrees) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _pearson(xs, ys):
    if len(xs) < 2:
        return float("nan")
    mx, my = _mean(xs), _mean(ys)
    xx = sum((x - mx) ** 2 for x in xs)
    yy = sum((y - my) ** 2 for y in ys)
    if xx == 0 or yy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(xx * yy)


def _paired_report(deltas):
    n = len(deltas)
    mean = _mean(deltas)
    sd = _sd(deltas)
    se = sd / math.sqrt(n) if n else float("nan")
    t_value = mean / se if se else (float("inf") if mean else 0.0)
    p_value = 2.0 * (1.0 - _t_cdf(abs(t_value), n - 1)) if n > 1 else float("nan")
    margin = _t_critical(n - 1) * se if n > 1 else float("nan")
    return {
        "n": n, "mean_delta": mean, "sd_delta": sd, "paired_t": t_value,
        "p_two_sided": p_value, "ci95": [mean - margin, mean + margin],
        "cohens_dz": mean / sd if sd else float("nan"),
    }


def _training_diagnostics(pair, step_start, step_end):
    result = {}
    for label in ("beta0_train", "beta_train"):
        path = pair.get(label)
        if not path:
            continue
        rows = _read_window(path, step_start, step_end, ("task_loss", "kl_mean", "kl_max"))
        result[label] = rows
    if "beta_train" in result:
        kl_mean = result["beta_train"].get("kl_mean")
        result["kl_collapse_flag"] = kl_mean is not None and abs(kl_mean) < 1e-5
    if "beta0_train" in result and "beta_train" in result:
        a = result["beta0_train"].get("task_loss")
        b = result["beta_train"].get("task_loss")
        result["task_loss_delta_beta_minus_zero"] = b - a if a is not None and b is not None else None
        result["relative_task_loss_increase"] = (
            (b - a) / max(abs(a), 1e-12) if a is not None and b is not None else None
        )
    return result


def build_report(manifest, step_start, step_end):
    if "comparisons" in manifest:
        reports = []
        for comparison in manifest["comparisons"]:
            submanifest = {
                "comparison": comparison.get("label", "unlabeled comparison"),
                "pairs": comparison["pairs"],
            }
            reports.append({
                "task_family": comparison.get("task_family"),
                "architecture": comparison.get("architecture"),
                **build_report(submanifest, step_start, step_end),
            })
        directions = [
            math.copysign(1, item["aggregate"]["ogs"]["mean_delta"])
            for item in reports
            if "ogs" in item.get("aggregate", {}) and item["aggregate"]["ogs"]["mean_delta"] != 0
        ]
        return {
            "preregistered_step_window": [step_start, step_end],
            "comparisons": reports,
            "two_by_two_consistency": {
                "ogs_effect_direction_consistent": len(directions) == len(reports) and len(set(directions)) <= 1,
                "criterion": "KL OGS effect sign should agree between lesson/classic task families and baseline/flagship architectures.",
            },
        }
    pairs = manifest["pairs"]
    if not pairs:
        raise ValueError("manifest needs at least one matched seed pair")
    columns = list(METRICS)
    rows, diagnostics = [], []
    for pair in pairs:
        if (
            pair.get("beta0_architecture") is not None
            and pair.get("beta_architecture") is not None
            and pair["beta0_architecture"] != pair["beta_architecture"]
        ):
            raise ValueError(f"seed {pair['seed']}: beta pair uses different architectures")
        if (
            pair.get("beta0_step_budget") is not None
            and pair.get("beta_step_budget") is not None
            and pair["beta0_step_budget"] != pair["beta_step_budget"]
        ):
            raise ValueError(f"seed {pair['seed']}: beta pair uses different step budgets")
        anchor = _read_window(pair["beta0_ogs"], step_start, step_end, columns)
        treatment = _read_window(pair["beta_ogs"], step_start, step_end, columns)
        row = {"seed": pair["seed"]}
        for metric in columns:
            if metric in anchor and metric in treatment:
                row[metric] = treatment[metric] - anchor[metric]
        rows.append(row)
        diagnostics.append(_training_diagnostics(pair, step_start, step_end))
    stats = {metric: _paired_report([r[metric] for r in rows if metric in r])
             for metric in columns if any(metric in r for r in rows)}
    anchor_ogs = [
        _read_window(pair["beta0_ogs"], step_start, step_end, ("ogs",))["ogs"]
        for pair in pairs
    ]
    delta_ogs = [r.get("ogs", float("nan")) for r in rows]
    mean_anchor = _mean(anchor_ogs)
    regression_r = _pearson([v - mean_anchor for v in anchor_ogs], delta_ogs)
    return {
        "comparison": manifest.get("comparison", "beta effect"),
        "preregistered_step_window": [step_start, step_end],
        "pairing": "within-seed beta-minus-zero; same architecture and step window",
        "n_seeds": len(pairs),
        "minimum_n_8_met": len(pairs) >= 8,
        "per_seed_deltas": rows,
        "aggregate": stats,
        "regression_to_mean": {
            "corr_anchor_ogs_offset_vs_beta_delta": regression_r,
            "possible_variance_dampening": math.isfinite(regression_r) and regression_r <= -0.5,
        },
        "training_diagnostics": diagnostics,
        "no_collapse_and_no_cost_redistribution": {
            "all_beta_runs_noncollapsed": all(not d.get("kl_collapse_flag", False) for d in diagnostics),
            "task_loss_deltas": [d.get("task_loss_delta_beta_minus_zero") for d in diagnostics],
            "relative_task_loss_increases": [d.get("relative_task_loss_increase") for d in diagnostics],
            "possible_cost_redistribution": any(
                d.get("relative_task_loss_increase") is not None
                and d["relative_task_loss_increase"] >= 0.05
                for d in diagnostics
            ),
            "interpretation": "Inspect task_loss and KL together; this report flags near-zero mean KL but does not infer causality.",
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--step-start", type=int, required=True,
                        help="inclusive preregistered lower step bound")
    parser.add_argument("--step-end", type=int, required=True,
                        help="inclusive preregistered upper step bound")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.step_end < args.step_start:
        parser.error("--step-end must be >= --step-start")
    with args.manifest.open(encoding="utf-8") as f:
        report = build_report(json.load(f), args.step_start, args.step_end)
    rendered = json.dumps(report, indent=2, allow_nan=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
