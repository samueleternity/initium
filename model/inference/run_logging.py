"""
file: inference/run_logging.py

Output for an inference run: per-episode CSV, window CSV (persistent mode),
summary JSON, and the console summary. (Named run_logging to avoid shadowing
the stdlib `logging` module.)
"""
from __future__ import annotations

import csv
import json
import os
import re
from typing import List

from inference.metrics import EpisodeResult


def make_run_id(ckpt_path: str, task_name: str, reset_experience: bool,
                n_episodes: int, suffix: str | None = None) -> str:
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    mode = "fresh" if reset_experience else f"persist{n_episodes}"
    rid = f"infer_{stem}_{task_name}_{mode}" + (f"_{suffix}" if suffix else "")
    return re.sub(r"[^A-Za-z0-9_.-]", "_", rid)


def write_episode_csv(path: str, results: List[EpisodeResult]) -> None:
    names = sorted({n for r in results for n in r.score.fields})
    header = ["episode", "reset_experience", "n_items", "n_correct", "item_acc",
              "perfect", "group", "elapsed_ms", "cache_event"]
    for n in names:
        header += [f"{n}_correct", f"{n}_total"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in results:
            s = r.score
            row = [r.index, int(r.reset_experience), s.n_items, s.n_correct, s.item_acc,
                   int(s.perfect), s.group, r.elapsed_ms, r.cache_event]
            for n in names:
                c, t = s.fields.get(n, ("", ""))
                row += [c, t]
            w.writerow(row)


def write_window_csv(path: str, windows: List[dict]) -> None:
    names = sorted({n for w in windows for n in w["field_acc"]})
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["window_index", "start_episode", "end_episode", "n_episodes",
                     "item_acc", "perfect_frac"] + [f"{n}_acc" for n in names])
        for w in windows:
            wr.writerow([w["window_index"], w["start_episode"], w["end_episode"], w["n_episodes"],
                         w["item_acc"], w["perfect_frac"]] + [w["field_acc"].get(n, "") for n in names])


def write_summary_json(path: str, summary: dict) -> None:
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)


def print_summary(s: dict) -> None:
    r = s["result"]
    print("\n===== Inference summary =====")
    print(f"run_id     : {s['run_id']}")
    print(f"checkpoint : {s['checkpoint']} (step {s['checkpoint_step']}, controller {s['controller']})")
    print(f"task       : {s['task']}")
    print(f"mode       : {s['mode']} (reset_experience={s['reset_experience']}) | "
          f"{r['n_episodes']} episodes | seed {s['seed']} | "
          f"sampled_writes={s['sampled_writes']} | ablate_memory={s['ablate_memory']}")
    print(f"overall    : item acc {r['item_acc']:.2f}% | perfect {r['perfect_frac']:.2f}%")
    if r["by_group"]:
        print("by group   : " + " | ".join(
            f"{g}: {v['item_acc']:.1f}% (n={v['n_episodes']})" for g, v in r["by_group"].items()))
    if r["field_acc"]:
        print("fields     : " + " | ".join(f"{n} {a:.1f}%" for n, a in r["field_acc"].items()))
    if s.get("windows"):
        print("windows    : " + " | ".join(
            f"[{w['start_episode']}-{w['end_episode']}] {w['item_acc']:.1f}%" for w in s["windows"]))
    a = s.get("adaptation")
    if a:
        print(f"adaptation : first {a['window']} eps {a['first_window_item_acc']:.2f}% -> "
              f"last {a['window']} eps {a['last_window_item_acc']:.2f}% "
              f"(delta {a['delta_item_acc']:+.2f}) | slope {a['slope_acc_per_episode']:+.4f} acc-pts/episode")
    fb = s.get("fresh_baseline")
    if fb:
        print(f"fresh base : item acc {fb['item_acc']:.2f}% | perfect {fb['perfect_frac']:.2f}% "
              f"(same episodes, reset each) -> persistent minus fresh: "
              f"{s['persistent_minus_fresh_item_acc']:+.2f} acc-pts")
    c = s.get("cache")
    if c:
        print(f"cache      : requested={c['requested']} | active={c['active'] or 'none'}"
              + (f" | disk={c['disk_dir']}" if c.get("disk_dir") else ""))
        for label, key in (("run", "report"), ("fresh baseline", "baseline_report")):
            for name, r in (c.get(key) or {}).items():
                print(f"  [{label}] {name}: {r['hits']}/{r['lookups']} hits ({r['hit_rate']:.0f}%) "
                      f"| ram {r['ram_hits']} disk {r['disk_hits']} | stores {r['stores']} "
                      f"| est. compute saved ~{r['est_saved_ms']:.0f} ms")
        v = c.get("verify")
        if v and v["checked"]:
            print(f"  verify: {v['checked']} hit(s) recomputed uncached | max |d output| "
                  f"{v['max_output_diff']:.2e} | max |d state| {v['max_state_diff']:.2e} "
                  f"| failures {v['failures']}")
        for note in c.get("notes", []):
            print(f"  note: {note}")
    print(f"speed      : mean {s['mean_episode_ms']:.1f} ms/episode")
    print(f"elapsed    : {s['elapsed_sec']:.1f}s")
    if s.get("log_files"):
        print("logs       : " + ", ".join(s["log_files"]))