"""
file: inference/run_inference.py

CLI entrypoint (counterpart of core_training.py's __main__). Flow:
  load checkpoint -> detect capabilities -> REFUSE if requested dataset type /
  dims are unsupported -> build task -> rebuild model + load weights ->
  (optional) set up caches -> run -> summarize/log.

Examples (from the project root):
  python -m inference.run_inference CKPT.pt
  python -m inference.run_inference CKPT.pt --inspect
  python -m inference.run_inference CKPT.pt --no-reset-experience --loop 50 --window 10 --compare-fresh
  python -m inference.run_inference CKPT.pt --dataset-link my_graph.csv --path-length 3 8
  # caching (see inference/cache/):
  python -m inference.run_inference CKPT.pt --shared-context --cache prefix
  python -m inference.run_inference CKPT.pt --shared-context --num-contexts 3 --cache prefix,result --cache-dir ./inference_cache
  python -m inference.run_inference CKPT.pt --no-reset-experience --loop 50 --cache result --cache-dir ./inference_cache
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import json
import random
import time

import torch

from inference.inference_config import (
    INFERENCE_LOG_DIR, DEFAULT_DATASET_TYPE, DEFAULT_DATASET_LINK, DEFAULT_NUM_EPISODES,
    DEFAULT_WINDOW, DEFAULT_SEED, DEFAULT_VERBOSE_N, PROGRESS_EVERY,
)
from inference.cache.cache_config import (
    DEFAULT_CACHE, DEFAULT_CACHE_RAM_MB, DEFAULT_CACHE_DISK_MB, DEFAULT_VERIFY_HITS,
)
from inference.cache.cache_registry import parse_cache_spec, setup_caches
from inference.cache.cached_engine import CachedInferenceEngine
from inference.checkpoint_io import load_checkpoint, describe_checkpoint
from inference.capabilities import (
    detect_capabilities, require_type_supported, require_dims_match, IncompatibleModelError,
)
from inference.tasks.task_registry import get_task
from inference.model_loader import load_model
from memory_manipulation.nvrtc_compat import patch_prod_jiterator
patch_prod_jiterator()  # environment workaround -- see that module's docstring
from inference.engine import InferenceEngine
from inference.metrics import aggregate, windowed, adaptation_trend
from inference import run_logging as rl


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Checkpoint-driven inference. Refuses to run if the requested dataset "
                    "type is not supported by the checkpoint.")
    p.add_argument("checkpoint", type=str, help="path to a core_training checkpoint (.pt), periodic or final")
    p.add_argument("--dataset-type", type=str, default=DEFAULT_DATASET_TYPE,
                   help="task type to run (graph | text | audio | video). Must be supported by the checkpoint.")
    p.add_argument("--dataset-link", type=str, default=DEFAULT_DATASET_LINK,
                   help="test data location. Omit for the built-in London Underground graph; "
                        "for graph, a .csv/.tsv/.txt/.json edge file (src,dst,line).")
    p.add_argument("--reset-experience", action=argparse.BooleanOptionalAction, default=True,
                   help="--reset-experience (default): every episode starts fresh (independent test). "
                        "--no-reset-experience: memory + controller state persist across episodes.")
    p.add_argument("--loop", type=int, default=None,
                   help="persistent mode only: number of consecutive episodes with state carried over.")
    p.add_argument("--num-episodes", type=int, default=None,
                   help=f"number of episodes (default {DEFAULT_NUM_EPISODES}); in persistent mode "
                        "use --loop instead.")
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                   help="persistent mode: episodes per window in the adaptation table.")
    p.add_argument("--compare-fresh", action="store_true",
                   help="persistent mode: also replay the SAME episodes with reset each episode and report the delta.")
    p.add_argument("--path-length", type=int, nargs=2, metavar=("MIN", "MAX"), default=None,
                   help="graph task: walk length range (default: the training-time OOD range).")
    p.add_argument("--perturbation-severity", type=float, default=None,
                   help="Robustness probe (Layer D): if set, episodes are built with "
                        "perturbation={'severity': X} - e.g. text/audio/video token-corruption / "
                        "additive-noise / frame-dropping severity in [0,1]. Graph task ignores this "
                        "(no perturbation hook wired for it yet).")
    p.add_argument("--shared-context", action="store_true",
                   help="graph task: every episode = ONE fixed edge listing (static prefix) + a fresh "
                        "random query. Required for the prefix cache to have anything to reuse.")
    p.add_argument("--num-contexts", type=int, default=1,
                   help="with --shared-context: number of distinct fixed edge listings, used round-robin.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", type=str, default=None, help="default: cuda:0 if available, else cpu")
    p.add_argument("--deterministic-write", action="store_true",
                   help="disable write-vector sampling (use mu only), even if the model was trained with beta>0.")
    p.add_argument("--ablate-memory", action="store_true",
                   help="skip memory read+write (functional-usage check).")
    p.add_argument("--ablate-combiner-stage", type=str, nargs="+", default=None,
                   help="split-graph checkpoints with a hybrid controller combiner only "
                        "(--split-graph-combiner-mode controller, variant like 'mamba+cfc'): "
                        "bypass one or more stages by name (as listed by --inspect, e.g. 'cfc') "
                        "or 0-based index, to measure that stage's contribution on real held-out "
                        "episodes. Compare against a run with this flag omitted for the delta.")
    p.add_argument("--report-moe-stats", action="store_true",
                   help="after the run, print each installed MoE sublayer's routing "
                        "diagnostics (CV(load), CV(importance), max load fraction) from its "
                        "last forward call -- confirms MoE is actually routing/balancing on "
                        "this data, independent of task accuracy.")
    p.add_argument("--verbose-n", type=int, default=DEFAULT_VERBOSE_N,
                   help="print per-item predictions for the first N episodes.")
    p.add_argument("--log-dir", type=str, default=INFERENCE_LOG_DIR)
    p.add_argument("--run-id-suffix", type=str, default=None)
    p.add_argument("--no-log", action="store_true", help="do not write CSV/JSON files.")
    p.add_argument("--model-config-override", type=str, default=None,
                   help='JSON dict merged into the checkpoint model_config, e.g. '
                        '\'{"split_graph_num_blocks": 3}\' (for legacy checkpoints).')
    p.add_argument("--inspect", action="store_true",
                   help="print the checkpoint's config + capabilities and exit.")

    g = p.add_argument_group("caching (inference/cache/)")
    g.add_argument("--cache", type=str, default=DEFAULT_CACHE,
                   help="off | prefix | result | prefix,result | all. 'prefix' = snapshot the model state "
                        "after a static prefix and resume queries from it (needs --shared-context and "
                        "reset-experience). 'result' = memoize whole-episode outputs (deterministic models).")
    g.add_argument("--no-cache", action="store_true", help="force caching off (overrides --cache).")
    g.add_argument("--cache-dir", type=str, default=None,
                   help="enable the disk tier here, so caches survive across CLI invocations "
                        "(one sub-folder per model fingerprint).")
    g.add_argument("--cache-ram-mb", type=float, default=DEFAULT_CACHE_RAM_MB, help="RAM tier size cap.")
    g.add_argument("--cache-disk-mb", type=float, default=DEFAULT_CACHE_DISK_MB, help="disk tier size cap.")
    g.add_argument("--cache-verify", type=int, default=DEFAULT_VERIFY_HITS,
                   help="recompute the first N cache hits without the cache and compare "
                        "(0 = off). On mismatch the cache is disabled for the rest of the run.")
    g.add_argument("--cache-clear", action="store_true", help="wipe this model's disk cache before running.")
    g.add_argument("--cache-allow-stochastic", action="store_true",
                   help="allow caching even if the model samples its write vectors (results are then "
                        "one random draw reused; verification is skipped).")
    args = p.parse_args(argv)

    if args.loop is not None and args.reset_experience:
        p.error("--loop requires --no-reset-experience (state must persist across episodes).")
    if args.compare_fresh and args.reset_experience:
        p.error("--compare-fresh requires --no-reset-experience.")
    if args.loop is not None and args.num_episodes is not None:
        p.error("use either --loop (persistent) or --num-episodes, not both.")
    for name in ("loop", "num_episodes"):
        v = getattr(args, name)
        if v is not None and v < 1:
            p.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.num_contexts > 1 and not args.shared_context:
        p.error("--num-contexts requires --shared-context")
    try:
        parse_cache_spec(args.cache)
    except ValueError as e:
        p.error(str(e))
    if args.model_config_override is not None:
        try:
            args.model_config_override = json.loads(args.model_config_override)
            assert isinstance(args.model_config_override, dict)
        except (json.JSONDecodeError, AssertionError):
            p.error("--model-config-override must be a JSON object")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"[inference] device: {device}")

    ckpt = load_checkpoint(args.checkpoint, args.model_config_override)
    caps = detect_capabilities(ckpt)
    print(f"[inference] checkpoint: {args.checkpoint}\n{describe_checkpoint(ckpt)}")
    print(f"[inference] capabilities: {caps.describe()}")
    if args.inspect:
        return 0

    # ---- compatibility gates (cheap, BEFORE building anything) --------------
    try:
        canon = require_type_supported(caps, args.dataset_type, args.checkpoint)
        task = get_task(canon, args.dataset_link, path_length_range=args.path_length,
                        shared_context=args.shared_context, num_contexts=args.num_contexts)
        require_dims_match(caps, task, args.checkpoint)
    except (IncompatibleModelError, NotImplementedError, ValueError, FileNotFoundError) as e:
        raise SystemExit(f"[inference] ABORT: {e}")
    print(f"[inference] task: {task.describe()}")

    torch.manual_seed(args.seed)
    loaded = load_model(ckpt, device, deterministic_write=args.deterministic_write)
    print(f"[inference] model loaded (step {loaded.step}, sampled_writes={loaded.sampled_writes})")
    if loaded.combiner_stage_kinds:
        print(f"[inference] combiner stages: {list(enumerate(loaded.combiner_stage_kinds))}")

    combiner_skip_stages = None
    if args.ablate_combiner_stage:
        kinds = loaded.combiner_stage_kinds
        if not kinds:
            raise SystemExit("[inference] ABORT: --ablate-combiner-stage requires a split-graph "
                              "checkpoint with --split-graph-combiner-mode controller; this "
                              "checkpoint has no combiner stages to ablate.")
        idx = set()
        for tok in args.ablate_combiner_stage:
            if tok.isdigit() and 0 <= int(tok) < len(kinds):
                idx.add(int(tok))
            elif tok in kinds:
                idx.add(kinds.index(tok))
            else:
                raise SystemExit(f"[inference] ABORT: unknown combiner stage {tok!r}; "
                                  f"this checkpoint's stages are {list(enumerate(kinds))}")
        combiner_skip_stages = frozenset(idx)
        print(f"[inference] ablating combiner stage(s) "
              f"{[(i, kinds[i]) for i in sorted(combiner_skip_stages)]}")

    reset = args.reset_experience
    if reset:
        n = args.num_episodes if args.num_episodes is not None else DEFAULT_NUM_EPISODES
    else:
        n = args.loop if args.loop is not None else (
            args.num_episodes if args.num_episodes is not None else DEFAULT_NUM_EPISODES)

    rng = random.Random(args.seed)
    perturbation = {"severity": args.perturbation_severity} if args.perturbation_severity is not None else None
    episodes = task.build_episodes(n, rng, perturbation=perturbation)  # materialized -> identical episodes for --compare-fresh

    # ---- caches ---------------------------------------------------------------
    caches, cache_notes, cache_fp = [], [], None
    if not args.no_cache:
        caches, cache_notes, cache_fp = setup_caches(
            args.cache, ckpt=ckpt, device=device, sampled_writes=loaded.sampled_writes,
            deterministic_write=args.deterministic_write, ablate_memory=args.ablate_memory,
            combiner_skip_stages=combiner_skip_stages,
            cache_dir=args.cache_dir, ram_mb=args.cache_ram_mb, disk_mb=args.cache_disk_mb,
            clear=args.cache_clear, allow_stochastic=args.cache_allow_stochastic)
    for note in cache_notes:
        print(f"[inference] cache: {note}")
    if caches:
        print(f"[inference] cache: {', '.join(c.name for c in caches)} | model fingerprint "
              f"{cache_fp[:12]} | RAM {args.cache_ram_mb:.0f} MB"
              + (f" | disk {args.cache_dir}" if args.cache_dir else ""))
        if any(c.wants_boundaries for c in caches) and not any(ep.cache_boundaries for ep in episodes):
            cache_notes.append("prefix cache is on but no episode declares a static prefix "
                               "(use --shared-context); it will not hit.")
            print(f"[inference] cache: {cache_notes[-1]}")

    if caches:
        engine = CachedInferenceEngine(
            loaded.rnn, loaded.output_proj, task, device, caches,
            ablate_memory=args.ablate_memory, combiner_skip_stages=combiner_skip_stages,
            verify_hits=(0 if loaded.sampled_writes else args.cache_verify))
    else:
        engine = InferenceEngine(loaded.rnn, loaded.output_proj, task, device,
                                 ablate_memory=args.ablate_memory,
                                 combiner_skip_stages=combiner_skip_stages)

    t0 = time.time()
    print(f"[inference] running {n} episodes | reset_experience={reset}")
    results = engine.run(episodes, reset_experience=reset, verbose_n=args.verbose_n,
                         progress_every=PROGRESS_EVERY)
    scores = [r.score for r in results]

    tags = [args.run_id_suffix,
            ("cache-" + "+".join(c.name for c in caches)) if caches else None,
            "sharedctx" if args.shared_context else None]
    suffix = "_".join(t for t in tags if t) or None

    summary = {
        "run_id": rl.make_run_id(args.checkpoint, task.name, reset, n, suffix),
        "checkpoint": args.checkpoint, "checkpoint_step": loaded.step, "controller": caps.controller_type,
        "supported_types": caps.supported_types, "capabilities_source": caps.source,
        "dataset_type": canon, "dataset_link": args.dataset_link, "task": task.describe(),
        "mode": "fresh" if reset else "persistent", "reset_experience": reset,
        "seed": args.seed, "sampled_writes": loaded.sampled_writes, "ablate_memory": args.ablate_memory,
        "result": aggregate(scores),
        "mean_episode_ms": sum(r.elapsed_ms for r in results) / max(len(results), 1),
    }
    if caches:
        summary["cache"] = {
            "requested": args.cache, "active": [c.name for c in caches],
            "model_fingerprint": cache_fp, "disk_dir": args.cache_dir,
            "report": engine.cache_report(reset=True), "verify": dict(engine.verify),
            "store": caches[0].store.info(), "notes": cache_notes + engine.notes,
        }
    elif cache_notes:
        summary["cache"] = {"requested": args.cache, "active": [], "notes": cache_notes,
                            "report": {}, "disk_dir": None}
    if not reset:
        summary["windows"] = windowed(scores, args.window)
        summary["adaptation"] = adaptation_trend(scores, args.window)
        if args.compare_fresh:
            print("[inference] replaying the same episodes with reset_experience=True (baseline)")
            fresh = engine.run(episodes, reset_experience=True, progress_every=0, label=":fresh")
            summary["fresh_baseline"] = aggregate([r.score for r in fresh])
            summary["persistent_minus_fresh_item_acc"] = (
                summary["result"]["item_acc"] - summary["fresh_baseline"]["item_acc"])
            if caches:
                summary["cache"]["baseline_report"] = engine.cache_report(reset=True)
                summary["cache"]["notes"] = cache_notes + engine.notes
    summary["elapsed_sec"] = time.time() - t0

    if not args.no_log:
        os.makedirs(args.log_dir, exist_ok=True)
        base = os.path.join(args.log_dir, summary["run_id"])
        files = [f"{base}_episodes.csv", f"{base}_summary.json"]
        rl.write_episode_csv(files[0], results)
        if not reset:
            files.insert(1, f"{base}_windows.csv")
            rl.write_window_csv(files[1], summary["windows"])
        summary["log_files"] = files
        rl.write_summary_json(files[-1], summary)

    if args.report_moe_stats:
        moe_layers = getattr(loaded.rnn, "moe_layers", None)
        if not moe_layers:
            print("[inference] --report-moe-stats: this checkpoint has no MoE layers installed.")
        else:
            print(f"[inference] MoE routing diagnostics ({len(moe_layers)} sublayer(s), "
                  f"from the LAST forward call only):")
            for i, layer in enumerate(moe_layers):
                diag = layer.last_diagnostics()
                if not diag:
                    print(f"  [{i}] no diagnostics recorded")
                    continue
                print(f"  [{i}] cv_load={diag['cv_load']:.4f} "
                      f"cv_importance={diag['cv_importance']:.4f} "
                      f"max_load_frac={diag['max_load_frac']:.4f}")

    rl.print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())