"""
file: inference/run_inference.py

CLI entrypoint (counterpart of core_training.py's __main__). Flow:
  load checkpoint -> detect capabilities -> REFUSE if requested dataset type /
  dims are unsupported -> build task -> rebuild model + load weights -> run ->
  summarize/log.

Examples (from the project root):
  python -m inference.run_inference CKPT.pt
  python -m inference.run_inference CKPT.pt --inspect
  python -m inference.run_inference CKPT.pt --no-reset-experience --loop 50 --window 10 --compare-fresh
  python -m inference.run_inference CKPT.pt --dataset-link my_graph.csv --path-length 3 8
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
from inference.checkpoint_io import load_checkpoint, describe_checkpoint
from inference.capabilities import (
    detect_capabilities, require_type_supported, require_dims_match, IncompatibleModelError,
)
from inference.tasks.task_registry import get_task
from inference.model_loader import load_model
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
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", type=str, default=None, help="default: cuda:0 if available, else cpu")
    p.add_argument("--deterministic-write", action="store_true",
                   help="disable write-vector sampling (use mu only), even if the model was trained with beta>0.")
    p.add_argument("--ablate-memory", action="store_true",
                   help="skip memory read+write (functional-usage check).")
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
        task = get_task(canon, args.dataset_link, path_length_range=args.path_length)
        require_dims_match(caps, task, args.checkpoint)
    except (IncompatibleModelError, NotImplementedError, ValueError, FileNotFoundError) as e:
        raise SystemExit(f"[inference] ABORT: {e}")
    print(f"[inference] task: {task.describe()}")

    torch.manual_seed(args.seed)
    loaded = load_model(ckpt, device, deterministic_write=args.deterministic_write)
    print(f"[inference] model loaded (step {loaded.step}, sampled_writes={loaded.sampled_writes})")

    reset = args.reset_experience
    if reset:
        n = args.num_episodes if args.num_episodes is not None else DEFAULT_NUM_EPISODES
    else:
        n = args.loop if args.loop is not None else (
            args.num_episodes if args.num_episodes is not None else DEFAULT_NUM_EPISODES)

    rng = random.Random(args.seed)
    episodes = task.build_episodes(n, rng)          # materialized -> identical episodes for --compare-fresh
    engine = InferenceEngine(loaded.rnn, loaded.output_proj, task, device, ablate_memory=args.ablate_memory)

    t0 = time.time()
    print(f"[inference] running {n} episodes | reset_experience={reset}")
    results = engine.run(episodes, reset_experience=reset, verbose_n=args.verbose_n,
                         progress_every=PROGRESS_EVERY)
    scores = [r.score for r in results]

    summary = {
        "run_id": rl.make_run_id(args.checkpoint, task.name, reset, n, args.run_id_suffix),
        "checkpoint": args.checkpoint, "checkpoint_step": loaded.step, "controller": caps.controller_type,
        "supported_types": caps.supported_types, "capabilities_source": caps.source,
        "dataset_type": canon, "dataset_link": args.dataset_link, "task": task.describe(),
        "mode": "fresh" if reset else "persistent", "reset_experience": reset,
        "seed": args.seed, "sampled_writes": loaded.sampled_writes, "ablate_memory": args.ablate_memory,
        "result": aggregate(scores),
    }
    if not reset:
        summary["windows"] = windowed(scores, args.window)
        summary["adaptation"] = adaptation_trend(scores, args.window)
        if args.compare_fresh:
            print("[inference] replaying the same episodes with reset_experience=True (baseline)")
            fresh = engine.run(episodes, reset_experience=True, progress_every=0, label=":fresh")
            summary["fresh_baseline"] = aggregate([r.score for r in fresh])
            summary["persistent_minus_fresh_item_acc"] = (
                summary["result"]["item_acc"] - summary["fresh_baseline"]["item_acc"])
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

    rl.print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())