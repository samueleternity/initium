"""
file: core_training.py

Dataset-agnostic DNC training core (stochastic write head + learned-prior KL,
selectable controller, AMP, LR schedule, checkpointing, periodic ID/OOD/
memory-dependency logging). All dataset specifics (data generation, encoding,
loss, curriculum, evaluation, OOD test) come from data/ via
    dataset = get_dataset(--dataset-type, --dataset-link)
(default: graph-traversal). See data/base_dataset.py for the interface.

Version notes (core-level):
- v3: capacity scale-up (MODEL_* constants), linear LR warmup, AMP
  init_scale=128 and GradScaler state saved/restored in checkpoints.
- v5 (Phase 2): learned, periodically-snapshotted prior N(mu_g, Sigma_g)
  (math in stochastic_write_head_v2.py); prior_snapshots.csv log;
  snapshot_step column; prior_state in checkpoints; run_id tag "_learnedprior".
- v6: snapshot_step in the console line; ood_rng state in checkpoints.
- v7+: pluggable controllers via MambaDNC (lstm/mamba/mamba2/mamba3/cfc/
  hybrids), MoE (Option 4), split-graph (Option 5), link-matrix ablation
  (Option 1), static/dynamic N (Option 2), dynamic beta. Each keeps its own
  CLI flag and run_id tag so runs never collide in logs/.
- v17: dataset split out of this file into data/. Core only talk to the
  dataset through the BaseDataset interface; curriculum's per-lesson memory
  size lives on the curriculum instance (curriculum.lesson_nr_cells).
- v18: constants moved out: train_config.py (loop/model/KL/prior/checkpoint)
  and controller_config.py (controllers, MoE, split-graph, link-matrix,
  Dynamic-N). Pure move, no value changes.
"""

import csv
import math
import os
import random
import shutil
import sys
import time
from contextlib import contextmanager
from typing import Any, TextIO

# Support both ``python -m initium.core_training`` and direct execution of
# ``src/initium/core_training.py`` from a source checkout.
_SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

import numpy as np
import torch
import torch.nn as nn

from initium.memory_manipulation.nvrtc_compat import patch_prod_jiterator

patch_prod_jiterator()  # environment workaround -- see module docstring

OGS_WEIGHTS = (1.0, 1.0, 1.0)
from initium.config.controller_config import (
    CFC_ACTIVATION,
    CFC_BACKBONE_DROPOUT,
    CFC_BACKBONE_LAYERS,
    CFC_BACKBONE_UNITS,
    CFC_MIXED_MEMORY,
    CFC_MODE,
    CFC_RESIDUAL,
    CONTROLLER_TYPE,
    DNC_READ_RESIDUAL_MAX_RATIO,
    DNC_READ_RESIDUAL_SCALE,
    DNC_READ_VARIANT,
    DNC_WRITE_OBSERVATION_VARIANCE,
    DNC_WRITE_VARIANT,
    DYNAMIC_N_CEILING,
    DYNAMIC_N_COOLDOWN_STEPS,
    DYNAMIC_N_EMA_DECAY,
    DYNAMIC_N_FLOOR,
    DYNAMIC_N_GROWTH_FACTOR,
    DYNAMIC_N_MODE,
    DYNAMIC_N_TRIGGER_FRAC,
    DYNAMIC_N_USAGE_HIGH,
    HYBRID_CFC_NUM_BLOCKS,
    ISOLATE_LINK_ABLATION,
    LINK_MATRIX_MODE,
    LINK_MATRIX_TOPK,
    MAMBA2_D_CONV,
    MAMBA2_D_STATE,
    MAMBA2_EXPAND,
    MAMBA2_HEADDIM,
    MAMBA2_NGROUPS,
    MAMBA3_D_STATE,
    MAMBA3_EXPAND,
    MAMBA3_HEADDIM,
    MAMBA3_ROPE_FRACTION,
    MAMBA_D_CONV,
    MAMBA_D_STATE,
    MAMBA_EXPAND,
    MOE_CAPACITY_FACTOR,
    MOE_ENABLED,
    MOE_EXPERT_DIM,
    MOE_LOAD_BALANCE_ALPHA,
    MOE_NUM_EXPERTS,
    MOE_TOP_K,
    SPLIT_GRAPH_COMBINE_READS,
    SPLIT_GRAPH_COMBINER_CFC_MULTI_SOURCE_MOE,
    SPLIT_GRAPH_COMBINER_MODE,
    SPLIT_GRAPH_COMBINER_NUM_BLOCKS,
    SPLIT_GRAPH_COMBINER_VARIANT,
    SPLIT_GRAPH_ENABLED,
    SPLIT_GRAPH_MAMBA_HEADDIM,
    SPLIT_GRAPH_MAMBA_VARIANT,
    SPLIT_GRAPH_NUM_BLOCKS,
    WORKSPACE_ENABLED,
    WORKSPACE_NO_SELECTION,
    WORKSPACE_NUM_SLOTS,
    WORKSPACE_RESET_EACH_STEP,
    WORKSPACE_TRAIN_MODE,
)
from initium.config.train_config import (
    AMP_INIT_SCALE,
    BATCH_SIZE,
    BETA_CTRL_ACC_TARGET,
    BETA_CTRL_EMA_DECAY,
    BETA_CTRL_LR,
    BETA_CTRL_MAX,
    BETA_CTRL_MIN,
    BETA_MODE,
    BETAS_TO_SWEEP,
    CHECKPOINT_DIR,
    CHECKPOINT_EVERY,
    DATASET_LINK,
    DATASET_TYPE,
    EVAL_EVERY,
    FREE_BITS,
    KL_ANNEAL_STEPS,
    LESSON_KL_DIP_STEPS,
    LOG_DIR,
    LOG_EVERY,
    LR,
    LR_DECAY_STEPS,
    LR_MIN,
    MODEL_CELL_SIZE,
    MODEL_HIDDEN_SIZE,
    MODEL_NR_CELLS,
    MODEL_READ_HEADS,
    OOD_EVAL_EPISODES,
    OOD_EVAL_EPISODES_PERIODIC,
    PRIOR_MAX_LOGVAR,
    PRIOR_MIN_LOGVAR,
    PRIOR_SNAPSHOT_EVERY,
    SEED,
    TOTAL_STEPS,
    USE_AMP,
    WARMUP_STEPS,
)
from initium.data.dataset_registry import get_dataset
from initium.data.graph_traversal.order_probe import (
    build_q99_probe_batch,
    evaluate_q99_order_sensitivity,
)
from initium.data.graph_traversal.relational_probe import (
    build_relational_probe,
    cosine_top1_accuracy,
    score_probe_cases,
)
from initium.data.prepared_dataset import load_prepared_dataset, save_prepared_dataset
from initium.data.surprise_recall.next_token_prediction import (
    SurprisePredictionHead,
    evaluate_prediction_signals,
    masked_next_token_loss,
)
from initium.diagnostics.segment_logging import SegmentDiagnostics
from initium.diagnostics.split_graph_forward_trace import SplitGraphForwardTrace
from initium.diagnostics.surprise_recall_logging import SurpriseRecallLogger
from initium.mamba_controller.mamba_controller import MambaDNC
from initium.mamba_controller.sgr_backbone_only import SurpriseRecallBackboneOnly
from initium.mamba_controller.split_graph_dnc import SplitGraphDNC
from initium.memory_manipulation.dnc_mechanisms import compile_memory_forward
from initium.memory_manipulation.dynamic_memory_resize import resize_memory
from initium.memory_manipulation.dynamic_n_controller import DynamicNController
from initium.memory_manipulation.link_matrix_ablation import patch_link_matrix
from initium.memory_manipulation.stochastic_write_head_v2 import (
    get_prior_state,
    install_stochastic_write_heads,
    load_prior_state,
    pop_total_kl,
    update_all_prior_snapshots,
)
from initium.MoE.moe_layer import SwitchMoE, pop_total_moe_aux_loss
from initium.workspace.broadcast import WorkspaceBroadcast

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def _arch_desc(
    controller,
    sg_enabled,
    sg_variant,
    sg_blocks,
    sg_comb_mode,
    sg_comb_variant,
    sg_comb_blocks,
    sg_combine_reads,
):
    if not sg_enabled:
        return f"controller={controller}"
    if sg_comb_mode == "controller":
        comb = f"controller:{sg_comb_variant}x{sg_comb_blocks}"
    else:
        comb = "linear" if sg_combine_reads else "none(no-combine)"
    return f"split-graph[backbone={sg_variant}x{sg_blocks} | combiner={comb}]"


def _first_nonfinite_report(tensor, name, step):
    """Silent when finite; prints once when not. Exists to bisect WHICH
    pipeline stage a NaN/Inf first appears at, since the interval-averaged
    training log only shows the aggregate, not the originating stage."""
    ok = torch.isfinite(tensor).all()
    if not ok:
        bad_frac = (~torch.isfinite(tensor)).float().mean().item()
        print(
            f"[NaN-TRACE] step {step}: non-finite in '{name}' ({bad_frac * 100:.2f}% of elements)"
        )
    return ok


def _capture_rng_state():
    """Capture all RNG streams used by task generation and model evaluation.

    Periodic dependency evaluations replay this state so the memory-on and
    memory-off scores use the same episodes, then restore the stream so the
    paired check does not perturb subsequent evaluation/training randomness.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _report_first_nonfinite_gradient(named_parameters, step):
    """Identify the first parameter whose unscaled gradient is non-finite."""
    for name, parameter in named_parameters:
        grad = parameter.grad
        if grad is not None and not torch.isfinite(grad).all():
            bad_fraction = (~torch.isfinite(grad)).float().mean().item()
            print(
                f"[NaN-TRACE] step {step}: first non-finite gradient in "
                f"'{name}' ({bad_fraction * 100:.2f}% of elements)"
            )
            return name
    print(
        f"[NaN-TRACE] step {step}: gradient norm is non-finite but no bad gradient tensor was found"
    )
    return None


def _format_moe_layer_reports(diagnostics: dict) -> list[str]:
    """Format per-bank routing/activation summaries for console output."""
    reports = []
    for layer_diag in diagnostics.get("moe_layers", []):
        load = ",".join(f"{value:.2f}" for value in layer_diag.get("expert_load_frac", []))
        importance = ",".join(
            f"{value:.2f}" for value in layer_diag.get("router_importance_frac", [])
        )
        valid_load = ",".join(
            f"{value:.2f}" for value in layer_diag.get("valid_expert_load_frac", [])
        )
        padded_load = ",".join(
            f"{value:.2f}" for value in layer_diag.get("padded_expert_load_frac", [])
        )
        valid_importance = ",".join(
            f"{value:.2f}" for value in layer_diag.get("valid_router_importance_frac", [])
        )
        padded_importance = ",".join(
            f"{value:.2f}" for value in layer_diag.get("padded_router_importance_frac", [])
        )
        source_routes = layer_diag.get("source_routes") or []
        source_summary = " ".join(
            f"src{i}:accepted={source.get('accepted_route_frac', 0.0):.1%},"
            f"drop={source.get('capacity_drop_frac', 0.0):.1%}"
            for i, source in enumerate(source_routes)
        )
        split_summary = ""
        if "valid_token_count" in layer_diag:
            valid_count = layer_diag.get("valid_token_count", 0.0)
            padded_count = layer_diag.get("padded_token_count", 0.0)
            split_summary = (
                f" valid[n={valid_count:.1f} load=[{valid_load}] imp=[{valid_importance}] "
                f"H={layer_diag.get('valid_router_entropy', 0.0):.2f} "
                f"pmax={layer_diag.get('valid_router_max_prob', 0.0):.2f} "
                f"drop={layer_diag.get('valid_capacity_drop_frac', 0.0):.1%}]"
                f" padded[n={padded_count:.1f} load=[{padded_load}] imp=[{padded_importance}] "
                f"H={layer_diag.get('padded_router_entropy', 0.0):.2f} "
                f"pmax={layer_diag.get('padded_router_max_prob', 0.0):.2f} "
                f"drop={layer_diag.get('padded_capacity_drop_frac', 0.0):.1%}]"
            )
        reports.append(
            f"moe-bank[{layer_diag['name']} "
            f"load=[{load}] importance=[{importance}] "
            f"entropy={layer_diag.get('router_entropy', 0.0):.2f} "
            f"pmax={layer_diag.get('router_max_prob', 0.0):.2f} "
            f"cv_load={layer_diag.get('cv_load', 0.0):.2f} "
            f"cv_imp={layer_diag.get('cv_importance', 0.0):.2f} "
            f"max_load={layer_diag.get('max_load_frac', 0.0):.2f} "
            f"topk_gate={layer_diag.get('topk_gate_mean', 0.0):.2f} "
            f"in_rms={layer_diag.get('input_rms', 0.0):.3g} "
            f"in_max={layer_diag.get('input_absmax', 0.0):.3g} "
            f"expert_rms/max={layer_diag.get('expert_output_rms', 0.0):.3g}/"
            f"{layer_diag.get('expert_output_absmax', 0.0):.3g} "
            f"moe_delta_rms/max={layer_diag.get('routed_output_rms', 0.0):.3g}/"
            f"{layer_diag.get('routed_output_absmax', 0.0):.3g} "
            f"drop={layer_diag.get('capacity_drop_frac', 0.0):.1%} "
            f"gate_mass={layer_diag.get('accepted_gate_mass', 0.0):.2f}"
            f"{split_summary}"
            f"{(' ' + source_summary) if source_summary else ''}]"
        )
    return reports


def _ogs_components(id_acc, ood_acc, ood_perfect, ood_memory_off, lesson):
    depth = max(int(lesson), 1)
    offset_normalized = max(0.0, 1.0 - abs(float(id_acc) - float(ood_acc)) / (100.0 * depth))
    components = {
        "offset_normalized": offset_normalized,
        "memory_attributable_ood": (
            (float(ood_acc) - float(ood_memory_off)) / 100.0 if ood_memory_off is not None else None
        ),
        "ood_perfect_frac": float(ood_perfect) / 100.0,
    }
    return components


def _ogs_score(components, weights):
    active = [(components[k], w) for k, w in zip(components, weights) if components[k] is not None]
    return sum(value * weight for value, weight in active) / max(sum(w for _, w in active), 1e-12)


def _write_ogs_row(writer, step, lesson, id_acc, ood_acc, ood_perfect, ood_memory_off, weights):
    components = _ogs_components(id_acc, ood_acc, ood_perfect, ood_memory_off, lesson)
    writer.writerow(
        [
            step,
            lesson,
            id_acc,
            ood_acc,
            components["offset_normalized"],
            components["ood_perfect_frac"],
            ood_acc,
            ood_memory_off,
            components["memory_attributable_ood"],
            *weights,
            _ogs_score(components, weights),
        ]
    )


def _print_ood_analysis(
    run_id,
    label,
    step,
    lesson,
    id_acc,
    ood_acc,
    ood_perfect,
    ood_memory_off,
    weights,
    beta,
    kl_loss,
    kl_diag,
):
    components = _ogs_components(id_acc, ood_acc, ood_perfect, ood_memory_off, lesson)
    memory_off = "n/a" if ood_memory_off is None else f"{ood_memory_off:.2f}%"
    memory_delta = components["memory_attributable_ood"]
    memory_delta_text = "n/a" if memory_delta is None else f"{memory_delta * 100:.2f} pp"
    print(
        f"[{run_id}] Step {step} {label} OOD attribution: "
        f"ID {id_acc:.2f}% -> OOD {ood_acc:.2f}% (offset {id_acc - ood_acc:.2f} pp; "
        f"normalized {components['offset_normalized']:.4f}) | "
        f"OOD perfect {ood_perfect:.2f}% | memory off {memory_off} | "
        f"memory-attributable OOD {memory_delta_text} | "
        f"OGS {_ogs_score(components, weights):.4f} "
        f"[offset {components['offset_normalized']:.4f}, "
        f"memory {memory_delta if memory_delta is not None else float('nan'):.4f}, "
        f"perfect {components['ood_perfect_frac']:.4f}]"
    )
    if beta > 0 and kl_diag:
        raw_kl = float(kl_loss.detach())
        print(
            f"[{run_id}] Step {step} KL attribution: beta {beta:.6g} | "
            f"L_KL {raw_kl:.6g} | beta*L_KL {beta * raw_kl:.6g} | "
            f"KL mean {kl_diag.get('kl_mean', float('nan')):.6g} | "
            f"clamp fraction {kl_diag.get('clamp_frac', float('nan')):.4f} | "
            f"prior snapshot {kl_diag.get('snapshot_step', 'n/a')}"
        )


def _capture_torch_rng():
    return (
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


@contextmanager
def _replay_torch_rng(state):
    """Replay evaluation randomness, then preserve the caller's advanced state."""
    advanced = _capture_torch_rng()
    torch.set_rng_state(state[0])
    if state[1] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state[1])
    try:
        yield
    finally:
        torch.set_rng_state(advanced[0])
        if advanced[1] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(advanced[1])


def save_checkpoint(
    path,
    rnn,
    output_proj,
    stochastic_heads,
    optimizer,
    curriculum,
    step,
    beta_target,
    run_id,
    scaler,
    ood_rng,
    controller_type=CONTROLLER_TYPE,
    link_matrix_mode=LINK_MATRIX_MODE,
    link_matrix_topk=LINK_MATRIX_TOPK,  # Static Option 1
    dynamic_n_mode=DYNAMIC_N_MODE,
    dynamic_n_state=None,  # Dynamic-N (macro-scale)
    moe_enabled=MOE_ENABLED,
    moe_num_experts=MOE_NUM_EXPERTS,
    moe_expert_dim=MOE_EXPERT_DIM,
    moe_capacity_factor=MOE_CAPACITY_FACTOR,
    moe_load_balance_alpha=MOE_LOAD_BALANCE_ALPHA,
    moe_top_k=MOE_TOP_K,
    split_graph_enabled=SPLIT_GRAPH_ENABLED,
    split_graph_variant=SPLIT_GRAPH_MAMBA_VARIANT,
    extra_model_config=None,
    sgr_prediction_decoder=None,
):
    """Save everything needed to resume training or re-run eval later:
    - model + output-projection + optimizer state
    - LR is NOT saved separately -- it's now a pure function of `step`
      (see lr_at_step()/set_lr() in run()), clamped at LR_DECAY_STEPS, so
      it's fully reconstructible from `step` alone on resume. This also
      sidesteps the CosineAnnealingLR periodicity bug (see chat log /
      Q21-Experiment-Log Section 5) that previously let LR drift back up
      after LR_DECAY_STEPS via the old scheduler's carried internal state.
    - the stochastic write head(s)' own params are already inside
      rnn.state_dict() (they were installed as submodules via the
      write_vector_transform swap), so no separate save needed for those
    - RNG state (python/numpy/torch/cuda) for reproducibility
    - curriculum lesson, so eval-time episode difficulty matches training
    - v3 fix: GradScaler state (scale, growth tracker, etc). Previously
      NOT saved -- `scaler = torch.amp.GradScaler(...)` was rebuilt fresh
      on every resume, throwing away whatever scale the prior leg had
      found and re-running a smaller version of the AMP overflow cascade
      right after every resume (see header note). `scaler` is now a
      required arg so this can't silently regress back to being dropped.
    - v5 (Phase 2): `prior_state` -- mu_g, Sigma_g (as prior_logvar), and
      last_snapshot_step for every installed stochastic write head (see
      get_prior_state() in stochastic_write_head.py). NOT covered by
      rnn.state_dict(): prior_mu/prior_logvar are registered buffers, so
      they normally WOULD ride along inside rnn.state_dict() automatically
      -- but they are captured here explicitly, as their own top-level
      checkpoint key, so this checkpoint's prior state is self-describing
      and independently loadable (e.g. by eval_from_checkpoint.py without
      needing to reload the entire rnn state_dict), and so a Phase-1-era
      reader that doesn't expect these buffers can still load
      rnn_state_dict unmodified. Mirrors the reasoning already applied to
      scaler_state_dict: model weights alone are not enough to reproduce
      this run's behavior on resume/re-eval.
    - v6: `ood_rng_state` -- the dedicated OOD-sampling random.Random
      instance's own state (via .getstate()), saved separately from the
      existing `rng_state` block (which only ever covered the global
      python/numpy/torch/cuda streams). Without this, a resumed run would
      silently re-seed ood_rng back to its start-of-process value
      (deterministic from `seed` alone -- see run()) instead of
      continuing the exact OOD walk sequence the pre-resume leg had
      reached, which is a smaller version of the same "silently evaluate
      against the wrong state" bug class prior_state (above) exists to
      catch -- it doesn't change the metric's validity (ood_rng was
      always decoupled from the training stream, which was the actual
      fix), but it does mean a resume's OOD trajectory isn't bit-for-bit
      continuous with the pre-resume leg unless this is restored. `ood_rng`
      is now a required arg so this can't silently regress the way the
      scaler-state omission originally did.
    - v7 (Alternate Phase 3, Step 1): `model_config` gained `controller_type`
      and (only when it's "mamba") `mamba_d_state`/`mamba_d_conv`/
      `mamba_expand`. Same "self-describing checkpoint" rationale as
      prior_state/scaler_state_dict above: an LSTM checkpoint's
      rnn_state_dict has `lstm_layer_0...` keys, a Mamba checkpoint's has
      `mamba_layer_0...` keys (see mamba_controller.py's MambaDNC), and
      without this field a reader has no way to know which model class to
      reconstruct before attempting `load_state_dict`.
    - Dynamic-N (macro-scale): `model_config["dynamic_n_mode"]`, same
      self-describing-checkpoint rationale as controller_type/
      link_matrix_mode above -- records WHICH mechanism was responsible
      for whatever nr_cells this checkpoint's rnn.memories[0].nr_cells
      already self-describes (curriculum lookup vs. usage-triggered
      growth), even though the resulting nr_cells value itself doesn't
      care which mechanism produced it. Also `dynamic_n_state` (top-level
      key, not inside model_config, same convention as prior_state /
      scaler_state_dict): DynamicNController's usage-EMA, cooldown
      counter, and full growth_history -- this is run *state*, not model
      architecture, so it lives alongside prior_state/ood_rng_state
      rather than inside model_config. None when this run isn't in
      dynamic_n_mode.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_config = {
        "input_size": rnn.input_size,
        "hidden_size": MODEL_HIDDEN_SIZE,
        "num_hidden_layers": getattr(rnn, "num_hidden_layers", None),
        "nr_cells": rnn.memories[0].nr_cells,
        "cell_size": MODEL_CELL_SIZE,
        "read_heads": MODEL_READ_HEADS,
        "controller_type": controller_type,  # v7
        "link_matrix_mode": link_matrix_mode,
        "link_matrix_topk": link_matrix_topk,  # Static Option 1
        "dynamic_n_mode": dynamic_n_mode,  # Dynamic-N (macro-scale)
        "moe_enabled": moe_enabled,
        "moe_num_experts": moe_num_experts,  # v8 (Option 4)
        "moe_expert_dim": moe_expert_dim,
        "moe_capacity_factor": moe_capacity_factor,
        "moe_load_balance_alpha": moe_load_balance_alpha,
        "moe_top_k": moe_top_k,
        "split_graph_enabled": split_graph_enabled,  # v9 (Option 5)
        "split_graph_variant": split_graph_variant,
    }
    if controller_type == "mamba":  # v7
        model_config.update(
            {
                "mamba_d_state": MAMBA_D_STATE,
                "mamba_d_conv": MAMBA_D_CONV,
                "mamba_expand": MAMBA_EXPAND,
            }
        )
    elif controller_type == "mamba2":
        model_config.update(
            {
                "mamba2_d_state": MAMBA2_D_STATE,
                "mamba2_d_conv": MAMBA2_D_CONV,
                "mamba2_expand": MAMBA2_EXPAND,
                "mamba2_headdim": MAMBA2_HEADDIM,
                "mamba2_ngroups": MAMBA2_NGROUPS,
            }
        )
    elif controller_type == "mamba3":
        model_config.update(
            {
                "mamba3_d_state": MAMBA3_D_STATE,
                "mamba3_expand": MAMBA3_EXPAND,
                "mamba3_headdim": MAMBA3_HEADDIM,
                "mamba3_rope_fraction": MAMBA3_ROPE_FRACTION,
            }
        )
    elif controller_type == "cfc":
        model_config.update(
            {
                "cfc_mode": CFC_MODE,
                "cfc_backbone_units": CFC_BACKBONE_UNITS,
                "cfc_backbone_layers": CFC_BACKBONE_LAYERS,
                "cfc_backbone_dropout": CFC_BACKBONE_DROPOUT,
                "cfc_activation": CFC_ACTIVATION,
                "cfc_mixed_memory": CFC_MIXED_MEMORY,
                "cfc_residual": CFC_RESIDUAL,
            }
        )
    elif "+" in controller_type:  # v14: hybrid chains, e.g. "mamba+cfc"
        model_config.update(
            {
                "mamba_d_state": MAMBA_D_STATE,
                "mamba_d_conv": MAMBA_D_CONV,
                "mamba_expand": MAMBA_EXPAND,
                "mamba2_d_state": MAMBA2_D_STATE,
                "mamba2_d_conv": MAMBA2_D_CONV,
                "mamba2_expand": MAMBA2_EXPAND,
                "mamba2_headdim": MAMBA2_HEADDIM,
                "mamba2_ngroups": MAMBA2_NGROUPS,
                "mamba3_d_state": MAMBA3_D_STATE,
                "mamba3_expand": MAMBA3_EXPAND,
                "mamba3_headdim": MAMBA3_HEADDIM,
                "mamba3_rope_fraction": MAMBA3_ROPE_FRACTION,
                "cfc_mode": CFC_MODE,
                "cfc_backbone_units": CFC_BACKBONE_UNITS,
                "cfc_backbone_layers": CFC_BACKBONE_LAYERS,
                "cfc_backbone_dropout": CFC_BACKBONE_DROPOUT,
                "cfc_activation": CFC_ACTIVATION,
                "cfc_mixed_memory": CFC_MIXED_MEMORY,
                "cfc_residual": CFC_RESIDUAL,
                "hybrid_cfc_num_blocks": HYBRID_CFC_NUM_BLOCKS,
            }
        )

    if extra_model_config:  # v19: dataset type / dims / split-graph details, read by inference/
        model_config.update(extra_model_config)

    checkpoint_payload = {
        "step": step,
        "run_id": run_id,
        "beta_target": beta_target,
        "model_config": model_config,
        "rnn_state_dict": rnn.state_dict(),
        "output_proj_state_dict": output_proj.state_dict(),
        "sgr_prediction_decoder_state_dict": (
            sgr_prediction_decoder.state_dict() if sgr_prediction_decoder is not None else None
        ),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),  # v3 fix
        "curriculum_lesson": curriculum.lesson,
        "prior_state": get_prior_state(stochastic_heads),  # v5 (Phase 2)
        "ood_rng_state": ood_rng.getstate(),  # v6
        "dynamic_n_state": dynamic_n_state,  # Dynamic-N (macro-scale); None if not dynamic_n_mode
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(checkpoint_payload, path)


def load_checkpoint_for_resume(path, device):
    """Load a checkpoint saved by save_checkpoint(), for CONTINUING training
    (as opposed to eval_from_checkpoint.py, which loads for eval only and
    doesn't need optimizer/scheduler/RNG state).

    map_location='cpu' (not `device`): torch.load's map_location moves
    EVERY tensor in the checkpoint, including the RNG-state ByteTensors.
    torch.cuda.set_rng_state_all() requires those to stay plain CPU
    ByteTensors -- if map_location drags them onto CUDA they become
    torch.cuda.ByteTensor and set_rng_state_all rejects them. Loading to
    CPU and letting rnn.load_state_dict()/optimizer.load_state_dict() do
    their own (automatic) device casting for the model/optimizer tensors
    avoids that without needing two different map_locations for one file.

    weights_only=False: this checkpoint stores non-tensor python/numpy RNG
    state alongside the tensors, and PyTorch >=2.6 defaults torch.load to
    weights_only=True. Safe here since it's a checkpoint we produced
    ourselves, not a downloaded/untrusted file.
    """
    return torch.load(path, map_location="cpu", weights_only=False)


def restore_rng_state(rng_state):
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(
        rng_state["torch"].cpu() if torch.is_tensor(rng_state["torch"]) else rng_state["torch"]
    )
    if rng_state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["cuda"])


def load_frozen_workspace_warm_start(
    rnn: nn.Module,
    output_proj: nn.Module,
    checkpoint: dict,
    expected_model_config: dict,
) -> list[str]:
    """Load a workspace-disabled baseline into a frozen-workspace model.

    Only the newly introduced workspace state may be absent. Every shared
    tensor must match by name and shape, and the recorded architecture must
    match outside workspace settings and the curriculum-dependent memory
    size. Returns the target workspace keys left at their fresh initialization.
    """
    checkpoint_config = checkpoint.get("model_config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError(
            "Frozen-workspace warm start requires a checkpoint with model_config metadata."
        )
    if checkpoint_config.get("workspace_enabled", False) or any(
        key.startswith("workspace.") or ".workspace." in key
        for key in checkpoint.get("rnn_state_dict", {})
    ):
        raise ValueError(
            "Frozen-workspace warm start expects a workspace-disabled baseline checkpoint."
        )

    ignored_config_keys = {
        "nr_cells",  # resized to the checkpoint size for load, then back to lesson 1
        "dnc_backward_trace",  # diagnostic instrumentation, not architecture
        "workspace_enabled",
        "workspace_num_slots",
        "workspace_no_selection",
        "workspace_reset_each_step",
        "workspace_train_mode",
        "workspace_train_scope",
        "workspace_gate_variant",
        "workspace_bistable_selection_scale",
        "workspace_bistable_noise_scale",
        "workspace_bistable_gradient_mode",
        "workspace_bptt_horizon",
        "workspace_reconsideration_enabled",
        "workspace_reconsideration_surprise_threshold",
        "workspace_reconsideration_trigger_mode",
        "workspace_bistable_deterministic",
        "workspace_stability_max_activation",
        # Exact implementation-only optimization; old checkpoints default
        # to base and remain weight/state compatible with either backend.
        "dnc_implementation",
        "dnc_compile_memory",  # exact execution optimization; state-dict compatible
        "dnc_link_sharpness",  # parameter-free read-addressing experiment
        "dnc_allocation_topk",  # parameter-free allocation experiment
        "batch_size",  # training hyperparameter, not model architecture
    }
    missing_config = sorted(
        key
        for key in expected_model_config
        if key not in ignored_config_keys and key not in checkpoint_config
    )
    unexpected_config = sorted(
        key
        for key in checkpoint_config
        if key not in ignored_config_keys and key not in expected_model_config
    )
    mismatched_config = sorted(
        key
        for key, expected in expected_model_config.items()
        if key not in ignored_config_keys
        and key in checkpoint_config
        and checkpoint_config[key] != expected
    )
    if missing_config or unexpected_config or mismatched_config:
        details = []
        if missing_config:
            details.append(f"missing metadata: {', '.join(missing_config)}")
        if unexpected_config:
            details.append(f"unrecognized metadata: {', '.join(unexpected_config)}")
        if mismatched_config:
            details.append(
                "configuration differs: "
                + ", ".join(
                    f"{key}={checkpoint_config[key]!r} (expected {expected_model_config[key]!r})"
                    for key in mismatched_config
                )
            )
        raise ValueError(
            "Checkpoint is not a compatible frozen-workspace baseline: " + "; ".join(details)
        )

    checkpoint_state = checkpoint.get("rnn_state_dict")
    if not isinstance(checkpoint_state, dict):
        raise ValueError("Checkpoint does not contain an rnn_state_dict.")
    target_state = rnn.state_dict()
    missing_keys = set(target_state) - set(checkpoint_state)
    unexpected_keys = set(checkpoint_state) - set(target_state)

    def is_workspace_key(key: str) -> bool:
        return key == "workspace" or key.startswith("workspace.") or ".workspace." in key

    missing_workspace_keys = sorted(key for key in missing_keys if is_workspace_key(key))
    missing_non_workspace = sorted(missing_keys - set(missing_workspace_keys))
    if missing_non_workspace or unexpected_keys:
        raise ValueError(
            "Checkpoint state does not match the frozen-workspace model. "
            f"Missing non-workspace keys={missing_non_workspace[:8]}, "
            f"unexpected keys={sorted(unexpected_keys)[:8]}"
        )
    shape_mismatches = [
        f"{key}: checkpoint{tuple(checkpoint_state[key].shape)} != model{tuple(target_state[key].shape)}"
        for key in target_state.keys() & checkpoint_state.keys()
        if torch.is_tensor(target_state[key])
        and torch.is_tensor(checkpoint_state[key])
        and target_state[key].shape != checkpoint_state[key].shape
    ]
    if shape_mismatches:
        raise ValueError(
            "Checkpoint has incompatible tensor shapes: " + "; ".join(shape_mismatches[:8])
        )

    load_result = rnn.load_state_dict(checkpoint_state, strict=False)
    if load_result.unexpected_keys or sorted(load_result.missing_keys) != missing_workspace_keys:
        raise RuntimeError(
            "Unexpected state mismatch while loading frozen-workspace baseline: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )
    output_proj.load_state_dict(checkpoint["output_proj_state_dict"])
    return missing_workspace_keys


# ==========================================
#              TRAINING LOOP
# ==========================================
def run(
    beta_target: float,
    run_id: str,
    seed: int = SEED,
    resume_from: str | None = None,
    controller: str = CONTROLLER_TYPE,  # v7 (Alternate Phase 3, Step 1)
    beta_mode: str = BETA_MODE,  # dynamic-beta toggle: "static" or "dynamic"
    total_steps: int = TOTAL_STEPS,  # see --total-steps.
    checkpoint_every: int = CHECKPOINT_EVERY,  # see --checkpoint-every.
    log_every: int = LOG_EVERY,  # see --log-every.
    link_matrix_mode: str = LINK_MATRIX_MODE,  # Static Option 1
    link_matrix_topk: int | None = LINK_MATRIX_TOPK,  # Static Option 1
    isolate_link_ablation: bool = ISOLATE_LINK_ABLATION,  # Static Option 1
    dynamic_n_mode: bool = DYNAMIC_N_MODE,  # Dynamic-N (macro-scale)
    dynamic_n_floor: int = DYNAMIC_N_FLOOR,
    dynamic_n_ceiling: int = DYNAMIC_N_CEILING,
    dynamic_n_trigger_frac: float = DYNAMIC_N_TRIGGER_FRAC,
    dynamic_n_cooldown_steps: int = DYNAMIC_N_COOLDOWN_STEPS,
    moe_enabled: bool = MOE_ENABLED,
    moe_diagnostics_enabled: bool = True,
    segment_diagnostics_enabled: bool = True,
    split_graph_moe_placement: str = "all",
    split_graph_moe_backbone_block_isolation: int | None = None,
    split_graph_moe_backbone_residual_scale: float = 1.0,
    moe_num_experts: int = MOE_NUM_EXPERTS,
    moe_expert_dim: int | None = MOE_EXPERT_DIM,
    moe_capacity_factor: float = MOE_CAPACITY_FACTOR,
    moe_load_balance_alpha: float = MOE_LOAD_BALANCE_ALPHA,
    moe_top_k: int = MOE_TOP_K,
    split_graph_combiner_cfc_multi_source_moe: bool = SPLIT_GRAPH_COMBINER_CFC_MULTI_SOURCE_MOE,
    split_graph_enabled: bool = SPLIT_GRAPH_ENABLED,
    split_graph_backbone_only: bool = False,
    split_graph_variant: str = SPLIT_GRAPH_MAMBA_VARIANT,
    split_graph_num_blocks: int = SPLIT_GRAPH_NUM_BLOCKS,
    split_graph_headdim: int = SPLIT_GRAPH_MAMBA_HEADDIM,
    split_graph_combine_reads: bool = SPLIT_GRAPH_COMBINE_READS,
    split_graph_combiner_mode: str = SPLIT_GRAPH_COMBINER_MODE,
    split_graph_combiner_variant: str = SPLIT_GRAPH_COMBINER_VARIANT,
    split_graph_combiner_num_blocks: int = SPLIT_GRAPH_COMBINER_NUM_BLOCKS,
    workspace_enabled: bool = WORKSPACE_ENABLED,
    workspace_num_slots: int = WORKSPACE_NUM_SLOTS,
    workspace_no_selection: bool = WORKSPACE_NO_SELECTION,
    workspace_reset_each_step: bool = WORKSPACE_RESET_EACH_STEP,
    workspace_train_mode: str = WORKSPACE_TRAIN_MODE,
    workspace_train_scope: str = "workspace",
    workspace_gate_variant: str = "continuous",
    workspace_bistable_selection_scale: float = 1.0,
    workspace_bistable_noise_scale: float = 0.75,
    workspace_bistable_gradient_mode: str = "truncated",
    workspace_bistable_debug: bool = False,
    workspace_bistable_anomaly_detection: bool = False,
    workspace_bistable_anomaly_step: int | None = None,
    workspace_bptt_horizon: int = 0,
    workspace_reconsideration_enabled: bool = False,
    workspace_reconsideration_surprise_threshold: float = 0.5,
    workspace_reconsideration_trigger_mode: str = "innovation",
    workspace_bistable_deterministic: bool = False,
    workspace_stability_max_activation: float = 100_000.0,
    eval_only: bool = False,
    eval_workspace_mode: str = "checkpoint",
    eval_seed: int | None = None,
    dnc_read_variant: str = DNC_READ_VARIANT,
    dnc_read_residual_scale: float = DNC_READ_RESIDUAL_SCALE,
    dnc_read_residual_max_ratio: float = DNC_READ_RESIDUAL_MAX_RATIO,
    dnc_write_variant: str = DNC_WRITE_VARIANT,
    dnc_write_observation_variance: float = DNC_WRITE_OBSERVATION_VARIANCE,
    dnc_link_sharpness: float = 1.0,
    dnc_allocation_topk: int | None = None,
    dnc_implementation: str = "base",
    memory_type: str = "dnc",
    memory_heads: int = 4,
    memory_dk: int = 64,
    memory_dv: int = 64,
    memory_read_lag: int = 1,
    memory_topk: int | None = None,
    memory_position: str = "none",
    delta_gate: bool = True,
    delta_chunk: int = 64,
    delta_use_fla: bool = False,
    dnc_compile_memory: bool = False,
    dnc_backward_trace: bool = False,
    dnc_backward_trace_step: int | None = None,
    split_graph_numeric_trace_step: int | None = None,
    split_graph_numeric_trace_on_error: bool = False,
    dataset_type: str = DATASET_TYPE,
    dataset_link: str | None = DATASET_LINK,
    test_dataset_link: str | None = None,
    prepared_dataset=None,
    ogs_weights: tuple[float, float, float] = OGS_WEIGHTS,
    batch_size: int = BATCH_SIZE,
    sgr_prediction_loss_weight: float = 0.0,
    sgr_prediction_eval_episodes: int = 128,
    sgr_phase2_arm: str | None = None,
    sgr_phase2_signal: str = "excess-surprise",
    sgr_gate_eval_override: str = "none",
    sgr_dnc_write_salience: bool = False,
    sgr_workspace_write_salience: bool = False,
):
    if log_every <= 0:
        raise ValueError("log_every must be positive")
    if memory_type not in {"none", "dnc", "kv", "delta"}:
        raise ValueError("memory_type must be none, dnc, kv, or delta")
    if memory_read_lag not in (0, 1):
        raise ValueError("memory_read_lag must be 0 or 1")
    if memory_position not in {"none", "rope"}:
        raise ValueError("memory_position must be none or rope")
    if memory_position != "none" and memory_type != "kv":
        raise ValueError("memory_position is only supported by KV memory")
    if memory_type != "dnc" and not split_graph_enabled:
        raise ValueError("non-DNC memory types require --split-graph")
    if split_graph_backbone_only and memory_type != "dnc":
        raise ValueError("--split-graph-backbone-only cannot be combined with a memory module")
    if memory_type != "dnc" and dynamic_n_mode:
        raise ValueError("--dynamic-n-mode applies only to DNC memory")
    # Deliberately does NOT touch LR_DECAY_STEPS - that's a separate
    # module-level constant, fixed at import time from the *original*
    # TOTAL_STEPS, and lr_at_step()/set_lr() below read it directly by
    # name, not through this parameter. A pilot run still anneals LR on
    # the full 120000-step schedule and simply stops early partway
    # through it, exactly as the --total-steps help text promises.
    dataset = get_dataset(
        dataset_type,
        dataset_link,
        test_dataset_link=test_dataset_link,
        prepared_dataset=prepared_dataset,
    )
    if split_graph_numeric_trace_step is not None:
        if not split_graph_enabled:
            raise ValueError("--split-graph-numeric-trace-step requires --split-graph")
        if eval_only:
            raise ValueError("--split-graph-numeric-trace-step requires a training run")
        if split_graph_numeric_trace_step < 0:
            raise ValueError("--split-graph-numeric-trace-step must be >= 0")
    if split_graph_numeric_trace_on_error and not split_graph_enabled:
        raise ValueError("--split-graph-numeric-trace-on-error requires --split-graph")
    if split_graph_numeric_trace_on_error and eval_only:
        raise ValueError("--split-graph-numeric-trace-on-error requires a training run")
    if split_graph_numeric_trace_on_error and split_graph_numeric_trace_step is not None:
        raise ValueError(
            "--split-graph-numeric-trace-on-error cannot be combined with "
            "--split-graph-numeric-trace-step"
        )
    sgr_prediction_enabled = sgr_prediction_loss_weight > 0.0
    sgr_phase4_enabled = (
        sgr_dnc_write_salience
        or sgr_workspace_write_salience
        or (workspace_reconsideration_enabled and sgr_prediction_enabled)
    )
    if not math.isfinite(sgr_prediction_loss_weight) or sgr_prediction_loss_weight < 0.0:
        raise ValueError("sgr_prediction_loss_weight must be finite and >= 0")
    if sgr_prediction_enabled:
        if getattr(dataset, "name", None) != "surprise-recall":
            raise ValueError("--sgr-pred-loss-weight requires --dataset-type surprise-recall")
        if not split_graph_enabled:
            raise ValueError("Phase 1 prediction loss requires --split-graph backbone features")
        if (moe_enabled or workspace_enabled) and not sgr_phase4_enabled:
            raise ValueError("Phase 1 signal isolation requires MoE and workspace to be disabled")
        if sgr_prediction_eval_episodes < 1:
            raise ValueError("sgr_prediction_eval_episodes must be positive")
    # Resolve the split-graph Mamba hyperparameters before model selection.
    # Phase 2 builds its dedicated backbone-only model in the first branch
    # below, so it does not enter the SplitGraphDNC branch where these values
    # were previously assigned. Run metadata still serializes them for both
    # model types, and Phase 2 needs the same variant-matched values.
    _sg_d_state, _sg_d_conv, _sg_expand = (
        (MAMBA3_D_STATE, MAMBA_D_CONV, MAMBA3_EXPAND)
        if split_graph_variant.startswith("mamba3")
        else (MAMBA2_D_STATE, MAMBA2_D_CONV, MAMBA2_EXPAND)
        if split_graph_variant.startswith("mamba2")
        else (MAMBA_D_STATE, MAMBA_D_CONV, MAMBA_EXPAND)
    )

    # The selected architecture is configured through the CLI and may be a
    # DNC, SplitGraphDNC, or a backbone-only module with different attributes.
    rnn: Any
    if sgr_phase2_arm is not None:
        valid_phase2_arms = {"content", "aux-only", "error", "blend", "shuffled", "oracle"}
        if sgr_phase2_arm not in valid_phase2_arms:
            raise ValueError(f"sgr_phase2_arm must be one of {sorted(valid_phase2_arms)}")
        if getattr(dataset, "name", None) != "surprise-recall":
            raise ValueError("Phase 2 arms require --dataset-type surprise-recall")
        if (
            not split_graph_enabled
            or split_graph_variant != "mamba1"
            or split_graph_num_blocks != 2
        ):
            raise ValueError("Phase 2 requires --split-graph-variant mamba1 and exactly 2 blocks")
        if moe_enabled or workspace_enabled or dnc_compile_memory or dnc_backward_trace:
            raise ValueError(
                "Phase 2 requires a backbone-only run with MoE/workspace/DNC tools off"
            )
        if dynamic_n_mode or link_matrix_mode != "dense" or isolate_link_ablation:
            raise ValueError("Phase 2 does not support memory sizing or link-matrix ablations")
        if sgr_phase2_signal not in {"surprisal", "excess-surprise"}:
            raise ValueError("sgr_phase2_signal must be surprisal or excess-surprise")
        if sgr_gate_eval_override not in {"none", "zero", "shuffle"}:
            raise ValueError("sgr_gate_eval_override must be none, zero, or shuffle")
        needs_decoder = sgr_phase2_arm in {"aux-only", "error", "blend", "shuffled"}
        if needs_decoder != sgr_prediction_enabled:
            raise ValueError(
                "Phase 2 arms aux-only/error/blend/shuffled require --sgr-pred-loss-weight > 0; "
                "content/oracle arms require it to be 0"
            )
    if split_graph_backbone_only:
        if not split_graph_enabled:
            raise ValueError("split_graph_backbone_only requires split_graph_enabled")
        if sgr_phase2_arm is not None:
            raise ValueError(
                "The Phase 2 SGR arms use their dedicated backbone-only model; "
                "omit split_graph_backbone_only"
            )
        if (
            dynamic_n_mode
            or link_matrix_mode != "dense"
            or isolate_link_ablation
            or dnc_compile_memory
            or dnc_backward_trace
            or dnc_read_variant != "cosine"
            or dnc_write_variant != "learned"
            or dnc_read_residual_scale != 1.0
            or dnc_read_residual_max_ratio != 0.5
            or dnc_write_observation_variance != 1.0
            or dnc_link_sharpness != 1.0
            or dnc_allocation_topk is not None
            or dnc_implementation != "base"
            or sgr_dnc_write_salience
        ):
            raise ValueError(
                "DNC-specific options (memory sizing, addressing, write/read variants, "
                "DNC traces/compilation, and DNC write salience) cannot be used with "
                "split_graph_backbone_only"
            )
        if split_graph_combiner_mode != "linear":
            raise ValueError(
                "split_graph_backbone_only has no sequential read combiner; "
                "use split_graph_combiner_mode='linear'"
            )
    if sgr_phase4_enabled:
        if getattr(dataset, "name", None) != "surprise-recall":
            raise ValueError("Phase 4 error consumers require --dataset-type surprise-recall")
        if not split_graph_enabled or split_graph_variant != "mamba1" or split_graph_num_blocks < 2:
            raise ValueError("Phase 4 SGR consumers require a split-graph Mamba-1 backbone")
        if not sgr_prediction_enabled:
            raise ValueError("Phase 4 error consumers require --sgr-pred-loss-weight > 0")
        if sgr_dnc_write_salience and dnc_compile_memory:
            raise ValueError(
                "DNC compiled-forward mode does not include the Phase 4 prediction-error path"
            )
        if sgr_workspace_write_salience and not workspace_enabled:
            raise ValueError("workspace write salience requires --workspace")
        if sgr_workspace_write_salience and workspace_num_slots < 1:
            raise ValueError("workspace write salience requires at least one workspace slot")
        if (
            workspace_reconsideration_enabled
            and workspace_reconsideration_trigger_mode.startswith("prediction_error")
            and workspace_num_slots < 1
        ):
            raise ValueError(
                "prediction-error reconsideration requires at least one workspace slot"
            )
        if sgr_dnc_write_salience and workspace_enabled and workspace_train_mode == "frozen":
            raise ValueError(
                "DNC write salience cannot be learned with a frozen-workspace run; "
                "use workspace_train_mode='coadapted'"
            )
        if sgr_phase2_arm is not None:
            raise ValueError("Phase 2 backbone-only arms cannot be combined with Phase 4 consumers")
    has_dnc = (
        not split_graph_backbone_only and sgr_phase2_arm is None and memory_type == "dnc"
    )
    periodic_ood_episodes = OOD_EVAL_EPISODES_PERIODIC
    final_ood_episodes = OOD_EVAL_EPISODES
    INPUT_DIM, TRIPLE_DIM = dataset.input_dim, dataset.output_dim
    beta_ctrl_acc_target = (
        BETA_CTRL_ACC_TARGET if BETA_CTRL_ACC_TARGET is not None else dataset.advance_threshold
    )

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"run_{run_id}.csv")
    resume_requested = resume_from is not None
    if eval_only and not resume_requested:
        raise ValueError("eval_only requires --resume with a checkpoint")
    if eval_workspace_mode not in ("checkpoint", "reset-each-step", "bypass"):
        raise ValueError("eval_workspace_mode must be 'checkpoint', 'reset-each-step', or 'bypass'")
    if not math.isfinite(workspace_reconsideration_surprise_threshold) or not (
        0.0 <= workspace_reconsideration_surprise_threshold <= 1.0
    ):
        raise ValueError(
            "workspace_reconsideration_surprise_threshold must be finite and in [0, 1]"
        )
    if workspace_reconsideration_trigger_mode not in (
        "innovation",
        "random_rate_matched",
        "prediction_error",
        "prediction_error_random_rate_matched",
    ):
        raise ValueError("unknown workspace reconsideration trigger mode")
    if eval_only and eval_workspace_mode != "checkpoint" and not workspace_enabled:
        raise ValueError("workspace evaluation modes require --workspace")
    resume_checkpoint = (
        load_checkpoint_for_resume(resume_from, device) if resume_requested else None
    )
    checkpoint_config = (
        (resume_checkpoint.get("model_config") or {}) if resume_checkpoint is not None else {}
    )
    checkpoint_state = (
        (resume_checkpoint.get("rnn_state_dict") or {}) if resume_checkpoint is not None else {}
    )
    checkpoint_has_workspace = any(
        key.startswith("workspace.") or ".workspace." in key for key in checkpoint_state
    )
    checkpoint_workspace_enabled = bool(
        checkpoint_config.get("workspace_enabled", False) or checkpoint_has_workspace
    )
    phase4_gate_control_warm_start = bool(
        resume_requested
        and workspace_enabled
        and workspace_train_scope == "gate"
        and checkpoint_workspace_enabled
        and (
            checkpoint_config.get("workspace_train_scope") != "gate"
            or checkpoint_config.get("workspace_train_mode") != workspace_train_mode
            or checkpoint_config.get("workspace_gate_variant", "continuous")
            != workspace_gate_variant
            or (
                workspace_gate_variant == "bistable"
                and checkpoint_config.get("workspace_bistable_gradient_mode", "unrolled")
                != workspace_bistable_gradient_mode
            )
        )
        and not eval_only
    )
    workspace_mode_warm_start = bool(
        resume_requested
        and workspace_enabled
        and checkpoint_workspace_enabled
        and checkpoint_config.get("workspace_train_mode") != workspace_train_mode
        and not eval_only
    )
    workspace_bptt_warm_start = bool(
        resume_requested
        and workspace_enabled
        and checkpoint_workspace_enabled
        and int(checkpoint_config.get("workspace_bptt_horizon", 0)) != workspace_bptt_horizon
        and not eval_only
    )
    workspace_reconsideration_warm_start = bool(
        resume_requested
        and workspace_enabled
        and checkpoint_workspace_enabled
        and (
            bool(checkpoint_config.get("workspace_reconsideration_enabled", False))
            != workspace_reconsideration_enabled
            or float(checkpoint_config.get("workspace_reconsideration_surprise_threshold", 0.5))
            != workspace_reconsideration_surprise_threshold
            or checkpoint_config.get("workspace_reconsideration_trigger_mode", "innovation")
            != workspace_reconsideration_trigger_mode
        )
        and not eval_only
    )
    gate_variant_warm_start = bool(
        resume_requested
        and workspace_enabled
        and workspace_gate_variant == "bistable"
        and checkpoint_workspace_enabled
        and checkpoint_config.get("workspace_gate_variant", "continuous") == "continuous"
        and not eval_only
    )
    workspace_condition_warm_start = bool(
        phase4_gate_control_warm_start
        or workspace_mode_warm_start
        or workspace_bptt_warm_start
        or workspace_reconsideration_warm_start
        or gate_variant_warm_start
    )
    frozen_workspace_warm_start = bool(
        resume_requested
        and workspace_enabled
        and workspace_train_mode == "frozen"
        and not checkpoint_workspace_enabled
    )
    if eval_only and frozen_workspace_warm_start:
        raise ValueError(
            "eval_only requires a checkpoint that already contains trained workspace weights; "
            "a workspace-disabled baseline would initialize a fresh workspace."
        )
    # A baseline-to-frozen-workspace warm start starts a new experiment: its
    # logs must be fresh, and it gets a fresh workspace-only optimizer.
    # Evaluation-only runs get independent, fresh logs keyed by their run ID;
    # they load checkpoint weights/state but must not append rows to the
    # original training logs.
    resuming = (
        resume_requested
        and not frozen_workspace_warm_start
        and not workspace_condition_warm_start
        and not eval_only
    )
    log_file = open(log_path, "a" if resuming else "w", newline="")
    log_writer = csv.writer(log_file)
    if not resuming:
        log_writer.writerow(
            [
                "step",
                "lesson",
                "beta_effective",
                "task_loss",
                "kl_loss",
                "kl_contrib",
                "total_loss",
                "digit_diversity",
                "kl_mean",
                "kl_max",
                "kl_min",
                "kl_std",
                "clamp_frac",
                "lr",
                "grad_norm",
                "amp_scale",
                "elapsed_sec",
                "snapshot_step",
                "gpu_mem_peak_mb",
                "param_count",
                "moe_aux_loss",
                "moe_cv_importance",
                "moe_cv_load",
                "moe_max_load_frac",
                "moe_capacity_drop_frac",
                "last_batch_reconsideration_eligible_frac",
                "last_batch_reconsideration_surprise_mean",
                "last_batch_reconsideration_trigger_frac",
                "last_batch_reconsideration_changed_frac",
                "last_batch_reconsideration_state_delta_rms",
                "sgr_prediction_loss",
                "sgr_error_signal_mean",
                "sgr_error_signal_abs_mean",
                "sgr_error_signal_positive_frac",
                "sgr_gate_signal_mean",
                "sgr_dnc_write_salience_coefficient",
                "sgr_workspace_write_scale_rms",
            ]
        )

    ood_log_path = os.path.join(LOG_DIR, f"run_{run_id}_ood.csv")
    ood_log_file = open(ood_log_path, "a" if resuming else "w", newline="")
    ood_log_writer = csv.writer(ood_log_file)
    if not resuming:
        ood_log_writer.writerow(
            [
                "step",
                "lesson",
                "id_triple_acc",
                "id_perfect_frac",
                "ood_triple_acc",
                "ood_perfect_frac",
                "ood_offset_triple",
                "ood_offset_perfect",
            ]
        )

    relational_probe_cases = (
        build_relational_probe(cell_size=MODEL_CELL_SIZE) if dataset_type == "graph" else None
    )
    relational_probe_file = None
    relational_probe_writer = None
    if relational_probe_cases is not None:
        relational_probe_file = open(
            os.path.join(LOG_DIR, f"run_{run_id}_relational_probe.csv"),
            "a" if resuming else "w",
            newline="",
        )
        relational_probe_writer = csv.writer(relational_probe_file)
        if not resuming:
            relational_probe_writer.writerow(
                [
                    "step",
                    "lesson",
                    "cosine_top1_accuracy",
                    "selected_read_top1_accuracy",
                    "selected_read_held_out_accuracy",
                    "cases",
                ]
            )

    track_a_file = None
    track_a_writer = None
    track_b_file = None
    track_b_writer = None
    if has_dnc and split_graph_enabled and dataset_type == "graph":
        track_a_file = open(
            os.path.join(LOG_DIR, f"run_{run_id}_track_a.csv"),
            "a" if resuming else "w",
            newline="",
        )
        track_a_writer = csv.writer(track_a_file)
        if not resuming:
            track_a_writer.writerow(
                [
                    "step",
                    "lesson",
                    "read_variant",
                    "id_triple_acc",
                    "ood_triple_acc",
                    "ood_offset_triple",
                    "stress_cosine_top1_acc",
                    "stress_selected_read_top1_acc",
                    "stress_held_out_acc",
                    "ordinary_read_score_cosine_rho_mean",
                    "ordinary_read_diagnostic_count",
                    "read_content_entropy_mean",
                    "read_content_max_weight_mean",
                    "relational_read_residual_rms_mean",
                    "relational_read_residual_to_cosine_rms_mean",
                ]
            )
        track_b_file = open(
            os.path.join(LOG_DIR, f"run_{run_id}_track_b.csv"),
            "a" if resuming else "w",
            newline="",
        )
        track_b_writer = csv.writer(track_b_file)
        if not resuming:
            track_b_writer.writerow(
                [
                    "event",
                    "step",
                    "lesson",
                    "write_variant",
                    "id_triple_acc",
                    "ood_triple_acc",
                    "ood_offset_triple",
                    "q99_gate",
                    "q99_tested_episodes",
                    "q99_tested_fact_episodes",
                    "q99_tested_query_episodes",
                    "q99_ordered_query_accuracy",
                    "q99_fact_permuted_accuracy",
                    "q99_fact_accuracy_delta",
                    "q99_fact_prediction_retention",
                    "q99_swapped_query_accuracy",
                    "q99_query_order_accuracy_drop",
                    "q99_query_prediction_flip_rate",
                    "q99_query_memory_incremental_order_effect",
                    "id_posterior_mean_variance",
                    "id_posterior_variance_min",
                    "id_posterior_variance_max",
                    "id_posterior_variance_reduction",
                    "id_posterior_mean_update_abs",
                    "id_active_write_cell_fraction",
                    "id_write_weight_mass",
                    "id_diagnostic_steps",
                    "ood_posterior_mean_variance",
                    "ood_posterior_variance_min",
                    "ood_posterior_variance_max",
                    "ood_posterior_variance_reduction",
                    "ood_posterior_mean_update_abs",
                    "ood_active_write_cell_fraction",
                    "ood_write_weight_mass",
                    "ood_diagnostic_steps",
                    "train_task_loss",
                    "train_kl_loss",
                    "train_total_loss",
                    "train_grad_norm",
                    "amp_scale",
                    "learning_rate",
                ]
            )

    ogs_log_path = os.path.join(LOG_DIR, f"run_{run_id}_ogs.csv")
    ogs_log_file = open(ogs_log_path, "a" if resuming else "w", newline="")
    ogs_log_writer = csv.writer(ogs_log_file)
    if not resuming:
        ogs_log_writer.writerow(
            [
                "step",
                "lesson",
                "id_acc",
                "ood_acc",
                "offset_normalized",
                "ood_perfect_frac",
                "ood_memory_on_acc",
                "ood_memory_off_acc",
                "memory_attributable_ood",
                "weight_offset",
                "weight_memory",
                "weight_perfect",
                "ogs",
            ]
        )

    # Functional-usage check (LB-9/Concept 6): same-cadence, same-distribution
    # companion to the ID/OOD log above, but with the model's own memory
    # path switched off (see evaluate_traversal's ablate_memory kwarg). A
    # standing per-run check, not a one-off sanity pass, per the roadmap's
    # explicit requirement -- catches rsDNC's silent bypass-path collapse
    # (still trains, still runs, loss curves look fine) that a KL- or
    # loss-only view can't see.
    mem_check_file: TextIO | None = None
    mem_check_writer: Any | None = None
    if has_dnc:
        mem_check_path = os.path.join(LOG_DIR, f"run_{run_id}_memory_dependency.csv")
        mem_check_file = open(mem_check_path, "a" if resuming else "w", newline="")
        mem_check_writer = csv.writer(mem_check_file)
        if not resuming:
            mem_check_writer.writerow(
                [
                    "step",
                    "lesson",
                    "id_triple_acc",
                    "id_perfect_frac",
                    "ablated_triple_acc",
                    "ablated_perfect_frac",
                    "memory_dependency_triple",
                    "memory_dependency_perfect",
                ]
            )

    combiner_stage_log_path = os.path.join(LOG_DIR, f"run_{run_id}_combiner_stage_dependency.csv")
    combiner_stage_log_file = open(combiner_stage_log_path, "a" if resuming else "w", newline="")
    combiner_stage_log_writer = csv.writer(combiner_stage_log_file)
    if not resuming:
        combiner_stage_log_writer.writerow(
            [
                "step",
                "lesson",
                "stage_index",
                "stage_kind",
                "id_triple_acc",
                "id_perfect_frac",
                "ablated_triple_acc",
                "ablated_perfect_frac",
                "stage_dependency_triple",
                "stage_dependency_perfect",
            ]
        )

    lesson_log_path = os.path.join(LOG_DIR, f"run_{run_id}_lesson_advances.csv")
    modality_log_path = os.path.join(LOG_DIR, f"run_{run_id}_modality_dependency.csv")
    modality_log_file = open(modality_log_path, "a" if resuming else "w", newline="")
    modality_log_writer = csv.writer(modality_log_file)
    if not resuming:
        modality_log_writer.writerow(
            [
                "step",
                "lesson",
                "modality",
                "id_acc",
                "id_perfect_frac",
                "ablated_acc",
                "ablated_perfect_frac",
                "modality_dependency_acc",
                "modality_dependency_perfect",
            ]
        )
    lesson_log_file = open(lesson_log_path, "a" if resuming else "w", newline="")
    lesson_log_writer = csv.writer(lesson_log_file)
    if not resuming:
        lesson_log_writer.writerow(["step", "new_lesson", "of_total_lessons"])

    # Static Option 1 watch-item: per-hop-count accuracy, logged as its own
    # trajectory instead of console-only. The link-matrix ablation/sparsify
    # finding this option validates (Session 009 synthesis) was established
    # on QA-style tasks, where each answer is effectively a 1-hop lookup and
    # content-based addressing dominates. Graph traversal chains multiple
    # hops per episode -- exactly where temporal/sequential-link addressing
    # could matter more than it does for QA. So the thing to watch isn't
    # whether aggregate triple_acc holds after ablating/sparsifying the link
    # matrix, it's whether accuracy holds *evenly across hop counts* or
    # collapses specifically at higher hops. hop_breakdown already computes
    # this at every advance-check; this file just persists it.
    hop_log_path = os.path.join(LOG_DIR, f"run_{run_id}_hop_breakdown.csv")
    hop_log_file = open(hop_log_path, "a" if resuming else "w", newline="")
    hop_log_writer = csv.writer(hop_log_file)
    if not resuming:
        hop_log_writer.writerow(
            [
                "step",
                "lesson",
                "eval_type",
                "hop_count",
                "triple_acc",
                "perfect_frac",
                "n_episodes",
            ]
        )

    # v5 (Phase 2): periodic "prior snapshot updated" log -- net-new file,
    # written every PRIOR_SNAPSHOT_EVERY steps by update_all_prior_snapshots()
    # below. This is the direct evidence for the Q48 periodic-snapshot
    # classification and for a collapse/drift check on the prior itself
    # (Sigma_g -> 0 would not show up in the existing kl_mean diagnostic,
    # which only ever describes q(v_t), never the prior it's compared
    # against). `snapshot_step` here is always == `step` (the update just
    # happened at this step); it's included anyway so this file's schema is
    # self-describing without cross-referencing the main log.
    prior_log_path = os.path.join(LOG_DIR, f"run_{run_id}_prior_snapshots.csv")
    prior_log_file = open(prior_log_path, "a" if resuming else "w", newline="")
    prior_log_writer = csv.writer(prior_log_file)
    if not resuming:
        prior_log_writer.writerow(
            [
                "step",
                "snapshot_step",
                "mu_g_norm",
                "sigma_g_mean",
                "sigma_g_min",
                "sigma_g_max",
                "trace_sigma_g",
                "n_samples",
                "raw_var_mean",
                "raw_var_max",
                "raw_hi_frac",
            ]
        )

    curriculum = dataset.make_curriculum()
    surprise_recall_logger = (
        SurpriseRecallLogger(
            LOG_DIR,
            run_id,
            append=resuming,
            phase1_enabled=sgr_prediction_enabled,
        )
        if getattr(dataset, "name", None) == "surprise-recall"
        else None
    )
    curriculum.advance_log_writer = lesson_log_writer  # log addition #4, read by maybe_advance
    curriculum.advance_log_file = lesson_log_file  # flushed after each write
    curriculum.hop_log_writer = hop_log_writer  # Static Option 1 watch-item
    curriculum.hop_log_file = hop_log_file

    field_log_path = os.path.join(LOG_DIR, f"run_{run_id}_field_breakdown.csv")
    field_log_file = open(field_log_path, "a" if resuming else "w", newline="")
    field_log_writer = csv.writer(field_log_file)
    if os.path.getsize(field_log_path) == 0:
        # v20: header is dataset-owned (see BaseDataset.field_log_header) instead of
        # hardcoded to graph's src/edge/dst schema, since text/audio/video/multimodal
        # now log their own value/cumsum breakdown into this same file.
        field_log_writer.writerow(["step", "lesson", "eval_type"] + dataset.field_log_header())
    curriculum.field_log_writer = field_log_writer
    curriculum.field_log_file = field_log_file

    # v5 (Phase 2): dedicated RNG stream for OOD (London Underground) walk
    # sampling, decoupled from the global `random` stream that curriculum ID
    # sampling uses. Constructed once per run (not re-seeded per call --
    # that was exactly the bug already fixed for
    # build_london_underground_eval()'s station/label mapping, see the v2
    # header note) and passed as `rng=` to every OOD evaluate_traversal()
    # call below. Seed is derived from this run's own seed so different
    # seeds in a panel still get different (but each internally
    # reproducible) OOD walk sequences, rather than all sharing one fixed
    # stream. Purely a cleanliness fix for the cross-seed comparisons Phase
    # 2 relies on -- see Section 2g in
    # Q21-Phase_1_Multi-seed_verification.md for the confound this removes.
    ood_rng = random.Random(seed * 1_000_003 + 17)

    starting_nr_cells = dynamic_n_floor if dynamic_n_mode else MODEL_NR_CELLS
    hidden_size, nr_cells, cell_size, read_heads = (
        MODEL_HIDDEN_SIZE,
        starting_nr_cells,
        MODEL_CELL_SIZE,
        MODEL_READ_HEADS,
    )

    print(
        f"[{run_id}] Model config: hidden={hidden_size} nr_cells={nr_cells} "
        f"cell_size={cell_size} read_heads={read_heads} | beta={beta_target} "
        f"| "
        + (
            f"split-graph[backbone={split_graph_variant}x{split_graph_num_blocks} | DNC=off]"
            if split_graph_backbone_only
            else _arch_desc(
                controller,
                split_graph_enabled,
                split_graph_variant,
                split_graph_num_blocks,
                split_graph_combiner_mode,
                split_graph_combiner_variant,
                split_graph_combiner_num_blocks,
                split_graph_combine_reads,
            )
        )
        + (f" | moe={moe_num_experts}e" if moe_enabled else "")
    )
    print(
        f"[{run_id}] diagnostics: moe={'on' if moe_diagnostics_enabled else 'off'} "
        f"segment={'on' if segment_diagnostics_enabled else 'off'}"
    )
    if moe_enabled:
        print(
            f"[{run_id}] MoE routing diagnostics="
            f"{'on' if moe_diagnostics_enabled else 'off'} | "
            "load-balancing auxiliary loss and routing remain active"
        )

    if isolate_link_ablation:
        curriculum.lesson_nr_cells = [MODEL_NR_CELLS] * len(curriculum.table)
        print(
            f"[{run_id}] isolate_link_ablation=True - nr_cells held fixed "
            f"at {MODEL_NR_CELLS} for the whole run (Option 2's resize "
            f"mechanism will never fire this run)."
        )
    if dynamic_n_mode:
        # Same isolation mechanism as isolate_link_ablation above (Concept
        # 16/SP-10): hold the curriculum-indexed lookup constant so
        # TraversalCurriculum.maybe_advance()'s own resize_memory() call
        # never fires -- DynamicNController is the only thing allowed to
        # resize nr_cells this run.
        curriculum.lesson_nr_cells = [dynamic_n_floor] * len(curriculum.table)
        print(
            f"[{run_id}] dynamic_n_mode=True - nr_cells starts at "
            f"{dynamic_n_floor} and grows via usage-triggered "
            f"DynamicNController (ceiling={dynamic_n_ceiling}, "
            f"trigger_frac={dynamic_n_trigger_frac}, "
            f"cooldown_steps={dynamic_n_cooldown_steps}); Option 2's "
            f"curriculum-indexed resize mechanism will never fire this run."
        )
        if isolate_link_ablation:
            print(
                f"[{run_id}] NOTE: isolate_link_ablation AND dynamic_n_mode "
                f"are both set -- these are two independent overrides of "
                f"LESSON_NR_CELLS (dynamic_n_mode's wins, since it's "
                f"checked second). Combining an isolated-Option-1 run "
                f"with Dynamic-N growth is unusual; make sure that's "
                f"actually what you meant to measure per Concept 16/SP-10."
            )

    if memory_type != "dnc":
        # Dataset curricula carry DNC cell-count schedules even when their
        # task is being run against a fixed-size/sequence memory. Prevent the
        # curriculum's generic advance hook from calling resize_memory on a
        # non-DNC module.
        curriculum.lesson_nr_cells = [nr_cells] * len(curriculum.table)
        print(
            f"[{run_id}] memory={memory_type}: DNC cell-resize schedule disabled; "
            f"reported nr_cells metadata held at {nr_cells}"
        )

    mamba_kwargs: dict[str, Any] = {}
    if controller == "mamba":
        mamba_kwargs = dict(
            mamba_d_state=MAMBA_D_STATE,
            mamba_d_conv=MAMBA_D_CONV,
            mamba_expand=MAMBA_EXPAND,
            moe_enabled=moe_enabled,
            moe_num_experts=moe_num_experts,
            moe_expert_dim=moe_expert_dim,
            moe_capacity_factor=moe_capacity_factor,
            moe_load_balance_alpha=moe_load_balance_alpha,
            moe_top_k=moe_top_k,
        )
    elif controller == "mamba2":
        mamba_kwargs = dict(
            mamba2_d_state=MAMBA2_D_STATE,
            mamba2_d_conv=MAMBA2_D_CONV,
            mamba2_expand=MAMBA2_EXPAND,
            mamba2_headdim=MAMBA2_HEADDIM,
            mamba2_ngroups=MAMBA2_NGROUPS,
            moe_enabled=moe_enabled,
            moe_num_experts=moe_num_experts,
            moe_expert_dim=moe_expert_dim,
            moe_capacity_factor=moe_capacity_factor,
            moe_load_balance_alpha=moe_load_balance_alpha,
            moe_top_k=moe_top_k,
        )

    if controller == "mamba3":
        mamba_kwargs = dict(
            mamba3_d_state=MAMBA3_D_STATE,
            mamba3_expand=MAMBA3_EXPAND,
            mamba3_headdim=MAMBA3_HEADDIM,
            mamba3_rope_fraction=MAMBA3_ROPE_FRACTION,
            moe_enabled=moe_enabled,
            moe_num_experts=moe_num_experts,
            moe_expert_dim=moe_expert_dim,
            moe_capacity_factor=moe_capacity_factor,
            moe_load_balance_alpha=moe_load_balance_alpha,
            moe_top_k=moe_top_k,
        )

    if controller == "cfc":
        mamba_kwargs = dict(
            cfc_mode=CFC_MODE,
            cfc_backbone_units=CFC_BACKBONE_UNITS,
            cfc_backbone_layers=CFC_BACKBONE_LAYERS,
            cfc_backbone_dropout=CFC_BACKBONE_DROPOUT,
            cfc_activation=CFC_ACTIVATION,
            cfc_mixed_memory=CFC_MIXED_MEMORY,
            cfc_residual=CFC_RESIDUAL,
            moe_enabled=moe_enabled,
            moe_top_k=moe_top_k,
        )

    if "+" in controller:  # v14: hybrid chain, e.g. "mamba+cfc"
        mamba_kwargs = dict(
            mamba_d_state=MAMBA_D_STATE,
            mamba_d_conv=MAMBA_D_CONV,
            mamba_expand=MAMBA_EXPAND,
            mamba2_d_state=MAMBA2_D_STATE,
            mamba2_d_conv=MAMBA2_D_CONV,
            mamba2_expand=MAMBA2_EXPAND,
            mamba2_headdim=MAMBA2_HEADDIM,
            mamba2_ngroups=MAMBA2_NGROUPS,
            mamba3_d_state=MAMBA3_D_STATE,
            mamba3_expand=MAMBA3_EXPAND,
            mamba3_headdim=MAMBA3_HEADDIM,
            mamba3_rope_fraction=MAMBA3_ROPE_FRACTION,
            cfc_mode=CFC_MODE,
            cfc_backbone_units=CFC_BACKBONE_UNITS,
            cfc_backbone_layers=CFC_BACKBONE_LAYERS,
            cfc_backbone_dropout=CFC_BACKBONE_DROPOUT,
            cfc_activation=CFC_ACTIVATION,
            cfc_mixed_memory=CFC_MIXED_MEMORY,
            cfc_residual=CFC_RESIDUAL,
            hybrid_cfc_num_blocks=HYBRID_CFC_NUM_BLOCKS,
            moe_enabled=moe_enabled,
            moe_top_k=moe_top_k,
        )

    if sgr_phase2_arm is not None:
        rnn = SurpriseRecallBackboneOnly(
            input_size=INPUT_DIM,
            hidden_size=hidden_size,
            nr_cells_metadata=nr_cells,
            variant="mamba1",
            d_state=MAMBA_D_STATE,
            d_conv=MAMBA_D_CONV,
            expand=MAMBA_EXPAND,
            delta_error_gating=True,
            device=device,
        ).to(device)
        rnn.phase2_arm = sgr_phase2_arm
        rnn.phase2_signal_mode = sgr_phase2_signal
        rnn.gate_eval_override = sgr_gate_eval_override
        print(
            f"[{run_id}] SGR Phase 2 arm={sgr_phase2_arm} signal={sgr_phase2_signal} "
            f"| backbone-only Mamba-1 x2 | DNC/MoE/workspace absent | "
            f"eval_gate_override={sgr_gate_eval_override}"
        )
    elif split_graph_enabled:
        if moe_enabled:
            print(
                f"[{run_id}] split-graph MoE placement={split_graph_moe_placement}: "
                f"backbone={'on' if split_graph_moe_placement in ('all', 'backbone') else 'off'}, "
                f"combiner={'on' if has_dnc and split_graph_moe_placement in ('all', 'combiner') else 'off'} "
                f"(see split_graph_dnc.py). A combiner_variant='cfc' additionally routes its two "
                f"natural input sources (backbone output, previous read vector) through "
                f"per-source specialized experts when active="
                f"{has_dnc and split_graph_moe_placement in ('all', 'combiner') and split_graph_combiner_cfc_multi_source_moe} "
                f"(see moe_layer.MultiSourceMoEBlock)."
            )
            if split_graph_moe_backbone_block_isolation is not None:
                routing_note = (
                    "Graph routing diagnostics split selected routes into valid and zero-padded timesteps."
                    if dataset_type == "graph"
                    else "Token-subset routing diagnostics are unavailable for this dataset."
                )
                print(
                    f"[{run_id}] backbone MoE block isolation={split_graph_moe_backbone_block_isolation} "
                    f"(zero-based per backbone stage; all other backbone blocks run without MoE). "
                    f"{routing_note}"
                )
            if split_graph_moe_backbone_residual_scale != 1.0:
                print(
                    f"[{run_id}] backbone MoE residual scale="
                    f"{split_graph_moe_backbone_residual_scale:g} | "
                    "applies to each backbone MoE residual, including the "
                    "broadcast-enriched final bank when workspace is active; "
                    "router, experts, and auxiliary routing loss remain active."
                )

        rnn = SplitGraphDNC(
            input_size=INPUT_DIM,
            hidden_size=hidden_size,
            nr_cells=nr_cells,
            cell_size=cell_size,
            read_heads=read_heads,
            num_backbone_blocks=split_graph_num_blocks,
            mamba_variant=split_graph_variant,
            mamba_d_state=_sg_d_state,
            mamba_d_conv=_sg_d_conv,
            mamba_expand=_sg_expand,
            mamba_headdim=split_graph_headdim,
            cfc_kwargs=dict(
                mode=CFC_MODE,
                backbone_units=CFC_BACKBONE_UNITS,
                backbone_layers=CFC_BACKBONE_LAYERS,
                backbone_dropout=CFC_BACKBONE_DROPOUT,
                activation=CFC_ACTIVATION,
                mixed_memory=CFC_MIXED_MEMORY,
                residual=CFC_RESIDUAL,
            ),
            combine_reads=split_graph_combine_reads,
            combiner_mode=split_graph_combiner_mode,
            combiner_variant=split_graph_combiner_variant,
            combiner_num_blocks=split_graph_combiner_num_blocks,
            independent_linears=True,
            moe_enabled=moe_enabled,
            moe_placement=split_graph_moe_placement,
            moe_backbone_block_isolation=split_graph_moe_backbone_block_isolation,
            moe_num_experts=moe_num_experts,
            moe_expert_dim=moe_expert_dim,
            moe_top_k=moe_top_k,
            moe_capacity_factor=moe_capacity_factor,
            moe_load_balance_alpha=moe_load_balance_alpha,
            moe_backbone_residual_scale=split_graph_moe_backbone_residual_scale,
            moe_cfc_multi_source=split_graph_combiner_cfc_multi_source_moe,
            dnc_enabled=not split_graph_backbone_only,
            memory_type=memory_type,
            memory_heads=memory_heads,
            memory_dk=memory_dk,
            memory_dv=memory_dv,
            memory_read_lag=memory_read_lag,
            memory_topk=memory_topk,
            memory_position=memory_position,
            delta_gate=delta_gate,
            delta_chunk=delta_chunk,
            delta_use_fla=delta_use_fla,
            dnc_read_variant=dnc_read_variant,
            dnc_read_residual_scale=dnc_read_residual_scale,
            dnc_read_residual_max_ratio=dnc_read_residual_max_ratio,
            dnc_write_variant=dnc_write_variant,
            dnc_write_observation_variance=dnc_write_observation_variance,
            dnc_link_sharpness=dnc_link_sharpness,
            dnc_allocation_topk=dnc_allocation_topk,
            dnc_implementation=dnc_implementation,
            sgr_write_salience_enabled=sgr_dnc_write_salience,
            sgr_workspace_salience_enabled=sgr_workspace_write_salience,
            workspace_enabled=workspace_enabled,
            workspace_num_slots=workspace_num_slots,
            workspace_no_selection=workspace_no_selection,
            workspace_reset_each_step=workspace_reset_each_step,
            workspace_train_mode=workspace_train_mode,
            workspace_gate_variant=workspace_gate_variant,
            workspace_bistable_selection_scale=workspace_bistable_selection_scale,
            workspace_bistable_noise_scale=workspace_bistable_noise_scale,
            workspace_bistable_gradient_mode=workspace_bistable_gradient_mode,
            workspace_bptt_horizon=workspace_bptt_horizon,
            workspace_reconsideration_enabled=workspace_reconsideration_enabled,
            workspace_reconsideration_surprise_threshold=(
                workspace_reconsideration_surprise_threshold
            ),
            workspace_reconsideration_trigger_mode=(workspace_reconsideration_trigger_mode),
            device=device,
        ).to(device)

        if workspace_bistable_debug:
            assert rnn.workspace is not None and rnn.workspace.bistable_gate is not None
            rnn.workspace.bistable_gate.debug_diagnostics = True
            print(
                f"[{run_id}] bistable deep diagnostics enabled | per-update forward "
                "conditioning, settling trajectory, raw gate gradients, clipping, "
                "parameter deltas and Adam moments will be printed"
            )

        if dnc_backward_trace and dnc_backward_trace_step is None:
            rnn.memories[0].set_backward_trace(True)
            print(
                f"[{run_id}] DNC backward trace enabled | watches usage, allocation, "
                "patched prod/cumprod, write/read addressing, temporal links, "
                "precedence, memory state and read vectors | forward equations unchanged"
            )
        print(
            f"[{run_id}] DNC compute precision=fp32 | memory interface, addressing, "
            "and recurrent state updates run outside CUDA autocast"
        )
    else:
        rnn = MambaDNC(
            input_size=INPUT_DIM,
            hidden_size=hidden_size,
            rnn_type=controller,  # v7: 'lstm' (default) or 'mamba'
            num_layers=1,
            nr_cells=nr_cells,
            cell_size=cell_size,
            read_heads=read_heads,
            batch_first=True,
            device=device,
            independent_linears=True,  # Phase 1: required so Memory exposes a
            # standalone write_vector_transform Linear for
            # install_stochastic_write_heads to swap out. Doesn't change the
            # addressing math - just which code path builds the (functionally
            # equivalent) per-head transforms. See header note + Section 5 of
            # Q21-Experiment-Log.md for why this makes Run 0 an anchor-by-
            # equivalence rather than a bit-identical beta=0 substitute.
            # v7: unaffected by the controller swap -- MambaDNC builds
            # dnc.memory.Memory identically regardless of rnn_type.
            dnc_implementation=dnc_implementation,
            **mamba_kwargs,
        ).to(device)

    if dnc_compile_memory and not isinstance(rnn, SurpriseRecallBackboneOnly):
        for memory in rnn.memories:
            compile_memory_forward(memory)
        print(
            f"[{run_id}] DNC memory forward compiled with torch.compile | "
            "first-call compilation overhead is expected"
        )

    split_graph_forward_trace = None
    if split_graph_numeric_trace_step is not None or split_graph_numeric_trace_on_error:
        if not isinstance(rnn, SplitGraphDNC):
            raise ValueError("split-graph numeric tracing requires SplitGraphDNC")
        split_graph_forward_trace = SplitGraphForwardTrace(
            LOG_DIR,
            run_id,
            split_graph_numeric_trace_step,
            trigger_on_error=split_graph_numeric_trace_on_error,
        )
        rnn._initium_forward_numeric_context = split_graph_forward_trace  # type: ignore[assignment]  # fmt: skip
        for module_name, module in rnn.named_modules():
            if module is rnn:
                prefix = "split_graph"
            elif module is rnn.backbone:
                prefix = "backbone"
            elif module.__class__.__name__ == "MambaBackboneParallel":
                prefix = module_name
            elif module.__class__.__name__ == "_ParallelBlock":
                prefix = module_name
            elif module.__class__.__name__ == "CfCControllerWrapper":
                prefix = "controller.cfc"
            elif module.__class__.__name__ == "CfCControllerBlock":
                prefix = module_name.replace("combiner_wrapper", "controller.cfc")
            else:
                continue

            def make_observer(module_prefix: str):
                return lambda boundary, tensor: split_graph_forward_trace.observe(
                    f"{module_prefix}.{boundary}", tensor
                )

            module._initium_forward_numeric_observer = make_observer(prefix)
        if split_graph_numeric_trace_on_error:
            print(
                f"[{run_id}] automatic split-graph forward numeric trace armed; "
                "detailed boundary tracing starts only after a trainer finite check "
                "fails, then replays that batch; output: "
                f"{split_graph_forward_trace.path}"
            )
        else:
            print(
                f"[{run_id}] split-graph forward numeric trace armed for absolute optimizer "
                f"step {split_graph_numeric_trace_step}; output: "
                f"{split_graph_forward_trace.path}"
            )

    def trace_split_graph_failure(
        step: int,
        failure: str,
        input_seq: torch.Tensor,
        target_digits: torch.Tensor,
        answer_mask: torch.Tensor,
        sequence_valid_mask: torch.Tensor | None,
        timespans: torch.Tensor | None,
    ) -> None:
        """Replay only the batch that tripped a finite check, with detailed tracing."""
        trace = split_graph_forward_trace
        if trace is None or not trace.activate_on_error(step, failure):
            return
        try:
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                replay_result = rnn(
                    input_seq,
                    (None, None, None),
                    reset_experience=True,
                    sequence_valid_mask=sequence_valid_mask,
                    timespans=timespans,
                    return_backbone_features=sgr_prediction_decoder is not None,
                )
                replay_output = replay_result[0]
                replay_features = replay_result[2] if len(replay_result) > 2 else None
                trace.set_time_index(None)
                trace.observe("rnn.output", replay_output)
                replay_logits = output_proj(replay_output.transpose(0, 1))
                trace.observe("task_output.logits", replay_logits)
                replay_task_loss = dataset.loss(replay_logits, target_digits, answer_mask)
                trace.observe("task_output.loss", replay_task_loss)
                if sgr_prediction_decoder is not None:
                    assert isinstance(replay_features, torch.Tensor)
                    replay_prediction_logits = sgr_prediction_decoder(replay_features[:, :-1])
                    trace.observe("sgr_prediction.logits", replay_prediction_logits)
                    replay_prediction_loss = masked_next_token_loss(
                        replay_prediction_logits,
                        input_seq,
                        dataset.last_stream_lengths,
                    )
                    trace.observe("sgr_prediction.loss", replay_prediction_loss)
            trace.finish_error_replay(step)
        except FloatingPointError as trace_error:
            print(
                f"[split-graph-forward-trace] automatic replay localized the first "
                f"non-finite boundary: {trace_error}",
                flush=True,
            )
            trace.close()
        except Exception as trace_error:
            print(
                f"[split-graph-forward-trace] automatic diagnostic replay failed "
                f"({type(trace_error).__name__}: {trace_error}); preserving the "
                "original training failure",
                flush=True,
            )
            trace.close()

    if memory_type in {"kv", "delta"}:
        print(
            f"[{run_id}] memory={memory_type} | heads={memory_heads} d_k={memory_dk} "
            f"d_v={memory_dv} read_lag={memory_read_lag} | sequence-parallel read path"
        )
    elif not has_dnc:
        print(f"[{run_id}] DNC absent | split-graph backbone-only model")
    else:
        print(
            f"[{run_id}] DNC implementation={dnc_implementation}"
            + (
                " | exact allocation path avoids one full usage-permutation sort per step"
                if dnc_implementation == "optimized"
                else ""
            )
            + (" | compiled memory forward" if dnc_compile_memory else "")
            + (
                f" | temporal-link sharpness exponent={dnc_link_sharpness:g}"
                if dnc_link_sharpness != 1.0
                else ""
            )
        )

    if split_graph_enabled:
        if workspace_enabled:
            workspace_active = isinstance(rnn, SplitGraphDNC) and rnn.workspace is not None
            write_mode = "random-control" if workspace_no_selection else "MoE-top-k"
            reset_workspace_each_step = workspace_reset_each_step or (
                eval_only and eval_workspace_mode == "reset-each-step"
            )
            persistence = "off (reset each timestep)" if reset_workspace_each_step else "on"
            print(
                f"[{run_id}] WORKSPACE {'active' if workspace_active else 'bypassed'} | "
                f"slots={workspace_num_slots} | write_selection={write_mode} | "
                f"persistence={persistence} | train_mode={workspace_train_mode} | "
                f"gate={workspace_gate_variant} | "
                f"bistable_gradient={workspace_bistable_gradient_mode} | "
                f"workspace_bptt_horizon={workspace_bptt_horizon or 'full'} | "
                f"reconsideration={'on' if workspace_reconsideration_enabled else 'off'}"
                f"(trigger={workspace_reconsideration_trigger_mode}, "
                f"threshold={workspace_reconsideration_surprise_threshold:g}) | "
                f"compute_precision={'fp32' if workspace_active else 'bypass'} | "
                "scope=final-backbone-MoE-bank; each specialist reads shared slots"
            )
        else:
            print(
                f"[{run_id}] WORKSPACE disabled | Selection-only split-graph run; "
                "no shared-slot writes or broadcast readback"
            )

    combiner_stage_kinds = getattr(
        getattr(rnn, "combiner_wrapper", None), "stage_kinds", None
    )  # set only for a hybrid ("<kind>+<kind>") split-graph combiner; None otherwise
    # v19: self-describing metadata for inference/ (capabilities + exact rebuild)
    extra_model_config = {
        "dataset_type": dataset_type,
        "dataset_name": dataset.name,
        "supported_dataset_types": [dataset_type],  # use ["*"] for a future multi-modal model
        "input_dim": INPUT_DIM,
        "output_dim": TRIPLE_DIM,
        "num_hidden_layers": getattr(rnn, "num_hidden_layers", None),
        "split_graph_num_blocks": split_graph_num_blocks,
        "split_graph_headdim": split_graph_headdim,
        "split_graph_combine_reads": split_graph_combine_reads,
        "split_graph_combiner_mode": split_graph_combiner_mode,
        "split_graph_combiner_variant": split_graph_combiner_variant,
        "split_graph_combiner_num_blocks": split_graph_combiner_num_blocks,
        "split_graph_mamba_hparams": (
            dict(d_state=_sg_d_state, d_conv=_sg_d_conv, expand=_sg_expand)
            if split_graph_enabled
            else None
        ),
        "split_graph_cfc_kwargs": (
            dict(
                mode=CFC_MODE,
                backbone_units=CFC_BACKBONE_UNITS,
                backbone_layers=CFC_BACKBONE_LAYERS,
                backbone_dropout=CFC_BACKBONE_DROPOUT,
                activation=CFC_ACTIVATION,
                mixed_memory=CFC_MIXED_MEMORY,
                residual=CFC_RESIDUAL,
            )
            if split_graph_enabled
            else None
        ),
        "split_graph_combiner_cfc_multi_source_moe": split_graph_combiner_cfc_multi_source_moe,
        "split_graph_backbone_only": split_graph_backbone_only,
        "dnc_enabled": has_dnc,
        "sgr_phase2_arm": sgr_phase2_arm,
        "sgr_phase2_signal": sgr_phase2_signal if sgr_phase2_arm is not None else None,
        "sgr_gate_eval_override": sgr_gate_eval_override if sgr_phase2_arm is not None else None,
        "sgr_phase4_dnc_write_salience": bool(sgr_dnc_write_salience),
        "sgr_phase4_workspace_write_salience": bool(sgr_workspace_write_salience),
        "sgr_prediction_error_signal": sgr_phase2_signal if sgr_prediction_enabled else None,
        "split_graph_moe_placement": split_graph_moe_placement,
        "split_graph_moe_backbone_block_isolation": split_graph_moe_backbone_block_isolation,
        "split_graph_moe_backbone_residual_scale": split_graph_moe_backbone_residual_scale,
        "dnc_read_variant": dnc_read_variant,
        "dnc_read_residual_scale": dnc_read_residual_scale,
        "dnc_read_residual_max_ratio": dnc_read_residual_max_ratio,
        "dnc_write_variant": dnc_write_variant,
        "dnc_write_observation_variance": dnc_write_observation_variance,
        "dnc_link_sharpness": dnc_link_sharpness,
        "dnc_allocation_topk": dnc_allocation_topk,
        "dnc_implementation": dnc_implementation,
        "dnc_compile_memory": dnc_compile_memory,
        "dnc_compute_precision": "fp32",
        "dnc_backward_trace": dnc_backward_trace,
        "batch_size": batch_size,
        "workspace_enabled": workspace_enabled,
        "workspace_num_slots": workspace_num_slots,
        "workspace_no_selection": workspace_no_selection,
        "workspace_reset_each_step": workspace_reset_each_step,
        "workspace_train_mode": workspace_train_mode,
        "workspace_gate_variant": workspace_gate_variant,
        "workspace_train_scope": workspace_train_scope,
        "workspace_bistable_selection_scale": workspace_bistable_selection_scale,
        "workspace_bistable_noise_scale": workspace_bistable_noise_scale,
        "workspace_bistable_gradient_mode": workspace_bistable_gradient_mode,
        "workspace_bptt_horizon": workspace_bptt_horizon,
        "workspace_reconsideration_enabled": workspace_reconsideration_enabled,
        "workspace_reconsideration_surprise_threshold": (
            workspace_reconsideration_surprise_threshold
        ),
        "workspace_reconsideration_trigger_mode": (workspace_reconsideration_trigger_mode),
        "workspace_bistable_deterministic": workspace_bistable_deterministic,
        "workspace_stability_max_activation": workspace_stability_max_activation,
    }
    if sgr_prediction_enabled:
        extra_model_config["sgr_prediction_decoder_enabled"] = True
        extra_model_config["sgr_prediction_loss_weight"] = float(sgr_prediction_loss_weight)
    controller_model_config: dict[str, Any] = {}
    if controller == "mamba":
        controller_model_config = {
            "mamba_d_state": MAMBA_D_STATE,
            "mamba_d_conv": MAMBA_D_CONV,
            "mamba_expand": MAMBA_EXPAND,
        }
    elif controller == "mamba2":
        controller_model_config = {
            "mamba2_d_state": MAMBA2_D_STATE,
            "mamba2_d_conv": MAMBA2_D_CONV,
            "mamba2_expand": MAMBA2_EXPAND,
            "mamba2_headdim": MAMBA2_HEADDIM,
            "mamba2_ngroups": MAMBA2_NGROUPS,
        }
    elif controller == "mamba3":
        controller_model_config = {
            "mamba3_d_state": MAMBA3_D_STATE,
            "mamba3_expand": MAMBA3_EXPAND,
            "mamba3_headdim": MAMBA3_HEADDIM,
            "mamba3_rope_fraction": MAMBA3_ROPE_FRACTION,
        }
    elif controller == "cfc":
        controller_model_config = {
            "cfc_mode": CFC_MODE,
            "cfc_backbone_units": CFC_BACKBONE_UNITS,
            "cfc_backbone_layers": CFC_BACKBONE_LAYERS,
            "cfc_backbone_dropout": CFC_BACKBONE_DROPOUT,
            "cfc_activation": CFC_ACTIVATION,
            "cfc_mixed_memory": CFC_MIXED_MEMORY,
            "cfc_residual": CFC_RESIDUAL,
        }
    elif "+" in controller:
        controller_model_config = {
            "mamba_d_state": MAMBA_D_STATE,
            "mamba_d_conv": MAMBA_D_CONV,
            "mamba_expand": MAMBA_EXPAND,
            "mamba2_d_state": MAMBA2_D_STATE,
            "mamba2_d_conv": MAMBA2_D_CONV,
            "mamba2_expand": MAMBA2_EXPAND,
            "mamba2_headdim": MAMBA2_HEADDIM,
            "mamba2_ngroups": MAMBA2_NGROUPS,
            "mamba3_d_state": MAMBA3_D_STATE,
            "mamba3_expand": MAMBA3_EXPAND,
            "mamba3_headdim": MAMBA3_HEADDIM,
            "mamba3_rope_fraction": MAMBA3_ROPE_FRACTION,
            "cfc_mode": CFC_MODE,
            "cfc_backbone_units": CFC_BACKBONE_UNITS,
            "cfc_backbone_layers": CFC_BACKBONE_LAYERS,
            "cfc_backbone_dropout": CFC_BACKBONE_DROPOUT,
            "cfc_activation": CFC_ACTIVATION,
            "cfc_mixed_memory": CFC_MIXED_MEMORY,
            "cfc_residual": CFC_RESIDUAL,
            "hybrid_cfc_num_blocks": HYBRID_CFC_NUM_BLOCKS,
        }
    memory_model_config = (
        {
            "memory_type": memory_type,
            "memory_heads": memory_heads,
            "memory_dk": memory_dk,
            "memory_dv": memory_dv,
            "memory_read_lag": memory_read_lag,
            "memory_topk": memory_topk,
            "memory_position": memory_position,
            "delta_gate": delta_gate,
            "delta_chunk": delta_chunk,
            "delta_use_fla": delta_use_fla,
        }
        if memory_type != "dnc"
        else {}
    )
    extra_model_config.update(memory_model_config)
    expected_resume_model_config: dict[str, Any] = {
        "input_size": INPUT_DIM,
        "hidden_size": hidden_size,
        "num_hidden_layers": getattr(rnn, "num_hidden_layers", None),
        "nr_cells": rnn.memories[0].nr_cells,
        "cell_size": cell_size,
        "read_heads": read_heads,
        "controller_type": controller,
        "link_matrix_mode": link_matrix_mode,
        "link_matrix_topk": link_matrix_topk,
        "dynamic_n_mode": dynamic_n_mode,
        "moe_enabled": moe_enabled,
        "moe_num_experts": moe_num_experts,
        "moe_expert_dim": moe_expert_dim,
        "moe_capacity_factor": moe_capacity_factor,
        "moe_load_balance_alpha": moe_load_balance_alpha,
        "moe_top_k": moe_top_k,
        "split_graph_enabled": split_graph_enabled,
        "split_graph_backbone_only": split_graph_backbone_only,
        "split_graph_variant": split_graph_variant,
        **memory_model_config,
        **controller_model_config,
        **extra_model_config,
    }

    if link_matrix_mode != "dense":
        patch_link_matrix(rnn, mode=link_matrix_mode, topk=link_matrix_topk)
        topk_note = f" topk={link_matrix_topk}" if link_matrix_mode == "sparse_topk" else ""
        print(
            f"[{run_id}] link_matrix_mode={link_matrix_mode}{topk_note} -- "
            f"Static Option 1 applied to rnn.memories."
        )

    output_proj = nn.Linear(
        hidden_size if not has_dnc else INPUT_DIM,
        TRIPLE_DIM,
    ).to(device)
    dataset.set_output_proj(output_proj)
    sgr_prediction_decoder = (
        SurprisePredictionHead(hidden_size, INPUT_DIM).to(device)
        if sgr_prediction_enabled
        else None
    )
    if isinstance(rnn, SurpriseRecallBackboneOnly):
        rnn.set_prediction_decoder(sgr_prediction_decoder)
    elif isinstance(rnn, SplitGraphDNC) and sgr_phase4_enabled:
        rnn.set_sgr_prediction_decoder(sgr_prediction_decoder, sgr_phase2_signal)
    if workspace_train_scope == "gate" and not segment_diagnostics_enabled:
        raise ValueError("Phase 4 gate-only stability checks require segment diagnostics")
    segment_diagnostics = SegmentDiagnostics(
        rnn,
        enabled=segment_diagnostics_enabled,
        activation_absmax_limit=(
            workspace_stability_max_activation if workspace_train_scope == "gate" else None
        ),
    )
    if workspace_train_scope == "gate" and isinstance(rnn, SplitGraphDNC):
        # PyTorch's Module __setattr__ typing only accepts parameters, buffers,
        # and submodules; this runtime diagnostic hook is a plain callback.
        rnn._initium_segment_observer = segment_diagnostics.observe_tensor  # type: ignore[assignment]  # fmt: skip
    segment_diagnostics.add_module(output_proj, "task_output")
    if segment_diagnostics_enabled and isinstance(rnn, SplitGraphDNC):
        combiner = rnn.combiner_wrapper
        if combiner is not None and hasattr(combiner, "in_adapter"):
            segment_diagnostics.add_module(combiner.in_adapter, "controller.cfc.in_adapter")
            for index, adapter in enumerate(getattr(combiner, "source_in_adapters", None) or ()):
                segment_diagnostics.add_module(adapter, f"controller.cfc.source_adapter.{index}")
            for index, block in enumerate(getattr(combiner, "blocks", ())):
                cfc_cell = getattr(block, "cfc", None)
                if isinstance(cfc_cell, nn.Module):
                    segment_diagnostics.add_module(cfc_cell, f"controller.cfc.cell.{index}")
    moe_layer_ids = {id(layer) for layer in getattr(rnn, "moe_layers", [])}
    for module_name, module in rnn.named_modules():
        if id(module) in moe_layer_ids:
            module.diagnostic_name = module_name or type(module).__name__
    for module in rnn.modules():
        if isinstance(module, SwitchMoE):
            module.set_diagnostics_enabled(moe_diagnostics_enabled)
    workspace_telemetry_enabled = bool(
        isinstance(rnn, SplitGraphDNC)
        and rnn.workspace is not None
        and (workspace_reconsideration_enabled or sgr_workspace_write_salience)
    )
    workspace_telemetry_log_file: TextIO | None = None
    workspace_telemetry_log_writer: Any | None = None
    if workspace_telemetry_enabled and not eval_only:
        workspace_telemetry_log_path = os.path.join(
            LOG_DIR, f"run_{run_id}_workspace_telemetry.csv"
        )
        telemetry_log_has_rows = os.path.isfile(workspace_telemetry_log_path) and (
            os.path.getsize(workspace_telemetry_log_path) > 0
        )
        workspace_telemetry_log_file = open(
            workspace_telemetry_log_path, "a" if resuming else "w", newline=""
        )
        workspace_telemetry_log_writer = csv.writer(workspace_telemetry_log_file)
        if not resuming or not telemetry_log_has_rows:
            workspace_telemetry_log_writer.writerow(
                [
                    "step",
                    "lesson",
                    "window_steps",
                    "reconsideration_trigger_mode",
                    "eligible_slots",
                    "triggered_slots",
                    "trigger_rate",
                    "changed_slots",
                    "changed_given_trigger",
                    "changed_of_eligible",
                    "state_delta_rms",
                    "prediction_error_mean",
                    "reconsideration_error_ema_mean",
                    "write_salience_weight_rms",
                    "write_salience_weight_absmax",
                    "write_salience_bias_rms",
                ]
            )
    if isinstance(rnn, SplitGraphDNC) and rnn.workspace is not None:
        # Phase 4 telemetry is required to interpret these interventions, so
        # collect workspace-only diagnostics whenever either consumer is active
        # even if the broader MoE diagnostics are disabled.
        rnn.workspace.set_diagnostics_enabled(
            moe_diagnostics_enabled or workspace_telemetry_enabled
        )
        rnn.workspace_bistable_deterministic = workspace_bistable_deterministic
        if workspace_train_scope == "gate":
            # PyTorch's Module __setattr__ typing rejects runtime callback hooks.
            rnn.workspace._initium_recurrent_state_observer = (
                segment_diagnostics.observe_recurrent_state  # type: ignore[assignment]
            )  # fmt: skip

    def emit_failure_diagnostics(channel: str, *, include_gradients: bool = False) -> None:
        segment_diagnostics.emit(channel, include_gradients=include_gradients)
        if dnc_backward_trace and split_graph_enabled:
            memory_diag = rnn.memories[0].pop_numerical_diagnostics()
            if memory_diag:
                print(
                    f"[dnc-numerics:{channel}] "
                    + " ".join(f"{name}={value:.4g}" for name, value in memory_diag.items())
                )
        if moe_enabled and moe_diagnostics_enabled:
            _, failure_moe_diag = pop_total_moe_aux_loss(rnn.moe_layers, include_diagnostics=True)
            reports = _format_moe_layer_reports(failure_moe_diag)
            if reports:
                print(f"[moe-diag:{channel}] " + " | ".join(reports))

    def report_bistable_evaluation(mode: str) -> None:
        if not (
            workspace_gate_variant == "bistable"
            and isinstance(rnn, SplitGraphDNC)
            and rnn.workspace is not None
        ):
            return
        diagnostics = rnn.workspace.consume_bistable_diagnostics()
        if not diagnostics or diagnostics["gate_candidate_count"].item() == 0:
            print(f"[{run_id}] [bistable:{mode}] no valid gate candidates were recorded")
            return
        histogram = ",".join(f"{value:.3f}" for value in diagnostics["gate_histogram"].tolist())
        writers = ",".join(f"{value:.3f}" for value in diagnostics["writer_frequency"].tolist())
        print(
            f"[{run_id}] [bistable:{mode}] p_mean="
            f"{diagnostics['gate_probability_mean'].item():.3f} "
            f"low/mid/high="
            f"{diagnostics['gate_low_frac'].item():.1%}/"
            f"{diagnostics['gate_mid_frac'].item():.1%}/"
            f"{diagnostics['gate_high_frac'].item():.1%} "
            f"commit_freq={diagnostics['write_frequency'].item():.1%} "
            f"selection_peak={diagnostics['selection_strength_mean'].item():.3f} "
            f"probability_histogram_10bin={histogram} "
            f"writer_frequency={writers}"
        )

    def format_bistable_forward_debug() -> str:
        if not workspace_bistable_debug or not isinstance(rnn, SplitGraphDNC):
            return ""
        if rnn.workspace is None:
            return ""
        diagnostics = rnn.workspace.last_diagnostics()
        fields = [
            (key.removeprefix("debug_"), value)
            for key, value in diagnostics.items()
            if key.startswith("debug_")
        ]
        fields.extend(
            (f"workspace_{key}", diagnostics[key])
            for key in (
                "slot_rms",
                "broadcast_to_specialist_ratio",
                "write_gate_mean",
                "read_gate",
            )
            if key in diagnostics
        )
        if not fields:
            return ""
        values = torch.stack([value.detach().float() for _, value in fields]).cpu().tolist()
        return " ".join(f"{name}={value:.3g}" for (name, _), value in zip(fields, values))

    def report_bistable_nonfinite(step_index: int, forward_debug: str, gradient_debug: str) -> None:
        if not workspace_bistable_debug:
            return
        print(
            f"[{run_id}] [bistable-debug:failure] step={step_index} "
            f"forward[{forward_debug or 'unavailable'}] "
            f"raw_gradients[{gradient_debug or 'unavailable'}]"
        )

    # Install the stochastic write head(s) BEFORE building the
    # optimizer, so their parameters (mu_transform, logvar_transform) are
    # included in optimizer.parameters(). mu_transform is initialized from
    # the original deterministic write_vector_transform's weights, so at
    # step 0 the sampled mean exactly matches Phase 0's write vector.
    # sample=True for all beta in BETAS_TO_SWEEP now that beta=0 is dropped
    # from the sweep (see header note) -- beta_target is always > 0.0 here.
    if not has_dnc:
        # Phase 2's dedicated backbone-only model has no DNC Memory modules;
        # the general split-graph backbone-only mode is identical in this
        # respect. Keep the shared training/checkpoint code's head list empty.
        stochastic_heads: list[Any] = []
    elif split_graph_enabled and dnc_write_variant == "kanerva-closed-form":
        # The Bayesian update consumes a deterministic observation from the
        # DNC write-value projection. Sampling a stochastic write head here
        # would make uncertainty sampling a second, confounded write change.
        stochastic_heads = []
    else:
        stochastic_heads = install_stochastic_write_heads(
            rnn, device=device, sample=(beta_target > 0.0)
        )

    if workspace_enabled and workspace_train_mode == "frozen":
        for parameter in rnn.parameters():
            parameter.requires_grad_(False)
        assert isinstance(rnn, SplitGraphDNC) and rnn.workspace is not None
        if workspace_train_scope == "gate":
            gate_parameters = (
                rnn.workspace.bistable_gate_parameters()
                if workspace_gate_variant == "bistable"
                else list(rnn.workspace.write_gate.parameters())
            )
            if rnn.workspace.prediction_error_to_write_gate is not None:
                gate_parameters = [
                    *gate_parameters,
                    *rnn.workspace.prediction_error_to_write_gate.parameters(),
                ]
            for parameter in gate_parameters:
                parameter.requires_grad_(True)
        else:
            for parameter in rnn.workspace.parameters():
                parameter.requires_grad_(True)
        for parameter in output_proj.parameters():
            parameter.requires_grad_(False)
    trainable_named_parameters = [
        (f"rnn.{name}", parameter)
        for name, parameter in rnn.named_parameters()
        if parameter.requires_grad
    ] + [
        (f"output_proj.{name}", parameter)
        for name, parameter in output_proj.named_parameters()
        if parameter.requires_grad
    ]
    if sgr_prediction_decoder is not None:
        trainable_named_parameters.extend(
            (f"sgr_prediction_decoder.{name}", parameter)
            for name, parameter in sgr_prediction_decoder.named_parameters()
            if parameter.requires_grad
        )
    trainable_parameters = [parameter for _, parameter in trainable_named_parameters]
    if not trainable_parameters:
        raise ValueError("training configuration has no trainable parameters")
    if (
        workspace_enabled
        and workspace_train_mode == "frozen"
        and workspace_train_scope == "gate"
        and workspace_gate_variant in ("continuous", "bistable")
    ):
        gate_prefix = (
            "rnn.workspace.bistable_gate."
            if workspace_gate_variant == "bistable"
            else "rnn.workspace.write_gate."
        )
        unexpected_trainable = [
            name for name, _ in trainable_named_parameters if not name.startswith(gate_prefix)
        ]
        if unexpected_trainable:
            raise RuntimeError(
                "Frozen gate-only training has parameters outside the selected write gate: "
                + ", ".join(unexpected_trainable)
            )
    optimizer = torch.optim.Adam(trainable_parameters, lr=LR)
    # Option 3 (Concept 24/LB-17): multi-axis efficiency accounting.
    # param_count is fixed for the life of this run -- static Option 2's
    # nr_cells resize never changes it (every Memory sublayer is sized by
    # cell_size/read_heads/input_size, never nr_cells, per the Option-2
    # analysis) -- so this is a one-time computation, not per-step.
    param_count = sum(p.numel() for p in rnn.parameters()) + sum(
        p.numel() for p in output_proj.parameters()
    )
    if sgr_prediction_decoder is not None:
        param_count += sum(p.numel() for p in sgr_prediction_decoder.parameters())
    amp_enabled = USE_AMP and device.type == "cuda"
    kl_on = bool(stochastic_heads) and stochastic_heads[0].sample
    print(
        f"[{run_id}] params {param_count} | kl_terms={'on' if kl_on else 'off'} | amp={'on' if amp_enabled else 'off'}"
    )
    if sgr_prediction_decoder is not None:
        if sgr_phase2_arm is not None:
            decoder_phase_label = "Phase 2"
        elif sgr_phase4_enabled:
            decoder_phase_label = "Phase 4"
        else:
            decoder_phase_label = "Phase 1"
        print(
            f"[{run_id}] SGR {decoder_phase_label} decoder enabled | "
            "pred_loss_weight="
            f"{sgr_prediction_loss_weight:g} | signal source=causal backbone features"
        )
    if sgr_phase4_enabled:
        consumers = []
        if sgr_dnc_write_salience:
            consumers.append("DNC write-strength")
        if sgr_workspace_write_salience:
            consumers.append("workspace write-gate")
        if workspace_reconsideration_enabled and workspace_reconsideration_trigger_mode.startswith(
            "prediction_error"
        ):
            consumers.append(
                f"workspace reconsideration ({workspace_reconsideration_trigger_mode}, "
                f"EMA decay={WorkspaceBroadcast._RECONSIDERATION_ERROR_EMA_DECAY:g})"
            )
        dnc_weight = (
            f", initial DNC salience coefficient="
            f"{rnn.memories[0].sgr_error_to_write_strength.detach().item():.3g}"
            if sgr_dnc_write_salience and hasattr(rnn.memories[0], "sgr_error_to_write_strength")
            else ""
        )
        consumer_summary = "; ".join(consumers) if consumers else "none (matched control arm)"
        print(
            f"[{run_id}] SGR Phase 4 consumers={consumer_summary} | "
            f"signal={sgr_phase2_signal} (detached){dnc_weight}"
        )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled, init_scale=AMP_INIT_SCALE)

    # Dynamic-N (macro-scale): None unless dynamic_n_mode -- every call site
    # below guards on `dynamic_n_ctrl is not None` so this is a true no-op
    # (not just a disabled-but-present object) when the flag is off.
    dynamic_n_ctrl = (
        DynamicNController(
            floor=dynamic_n_floor,
            ceiling=dynamic_n_ceiling,
            growth_factor=DYNAMIC_N_GROWTH_FACTOR,
            trigger_frac=dynamic_n_trigger_frac,
            ema_decay=DYNAMIC_N_EMA_DECAY,
            cooldown_steps=dynamic_n_cooldown_steps,
        )
        if dynamic_n_mode
        else None
    )
    dynamic_n_log_writer = None
    if dynamic_n_ctrl is not None:
        # Own CSV, same one-file-per-mechanism convention as
        # run_{run_id}_lesson_advances.csv / _prior_snapshots.csv above.
        dynamic_n_log_path = os.path.join(LOG_DIR, f"run_{run_id}_dynamic_n_growth.csv")
        dynamic_n_log_file = open(dynamic_n_log_path, "a" if resuming else "w", newline="")
        dynamic_n_log_writer = csv.writer(dynamic_n_log_file)
        if not resuming:
            dynamic_n_log_writer.writerow(
                ["step", "lesson", "old_nr_cells", "new_nr_cells", "usage_ema"]
            )

    # --- LR schedule -----------------------------------------------------
    # FIX (see chat log / Q21-Experiment-Log Section 5): CosineAnnealingLR's
    # formula -- eta_min + 0.5*(base_lr-eta_min)*(1+cos(pi*last_epoch/T_max))
    # -- is PERIODIC in last_epoch with period 2*T_max, not a one-way ramp
    # that flatlines at eta_min once last_epoch exceeds T_max. Every run
    # that trains past LR_DECAY_STEPS=20000 (i.e. every extended/switch-late
    # run in this project, since TOTAL_STEPS=80000) was therefore climbing
    # back toward LR=3e-4 after each trough at step 20000, 60000, ... and
    # peaking again at step 40000, 80000, ... -- confirmed empirically via
    # the added LR log column (observed ~2.94e-4 at step 78000, matching
    # the closed-form prediction almost exactly). The original 20,000-step
    # beta-sweep (Run 1 beta in {0.01,0.1,1.0}) is NOT affected -- it ends
    # exactly at the first trough -- but the extended beta=0 run and both
    # switch-late runs (beta=0.001, beta=0.01) trained/evaluated well past
    # that point under a partially-cycling, not monotonically-decaying, LR.
    #
    # Fix: don't use torch's stateful scheduler at all (its internal
    # last_epoch counter is also what silently carried the drift across
    # checkpoint resumes). Instead recompute LR directly from the absolute
    # step every iteration, with the step clamped at LR_DECAY_STEPS so the
    # schedule flatlines at LR_MIN once reached, exactly matching the
    # "cosine anneal from LR to LR_MIN over this many steps" comment this
    # constant always had -- it just was never enforced past that point.
    #
    # v3 addition: linear warmup 0 -> LR over the first WARMUP_STEPS,
    # before the cosine anneal (which now spans WARMUP_STEPS..LR_DECAY_STEPS
    # instead of 0..LR_DECAY_STEPS, so it still reaches LR_MIN exactly at
    # LR_DECAY_STEPS as before). Since LR is still a pure function of the
    # absolute step, this stays resume-safe for the same reason the original
    # fix was -- a resume past WARMUP_STEPS just lands in the cosine part,
    # no separate warmup state to save/restore.
    def lr_at_step(s):
        if s < WARMUP_STEPS:
            return LR * (s + 1) / WARMUP_STEPS
        s_clamped = min(s, LR_DECAY_STEPS)
        decay_span = max(1, LR_DECAY_STEPS - WARMUP_STEPS)
        progress = (s_clamped - WARMUP_STEPS) / decay_span
        return LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * progress))

    def set_lr(step_now):
        lr_now = lr_at_step(step_now)
        for g in optimizer.param_groups:
            g["lr"] = lr_now
        return lr_now

    step = 0
    anneal_start_step = 0  # Phase 1: absolute step the KL anneal ramp is measured from.
    # Stays 0 for a normal fresh run or a same-beta resume (anneal counted
    # from the true start, as before). Reset to the resume step below when
    # the checkpoint's beta_target differs from this run's -- i.e. a
    # deliberate deterministic(or other-beta)->this-beta switch -- so the
    # ramp restarts at 0 from the switch point instead of reading the
    # already-large absolute step and jumping straight to full beta_target
    # on the first post-switch update.
    lesson_dip_start_step = -(
        10**9
    )  # idle at run start (so the dip ramp is already at 1.0 immediately);
    # reset to `step` on every lesson advance instead
    if resume_requested:
        ckpt = resume_checkpoint
        if ckpt is None:
            raise RuntimeError("resume_requested was set without a loaded checkpoint")
        # Static Option 2 resume fix: rnn was just constructed above at
        # LESSON_NR_CELLS[0] (curriculum.lesson doesn't exist yet at that
        # point in run()). If this checkpoint is mid-schedule, the live
        # model's Memory is still the wrong size -- maybe_advance() only
        # resizes on a LIVE lesson transition, which never replays for
        # lessons already passed before this resume. Must happen BEFORE
        # rnn.load_state_dict() below, so load_state_dict fills the
        # resized module's parameters with the checkpoint's trained values
        # directly, instead of transplanting fresh-init weights that then
        # get immediately overwritten anyway.
        # Prefer the checkpoint's own recorded nr_cells (self-describing,
        # Edit A above) over recomputing from the current LESSON_NR_CELLS
        # list, in case the schedule was edited since this checkpoint was
        # produced.
        ckpt_nr_cells = ckpt.get("model_config", {}).get("nr_cells")
        if ckpt_nr_cells is None:
            ckpt_nr_cells = curriculum.lesson_nr_cells[ckpt["curriculum_lesson"]]
            print(
                f"[{run_id}] WARNING: checkpoint predates per-lesson "
                f"nr_cells recording (static Option 2) -- inferring "
                f"nr_cells={ckpt_nr_cells} from LESSON_NR_CELLS"
                f"[{ckpt['curriculum_lesson']}] instead of a recorded value."
            )
        if not isinstance(rnn, SurpriseRecallBackboneOnly):
            resize_memory(rnn, ckpt_nr_cells, device=device, optimizer=optimizer)

        if frozen_workspace_warm_start:
            if ckpt.get("beta_target") != beta_target:
                raise ValueError(
                    "Frozen-workspace warm start requires the baseline checkpoint "
                    f"beta ({ckpt.get('beta_target')}) to match this run ({beta_target})."
                )
            missing_workspace_keys = load_frozen_workspace_warm_start(
                rnn,
                output_proj,
                ckpt,
                expected_resume_model_config,
            )
            # The baseline was saved at a later curriculum lesson, where
            # memory has more cells. Start the workspace experiment at lesson
            # 1 with that lesson's configured memory size and a fresh schedule.
            resize_memory(rnn, starting_nr_cells, device=device, optimizer=optimizer)
            # Memory resize constructs a replacement module with fresh
            # Parameter objects; reapply the frozen policy after transplant.
            for parameter in rnn.parameters():
                parameter.requires_grad_(False)
            assert isinstance(rnn, SplitGraphDNC) and rnn.workspace is not None
            if workspace_train_scope == "gate":
                gate_parameters = (
                    rnn.workspace.bistable_gate_parameters()
                    if workspace_gate_variant == "bistable"
                    else list(rnn.workspace.write_gate.parameters())
                )
                if rnn.workspace.prediction_error_to_write_gate is not None:
                    gate_parameters = [
                        *gate_parameters,
                        *rnn.workspace.prediction_error_to_write_gate.parameters(),
                    ]
                for parameter in gate_parameters:
                    parameter.requires_grad_(True)
            else:
                for parameter in rnn.workspace.parameters():
                    parameter.requires_grad_(True)
            # Model construction and checkpoint loading consume RNG state.
            # Start the new workspace experiment from the requested seed.
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            curriculum.lesson = 0
            step = 0
            print(
                f"[{run_id}] Warm-started frozen workspace from {resume_from} "
                f"(baseline step {ckpt['step']}, lesson "
                f"{ckpt.get('curriculum_lesson', 0) + 1}); loaded matching "
                f"backbone/combiner/MoE/DNC/task-head weights, kept "
                f"{len(missing_workspace_keys)} workspace tensors freshly initialized, "
                f"reset optimizer, scaler, RNG, and curriculum to step 0 / lesson 1."
            )
        elif gate_variant_warm_start:
            incompatible = rnn.load_state_dict(ckpt["rnn_state_dict"], strict=False)
            expected_missing = {
                key for key in rnn.state_dict() if key.startswith("workspace.bistable_gate.")
            }
            if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
                raise ValueError(
                    "Bistable gate warm start only permits newly initialized "
                    "workspace.bistable_gate parameters; checkpoint state differs "
                    f"(missing={incompatible.missing_keys[:8]}, "
                    f"unexpected={incompatible.unexpected_keys[:8]})."
                )
            output_proj.load_state_dict(ckpt["output_proj_state_dict"])
            print(
                f"[{run_id}] Initialized bistable gate from continuous-workspace "
                f"checkpoint {resume_from} at step {step}; all shared model and "
                "workspace tensors loaded, gate parameters initialized fresh, "
                "optimizer reset."
            )
        elif workspace_condition_warm_start:
            rnn.load_state_dict(ckpt["rnn_state_dict"])
            output_proj.load_state_dict(ckpt["output_proj_state_dict"])
            print(
                f"[{run_id}] Started a fresh Phase 4 condition from {resume_from} "
                f"at step {step}; workspace/model tensors loaded, optimizer reset."
            )
        else:
            rnn.load_state_dict(ckpt["rnn_state_dict"])
            output_proj.load_state_dict(ckpt["output_proj_state_dict"])
            if not eval_only:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if sgr_prediction_decoder is not None:
            prediction_state = ckpt.get("sgr_prediction_decoder_state_dict")
            if prediction_state is None:
                raise ValueError(
                    "Phase 1 resume requires a checkpoint with a trained SGR prediction decoder"
                )
            sgr_prediction_decoder.load_state_dict(prediction_state)
        # v3 fix: restore GradScaler state (scale, growth tracker) instead
        # of leaving `scaler` at its freshly-constructed init_scale -- see
        # header note. `.get(...)` instead of ["..."] so a pre-v3 checkpoint
        # (saved before this key existed) still loads, just without this fix.
        if (
            not frozen_workspace_warm_start
            and not workspace_condition_warm_start
            and "scaler_state_dict" in ckpt
        ):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        elif not frozen_workspace_warm_start and not workspace_condition_warm_start:
            print(
                f"[{run_id}] WARNING: checkpoint predates GradScaler-state "
                f"saving (v3 fix) -- scaler starting fresh at "
                f"init_scale={scaler.get_scale()} instead of resuming the "
                # (continued below, unchanged)
                f"prior leg's scale."
            )
        # NOT loading scheduler_state_dict: LR is now recomputed from the
        # absolute step every iteration via set_lr() (see above), clamped
        # at LR_DECAY_STEPS -- this sidesteps the periodicity bug entirely,
        # and also means old checkpoints' scheduler_state_dict (which may
        # already reflect the buggy drifted state) is simply ignored rather
        # than needing migration.
        if not frozen_workspace_warm_start:
            curriculum.lesson = ckpt["curriculum_lesson"]
        # Dynamic-N (macro-scale): restore the usage-EMA/cooldown/growth
        # history, so a resumed run doesn't silently reset the trigger's
        # sustained-saturation window back to 0 (which would just delay the
        # next growth event by however long the EMA takes to re-climb --
        # not incorrect, but not reproducible either) and so growth_history
        # stays complete across resumed legs. `.get(...)` so a checkpoint
        # saved before dynamic_n_mode existed, or a checkpoint from a
        # non-dynamic-N run, still loads -- the controller (if this run
        # even has one) just starts fresh, same as a brand-new run would.
        if dynamic_n_ctrl is not None and not frozen_workspace_warm_start:
            if ckpt.get("dynamic_n_state") is not None:
                dynamic_n_ctrl.load_state_dict(ckpt["dynamic_n_state"])
            else:
                print(
                    f"[{run_id}] WARNING: checkpoint has no dynamic_n_state "
                    f"(pre-dates Dynamic-N, or was saved by a non-dynamic-N "
                    f"run) -- usage EMA/cooldown/growth_history starting "
                    f"fresh instead of resuming the prior leg's."
                )
        # v5 (Phase 2): restore (mu_g, Sigma_g, last_snapshot_step) for every
        # installed stochastic write head. Without this, resuming would
        # silently continue training/evaluating against whatever prior
        # snapshot happens to sit in each freshly-constructed head's
        # zero-init buffers (i.e. silently reset to N(0,I)) instead of the
        # snapshot this checkpoint was actually trained against -- the same
        # class of bug the write_mode deterministic/sampled mismatch check
        # already exists to catch in eval_from_checkpoint.py. `.get(...)`
        # instead of ["..."] so a pre-v5 (Phase 1) checkpoint, which has no
        # prior state to restore at all, still loads -- just starting the
        # learned prior fresh at N(0,I), same as a brand-new run would.
        if not frozen_workspace_warm_start and "prior_state" in ckpt:
            load_prior_state(stochastic_heads, ckpt["prior_state"])
        elif not frozen_workspace_warm_start:
            print(
                f"[{run_id}] WARNING: checkpoint predates the learned-prior "
                f"snapshot (v5/Phase 2) - prior starting fresh at N(0,I) "
                f"instead of resuming a prior snapshot."
            )
        # v6: restore the dedicated OOD-sampling RNG's state, so a resumed
        # run's OOD walk sequence continues from exactly where the
        # pre-resume leg left off, rather than silently restarting from
        # ood_rng's fresh, seed-derived starting state (see save_checkpoint()
        # docstring). `.get(...)` so a pre-v6 checkpoint still loads --
        # ood_rng just starts fresh in that case, identical to today's
        # behavior before this fix.
        if not frozen_workspace_warm_start and "ood_rng_state" in ckpt:
            ood_rng.setstate(ckpt["ood_rng_state"])
        elif not frozen_workspace_warm_start:
            print(
                f"[{run_id}] WARNING: checkpoint predates ood_rng-state "
                f"saving (v6 fix) -- OOD sampling restarting from its "
                f"fresh, seed-derived state instead of continuing the "
                f"pre-resume leg's exact walk sequence."
            )
        if not frozen_workspace_warm_start:
            restore_rng_state(ckpt["rng_state"])
            step = ckpt["step"]
        if eval_only and eval_seed is not None:
            # An explicit evaluation seed overrides checkpoint RNG state so
            # separate checkpoints/modes can be compared on identical
            # generated episodes. Omit it to replay the checkpoint's stream.
            random.seed(eval_seed)
            np.random.seed(eval_seed)
            torch.manual_seed(eval_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(eval_seed)
            ood_rng.seed(eval_seed)
        if not frozen_workspace_warm_start and ckpt["beta_target"] != beta_target:
            anneal_start_step = step  # Phase 1: restart the ramp here
            print(
                f"[{run_id}] beta_target changed on resume "
                f"({ckpt['beta_target']} -> {beta_target}); KL anneal "
                f"restarted from step {anneal_start_step}, ramping over "
                f"the next {KL_ANNEAL_STEPS} steps."
            )
        if eval_only:
            if not isinstance(rnn, SplitGraphDNC) or rnn.workspace is None:
                if eval_workspace_mode != "checkpoint":
                    raise ValueError(
                        "workspace evaluation mode requires a checkpoint with an active workspace"
                    )
            else:
                if eval_workspace_mode == "reset-each-step":
                    rnn.workspace_reset_each_step = True
                elif eval_workspace_mode == "bypass":
                    rnn.workspace_eval_bypass = True
            print(
                f"[{run_id}] Evaluation-only checkpoint load | step={step} | "
                f"workspace_mode={eval_workspace_mode} | "
                f"eval_seed={eval_seed if eval_seed is not None else 'checkpoint RNG'} | "
                "optimizer state not loaded"
            )
        elif not frozen_workspace_warm_start:
            print(
                f"[{run_id}] Resumed from {resume_from} at step {step} "
                f"(lesson {curriculum.lesson + 1}/{len(curriculum.table)}); "
                f"continuing to total_steps={total_steps}"
            )
        if step >= total_steps and not eval_only:
            print(
                f"[{run_id}] Checkpoint step {step} already >= total_steps "
                f"{total_steps} -- nothing to do. Raise --total-steps if you "
                f"want to extend further."
            )

    run_mode_label = "Evaluation-only" if eval_only else "Training"
    print(
        f"\n=== [{run_id}] {run_mode_label} (beta_target={beta_target}, beta_mode={beta_mode}) "
        f"{'[warm-started]' if frozen_workspace_warm_start else '[resumed]' if resuming else ''} ==="
    )
    rnn.train(not eval_only)
    output_proj.train(not eval_only)
    if not eval_only:
        set_lr(step)  # Phase 1 FIX: ensure correct clamped LR from the very first
    # iteration -- otherwise a resumed run's first step would briefly use
    # whatever LR was saved inside optimizer_state_dict at checkpoint time
    # (potentially still reflecting the old buggy drifted value).
    running_task_loss, running_kl_loss, running_div, running_grad_norm = 0.0, 0.0, 0, 0.0
    running_sgr_prediction_loss: Any = 0.0
    running_log_steps = 0
    last_train_sgr_signal_diagnostics: dict[str, torch.Tensor] = {}
    workspace_reconsideration_window: dict[str, torch.Tensor] | None = None
    if workspace_telemetry_enabled and workspace_reconsideration_enabled:
        workspace_reconsideration_window = {
            name: torch.zeros((), device=device, dtype=torch.float64)
            for name in (
                "reconsideration_eligible_count",
                "reconsideration_trigger_count",
                "reconsideration_changed_count",
                "reconsideration_state_delta_sq_sum",
            )
        }
    running_postclip_grad_norm = 0.0
    running_finite_grad_steps = 0
    running_nonfinite_grad_steps = 0
    running_clipped_grad_steps = 0
    max_grad_norm = 10.0
    bistable_grad_probe = None
    if workspace_train_scope == "gate" and workspace_gate_variant == "bistable":
        # SegmentDiagnostics reports parameter gradients after clipping. Keep
        # a separate pre-clip trace so an unexpectedly large global norm can
        # be attributed to the exact trainable tensors that produced it.
        bistable_grad_probe = {
            name: {
                "sum_sq": torch.zeros((), device=device, dtype=torch.float64),
                "max": torch.zeros((), device=device, dtype=torch.float64),
                "max_step": torch.full((), -1, device=device, dtype=torch.int64),
            }
            for name, _ in trainable_named_parameters
        }
        bistable_grad_probe_steps = 0
        bistable_grad_probe_global_max = torch.zeros((), device=device, dtype=torch.float64)
        bistable_grad_probe_global_max_step = torch.full((), -1, device=device, dtype=torch.int64)
        bistable_grad_probe_relative_error = torch.zeros((), device=device, dtype=torch.float64)

    # Dynamic-beta controller state (no-op / unused when beta_mode == "static").
    # Initialized post-resume so a resumed dynamic run starts its health check
    # from the just-restored scaler scale rather than a fresh-process default.
    beta_constraint_ema = 0.0
    scale_at_last_eval = scaler.get_scale()
    grad_was_finite_since_eval = True
    consecutive_nonfinite_grad_steps = 0

    t0 = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # Log addition #5: cumulative wall-clock for this run(). Starts fresh on
    # a resumed leg too (i.e. measures THIS process's elapsed time, not
    # elapsed time since the original run began across all resumed legs --
    # legs already have their own timestamped log rows if that's needed).
    t_run_start = time.time()

    while step < total_steps and not eval_only:
        if split_graph_forward_trace is not None:
            split_graph_forward_trace.begin_step(step)
        dnc_trace_this_step = (
            dnc_backward_trace
            and isinstance(rnn, SplitGraphDNC)
            and dnc_backward_trace_step is not None
            and step == dnc_backward_trace_step
        )
        if dnc_backward_trace_step is not None and isinstance(rnn, SplitGraphDNC):
            # DNC boundary hooks synchronize while checking finiteness. Install
            # them only for the selected absolute step during targeted tracing.
            rnn.memories[0].set_backward_trace(dnc_trace_this_step, step=step)
        anomaly_this_step = (
            workspace_bistable_anomaly_detection
            and workspace_bistable_anomaly_step is not None
            and step == workspace_bistable_anomaly_step
        )
        segment_diagnostics.set_nonfinite_trace(anomaly_this_step)
        if isinstance(rnn, SplitGraphDNC):
            combiner = rnn.combiner_wrapper
            if combiner is not None and hasattr(combiner, "set_gradient_trace"):
                combiner.set_gradient_trace(anomaly_this_step, step=step)
        if anomaly_this_step:
            # Anomaly detection must be enabled before this step's forward so
            # PyTorch can retain the originating stack for backward failures.
            torch.autograd.set_detect_anomaly(True, check_nan=True)
        batch = dataset.sample_batch(curriculum, batch_size)
        if len(batch) == 3:
            input_seq, target_digits, answer_mask = batch
            timespans = None
        elif len(batch) == 4:
            input_seq, target_digits, answer_mask, timespans = batch
        else:
            raise ValueError(
                "dataset.sample_batch() must return (inputs, targets, mask) or "
                "(inputs, targets, mask, timespans)"
            )

        input_seq = input_seq.to(device, non_blocking=True)
        target_digits = target_digits.to(device, non_blocking=True)
        answer_mask = answer_mask.to(device, non_blocking=True)
        if timespans is not None:
            timespans = timespans.to(device, non_blocking=True)
        # The Phase 2 backbone-only route does not enter the split-graph DNC
        # branch below, where this diagnostic mask is normally initialized.
        # Keep it defined for all architectures so failure tracing cannot
        # mask the original non-finite activation with UnboundLocalError.
        sequence_valid_mask = None
        check_finite_this_step = True
        if check_finite_this_step:
            if not _first_nonfinite_report(input_seq, "input_seq", step):
                emit_failure_diagnostics("train")
                raise FloatingPointError(
                    f"[{run_id}] non-finite training input at step {step}; stopping."
                )
        hidden: Any = (None, None, None)
        optimizer.zero_grad(set_to_none=True)

        segment_diagnostics.set_channel("train")
        segment_diagnostics.set_step(step)
        if dnc_backward_trace and split_graph_enabled:
            rnn.memories[0].set_backward_trace(True, step=step)
        bistable_forward_debug = ""
        prediction_loss = None
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            if isinstance(rnn, SurpriseRecallBackboneOnly):
                oracle_flags = None
                if sgr_phase2_arm == "oracle":
                    oracle_flags = torch.zeros(
                        input_seq.shape[:2], device=device, dtype=torch.float32
                    )
                    for batch_index, example in enumerate(dataset.last_batch_examples):
                        for position, kind in zip(example.anomaly_positions, example.anomaly_types):
                            if kind in {"A1", "A2", "A4", "A5"}:
                                oracle_flags[batch_index, position] = 1.0
                output, hidden, backbone_features, prediction_logits = rnn(
                    input_seq,
                    hidden,
                    reset_experience=True,
                    return_backbone_features=True,
                    return_prediction_logits=True,
                    oracle_flags=oracle_flags,
                    stream_lengths=dataset.last_stream_lengths,
                )
                if prediction_logits is not None:
                    prediction_loss = masked_next_token_loss(
                        prediction_logits,
                        input_seq,
                        dataset.last_stream_lengths,
                    )
            elif split_graph_enabled:
                # Graph episodes are variable length and collate_fn pads them
                # with all-zero feature rows. This mask is diagnostic-only for
                # MoE routing: routing/capacity behavior remains unchanged.
                sequence_valid_mask = (
                    input_seq.ne(0).any(dim=-1) if moe_enabled and dataset_type == "graph" else None
                )
                if sgr_prediction_decoder is not None:
                    if sgr_phase4_enabled:
                        output, hidden, backbone_features, prediction_logits = rnn(
                            input_seq,
                            hidden,
                            reset_experience=True,
                            sequence_valid_mask=sequence_valid_mask,
                            timespans=timespans,
                            return_backbone_features=True,
                            return_prediction_logits=True,
                        )
                        prediction_loss = masked_next_token_loss(
                            prediction_logits, input_seq, dataset.last_stream_lengths
                        )
                    else:
                        output, hidden, backbone_features = rnn(
                            input_seq,
                            hidden,
                            reset_experience=True,
                            sequence_valid_mask=sequence_valid_mask,
                            timespans=timespans,
                            return_backbone_features=True,
                        )
                else:
                    output, hidden = rnn(
                        input_seq,
                        hidden,
                        reset_experience=True,
                        sequence_valid_mask=sequence_valid_mask,
                        timespans=timespans,
                    )
            else:
                output, hidden = rnn(input_seq, hidden, reset_experience=True)
            if sgr_phase2_arm is not None or sgr_phase4_enabled:
                last_train_sgr_signal_diagnostics = dict(
                    getattr(rnn, "last_sgr_signal_diagnostics", {})
                )
            bistable_forward_debug = format_bistable_forward_debug()
            if workspace_train_scope == "gate":
                exceeded, maximum, segment = segment_diagnostics.activation_absmax_exceeded("train")
                if exceeded:
                    emit_failure_diagnostics("train")
                    raise FloatingPointError(
                        f"[{run_id}] Phase 4 activation safety limit "
                        f"{workspace_stability_max_activation:.6g} exceeded in "
                        f"{segment} (observed absmax={maximum:.6g}) at step {step}; "
                        "stopping before backward/optimizer update."
                    )
            if split_graph_forward_trace is not None and split_graph_forward_trace.active:
                split_graph_forward_trace.set_time_index(None)
                split_graph_forward_trace.observe("rnn.output", output)
            if check_finite_this_step and not _first_nonfinite_report(output, "rnn_output", step):
                emit_failure_diagnostics("train")
                output = None
                hidden = None
                trace_split_graph_failure(
                    step,
                    "non-finite recurrent output",
                    input_seq,
                    target_digits,
                    answer_mask,
                    sequence_valid_mask,
                    timespans,
                )
                raise FloatingPointError(
                    f"[{run_id}] non-finite recurrent output at step {step}; "
                    "stopping before backward/optimizer update to protect the checkpoint."
                )
            output = output.transpose(0, 1)  # (B, T, 92); Linear accepts non-contiguous input.
            output = output_proj(output)  # (B, T, 90)
            if split_graph_forward_trace is not None and split_graph_forward_trace.active:
                split_graph_forward_trace.observe("task_output.logits", output)
            if check_finite_this_step and not _first_nonfinite_report(
                output, "output_logits", step
            ):
                emit_failure_diagnostics("train")
                output = None
                hidden = None
                trace_split_graph_failure(
                    step,
                    "non-finite task output logits",
                    input_seq,
                    target_digits,
                    answer_mask,
                    sequence_valid_mask,
                    timespans,
                )
                raise FloatingPointError(
                    f"[{run_id}] non-finite output logits at step {step}; "
                    "stopping before backward/optimizer update to protect the checkpoint."
                )
            task_loss = dataset.loss(output, target_digits, answer_mask)
            if split_graph_forward_trace is not None and split_graph_forward_trace.active:
                split_graph_forward_trace.observe("task_output.loss", task_loss)
            if (
                sgr_prediction_decoder is not None
                and not isinstance(rnn, SurpriseRecallBackboneOnly)
                and not sgr_phase4_enabled
            ):
                prediction_logits = sgr_prediction_decoder(backbone_features[:, :-1])
                prediction_loss = masked_next_token_loss(
                    prediction_logits,
                    input_seq,
                    dataset.last_stream_lengths,
                )

        if check_finite_this_step and not _first_nonfinite_report(task_loss, "task_loss", step):
            emit_failure_diagnostics("train")
            output = None
            task_loss = None
            hidden = None
            trace_split_graph_failure(
                step,
                "non-finite task loss",
                input_seq,
                target_digits,
                answer_mask,
                sequence_valid_mask,
                timespans,
            )
            raise FloatingPointError(
                f"[{run_id}] non-finite task loss at step {step}; "
                "stopping before backward/optimizer update to protect the checkpoint."
            )
        if (
            prediction_loss is not None
            and check_finite_this_step
            and not _first_nonfinite_report(prediction_loss, "sgr_prediction_loss", step)
        ):
            emit_failure_diagnostics("train")
            prediction_loss = None
            output = None
            hidden = None
            trace_split_graph_failure(
                step,
                "non-finite SGR prediction loss",
                input_seq,
                target_digits,
                answer_mask,
                sequence_valid_mask,
                timespans,
            )
            raise FloatingPointError(
                f"[{run_id}] non-finite SGR prediction loss at step {step}; stopping."
            )
        kl_loss, kl_diag = pop_total_kl(stochastic_heads, free_bits=FREE_BITS)  # Phase 1
        if check_finite_this_step and not _first_nonfinite_report(kl_loss, "kl_loss", step):
            emit_failure_diagnostics("train")
            raise FloatingPointError(
                f"[{run_id}] non-finite KL loss at step {step}; stopping before backward."
            )
        moe_aux_loss, moe_diag = (
            pop_total_moe_aux_loss(
                rnn.moe_layers,
                include_diagnostics=(
                    moe_diagnostics_enabled and running_log_steps + 1 >= log_every
                ),
            )
            if moe_enabled
            else (torch.zeros((), device=device), {})
        )  # v8 (Option 4): Switch-style load-balancing loss, already scaled
        # by moe_load_balance_alpha inside SwitchMoE.pop_aux_loss() -- no
        # extra weighting applied here, unlike beta_eff*kl_loss below.
        if check_finite_this_step and not _first_nonfinite_report(
            moe_aux_loss, "moe_aux_loss", step
        ):
            emit_failure_diagnostics("train")
            raise FloatingPointError(
                f"[{run_id}] non-finite MoE auxiliary loss at step {step}; "
                "stopping before backward."
            )
        global_ramp = min(1.0, (step - anneal_start_step) / max(1, KL_ANNEAL_STEPS))
        lesson_dip_ramp = min(1.0, (step - lesson_dip_start_step) / max(1, LESSON_KL_DIP_STEPS))
        beta_eff = beta_target * global_ramp * lesson_dip_ramp
        loss = (
            task_loss
            + (sgr_prediction_loss_weight * prediction_loss if prediction_loss is not None else 0.0)
            + beta_eff * kl_loss
            + moe_aux_loss
        )
        if split_graph_forward_trace is not None and split_graph_forward_trace.active:
            split_graph_forward_trace.observe("training.total_loss", loss)
        if check_finite_this_step and not _first_nonfinite_report(loss, "total_loss", step):
            emit_failure_diagnostics("train")
            raise FloatingPointError(
                f"[{run_id}] non-finite total loss at step {step}; stopping before backward."
            )
        if split_graph_forward_trace is not None:
            split_graph_forward_trace.end_step(step)
        try:
            scaler.scale(loss).backward()
        except RuntimeError:
            if anomaly_this_step:
                # Anomaly detection raises as soon as one backward node emits
                # NaN, before the normal post-backward failure path can flush
                # module-boundary summaries. Emit the hooks collected up to
                # that point while they still carry this step's evidence.
                emit_failure_diagnostics("train", include_gradients=True)
                segment_diagnostics.set_nonfinite_trace(False)
            raise
        if anomaly_this_step:
            torch.autograd.set_detect_anomaly(False)
            segment_diagnostics.set_nonfinite_trace(False)
        if dnc_trace_this_step and isinstance(rnn, SplitGraphDNC):
            rnn.memories[0].set_backward_trace(False, step=step)
        scaler.unscale_(optimizer)
        bistable_debug_gradient_stats: dict[str, tuple[float, float, int]] = {}
        bistable_debug_parameter_before: dict[str, torch.Tensor] = {}
        if workspace_bistable_debug and isinstance(rnn, SplitGraphDNC):
            assert rnn.workspace is not None and rnn.workspace.bistable_gate is not None
            for name, parameter in rnn.workspace.bistable_gate.named_parameters():
                bistable_debug_parameter_before[name] = parameter.detach().clone()
                gradient = parameter.grad
                if gradient is None:
                    bistable_debug_gradient_stats[name] = (0.0, 0.0, 0)
                    continue
                finite_mask = torch.isfinite(gradient)
                finite_gradient = torch.where(finite_mask, gradient, 0.0).detach().float()
                finite_count = int(finite_mask.sum().item())
                nonfinite_count = gradient.numel() - finite_count
                norm = float(torch.linalg.vector_norm(finite_gradient.to(torch.float64)).cpu())
                absmax = float(finite_gradient.abs().max().cpu())
                bistable_debug_gradient_stats[name] = (
                    norm,
                    absmax,
                    nonfinite_count,
                )
        independent_grad_norm = None
        if bistable_grad_probe is not None:
            step_parameter_norms = []
            for name, parameter in trainable_named_parameters:
                if parameter.grad is None:
                    continue
                parameter_norm = torch.linalg.vector_norm(
                    parameter.grad.detach().to(dtype=torch.float64)
                )
                probe = bistable_grad_probe[name]
                probe["sum_sq"].add_(parameter_norm.square())
                probe["max_step"].copy_(
                    torch.where(
                        parameter_norm > probe["max"],
                        torch.full_like(probe["max_step"], step),
                        probe["max_step"],
                    )
                )
                probe["max"].copy_(torch.maximum(probe["max"], parameter_norm))
                step_parameter_norms.append(parameter_norm)
            if step_parameter_norms:
                independent_grad_norm = torch.linalg.vector_norm(torch.stack(step_parameter_norms))
                bistable_grad_probe_global_max_step.copy_(
                    torch.where(
                        independent_grad_norm > bistable_grad_probe_global_max,
                        torch.full_like(bistable_grad_probe_global_max_step, step),
                        bistable_grad_probe_global_max_step,
                    )
                )
                bistable_grad_probe_global_max.copy_(
                    torch.maximum(bistable_grad_probe_global_max, independent_grad_norm)
                )
                bistable_grad_probe_steps += 1
        bistable_gate_grad_norm = 0.0
        if workspace_gate_variant == "bistable":
            assert isinstance(rnn, SplitGraphDNC) and rnn.workspace is not None
            assert rnn.workspace.bistable_gate is not None
            gate_gradient_items = [
                (name, parameter.grad.detach())
                for name, parameter in rnn.workspace.bistable_gate.named_parameters()
                if parameter.grad is not None
            ]
            if not gate_gradient_items:
                if workspace_bistable_debug:
                    report_bistable_nonfinite(
                        step,
                        bistable_forward_debug,
                        "no bistable gate parameter received a gradient",
                    )
                emit_failure_diagnostics("train", include_gradients=True)
                raise FloatingPointError(
                    f"[{run_id}] bistable write gate received no gradients at step {step}."
                )
            gate_gradients = [gradient for _, gradient in gate_gradient_items]
            bistable_gate_grad_norm = float(
                torch.linalg.vector_norm(
                    torch.cat(
                        [gradient.to(torch.float64).reshape(-1) for gradient in gate_gradients]
                    )
                ).cpu()
            )
            if not math.isfinite(bistable_gate_grad_norm) or bistable_gate_grad_norm == 0.0:
                gradient_debug_parts = []
                for name, gradient in gate_gradient_items:
                    finite = torch.isfinite(gradient)
                    finite_values = gradient[finite]
                    finite_norm = (
                        float(torch.linalg.vector_norm(finite_values.to(torch.float64)).cpu())
                        if finite_values.numel()
                        else 0.0
                    )
                    finite_absmax = (
                        float(finite_values.abs().max().cpu()) if finite_values.numel() else 0.0
                    )
                    bad_count = gradient.numel() - finite_values.numel()
                    gradient_debug_parts.append(
                        f"{name}:finite_norm={finite_norm:.3g},"
                        f"finite_absmax={finite_absmax:.3g},nonfinite={bad_count}"
                    )
                gradient_debug = " ".join(gradient_debug_parts)
                if workspace_bistable_debug:
                    report_bistable_nonfinite(step, bistable_forward_debug, gradient_debug)
                emit_failure_diagnostics("train", include_gradients=True)
                raise FloatingPointError(
                    f"[{run_id}] bistable write gate gradient norm is "
                    f"{bistable_gate_grad_norm} at step {step}; "
                    f"parameter-gradient details: {gradient_debug}."
                )
        # Log addition #2: capture the pre-clip gradient norm instead of
        # discarding clip_grad_norm_'s return value. Gives a β=0 baseline
        # gradient-scale trajectory to compare a future β>0 switch-in
        # against, useful for telling a genuine instability apart from the
        # expected one-off "shock" when sampling noise is switched on.
        # Clip every parameter in this optimizer, including output_proj.
        # Clipping only rnn.parameters() left the trainable prediction head
        # unbounded while its gradients were still being applied by Adam.
        if bistable_grad_probe is not None:
            # Gate-only bistable runs have three scalar trainables. Their raw
            # gradients can be large but finite; clipping_grad_norm_'s usual
            # FP32 sum-of-squares can overflow before it gets a chance to
            # rescale them. Compute the norm in FP64 and scale the original
            # gradients directly so Adam never sees the unbounded values.
            grad_norm = independent_grad_norm
            if grad_norm is None:
                grad_norm = torch.zeros((), device=device, dtype=torch.float64)
            clip_coefficient = (max_grad_norm / grad_norm.clamp_min(1e-12)).clamp(max=1.0)
            for parameter in trainable_parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(clip_coefficient.to(parameter.grad.dtype))
            postclip_grad_norm = grad_norm * clip_coefficient
        else:
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, max_norm=max_grad_norm, error_if_nonfinite=True
                )
                # clip_grad_norm_ scales every gradient by this shared
                # coefficient. Derive the resulting norm from its returned
                # pre-clip norm instead of making another full pass over all
                # model gradients on every training step.
                clip_coefficient = (max_grad_norm / (grad_norm + 1e-6)).clamp(max=1.0)
                postclip_grad_norm = grad_norm * clip_coefficient
            except RuntimeError as error:
                if "non-finite" not in str(error):
                    raise
                # error_if_nonfinite raises before applying the clip coefficient,
                # preserving the original bad gradients for the diagnostic below.
                grad_norm = torch.tensor(float("nan"), device=device)
                postclip_grad_norm = torch.tensor(float("nan"), device=device)
        grad_norm_value = float(grad_norm)
        postclip_grad_norm_value = float(postclip_grad_norm)
        bistable_optimizer_updated = False
        if independent_grad_norm is not None and math.isfinite(grad_norm_value):
            relative_error = (
                independent_grad_norm - grad_norm.detach().to(dtype=torch.float64)
            ).abs() / independent_grad_norm.clamp_min(1e-12)
            bistable_grad_probe_relative_error.copy_(
                torch.maximum(bistable_grad_probe_relative_error, relative_error)
            )
        if not math.isfinite(grad_norm_value):
            if workspace_bistable_debug:
                gradient_debug = " ".join(
                    f"{name}:norm={norm:.3g},absmax={absmax:.3g},nonfinite={bad}"
                    for name, (norm, absmax, bad) in bistable_debug_gradient_stats.items()
                )
                report_bistable_nonfinite(step, bistable_forward_debug, gradient_debug)
            grad_was_finite_since_eval = False
            consecutive_nonfinite_grad_steps += 1
            running_nonfinite_grad_steps += 1
            _report_first_nonfinite_gradient(
                [(f"rnn.{name}", parameter) for name, parameter in rnn.named_parameters()]
                + [
                    (f"output_proj.{name}", parameter)
                    for name, parameter in output_proj.named_parameters()
                ],
                step,
            )
            emit_failure_diagnostics("train", include_gradients=True)
            # Skip this optimizer update. GradScaler's found_inf scan happens
            # during unscale_; the explicit finite-norm check also covers
            # non-finite gradients that arise after that scan.
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            if consecutive_nonfinite_grad_steps >= 8:
                raise FloatingPointError(
                    f"[{run_id}] {consecutive_nonfinite_grad_steps} consecutive optimizer "
                    f"steps had non-finite gradients (AMP scale={scaler.get_scale():.4g}); "
                    "stopping rather than silently training without parameter updates."
                )
        else:
            consecutive_nonfinite_grad_steps = 0
            scaler.step(optimizer)
            scaler.update()
            bistable_optimizer_updated = True
        if workspace_bistable_debug and isinstance(rnn, SplitGraphDNC):
            assert rnn.workspace is not None and rnn.workspace.bistable_gate is not None
            clip_scale = (
                min(1.0, max_grad_norm / max(grad_norm_value, 1e-12))
                if math.isfinite(grad_norm_value)
                else float("nan")
            )
            parameter_reports = []
            for name, parameter in rnn.workspace.bistable_gate.named_parameters():
                grad_norm_raw, grad_absmax, bad_count = bistable_debug_gradient_stats.get(
                    name, (0.0, 0.0, 0)
                )
                before = bistable_debug_parameter_before.get(name, parameter.detach())
                after = parameter.detach()
                delta = after - before
                state = optimizer.state.get(parameter, {})
                exp_avg = state.get("exp_avg")
                exp_avg_sq = state.get("exp_avg_sq")
                step_state = state.get("step")
                adam_step = (
                    int(step_state.item()) if torch.is_tensor(step_state) else int(step_state or 0)
                )
                m_absmax = float(exp_avg.detach().abs().max().cpu()) if exp_avg is not None else 0.0
                v_max = float(exp_avg_sq.detach().max().cpu()) if exp_avg_sq is not None else 0.0
                parameter_reports.append(
                    f"{name}[v={float(after.cpu()):.4g},g_norm_raw={grad_norm_raw:.3g},"
                    f"g_absmax_raw={grad_absmax:.3g},g_nonfinite={bad_count},"
                    f"delta={float(delta.cpu()):.3g},m_absmax={m_absmax:.3g},"
                    f"v_max={v_max:.3g},adam_step={adam_step}]"
                )
            print(
                f"[{run_id}] [bistable-debug:update] step={step} "
                f"loss={float(loss.detach().cpu()):.5g} "
                f"task={float(task_loss.detach().cpu()):.5g} "
                f"kl={float(kl_loss.detach().cpu()):.5g} beta={beta_eff:.4g} "
                f"moe_aux={float(moe_aux_loss.detach().cpu()):.5g} "
                f"lr={optimizer.param_groups[0]['lr']:.5g} "
                f"preclip_global_norm={grad_norm_value:.4g} "
                f"clip_scale={clip_scale:.4g} max_grad_norm={max_grad_norm:g} "
                f"updated={bistable_optimizer_updated} amp_scale={scaler.get_scale():.4g} "
                f"forward[{bistable_forward_debug or 'unavailable'}] "
                f"parameters[{'; '.join(parameter_reports)}]"
            )
            # Segment hooks collect this optimizer step's per-call gate/slot,
            # controller, and DNC backward statistics. Emitting here gives a
            # step-resolved view, while the normal 100-step aggregation stays
            # unchanged for non-debug runs.
            segment_diagnostics.emit("train", include_gradients=True)
        # Log addition #3: AMP loss-scale value. A collapsing/repeatedly
        # halved scale is the standard AMP symptom of inf/NaN gradients --
        # the exp()/logvar path in the stochastic write head is the most
        # numerically fragile part of this pipeline, so a clean β=0 baseline
        # of scaler behavior is useful context once β>0 introduces that path.
        amp_scale = scaler.get_scale()
        running_task_loss += task_loss.detach()
        running_kl_loss += kl_loss.detach()
        if prediction_loss is not None:
            running_sgr_prediction_loss += prediction_loss.detach()
        running_div += dataset.diversity(output.detach(), target_digits, answer_mask)
        if math.isfinite(grad_norm_value):
            running_grad_norm += grad_norm_value
            running_postclip_grad_norm += postclip_grad_norm_value
            running_finite_grad_steps += 1
            if grad_norm_value > max_grad_norm:
                running_clipped_grad_steps += 1
        if workspace_reconsideration_window is not None:
            assert isinstance(rnn, SplitGraphDNC) and rnn.workspace is not None
            step_workspace_diag = rnn.workspace.last_diagnostics()
            for name, total in workspace_reconsideration_window.items():
                value = step_workspace_diag.get(name)
                if value is not None:
                    total.add_(value.detach())
        step += 1
        running_log_steps += 1
        current_lr = set_lr(step)  # Phase 1 FIX: clamped manual LR, replaces scheduler.step()

        # Dynamic-N (usage-triggered), macro-scale: record this just-finished
        # episode's memory-usage saturation and, if the trigger has been
        # sustained past cooldown, resize nr_cells before the NEXT episode's
        # forward pass. Placed here -- after scaler.step()/scaler.update()
        # above, using `hidden` from the forward pass that just completed --
        # so a resize never touches a live autograd graph: this episode's
        # backward pass and optimizer step are both already done, and the
        # very next `hidden = (None, None, None)` at the top of the loop
        # means nothing carries the old size forward anyway (same "no
        # cross-episode memory content to preserve" fact static Option 2
        # relies on -- see dynamic_memory_resize.py's module docstring).
        # This record()/should_grow() split (see dynamic_n_controller.py) is
        # deliberately timing-agnostic -- it's what lets a future mid-episode
        # (micro-scale) variant reuse the same decision core fed from a
        # per-timestep usage reading instead, without redesigning the
        # EMA/cooldown logic.
        if dynamic_n_ctrl is not None:
            with torch.no_grad():
                # mem_hidden's shape differs by controller: MambaDNC's
                # _init_hidden returns mhx as a bare dict for rnn_type!=
                # 'mamba' (deferring to stock dnc.DNC._init_hidden, share_
                # memory_between_layers=True case), but wraps it in a
                # single-entry list for rnn_type=='mamba' (see
                # mamba_controller.py's _init_hidden memory-state branch,
                # which is otherwise byte-identical to upstream). Both cases
                # are layer 0 of a single shared Memory in this project, so
                # this just picks the right unwrap for whichever controller
                # is active rather than hardcoding one shape.
                mem_hidden = hidden[1]
                mhx = mem_hidden[0] if isinstance(mem_hidden, list) else mem_hidden
                usage = mhx["usage_vector"].float()
                frac_saturated = (usage > DYNAMIC_N_USAGE_HIGH).float().mean().item()
            dynamic_n_ctrl.record(frac_saturated)
            last_frac_saturated = frac_saturated
            current_nr_cells = rnn.memories[0].nr_cells
            new_n = dynamic_n_ctrl.should_grow(current_nr_cells)
            if new_n is not None:
                ema_at_trigger = dynamic_n_ctrl.ema  # captured before mark_grown() resets it to 0.0
                resize_memory(rnn, new_n, device=device, optimizer=optimizer)
                dynamic_n_ctrl.mark_grown(step, current_nr_cells, new_n)
                print(
                    f"[{run_id}] Step {step} Dynamic-N growth: nr_cells "
                    f"{current_nr_cells} -> {new_n} (usage-saturation EMA "
                    f"{ema_at_trigger:.3f} had reached the "
                    f"{dynamic_n_ctrl.trigger_frac:.2f} trigger, sustained "
                    f"past cooldown)"
                )
                assert dynamic_n_log_writer is not None and dynamic_n_log_file is not None
                dynamic_n_log_writer.writerow(
                    [
                        step,
                        curriculum.lesson + 1,
                        current_nr_cells,
                        new_n,
                        ema_at_trigger,
                    ]
                )
                dynamic_n_log_file.flush()

        if running_log_steps >= log_every:
            window_steps = running_log_steps
            elapsed = time.time() - t0
            t0 = time.time()
            total_elapsed = time.time() - t_run_start  # log addition #5
            avg_task = (
                running_task_loss.item()
                if torch.is_tensor(running_task_loss)
                else running_task_loss
            ) / window_steps
            avg_sgr_prediction_loss = (
                (
                    running_sgr_prediction_loss.item()
                    if torch.is_tensor(running_sgr_prediction_loss)
                    else running_sgr_prediction_loss
                )
                / window_steps
                if sgr_prediction_decoder is not None
                else None
            )
            if surprise_recall_logger is not None:
                surprise_recall_logger.write_training_batch(
                    step,
                    curriculum.lesson + 1,
                    dataset.last_batch_examples,
                    task_loss=avg_task,
                    prediction_loss=avg_sgr_prediction_loss,
                )
            avg_kl = (
                running_kl_loss.item() if torch.is_tensor(running_kl_loss) else running_kl_loss
            ) / window_steps
            avg_div = running_div / window_steps
            avg_grad_norm = running_grad_norm / max(running_finite_grad_steps, 1)
            avg_postclip_grad_norm = running_postclip_grad_norm / max(running_finite_grad_steps, 1)
            kl_contrib = (
                beta_eff * avg_kl
            )  # actual beta*L_KL added to total loss, vs. avg_kl (pre-beta, raw clamped sum)
            gpu_mem_peak_mb = (
                torch.cuda.max_memory_allocated(device) / 1e6 if torch.cuda.is_available() else 0.0
            )
            dyn_n_str = ""
            if dynamic_n_ctrl is not None:
                dyn_n_str = (
                    f"| dynN[nr_cells {rnn.memories[0].nr_cells} frac_sat {last_frac_saturated:.3f} "
                    f"ema {dynamic_n_ctrl.ema:.3f} cooldown {dynamic_n_ctrl.cooldown_remaining}] "
                )

            parts = [
                f"[{run_id}] Step {step}/{total_steps} | Lesson {curriculum.lesson + 1}/{len(curriculum.table)}",
                f"L_task {avg_task:.4f}",
            ]
            if avg_sgr_prediction_loss is not None:
                parts.append(f"L_pred {avg_sgr_prediction_loss:.4f}")
            sgr_signal_values = {
                key: float(value.detach().float().item())
                for key, value in last_train_sgr_signal_diagnostics.items()
            }
            if sgr_phase2_arm is not None or sgr_phase4_enabled:
                parts.append(
                    "sgr_error["
                    f"mean={sgr_signal_values.get('mean', 0.0):.3f} "
                    f"abs_mean={sgr_signal_values.get('abs_mean', 0.0):.3f} "
                    f"positive={sgr_signal_values.get('positive_frac', 0.0):.1%} "
                    f"gate_mean={sgr_signal_values.get('gate_mean', 0.0):.3f}]"
                )
                if sgr_dnc_write_salience:
                    parts.append(
                        "dnc_error_write_scale="
                        f"{rnn.memories[0].sgr_error_to_write_strength.detach().item():.4g}"
                    )
            if kl_on:
                parts += [
                    f"L_KL {avg_kl:.4f}",
                    f"beta {beta_eff:.4f}",
                    f"KL_contrib {kl_contrib:.4f}",
                    f"KL[mean {kl_diag['kl_mean']:.4f} max {kl_diag['kl_max']:.4f}]",
                    f"clamp_frac {kl_diag['clamp_frac']:.4f}",
                    f"floor_frac {kl_diag['floor_frac']:.4f}",
                    f"snapshot_step {kl_diag['snapshot_step']}",
                ]
            parts += [
                f"diversity {avg_div:.2f}",
                f"{window_steps / elapsed:.2f} steps/s",
                f"LR {current_lr:.6f}",
                f"grad_norm {avg_grad_norm:.4g}",
                f"postclip_grad_norm {avg_postclip_grad_norm:.4g}",
                f"nonfinite_grad_steps {running_nonfinite_grad_steps}",
                f"clipped_grad_steps {running_clipped_grad_steps}/{running_finite_grad_steps}",
            ]
            if amp_enabled:
                parts.append(f"amp_scale {amp_scale:.1f}")
            parts.append(f"gpu_mem_peak_mb {gpu_mem_peak_mb:.1f}")
            if moe_enabled and moe_diagnostics_enabled:
                parts.append(f"moe_aux {float(moe_aux_loss.detach()):.4f}")
                parts += [
                    f"moe_cv_load {moe_diag.get('moe_cv_load', 0.0):.4f}",
                    f"moe_max_load_frac {moe_diag.get('moe_max_load_frac', 0.0):.4f}",
                    f"moe_capacity_drop {moe_diag.get('moe_capacity_drop_frac', 0.0):.1%}",
                ]
                parts.extend(_format_moe_layer_reports(moe_diag))
            workspace_diag: dict[str, Any] = {}
            if workspace_enabled and isinstance(rnn, SplitGraphDNC):
                if rnn.workspace is None:
                    parts.append("workspace[bypassed: slots=0]")
                else:
                    workspace_diag = rnn.workspace.last_diagnostics()
                    if workspace_diag:
                        parts.append(
                            "workspace["
                            f"write_mass {workspace_diag['write_mass'].item():.3f} "
                            f"active_writes {workspace_diag['active_write_frac'].item():.1%} "
                            f"write_gate {workspace_diag['write_gate_mean'].item():.3f} "
                            f"slot_rms {workspace_diag['slot_rms'].item():.3f} "
                            f"broadcast/specialist {workspace_diag['broadcast_to_specialist_ratio'].item():.3f} "
                            f"read_gate {workspace_diag['read_gate'].item():.3f}]"
                        )
                        if "prediction_error_write_scale_rms" in workspace_diag:
                            parts.append(
                                "sgr_workspace_write_scale_rms="
                                f"{workspace_diag['prediction_error_write_scale_rms'].item():.4g}"
                            )
                        if workspace_reconsideration_enabled:
                            parts.append(
                                "reconsideration["
                                f"eligible "
                                f"{workspace_diag['reconsideration_eligible_frac'].item():.1%} "
                                f"surprise "
                                f"{workspace_diag['reconsideration_surprise_mean'].item():.3f} "
                                f"trigger "
                                f"{workspace_diag['reconsideration_trigger_frac'].item():.1%} "
                                f"changed "
                                f"{workspace_diag['reconsideration_changed_frac'].item():.1%} "
                                f"delta_rms "
                                f"{workspace_diag['reconsideration_state_delta_rms'].item():.3f}]"
                            )
                            if "prediction_error_mean" in workspace_diag:
                                parts.append(
                                    "sgr_reconsideration_error["
                                    f"raw={workspace_diag['prediction_error_mean'].item():.3f} "
                                    f"ema={workspace_diag['reconsideration_prediction_error_mean'].item():.3f}]"
                                )
                        if "gate_low_frac" in workspace_diag:
                            print(
                                f"[{run_id}] [bistable] p_mean="
                                f"{workspace_diag['gate_probability_mean'].item():.3f} "
                                f"low/mid/high="
                                f"{workspace_diag['gate_low_frac'].item():.1%}/"
                                f"{workspace_diag['gate_mid_frac'].item():.1%}/"
                                f"{workspace_diag['gate_high_frac'].item():.1%} "
                                f"commit_freq={workspace_diag['write_frequency'].item():.1%} "
                                f"selection_peak="
                                f"{workspace_diag['selection_strength_mean'].item():.3f} "
                                f"gate_grad_norm={bistable_gate_grad_norm:.3g}"
                            )
                            if "gate_histogram" in workspace_diag:
                                histogram = ",".join(
                                    f"{value:.2f}"
                                    for value in workspace_diag["gate_histogram"].tolist()
                                )
                                print(
                                    f"[{run_id}] [bistable] probability_histogram_10bin={histogram}"
                                )
                            if "writer_frequency" in workspace_diag:
                                writer_frequency = ",".join(
                                    f"{value:.3f}"
                                    for value in workspace_diag["writer_frequency"].tolist()
                                )
                                print(f"[{run_id}] [workspace] writer_frequency={writer_frequency}")
            if dyn_n_str:
                parts.append(dyn_n_str.strip(" |"))
            if bistable_grad_probe is not None:
                probe_steps = max(bistable_grad_probe_steps, 1)
                per_parameter: list[tuple[float, str, float, int]] = []
                for name, probe in bistable_grad_probe.items():
                    rms = (probe["sum_sq"] / probe_steps).sqrt()
                    maximum_tensor = probe["max"]
                    maximum_step = int(probe["max_step"].cpu())
                    per_parameter.append(
                        (
                            float(maximum_tensor.cpu()),
                            name,
                            float(rms.cpu()),
                            maximum_step,
                        )
                    )
                per_parameter.sort(reverse=True)
                parameter_summary = (
                    ", ".join(
                        f"{name}:rms={rms:.3g},max={maximum:.3g}@step{maximum_step}"
                        for maximum, name, rms, maximum_step in per_parameter[:8]
                    )
                    or "none"
                )
                parts.append(
                    "bistable_grad_probe["
                    f"trainable={len(trainable_named_parameters)} steps={bistable_grad_probe_steps} "
                    f"independent_global_max={float(bistable_grad_probe_global_max.cpu()):.3g}"
                    f"@step{int(bistable_grad_probe_global_max_step.cpu())} "
                    f"max_norm_relative_error={float(bistable_grad_probe_relative_error.cpu()):.3g} "
                    f"parameters={parameter_summary}]"
                )
                for probe in bistable_grad_probe.values():
                    probe["sum_sq"].zero_()
                    probe["max"].zero_()
                bistable_grad_probe_steps = 0
                bistable_grad_probe_global_max.zero_()
                bistable_grad_probe_global_max_step.fill_(-1)
                bistable_grad_probe_relative_error.zero_()
            print(" | ".join(parts))
            if workspace_telemetry_enabled:
                telemetry_parts = [
                    f"[{run_id}] [workspace-telemetry] step={step} window_steps={window_steps}"
                ]
                eligible_count = None
                trigger_count = None
                changed_count = None
                trigger_rate = None
                changed_given_trigger = None
                changed_of_eligible = None
                state_delta_rms = None
                if workspace_reconsideration_window is not None:
                    count_values = (
                        torch.stack(
                            [
                                workspace_reconsideration_window[name]
                                for name in (
                                    "reconsideration_eligible_count",
                                    "reconsideration_trigger_count",
                                    "reconsideration_changed_count",
                                    "reconsideration_state_delta_sq_sum",
                                )
                            ]
                        )
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    eligible_count, trigger_count, changed_count, state_delta_sq_sum = (
                        float(value) for value in count_values
                    )
                    trigger_rate = trigger_count / max(eligible_count, 1.0)
                    changed_given_trigger = changed_count / max(trigger_count, 1.0)
                    changed_of_eligible = changed_count / max(eligible_count, 1.0)
                    state_delta_rms = math.sqrt(state_delta_sq_sum / max(trigger_count, 1.0))
                    telemetry_parts.append(
                        f"reconsideration_trigger={workspace_reconsideration_trigger_mode} "
                        f"eligible_slots={eligible_count:.0f} "
                        f"triggered_slots={trigger_count:.0f} "
                        f"trigger_rate={trigger_rate:.2%} "
                        f"changed_slots={changed_count:.0f} "
                        f"changed_given_trigger={changed_given_trigger:.2%} "
                        f"changed_of_eligible={changed_of_eligible:.2%} "
                        f"state_delta_rms={state_delta_rms:.4g}"
                    )
                write_scale_rms = None
                write_scale_absmax = None
                bias_rms = None
                if (
                    isinstance(rnn, SplitGraphDNC)
                    and rnn.workspace is not None
                    and rnn.workspace.prediction_error_to_write_gate is not None
                ):
                    write_gate = rnn.workspace.prediction_error_to_write_gate
                    weight = write_gate.weight.detach().float()
                    write_scale_rms = weight.square().mean().sqrt().item()
                    write_scale_absmax = weight.abs().max().item()
                    bias = getattr(write_gate, "bias", None)
                    bias_rms = (
                        bias.detach().float().square().mean().sqrt().item()
                        if bias is not None
                        else 0.0
                    )
                    telemetry_parts.append(
                        f"write_salience_weight_rms={write_scale_rms:.4g} "
                        f"weight_absmax={write_scale_absmax:.4g} "
                        f"bias_rms={bias_rms:.4g}"
                    )
                elif sgr_workspace_write_salience:
                    telemetry_parts.append("write_salience=enabled_but_projection_missing")
                else:
                    write_scale_rms = None
                    write_scale_absmax = None
                    bias_rms = None
                    telemetry_parts.append("write_salience=off")
                print(" | ".join(telemetry_parts))
                if workspace_telemetry_log_writer is not None:
                    prediction_error_mean = workspace_diag.get("prediction_error_mean")
                    reconsideration_error_ema_mean = workspace_diag.get(
                        "reconsideration_prediction_error_mean"
                    )
                    workspace_telemetry_log_writer.writerow(
                        [
                            step,
                            curriculum.lesson + 1,
                            window_steps,
                            (
                                workspace_reconsideration_trigger_mode
                                if workspace_reconsideration_enabled
                                else None
                            ),
                            eligible_count,
                            trigger_count,
                            trigger_rate,
                            changed_count,
                            changed_given_trigger,
                            changed_of_eligible,
                            state_delta_rms,
                            (
                                float(prediction_error_mean.item())
                                if torch.is_tensor(prediction_error_mean)
                                else None
                            ),
                            (
                                float(reconsideration_error_ema_mean.item())
                                if torch.is_tensor(reconsideration_error_ema_mean)
                                else None
                            ),
                            write_scale_rms,
                            write_scale_absmax,
                            bias_rms,
                        ]
                    )
                    assert workspace_telemetry_log_file is not None
                    workspace_telemetry_log_file.flush()
            if not workspace_bistable_debug:
                segment_diagnostics.emit("train", include_gradients=True)
            if dnc_backward_trace and split_graph_enabled:
                memory_diag = rnn.memories[0].pop_numerical_diagnostics()
                if memory_diag:
                    print(
                        f"[dnc-numerics:train] step={step} "
                        + " ".join(f"{name}={value:.4g}" for name, value in memory_diag.items())
                    )

            if (
                track_b_writer is not None
                and track_b_file is not None
                and dnc_write_variant == "kanerva-closed-form"
            ):
                train_total_loss = avg_task + kl_contrib
                track_b_writer.writerow(
                    [
                        "training_window",
                        step,
                        curriculum.lesson + 1,
                        dnc_write_variant,
                        None,
                        None,
                        None,
                        *([None] * 28),
                        avg_task,
                        avg_kl,
                        train_total_loss,
                        avg_grad_norm,
                        amp_scale,
                        current_lr,
                    ]
                )
                track_b_file.flush()
                print(
                    f"[{run_id}] [Track B / train] task/total loss="
                    f"{avg_task:.5f}/{train_total_loss:.5f} | "
                    f"grad_norm={avg_grad_norm:.4f} | amp_scale={amp_scale:.1f} | "
                    f"lr={current_lr:.6g}"
                )

            log_writer.writerow(
                [
                    step,
                    curriculum.lesson + 1,
                    beta_eff,
                    avg_task,
                    avg_kl,
                    kl_contrib,
                    avg_task + kl_contrib,
                    avg_div,
                    kl_diag["kl_mean"],
                    kl_diag["kl_max"],
                    kl_diag["kl_min"],
                    kl_diag["kl_std"],
                    kl_diag["clamp_frac"],
                    current_lr,
                    avg_grad_norm,
                    amp_scale,
                    total_elapsed,
                    kl_diag["snapshot_step"],
                    gpu_mem_peak_mb,
                    param_count,
                    float(moe_aux_loss.detach()),
                    moe_diag.get("moe_cv_importance", 0.0),
                    moe_diag.get("moe_cv_load", 0.0),
                    moe_diag.get("moe_max_load_frac", 0.0),
                    moe_diag.get("moe_capacity_drop_frac", 0.0),
                    float(workspace_diag.get("reconsideration_eligible_frac", 0.0)),
                    float(workspace_diag.get("reconsideration_surprise_mean", 0.0)),
                    float(workspace_diag.get("reconsideration_trigger_frac", 0.0)),
                    float(workspace_diag.get("reconsideration_changed_frac", 0.0)),
                    float(workspace_diag.get("reconsideration_state_delta_rms", 0.0)),
                    avg_sgr_prediction_loss,
                    sgr_signal_values.get("mean"),
                    sgr_signal_values.get("abs_mean"),
                    sgr_signal_values.get("positive_frac"),
                    sgr_signal_values.get("gate_mean"),
                    (
                        float(rnn.memories[0].sgr_error_to_write_strength.detach().item())
                        if sgr_dnc_write_salience
                        else None
                    ),
                    (
                        float(workspace_diag["prediction_error_write_scale_rms"].item())
                        if "prediction_error_write_scale_rms" in workspace_diag
                        else None
                    ),
                ]
            )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)  # so next window's peak isn't cumulative
            log_file.flush()
            running_task_loss, running_kl_loss, running_div, running_grad_norm = 0.0, 0.0, 0, 0.0
            running_sgr_prediction_loss = 0.0
            running_log_steps = 0
            running_postclip_grad_norm = 0.0
            running_finite_grad_steps = 0
            running_nonfinite_grad_steps = 0
            running_clipped_grad_steps = 0
            if workspace_reconsideration_window is not None:
                for total in workspace_reconsideration_window.values():
                    total.zero_()

        if kl_on and step % PRIOR_SNAPSHOT_EVERY == 0:
            # v5 (Phase 2): refit (mu_g, Sigma_g) from the writes
            # accumulated since the last snapshot, freeze the result into
            # every stochastic write head's buffers, and log a summary.
            # This call -- and only this call -- is where the prior's
            # non-locality lives (see stochastic_write_head.py v2,
            # update_prior_snapshot); it happens here, outside the
            # per-step backward pass above, never inside it. This is a
            # brand-new console line and a brand-new log file -- it does
            # not touch the per-step console line above, or any other
            # existing print/log call in this loop.
            snap_diag = update_all_prior_snapshots(
                stochastic_heads,
                step,
                min_logvar=PRIOR_MIN_LOGVAR,
                max_logvar=PRIOR_MAX_LOGVAR,
            )
            print(
                f"[{run_id}] Step {step} prior snapshot updated | "
                f"||mu_g|| {snap_diag['mu_g_norm']:.4f} | "
                f"diag(Sigma_g) mean/min/max {snap_diag['sigma_g_mean']:.4f}/"
                f"{snap_diag['sigma_g_min']:.4f}/{snap_diag['sigma_g_max']:.4f} | "
                f"trace(Sigma_g) {snap_diag['trace_sigma_g']:.4f} | "
                f"n_samples {snap_diag['n_samples']} "
                f" | raw_var mean/max {snap_diag['raw_var_mean']:.3f}/{snap_diag['raw_var_max']:.3f} hi_frac {snap_diag['raw_hi_frac']:.2f}"
            )
            prior_log_writer.writerow(
                [
                    step,
                    snap_diag["snapshot_step"],
                    snap_diag["mu_g_norm"],
                    snap_diag["sigma_g_mean"],
                    snap_diag["sigma_g_min"],
                    snap_diag["sigma_g_max"],
                    snap_diag["trace_sigma_g"],
                    snap_diag["n_samples"],
                    snap_diag["raw_var_mean"],
                    snap_diag["raw_var_max"],
                    snap_diag["raw_hi_frac"],
                ]
            )
            prior_log_file.flush()

        if step % EVAL_EVERY == 0:
            pre_advance_lesson = curriculum.lesson  # capture before maybe_advance can bump it
            id_stage_memory = rnn.memories[0]
            if hasattr(id_stage_memory, "begin_stage_diagnostics"):
                id_stage_memory.begin_stage_diagnostics()
            id_eval_rng_state = _capture_rng_state()
            segment_diagnostics.set_channel("eval-id")
            _, id_triple_acc, id_perfect_frac = curriculum.maybe_advance(
                rnn, device, step=step, optimizer=optimizer
            )
            if surprise_recall_logger is not None:
                surprise_recall_logger.write_evaluation(
                    step, pre_advance_lesson + 1, "id", dataset.last_eval_diagnostics
                )
            if sgr_prediction_decoder is not None:
                sgr_phase1_eval = evaluate_prediction_signals(
                    rnn.backbone,
                    sgr_prediction_decoder,
                    device,
                    episodes_per_lesson=sgr_prediction_eval_episodes,
                    lesson_indices=(pre_advance_lesson,),
                    seed=seed + 730_201,
                )
                if surprise_recall_logger is not None:
                    surprise_recall_logger.write_phase1_evaluation(
                        step, "periodic-active-lesson", sgr_phase1_eval
                    )
            segment_diagnostics.emit("eval-id")
            id_stage_diagnostics = (
                id_stage_memory.finish_stage_diagnostics()
                if hasattr(id_stage_memory, "finish_stage_diagnostics")
                else {}
            )
            if (
                has_dnc
                and relational_probe_writer is not None
                and relational_probe_file is not None
                and relational_probe_cases is not None
            ):
                probe_metrics = score_probe_cases(rnn.memories[0], relational_probe_cases)
                relational_probe_writer.writerow(
                    [
                        step,
                        curriculum.lesson + 1,
                        cosine_top1_accuracy(relational_probe_cases),
                        probe_metrics["accuracy"],
                        probe_metrics["held_out_accuracy"],
                        int(probe_metrics["cases"]),
                    ]
                )
                relational_probe_file.flush()

            if curriculum.lesson != pre_advance_lesson:
                lesson_dip_start_step = step
                if kl_on:
                    print(
                        f"[{run_id}] Step {step} lesson advance {pre_advance_lesson + 1}->"
                        f"{curriculum.lesson + 1}: KL briefly dipping and ramping back up "
                        f"over the next {LESSON_KL_DIP_STEPS} steps (global anneal unaffected)."
                    )
            if beta_mode == "dynamic":
                # Hard safety ceiling -- enforced every eval cycle unconditionally,
                # independent of the health gate below. Without this, starting
                # beta_target above [BETA_CTRL_MIN, BETA_CTRL_MAX] (e.g. testing
                # 0.001) could ride out the whole KL_ANNEAL_STEPS window unclamped
                # if early-training grad instability keeps failing the health
                # check on every eval cycle -- i.e. behave identically to static
                # mode at the known-collapse value, silently. This is a clamp,
                # not a rate-limited step: it can move beta_target more than
                # BETA_CTRL_LR in one cycle, on purpose.
                if not (BETA_CTRL_MIN <= beta_target <= BETA_CTRL_MAX):
                    clamped = float(np.clip(beta_target, BETA_CTRL_MIN, BETA_CTRL_MAX))
                    print(
                        f"[{run_id}] Step {step} beta_target {beta_target:.6f} outside "
                        f"[{BETA_CTRL_MIN}, {BETA_CTRL_MAX}] -- hard-clamping to {clamped:.6f} "
                        f"(unconditional, ignores health gate below)"
                    )
                    beta_target = clamped

                scale_backed_off = scaler.get_scale() < scale_at_last_eval
                healthy = grad_was_finite_since_eval and not scale_backed_off
                if healthy:
                    acc_frac = id_triple_acc / 100.0
                    raw_constraint = acc_frac - beta_ctrl_acc_target
                    beta_constraint_ema = (
                        BETA_CTRL_EMA_DECAY * beta_constraint_ema
                        + (1 - BETA_CTRL_EMA_DECAY) * raw_constraint
                    )
                    beta_target = float(
                        np.clip(
                            beta_target + BETA_CTRL_LR * beta_constraint_ema,
                            BETA_CTRL_MIN,
                            BETA_CTRL_MAX,
                        )
                    )
                    print(
                        f"[{run_id}] Step {step} dynamic-beta update | "
                        f"acc {acc_frac:.4f} target {beta_ctrl_acc_target:.4f} | "
                        f"constraint_ema {beta_constraint_ema:.4f} | beta -> {beta_target:.6f}"
                    )
                else:
                    print(
                        f"[{run_id}] Step {step} dynamic-beta update SKIPPED "
                        f"(grad_finite={grad_was_finite_since_eval}, "
                        f"scale_backoff={scale_backed_off}) | beta stays {beta_target:.6f}"
                    )
                scale_at_last_eval = scaler.get_scale()
                grad_was_finite_since_eval = True

            ood_field_log: dict[tuple[int, int], list[int]] = {}
            ood_rng_state = ood_rng.getstate()
            ood_torch_rng = _capture_torch_rng()
            ood_stage_memory = rnn.memories[0]
            if hasattr(ood_stage_memory, "begin_stage_diagnostics"):
                ood_stage_memory.begin_stage_diagnostics()
            segment_diagnostics.set_channel("eval-ood")
            ood_triple_acc, ood_perfect_frac, ood_hop_breakdown = dataset.evaluate_ood(
                rnn,
                device,
                num_episodes=periodic_ood_episodes,
                rng=ood_rng,
                field_log=ood_field_log,
            )
            if surprise_recall_logger is not None:
                surprise_recall_logger.write_evaluation(
                    step, curriculum.lesson + 1, "ood", dataset.last_eval_diagnostics
                )
            segment_diagnostics.emit("eval-ood")
            ood_stage_diagnostics = (
                ood_stage_memory.finish_stage_diagnostics()
                if hasattr(ood_stage_memory, "finish_stage_diagnostics")
                else {}
            )
            ood_ablated_acc = None
            if has_dnc and hasattr(dataset, "evaluate_ood_ablated"):
                ood_ablated_rng = random.Random()
                ood_ablated_rng.setstate(ood_rng_state)
                segment_diagnostics.set_channel("eval-ood-memory-off")
                with _replay_torch_rng(ood_torch_rng):
                    ood_ablated_acc, _ood_ablated_perfect = dataset.evaluate_ood_ablated(
                        rnn, device, periodic_ood_episodes, ood_ablated_rng
                    )
                segment_diagnostics.emit("eval-ood-memory-off")

            ood_log_writer.writerow(
                [
                    step,
                    curriculum.lesson + 1,
                    id_triple_acc,
                    id_perfect_frac,
                    ood_triple_acc,
                    ood_perfect_frac,
                    id_triple_acc - ood_triple_acc,
                    id_perfect_frac - ood_perfect_frac,
                ]
            )
            ood_log_file.flush()
            if (
                has_dnc
                and track_a_writer is not None
                and track_a_file is not None
                and relational_probe_cases is not None
            ):
                rho_key = "ordinary_read_score_cosine_correlation_mean"
                rho = id_stage_diagnostics.get(
                    rho_key,
                    1.0 if dnc_read_variant == "cosine" else None,
                )
                rho_count = id_stage_diagnostics.get("diagnostic_steps", 0.0)
                stress_metrics = score_probe_cases(rnn.memories[0], relational_probe_cases)
                track_a_writer.writerow(
                    [
                        step,
                        pre_advance_lesson + 1,
                        dnc_read_variant,
                        id_triple_acc,
                        ood_triple_acc,
                        id_triple_acc - ood_triple_acc,
                        cosine_top1_accuracy(relational_probe_cases),
                        stress_metrics["accuracy"],
                        stress_metrics["held_out_accuracy"],
                        rho,
                        rho_count,
                        id_stage_diagnostics.get("read_content_entropy_mean"),
                        id_stage_diagnostics.get("read_content_max_weight_mean"),
                        id_stage_diagnostics.get("relational_read_residual_rms_mean"),
                        id_stage_diagnostics.get("relational_read_residual_to_cosine_rms_mean"),
                    ]
                )
                track_a_file.flush()
                print(
                    f"[{run_id}] [Track A] read={dnc_read_variant} | "
                    f"ID/OOD/offset={id_triple_acc:.2f}/{ood_triple_acc:.2f}/"
                    f"{id_triple_acc - ood_triple_acc:.2f} | "
                    f"stress cosine/read/held-out="
                    f"{cosine_top1_accuracy(relational_probe_cases):.3f}/"
                    f"{stress_metrics['accuracy']:.3f}/"
                    f"{stress_metrics['held_out_accuracy']:.3f} | "
                    f"ordinary MLP-cosine rho="
                    f"{rho if rho is not None else float('nan'):.4f} | "
                    f"read entropy/max="
                    f"{id_stage_diagnostics.get('read_content_entropy_mean', float('nan')):.3f}/"
                    f"{id_stage_diagnostics.get('read_content_max_weight_mean', float('nan')):.3f} | "
                    f"residual/cosine rms="
                    f"{id_stage_diagnostics.get('relational_read_residual_to_cosine_rms_mean', float('nan')):.3f}"
                )
            if track_b_writer is not None and track_b_file is not None:
                id_write_values = [
                    id_stage_diagnostics.get("posterior_variance_mean"),
                    id_stage_diagnostics.get("posterior_variance_min"),
                    id_stage_diagnostics.get("posterior_variance_max"),
                    id_stage_diagnostics.get("posterior_variance_reduction_mean"),
                    id_stage_diagnostics.get("posterior_mean_update_abs_mean"),
                    id_stage_diagnostics.get("active_write_cell_fraction_mean"),
                    id_stage_diagnostics.get("write_weight_mass_mean"),
                    id_stage_diagnostics.get("diagnostic_steps"),
                ]
                ood_write_values = [
                    ood_stage_diagnostics.get("posterior_variance_mean"),
                    ood_stage_diagnostics.get("posterior_variance_min"),
                    ood_stage_diagnostics.get("posterior_variance_max"),
                    ood_stage_diagnostics.get("posterior_variance_reduction_mean"),
                    ood_stage_diagnostics.get("posterior_mean_update_abs_mean"),
                    ood_stage_diagnostics.get("active_write_cell_fraction_mean"),
                    ood_stage_diagnostics.get("write_weight_mass_mean"),
                    ood_stage_diagnostics.get("diagnostic_steps"),
                ]
                track_b_writer.writerow(
                    [
                        "evaluation",
                        step,
                        pre_advance_lesson + 1,
                        dnc_write_variant,
                        id_triple_acc,
                        ood_triple_acc,
                        id_triple_acc - ood_triple_acc,
                        *([None] * 12),
                        *id_write_values,
                        *ood_write_values,
                        *([None] * 6),
                    ]
                )
                track_b_file.flush()
                if dnc_write_variant == "kanerva-closed-form":
                    print(
                        f"[{run_id}] [Track B] write={dnc_write_variant} | "
                        f"posterior variance ID/OOD="
                        f"{id_stage_diagnostics.get('posterior_variance_mean', float('nan')):.5f}/"
                        f"{ood_stage_diagnostics.get('posterior_variance_mean', float('nan')):.5f} | "
                        f"mean update ID/OOD="
                        f"{id_stage_diagnostics.get('posterior_mean_update_abs_mean', float('nan')):.6g}/"
                        f"{ood_stage_diagnostics.get('posterior_mean_update_abs_mean', float('nan')):.6g} | "
                        f"active cells ID/OOD="
                        f"{id_stage_diagnostics.get('active_write_cell_fraction_mean', float('nan')):.4f}/"
                        f"{ood_stage_diagnostics.get('active_write_cell_fraction_mean', float('nan')):.4f}"
                    )
            _write_ogs_row(
                ogs_log_writer,
                step,
                pre_advance_lesson + 1,
                id_triple_acc,
                ood_triple_acc,
                ood_perfect_frac,
                ood_ablated_acc,
                ogs_weights,
            )
            ogs_log_file.flush()
            _print_ood_analysis(
                run_id,
                "periodic",
                step,
                pre_advance_lesson + 1,
                id_triple_acc,
                ood_triple_acc,
                ood_perfect_frac,
                ood_ablated_acc,
                ogs_weights,
                beta_eff,
                kl_loss,
                kl_diag,
            )
            dataset.write_field_log(
                field_log_writer, field_log_file, step, curriculum.lesson + 1, "ood", ood_field_log
            )
            print(
                f"[{run_id}] Step {step} periodic OOD check: "
                f"ID {id_triple_acc:.2f}% | OOD {ood_triple_acc:.2f}% | "
                f"offset {id_triple_acc - ood_triple_acc:.2f}"
            )
            for hops, (acc, pf, n) in sorted(ood_hop_breakdown.items()):
                hop_log_writer.writerow([step, curriculum.lesson + 1, "ood", hops, acc, pf, n])
            hop_log_file.flush()

            # Functional-usage check: same lesson distribution the ID eval
            # above just used (pre_advance_lesson, not curriculum.lesson --
            # this call may have just advanced it).
            # Replay the exact ID evaluation episodes with memory bypassed.
            # Comparing two independently sampled episode sets made this
            # purported causal dependency metric too noisy to diagnose
            # whether the model actually used memory. Restore the post-ID
            # stream afterward so the paired evaluation consumes no extra
            # randomness from the run.
            if not has_dnc:
                print(
                    f"[{run_id}] Step {step} DNC memory-dependency check: n/a "
                    "(DNC is disabled for this split-graph run)."
                )
            else:
                rng_state_after_other_evaluations = _capture_rng_state()
                _restore_rng_state(id_eval_rng_state)
                segment_diagnostics.set_channel("eval-id-memory-off")
                try:
                    ablated_triple_acc, ablated_perfect_frac = dataset.evaluate_id_ablated(
                        rnn, device, curriculum, pre_advance_lesson
                    )
                    if surprise_recall_logger is not None:
                        surprise_recall_logger.write_evaluation(
                            step,
                            pre_advance_lesson + 1,
                            "id-memory-off",
                            dataset.last_eval_diagnostics,
                        )
                finally:
                    _restore_rng_state(rng_state_after_other_evaluations)
                segment_diagnostics.emit("eval-id-memory-off")

                assert mem_check_writer is not None and mem_check_file is not None
                mem_check_writer.writerow(
                    [
                        step,
                        pre_advance_lesson + 1,
                        id_triple_acc,
                        id_perfect_frac,
                        ablated_triple_acc,
                        ablated_perfect_frac,
                        id_triple_acc - ablated_triple_acc,
                        id_perfect_frac - ablated_perfect_frac,
                    ]
                )
                mem_check_file.flush()
                print(
                    f"[{run_id}] Step {step} memory-dependency check: "
                    f"ID (memory on) {id_triple_acc:.2f}% | ablated (memory off) "
                    f"{ablated_triple_acc:.2f}% | dependency "
                    f"{id_triple_acc - ablated_triple_acc:.2f}"
                )

            if combiner_stage_kinds is not None:
                # Same pattern as the memory-dependency check above, but
                # ablating one combiner stage (e.g. the CfC block in a
                # "mamba+cfc" combiner) at a time, so each stage's actual
                # contribution to accuracy is measured directly instead of
                # inferred from loss curves / plateau shape.
                for stage_idx, stage_kind in enumerate(combiner_stage_kinds):
                    segment_diagnostics.set_channel(f"eval-combiner-{stage_idx}")
                    stage_triple_acc, stage_perfect_frac = (
                        dataset.evaluate_id_combiner_stage_ablated(
                            rnn, device, curriculum, pre_advance_lesson, skip_stages={stage_idx}
                        )
                    )
                    segment_diagnostics.emit(f"eval-combiner-{stage_idx}")
                    combiner_stage_log_writer.writerow(
                        [
                            step,
                            pre_advance_lesson + 1,
                            stage_idx,
                            stage_kind,
                            id_triple_acc,
                            id_perfect_frac,
                            stage_triple_acc,
                            stage_perfect_frac,
                            id_triple_acc - stage_triple_acc,
                            id_perfect_frac - stage_perfect_frac,
                        ]
                    )
                    combiner_stage_log_file.flush()
                    print(
                        f"[{run_id}] Step {step} combiner-stage-dependency check "
                        f"[{stage_idx}:{stage_kind}]: ID (full) {id_triple_acc:.2f}% | "
                        f"ablated ({stage_kind} off) {stage_triple_acc:.2f}% | "
                        f"dependency {id_triple_acc - stage_triple_acc:.2f}"
                    )

            if getattr(dataset, "modalities", None):
                for modality in dataset.modalities:
                    m_acc, m_pf = dataset.evaluate_modality_ablated(
                        rnn, device, curriculum, pre_advance_lesson, modality
                    )
                    modality_log_writer.writerow(
                        [
                            step,
                            pre_advance_lesson + 1,
                            modality,
                            id_triple_acc,
                            id_perfect_frac,
                            m_acc,
                            m_pf,
                            id_triple_acc - m_acc,
                            id_perfect_frac - m_pf,
                        ]
                    )
                    modality_log_file.flush()
                    print(
                        f"[{run_id}] Step {step} modality-dependency check "
                        f"[{modality}]: full {id_triple_acc:.2f}% | ablated ({modality} off) "
                        f"{m_acc:.2f}% | dependency {id_triple_acc - m_acc:.2f}"
                    )

        if step % checkpoint_every == 0 or step == total_steps:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"{run_id}_step{step}.pt")
            save_checkpoint(
                ckpt_path,
                rnn,
                output_proj,
                stochastic_heads,
                optimizer,
                curriculum,
                step,
                beta_target,
                run_id,
                scaler,
                ood_rng,
                controller_type=controller,
                link_matrix_mode=link_matrix_mode,
                link_matrix_topk=link_matrix_topk,
                dynamic_n_mode=dynamic_n_mode,
                dynamic_n_state=(
                    dynamic_n_ctrl.state_dict() if dynamic_n_ctrl is not None else None
                ),
                moe_enabled=moe_enabled,
                moe_num_experts=moe_num_experts,
                moe_expert_dim=moe_expert_dim,
                moe_capacity_factor=moe_capacity_factor,
                moe_load_balance_alpha=moe_load_balance_alpha,
                moe_top_k=moe_top_k,
                split_graph_enabled=split_graph_enabled,
                split_graph_variant=split_graph_variant,
                extra_model_config=extra_model_config,
                sgr_prediction_decoder=sgr_prediction_decoder,
            )
            latest_path = os.path.join(CHECKPOINT_DIR, f"{run_id}_latest.pt")
            shutil.copyfile(ckpt_path, latest_path)

    completion_label = "Evaluation-only pass" if eval_only else "Training complete"
    if eval_only:
        # No optimization step ran, so there is no effective training beta or
        # KL measurement to attribute in the shared final-reporting path.
        beta_eff = 0.0
        kl_loss = None
        kl_diag = {}
    print(f"\n[{run_id}] {completion_label}. Final evaluation on training-distribution lesson:")
    final_id_stage_memory = rnn.memories[0]
    if isinstance(rnn, SplitGraphDNC) and rnn.workspace is not None:
        rnn.workspace.reset_bistable_diagnostics()
    if hasattr(final_id_stage_memory, "begin_stage_diagnostics"):
        final_id_stage_memory.begin_stage_diagnostics()
    segment_diagnostics.set_channel("eval-final-id")
    _, id_triple_acc, id_perfect_frac = curriculum.maybe_advance(
        rnn, device, step=step, optimizer=optimizer, allow_advance=not eval_only
    )
    if surprise_recall_logger is not None:
        surprise_recall_logger.write_evaluation(
            step, curriculum.lesson + 1, "final-id", dataset.last_eval_diagnostics
        )
    final_sgr_phase1_eval = None
    if sgr_prediction_decoder is not None:
        final_sgr_phase1_eval = evaluate_prediction_signals(
            rnn.backbone,
            sgr_prediction_decoder,
            device,
            episodes_per_lesson=sgr_prediction_eval_episodes,
            seed=seed + 730_201,
        )
        if surprise_recall_logger is not None:
            surprise_recall_logger.write_phase1_evaluation(
                step, "final-all-lessons", final_sgr_phase1_eval
            )
    segment_diagnostics.emit("eval-final-id")
    report_bistable_evaluation("ID")
    final_id_stage_diagnostics = (
        final_id_stage_memory.finish_stage_diagnostics()
        if hasattr(final_id_stage_memory, "finish_stage_diagnostics")
        else {}
    )
    if (
        has_dnc
        and relational_probe_writer is not None
        and relational_probe_file is not None
        and relational_probe_cases is not None
    ):
        probe_metrics = score_probe_cases(rnn.memories[0], relational_probe_cases)
        relational_probe_writer.writerow(
            [
                step,
                curriculum.lesson + 1,
                cosine_top1_accuracy(relational_probe_cases),
                probe_metrics["accuracy"],
                probe_metrics["held_out_accuracy"],
                int(probe_metrics["cases"]),
            ]
        )
        relational_probe_file.flush()

    print(f"\n[{run_id}] Generalization test:")

    final_ood_rng_state = ood_rng.getstate()
    final_ood_torch_rng = _capture_torch_rng()
    final_ood_field_log: dict[tuple[int, int], list[int]] = {}
    final_ood_stage_memory = rnn.memories[0]
    if isinstance(rnn, SplitGraphDNC) and rnn.workspace is not None:
        rnn.workspace.reset_bistable_diagnostics()
    if hasattr(final_ood_stage_memory, "begin_stage_diagnostics"):
        final_ood_stage_memory.begin_stage_diagnostics()
    segment_diagnostics.set_channel("eval-final-ood")
    ood_triple_acc, ood_perfect_frac, ood_hop_breakdown = dataset.evaluate_ood(
        rnn,
        device,
        num_episodes=final_ood_episodes,
        rng=ood_rng,
        verbose_n=10,
        field_log=final_ood_field_log,
    )
    if surprise_recall_logger is not None:
        surprise_recall_logger.write_evaluation(
            step, curriculum.lesson + 1, "final-ood", dataset.last_eval_diagnostics
        )
    segment_diagnostics.emit("eval-final-ood")
    report_bistable_evaluation("OOD")
    final_ood_stage_diagnostics = (
        final_ood_stage_memory.finish_stage_diagnostics()
        if hasattr(final_ood_stage_memory, "finish_stage_diagnostics")
        else {}
    )

    ood_offset_triple = id_triple_acc - ood_triple_acc
    ood_offset_perfect = id_perfect_frac - ood_perfect_frac

    if (
        has_dnc
        and track_a_writer is not None
        and track_a_file is not None
        and relational_probe_cases is not None
    ):
        final_stress_metrics = score_probe_cases(rnn.memories[0], relational_probe_cases)
        rho = final_id_stage_diagnostics.get(
            "ordinary_read_score_cosine_correlation_mean",
            1.0 if dnc_read_variant == "cosine" else None,
        )
        track_a_writer.writerow(
            [
                step,
                curriculum.lesson + 1,
                dnc_read_variant,
                id_triple_acc,
                ood_triple_acc,
                ood_offset_triple,
                cosine_top1_accuracy(relational_probe_cases),
                final_stress_metrics["accuracy"],
                final_stress_metrics["held_out_accuracy"],
                rho,
                final_id_stage_diagnostics.get("diagnostic_steps", 0.0),
                final_id_stage_diagnostics.get("read_content_entropy_mean"),
                final_id_stage_diagnostics.get("read_content_max_weight_mean"),
                final_id_stage_diagnostics.get("relational_read_residual_rms_mean"),
                final_id_stage_diagnostics.get("relational_read_residual_to_cosine_rms_mean"),
            ]
        )
        track_a_file.flush()
        residual_ratio = final_id_stage_diagnostics.get(
            "relational_read_residual_to_cosine_rms_mean"
        )
        residual_ratio_text = (
            f"{residual_ratio:.3f}"
            if residual_ratio is not None
            else f"n/a (not defined for {dnc_read_variant} read)"
        )
        print(
            f"[{run_id}] [Track A / final] stress cosine/read/held-out="
            f"{cosine_top1_accuracy(relational_probe_cases):.3f}/"
            f"{final_stress_metrics['accuracy']:.3f}/"
            f"{final_stress_metrics['held_out_accuracy']:.3f} | "
            f"ordinary MLP-cosine rho="
            f"{rho if rho is not None else float('nan'):.4f} | "
            f"read entropy/max="
            f"{final_id_stage_diagnostics.get('read_content_entropy_mean', float('nan')):.3f}/"
            f"{final_id_stage_diagnostics.get('read_content_max_weight_mean', float('nan')):.3f} | "
            f"residual/cosine rms={residual_ratio_text}"
        )
    if track_b_writer is not None and track_b_file is not None:

        def _write_diagnostic_values(diagnostics):
            return [
                diagnostics.get("posterior_variance_mean"),
                diagnostics.get("posterior_variance_min"),
                diagnostics.get("posterior_variance_max"),
                diagnostics.get("posterior_variance_reduction_mean"),
                diagnostics.get("posterior_mean_update_abs_mean"),
                diagnostics.get("active_write_cell_fraction_mean"),
                diagnostics.get("write_weight_mass_mean"),
                diagnostics.get("diagnostic_steps"),
            ]

        track_b_writer.writerow(
            [
                "final_evaluation",
                step,
                curriculum.lesson + 1,
                dnc_write_variant,
                id_triple_acc,
                ood_triple_acc,
                ood_offset_triple,
                *([None] * 12),
                *_write_diagnostic_values(final_id_stage_diagnostics),
                *_write_diagnostic_values(final_ood_stage_diagnostics),
                *([None] * 6),
            ]
        )
        track_b_file.flush()
        if dnc_write_variant == "kanerva-closed-form":
            print(
                f"[{run_id}] [Track B / final] posterior variance ID/OOD="
                f"{final_id_stage_diagnostics.get('posterior_variance_mean', float('nan')):.5f}/"
                f"{final_ood_stage_diagnostics.get('posterior_variance_mean', float('nan')):.5f} | "
                f"mean update ID/OOD="
                f"{final_id_stage_diagnostics.get('posterior_mean_update_abs_mean', float('nan')):.6g}/"
                f"{final_ood_stage_diagnostics.get('posterior_mean_update_abs_mean', float('nan')):.6g}"
            )

    ood_log_writer.writerow(
        [
            step,
            curriculum.lesson + 1,
            id_triple_acc,
            id_perfect_frac,
            ood_triple_acc,
            ood_perfect_frac,
            ood_offset_triple,
            ood_offset_perfect,
        ]
    )
    ood_log_file.flush()
    final_ood_ablated_acc = None
    if has_dnc and hasattr(dataset, "evaluate_ood_ablated"):
        final_ood_ablated_rng = random.Random()
        final_ood_ablated_rng.setstate(final_ood_rng_state)
        segment_diagnostics.set_channel("eval-final-ood-memory-off")
        with _replay_torch_rng(final_ood_torch_rng):
            final_ood_ablated_acc, _final_ood_ablated_perfect = dataset.evaluate_ood_ablated(
                rnn, device, final_ood_episodes, final_ood_ablated_rng
            )
        segment_diagnostics.emit("eval-final-ood-memory-off")

    q99_metrics = None
    if split_graph_enabled and dnc_write_variant == "kanerva-closed-form":
        q99_inputs, q99_targets, q99_answer_mask = build_q99_probe_batch()
        q99_metrics = evaluate_q99_order_sensitivity(
            rnn,
            output_proj,
            q99_inputs,
            q99_targets,
            q99_answer_mask,
        )
        print(
            f"[{run_id}] [Track B / Q99] post-training gate={q99_metrics['gate']} | "
            f"episodes={q99_metrics['tested_episodes']} | "
            f"ordered/fact-permuted accuracy="
            f"{q99_metrics['ordered_query_accuracy']:.2f}/"
            f"{q99_metrics['fact_permuted_accuracy']:.2f}% "
            f"(delta {q99_metrics['fact_accuracy_delta']:+.2f} pp) | "
            f"ordered/swapped query accuracy="
            f"{q99_metrics['ordered_query_accuracy']:.2f}/"
            f"{q99_metrics['swapped_query_accuracy']:.2f}% "
            f"(drop {q99_metrics['query_order_accuracy_drop']:+.2f} pp) | "
            f"query memory effect="
            f"{q99_metrics['query_memory_incremental_order_effect']:.6g}"
        )
        if track_b_writer is not None and track_b_file is not None:
            track_b_writer.writerow(
                [
                    "q99_semantic_order_check",
                    step,
                    curriculum.lesson + 1,
                    dnc_write_variant,
                    id_triple_acc,
                    ood_triple_acc,
                    ood_offset_triple,
                    q99_metrics["gate"],
                    q99_metrics["tested_episodes"],
                    q99_metrics["tested_fact_episodes"],
                    q99_metrics["tested_query_episodes"],
                    q99_metrics["ordered_query_accuracy"],
                    q99_metrics["fact_permuted_accuracy"],
                    q99_metrics["fact_accuracy_delta"],
                    q99_metrics["fact_prediction_retention"],
                    q99_metrics["swapped_query_accuracy"],
                    q99_metrics["query_order_accuracy_drop"],
                    q99_metrics["query_prediction_flip_rate"],
                    q99_metrics["query_memory_incremental_order_effect"],
                    *([None] * 16),
                    *([None] * 6),
                ]
            )
            track_b_file.flush()
    _write_ogs_row(
        ogs_log_writer,
        step,
        curriculum.lesson + 1,
        id_triple_acc,
        ood_triple_acc,
        ood_perfect_frac,
        final_ood_ablated_acc,
        ogs_weights,
    )
    ogs_log_file.flush()
    _print_ood_analysis(
        run_id,
        "final",
        step,
        curriculum.lesson + 1,
        id_triple_acc,
        ood_triple_acc,
        ood_perfect_frac,
        final_ood_ablated_acc,
        ogs_weights,
        beta_eff,
        kl_loss,
        kl_diag,
    )

    for hops, (acc, pf, n) in sorted(ood_hop_breakdown.items()):
        hop_log_writer.writerow([step, curriculum.lesson + 1, "ood", hops, acc, pf, n])
    hop_log_file.flush()

    summary = {
        "run_id": run_id,
        "beta_target": beta_target,
        "id_triple_acc": id_triple_acc,
        "id_perfect_frac": id_perfect_frac,
        "ood_triple_acc": ood_triple_acc,
        "ood_perfect_frac": ood_perfect_frac,
        "ood_offset_triple": ood_offset_triple,
        "ood_offset_perfect": ood_offset_perfect,
        "total_elapsed_sec": time.time() - t_run_start,
    }
    if final_sgr_phase1_eval is not None:
        summary["sgr_prediction_loss_weight"] = float(sgr_prediction_loss_weight)
        for kind, values in final_sgr_phase1_eval["per_type"].items():
            summary[f"phase1_{kind.lower()}_raw_auroc"] = values["raw_auroc_vs_clean_normal"]
            summary[f"phase1_{kind.lower()}_excess_auroc"] = values["excess_auroc_vs_clean_normal"]
        summary["phase1_a6_vs_anomalies_raw_auroc"] = final_sgr_phase1_eval["a6_vs_true_anomalies"][
            "raw_auroc"
        ]
        summary["phase1_a6_vs_anomalies_excess_auroc"] = final_sgr_phase1_eval[
            "a6_vs_true_anomalies"
        ]["excess_auroc"]
        summary["phase1_next_token_ece"] = final_sgr_phase1_eval["calibration"]["ece"]
        summary["phase1_a5_mean_tokens_to_recovery"] = final_sgr_phase1_eval["a5_recovery"][
            "mean_tokens_to_recovery"
        ]
    if q99_metrics is not None:
        summary["q99_gate"] = q99_metrics["gate"]
        summary["q99_fact_permuted_accuracy"] = q99_metrics["fact_permuted_accuracy"]
        summary["q99_query_order_accuracy_drop"] = q99_metrics["query_order_accuracy_drop"]
    print(f"\n[{run_id}] SUMMARY: {summary}")

    log_writer.writerow([])
    log_writer.writerow(["SUMMARY"] + list(summary.keys()))
    log_writer.writerow([""] + list(summary.values()))
    log_file.close()
    if workspace_telemetry_log_file is not None:
        workspace_telemetry_log_file.close()
    ood_log_file.close()
    if relational_probe_file is not None:
        relational_probe_file.close()
    if track_a_file is not None:
        track_a_file.close()
    if track_b_file is not None:
        track_b_file.close()
    ogs_log_file.close()
    if mem_check_file is not None:
        mem_check_file.close()
    lesson_log_file.close()
    prior_log_file.close()
    field_log_file.close()  # v16
    combiner_stage_log_file.close()
    modality_log_file.close()
    if dynamic_n_ctrl is not None:
        dynamic_n_log_file.close()

    segment_diagnostics.close()
    if surprise_recall_logger is not None:
        surprise_recall_logger.close()
    if split_graph_forward_trace is not None:
        split_graph_forward_trace.close()
    return summary


# ==========================================
# 11. GATE
# ==========================================
if __name__ == "__main__":
    import argparse
    import sys

    class _OptimizedDNCAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            namespace.dnc_implementation = "optimized"
            namespace.dnc_compile_memory = True
            namespace._dnc_optimized_bundle = True

    parser = argparse.ArgumentParser(
        description="Phase 2 KL sweep (learned, periodically-snapshotted "
        "prior). Positional beta for single-beta mode; "
        "no beta runs the full BETAS_TO_SWEEP list sequentially."
    )
    parser.add_argument(
        "beta",
        nargs="?",
        type=float,
        default=None,
        help=f"beta to run, one of {BETAS_TO_SWEEP}. Omit to run the full sweep sequentially.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="path to a checkpoint .pt (requires --beta / positional beta). Normally "
        "restores model, optimizer, RNG, curriculum, and step. With --workspace "
        "--workspace-train-mode=frozen, a compatible workspace-disabled baseline "
        "instead warm-starts matching weights and begins a fresh workspace training run.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load --resume checkpoint and run only the final ID/OOD evaluation; no training updates or optimizer-state restore.",
    )
    parser.add_argument(
        "--eval-workspace-mode",
        choices=("checkpoint", "reset-each-step", "bypass"),
        default="checkpoint",
        help="Evaluation-only workspace counterfactual: preserve checkpoint behavior, clear slots before every timestep, or bypass workspace readback while retaining the selected MoE residual.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="Evaluation-only RNG seed. Use the same value across checkpoints to evaluate identical generated ID/OOD episodes; omit to replay checkpoint RNG state.",
    )
    parser.add_argument(
        "--run-id-suffix",
        type=str,
        default=None,
        help="appended to the derived run_id (e.g. 'switch') so this run's checkpoints "
        "(checkpoints/beta_XXX<suffix>_stepN.pt) and log "
        "(logs/run_beta_XXX<suffix>.csv) don't collide with an existing run "
        "that used the same beta value -- e.g. resuming a beta=0 checkpoint into a "
        "beta=0.01 run would otherwise reuse the same run_id/files as a prior plain "
        "beta=0.01 sweep run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="random seed for torch/random/numpy (default: SEED module constant)",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=CONTROLLER_TYPE,
        choices=[
            "lstm",
            "mamba",
            "mamba2",
            "mamba3",
            "cfc",
            "mamba+cfc",
            "mamba2+cfc",
            "mamba3+cfc",
        ],
        help="DNC controller type. 'lstm' (default) "
        "reproduces Phase 2 exactly. 'mamba' swaps in the "
        "Mamba-1 controller from mamba_controller.py. "
        "'mamba2' swaps in the standalone, "
        "per-timestep Mamba-2 controller from "
        "mamba_controller/mamba2_controller.py - distinct "
        "from --split-graph-variant=mamba2, which is a "
        "parallel whole-sequence backbone, not an "
        "interleaved controller. Both require mamba-ssm.",
    )
    parser.add_argument(
        "--beta-mode",
        type=str,
        default=BETA_MODE,
        choices=["static", "dynamic"],
        help="'static' (default): beta_target is fixed for the run, "
        "current sweep behavior, unchanged. 'dynamic': beta_target "
        "starts at the given positional beta and is then adjusted "
        "every EVAL_EVERY steps by a task-accuracy-constrained "
        "controller, bounded to [BETA_CTRL_MIN, BETA_CTRL_MAX].",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Override the training loop's stopping point for a "
        "cheap pilot run (e.g. the seed-0/seed-2, short-budget staging pass "
        "suggested before committing to the full 4-seed x 3-config x "
        f"{TOTAL_STEPS}-step sweep). Defaults to the module constant "
        f"TOTAL_STEPS ({TOTAL_STEPS}) if omitted. Does NOT change "
        "LR_DECAY_STEPS -- a pilot run still anneals on the full schedule.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Override how often (in steps) a periodic safety checkpoint "
        "is saved, in addition to the end-of-run checkpoint. Defaults "
        f"to the module constant CHECKPOINT_EVERY ({CHECKPOINT_EVERY}) "
        "if omitted.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=LOG_EVERY,
        metavar="N",
        help=f"print the training summary and emit grouped training diagnostics every N steps (default: {LOG_EVERY}).",
    )
    parser.add_argument(
        "--link-matrix-mode",
        type=str,
        default=LINK_MATRIX_MODE,
        choices=["dense", "ablated", "sparse_topk"],
        help="Static Option 1: link-matrix ablation/sparsification at "
        "fixed N. 'dense' (default): unchanged Memory behavior. "
        "'ablated': link matrix never updated (temporal addressing "
        "contributes nothing). 'sparse_topk': link matrix updated "
        "as normal, then each row keeps only its --link-matrix-topk "
        "largest-magnitude entries.",
    )
    parser.add_argument(
        "--link-matrix-topk",
        type=int,
        default=LINK_MATRIX_TOPK,
        help="Required when --link-matrix-mode=sparse_topk: how many "
        "entries each link-matrix row keeps per step.",
    )
    parser.add_argument(
        "--isolate-link-ablation",
        action="store_true",
        help="Hold nr_cells fixed at MODEL_NR_CELLS for the whole run "
        "(overrides LESSON_NR_CELLS to a constant list), so Option "
        "2's curriculum-indexed resize never fires. Use this for "
        "the isolated Option-1 run the roadmap's ordering requires "
        "-- do not combine with Option 2 in the same run.",
    )
    parser.add_argument(
        "--dynamic-n-mode",
        action="store_true",
        help="Dynamic-N, macro-scale: usage-triggered nr_cells growth "
        "between episodes, instead of LESSON_NR_CELLS's "
        "curriculum-indexed lookup (which this disables, same "
        "isolation as --isolate-link-ablation). Model starts at "
        "--dynamic-n-floor and grows toward --dynamic-n-ceiling "
        "whenever memory-usage saturation stays above "
        "--dynamic-n-trigger-frac past the post-growth cooldown.",
    )
    parser.add_argument(
        "--dynamic-n-floor",
        type=int,
        default=DYNAMIC_N_FLOOR,
        help="Starting (and minimum) nr_cells under --dynamic-n-mode.",
    )
    parser.add_argument(
        "--dynamic-n-ceiling",
        type=int,
        default=DYNAMIC_N_CEILING,
        help="Hard ceiling nr_cells will never grow past under "
        "--dynamic-n-mode (Strategy doc: sparse-link-matrix "
        "approximation only validated to N=512).",
    )
    parser.add_argument(
        "--dynamic-n-trigger-frac",
        type=float,
        default=DYNAMIC_N_TRIGGER_FRAC,
        help="Growth fires when the EMA fraction of memory cells with "
        "usage above DYNAMIC_N_USAGE_HIGH exceeds this, past cooldown.",
    )
    parser.add_argument(
        "--dynamic-n-cooldown-steps",
        type=int,
        default=DYNAMIC_N_COOLDOWN_STEPS,
        help="Steps to wait after a growth event before another can fire, "
        "so the model's addressing policy gets time to re-settle.",
    )
    parser.add_argument(
        "--moe",
        action="store_true",
        help="Alternative Phase 3, Step 2, Option 4: interleave external "
        "SwitchMoE blocks between Mamba controller blocks (mamba_controller.py "
        "/ moe_layer.py).",
    )
    parser.add_argument(
        "--moe-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect and print detailed MoE routing/activation diagnostics "
        "(enabled by default; use --no-moe-diagnostics to reduce diagnostic "
        "overhead, including workspace activity summaries). Expert routing, "
        "workspace computation, and the load-balancing auxiliary loss stay active.",
    )
    parser.add_argument(
        "--segment-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect activation/gradient summaries for model segments (enabled by default; "
        "use --no-segment-diagnostics to avoid per-module hooks in long recurrent sequences).",
    )
    parser.add_argument(
        "--moe-placement",
        choices=("all", "backbone", "combiner"),
        default="all",
        help="For split-graph MoE, enable experts in both components (all), "
        "only the parallel backbone, or only the sequential combiner. Default: all.",
    )
    parser.add_argument(
        "--moe-backbone-block-isolation",
        type=int,
        default=None,
        metavar="INDEX",
        help="For split-graph backbone MoE, install a bank only after the selected "
        "zero-based backbone block; other blocks run without MoE. Use one run per "
        "block index in every backbone stage to isolate routing (requires --moe and "
        "--moe-placement backbone).",
    )
    parser.add_argument(
        "--moe-backbone-residual-scale",
        type=float,
        default=1.0,
        metavar="SCALE",
        help="Multiply each active split-graph backbone MoE delta by SCALE "
        "before adding it to the backbone residual (default: 1.0, unchanged). "
        "With --workspace, this also scales the final-bank broadcast contribution. "
        "Use with --moe-placement backbone or all; with all, combiner MoE "
        "residuals remain unscaled. In an isolation run, only the selected "
        "block's MoE residual is scaled.",
    )
    parser.add_argument(
        "--workspace",
        action="store_true",
        help="Phase 0 shared workspace: competitively write selected MoE expert outputs to persistent slots, then broadcast slots back to every specialist. Requires --split-graph and --moe.",
    )
    parser.add_argument(
        "--workspace-num-slots",
        type=int,
        default=WORKSPACE_NUM_SLOTS,
        help="Persistent workspace slot count (0 is the plumbing/bypass control).",
    )
    parser.add_argument(
        "--workspace-no-selection",
        action="store_true",
        help="Control: use random, unweighted specialist writes instead of MoE-selected writes.",
    )
    parser.add_argument(
        "--workspace-reset-each-step",
        action="store_true",
        help="Persistence ablation: reset workspace before each timestep's write/read cycle.",
    )
    parser.add_argument(
        "--workspace-train-mode",
        choices=["coadapted", "frozen"],
        default=WORKSPACE_TRAIN_MODE,
        help="coadapted trains the full model; frozen trains workspace parameters and freezes substrate/task-head parameters.",
    )
    parser.add_argument(
        "--workspace-train-scope",
        choices=("workspace", "gate"),
        default="workspace",
        help="With frozen mode, train all workspace parameters or only the active write gate. Gate-only supports matched continuous/bistable Phase 4 controls.",
    )
    parser.add_argument(
        "--workspace-gate-variant",
        choices=("continuous", "bistable"),
        default="continuous",
        help="Use the existing continuous update gate or the experimental recurrent stochastic bistable write gate.",
    )
    parser.add_argument(
        "--workspace-bistable-selection-scale",
        type=float,
        default=1.0,
        help="Scale the content-selection drive in the bistable gate; vary this for the preregistered selection-strength ablation.",
    )
    parser.add_argument(
        "--workspace-bistable-noise-scale",
        type=float,
        default=0.75,
        help="Independent background-noise scale for bistable ignition; set to 0 for the noise ablation.",
    )
    parser.add_argument(
        "--workspace-bistable-gradient-mode",
        choices=("truncated", "unrolled"),
        default="truncated",
        help=(
            "Backward mode for bistable settling. truncated preserves the full "
            "forward iteration but detaches intermediate settling states; "
            "unrolled backpropagates through every iteration."
        ),
    )
    parser.add_argument(
        "--workspace-bptt-horizon",
        type=int,
        default=0,
        metavar="STEPS",
        help=(
            "Truncate gradients carried by persistent workspace slots every STEPS "
            "sequence positions while preserving the forward trajectory; 0 keeps "
            "full-sequence BPTT."
        ),
    )
    parser.add_argument(
        "--workspace-reconsideration",
        action="store_true",
        help=(
            "Phase 5: reopen writes to previously occupied workspace slots when "
            "the current routed proposal has sufficiently high normalized "
            "content innovation. Uses no labels or future tokens."
        ),
    )
    parser.add_argument(
        "--workspace-reconsideration-surprise-threshold",
        type=float,
        default=0.5,
        metavar="THRESHOLD",
        help=(
            "Phase 5 normalized innovation threshold in [0, 1] for reconsidering "
            "an occupied slot (default: 0.5)."
        ),
    )
    parser.add_argument(
        "--workspace-reconsideration-trigger",
        choices=(
            "innovation",
            "random_rate_matched",
            "prediction_error",
            "prediction_error_random_rate_matched",
        ),
        default="innovation",
        help=(
            "Reconsider using normalized innovation or SGR prediction error; "
            "the random_rate_matched modes provide firing-rate-matched null controls."
        ),
    )
    parser.add_argument(
        "--workspace-bistable-deterministic",
        action="store_true",
        help="Disable independent gate noise for a deterministic diagnostic; normal bistable evaluation remains stochastic and seed-reproducible.",
    )
    parser.add_argument(
        "--bistable-debug",
        action="store_true",
        help=(
            "Print per-update bistable input conditioning, all settling iterations, "
            "raw gate gradients, clipping scale, parameter updates, and Adam moments. "
            "Diagnostic only; expect slower training and high console volume."
        ),
    )
    parser.add_argument(
        "--bistable-anomaly-detection",
        action="store_true",
        help=(
            "Enable PyTorch autograd anomaly detection for a bistable workspace run. "
            "Reports the forward callsite of the first backward operation returning "
            "NaN; diagnostic only and substantially slower."
        ),
    )
    parser.add_argument(
        "--bistable-anomaly-step",
        type=int,
        default=None,
        metavar="STEP",
        help=(
            "Restrict --bistable-anomaly-detection to one absolute training step. "
            "Use the failing step from the log to avoid the cost on earlier steps."
        ),
    )
    parser.add_argument(
        "--workspace-stability-max-activation",
        type=float,
        default=100_000.0,
        help="Phase 4 absolute activation maximum; abort before backward/update if any instrumented segment exceeds this value.",
    )
    parser.add_argument(
        "--moe-num-experts",
        type=int,
        default=MOE_NUM_EXPERTS,
        help="Number of experts per MoE block (>=4 enforced by SwitchMoE; "
        "roadmap recommends 8+, per Dead-End #45).",
    )
    parser.add_argument(
        "--moe-expert-dim",
        type=int,
        default=MOE_EXPERT_DIM,
        help="Hidden dim of each expert FFN. Defaults to 3x the controller's "
        "hidden_size (MoE-Mamba's own '3:3' active-parameter ratio) if omitted.",
    )
    parser.add_argument(
        "--moe-capacity-factor",
        type=float,
        default=MOE_CAPACITY_FACTOR,
        help="Switch-style expert capacity buffer above an even token split.",
    )
    parser.add_argument(
        "--moe-load-balance-alpha",
        type=float,
        default=MOE_LOAD_BALANCE_ALPHA,
        help="Weight on the Switch-style auxiliary load-balancing loss.",
    )
    parser.add_argument("--moe-top-k", type=int, default=MOE_TOP_K, help="Number of top-k")
    parser.add_argument(
        "--split-graph",
        action="store_true",
        help="Alternate Phase 3, Step 2, Option 5: enable the "
        "split-compute-graph controller (parallel Mamba backbone + "
        "sequential memory addressing). DISABLED unless this flag is "
        "passed -- roadmap flags this as untested/isolated-ablation-"
        "required. Overrides --controller entirely when set.",
    )
    parser.add_argument(
        "--split-graph-backbone-only",
        action="store_true",
        help=(
            "Build the split-graph backbone without DNC memory or its sequential "
            "read combiner. Keeps backbone MoE, workspace, task heads, and general "
            "training/evaluation features. DNC-specific memory/addressing options "
            "are unavailable. Requires --split-graph."
        ),
    )
    parser.add_argument(
        "--split-graph-variant",
        type=str,
        default=SPLIT_GRAPH_MAMBA_VARIANT,
        choices=["mamba1", "mamba2", "mamba3", "cfc", "mamba1+cfc", "mamba2+cfc", "mamba3+cfc"],
        help="Backbone variant for --split-graph. 'mamba2' requires "
        "mamba_ssm.modules.mamba2.Mamba2 to be importable. - same with 'mamba3'",
    )
    parser.add_argument(
        "--split-graph-num-blocks",
        type=int,
        default=SPLIT_GRAPH_NUM_BLOCKS,
        help="Number of stacked backbone blocks for --split-graph.",
    )
    parser.add_argument(
        "--split-graph-headdim",
        type=int,
        default=SPLIT_GRAPH_MAMBA_HEADDIM,
        help="Mamba2 headdim (mamba2 variant only).",
    )
    parser.add_argument(
        "--split-graph-no-combine",
        action="store_true",
        help="Built-in ablation for the roadmap's required 'read now, "
        "decide next hop' verification: disables the read-vector "
        "combiner entirely, so addressing never sees any read vector. "
        "Only meaningful with --split-graph-combiner-mode=linear.",
    )
    parser.add_argument(
        "--split-graph-combiner-mode",
        type=str,
        default=SPLIT_GRAPH_COMBINER_MODE,
        choices=["linear", "controller"],
        help="'linear' (default) is the original, already-efficient "
        "stateless combiner. 'controller' drives the sequential "
        "addressing step with a real interleaved Mamba controller "
        "cell instead (see --split-graph-combiner-variant).",
    )
    parser.add_argument(
        "--split-graph-combiner-variant",
        type=str,
        default=SPLIT_GRAPH_COMBINER_VARIANT,
        choices=[
            "mamba1",
            "mamba2",
            "mamba3",
            "cfc",
            "mamba+cfc",
            "mamba2+cfc",
            "mamba3+cfc",
            "cfc+mamba",
            "cfc+cfc",
            "mamba+mamba",
            "mamba2+mamba2",
            "mamba3+mamba3",
        ],
        help="v11: which controller drives the sequential combiner when "
        "--split-graph-combiner-mode=controller. 'mamba1' is the "
        "recommended/efficient choice; 'mamba2' is wired but not "
        "expected to be used (known inefficient, same reason plain "
        "--controller mamba2 runs aren't pursued). Same-kind stacks "
        "(e.g. 'cfc+cfc') isolate depth from cross-kind mixing -- "
        "hybrid_controller.py's STAGE_KINDS already supports "
        "duplicated kinds, this just widens the CLI's accepted list.",
    )
    parser.add_argument(
        "--split-graph-combiner-num-blocks",
        type=int,
        default=SPLIT_GRAPH_COMBINER_NUM_BLOCKS,
        help="Number of stacked blocks in the controller combiner "
        "(--split-graph-combiner-mode=controller only). Kept small "
        "by default -- this step is meant to stay cheap.",
    )
    parser.add_argument(
        "--memory-type",
        choices=("none", "dnc", "kv", "delta"),
        default="dnc",
        help=(
            "Memory module for split-graph controllers. KV and delta operate in parallel over "
            "controller features; DNC preserves the recurrent read/write path."
        ),
    )
    parser.add_argument("--memory-heads", type=int, default=4)
    parser.add_argument("--memory-dk", type=int, default=64)
    parser.add_argument("--memory-dv", type=int, default=64)
    parser.add_argument("--memory-read-lag", type=int, choices=(0, 1), default=1)
    parser.add_argument("--memory-topk", type=int, default=None)
    parser.add_argument("--memory-position", choices=("none", "rope"), default="none")
    parser.add_argument(
        "--delta-gate", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--delta-chunk", type=int, default=64)
    parser.add_argument("--delta-use-fla", action="store_true")
    parser.add_argument(
        "--dnc-read-variant",
        choices=["cosine", "relational-mlp", "relational-residual"],
        default=DNC_READ_VARIANT,
        help="Read content score for --split-graph. relational-mlp replaces "
        "cosine similarity; relational-residual adds a learned pairwise score "
        "to cosine and starts exactly at the DNC baseline. DNC write addressing "
        "and temporal links remain intact.",
    )
    parser.add_argument(
        "--dnc-read-residual-scale",
        type=float,
        default=DNC_READ_RESIDUAL_SCALE,
        help="Gain applied before tanh to the normalized, centered relational correction; "
        "ignored by other read variants. The correction is still bounded by the max-ratio.",
    )
    parser.add_argument(
        "--dnc-read-residual-max-ratio",
        type=float,
        default=DNC_READ_RESIDUAL_MAX_RATIO,
        help="Maximum absolute relational correction as a fraction of per-query cosine RMS; "
        "ignored by other read variants.",
    )
    parser.add_argument(
        "--dnc-write-variant",
        choices=["learned", "kanerva-closed-form"],
        default=DNC_WRITE_VARIANT,
        help="Write update for --split-graph. kanerva-closed-form performs an "
        "online Gaussian posterior update at each timestep.",
    )
    parser.add_argument(
        "--dnc-write-observation-variance",
        type=float,
        default=DNC_WRITE_OBSERVATION_VARIANCE,
        help="Fixed positive observation variance for the Kanerva-style write.",
    )
    parser.add_argument(
        "--dnc-link-sharpness",
        type=float,
        nargs="?",
        const=2.0,
        default=1.0,
        metavar="EXPONENT",
        help="Sharpen the shape of forward/backward temporal-link reads using a "
        "fixed exponent inspired by arXiv:1904.10278 while preserving each "
        "vector's original mass (default exponent 2 when specified; 1 leaves "
        "the DNC unchanged). Adds no parameters, so it can resume from a "
        "baseline checkpoint.",
    )
    parser.add_argument(
        "--dnc-allocation-topk",
        type=int,
        default=None,
        metavar="K",
        help="Approximate DNC allocation using only the K least-used cells. "
        "This can reduce allocation selection work, but omits allocation mass "
        "outside the selected cells and may change learning behavior.",
    )
    parser.add_argument(
        "--dnc-backward-trace",
        action="store_true",
        help="Trace the first non-finite gradient through DNC internals and report "
        "windowed memory/read/link magnitudes. Diagnostic only; adds synchronization.",
    )
    parser.add_argument(
        "--dnc-backward-trace-step",
        type=int,
        default=None,
        metavar="STEP",
        help=(
            "Restrict --dnc-backward-trace hooks to one absolute training step "
            "to reduce diagnostic overhead."
        ),
    )
    parser.add_argument(
        "--split-graph-numeric-trace-step",
        type=int,
        default=None,
        metavar="STEP",
        help=(
            "Trace finite counts, dtypes, shapes, and peak finite magnitudes at split-graph "
            "backbone/CfC/DNC boundaries for one absolute training step; stop at the first "
            "observed non-finite tensor. Writes run_<id>_splitgraph_forward_trace.csv. "
            "This diagnostic synchronizes CUDA and should be targeted at a replayable step."
        ),
    )
    parser.add_argument(
        "--split-graph-numeric-trace-on-error",
        action="store_true",
        help=(
            "Keep split-graph numeric tracing dormant during training. When an "
            "existing finite check detects a non-finite forward result, replay "
            "that same batch once and trace split-graph boundaries to the first "
            "non-finite tensor. Writes run_<id>_splitgraph_forward_trace.csv."
        ),
    )
    dnc_impl_group = parser.add_mutually_exclusive_group()
    dnc_impl_group.add_argument(
        "--base-dnc",
        dest="dnc_implementation",
        action="store_const",
        const="base",
        help="Use the upstream DNC allocation implementation (default).",
    )
    dnc_impl_group.add_argument(
        "--optimized-dnc",
        action=_OptimizedDNCAction,
        nargs=0,
        help="Enable exact fast allocation and compiled memory forward. For "
        "--split-graph, also enable top-64 allocation and exponent-2 temporal "
        "link sharpening as separate functional experiments.",
    )
    parser.set_defaults(dnc_implementation="base", _dnc_optimized_bundle=False)
    dnc_impl_group.add_argument(
        "--dnc-fast-allocation",
        dest="dnc_implementation",
        action="store_const",
        const="optimized",
        help="Use the exact allocation path with one usage-permutation sort removed (without compilation).",
    )
    parser.add_argument(
        "--dnc-compile-memory",
        action="store_true",
        help="Compile the DNC memory forward with torch.compile; keeps DNC equations unchanged.",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default=DATASET_TYPE,
        choices=[
            "graph",
            "text",
            "audio",
            "video",
            "multimodal",
            "surprise-recall",
        ],
        help="Dataset plugged into the core loop (data/dataset_registry.py).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Training episodes per optimizer step (default: {BATCH_SIZE}); "
        "larger values may improve device utilization at higher memory use and "
        "change the effective gradient batch.",
    )
    parser.add_argument(
        "--sgr-pred-loss-weight",
        type=float,
        default=0.0,
        help="Enable the SGR next-token decoder with this auxiliary CE weight. "
        "Requires --dataset-type surprise-recall and --split-graph.",
    )
    parser.add_argument(
        "--sgr-pred-eval-episodes",
        type=int,
        default=128,
        help="Fixed probe episodes per lesson for periodic active-lesson and final all-lesson "
        "Phase 1 signal evaluation (default: 128).",
    )
    parser.add_argument(
        "--sgr-phase2-arm",
        choices=["content", "aux-only", "error", "blend", "shuffled", "oracle"],
        default=None,
        help="Run one isolated Phase 2 SGR arm with a true two-block Mamba-1 backbone and no DNC. "
        "Decoder arms (aux-only/error/blend/shuffled) require --sgr-pred-loss-weight > 0.",
    )
    parser.add_argument(
        "--sgr-phase2-signal",
        choices=["surprisal", "excess-surprise"],
        default="excess-surprise",
        help="Prediction-error signal used by Phase 2 error/blend/shuffled arms.",
    )
    parser.add_argument(
        "--sgr-dnc-write-salience",
        action="store_true",
        help="Phase 4: add detached SGR prediction error to the DNC write-strength logits.",
    )
    parser.add_argument(
        "--sgr-workspace-write-salience",
        action="store_true",
        help="Phase 4: add detached SGR prediction error to the workspace write-gate logits; requires --workspace.",
    )
    parser.add_argument(
        "--sgr-gate-eval-override",
        choices=["none", "zero", "shuffle"],
        default="none",
        help="Inference-only pathway-dependency ablation for Phase 2 gated arms.",
    )
    parser.add_argument(
        "--dataset-link",
        type=str,
        default=DATASET_LINK,
        help="Real training-data source for --dataset-type: a graph edge file "
        "(graph), a text file (text), an audio file (audio, needs torchaudio), "
        "or a video file (video, needs opencv-python). Omit (or 'graph-traversal' "
        "for graph) to keep the built-in synthetic data for that type.",
    )
    parser.add_argument(
        "--test-dataset-link",
        type=str,
        default=None,
        help="Path to a real held-out TEST source (a second text/audio/video file, "
        "or a graph edge file), used instead of the default synthetic/seeded "
        "held-out split -- the modality's analogue of the graph dataset's "
        "London Underground test graph. Omit to keep default behavior: for "
        "graph, the built-in London Underground; for text/audio/video, a "
        "disjoint-key held-out slice of --dataset-link (or the fully synthetic "
        "seeded table if --dataset-link is also omitted).",
    )
    parser.add_argument(
        "--ogs-weights",
        type=float,
        nargs=3,
        default=list(OGS_WEIGHTS),
        metavar=("OFFSET", "MEMORY", "PERFECT"),
        help="nonnegative OGS component weights in the order shown.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        metavar="DIRECTORY",
        help="save the prepared dataset under DIRECTORY/<type>/<name> [prepared].",
    )
    parser.add_argument(
        "--load-prepared",
        type=str,
        default=None,
        metavar="DIRECTORY",
        help="load an already prepared dataset directory; --dataset-link is not needed.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="prepare and optionally save the dataset, then exit without training.",
    )
    args = parser.parse_args()

    if args.prepare_only and args.load_prepared:
        parser.error("--prepare-only cannot be combined with --load-prepared")
    if args.dnc_write_observation_variance <= 0:
        parser.error("--dnc-write-observation-variance must be positive")
    if args._dnc_optimized_bundle and args.split_graph:
        if args.dnc_link_sharpness == 1.0:
            args.dnc_link_sharpness = 2.0
        if args.dnc_allocation_topk is None:
            args.dnc_allocation_topk = 64
    if args.dnc_link_sharpness <= 0:
        parser.error("--dnc-link-sharpness must be positive")
    if args.dnc_allocation_topk is not None and args.dnc_allocation_topk <= 0:
        parser.error("--dnc-allocation-topk must be positive")
    if args.dnc_allocation_topk is not None and not args.split_graph:
        parser.error("--dnc-allocation-topk requires --split-graph")
    if args.memory_type != "dnc" and not args.split_graph:
        parser.error("non-DNC memory types require --split-graph")
    if args.split_graph_backbone_only and args.memory_type != "dnc":
        parser.error("--split-graph-backbone-only cannot be combined with --memory-type")
    if args.memory_type != "dnc" and args.dynamic_n_mode:
        parser.error("--dynamic-n-mode is only supported with DNC memory")
    if min(args.memory_heads, args.memory_dk, args.memory_dv, args.delta_chunk) <= 0:
        parser.error("memory dimensions and --delta-chunk must be positive")
    if args.memory_topk is not None and args.memory_topk <= 0:
        parser.error("--memory-topk must be positive")
    if args.memory_topk is not None and args.memory_type != "kv":
        parser.error("--memory-topk requires --memory-type kv")
    if args.delta_use_fla and args.memory_type != "delta":
        parser.error("--delta-use-fla requires --memory-type delta")
    if args.delta_use_fla and args.delta_chunk not in {16, 32, 64}:
        parser.error("--delta-use-fla requires --delta-chunk 16, 32, or 64")
    if args.memory_type != "dnc" and (
        args.dnc_compile_memory or args.dnc_backward_trace or args.dnc_implementation != "base"
    ):
        parser.error(
            "DNC compilation, tracing, and implementation flags require --memory-type dnc"
        )
    if args.memory_type != "dnc" and (
        args.link_matrix_mode != "dense"
        or args.isolate_link_ablation
        or args.dnc_allocation_topk is not None
        or args.dnc_read_variant != "cosine"
        or args.dnc_write_variant != "learned"
        or args.dnc_read_residual_scale != 1.0
        or args.dnc_read_residual_max_ratio != 0.5
        or args.dnc_write_observation_variance != 1.0
        or args.dnc_link_sharpness != 1.0
        or args.dnc_backward_trace_step is not None
        or args.sgr_dnc_write_salience
    ):
        parser.error("DNC addressing and write-salience options require --memory-type dnc")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    if args.dnc_link_sharpness != 1.0 and not args.split_graph:
        parser.error("--dnc-link-sharpness requires --split-graph")
    if args.dnc_backward_trace and not args.split_graph:
        parser.error("--dnc-backward-trace requires --split-graph")
    if args.dnc_backward_trace_step is not None and not args.dnc_backward_trace:
        parser.error("--dnc-backward-trace-step requires --dnc-backward-trace")
    if args.dnc_backward_trace_step is not None and args.dnc_backward_trace_step < 0:
        parser.error("--dnc-backward-trace-step must be >= 0")
    if args.split_graph_numeric_trace_step is not None and not args.split_graph:
        parser.error("--split-graph-numeric-trace-step requires --split-graph")
    if args.split_graph_numeric_trace_step is not None and args.split_graph_numeric_trace_step < 0:
        parser.error("--split-graph-numeric-trace-step must be >= 0")
    if args.split_graph_numeric_trace_step is not None and args.eval_only:
        parser.error("--split-graph-numeric-trace-step requires a training run")
    if args.split_graph_numeric_trace_on_error and not args.split_graph:
        parser.error("--split-graph-numeric-trace-on-error requires --split-graph")
    if args.split_graph_numeric_trace_on_error and args.eval_only:
        parser.error("--split-graph-numeric-trace-on-error requires a training run")
    if args.split_graph_numeric_trace_on_error and args.split_graph_numeric_trace_step is not None:
        parser.error(
            "--split-graph-numeric-trace-on-error cannot be combined with "
            "--split-graph-numeric-trace-step"
        )
    if args.dnc_read_residual_scale < 0:
        parser.error("--dnc-read-residual-scale must be non-negative")
    if args.dnc_read_residual_max_ratio < 0:
        parser.error("--dnc-read-residual-max-ratio must be non-negative")
    if args.split_graph_backbone_only:
        if not args.split_graph:
            parser.error("--split-graph-backbone-only requires --split-graph")
        if args.split_graph_combiner_mode != "linear":
            parser.error(
                "--split-graph-backbone-only has no sequential combiner; "
                "omit --split-graph-combiner-mode=controller"
            )
        if args.moe_placement == "combiner":
            parser.error("--split-graph-backbone-only supports backbone MoE, not combiner MoE")
        if (
            args.dynamic_n_mode
            or args.link_matrix_mode != "dense"
            or args.isolate_link_ablation
            or args.dnc_compile_memory
            or args.dnc_backward_trace
            or args.dnc_read_variant != "cosine"
            or args.dnc_write_variant != "learned"
            or args.dnc_read_residual_scale != 1.0
            or args.dnc_read_residual_max_ratio != 0.5
            or args.dnc_write_observation_variance != 1.0
            or args.dnc_link_sharpness != 1.0
            or args.dnc_allocation_topk is not None
            or args.dnc_implementation != "base"
            or args.sgr_dnc_write_salience
        ):
            parser.error(
                "--split-graph-backbone-only cannot be combined with DNC-specific "
                "memory, addressing, tracing, or write-salience options"
            )
        if args.sgr_phase2_arm is not None:
            parser.error(
                "Phase 2 SGR arms already use their dedicated backbone-only model; "
                "omit --split-graph-backbone-only"
            )
    if not args.split_graph and (
        args.dnc_read_variant != "cosine" or args.dnc_write_variant != "learned"
    ):
        parser.error("--dnc-read/write-variant require --split-graph")
    if args.prepare_only and not args.save:
        parser.error("--prepare-only requires --save DIRECTORY")
    if args.save and args.load_prepared:
        parser.error("--save and --load-prepared cannot be combined")
    if any(weight < 0 for weight in args.ogs_weights) or sum(args.ogs_weights) <= 0:
        parser.error("--ogs-weights must be nonnegative and at least one must be positive")
    if args.load_prepared and args.dataset_type == "multimodal":
        parser.error("--load-prepared currently supports graph, text, audio, and video datasets")
    prepared_dataset = None
    if args.load_prepared:
        prepared_dataset = load_prepared_dataset(args.load_prepared, args.dataset_type)
    elif args.save or args.prepare_only:
        if args.dataset_type == "multimodal":
            parser.error("prepared dataset saving currently supports graph, text, audio, and video")
        prepared_dataset = get_dataset(
            args.dataset_type,
            args.dataset_link,
            test_dataset_link=args.test_dataset_link,
        )

        if args.save:
            save_prepared_dataset(prepared_dataset, args.save, args.dataset_type, args.dataset_link)
        if args.prepare_only:
            raise SystemExit(0)

    if args.resume is not None and args.beta is None:
        raise SystemExit(
            "--resume requires the beta positional arg too, e.g.:\n"
            f"  python3 {sys.argv[0]} 0.02 --resume checkpoints/beta_0p02_latest.pt"
        )
    if args.eval_only and args.resume is None:
        parser.error("--eval-only requires --resume <checkpoint.pt>")
    if not args.eval_only and args.eval_workspace_mode != "checkpoint":
        parser.error("--eval-workspace-mode is only valid with --eval-only")
    if args.eval_seed is not None and not args.eval_only:
        parser.error("--eval-seed is only valid with --eval-only")
    if args.sgr_phase2_arm is not None:
        if args.dataset_type != "surprise-recall":
            parser.error("--sgr-phase2-arm requires --dataset-type surprise-recall")
        if (
            not args.split_graph
            or args.split_graph_variant != "mamba1"
            or args.split_graph_num_blocks != 2
        ):
            parser.error(
                "Phase 2 requires --split-graph --split-graph-variant mamba1 --split-graph-num-blocks 2"
            )
        if args.moe or args.workspace or args.dynamic_n_mode or args.dnc_compile_memory:
            parser.error(
                "Phase 2 is backbone-only; disable MoE, workspace, Dynamic-N, and DNC compilation"
            )
        if (
            args.dnc_backward_trace
            or args.split_graph_numeric_trace_step is not None
            or args.split_graph_numeric_trace_on_error
        ):
            parser.error(
                "DNC/split-graph numeric tracing options are not applicable to the Phase 2 backbone-only model"
            )
        decoder_arm = args.sgr_phase2_arm in {"aux-only", "error", "blend", "shuffled"}
        if decoder_arm and args.sgr_pred_loss_weight <= 0:
            parser.error("Phase 2 decoder arms require --sgr-pred-loss-weight > 0")
        if not decoder_arm and args.sgr_pred_loss_weight != 0:
            parser.error("content/oracle Phase 2 arms require --sgr-pred-loss-weight 0")
        if args.sgr_gate_eval_override != "none" and args.sgr_phase2_arm == "aux-only":
            parser.error("--sgr-gate-eval-override is only meaningful for gated Phase 2 arms")
        if args.sgr_dnc_write_salience or args.sgr_workspace_write_salience:
            parser.error("Phase 2 arms cannot be combined with Phase 4 salience consumers")
        if args.workspace_reconsideration and args.sgr_pred_loss_weight > 0.0:
            parser.error("Phase 2 arms cannot be combined with Phase 4 workspace controls")
    elif args.sgr_gate_eval_override != "none":
        parser.error("--sgr-gate-eval-override requires --sgr-phase2-arm")
    sgr_phase4_requested = (
        args.sgr_dnc_write_salience
        or args.sgr_workspace_write_salience
        or (args.workspace_reconsideration and args.sgr_pred_loss_weight > 0.0)
    )
    if sgr_phase4_requested:
        if args.sgr_phase2_arm is not None:
            parser.error("Phase 2 arms cannot be combined with Phase 4 consumers or controls")
        if args.dataset_type != "surprise-recall":
            parser.error(
                "Phase 4 prediction-error consumers require --dataset-type surprise-recall"
            )
        if (
            not args.split_graph
            or args.split_graph_variant != "mamba1"
            or args.split_graph_num_blocks < 2
        ):
            parser.error(
                "Phase 4 SGR consumers require split-graph Mamba-1 with at least two blocks"
            )
        if args.sgr_pred_loss_weight <= 0:
            parser.error("Phase 4 SGR consumers require --sgr-pred-loss-weight > 0")
        if args.sgr_dnc_write_salience and args.dnc_compile_memory:
            parser.error(
                "--dnc-compile-memory does not compile the Phase 4 prediction-error write path"
            )
        if args.sgr_workspace_write_salience and not args.workspace:
            parser.error("--sgr-workspace-write-salience requires --workspace")
        if args.sgr_workspace_write_salience and args.workspace_num_slots < 1:
            parser.error("--sgr-workspace-write-salience requires --workspace-num-slots >= 1")
        if args.sgr_dnc_write_salience and args.workspace and args.workspace_train_mode == "frozen":
            parser.error(
                "DNC write salience cannot be learned with --workspace-train-mode frozen; "
                "use coadapted"
            )
        if (
            args.workspace_reconsideration_trigger.startswith("prediction_error")
            and args.workspace_reconsideration
            and args.workspace_num_slots < 1
        ):
            parser.error("prediction-error reconsideration requires --workspace-num-slots >= 1")
    if args.eval_only and args.eval_workspace_mode != "checkpoint" and not args.workspace:
        parser.error("non-default --eval-workspace-mode requires --workspace")
    if args.eval_only and args.run_id_suffix is None:
        args.run_id_suffix = f"eval-{args.eval_workspace_mode}"
        if args.eval_seed is not None:
            args.run_id_suffix += f"-seed{args.eval_seed}"

    if args.link_matrix_mode == "sparse_topk" and args.link_matrix_topk is None:
        raise SystemExit("--link-matrix-mode=sparse_topk requires --link-matrix-topk")
    if args.workspace and not args.split_graph:
        parser.error("--workspace requires --split-graph")
    if args.workspace and not args.moe:
        parser.error("--workspace requires --moe")
    if args.moe_placement != "all" and not args.moe:
        parser.error("--moe-placement backbone/combiner requires --moe")
    if args.moe_placement != "all" and not args.split_graph:
        parser.error("--moe-placement backbone/combiner requires --split-graph")
    if args.moe_placement == "combiner" and args.split_graph_combiner_mode != "controller":
        parser.error("--moe-placement combiner requires --split-graph-combiner-mode controller")
    if args.moe_backbone_block_isolation is not None:
        if not args.moe or not args.split_graph:
            parser.error("--moe-backbone-block-isolation requires --moe and --split-graph")
        if args.moe_placement != "backbone":
            parser.error(
                "--moe-backbone-block-isolation requires --moe-placement backbone "
                "to keep combiner MoE out of the isolation run"
            )
        if args.workspace:
            parser.error("--moe-backbone-block-isolation cannot be combined with --workspace")
        if not 0 <= args.moe_backbone_block_isolation < args.split_graph_num_blocks:
            parser.error(
                "--moe-backbone-block-isolation must be in [0, "
                f"{args.split_graph_num_blocks - 1}] for the configured number of blocks"
            )
    if (
        not math.isfinite(args.moe_backbone_residual_scale)
        or args.moe_backbone_residual_scale < 0.0
    ):
        parser.error("--moe-backbone-residual-scale must be finite and >= 0")
    if args.moe_backbone_residual_scale != 1.0:
        if not args.moe or not args.split_graph:
            parser.error("--moe-backbone-residual-scale requires --moe and --split-graph")
        if args.moe_placement not in ("backbone", "all"):
            parser.error("--moe-backbone-residual-scale requires --moe-placement backbone or all")
    if args.workspace and args.moe_placement == "combiner":
        parser.error("--workspace requires backbone MoE; use --moe-placement backbone or all")
    if args.workspace_num_slots < 0:
        parser.error("--workspace-num-slots must be >= 0")
    if args.workspace_bptt_horizon < 0:
        parser.error("--workspace-bptt-horizon must be >= 0")
    if (
        not math.isfinite(args.workspace_reconsideration_surprise_threshold)
        or not 0.0 <= args.workspace_reconsideration_surprise_threshold <= 1.0
    ):
        parser.error("--workspace-reconsideration-surprise-threshold must be finite and in [0, 1]")
    if args.workspace_reconsideration and not args.workspace:
        parser.error("--workspace-reconsideration requires --workspace")
    if (
        args.workspace_reconsideration_trigger != "innovation"
        and not args.workspace_reconsideration
    ):
        parser.error(
            "--workspace-reconsideration-trigger=random_rate_matched requires "
            "--workspace-reconsideration"
        )
    if args.workspace_reconsideration and not args.moe_diagnostics:
        parser.error("--workspace-reconsideration requires --moe-diagnostics for trigger telemetry")
    if args.workspace_train_scope == "gate" and not args.workspace:
        parser.error("--workspace-train-scope=gate requires --workspace")
    if args.workspace_gate_variant == "bistable" and (
        not args.workspace or args.workspace_num_slots < 2
    ):
        parser.error(
            "--workspace-gate-variant=bistable requires --workspace and at least two slots"
        )
    if args.workspace_gate_variant == "bistable" and not args.moe_diagnostics:
        parser.error("bistable gate histogram/diagnostics require --moe-diagnostics")
    if args.bistable_debug and args.workspace_gate_variant != "bistable":
        parser.error("--bistable-debug requires --workspace-gate-variant=bistable")
    if args.bistable_debug and (not args.workspace or not args.split_graph):
        parser.error("--bistable-debug requires --workspace and --split-graph")
    if args.bistable_anomaly_detection and args.workspace_gate_variant != "bistable":
        parser.error("--bistable-anomaly-detection requires --workspace-gate-variant=bistable")
    if args.bistable_anomaly_detection and (not args.workspace or not args.split_graph):
        parser.error("--bistable-anomaly-detection requires --workspace and --split-graph")
    if args.bistable_anomaly_step is not None and not args.bistable_anomaly_detection:
        parser.error("--bistable-anomaly-step requires --bistable-anomaly-detection")
    if args.bistable_anomaly_step is not None and args.bistable_anomaly_step < 0:
        parser.error("--bistable-anomaly-step must be >= 0")
    if (
        not math.isfinite(args.workspace_bistable_selection_scale)
        or args.workspace_bistable_selection_scale < 0
    ):
        parser.error("--workspace-bistable-selection-scale must be finite and >= 0")
    if (
        not math.isfinite(args.workspace_bistable_noise_scale)
        or args.workspace_bistable_noise_scale < 0
    ):
        parser.error("--workspace-bistable-noise-scale must be finite and >= 0")
    if (
        not math.isfinite(args.workspace_stability_max_activation)
        or args.workspace_stability_max_activation <= 0
    ):
        parser.error("--workspace-stability-max-activation must be finite and > 0")

    if args.beta is not None:
        betas_to_run = [args.beta]
        if args.beta not in BETAS_TO_SWEEP:
            print(
                f"WARNING: {args.beta} is not in the planned sweep "
                f"{BETAS_TO_SWEEP} -- running it anyway, but it won't be "
                f"picked up by the combined summary step below."
            )
    else:
        betas_to_run = BETAS_TO_SWEEP

    all_summaries = []
    if args.bistable_anomaly_detection:
        # Anomaly detection must cover both forward graph construction and
        # backward execution in order to print the originating forward stack.
        # Keep it opt-in because it retains extra graph metadata and is slow.
        if args.bistable_anomaly_step is None:
            print(
                "[bistable-anomaly] PyTorch autograd anomaly detection enabled "
                "for every step; the run will be substantially slower."
            )
            torch.autograd.set_detect_anomaly(True, check_nan=True)
        else:
            print(
                f"[bistable-anomaly] anomaly detection restricted to training "
                f"step {args.bistable_anomaly_step}; all other steps use normal autograd."
            )
    for beta in betas_to_run:
        # These are made to avoid checkpoint collisions between different types of runs.
        run_id = f"beta_{beta}_seed{args.seed}_learnedprior".replace(".", "p")
        if args.memory_type != "dnc":
            run_id = (
                f"{run_id}_mem{args.memory_type}_h{args.memory_heads}"
                f"_dk{args.memory_dk}_dv{args.memory_dv}_lag{args.memory_read_lag}"
            )
            if args.memory_type == "kv":
                run_id += f"_{args.memory_position}"
                if args.memory_topk is not None:
                    run_id += f"_topk{args.memory_topk}"
            if args.memory_type == "delta":
                run_id += f"_chunk{args.delta_chunk}_{'gate' if args.delta_gate else 'nogate'}"
                if args.delta_use_fla:
                    run_id += "_fla"
        if args.controller == "mamba":
            run_id = f"{run_id}_mambactrl"
        elif args.controller == "mamba2":
            run_id = f"{run_id}_mamba2ctrl"
        elif args.controller == "mamba3":
            run_id = f"{run_id}_mamba3ctrl"
        elif args.controller == "cfc":
            run_id = f"{run_id}_cfcctrl"
        elif "+" in args.controller:
            run_id = f"{run_id}_{args.controller.replace('+', '')}ctrl"
        if args.beta_mode == "dynamic":
            run_id = f"{run_id}_dynbeta"
        if args.link_matrix_mode != "dense":
            tag = (
                args.link_matrix_mode
                if args.link_matrix_mode != "sparse_topk"
                else f"sparsetopk{args.link_matrix_topk}"
            )
            run_id = f"{run_id}_link{tag}"
        if args.dynamic_n_mode:
            run_id = f"{run_id}_dynN"
        if args.moe:
            run_id = f"{run_id}_moe{args.moe_num_experts}e"
            if not args.moe_diagnostics:
                run_id += "_moediagoff"
            if args.split_graph and args.moe_placement != "all":
                run_id += f"_{args.moe_placement}only"
            if args.moe_backbone_block_isolation is not None:
                run_id += f"_backboneblock{args.moe_backbone_block_isolation}only"
            if args.moe_backbone_residual_scale != 1.0:
                run_id += f"_backbonerscale{args.moe_backbone_residual_scale:g}"
        if args.workspace:
            run_id += f"_workspace{args.workspace_num_slots}s_{args.workspace_train_mode}"
            if args.workspace_gate_variant != "continuous" or args.workspace_train_scope == "gate":
                run_id += f"_gate{args.workspace_gate_variant}_{args.workspace_train_scope}"
            if args.workspace_gate_variant == "bistable":
                run_id += (
                    f"_sel{args.workspace_bistable_selection_scale:g}"
                    f"_noise{args.workspace_bistable_noise_scale:g}"
                    f"_grad{args.workspace_bistable_gradient_mode}"
                )
                if args.workspace_bistable_deterministic:
                    run_id += "_detgate"
            if args.workspace_bptt_horizon:
                run_id += f"_wbptt{args.workspace_bptt_horizon}"
            if args.workspace_reconsideration:
                threshold_tag = f"{args.workspace_reconsideration_surprise_threshold:g}".replace(
                    ".", "p"
                ).replace("-", "m")
                trigger_tag = {
                    "innovation": "innovation",
                    "random_rate_matched": "randommatched",
                    "prediction_error": "prederror",
                    "prediction_error_random_rate_matched": "prederrormatched",
                }[args.workspace_reconsideration_trigger]
                run_id += f"_reconsider{trigger_tag}{threshold_tag}"
            if args.workspace_no_selection:
                run_id += "_randomwrite"
            if args.workspace_reset_each_step:
                run_id += "_resetstep"
        if args.split_graph:
            run_id = f"{run_id}_splitgraph_{args.split_graph_variant.replace('+', '')}"
            if args.split_graph_backbone_only:
                run_id += "_backboneonly"
            if args.dnc_backward_trace:
                run_id += "_dncbacktrace"
            if args.dnc_read_variant != "cosine" or args.dnc_write_variant != "learned":
                run_id += f"_read-{args.dnc_read_variant}_write-{args.dnc_write_variant}"
            if args.dnc_read_variant == "relational-residual":
                run_id += f"_rscale{args.dnc_read_residual_scale:g}"
                run_id += f"_rmax{args.dnc_read_residual_max_ratio:g}"
            if args.dnc_write_variant == "kanerva-closed-form":
                run_id += f"_obsvar{args.dnc_write_observation_variance:g}"
            if args.dnc_link_sharpness != 1.0:
                run_id += f"_linksharp{args.dnc_link_sharpness:g}"
            if args.dnc_allocation_topk is not None:
                run_id += f"_allocK{args.dnc_allocation_topk}"
            if args.split_graph_combiner_mode == "controller":
                run_id = f"{run_id}_combctrl{args.split_graph_combiner_variant.replace('+', '')}"
        if args.dnc_implementation == "optimized":
            run_id += "_optimizeddnc" if args.dnc_compile_memory else "_dncfastalloc"
        elif args.dnc_compile_memory:
            run_id += "_dnccompiled"
        if args.dataset_type != "graph":
            run_id = f"{run_id}_ds{args.dataset_type}"
        if args.sgr_pred_loss_weight > 0:
            pred_weight_tag = f"{args.sgr_pred_loss_weight:g}".replace(".", "p")
            run_id += f"_sgrpred{pred_weight_tag}"
        if args.sgr_phase2_arm is not None:
            run_id += f"_sgrphase2-{args.sgr_phase2_arm}_{args.sgr_phase2_signal}"
            if args.sgr_gate_eval_override != "none":
                run_id += f"_gateeval-{args.sgr_gate_eval_override}"
        if args.sgr_dnc_write_salience:
            run_id += f"_sgrdncwrite-{args.sgr_phase2_signal}"
        if args.sgr_workspace_write_salience:
            run_id += f"_sgrworkspacewrite-{args.sgr_phase2_signal}"
        if args.workspace_reconsideration_trigger.startswith("prediction_error"):
            run_id += (
                f"_sgrreconsider-{args.workspace_reconsideration_trigger}_{args.sgr_phase2_signal}"
            )
        if args.batch_size != BATCH_SIZE:
            run_id += f"_bs{args.batch_size}"
        if args.run_id_suffix:
            run_id = f"{run_id}_{args.run_id_suffix}"
        summary = run(
            beta_target=beta,
            run_id=run_id,
            seed=args.seed,
            resume_from=args.resume,
            controller=args.controller,  # v7
            beta_mode=args.beta_mode,
            total_steps=(args.total_steps if args.total_steps is not None else TOTAL_STEPS),
            checkpoint_every=(
                args.checkpoint_every if args.checkpoint_every is not None else CHECKPOINT_EVERY
            ),
            log_every=args.log_every,
            link_matrix_mode=args.link_matrix_mode,
            link_matrix_topk=args.link_matrix_topk,
            isolate_link_ablation=args.isolate_link_ablation,
            dynamic_n_mode=args.dynamic_n_mode,
            dynamic_n_floor=args.dynamic_n_floor,
            dynamic_n_ceiling=args.dynamic_n_ceiling,
            dynamic_n_trigger_frac=args.dynamic_n_trigger_frac,
            dynamic_n_cooldown_steps=args.dynamic_n_cooldown_steps,
            moe_enabled=args.moe,
            moe_diagnostics_enabled=args.moe_diagnostics,
            segment_diagnostics_enabled=args.segment_diagnostics,
            split_graph_moe_placement=args.moe_placement,
            split_graph_moe_backbone_block_isolation=args.moe_backbone_block_isolation,
            split_graph_moe_backbone_residual_scale=args.moe_backbone_residual_scale,
            moe_num_experts=args.moe_num_experts,
            moe_expert_dim=args.moe_expert_dim,
            moe_capacity_factor=args.moe_capacity_factor,
            moe_load_balance_alpha=args.moe_load_balance_alpha,
            moe_top_k=args.moe_top_k,
            split_graph_enabled=args.split_graph,
            split_graph_backbone_only=args.split_graph_backbone_only,
            split_graph_variant=args.split_graph_variant,
            split_graph_num_blocks=args.split_graph_num_blocks,
            split_graph_headdim=args.split_graph_headdim,
            split_graph_combine_reads=not args.split_graph_no_combine,
            split_graph_combiner_mode=args.split_graph_combiner_mode,
            split_graph_combiner_variant=args.split_graph_combiner_variant,
            split_graph_combiner_num_blocks=args.split_graph_combiner_num_blocks,
            workspace_enabled=args.workspace,
            workspace_num_slots=args.workspace_num_slots,
            workspace_no_selection=args.workspace_no_selection,
            workspace_reset_each_step=args.workspace_reset_each_step,
            workspace_train_mode=args.workspace_train_mode,
            workspace_train_scope=args.workspace_train_scope,
            workspace_gate_variant=args.workspace_gate_variant,
            workspace_bistable_selection_scale=args.workspace_bistable_selection_scale,
            workspace_bistable_noise_scale=args.workspace_bistable_noise_scale,
            workspace_bistable_gradient_mode=args.workspace_bistable_gradient_mode,
            workspace_bistable_debug=args.bistable_debug,
            workspace_bistable_anomaly_detection=args.bistable_anomaly_detection,
            workspace_bistable_anomaly_step=args.bistable_anomaly_step,
            workspace_bptt_horizon=args.workspace_bptt_horizon,
            workspace_reconsideration_enabled=args.workspace_reconsideration,
            workspace_reconsideration_surprise_threshold=(
                args.workspace_reconsideration_surprise_threshold
            ),
            workspace_reconsideration_trigger_mode=(args.workspace_reconsideration_trigger),
            workspace_bistable_deterministic=args.workspace_bistable_deterministic,
            workspace_stability_max_activation=args.workspace_stability_max_activation,
            eval_only=args.eval_only,
            eval_workspace_mode=args.eval_workspace_mode,
            eval_seed=args.eval_seed,
            dnc_read_variant=args.dnc_read_variant,
            dnc_read_residual_scale=args.dnc_read_residual_scale,
            dnc_read_residual_max_ratio=args.dnc_read_residual_max_ratio,
            dnc_write_variant=args.dnc_write_variant,
            dnc_write_observation_variance=args.dnc_write_observation_variance,
            dnc_link_sharpness=args.dnc_link_sharpness,
            dnc_allocation_topk=args.dnc_allocation_topk,
            dnc_implementation=args.dnc_implementation,
            memory_type=args.memory_type,
            memory_heads=args.memory_heads,
            memory_dk=args.memory_dk,
            memory_dv=args.memory_dv,
            memory_read_lag=args.memory_read_lag,
            memory_topk=args.memory_topk,
            memory_position=args.memory_position,
            delta_gate=args.delta_gate,
            delta_chunk=args.delta_chunk,
            delta_use_fla=args.delta_use_fla,
            dnc_compile_memory=args.dnc_compile_memory,
            dnc_backward_trace=args.dnc_backward_trace,
            dnc_backward_trace_step=args.dnc_backward_trace_step,
            split_graph_numeric_trace_step=args.split_graph_numeric_trace_step,
            split_graph_numeric_trace_on_error=args.split_graph_numeric_trace_on_error,
            dataset_type=args.dataset_type,
            dataset_link=args.dataset_link,
            test_dataset_link=args.test_dataset_link,
            prepared_dataset=prepared_dataset,
            ogs_weights=tuple(args.ogs_weights),
            batch_size=args.batch_size,
            sgr_prediction_loss_weight=args.sgr_pred_loss_weight,
            sgr_prediction_eval_episodes=args.sgr_pred_eval_episodes,
            sgr_phase2_arm=args.sgr_phase2_arm,
            sgr_phase2_signal=args.sgr_phase2_signal,
            sgr_gate_eval_override=args.sgr_gate_eval_override,
            sgr_dnc_write_salience=args.sgr_dnc_write_salience,
            sgr_workspace_write_salience=args.sgr_workspace_write_salience,
        )
        all_summaries.append(summary)

    print("\n===== Sweep summary (this process) =====")
    for s in all_summaries:
        print(s)

    # Per-process partial summary - when running one beta per process, each
    # writes its own file; merge them by hand (or a short follow-up script)
    # once all processes finish. Only the no-arg (full sequential) path
    # produces a complete combined summary in one file.
    summary_suffix = "_".join(f"b{b}".replace(".", "p") for b in betas_to_run)
    summary_path = os.path.join(LOG_DIR, f"gate1_sweep_summary_{summary_suffix}.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
        w.writeheader()
        for s in all_summaries:
            w.writerow(s)
    print(f"\nWrote summary to {summary_path}")
