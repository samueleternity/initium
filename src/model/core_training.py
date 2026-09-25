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
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from memory_manipulation.nvrtc_compat import patch_prod_jiterator

patch_prod_jiterator()  # environment workaround -- see module docstring

from config.controller_config import (
    CFC_ACTIVATION,
    CFC_BACKBONE_DROPOUT,
    CFC_BACKBONE_LAYERS,
    CFC_BACKBONE_UNITS,
    CFC_MIXED_MEMORY,
    CFC_MODE,
    CFC_RESIDUAL,
    CONTROLLER_TYPE,
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
)
from config.train_config import (
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
from data.dataset_registry import get_dataset
from mamba_controller.mamba_controller import MambaDNC
from mamba_controller.split_graph_dnc import SplitGraphDNC
from memory_manipulation.dynamic_memory_resize import resize_memory
from memory_manipulation.dynamic_n_controller import DynamicNController
from memory_manipulation.link_matrix_ablation import patch_link_matrix
from memory_manipulation.stochastic_write_head_v2 import (
    get_prior_state,
    install_stochastic_write_heads,
    load_prior_state,
    pop_total_kl,
    update_all_prior_snapshots,
)
from MoE.moe_layer import pop_total_moe_aux_loss

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
    pipeline stage a NaN/Inf first appears at, since the LOG_EVERY-averaged
    training log only shows the aggregate, not the originating stage."""
    ok = torch.isfinite(tensor).all()
    if not ok:
        bad_frac = (~torch.isfinite(tensor)).float().mean().item()
        print(
            f"[NaN-TRACE] step {step}: non-finite in '{name}' ({bad_frac * 100:.2f}% of elements)"
        )
    return ok


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

    torch.save(
        {
            "step": step,
            "run_id": run_id,
            "beta_target": beta_target,
            "model_config": model_config,
            "rnn_state_dict": rnn.state_dict(),
            "output_proj_state_dict": output_proj.state_dict(),
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
        },
        path,
    )


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
    link_matrix_mode: str = LINK_MATRIX_MODE,  # Static Option 1
    link_matrix_topk: int | None = LINK_MATRIX_TOPK,  # Static Option 1
    isolate_link_ablation: bool = ISOLATE_LINK_ABLATION,  # Static Option 1
    dynamic_n_mode: bool = DYNAMIC_N_MODE,  # Dynamic-N (macro-scale)
    dynamic_n_floor: int = DYNAMIC_N_FLOOR,
    dynamic_n_ceiling: int = DYNAMIC_N_CEILING,
    dynamic_n_trigger_frac: float = DYNAMIC_N_TRIGGER_FRAC,
    dynamic_n_cooldown_steps: int = DYNAMIC_N_COOLDOWN_STEPS,
    moe_enabled: bool = MOE_ENABLED,
    moe_num_experts: int = MOE_NUM_EXPERTS,
    moe_expert_dim: int | None = MOE_EXPERT_DIM,
    moe_capacity_factor: float = MOE_CAPACITY_FACTOR,
    moe_load_balance_alpha: float = MOE_LOAD_BALANCE_ALPHA,
    moe_top_k: int = MOE_TOP_K,
    split_graph_combiner_cfc_multi_source_moe: bool = SPLIT_GRAPH_COMBINER_CFC_MULTI_SOURCE_MOE,
    split_graph_enabled: bool = SPLIT_GRAPH_ENABLED,
    split_graph_variant: str = SPLIT_GRAPH_MAMBA_VARIANT,
    split_graph_num_blocks: int = SPLIT_GRAPH_NUM_BLOCKS,
    split_graph_headdim: int = SPLIT_GRAPH_MAMBA_HEADDIM,
    split_graph_combine_reads: bool = SPLIT_GRAPH_COMBINE_READS,
    split_graph_combiner_mode: str = SPLIT_GRAPH_COMBINER_MODE,
    split_graph_combiner_variant: str = SPLIT_GRAPH_COMBINER_VARIANT,
    split_graph_combiner_num_blocks: int = SPLIT_GRAPH_COMBINER_NUM_BLOCKS,
    dataset_type: str = DATASET_TYPE,
    dataset_link: str | None = DATASET_LINK,
    test_dataset_link: str | None = None,
):
    # Deliberately does NOT touch LR_DECAY_STEPS - that's a separate
    # module-level constant, fixed at import time from the *original*
    # TOTAL_STEPS, and lr_at_step()/set_lr() below read it directly by
    # name, not through this parameter. A pilot run still anneals LR on
    # the full 120000-step schedule and simply stops early partway
    # through it, exactly as the --total-steps help text promises.
    dataset = get_dataset(dataset_type, dataset_link, test_dataset_link=test_dataset_link)
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
    resuming = resume_from is not None
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
                "total_loss",
                "digit_diversity",
                "kl_mean",
                "kl_max",
                "kl_min",
                "kl_std",
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

    # Functional-usage check (LB-9/Concept 6): same-cadence, same-distribution
    # companion to the ID/OOD log above, but with the model's own memory
    # path switched off (see evaluate_traversal's ablate_memory kwarg). A
    # standing per-run check, not a one-off sanity pass, per the roadmap's
    # explicit requirement -- catches rsDNC's silent bypass-path collapse
    # (still trains, still runs, loss curves look fine) that a KL- or
    # loss-only view can't see.
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
        + _arch_desc(
            controller,
            split_graph_enabled,
            split_graph_variant,
            split_graph_num_blocks,
            split_graph_combiner_mode,
            split_graph_combiner_variant,
            split_graph_combiner_num_blocks,
            split_graph_combine_reads,
        )
        + (f" | moe={moe_num_experts}e" if moe_enabled else "")
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
            moe_enabled=moe_enabled,  # MambaDNC raises if True (not wired for cfc)
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
            moe_enabled=moe_enabled,  # MambaDNC raises if True (not wired for hybrids)
            moe_top_k=moe_top_k,
        )

    if split_graph_enabled:
        if moe_enabled:
            print(
                f"[{run_id}] moe_enabled=True with split_graph_enabled=True: MoE is now "
                f"wired into both the parallel backbone and the sequential combiner (see "
                f"split_graph_dnc.py). A combiner_variant='cfc' additionally routes its two "
                f"natural input sources (backbone output, previous read vector) through "
                f"per-source specialized experts when "
                f"split_graph_combiner_cfc_multi_source_moe={split_graph_combiner_cfc_multi_source_moe} "
                f"(see moe_layer.MultiSourceMoEBlock)."
            )

        # v10: variant-matched hyperparameter defaults instead of always
        # reusing the Mamba-1 constants here -- harmless previously (any
        # d_state/headdim is accepted), but not what the SSD paper's own
        # Mamba-2 defaults are.
        _sg_d_state, _sg_d_conv, _sg_expand = (
            (MAMBA3_D_STATE, MAMBA_D_CONV, MAMBA3_EXPAND)
            if split_graph_variant.startswith("mamba3")
            else (MAMBA2_D_STATE, MAMBA2_D_CONV, MAMBA2_EXPAND)
            if split_graph_variant.startswith("mamba2")
            else (MAMBA_D_STATE, MAMBA_D_CONV, MAMBA_EXPAND)
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
            moe_num_experts=moe_num_experts,
            moe_expert_dim=moe_expert_dim,
            moe_top_k=moe_top_k,
            moe_capacity_factor=moe_capacity_factor,
            moe_load_balance_alpha=moe_load_balance_alpha,
            moe_cfc_multi_source=split_graph_combiner_cfc_multi_source_moe,
            device=device,
        ).to(device)
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
            **mamba_kwargs,
        ).to(device)

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
    }

    if link_matrix_mode != "dense":
        patch_link_matrix(rnn, mode=link_matrix_mode, topk=link_matrix_topk)
        topk_note = f" topk={link_matrix_topk}" if link_matrix_mode == "sparse_topk" else ""
        print(
            f"[{run_id}] link_matrix_mode={link_matrix_mode}{topk_note} -- "
            f"Static Option 1 applied to rnn.memories."
        )

    output_proj = nn.Linear(INPUT_DIM, TRIPLE_DIM).to(device)
    dataset.set_output_proj(output_proj)

    # Install the stochastic write head(s) BEFORE building the
    # optimizer, so their parameters (mu_transform, logvar_transform) are
    # included in optimizer.parameters(). mu_transform is initialized from
    # the original deterministic write_vector_transform's weights, so at
    # step 0 the sampled mean exactly matches Phase 0's write vector.
    # sample=True for all beta in BETAS_TO_SWEEP now that beta=0 is dropped
    # from the sweep (see header note) -- beta_target is always > 0.0 here.
    stochastic_heads = install_stochastic_write_heads(
        rnn, device=device, sample=(beta_target > 0.0)
    )

    optimizer = torch.optim.Adam(list(rnn.parameters()) + list(output_proj.parameters()), lr=LR)
    # Option 3 (Concept 24/LB-17): multi-axis efficiency accounting.
    # param_count is fixed for the life of this run -- static Option 2's
    # nr_cells resize never changes it (every Memory sublayer is sized by
    # cell_size/read_heads/input_size, never nr_cells, per the Option-2
    # analysis) -- so this is a one-time computation, not per-step.
    param_count = sum(p.numel() for p in rnn.parameters()) + sum(
        p.numel() for p in output_proj.parameters()
    )
    amp_enabled = USE_AMP and device.type == "cuda"
    kl_on = stochastic_heads[0].sample  # False when beta==0 -> no KL/prior terms in console
    print(
        f"[{run_id}] params {param_count} | kl_terms={'on' if kl_on else 'off'} | amp={'on' if amp_enabled else 'off'}"
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
    if resuming:
        ckpt = load_checkpoint_for_resume(resume_from, device)
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
        resize_memory(rnn, ckpt_nr_cells, device=device, optimizer=optimizer)

        rnn.load_state_dict(ckpt["rnn_state_dict"])
        output_proj.load_state_dict(ckpt["output_proj_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # v3 fix: restore GradScaler state (scale, growth tracker) instead
        # of leaving `scaler` at its freshly-constructed init_scale -- see
        # header note. `.get(...)` instead of ["..."] so a pre-v3 checkpoint
        # (saved before this key existed) still loads, just without this fix.
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        else:
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
        if dynamic_n_ctrl is not None:
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
        if "prior_state" in ckpt:
            load_prior_state(stochastic_heads, ckpt["prior_state"])
        else:
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
        if "ood_rng_state" in ckpt:
            ood_rng.setstate(ckpt["ood_rng_state"])
        else:
            print(
                f"[{run_id}] WARNING: checkpoint predates ood_rng-state "
                f"saving (v6 fix) -- OOD sampling restarting from its "
                f"fresh, seed-derived state instead of continuing the "
                f"pre-resume leg's exact walk sequence."
            )
        restore_rng_state(ckpt["rng_state"])
        step = ckpt["step"]
        if ckpt["beta_target"] != beta_target:
            anneal_start_step = step  # Phase 1: restart the ramp here
            print(
                f"[{run_id}] beta_target changed on resume "
                f"({ckpt['beta_target']} -> {beta_target}); KL anneal "
                f"restarted from step {anneal_start_step}, ramping over "
                f"the next {KL_ANNEAL_STEPS} steps."
            )
        print(
            f"[{run_id}] Resumed from {resume_from} at step {step} "
            f"(lesson {curriculum.lesson + 1}/{len(curriculum.table)}); "
            f"continuing to total_steps={total_steps}"
        )
        if step >= total_steps:
            print(
                f"[{run_id}] Checkpoint step {step} already >= total_steps "
                f"{total_steps} -- nothing to do. Raise --total-steps if you "
                f"want to extend further."
            )

    print(
        f"\n=== [{run_id}] Training (beta_target={beta_target}, beta_mode={beta_mode}) "
        f"{'[resumed]' if resuming else ''} ==="
    )
    rnn.train()
    set_lr(step)  # Phase 1 FIX: ensure correct clamped LR from the very first
    # iteration -- otherwise a resumed run's first step would briefly use
    # whatever LR was saved inside optimizer_state_dict at checkpoint time
    # (potentially still reflecting the old buggy drifted value).
    running_task_loss, running_kl_loss, running_div, running_grad_norm = 0.0, 0.0, 0, 0.0

    # Dynamic-beta controller state (no-op / unused when beta_mode == "static").
    # Initialized post-resume so a resumed dynamic run starts its health check
    # from the just-restored scaler scale rather than a fresh-process default.
    beta_constraint_ema = 0.0
    scale_at_last_eval = scaler.get_scale()
    grad_was_finite_since_eval = True

    t0 = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # Log addition #5: cumulative wall-clock for this run(). Starts fresh on
    # a resumed leg too (i.e. measures THIS process's elapsed time, not
    # elapsed time since the original run began across all resumed legs --
    # legs already have their own timestamped log rows if that's needed).
    t_run_start = time.time()

    while step < total_steps:
        input_seq, target_digits, answer_mask = dataset.sample_batch(curriculum, BATCH_SIZE)

        input_seq = input_seq.to(device, non_blocking=True)
        target_digits = target_digits.to(device, non_blocking=True)
        answer_mask = answer_mask.to(device, non_blocking=True)
        _first_nonfinite_report(input_seq, "input_seq", step)
        hidden = (None, None, None)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            output, hidden = rnn(input_seq, hidden, reset_experience=True)
            _first_nonfinite_report(output, "rnn_output", step)
            output = output.transpose(0, 1).contiguous()  # (B, T, 92)
            output = output_proj(output)  # (B, T, 90)
            task_loss = dataset.loss(output, target_digits, answer_mask)

        _first_nonfinite_report(task_loss, "task_loss", step)
        kl_loss, kl_diag = pop_total_kl(stochastic_heads, free_bits=FREE_BITS)  # Phase 1
        _first_nonfinite_report(kl_loss, "kl_loss", step)
        moe_aux_loss, moe_diag = (
            pop_total_moe_aux_loss(rnn.moe_layers)
            if moe_enabled
            else (torch.zeros((), device=device), {})
        )  # v8 (Option 4): Switch-style load-balancing loss, already scaled
        # by moe_load_balance_alpha inside SwitchMoE.pop_aux_loss() -- no
        # extra weighting applied here, unlike beta_eff*kl_loss below.
        global_ramp = min(1.0, (step - anneal_start_step) / max(1, KL_ANNEAL_STEPS))
        lesson_dip_ramp = min(1.0, (step - lesson_dip_start_step) / max(1, LESSON_KL_DIP_STEPS))
        beta_eff = beta_target * global_ramp * lesson_dip_ramp
        loss = (
            task_loss + beta_eff * kl_loss + moe_aux_loss
        )  # Phase 1 + Option 4: L = L_task + beta*L_KL + L_moe_aux
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # Log addition #2: capture the pre-clip gradient norm instead of
        # discarding clip_grad_norm_'s return value. Gives a β=0 baseline
        # gradient-scale trajectory to compare a future β>0 switch-in
        # against, useful for telling a genuine instability apart from the
        # expected one-off "shock" when sampling noise is switched on.
        grad_norm = torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=10.0)
        if not math.isfinite(float(grad_norm)):
            grad_was_finite_since_eval = False
            # clip_grad_norm_ computes clip_coef = max_norm/(total_norm+eps)
            # and multiplies EVERY gradient by it. A NaN total_norm therefore
            # NaNs out every gradient in the model, including ones that were
            # finite. GradScaler's found_inf was recorded during unscale_,
            # BEFORE this clip, so it can be clean while the post-clip grads
            # are all NaN -- scaler.step() then applies them and the weights
            # are permanently poisoned, which is exactly the one-way collapse
            # at step 23700. Zero the grads and skip the step instead: this
            # costs one wasted batch and is fully recoverable.
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
        else:
            scaler.step(optimizer)
            scaler.update()
        # Log addition #3: AMP loss-scale value. A collapsing/repeatedly
        # halved scale is the standard AMP symptom of inf/NaN gradients --
        # the exp()/logvar path in the stochastic write head is the most
        # numerically fragile part of this pipeline, so a clean β=0 baseline
        # of scaler behavior is useful context once β>0 introduces that path.
        amp_scale = scaler.get_scale()
        running_task_loss += task_loss.item()
        running_kl_loss += kl_loss.item()
        running_div += dataset.diversity(output.detach(), target_digits, answer_mask)
        running_grad_norm += float(grad_norm)
        step += 1
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

        if step % LOG_EVERY == 0:
            elapsed = time.time() - t0
            t0 = time.time()
            total_elapsed = time.time() - t_run_start  # log addition #5
            avg_task = running_task_loss / LOG_EVERY
            avg_kl = running_kl_loss / LOG_EVERY
            avg_div = running_div / LOG_EVERY
            avg_grad_norm = running_grad_norm / LOG_EVERY
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
                f"{LOG_EVERY / elapsed:.2f} steps/s",
                f"LR {current_lr:.6f}",
                f"grad_norm {avg_grad_norm:.4f}",
            ]
            if amp_enabled:
                parts.append(f"amp_scale {amp_scale:.1f}")
            parts.append(f"gpu_mem_peak_mb {gpu_mem_peak_mb:.1f}")
            if moe_enabled:
                parts += [
                    f"moe_aux {float(moe_aux_loss.detach()):.4f}",
                    f"moe_cv_load {moe_diag.get('moe_cv_load', 0.0):.4f}",
                    f"moe_max_load_frac {moe_diag.get('moe_max_load_frac', 0.0):.4f}",
                ]
            if dyn_n_str:
                parts.append(dyn_n_str.strip(" |"))
            print(" | ".join(parts))

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
                ]
            )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)  # so next window's peak isn't cumulative
            log_file.flush()
            running_task_loss, running_kl_loss, running_div, running_grad_norm = 0.0, 0.0, 0, 0.0

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
            _, id_triple_acc, id_perfect_frac = curriculum.maybe_advance(
                rnn, device, step=step, optimizer=optimizer
            )

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
            ood_triple_acc, ood_perfect_frac, ood_hop_breakdown = dataset.evaluate_ood(
                rnn,
                device,
                num_episodes=OOD_EVAL_EPISODES_PERIODIC,
                rng=ood_rng,
                field_log=ood_field_log,
            )

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
            ablated_triple_acc, ablated_perfect_frac = dataset.evaluate_id_ablated(
                rnn, device, curriculum, pre_advance_lesson
            )

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
                f"{ablated_triple_acc:.2f}% | dependency {id_triple_acc - ablated_triple_acc:.2f}"
            )

            if combiner_stage_kinds is not None:
                # Same pattern as the memory-dependency check above, but
                # ablating one combiner stage (e.g. the CfC block in a
                # "mamba+cfc" combiner) at a time, so each stage's actual
                # contribution to accuracy is measured directly instead of
                # inferred from loss curves / plateau shape.
                for stage_idx, stage_kind in enumerate(combiner_stage_kinds):
                    stage_triple_acc, stage_perfect_frac = (
                        dataset.evaluate_id_combiner_stage_ablated(
                            rnn, device, curriculum, pre_advance_lesson, skip_stages={stage_idx}
                        )
                    )
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
            )
            latest_path = os.path.join(CHECKPOINT_DIR, f"{run_id}_latest.pt")
            shutil.copyfile(ckpt_path, latest_path)

    print(f"\n[{run_id}] Training complete. Final evaluation on training-distribution lesson:")
    _, id_triple_acc, id_perfect_frac = curriculum.maybe_advance(
        rnn, device, step=step, optimizer=optimizer
    )

    print(f"\n[{run_id}] Generalization test:")

    ood_triple_acc, ood_perfect_frac, ood_hop_breakdown = dataset.evaluate_ood(
        rnn,
        device,
        num_episodes=OOD_EVAL_EPISODES,
        rng=ood_rng,
        verbose_n=10,
    )

    ood_offset_triple = id_triple_acc - ood_triple_acc
    ood_offset_perfect = id_perfect_frac - ood_perfect_frac

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
    print(f"\n[{run_id}] SUMMARY: {summary}")

    log_writer.writerow([])
    log_writer.writerow(["SUMMARY"] + list(summary.keys()))
    log_writer.writerow([""] + list(summary.values()))
    log_file.close()
    ood_log_file.close()
    lesson_log_file.close()
    prior_log_file.close()
    field_log_file.close()  # v16
    combiner_stage_log_file.close()
    modality_log_file.close()
    if dynamic_n_ctrl is not None:
        dynamic_n_log_file.close()

    return summary


# ==========================================
# 11. GATE
# ==========================================
if __name__ == "__main__":
    import argparse
    import sys

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
        help="path to a checkpoint .pt to resume from (requires --beta / positional beta "
        "to also be given, so run_id/beta_target can be matched against the checkpoint). "
        "Training continues from the checkpoint's step up to TOTAL_STEPS.",
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
        help="v11: 'linear' (default) is the original, already-efficient "
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
        "--dataset-type",
        type=str,
        default=DATASET_TYPE,
        choices=["graph", "text", "audio", "video", "multimodal"],
        help="Dataset plugged into the core loop (data/dataset_registry.py). "
        "Only 'graph' (graph-traversal) is implemented so far.",
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
    args = parser.parse_args()

    if args.resume is not None and args.beta is None:
        raise SystemExit(
            "--resume requires the beta positional arg too, e.g.:\n"
            f"  python3 {sys.argv[0]} 0.02 --resume checkpoints/beta_0p02_latest.pt"
        )

    if args.link_matrix_mode == "sparse_topk" and args.link_matrix_topk is None:
        raise SystemExit("--link-matrix-mode=sparse_topk requires --link-matrix-topk")

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
    for beta in betas_to_run:
        # These are made to avoid checkpoint collisions between different types of runs.
        run_id = f"beta_{beta}_seed{args.seed}_learnedprior".replace(".", "p")
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
        if args.split_graph:
            run_id = f"{run_id}_splitgraph_{args.split_graph_variant.replace('+', '')}"
            if args.split_graph_combiner_mode == "controller":
                run_id = f"{run_id}_combctrl{args.split_graph_combiner_variant.replace('+', '')}"
        if args.dataset_type != "graph":
            run_id = f"{run_id}_ds{args.dataset_type}"
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
            link_matrix_mode=args.link_matrix_mode,
            link_matrix_topk=args.link_matrix_topk,
            isolate_link_ablation=args.isolate_link_ablation,
            dynamic_n_mode=args.dynamic_n_mode,
            dynamic_n_floor=args.dynamic_n_floor,
            dynamic_n_ceiling=args.dynamic_n_ceiling,
            dynamic_n_trigger_frac=args.dynamic_n_trigger_frac,
            dynamic_n_cooldown_steps=args.dynamic_n_cooldown_steps,
            moe_enabled=args.moe,
            moe_num_experts=args.moe_num_experts,
            moe_expert_dim=args.moe_expert_dim,
            moe_capacity_factor=args.moe_capacity_factor,
            moe_load_balance_alpha=args.moe_load_balance_alpha,
            moe_top_k=args.moe_top_k,
            split_graph_enabled=args.split_graph,
            split_graph_variant=args.split_graph_variant,
            split_graph_num_blocks=args.split_graph_num_blocks,
            split_graph_headdim=args.split_graph_headdim,
            split_graph_combine_reads=not args.split_graph_no_combine,
            split_graph_combiner_mode=args.split_graph_combiner_mode,
            split_graph_combiner_variant=args.split_graph_combiner_variant,
            split_graph_combiner_num_blocks=args.split_graph_combiner_num_blocks,
            dataset_type=args.dataset_type,
            dataset_link=args.dataset_link,
            test_dataset_link=args.test_dataset_link,
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
