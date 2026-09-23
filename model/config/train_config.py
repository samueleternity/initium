"""
file: train_config.py

Core training-loop configuration (dataset-agnostic). Pure constants; imported
explicitly by core_training.py. Controller / memory-feature options live in
controller_config.py.
"""
import math

# ---- Loop / optimisation ----------------------------------------------------
BATCH_SIZE = 16              # bucketed, so padding waste stays low even >1
TOTAL_STEPS = 120000
LOG_EVERY = 100
EVAL_EVERY = 1000            # curriculum-advance eval cadence
LR = 3e-4
LR_MIN = 3e-5
LR_DECAY_STEPS = TOTAL_STEPS # cosine anneal length. Derived from TOTAL_STEPS here on
                             # purpose: --total-steps does NOT change it (pilot runs
                             # still anneal on the full schedule).
WARMUP_STEPS = 1000          # linear LR warmup 0 -> LR before the cosine anneal
USE_AMP = True
SEED = 0
AMP_INIT_SCALE = 128.0       # GradScaler start scale (default 65536 wasted ~14k steps
                             # halving its way down before training worked)

# ---- Model size -------------------------------------------------------------
MODEL_HIDDEN_SIZE = 512
MODEL_NR_CELLS = 256         # link matrix is O(N^2) in memory, so left unchanged
MODEL_CELL_SIZE = 192
MODEL_READ_HEADS = 8

# ---- Dataset selection (see data/dataset_registry.py) -----------------------
# Overridable via --dataset-type / --dataset-link.
DATASET_TYPE = "graph"       # "graph" | "text" | "audio" | "video" (only graph implemented)
DATASET_LINK = None          # None / "graph-traversal" -> built-in synthetic graph curriculum

# ---- KL -----------------------------------------------------------
BETAS_TO_SWEEP = [0.0]
KL_ANNEAL_STEPS = 8000       # ramp beta 0 -> target over this many steps
LESSON_KL_DIP_STEPS = 100
FREE_BITS = 0.02             # per-dimension KL floor (nats); 0.0 disables
LOG_DIR = "./logs"
OOD_EVAL_EPISODES = 200                # terminal OOD eval
OOD_EVAL_EPISODES_PERIODIC = 50        # lighter periodic OOD read every EVAL_EVERY steps

# ---- Dynamic beta (GECO-style dual ascent), task-accuracy constrained --------
# beta_target is reused as the controller's live value (checkpointed/logged as before).
BETA_MODE = "static"          # "static" or "dynamic"
BETA_CTRL_ACC_TARGET = None   # accuracy floor (0-1 frac); None -> dataset.advance_threshold
BETA_CTRL_LR = 2e-5           # max step per EVAL_EVERY cycle at constraint==1.0
BETA_CTRL_MIN = 0.0
BETA_CTRL_MAX = 2e-4          # hard ceiling -- 1e-3 is known to collapse
BETA_CTRL_EMA_DECAY = 0.9     # smooths the constraint signal across eval cycles

# ---- Learned prior ---------------------------------------------
# See stochastic_write_head_v2.py (update_prior_snapshot) for what these gate.
PRIOR_SNAPSHOT_EVERY = 2000
PRIOR_MIN_LOGVAR = -6.0       # floor on log(Sigma_g), prevents silent collapse
PRIOR_MAX_LOGVAR = math.log(6)       # symmetric ceiling

# ---- Checkpointing ----------------------------------------------------------
CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_EVERY = 2000       # periodic safety checkpoint, in addition to end-of-run