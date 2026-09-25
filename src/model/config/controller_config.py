"""
file: controller_config.py

Controller and memory-feature configuration. Each block is only meaningful for
its own controller / feature; defaults reproduce the baseline LSTM + dense
fixed-N run. Imported explicitly by core_training.py.
"""

# ---- Controller selection ---------------------------------------------------
# "lstm" reproduces Phase 2 exactly. Overridable via --controller.
CONTROLLER_TYPE = "lstm"

# Mamba-1 (paper defaults, Gu & Dao 2024, Sec 3.4)
MAMBA_D_STATE = 16
MAMBA_D_CONV = 4
MAMBA_EXPAND = 2

# Mamba-2 standalone per-timestep controller (--controller mamba2). NOT the same
# as SPLIT_GRAPH_MAMBA_VARIANT="mamba2" (whole-sequence backbone, no memory dependency).
MAMBA2_D_STATE = 64
MAMBA2_D_CONV = 4
MAMBA2_EXPAND = 2
MAMBA2_HEADDIM = 64
MAMBA2_NGROUPS = 1

# Mamba-3 (SISO). d_state=64 matches the Mamba-2 setting. If the interleaved run
# OOMs, drop MAMBA3_D_STATE to 32 first.
MAMBA3_D_STATE = 64
MAMBA3_EXPAND = 2
MAMBA3_HEADDIM = 64
MAMBA3_ROPE_FRACTION = 0.5

# CfC (LNN, ncps) -- CONTROLLER_TYPE == "cfc"
CFC_MODE = "default"  # "default" | "pure" | "no_gate"
CFC_BACKBONE_UNITS = 512
CFC_BACKBONE_LAYERS = 1
CFC_BACKBONE_DROPOUT = 0.0
CFC_ACTIVATION = "lecun_tanh"
CFC_MIXED_MEMORY = False  # True = CfC-mmRNN (adds an LSTM cell)
CFC_RESIDUAL = True
HYBRID_CFC_NUM_BLOCKS = (
    1  # CfC stage depth for "<mamba*>+cfc" (Mamba stage depth = num_hidden_layers)
)

# ---- Option 4: MoE in controller (mamba / mamba2 / mamba3 only) --------------
# num_experts >= 4 required by SwitchMoE; expert_dim None -> 3*hidden_size;
# alpha=0.01 is Switch Transformers' tuned value.
MOE_ENABLED = False
MOE_NUM_EXPERTS = 8
MOE_EXPERT_DIM: int | None = None
MOE_CAPACITY_FACTOR = 1.5
MOE_LOAD_BALANCE_ALPHA = 0.01
MOE_LOAD_BALANCE_ALPHA = 0.01
MOE_TOP_K = 1  # experts simultaneously active per token/source (Top-K routing)
SPLIT_GRAPH_COMBINER_CFC_MULTI_SOURCE_MOE = True  # combiner_variant="cfc" + moe: route
# [backbone_output, prev_read_vector] through
# per-source specialized experts (see
# cfc_controller.py / moe_layer.MultiSourceMoEBlock)

# ---- Option 5: split-compute-graph controller (off unless --split-graph) ------
SPLIT_GRAPH_ENABLED = False
SPLIT_GRAPH_MAMBA_VARIANT = "mamba1"  # "mamba1" | "mamba2" | "mamba3" | "cfc" | "<mamba*>+cfc"
SPLIT_GRAPH_NUM_BLOCKS = 2
SPLIT_GRAPH_MAMBA_HEADDIM = 64  # mamba2/3 only
SPLIT_GRAPH_COMBINE_READS = True  # False = built-in ablation
SPLIT_GRAPH_COMBINER_MODE = "linear"  # "linear" (default) | "controller"
SPLIT_GRAPH_COMBINER_VARIANT = "mamba1"
SPLIT_GRAPH_COMBINER_NUM_BLOCKS = 1

# ---- Option 1: link-matrix ablation/sparsification at fixed N -----------------
LINK_MATRIX_MODE = "dense"  # "dense" | "ablated" | "sparse_topk"
LINK_MATRIX_TOPK: int | None = None  # int, required only for "sparse_topk"
ISOLATE_LINK_ABLATION = False  # True: hold nr_cells fixed at MODEL_NR_CELLS for the
# whole run so only the link-matrix change is measured

# ---- Dynamic-N, macro-scale (usage-triggered nr_cells growth between episodes) ---
# Mutually exclusive with curriculum-indexed resize (the core overrides the
# curriculum's lesson_nr_cells to a constant when this is on).
DYNAMIC_N_MODE = False
DYNAMIC_N_FLOOR = 128  # starting nr_cells
DYNAMIC_N_CEILING = 512  # hard ceiling (sparse-link approx only validated to 512)
DYNAMIC_N_GROWTH_FACTOR = 2.0  # 128 -> 256 -> 512
DYNAMIC_N_USAGE_HIGH = 0.90  # a cell counts as "saturated" above this usage
DYNAMIC_N_TRIGGER_FRAC = 0.75  # growth when EMA of saturated fraction exceeds this
DYNAMIC_N_EMA_DECAY = 0.98  # "sustained, not a single-step spike" window
DYNAMIC_N_COOLDOWN_STEPS = 2000  # wait after a growth event before another
