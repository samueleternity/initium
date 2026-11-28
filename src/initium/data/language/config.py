"""Explicit, serializable configuration for the language/world-state run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class LossWeights:
    text: float = 1.0
    same_state: float = 0.1
    transition: float = 0.1
    logic: float = 0.1
    dialogue: float = 0.25

    def __post_init__(self):
        values = asdict(self)
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("loss weights must be finite and nonnegative")
        if not any(value > 0 for value in values.values()):
            raise ValueError("at least one objective loss weight must be positive")


@dataclass(frozen=True)
class ArchitectureConfig:
    """The requested split-graph Mamba1 + parallel dense workspace candidate."""

    split_graph: bool = True
    split_graph_variant: str = "mamba1"
    split_graph_num_blocks: int = 4
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    memory_type: str = "none"
    workspace_token_id_input: bool = True
    workspace_parallel_dense: bool = True
    workspace_block_index: int = 1
    workspace_num_slots: int = 128
    workspace_key_dim: int = 64
    workspace_scan_chunk: int = 32
    modulation: bool = True
    modulation_targets: tuple[str, ...] = ("dt",)
    modulation_blocks: tuple[int, ...] = (3,)
    modulation_source: str = "workspace"
    modulation_rank: int = 32
    modulation_alphas: dict[str, float] = field(default_factory=lambda: {
        "dt": 0.5, "a": 0.5, "B": 0.3, "C": 0.3, "D": 0.3,
    })
    modulation_lambda: float = 0.0
    modulation_temporal_lambda: float = 0.0
    modulation_lr_scale: float = 1.0
    modulation_alpha_warmup_steps: int = 3000
    neuromod_combine: tuple[str, ...] = (
        "plasticity", "gate", "consolidation", "responsiveness"
    )
    neuromod: str = "off"
    neuromod_groups: tuple[str, ...] = ("backbone", "moe", "workspace", "modulation", "task_head")
    neuromod_retention_control: bool = True
    neuromod_loss_outlier_gating: bool = True
    neuromod_consolidation_lambda: float = 0.1
    neuromod_range: float = 2.0
    neuromod_every: int = 50
    neuromod_slew: float = 0.05
    neuromod_seed: int = 0
    neuromod_gate_kappa: float = 2.0
    fp32: bool = True
    total_steps: int = 130_000
    checkpoint_every: int = 5_000
    seed: int = 0

    def __post_init__(self):
        if not self.split_graph or self.split_graph_variant != "mamba1":
            raise ValueError("the language model assembly requires split-graph Mamba1")
        if self.split_graph_num_blocks < 1 or self.workspace_block_index < 0:
            raise ValueError("split-graph block count must be positive and workspace block nonnegative")
        if self.workspace_parallel_dense and self.split_graph_num_blocks <= self.workspace_block_index + 1:
            raise ValueError("parallel workspace requires blocks after workspace_block_index")
        if min(self.workspace_num_slots, self.workspace_key_dim, self.workspace_scan_chunk) < 1:
            raise ValueError("workspace dimensions and scan chunk must be positive")
        if self.memory_type != "none":
            raise ValueError("the selected parallel dense workspace requires memory_type='none'")
        if not self.workspace_token_id_input:
            raise ValueError("language model assembly requires workspace token-ID input")
        if not self.fp32:
            raise ValueError("the standalone language trainer currently supports FP32 only")
        if self.modulation and not self.workspace_parallel_dense:
            raise ValueError("modulation requires the parallel dense workspace")
        if min(self.total_steps, self.checkpoint_every, self.neuromod_every) < 1:
            raise ValueError("training/checkpoint/neuromodulator intervals must be positive")
        if (not math.isfinite(self.neuromod_range) or self.neuromod_range < 1
                or not math.isfinite(self.neuromod_slew) or self.neuromod_slew < 0):
            raise ValueError("neuromodulator range must be >=1 and slew nonnegative")
        if self.neuromod_consolidation_lambda < 0:
            raise ValueError("consolidation lambda must be nonnegative")
        valid_arms = {"lr", "plasticity", "gate", "consolidation", "responsiveness", "random"}
        unknown = set(self.neuromod_combine) - valid_arms
        if unknown:
            raise ValueError(f"unknown neuromodulation arms: {sorted(unknown)}")
        if "random" in self.neuromod_combine and len(self.neuromod_combine) > 1:
            raise ValueError("random neuromodulation cannot be combined with signal-driven arms")
        effective_arms = set(self.neuromod_combine)
        if self.neuromod != "off":
            effective_arms.add(self.neuromod)
        if "random" in effective_arms and len(effective_arms) > 1:
            raise ValueError("random neuromodulation cannot be combined with signal-driven arms")
        if self.neuromod_retention_control and "plasticity" not in effective_arms:
            raise ValueError("retention control requires the plasticity arm")
        if "responsiveness" in effective_arms and not self.modulation:
            raise ValueError("neuromod responsiveness requires modulation")
        if min(self.mamba_d_state, self.mamba_d_conv, self.mamba_expand, self.modulation_rank) < 1:
            raise ValueError("Mamba state/convolution/expansion and modulation rank must be positive")
        if self.modulation_alpha_warmup_steps < 0:
            raise ValueError("modulation alpha warmup steps must be nonnegative")
        if self.modulation_source not in ("workspace", "backbone", "static"):
            raise ValueError("unsupported modulation source")
        if any(target not in ("dt", "a", "B", "C", "D") for target in self.modulation_targets):
            raise ValueError("unsupported modulation target")
        if self.modulation and (not self.modulation_blocks or min(self.modulation_blocks) <= self.workspace_block_index
                                or max(self.modulation_blocks) >= self.split_graph_num_blocks):
            raise ValueError("modulation blocks must be after workspace block and inside backbone")
        if self.neuromod not in {"off", "lr", "plasticity", "gate", "responsiveness", "consolidation", "random"}:
            raise ValueError("unsupported neuromodulator mode")
        if (not math.isfinite(self.neuromod_gate_kappa) or self.neuromod_gate_kappa <= 0
                or not math.isfinite(self.modulation_lr_scale) or self.modulation_lr_scale <= 0):
            raise ValueError("neuromodulator gate kappa and modulation LR scale must be positive")
        valid_groups = {"backbone", "moe", "workspace", "modulation", "task_head"}
        if not self.neuromod_groups or set(self.neuromod_groups) - valid_groups:
            raise ValueError("unknown neuromodulator parameter group")
        if any(not math.isfinite(value) or value < 0 for value in self.modulation_alphas.values()):
            raise ValueError("modulation alpha values must be nonnegative")
        if (any(not math.isfinite(value) for value in
                (self.modulation_lambda, self.modulation_temporal_lambda))
                or min(self.modulation_lambda, self.modulation_temporal_lambda) < 0):
            raise ValueError("modulation regularization weights must be nonnegative")

    @property
    def effective_neuromod_arms(self) -> tuple[str, ...]:
        arms = set(self.neuromod_combine)
        if self.neuromod != "off":
            arms.add(self.neuromod)
        return tuple(sorted(arms))


@dataclass(frozen=True)
class LanguageRunConfig:
    schema_version: int = 1
    tokenizer_path: str = "tokenizer/tokenizer.json"
    vocab_size: int = 32_000
    max_sequence_tokens: int = 2_048
    shard_tokens: int = 1_000_000
    seed: int = 0
    hidden_size: int = 512
    batch_size: int = 16
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 1_000
    evaluate_every: int = 1_000
    evaluation_batches: int = 8
    max_grad_norm: float = 1.0
    log_every: int = 20
    objective_sampling_weights: dict[str, float] = field(default_factory=lambda: {
        "text": 0.70, "same_state": 0.10, "transition": 0.10, "logic": 0.05,
        "dialogue": 0.05,
    })
    loss_weights: LossWeights = field(default_factory=LossWeights)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)

    def __post_init__(self):
        if self.vocab_size < 320 or self.max_sequence_tokens < 2 or self.shard_tokens < 2:
            raise ValueError("vocab_size, max_sequence_tokens, and shard_tokens are too small")
        if min(self.hidden_size, self.batch_size, self.evaluate_every,
               self.evaluation_batches, self.log_every) < 1:
            raise ValueError("model, batch, evaluation, and logging dimensions must be positive")
        if (not math.isfinite(self.learning_rate) or not math.isfinite(self.minimum_learning_rate)
                or not math.isfinite(self.weight_decay) or not math.isfinite(self.max_grad_norm)
                or min(self.learning_rate, self.minimum_learning_rate) <= 0
                or self.minimum_learning_rate > self.learning_rate or self.weight_decay < 0
                or self.max_grad_norm <= 0 or self.warmup_steps < 0):
            raise ValueError("learning rates must be positive and weight decay nonnegative")
        if any(not math.isfinite(value) or value < 0
               for value in self.objective_sampling_weights.values()):
            raise ValueError("objective sampling weights must be nonnegative")

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LanguageRunConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        weights = LossWeights(**raw.pop("loss_weights", {}))
        architecture_raw = raw.pop("architecture", {})
        for key in ("neuromod_combine", "neuromod_groups", "modulation_targets", "modulation_blocks"):
            if key in architecture_raw:
                architecture_raw[key] = tuple(architecture_raw[key])
        architecture = ArchitectureConfig(**architecture_raw)
        return cls(**raw, loss_weights=weights, architecture=architecture)


def core_training_args(config: ArchitectureConfig | None = None) -> list[str]:
    """Return the verified architecture flags for ``core_training``.

    Data/objective training is provided by this package's multimodal loop
    interfaces; the legacy ``core_training`` CLI only knows the episodic
    dataset contract, so this helper intentionally emits architecture flags
    only and does not invent a ``dataset-type``/track pairing.
    """
    cfg = config or ArchitectureConfig()
    args = [
        "--split-graph", "--split-graph-variant", cfg.split_graph_variant,
        "--split-graph-num-blocks", str(cfg.split_graph_num_blocks),
        "--memory-type", cfg.memory_type, "--workspace-token-id-input",
        "--workspace-parallel-dense", "--workspace-block-index", str(cfg.workspace_block_index),
        "--workspace-num-slots", str(cfg.workspace_num_slots), "--workspace-key-dim",
        str(cfg.workspace_key_dim), "--workspace-scan-chunk", str(cfg.workspace_scan_chunk),
        "--neuromod-combine", *cfg.neuromod_combine,
        "--neuromod-consolidation-lambda", str(cfg.neuromod_consolidation_lambda),
        "--neuromod-range", str(cfg.neuromod_range), "--neuromod-every",
        str(cfg.neuromod_every), "--neuromod-slew", str(cfg.neuromod_slew),
        "--total-steps", str(cfg.total_steps), "--checkpoint-every", str(cfg.checkpoint_every),
        "--seed", str(cfg.seed), "--fp32", "--no-segment-diagnostics",
    ]
    if cfg.modulation:
        args.append("--modulation")
    if cfg.neuromod_loss_outlier_gating:
        args.append("--neuromod-loss-outlier-gating")
    if not cfg.fp32:
        args.remove("--fp32")
    return args
