"""
file: inference/capabilities.py

Decides, FROM THE CHECKPOINT ITSELF, which dataset types a model can run, and
refuses to continue when the requested type/dimensions don't match.

Sources of truth (in order):
  1. model_config["supported_dataset_types"] (written by core_training v19+).
     May contain "*" = any type whose dims match the weights.
  2. Legacy checkpoints (no such key): infer from the tensor dims - every
     implemented task type whose (input_dim, output_dim) equals the model's.
     Dims that match nothing -> supported list is empty -> run refuses.
Independently of (1)/(2), the task's dims must ALWAYS equal the model's dims.
"""

from __future__ import annotations

from dataclasses import dataclass

from initium.inference.inference_config import WILDCARD_TYPE
from initium.inference.tasks.task_registry import IMPLEMENTED_TYPES, canonical_type, get_task_class


class IncompatibleModelError(Exception):
    pass


@dataclass
class ModelCapabilities:
    supported_types: list[str]
    input_dim: int
    output_dim: int
    source: str  # "recorded" | "inferred-legacy" | "unknown"
    controller_type: str

    def supports(self, canon_type: str) -> bool:
        return WILDCARD_TYPE in self.supported_types or canon_type in self.supported_types

    def describe(self) -> str:
        return (
            f"supported types={self.supported_types} (source: {self.source}) | "
            f"input_dim={self.input_dim} output_dim={self.output_dim} | "
            f"controller={self.controller_type}"
        )


def detect_capabilities(ckpt: dict) -> ModelCapabilities:
    cfg = ckpt["model_config"]
    out_dim, proj_in = ckpt["output_proj_state_dict"]["weight"].shape
    input_dim = int(cfg.get("input_dim", cfg.get("input_size", proj_in)))
    output_dim = int(out_dim)

    recorded = cfg.get("supported_dataset_types")
    if recorded:
        types = [WILDCARD_TYPE if t == WILDCARD_TYPE else canonical_type(t) for t in recorded]
        source = "recorded"
    else:
        types = []
        for t in IMPLEMENTED_TYPES:
            cls = get_task_class(t)
            if (cls.input_dim, cls.output_dim) == (input_dim, output_dim):
                types.append(t)
        source = "inferred-legacy" if types else "unknown"

    return ModelCapabilities(
        types, input_dim, output_dim, source, cfg.get("controller_type", "lstm")
    )


def require_type_supported(caps: ModelCapabilities, dataset_type: str, ckpt_path: str) -> str:
    canon = canonical_type(dataset_type)
    if not caps.supports(canon):
        raise IncompatibleModelError(
            f"checkpoint '{ckpt_path}' cannot run dataset type '{canon}'. "
            f"It supports {caps.supported_types} ({caps.source}). Not continuing."
        )
    return canon


def require_dims_match(caps: ModelCapabilities, task, ckpt_path: str) -> None:
    if (task.input_dim, task.output_dim) != (caps.input_dim, caps.output_dim):
        raise IncompatibleModelError(
            f"task '{task.name}' produces input_dim={task.input_dim}/output_dim={task.output_dim} "
            f"but checkpoint '{ckpt_path}' expects input_dim={caps.input_dim}/"
            f"output_dim={caps.output_dim}. Not continuing."
        )
