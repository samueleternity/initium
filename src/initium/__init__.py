"""Initium's stable architecture API.

Import mechanism-layer components from this package, for example::

    from initium import SplitGraphDNC, MambaDNC, resize_memory

Training and inference entry points remain in their existing modules
(``initium.core_training`` and ``initium.inference``). Dataset encoders,
``DigitCodec``, and task/curriculum implementations under ``initium.data``
are intentionally internal: they describe replaceable representations, not
the architecture contract. Exports are lazy so importing the package does
not import a training CLI or optional model dependencies unnecessarily.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    # DNC architectures
    "SplitGraphDNC",
    "MambaDNC",
    # Controller mechanisms and compositions
    "MambaControllerWrapper",
    "Mamba2ControllerWrapper",
    "Mamba3ControllerWrapper",
    "CfCControllerWrapper",
    "ChainedControllerWrapper",
    "build_hybrid_controller",
    # Routing and stochastic memory writes
    "StochasticWriteHead",
    "SwitchMoE",
    "MoEBlock",
    # Memory and checkpoint mechanics
    "resize_memory",
    "load_checkpoint",
    "describe_checkpoint",
    "load_model",
    "LoadedModel",
]

_EXPORTS = {
    "SplitGraphDNC": "initium.mamba_controller.split_graph_dnc",
    "MambaDNC": "initium.mamba_controller.mamba_controller",
    "MambaControllerWrapper": "initium.mamba_controller.mamba_controller",
    "Mamba2ControllerWrapper": "initium.mamba_controller.mamba2_controller",
    "Mamba3ControllerWrapper": "initium.mamba_controller.mamba3_controller",
    "CfCControllerWrapper": "initium.LNN_controller.cfc_controller",
    "ChainedControllerWrapper": "initium.LNN_controller.chained_controller",
    "build_hybrid_controller": "initium.LNN_controller.hybrid_controller",
    "StochasticWriteHead": "initium.memory_manipulation.stochastic_write_head_v2",
    "SwitchMoE": "initium.MoE.moe_layer",
    "MoEBlock": "initium.MoE.moe_layer",
    "resize_memory": "initium.memory_manipulation.dynamic_memory_resize",
    "load_checkpoint": "initium.inference.checkpoint_io",
    "describe_checkpoint": "initium.inference.checkpoint_io",
    "load_model": "initium.inference.model_loader",
    "LoadedModel": "initium.inference.model_loader",
}

if TYPE_CHECKING:
    from initium.LNN_controller.cfc_controller import CfCControllerWrapper as CfCControllerWrapper
    from initium.LNN_controller.chained_controller import ChainedControllerWrapper as ChainedControllerWrapper
    from initium.LNN_controller.hybrid_controller import build_hybrid_controller as build_hybrid_controller
    from initium.MoE.moe_layer import MoEBlock as MoEBlock
    from initium.MoE.moe_layer import SwitchMoE as SwitchMoE
    from initium.inference.checkpoint_io import describe_checkpoint as describe_checkpoint
    from initium.inference.checkpoint_io import load_checkpoint as load_checkpoint
    from initium.inference.model_loader import LoadedModel as LoadedModel
    from initium.inference.model_loader import load_model as load_model
    from initium.mamba_controller.mamba2_controller import Mamba2ControllerWrapper as Mamba2ControllerWrapper
    from initium.mamba_controller.mamba3_controller import Mamba3ControllerWrapper as Mamba3ControllerWrapper
    from initium.mamba_controller.mamba_controller import MambaControllerWrapper as MambaControllerWrapper
    from initium.mamba_controller.mamba_controller import MambaDNC as MambaDNC
    from initium.mamba_controller.split_graph_dnc import SplitGraphDNC as SplitGraphDNC
    from initium.memory_manipulation.dynamic_memory_resize import resize_memory as resize_memory
    from initium.memory_manipulation.stochastic_write_head_v2 import StochasticWriteHead as StochasticWriteHead


def __getattr__(name: str) -> Any:
    """Resolve only names explicitly listed in the stable public API."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
