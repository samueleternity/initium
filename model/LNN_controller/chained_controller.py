"""
file: LNN_controller/chained_controller.py -- v1

Generic composer: runs several controller wrappers back-to-back per DNC
timestep, each feeding the next (stage i's d_model must equal stage i+1's
input width). Works with ANY wrapper following the controller-wrapper
protocol in cfc_controller.py's docstring: MambaControllerWrapper,
Mamba2ControllerWrapper, Mamba3ControllerWrapper, CfCControllerWrapper, or
future ones. Example (Mamba backbone -> CfC head):

    ChainedControllerWrapper([
        MambaControllerWrapper(in_dim=in_dim, d_model=d, num_blocks=2),
        CfCControllerWrapper(in_dim=d, d_model=d, num_blocks=1),
    ])

State is a list with one entry per stage. Also usable as a SplitGraphDNC
combiner_wrapper (same protocol).
"""
from __future__ import annotations

import torch.nn as nn


def _in_features(stage) -> int:
    a = stage.in_adapter
    return a.in_features if isinstance(a, nn.Linear) else stage.d_model


class ChainedControllerWrapper(nn.Module):
    def __init__(self, stages):
        super().__init__()
        stages = list(stages)
        assert len(stages) >= 1, "ChainedControllerWrapper needs at least one stage"
        for prev, nxt in zip(stages[:-1], stages[1:]):
            assert _in_features(nxt) == prev.d_model, (
                f"stage width mismatch: previous d_model={prev.d_model}, next stage input={_in_features(nxt)}"
            )
        self.stages = nn.ModuleList(stages)
        self.d_model = stages[-1].d_model
        self.in_adapter = stages[0].in_adapter  # same object (not re-registered under a new path issue: shared reference)
        # plain python list (NOT ModuleList): the MoE blocks are already registered inside their stages
        self.moe_blocks = [b for s in stages if getattr(s, "moe_enabled", False) for b in s.moe_blocks]
        self.moe_enabled = len(self.moe_blocks) > 0

    def init_state(self, batch_size, device=None, dtype=None):
        return [s.init_state(batch_size, device=device, dtype=dtype) for s in self.stages]

    def forward(self, input, hx):
        if hx is None:
            hx = [None] * len(self.stages)
        x, new_hx = input, []
        for stage, state in zip(self.stages, hx):
            x, new_state = stage(x, state)
            new_hx.append(new_state)
        return x, new_hx