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
    def __init__(self, stages, stage_kinds=None):
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
        # Names each stage (e.g. ["mamba", "cfc"] for "mamba+cfc") so the
        # training loop can log which stage a given ablation index refers
        # to. None only for the [f"stage{i}"...] fallback below when a
        # caller doesn't pass kinds.
        self.stage_kinds = list(stage_kinds) if stage_kinds is not None else [f"stage{i}" for i in range(len(self.stages))]

    def init_state(self, batch_size, device=None, dtype=None):
        return [s.init_state(batch_size, device=device, dtype=dtype) for s in self.stages]

    def forward(self, input, hx, skip_stages=None):
        """skip_stages: optional set/list of stage indices to bypass, for
        measuring each stage's contribution the same way pass_through_memory
        measures Memory's contribution (see evaluate_traversal's
        ablate_memory / SplitGraphDNC's combiner_skip_stages). Stage 0 still
        gets its in_adapter applied (so downstream stages see the right
        width) but skips its actual recurrent computation; every later index
        is skipped entirely (its d_model-in == d_model-out, so identity is
        exact)."""
        if hx is None:
            hx = [None] * len(self.stages)
        skip_stages = set(skip_stages) if skip_stages else set()
        x, new_hx = input, []
        for i, (stage, state) in enumerate(zip(self.stages, hx)):
            if i in skip_stages:
                if i == 0:
                    x = stage.in_adapter(x.squeeze(1)).unsqueeze(1)
                new_hx.append(state)
                continue
            x, new_state = stage(x, state)
            new_hx.append(new_state)
        return x, new_hx