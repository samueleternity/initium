"""
file: mamba_backbone_parallel.py

Alternate Phase 3, Step 2, Option 5 ("split the compute graph, don't try to
parallelize the addressing itself") -- see Experiment-Roadmap.md.

This module implements ONLY the "(a) Mamba backbone" half of the Option 5
split: a stack of Mamba (v1) or Mamba-2 blocks driven over the WHOLE
sequence at once via each block's own native, library-provided forward()
(the fast, parallel-scan / SSD path), with NO per-timestep interleaving of
memory reads.

This is a deliberate, much simpler sibling to mamba_controller.py's
MambaControllerCell/Block/Wrapper (Alternative Phase 3, Step 1). That file
had to hand-roll a BPTT-safe manual step() specifically because the DNC's
per-timestep interleaving loop needed to call the controller one token at a
time while chaining (conv_state, ssm_state) through a Python loop under
full backprop, and mamba_ssm's own .step() mutates its state tensors
in-place (fatal for BPTT -- see that file's module docstring for the full
diagnosis).

Option 5's entire point is to remove that constraint for the backbone: since
this backbone's per-token computation has NO dependency on M_{t-1} (the
memory's state at the previous timestep) -- it only ever consumes the raw
input sequence X, never a read vector -- there is nothing forcing it into a
sequential Python loop. It can therefore call mamba_ssm's OWN whole-sequence
forward() directly, which already uses the fast parallel selective-scan
(Mamba-1) or the chunked, matmul-based SSD algorithm (Mamba-2, Dao & Gu
2024) -- no custom step function, no in-place-op workaround needed, because
nothing here is chained through a live memory-state Python loop.

The other half of the Option 5 split -- the "genuinely sequential, but much
smaller and cheaper, memory read/write/addressing step" that consumes this
backbone's parallel output as an input stream -- lives in
split_graph_dnc.py, not here. This file only produces H = backbone(X); it
has no notion of DNC, Memory, or read vectors at all, by design (see
Concept 6 / LB-9's caution about structural-vs-functional separation --
keeping this file's API surface structurally incapable of touching memory
state is a guarantee, not just a convention).

Supports two variants, both from the mamba-ssm package (per the roadmap's
"add Mamba-2 first if Option 5 needs it, otherwise Mamba-1 is already
implemented" instruction -- Option 5 benefits directly from Mamba-2's own
SSD chunked-scan algorithm, since that algorithm is specifically optimized
for exactly the "process the whole sequence at once" call pattern Option 5
newly unlocks for the backbone -- see the SSD paper, Sections 6 and 9.3,
already in corpus):

  - "mamba1": mamba_ssm.modules.mamba_simple.Mamba
  - "mamba2": mamba_ssm.modules.mamba2.Mamba2

Both are used exactly as their own package intends (whole-sequence
forward(hidden_states) call, batch_first (B, L, D) tensors) -- this file
does not reimplement or bypass anything of either class, unlike
mamba_controller.py's cell.

NOTE on Mamba2 kwargs: mamba_ssm's Mamba2 additionally requires `headdim`
(d_model must be divisible by it -- SSD paper's own convention is
head_dim in {64, 128}). This has no analogue in Mamba-1 and is therefore
a new, Mamba2-only constructor argument here, same pattern as
mamba_d_state/mamba_d_conv/mamba_expand having no LSTM analogue in
mamba_controller.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from mamba_ssm.modules.mamba_simple import Mamba as Mamba1
except ImportError as _mamba1_err:  # pragma: no cover - environment-dependent
    Mamba1 = None
    _MAMBA1_IMPORT_ERROR = (
        "mamba_backbone_parallel.py requires the `mamba-ssm` package for "
        f"variant='mamba1'. Original import error: {_mamba1_err}"
    )
else:
    _MAMBA1_IMPORT_ERROR = None

try:
    from mamba_ssm.modules.mamba2 import Mamba2
except ImportError as _mamba2_err:  # pragma: no cover - environment-dependent
    Mamba2 = None
    _MAMBA2_IMPORT_ERROR = (
        "mamba_backbone_parallel.py requires the `mamba-ssm` package "
        "(with Mamba-2 support -- mamba_ssm.modules.mamba2.Mamba2) for "
        f"variant='mamba2'. Original import error: {_mamba2_err}"
    )
else:
    _MAMBA2_IMPORT_ERROR = None

try:
    from mamba_ssm.modules.mamba3 import Mamba3
except ImportError as _mamba3_err:  # pragma: no cover - environment-dependent
    Mamba3 = None
    _MAMBA3_IMPORT_ERROR = (
        "mamba_backbone_parallel.py: variant='mamba3' needs mamba-ssm installed from GitHub main "
        f"(Mamba3 is not in the v2.3.1 wheel). Original import error: {_mamba3_err}"
    )
else:
    _MAMBA3_IMPORT_ERROR = None

import os as _os

# bf16 (the repo's own precision) when the GPU supports it; otherwise the fp32-under-fp16-autocast trick.
_MAMBA3_BF16_OK = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] >= 8
    and _os.environ.get("MAMBA3_FORCE_FP32", "0") != "1"
)


def _require_variant(variant: str) -> None:
    if variant == "mamba1" and Mamba1 is None:
        raise ImportError(_MAMBA1_IMPORT_ERROR)
    if variant == "mamba2" and Mamba2 is None:
        raise ImportError(_MAMBA2_IMPORT_ERROR)
    if variant == "mamba3" and Mamba3 is None:
        raise ImportError(_MAMBA3_IMPORT_ERROR)
    if variant not in ("mamba1", "mamba2", "mamba3"):
        raise ValueError(
            f"mamba_backbone_parallel: unknown variant {variant!r}, expected 'mamba1' or 'mamba2'."
        )


class _ParallelBlock(nn.Module):
    """Pre-norm residual wrapper around one whole-sequence Mamba block,
    mirroring mamba_ssm's own Block (Add -> LN -> Mixer) and
    mamba_controller.py's MambaControllerBlock, but calling the mixer's own
    forward() over the full (B, L, D) sequence at once instead of a
    per-timestep step(). No custom state threading of any kind -- the
    mixer manages its own internal scan/chunking entirely within this one
    call.
    """

    def __init__(
        self,
        d_model: int,
        variant: str,
        mamba_kwargs: dict,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        _require_variant(variant)
        self.norm = nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.variant = variant
        if variant == "mamba3":
            self.mixer = Mamba3(d_model=d_model, device=device, dtype=dtype, **mamba_kwargs)
        elif variant == "mamba1":
            self.mixer = Mamba1(d_model=d_model, device=device, dtype=dtype, **mamba_kwargs)
        else:
            self.mixer = Mamba2(d_model=d_model, device=device, dtype=dtype, **mamba_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D). mamba_ssm's own forward() handles the whole
        # sequence's parallel scan / SSD chunked-scan internally -- this is
        # the entire mechanism that makes Option 5 different from
        # mamba_controller.py: one call per block, not one call per DNC
        # timestep.
        #
        # FIX (chronic grad_norm nan under fp16 AMP, Mamba-2 variant only in
        # practice but applied to both for symmetry): same wrapper trick
        # already used by mamba_controller.py's MambaControllerCell.step()
        # and mamba2_controller.py's Mamba2ControllerCell.step() -- force
        # this call to run in real fp32, outside ambient fp16 autocast,
        # rather than reimplementing the mixer's internal math by hand the
        # way those cells do for their own per-timestep step(). Unlike those
        # cells, self.mixer here is the library's own whole-sequence
        # forward() (that's the entire point of the parallel backbone), so
        # we can't insert clamps mid-computation the way step() does --
        # instead we just deny it fp16 entirely. Mamba-2's SSD algorithm
        # computes exp() of a cumulative sum of (dt * A) over a whole chunk
        # (paper Section 6, Listing 1's segsum), which is far more prone to
        # fp16 underflow/overflow than Mamba-1's fused selective-scan
        # kernel -- this is the direct cause of the amp_scale-collapse-to-
        # 0.0 pattern observed in mamba2-backbone split-graph runs that
        # mamba1-backbone runs didn't show. Costs some speed relative to
        # running under fp16 (this block no longer benefits from tensor-core
        # fp16 matmuls), but matches every other Mamba SSM computation path
        # in this project, all of which already force fp32 for this reason.
        if self.variant == "mamba3" and _MAMBA3_BF16_OK:
            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=True):
                out = self.mixer(self.norm(x.float()).to(torch.bfloat16))
        else:
            with torch.autocast(device_type=x.device.type, enabled=False):
                out = self.mixer(self.norm(x.float()))
        return x + out.to(x.dtype)


class MambaBackboneParallel(nn.Module):
    """Stack of `num_blocks` whole-sequence Mamba blocks. Pure sequence-to-
    sequence map (B, L, in_dim) -> (B, L, d_model), with NO awareness of
    DNC, Memory, or read vectors -- see module docstring.

    variant: "mamba1" (mamba_ssm.modules.mamba_simple.Mamba, default
        Mamba-1 hyperparameters d_state=16/d_conv=4/expand=2, this
        project's original Alternate-Phase-3 choice) or "mamba2"
        (mamba_ssm.modules.mamba2.Mamba2, needs `headdim` in addition;
        default 64, matching the SSD paper's own convention of head_dim in
        {64, 128}).
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        num_blocks: int = 2,
        variant: str = "mamba1",
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,  # mamba2-only
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        _require_variant(variant)
        self.variant = variant
        self.d_model = d_model

        # Same role as MambaControllerWrapper's in_adapter: DNC's raw
        # per-timestep input (input_size) generally differs from
        # hidden_size/d_model. Unlike the interleaved wrapper, this is
        # applied to the WHOLE sequence at once (a single
        # (B, L, in_dim) -> (B, L, d_model) matmul), not per-timestep.
        self.in_adapter: nn.Module = (
            nn.Identity()
            if in_dim == d_model
            else nn.Linear(in_dim, d_model, device=device, dtype=dtype)
        )

        mamba_kwargs = dict(d_state=d_state, d_conv=d_conv, expand=expand)
        if variant == "mamba2":
            mamba_kwargs["headdim"] = headdim
        if variant == "mamba3":
            mamba_kwargs["headdim"] = headdim
            mamba_kwargs["chunk_size"] = (
                64 if _MAMBA3_BF16_OK else 32
            )  # repo: 64 bf16 / 32 otherwise

        self.blocks = nn.ModuleList(
            [
                _ParallelBlock(d_model, variant, mamba_kwargs, device=device, dtype=dtype)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, in_dim) -> (B, L, d_model). Called EXACTLY ONCE per
        # training step (not once per DNC timestep, unlike
        # MambaControllerWrapper) -- this single call is the entire
        # "backbone half" of the Option 5 split.
        h = self.in_adapter(x)
        for block in self.blocks:
            h = block(h)
        return h
