"""
mamba_chunk_controller.py -- v1 (new file)

Alternate Phase 3, Step 2 (see Experiment-Roadmap.md, "Attempt to modify DNC
in a way that it can in one way or another simulate SSMs parallelism...
Option 2: Chunked/blockwise approximate parallelism").

This file adds a CHUNK-PARALLEL forward path for the Mamba-1 controller
defined in `mamba_controller.py` (v7, Alternate Phase 3 Step 1). It does
NOT modify mamba_controller.py in any way -- every class here either
subclasses or wraps a class from that file, reusing its parameter
construction, initialization, and single-step `.step()` path completely
unmodified. mamba_controller.py's own smoke-test-verified guarantee ("byte-
identical to `Mamba.step()`'s non-fast-path branch, out-of-place") is
therefore inherited unchanged; nothing here can silently regress it.

--------------------------------------------------------------------------
What "chunk-parallel" means here, precisely
--------------------------------------------------------------------------
`MambaControllerCell.step()` (mamba_controller.py) computes ONE S6 update
per Python call: `state -> state'`, `x_t -> y_t`. Called C times in a
Python for-loop (which is what the plain, sequential DNC forward loop
already does, once per real timestep) has O(C) *sequential* Python-level
and autograd-graph-node depth.

`MambaChunkControllerCell.forward_chunk()` (this file) computes the exact
same S6 recurrence for a whole block of C consecutive timesteps in ONE
call, using real batched tensor ops for every part of the computation that
doesn't have a genuine cross-timestep data dependency (in_proj, the causal
conv -- via a single grouped `F.conv1d` over the whole chunk instead of a
per-step windowed sum, x_proj, dt_proj/softplus, the (Δ,A,B,C) construction)
and `dnc_parallel_scan.py`'s O(log C)-depth associative scan for the one
part that IS a genuine cross-timestep recurrence (the SSM state update
h_t = dA_t*h_{t-1} + dB_t*x_t). This is the literal, direct translation of
Mamba's own "hardware-aware parallel scan" idea (Gu & Dao 2024, Section
3.3: "we overcome this with ... a parallel scan") into the one place this
codebase's existing Mamba wiring had NOT yet applied it -- Step 1
(mamba_controller.py) deliberately used the portable, always-correct
per-step formula everywhere, specifically so Step 2 could later replace
its C-times-sequential outer loop with a real scan without having to first
untangle any BPTT-unsafe in-place state mutation (see mamba_controller.py's
own "Why not just call Mamba.step()?" section -- the same hazard that
motivated writing `.step()` out-of-place in the first place is exactly why
the chunk version below is ALSO written entirely out-of-place).

`forward_chunk()` is mathematically exact (not an approximation) relative
to calling `.step()` C times in a row with the same inputs and the same
starting `(conv_state, ssm_state)` -- see "Exactness" below. The
APPROXIMATION in this codebase's chunked design lives entirely one layer up,
in `chunked_parallel_dnc.py` (the DNC-level read-vector freezing), not here.

--------------------------------------------------------------------------
Exactness of forward_chunk() vs. step() x C -- why, and what was checked
--------------------------------------------------------------------------
Every one of the four building blocks below is provably identical to
stacking C `.step()` calls, not merely close:

  1. in_proj / x_proj / dt_proj are `nn.Linear`, which is already defined
     to act independently and identically on every element along any
     leading batch-like dimension (that is what "Linear" means) -- calling
     it once on a (B, C, *) tensor is definitionally the same as calling it
     C times on (B, *) slices, not an approximation of doing so.
  2. The causal depthwise conv: `.step()` keeps a rolling window
     `conv_state` of the last `d_conv` raw inputs and computes
     `sum(window * conv_weight)`, one dot product per step.
     `forward_chunk()` builds ONE extended window
     `[conv_state[:,:,1:], x_1, ..., x_C]` (length `d_conv-1+C`) and runs a
     single valid-mode (`padding=0`) grouped `F.conv1d` over it. Standard
     1D convolution/correlation is itself defined as a sliding dot product
     over exactly this kind of window -- the `t`-th output of that single
     conv1d call is, by the operation's own definition, the same dot
     product `.step()` would have computed for chunk-position `t`. This is
     not an approximation of the rolling-window recurrence; it IS the
     rolling-window recurrence, vectorized.
  3. The (Δ, A, B, C) construction (softplus, `-exp(A_log)`, the two
     `einsum`s forming `dA_t`/`dB_t`) has no cross-timestep dependency at
     all in either version -- every timestep's (Δ_t, A, B_t, C_t) depends
     only on that timestep's own conv output, so batching this over the
     chunk changes nothing about the values computed, only how many Python
     calls it takes.
  4. The SSM recurrence itself: `dnc_parallel_scan.py`'s module docstring
     and this repository's own validation script prove the Hillis-Steele
     scan there produces bit-identical (up to ordinary floating-point
     associativity error, ~1e-6 to ~1e-8 depending on dtype/chunk length --
     see that file's docstring) results to the naive sequential recurrence
     it replaces, at Mamba's actual (B, C, d_inner, d_state) tensor shape,
     across randomized trials spanning chunk lengths 1-32, INCLUDING the
     `C == 1` case reducing to the exact single-step formula algebraically
     (the scan's doubling loop provably does not execute when `C == 1`).

Taken together: `forward_chunk()` over a chunk of length C, given the same
starting `(conv_state, ssm_state)`, produces the same `(new_conv_state,
new_ssm_state, [y_1..y_C])` as C sequential `.step()` calls, up to ordinary
floating-point non-associativity (the same caveat that already applies to,
e.g., `nn.LSTM`'s cuDNN fused kernel vs. a hand-rolled per-step LSTM cell --
this is not specific to this module). This is what makes `chunk_size=1`
(i.e. one `forward_chunk()` call per real timestep, C=1 each time) the
mandatory, checkable regression test for "did wiring this in actually
change anything it shouldn't have" -- see `chunked_parallel_dnc.py`.

--------------------------------------------------------------------------
What forward_chunk() is NOT
--------------------------------------------------------------------------
It is not a replacement for `mamba-ssm`'s own fused `selective_scan_fn` /
`causal_conv1d_fn` CUDA kernels (which additionally fuse every step into
one kernel launch resident in GPU SRAM, per Gu & Dao 2024 Section 3.3's
"kernel fusion" + "recomputation" techniques) -- this module is pure
PyTorch, portable, and does not require the CUDA build toolchain
`mamba-ssm`'s fast path needs. It gets the *algorithmic* class of speedup
(O(log C) sequential depth instead of O(C), i.e. genuine parallel-scan
scaling) without the *kernel-fusion* class of speedup. Both are real and
distinct contributions in the Mamba paper; this file claims only the
former, honestly, per the roadmap's "Important information to Step 2"
section's own framing of what a from-scratch parallel-scan wrapper can and
cannot claim relative to the reference CUDA kernels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from model.controller.mamba_controller import (  # noqa: F401 -- MambaControllerBlock re-exported for callers that only need the sequential path
    MambaControllerBlock,
    MambaControllerCell,
    _require_mamba_ssm,
)
from Parallelization_Attempt.dnc_parallel_scan import selective_scan_chunk


# ==========================================================================
# 1. MambaChunkControllerCell -- chunk-parallel S6 update
# ==========================================================================
class MambaChunkControllerCell(MambaControllerCell):
    """Subclasses `MambaControllerCell` purely to add `forward_chunk()`.

    Construction (`__init__`), the parameter container (`self.mamba`), and
    the single-step `step()` method are all inherited UNCHANGED from
    `mamba_controller.py` -- this class adds exactly one new method and
    changes nothing else, so it remains usable anywhere a
    `MambaControllerCell` is (e.g. for a `chunk_size=1` run, `step()` and
    `forward_chunk()` are both available and, per this file's module
    docstring, provably agree).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mamba.in_proj.weight.is_cuda:
            self.forward_chunk = torch.compile(self.forward_chunk, dynamic=False)

    def forward_chunk(
        self,
        hidden_states_chunk: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Chunk-parallel S6 update. `hidden_states_chunk`: (B, C, d_model).

        Returns (out_chunk, new_conv_state, new_ssm_state):
          out_chunk:      (B, C, d_model)
          new_conv_state: (B, d_inner, d_conv)   -- same shape/convention as step()
          new_ssm_state:  (B, d_inner, d_state)  -- same shape/convention as step()

        See module docstring, "Exactness of forward_chunk() vs. step() x C",
        for the line-by-line correspondence to `step()`.
        """
        m = self.mamba
        dtype = hidden_states_chunk.dtype
        B, C, _ = hidden_states_chunk.shape

        # ---- input/gate projection, whole chunk at once -------------------
        xz = m.in_proj(hidden_states_chunk)  # (B, C, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # (B, C, d_inner) each

        # ---- causal depthwise conv, ONE grouped conv1d over the chunk -----
        # extended = [last (d_conv-1) raw x's from before this chunk] ++
        #            [this chunk's C raw x's], laid out as (B, d_inner, L)
        # so that a single valid-mode (padding=0) grouped conv1d produces
        # exactly the C outputs the rolling-window step() recurrence would
        # have produced one at a time -- see module docstring point 2.
        x_bdc = x.transpose(1, 2)  # (B, d_inner, C)
        extended = torch.cat([conv_state[:, :, 1:], x_bdc], dim=-1)  # (B, d_inner, d_conv-1+C)
        x_conv_bdc = F.conv1d(
            extended, m.conv1d.weight, bias=m.conv1d.bias, groups=self.d_inner, padding=0
        )  # (B, d_inner, C) -- valid-mode conv: (d_conv-1+C) - d_conv + 1 == C
        x_conv_bdc = m.act(x_conv_bdc).to(dtype=dtype)
        x_conv = x_conv_bdc.transpose(1, 2)  # (B, C, d_inner)

        new_conv_state = extended[:, :, -self.d_conv:]  # (B, d_inner, d_conv), same convention as step()

        # ---- input-dependent selection parameters (Delta, B, C), whole chunk ----
        x_db = m.x_proj(x_conv)  # (B, C, dt_rank + 2*d_state)
        dt, Bparam, Cparam = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, m.dt_proj.weight)  # bias added below, inside softplus -- (B, C, d_inner)
        dt = F.softplus(dt + m.dt_proj.bias.to(dtype=dt.dtype))  # (B, C, d_inner)
        A = -torch.exp(m.A_log.float())  # (d_inner, d_state) -- no chunk dependency, computed once

        # ---- selective-scan over the whole chunk, via the parallel scan ----
        dA = torch.exp(torch.einsum("bcd,dn->bcdn", dt, A))  # (B, C, d_inner, d_state)
        dB = torch.einsum("bcd,bcn->bcdn", dt, Bparam)  # (B, C, d_inner, d_state)
        dBx = dB * x_conv.unsqueeze(-1)  # (B, C, d_inner, d_state)

        h = selective_scan_chunk(dA, dBx, ssm_state.float(), time_dim=1)  # (B, C, d_inner, d_state)

        y = torch.einsum("bcdn,bcn->bcd", h.to(dtype), Cparam)  # (B, C, d_inner)
        y = y + m.D.to(dtype) * x_conv
        y = y * m.act(z)  # gated output, (B, C, d_inner)

        out = m.out_proj(y)  # (B, C, d_model)
        new_ssm_state = h[:, -1].to(dtype=ssm_state.dtype)  # (B, d_inner, d_state)
        return out, new_conv_state, new_ssm_state


# ==========================================================================
# 2. MambaChunkControllerBlock -- pre-norm residual wrapper, chunk-mode
# ==========================================================================
class MambaChunkControllerBlock(torch.nn.Module):
    """Add -> LN -> Mixer residual block around one
    `MambaChunkControllerCell`, structurally identical to
    `mamba_controller.MambaControllerBlock` (same LayerNorm-then-residual
    pattern) but built directly around the chunk-capable cell, and exposing
    both `step()` (for chunk_size=1 / equivalence testing) and
    `forward_chunk()`.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        layer_idx: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.cell = MambaChunkControllerCell(
            d_model, d_state=d_state, d_conv=d_conv, expand=expand,
            layer_idx=layer_idx, device=device, dtype=dtype,
        )

    def init_state(self, batch_size: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        return self.cell.init_state(batch_size, device=device, dtype=dtype)

    def step(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        """Single-timestep path -- delegates to the (inherited, unmodified)
        `MambaControllerCell.step()`. Byte-for-byte the same computation as
        `mamba_controller.MambaControllerBlock.step()`; kept here so a
        `MambaChunkControllerBlock` can be driven one step at a time too
        (used by the chunk_size=1 equivalence check in this repo's smoke
        tests, and available to any caller that wants to mix step-mode and
        chunk-mode calls on the same block)."""
        conv_state, ssm_state = state
        out, new_conv_state, new_ssm_state = self.cell.step(self.norm(x), conv_state, ssm_state)
        return x + out, (new_conv_state, new_ssm_state)

    def forward_chunk(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        """Chunk-parallel path. `x`: (B, C, d_model). Returns
        (x + out, new_state), same residual/pre-norm structure as `step()`,
        just over a whole chunk at once."""
        conv_state, ssm_state = state
        out, new_conv_state, new_ssm_state = self.cell.forward_chunk(self.norm(x), conv_state, ssm_state)
        return x + out, (new_conv_state, new_ssm_state)


# ==========================================================================
# 3. MambaChunkControllerWrapper -- stack of blocks, chunk-capable
# ==========================================================================
class MambaChunkControllerWrapper(torch.nn.Module):
    """Stacks `num_blocks` `MambaChunkControllerBlock`s. Exposes:

      - `step(x_unsq, hx)`: the exact `dnc.dnc.DNC._layer_forward`-compatible
        single-timestep calling convention `MambaControllerWrapper.forward`
        (mamba_controller.py) already provides -- kept here under the name
        `step` (not `forward`) to make the two call sites unambiguous at
        every call site in `chunked_parallel_dnc.py` (that file is the only
        caller, and it always makes an explicit choice between the two).
      - `forward_chunk(x_chunk, hx)`: `x_chunk` shape (B, C, in_dim), runs
        every block's `forward_chunk` over the whole chunk. This is the
        method `chunked_parallel_dnc.py` calls once per chunk when
        `CONTROLLER_TYPE == "mamba"`.
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        num_blocks: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        _require_mamba_ssm()
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.in_adapter: torch.nn.Module = (
            torch.nn.Identity() if in_dim == d_model else torch.nn.Linear(in_dim, d_model, device=device, dtype=dtype)
        )
        self.blocks = torch.nn.ModuleList(
            [
                MambaChunkControllerBlock(
                    d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                    layer_idx=i, device=device, dtype=dtype,
                )
                for i in range(num_blocks)
            ]
        )

    def init_state(self, batch_size: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        if device is None or dtype is None:
            p = next(self.parameters())
            device = device if device is not None else p.device
            dtype = dtype if dtype is not None else p.dtype
        return [blk.init_state(batch_size, device=device, dtype=dtype) for blk in self.blocks]

    def step(self, input: torch.Tensor, hx):
        """Single-timestep path, identical calling convention to
        `mamba_controller.MambaControllerWrapper.forward`. `input`: (B, 1, in_dim)."""
        assert input.dim() == 3 and input.size(1) == 1, (
            "MambaChunkControllerWrapper.step only supports single-timestep "
            f"calls (got shape {tuple(input.shape)})."
        )
        x = input.squeeze(1)
        x = self.in_adapter(x)
        if hx is None:
            hx = self.init_state(x.size(0), device=x.device, dtype=x.dtype)
        new_hx = []
        for block, state in zip(self.blocks, hx):
            x, new_state = block.step(x, state)
            new_hx.append(new_state)
        return x.unsqueeze(1), new_hx

    def forward_chunk(self, input_chunk: torch.Tensor, hx):
        """Chunk-parallel path. `input_chunk`: (B, C, in_dim). Returns
        (out_chunk, new_hx) with `out_chunk`: (B, C, d_model)."""
        assert input_chunk.dim() == 3, (
            f"MambaChunkControllerWrapper.forward_chunk expects (B, C, in_dim), got {tuple(input_chunk.shape)}"
        )
        x = self.in_adapter(input_chunk)  # (B, C, d_model)
        if hx is None:
            hx = self.init_state(x.size(0), device=x.device, dtype=x.dtype)
        new_hx = []
        for block, state in zip(self.blocks, hx):
            x, new_state = block.forward_chunk(x, state)
            new_hx.append(new_state)
        return x, new_hx
