"""
mamba_controller.py -- v1 (new file)

Alternate Phase 3, Step 1 (see Experiment-Roadmap.md, "Alternative Phase 3 -
fits better to the programs architecture"): wire a Mamba-1 (selective SSM)
controller into the pytorch-dnc `DNC` in place of the LSTM controller used
by Phase 2 (`PHASE_2_graph_traversal_KL_prior.py`), while leaving memory
addressing (content lookup, allocation, temporal link matrix, read modes)
and the Phase 1/2 stochastic-write-head KL machinery completely untouched.

Library used: `mamba-ssm` (per the roadmap's explicit instruction, Mamba-1
is tried first, not Mamba-2/S4, because it's the simplest option). This
file reuses `mamba_ssm.modules.mamba_simple.Mamba` for its parameters and
initialization (in_proj, conv1d, x_proj, dt_proj, A_log, D, out_proj -- the
exact S6 parameterization from the Mamba paper, Algorithm 2 / Section 3.6),
but does NOT use that class's own `.forward()` or `.step()` methods. Why,
and what this file does instead, is the one piece of non-obvious design in
here -- see "Why not just call Mamba.step()?" below before editing the
step() math in MambaControllerCell.

--------------------------------------------------------------------------
Why a separate file (as requested)
--------------------------------------------------------------------------
pytorch-dnc's `DNC.__init__` builds `self.rnns[layer]` as a `nn.LSTM` /
`nn.GRU` / `nn.RNN` directly inline (see `dnc/dnc.py` in
https://github.com/ixaxaar/pytorch-dnc), and `DNC._init_hidden` has an
LSTM-specific branch for constructing the initial controller hidden state.
Neither `rnn_type='mamba'` nor a `nn.Module` with a fundamentally different
hidden-state shape (Mamba needs a `(conv_state, ssm_state)` pair per block,
not an `(h, c)` LSTM pair) is supported by the stock class. Rather than
monkey-patching or forking pytorch-dnc, this file defines:

  1. `MambaControllerCell`   -- one Mamba-1 block, single-timestep, BPTT-safe.
  2. `MambaControllerBlock`  -- pre-norm residual wrapper around (1), mirroring
                                 mamba_ssm's own `Block` (Add -> LN -> Mixer).
  3. `MambaControllerWrapper`-- stacks N `MambaControllerBlock`s and exposes
                                 the exact calling convention pytorch-dnc's
                                 `DNC._layer_forward` expects from
                                 `self.rnns[layer]`: callable as
                                 `module(x_unsqueezed_at_dim1, hx) -> (out_unsqueezed, new_hx)`,
                                 the same shape contract as `nn.LSTM(...,
                                 batch_first=True)` called one timestep at a
                                 time (`x.unsqueeze(1)`).
  4. `MambaDNC`               -- subclasses `dnc.DNC`. For
                                 `rnn_type != 'mamba'` it defers to the stock
                                 `DNC.__init__`/`_init_hidden` untouched (so
                                 this class is a strict drop-in superset --
                                 `rnn_type='lstm'` behaves byte-identically
                                 to plain `dnc.DNC`, which is what makes an
                                 LSTM-vs-Mamba A/B comparison a one-line
                                 change in the training script). For
                                 `rnn_type == 'mamba'` it builds
                                 `self.rnns[layer]` as a `MambaControllerWrapper`
                                 instead, and overrides `_init_hidden`'s
                                 controller-state branch to build
                                 `(conv_state, ssm_state)` pairs instead of
                                 `(h, c)`. Memory construction
                                 (`dnc.memory.Memory`), the output projection,
                                 and `forward()`/`_layer_forward()` are all
                                 inherited from `dnc.DNC` UNCHANGED -- this is
                                 what guarantees addressing stays untouched
                                 and that `install_stochastic_write_heads()`
                                 from `stochastic_write_head_v2.py` keeps
                                 working with zero modification (it only ever
                                 looks for `dnc.memory.Memory` submodules and
                                 patches `write_vector_transform`; it never
                                 looks at the controller).

--------------------------------------------------------------------------
Why not just call Mamba.step()?  (the one thing worth reading carefully)
--------------------------------------------------------------------------
`mamba_ssm.modules.mamba_simple.Mamba` ships two ways to run the model:

  - `.forward(hidden_states)`: takes the WHOLE sequence at once (B, L, D)
    and is only fast because it uses `causal_conv1d_fn` / `selective_scan_fn`
    CUDA kernels (or an FFT-free full-sequence scan) that assume every
    timestep's input is already available. This doesn't fit here: DNC has
    to interleave one controller step with one memory read/write at a time
    (the read vector from step t feeds the controller's input at step t+1),
    so the whole point of wiring Mamba in as *DNC's controller* is defeated
    if we can't get the model one token at a time.

  - `.step(hidden_states, conv_state, ssm_state)`: the autoregressive
    decoding path, exactly one token at a time -- this is the right
    granularity. BUT it is written for inference, and mutates its own state
    tensors IN PLACE via `conv_state.copy_(...)` / `ssm_state.copy_(...)`
    (see mamba_ssm/modules/mamba_simple.py). That's fine under
    `torch.no_grad()` for text generation, but it is NOT safe for
    backprop-through-time: the in-place `.copy_()` overwrites the exact
    tensor version autograd needs to compute the backward pass through the
    *previous* timestep's `ssm_state * dA` multiply. Concretely, PyTorch
    raises:

        RuntimeError: one of the variables needed for gradient computation
        has been modified by an inplace operation ...

    the moment `.backward()` is called on a loss that was accumulated over
    more than one `.step()` call chained through the same state tensors --
    which is exactly what a T-step DNC unroll does. (Verified directly: a
    minimal repro of this exact `state.copy_(new_state)`-across-a-python-
    loop pattern throws this RuntimeError on `.backward()`, while the
    identical recurrence written out-of-place -- `state = state * dA + ...`,
    no `.copy_()` -- backpropagates cleanly through an arbitrary number of
    chained steps.)

  So: `MambaControllerCell.step()` below reimplements the SAME math as
  `Mamba.step()`'s non-fast-path branch (the branch it falls back to when
  `causal_conv1d_update`/`selective_state_update` aren't available -- i.e.
  the portable, dependency-light path, matching the roadmap's "Mamba-1
  first because it's simplest"), line-for-line equivalent, EXCEPT every
  in-place `.copy_()` is replaced with a plain out-of-place tensor
  expression that returns a NEW state tensor each call, exactly the same
  pattern `nn.LSTM` already uses (a step returns new `(h, c)`, it doesn't
  mutate the old `(h, c)` in place). This is what makes it safe to chain
  inside pytorch-dnc's own per-timestep Python loop under full BPTT.

  We still reuse the actual `mamba_ssm.modules.mamba_simple.Mamba` class
  for parameter construction (`in_proj`, `conv1d`, `x_proj`, `dt_proj`,
  `A_log`, `D`, `out_proj`) and all of its initialization logic (S4D-real
  `A_log` init, the dt-bias inverse-softplus init, etc.) -- we only bypass
  its `forward`/`step` methods, never its `__init__`.

--------------------------------------------------------------------------
Design choices specific to this wiring (Step 1 scope)
--------------------------------------------------------------------------
- `d_model` for every Mamba block is fixed to DNC's `hidden_size`
  (`output_size` in pytorch-dnc's naming), since the controller's output
  (post-clip) is fed directly to `Memory.forward()` as the interface vector
  `ξ`, which must have exactly that width. A small `nn.Linear` "in_adapter"
  handles the one dimension mismatch pytorch-dnc's own LSTM absorbs
  implicitly in its first-layer weight matrix: `nn_input_size` (=
  `input_size + read_vectors_size`) -> `hidden_size`, applied once before
  the Mamba block stack. Every subsequent stacked block already operates at
  `hidden_size`, so no further adapter is needed (identical in spirit to
  how `nn.LSTM(num_layers=N)`'s layer-0 weight differs in shape from
  layers 1..N-1).
- `num_hidden_layers` (pytorch-dnc's existing knob, previously "how many
  stacked LSTM layers form this DNC layer's controller", default 2) is
  reused, unchanged in meaning, as "how many stacked Mamba blocks form this
  DNC layer's controller" -- so Phase 2's existing model-capacity constants
  don't need a new knob to keep meaning "the controller has some depth".
- `mamba_d_state` / `mamba_d_conv` / `mamba_expand` are new, Mamba-specific
  hyperparameters (defaults 16 / 4 / 2, the Mamba-1 paper's own defaults),
  exposed as constructor kwargs on `MambaDNC` with no analogue in the LSTM
  path -- there's nothing to keep in sync with, since LSTM has no such
  parameters.
- Initial `(conv_state, ssm_state)` are zero-initialized (matching
  `Mamba.allocate_inference_cache`'s own convention), NOT
  `xavier_uniform_`-initialized the way pytorch-dnc seeds its LSTM `h0`.
  This is a deliberate divergence, not an oversight: `xavier_uniform_` on a
  conv/ssm state tensor has no principled meaning in this parameterization
  (Mamba's own reference implementation always starts both at exactly
  zero), so keeping the library's own convention here is more faithful
  than forcing LSTM's init convention onto a structurally different state.
- Everything here targets Mamba-1 (`mamba_ssm.modules.mamba_simple.Mamba`)
  specifically, not Mamba-2/S4, per the roadmap's explicit sequencing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dnc import DNC
from dnc.memory import Memory
from dnc.util import cuda

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError as _mamba_import_error:  # pragma: no cover - environment-dependent
    Mamba = None
    _MAMBA_IMPORT_ERROR = (
        "mamba_controller.py requires the `mamba-ssm` package "
        "(`pip install mamba-ssm`, which needs a CUDA build toolchain / "
        "nvcc available at install time -- see mamba-ssm's own install "
        "instructions). Original import error: "
        f"{_mamba_import_error}"
    )
else:
    _MAMBA_IMPORT_ERROR = None


def _require_mamba_ssm() -> None:
    if Mamba is None:
        raise ImportError(_MAMBA_IMPORT_ERROR)


# ==========================================================================
# 1. MambaControllerCell -- one Mamba-1 block, single-timestep, BPTT-safe
# ==========================================================================
class MambaControllerCell(nn.Module):
    """One Mamba-1 (selective SSM) block, driven one timestep at a time.

    Reuses `mamba_ssm.modules.mamba_simple.Mamba` purely as a parameter
    container (correct shapes + the library's own initialization), and
    reimplements the single-step S6 recurrence out-of-place so it can be
    backpropagated through an arbitrary number of chained calls -- see the
    module docstring's "Why not just call Mamba.step()?" section for why
    the library's own `.step()` cannot be used here as-is.

    State: `(conv_state, ssm_state)`, shapes `(B, d_inner, d_conv)` and
    `(B, d_inner, d_state)` respectively -- same shapes as
    `Mamba.allocate_inference_cache()` produces, so this stays a drop-in
    match for the library's own state convention even though the state is
    threaded functionally here rather than mutated in place.
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
        _require_mamba_ssm()

        # use_fast_path=False: purely defensive/documentary. We never call
        # `self.mamba.forward()` or `self.mamba.step()` on this instance --
        # only its submodules (in_proj, conv1d, ...) are used, by `step()`
        # below -- so this flag has no effect on us either way. Set False so
        # nothing accidentally routes through the causal_conv1d/
        # mamba_inner_fn CUDA fast path if some other code ever calls
        # `.forward()` on `self.mamba` directly.
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            layer_idx=layer_idx,
            use_fast_path=False,
            device=device,
            dtype=dtype,
        )
        self.d_model = d_model
        self.d_inner = self.mamba.d_inner
        self.d_state = self.mamba.d_state
        self.d_conv = self.mamba.d_conv
        self.dt_rank = self.mamba.dt_rank

    def init_state(
        self, batch_size: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-initialized (conv_state, ssm_state), matching
        `Mamba.allocate_inference_cache`'s shapes/convention (see module
        docstring for why this stays zero-init rather than xavier-init)."""
        conv_state = torch.zeros(batch_size, self.d_inner, self.d_conv, device=device, dtype=dtype)
        ssm_state = torch.zeros(batch_size, self.d_inner, self.d_state, device=device, dtype=dtype)
        return conv_state, ssm_state

    def step(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-timestep S6 update. `hidden_states`: (B, d_model).

        Line-for-line equivalent to `Mamba.step()`'s non-fast-path branch
        (mamba_ssm/modules/mamba_simple.py), except every state update is
        out-of-place (`new_state = ...`, no `.copy_()`), so this function is
        safe to call repeatedly inside a Python loop under full
        backprop-through-time -- see module docstring.
        """
        m = self.mamba
        dtype = hidden_states.dtype

        # ---- input/gate projection --------------------------------------
        xz = m.in_proj(hidden_states)  # (B, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # (B, d_inner) each

        # ---- causal depthwise conv, functional (rolling window) ---------
        # conv_state holds the last (d_conv - 1) x's; append the new x and
        # drop the oldest column -- out-of-place shift, mirrors what
        # Mamba.step() does via conv_state.copy_(roll(...)) but without
        # mutating the input tensor.
        new_conv_state = torch.cat([conv_state[:, :, 1:], x.unsqueeze(-1)], dim=-1)
        conv_weight = m.conv1d.weight.squeeze(1)  # (d_inner, 1, d_conv) -> (d_inner, d_conv)
        x_conv = torch.sum(new_conv_state * conv_weight, dim=-1)  # (B, d_inner)
        if m.conv1d.bias is not None:
            x_conv = x_conv + m.conv1d.bias
        x_conv = m.act(x_conv).to(dtype=dtype)

        # ---- input-dependent selection parameters (Delta, B, C) ---------
        x_db = m.x_proj(x_conv)  # (B, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, m.dt_proj.weight)  # bias added below, inside softplus
        dt = F.softplus(dt + m.dt_proj.bias.to(dtype=dt.dtype))  # (B, d_inner)
        # ---- selective-scan single step, out-of-place --------------------
        # FIX (grad_norm nan, independent of beta -- see chat log): under
        # torch.amp.autocast, einsum is an autocast-to-fp16 op, so A's
        # .float() cast above was silently undone here, and m.A_log has no
        # upper bound -- torch.exp(m.A_log.float()) can overflow to +inf
        # while softplus(dt) can underflow to exactly 0.0 in fp16 range,
        # producing 0.0 * -inf == nan that poisons ssm_state on every step,
        # for every beta, since this path has no dependency on beta_eff.
        # Force the whole recurrence to real fp32 regardless of the outer
        # autocast context (matching how mamba_ssm's own CUDA kernel always
        # accumulates the scan in fp32 even with fp16 activations), and
        # clamp A_log so A can never reach -inf in the first place.
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            dt32 = dt.float().clamp(min=1e-6)
            B32 = B.float()
            C32 = C.float()
            x_conv32 = x_conv.float()
            A_log_c = m.A_log.float().clamp(max=20.0)  # exp(20) already far above any dt*A this model needs
            A32 = -torch.exp(A_log_c)  # (d_inner, d_state)
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt32, A32))
            dB = torch.einsum("bd,bn->bdn", dt32, B32)
            new_ssm_state32 = ssm_state.float() * dA + x_conv32.unsqueeze(-1) * dB  # (B, d_inner, d_state)
            y32 = torch.einsum("bdn,bn->bd", new_ssm_state32, C32)
            y32 = y32 + m.D.float() * x_conv32
            y32 = y32 * m.act(z).float()  # gated output
        new_ssm_state = new_ssm_state32.to(dtype)
        y = y32.to(dtype)

        out = m.out_proj(y)  # (B, d_model)
        return out, new_conv_state, new_ssm_state


# ==========================================================================
# 2. MambaControllerBlock -- pre-norm residual wrapper around one cell
# ==========================================================================
class MambaControllerBlock(nn.Module):
    """Add -> LN -> Mixer residual block around one `MambaControllerCell`,
    mirroring `mamba_ssm.modules.block.Block`'s own pattern (pre-norm,
    residual returned separately upstream for fused-kernel reasons that
    don't apply to us; here we just add it back directly since we're not
    using the fused add+norm kernel)."""

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
        self.norm = nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.cell = MambaControllerCell(
            d_model, d_state=d_state, d_conv=d_conv, expand=expand,
            layer_idx=layer_idx, device=device, dtype=dtype,
        )

    def init_state(self, batch_size: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        return self.cell.init_state(batch_size, device=device, dtype=dtype)

    def step(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        conv_state, ssm_state = state
        out, new_conv_state, new_ssm_state = self.cell.step(self.norm(x), conv_state, ssm_state)
        return x + out, (new_conv_state, new_ssm_state)


# ==========================================================================
# 3. MambaControllerWrapper -- stack of blocks, nn.LSTM-compatible call API
# ==========================================================================
class MambaControllerWrapper(nn.Module):
    """Stacks `num_blocks` `MambaControllerBlock`s and exposes the exact
    calling convention `dnc.dnc.DNC._layer_forward` expects from
    `self.rnns[layer]`:

        out_unsq, new_hx = wrapper(x_unsq, hx)

    where `x_unsq` has shape (B, 1, in_dim) (DNC always calls with
    `input.unsqueeze(1)`, one timestep at a time) and `out_unsq` has shape
    (B, 1, d_model). `hx` is a list of `(conv_state, ssm_state)` tuples, one
    per stacked block, or `None` for a fresh zero state.
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
        self.d_model = d_model
        self.num_blocks = num_blocks
        # Dimension-matching adapter for the first block only -- mirrors how
        # nn.LSTM(num_layers=N)'s layer-0 weight matrix differs in shape
        # from layers 1..N-1 (layer 0 maps in_dim -> hidden, the rest map
        # hidden -> hidden). Identity when already matched (in_dim == d_model),
        # e.g. for DNC layers beyond the first when num_layers > 1.
        self.in_adapter: nn.Module = (
            nn.Identity() if in_dim == d_model else nn.Linear(in_dim, d_model, device=device, dtype=dtype)
        )
        self.blocks = nn.ModuleList(
            [
                MambaControllerBlock(
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

    def forward(self, input: torch.Tensor, hx):
        # input: (B, 1, in_dim) -- DNC's `_layer_forward` always calls with
        # `input.unsqueeze(1)`, a single timestep at a time, since memory
        # read/write must be interleaved between controller steps.
        assert input.dim() == 3 and input.size(1) == 1, (
            "MambaControllerWrapper only supports single-timestep calls "
            f"(got shape {tuple(input.shape)}); this mirrors how DNC drives "
            "nn.LSTM one step at a time, never a full sequence at once."
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


# ==========================================================================
# 4. MambaDNC -- dnc.DNC subclass that can build a Mamba-1 controller
# ==========================================================================
class MambaDNC(DNC):
    """`dnc.DNC` subclass that adds `rnn_type='mamba'` as a third controller
    option alongside pytorch-dnc's existing `'rnn'`/`'gru'`/`'lstm'`.

    For any `rnn_type` other than `'mamba'`, `__init__`/`_init_hidden` defer
    entirely to `dnc.DNC`'s own implementation -- this class is a strict
    drop-in superset, so switching the Phase 2 training script between LSTM
    and Mamba controllers is a one-line change (`rnn_type='lstm'` vs.
    `rnn_type='mamba'`), not a code fork. All memory/addressing construction
    (`dnc.memory.Memory`, the output projection) is copied verbatim from
    `dnc.DNC.__init__` for the `'mamba'` branch below, to guarantee it can
    never silently drift from upstream's addressing logic.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rnn_type: str = "lstm",
        num_layers: int = 1,
        num_hidden_layers: int = 2,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0,
        nr_cells: int = 5,
        read_heads: int = 2,
        cell_size: int = 10,
        nonlinearity: str = "tanh",
        independent_linears: bool = False,
        share_memory_between_layers: bool = True,
        debug: bool = False,
        clip: float = 20,
        device: torch.device | None = None,
        # Mamba-specific hyperparameters (no LSTM/GRU/RNN analogue -- see
        # module docstring's "Design choices" section). Defaults are
        # Mamba-1's own paper defaults.
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
    ):
        if rnn_type.lower() != "mamba":
            # Not our concern -- defer completely to the stock DNC. This is
            # what makes MambaDNC a true drop-in superset rather than a fork.
            super().__init__(
                input_size=input_size,
                hidden_size=hidden_size,
                rnn_type=rnn_type,
                num_layers=num_layers,
                num_hidden_layers=num_hidden_layers,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout,
                nr_cells=nr_cells,
                read_heads=read_heads,
                cell_size=cell_size,
                nonlinearity=nonlinearity,
                independent_linears=independent_linears,
                share_memory_between_layers=share_memory_between_layers,
                debug=debug,
                clip=clip,
                device=device,
            )
            return

        _require_mamba_ssm()

        # ---- rnn_type == "mamba": custom construction --------------------
        # Deliberately NOT calling DNC.__init__ (it hardcodes nn.RNN/GRU/LSTM
        # construction inline with no extension point) -- instead, replicate
        # its non-controller-specific bookkeeping verbatim (same attribute
        # names, same Memory/output construction) and only swap in the
        # Mamba controller for `self.rnns`.
        nn.Module.__init__(self)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.num_hidden_layers = num_hidden_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.nr_cells = nr_cells
        self.read_heads = read_heads
        self.cell_size = cell_size
        self.nonlinearity = nonlinearity
        self.independent_linears = independent_linears
        self.share_memory_between_layers = share_memory_between_layers
        self.debug = debug
        self.clip = clip
        self.device = device

        # Mamba-only bookkeeping, stored for __repr__ / checkpoint metadata.
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand

        self.w = self.cell_size
        self.r = self.read_heads
        self.read_vectors_size = self.read_heads * self.cell_size
        self.output_size = self.hidden_size
        self.nn_input_size = self.input_size + self.read_vectors_size
        self.nn_output_size = self.output_size + self.read_vectors_size

        self.rnns: list[nn.Module] = []
        self.memories: list[Memory] = []

        for layer in range(self.num_layers):
            in_dim = self.nn_input_size if layer == 0 else self.nn_output_size
            controller = MambaControllerWrapper(
                in_dim=in_dim,
                d_model=self.output_size,
                num_blocks=self.num_hidden_layers,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                device=device,
            )
            self.rnns.append(controller)
            # setattr so this shows up as a proper submodule for autograd /
            # optimizer.parameters(), same trick dnc.DNC itself uses for its
            # own self.rnns entries.
            setattr(self, "mamba_layer_" + str(layer), controller)

            # memories for each layer -- copied verbatim from dnc.DNC.__init__
            if not self.share_memory_between_layers:
                self.memories.append(
                    Memory(
                        input_size=self.output_size,
                        nr_cells=self.nr_cells,
                        cell_size=self.w,
                        read_heads=self.r,
                        device=self.device,
                        independent_linears=self.independent_linears,
                    )
                )
                setattr(self, "rnn_layer_memory_" + str(layer), self.memories[layer])

        # only one memory shared by all layers -- copied verbatim from
        # dnc.DNC.__init__
        if self.share_memory_between_layers:
            self.memories.append(
                Memory(
                    input_size=self.output_size,
                    nr_cells=self.nr_cells,
                    cell_size=self.w,
                    read_heads=self.r,
                    device=self.device,
                    independent_linears=self.independent_linears,
                )
            )
            setattr(self, "rnn_layer_memory_shared", self.memories[0])

        # final output layer -- copied verbatim from dnc.DNC.__init__
        self.output = nn.Linear(self.nn_output_size, self.input_size)
        torch.nn.init.kaiming_uniform_(self.output.weight)

        if self.device is not None and self.device.type == "cuda":
            self.to(self.device)

    def _init_hidden(self, hx, batch_size: int, reset_experience: bool):
        if self.rnn_type.lower() != "mamba":
            return super()._init_hidden(hx, batch_size, reset_experience)

        # ---- controller-state branch: Mamba (conv_state, ssm_state) ------
        if hx is not None:
            chx, mhx, last_read = hx
        else:
            chx, mhx, last_read = None, None, None

        if chx is None:
            chx = [
                self.rnns[layer].init_state(batch_size, device=self.device)
                for layer in range(self.num_layers)
            ]

        if last_read is None:
            last_read = cuda(torch.zeros(batch_size, self.w * self.r), device=self.device)

        # ---- memory-state branch: byte-identical to dnc.DNC._init_hidden -
        # (copied verbatim -- this logic is controller-agnostic and must not
        # drift from upstream's own resume/reset semantics)
        if mhx is None:
            if self.share_memory_between_layers:
                mhx = [self.memories[0].reset(batch_size, erase=reset_experience)]
            else:
                mhx = [m.reset(batch_size, erase=reset_experience) for m in self.memories]
        else:
            if self.share_memory_between_layers:
                if len(mhx) == 0 or mhx[0] is None:
                    mhx = [self.memories[0].reset(batch_size, erase=reset_experience)]
                else:
                    mhx = [self.memories[0].reset(batch_size, mhx[0], erase=reset_experience)]
            else:
                if len(mhx) == 0:
                    mhx = [m.reset(batch_size, erase=reset_experience) for m in self.memories]
                else:
                    new_mhx = []
                    for i, m in enumerate(self.memories):
                        if i < len(mhx) and mhx[i] is not None:
                            new_mhx.append(m.reset(batch_size, mhx[i], erase=reset_experience))
                        else:
                            new_mhx.append(m.reset(batch_size, erase=reset_experience))
                    mhx = new_mhx

        return chx, mhx, last_read
