"""
deer_parallel_dnc.py -- v1 (new file, updated with torch.no_grad() fix)

Alternate Phase 3, Step 2, Option 3 (see Experiment-Roadmap.md, "Attempt to
modify DNC in a way that it can in one way or another simulate SSMs
parallelism" -> "3. Iterative fixed-point parallelization of the full
nonlinear recurrence"): the DNC-specific wiring around the generic
quasi-DEER Newton engine in `newton_associative_scan.py` (v1, unmodified --
that file is imported here exactly as written, per its own module
docstring: "DNC-specific wiring ... lives in the sibling file
`deer_parallel_dnc.py`, which imports this module").

--------------------------------------------------------------------------
Why this builds on MambaDNC (Step 1), NOT ChunkedParallelDNC (Option 2)
--------------------------------------------------------------------------
Per the roadmap's post-mortem on Option 2 (chunking): chunking's damage
came from a structural approximation baked into every forward pass -- reads
inside a chunk are frozen regardless of how well-trained the model is --
and that approximation error turned out to suppress the phase-transition
learning dynamics rather than just trading it for speed. DEER's entire
selling point over that is "exact at convergence, so no structural error to
worry about at all": it solves the TRUE, un-approximated recurrence via
Newton iteration. That guarantee only holds if the function being
linearized is the true sequential transition, not an already-approximate
one. `ChunkedParallelDNC.forward()` is built entirely around "freeze state
for a chunk, reconcile at the boundary" -- differentiating through that
would mean DEER converges exactly to the CHUNKED model's (suppressed-
phase-transition) behavior, defeating the point of switching to it. So this
file subclasses `MambaDNC` (mamba_controller.py, Step 1) directly: the
exact sequential controller + DNC step, with no chunking assumptions baked
in anywhere in the code path being linearized.

--------------------------------------------------------------------------
What this file actually builds
--------------------------------------------------------------------------
`newton_associative_scan.py`'s `deer_quasi_newton_solve` is fully generic:
it knows nothing about DNC, Mamba, or memory matrices -- it just needs a
`per_sample_step(state_vec, input_vec) -> new_state_vec` closure operating
on flat, unbatched tensors, plus a `(state_pytree, spec)` pack/unpack pair
built by `make_state_spec`/`pack_state`/`unpack_state` for whatever nested
dict/list/tuple/Tensor structure the real state happens to have. This file
supplies exactly that closure for one MambaDNC layer's per-timestep
transition, plus everything DNC-specific that the generic engine
deliberately doesn't know about:

  1. **State pytree definition.** The recurrent state this file feeds to
     the Newton solver is the 4-tuple `(chx, mhx, last_read, output)` --
     controller state, memory state, last read vectors, and the
     controller's own raw per-timestep output -- exactly the tuple
     `newton_associative_scan.py`'s own `make_state_spec` docstring
     anticipates ("e.g. a DNC layer's `(chx, mhx, last_read, output)`
     tuple, whatever its internal structure actually is"). `output` is
     folled into the state even though it doesn't feed back into the
     recurrence (only `chx`/`mhx`/`last_read` do) -- it's a pure function
     of `(chx_{t-1}, x_t)` at each step, so Newton converges it "for free"
     alongside everything else, and having it land inside the converged
     trajectory `y` means the whole output sequence can be reconstructed
     with zero extra sequential work (see point 3).
  2. **The transition closure**, built from `MambaDNC._layer_forward` --
     pytorch-dnc's own per-timestep, per-layer step function (controller
     call, then `Memory.forward` for content/allocation addressing, read,
     and write) -- called with a synthetic batch-of-1 dimension added
     (`per_sample_step` operates on genuinely unbatched (D,)/(X,) tensors,
     since `deer_quasi_newton_solve` vmaps it over every (t, b) pair
     itself). This file assumes `num_layers == 1` and
     `share_memory_between_layers == True`, asserted in `__init__` --
     the only configuration `Alter_PHASE3_mamba_step2.py` ever
     constructs; extending the state packing to a real multi-layer stack
     is unimplemented.
  3. **Parallel output reconstruction.** Because `output` and `last_read`
     both live inside the converged Newton trajectory `y` (shape
     `(B, T, D)`), the final task-output sequence
     `self.output(cat([output_t, last_read_t]))` is reconstructed for
     every `t` in one vectorized call -- no sequential replay needed for
     this part, preserving DEER's parallelism where it actually matters.
  4. **Handling the stochastic write head** (`stochastic_write_head_v2.py`,
     imported and used completely UNMODIFIED -- per the task's
     instruction not to touch anything else about the model). Two
     distinct problems, two distinct fixes:
       a. *Making the transition function well-defined.* Newton's method
          needs to find a fixed point of a FIXED function. But
          `StochasticWriteHead.forward()` draws `eps = torch.randn_like(std)`
          fresh on every call -- and the Newton solve calls the transition
          function for every timestep many times over (once per Newton
          round, plus once more inside every round's `jacrev` diagonal-
          Jacobian evaluation). If eps were re-drawn every call, the
          "function" being solved for would be a different function each
          time, and there would be nothing to converge to. Fix: pre-sample
          one noise tensor `eps_seq` of shape `(B, T, cell_size)` up
          front, treated as part of the (fixed, given) external input at
          every timestep -- exactly how the DEER/quasi-DEER literature
          itself treats "any (possibly random) input dependence" (Lim et
          al. 2024 suppress it in their own notation; Gonzalez et al. 2025
          note explicitly that DEER/ELK parallelize "any discrete-time
          nonlinear dynamical system ... that may or may not include
          stochasticity"). The `_inject_fixed_noise` context manager below
          makes `torch.randn_like` return the pre-sampled value for the
          duration of one transition-function call, restoring the real
          `torch.randn_like` immediately after -- see that function's own
          docstring for the full argument.
       b. *Not corrupting the KL / prior-snapshot bookkeeping.* The write
          head accumulates `_kl_terms`/`_recent_writes` as a SIDE EFFECT of
          `forward()`, guarded by `self.training`. Left on during the
          Newton solve, every one of those many redundant per-timestep
          evaluations would append another (non-final, intermediate-
          Newton-iterate) entry -- corrupting `pop_total_kl()`'s later
          read for this training step. Fix: the write head is put in
          `.eval()` mode (silencing bookkeeping, but NOT changing what
          value it samples -- `v`'s formula depends only on `self.sample`,
          never on `self.training`, so this doesn't perturb the dynamics
          at all) for the whole Newton solve, then restored to its real
          mode and driven with exactly ONE additional vectorized call over
          the converged trajectory afterward, so `_kl_terms`/
          `_recent_writes` end up populated with exactly one (B*T,
          cell_size)-shaped entry per real forward pass -- the same
          eventual shape `pop_kl`/`update_prior_snapshot` already handle
          (they reshape/concatenate along dim 0 regardless of how many
          list entries it came from), just computed in one shot instead of
          T sequential ones, because `output_traj` (needed to recompute
          mu/logvar) is already sitting in the converged trajectory `y`
          from point 3 -- no extra `_layer_forward` calls needed.
  5. **Correctness self-check against the sequential rollout**
     (`deer_vs_sequential_max_abs_error`, near the bottom of this file):
     runs both a genuine `for t in range(T)` sequential rollout through
     `_layer_forward` and a DEER solve, seeded with the SAME `eps_seq`, and
     reports the max-abs-error between their output sequences -- the same
     style of check `dnc_parallel_scan.py` used for its own scan primitive,
     and the check the roadmap calls out explicitly ("run DEER at enough
     Newton rounds to fully converge and confirm it reproduces MambaDNC's
     sequential rollout on a fixed batch/seed to near machine precision").
     This is a reusable utility function, not a test harness invoked at
     import time.
  6. **Damping / trust-region fallback**: not reimplemented here --
     `deer_quasi_newton_solve`'s `damping`/`max_jac_diag_abs` parameters
     (see newton_associative_scan.py's own module docstring, "Numerical
     safeguard") are simply exposed as constructor kwargs on
     `DEERParallelDNC` and threaded straight through, since the roadmap's
     own risk flag ("DNC's cosine-similarity addressing has flat,
     low-gradient regions ... a poorly-conditioned Jacobian there could
     make Newton's method converge slowly or unstably") is precisely what
     those two knobs already exist to guard against.
  7. **Diagonal-Jacobian memory knob** (v3): `deer_jac_chunk_size` is
     likewise exposed as a constructor kwarg and threaded straight through
     to `deer_quasi_newton_solve`'s `jac_chunk_size` -- see the v3 module-
     docstring note above for why this exists (v1's dense-Jacobian
     diagonal extraction OOMs at this program's actual memory-matrix
     size).

Reused unchanged: `associative_scan_affine`/`selective_scan_chunk`
(`dnc_parallel_scan.py`, via `newton_associative_scan.py`'s import of
`selective_scan_chunk`) as the linear solver called once per Newton round.
Nothing in this file re-derives or re-implements that scan.

v2 (this revision): two correctness fixes, no change to the Newton/DEER math
or the training-script call sites.

1. DEERParallelDNC now asserts rnn_type == "mamba" in __init__. make_state_spec
   (n_leading_dims=1) assumes every state leaf is already batch-first, which
   holds for MambaDNC's state (a list of per-block (conv_state, ssm_state)
   tuples) but NOT for dnc.DNC's own LSTM/GRU/RNN state, which is shaped
   (num_hidden_layers, B, hidden) -- an extra leading dim pack_state would
   silently mis-flatten as if it were the batch dimension. Since
   Alter_PHASE3_mamba_step2.py's --use-deer flag can be combined with the
   default --controller lstm, this was reachable and would have produced
   silently wrong (not crashing) results rather than an error. Per the
   roadmap, MambaDNC is the only nonlinear model DEER is specified to
   linearize around, so this is a hard requirement, not a new limitation.
2. deer_vs_sequential_max_abs_error no longer leaves the global RNG rewound
   after it runs. It previously called torch.manual_seed(seed) directly;
   since this function is invoked once, at startup, with the SAME seed the
   run already used to seed model init and the first sample_batch() call,
   that call silently rewound the global RNG stream back to that earlier
   point -- every training batch sampled after this startup check would
   repeat the early post-seed draws (model init noise, the first curriculum
   sample) instead of continuing to advance. This is the same "global
   reseed mid-run" bug class already fixed once for
   build_london_underground_eval() (see that function's v2 note in the
   training script); it resurfaced here because this check runs inline in
   run() rather than through a dedicated RNG stream. Fixed by saving/
   restoring torch's CPU and CUDA RNG state around the seeded eps_seq draw,
   so the check itself stays reproducible (same seed -> same eps_seq) but
   training's RNG stream is left exactly as it was.

v3 (this revision): threads `newton_associative_scan.py`'s v2
`jac_chunk_size` knob through as a new `DEERParallelDNC` constructor kwarg,
`deer_jac_chunk_size` (default 128). This is purely plumbing -- no change
to what DEER computes -- needed because this program's actual
`nr_cells=256` DNC config makes the packed per-timestep state `D` (chx +
mhx + last_read + output, with `mhx`'s `link_matrix` alone contributing
`nr_cells^2 = 65,536` elements) exceed 100,000, and `newton_associative_
scan.py` v1's dense-per-sample-Jacobian diagonal extraction OOMs at that
scale (see that file's v2 module-docstring section, "Diagonal-Jacobian
computation", for the full story -- reported failure: a `jacrev`-driven
`vmap` call trying to allocate ~47,605 GiB). `deer_quasi_newton_solve`
itself now defaults to memory-safe chunked-JVP computation regardless of
whether this kwarg is touched; exposing it here just lets
`Alter_PHASE3_mamba_step2.py` tune the memory/speed tradeoff per-GPU the
same way `deer_damping`/`deer_max_jac_diag_abs` are already exposed (point
6 above), without editing this file again.

Everything else below is unchanged from v2.
"""

from __future__ import annotations

import contextlib
from typing import Optional

import torch

from model.controller.mamba_controller import MambaDNC
from DEER.newton_associative_scan import (
    make_state_spec,
    pack_state,
    unpack_state,
    deer_quasi_newton_solve,
    deer_adjoint_scan,
    _batched_step_fn,
)
from model.memory_manipulation.stochastic_write_head_v2 import StochasticWriteHead
from DEER.analytic_diag_jac import build_analytic_diag_fn

__all__ = [
    "DEERParallelDNC",
    "deer_vs_sequential_max_abs_error",
]


# ==========================================================================
# 0. Small generic pytree helpers (batch-dim massaging only -- NOT part of
#    the reusable engine in newton_associative_scan.py, which is left
#    untouched; these are DNC-wiring-local plumbing for the fact that
#    `_layer_forward` expects a real batch dimension while
#    `per_sample_step` (per deer_quasi_newton_solve's contract) is called
#    on genuinely unbatched per-example tensors under vmap).
# ==========================================================================
def _tree_unsqueeze0(tree):
    """Add a synthetic batch-of-1 leading dim to every Tensor leaf of an
    arbitrary dict/list/tuple/Tensor/None tree, preserving structure."""
    if tree is None:
        return None
    if isinstance(tree, torch.Tensor):
        return tree.unsqueeze(0)
    if isinstance(tree, dict):
        return {k: _tree_unsqueeze0(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        out = [_tree_unsqueeze0(v) for v in tree]
        return tuple(out) if isinstance(tree, tuple) else out
    raise TypeError(f"_tree_unsqueeze0: unsupported leaf type {type(tree)!r}")


def _tree_squeeze0(tree):
    """Inverse of `_tree_unsqueeze0`: drop the leading batch-of-1 dim."""
    if tree is None:
        return None
    if isinstance(tree, torch.Tensor):
        return tree.squeeze(0)
    if isinstance(tree, dict):
        return {k: _tree_squeeze0(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        out = [_tree_squeeze0(v) for v in tree]
        return tuple(out) if isinstance(tree, tuple) else out
    raise TypeError(f"_tree_squeeze0: unsupported leaf type {type(tree)!r}")


def _find_stochastic_write_head(model) -> StochasticWriteHead:
    """Locate the single installed `StochasticWriteHead` on `model`
    (installed via `install_stochastic_write_heads`, stochastic_write_
    head_v2.py, unmodified). Requires exactly one -- the only
    configuration this file's num_layers==1 /
    share_memory_between_layers==True assumption produces."""
    heads = [m for m in model.modules() if isinstance(m, StochasticWriteHead)]
    if len(heads) != 1:
        raise RuntimeError(
            f"deer_parallel_dnc.py expects exactly one installed "
            f"StochasticWriteHead on the model (num_layers=1, "
            f"share_memory_between_layers=True is the only configuration "
            f"this file supports); found {len(heads)}. Call "
            f"install_stochastic_write_heads(model) before using DEER."
        )
    return heads[0]


@contextlib.contextmanager
def _inject_fixed_noise(eps_source):
    """Temporarily replace `torch.randn_like` so that
    `StochasticWriteHead.forward`'s (stochastic_write_head_v2.py,
    UNMODIFIED) `eps = torch.randn_like(std)` call returns a
    pre-determined noise tensor instead of a fresh draw.

    Why this is necessary: DEER's Newton iteration evaluates the SAME
    per-timestep transition function many times over (once per Newton
    round, plus once more inside every round's `jacrev` diagonal-Jacobian
    computation) while solving for one fixed trajectory. If the write
    head's forward() sampled a NEW eps on every one of those evaluations,
    the "function" being solved for would be a different function on every
    call, and Newton's method would have no fixed target to converge to.
    Treating the write noise as part of the fixed, given external input at
    every timestep -- exactly how DEER/quasi-DEER treat "any (possibly
    random) input dependence" (Lim et al. 2024, Section 2, suppress it in
    their own notation; Gonzalez et al. 2025, Section 1, note their
    algorithms parallelize "any discrete-time nonlinear dynamical system
    ... that may or may not include stochasticity") -- keeps the
    linearized problem well posed and reproduces bit-for-bit what a
    sequential rollout using the same noise sequence would produce (see
    `deer_vs_sequential_max_abs_error` below).

    `eps_source` is a zero-argument callable returning the tensor to hand
    back; it is called once per `torch.randn_like` invocation inside the
    `with` block (in this file, exactly one call happens per invocation,
    from the write head -- nothing else in the `_layer_forward` call tree
    samples randomness: Mamba's own step is deterministic given its state,
    and DNC's content/allocation/temporal-link addressing is deterministic
    given the memory state). Restores the real `torch.randn_like` on exit,
    including on exception, so nothing about the rest of the process's
    randomness is disturbed once this context manager returns.
    """
    original_randn_like = torch.randn_like

    def _patched_randn_like(input_tensor, *args, **kwargs):
        noise = eps_source()
        if noise.shape != input_tensor.shape:
            raise RuntimeError(
                f"_inject_fixed_noise: injected noise shape "
                f"{tuple(noise.shape)} does not match the shape "
                f"torch.randn_like was called with "
                f"({tuple(input_tensor.shape)}) -- eps_seq's cell_size/"
                f"batch layout is out of sync with the write head's std "
                f"tensor."
            )
        return noise.to(dtype=input_tensor.dtype, device=input_tensor.device)

    torch.randn_like = _patched_randn_like
    try:
        yield
    finally:
        torch.randn_like = original_randn_like



class _DEERImplicitSolve(torch.autograd.Function):
    """Bridges the no-grad quasi-DEER forward solve back into autograd via
    implicit differentiation instead of unrolling backprop through the
    Newton iteration (see newton_associative_scan.py's deer_adjoint_scan).
    Memory no longer scales with deer_max_newton_iters / deer_jac_chunk_size
    / deer_jac_sample_batch_size -- those now only affect the forward solve,
    which builds no autograd graph at all.
    """

    @staticmethod
    def forward(ctx, per_sample_step, x_seq, init_state_vec, model, solver_kwargs):
        y, diag_jac, diagnostics = deer_quasi_newton_solve(
            per_sample_step, x_seq, init_state_vec, **solver_kwargs
        )
        ctx.save_for_backward(y, diag_jac, init_state_vec, x_seq)
        ctx.per_sample_step = per_sample_step
        ctx.model = model
        ctx.step_sample_batch_size = (
            solver_kwargs.get("step_sample_batch_size")
            or solver_kwargs.get("jac_sample_batch_size", 1)
        )
        model._last_deer_diagnostics = diagnostics
        return y

    @staticmethod
    def backward(ctx, grad_y):
        y, diag_jac, init_state_vec, x_seq = ctx.saved_tensors
        per_sample_step = ctx.per_sample_step
        model = ctx.model
        B, T, D = y.shape
        X = x_seq.shape[-1]

        mu = deer_adjoint_scan(diag_jac, grad_y)  # (B, T, D)

        x_seq_leaf = x_seq.detach().requires_grad_(True)
        params = [p for p in model.parameters() if p.requires_grad]
        need_init_grad = init_state_vec.requires_grad

        # Same reason the forward solve silences it: this re-evaluation
        # must not append a second, redundant bookkeeping entry.
        write_head = model._find_write_head()
        was_training_bw = write_head.training
        write_head.eval()
        try:
            with torch.enable_grad():
                y_prev = torch.cat(
                    [init_state_vec.unsqueeze(1), y[:, :-1, :]], dim=1
                )
                f_out = _batched_step_fn(
                    per_sample_step,
                    y_prev.reshape(B * T, D),
                    x_seq_leaf.reshape(B * T, X),
                    step_sample_batch_size=ctx.step_sample_batch_size,
                ).reshape(B, T, D)

                grad_targets = [x_seq_leaf] + params + (
                    [init_state_vec] if need_init_grad else []
                )
                grads = torch.autograd.grad(
                    f_out, grad_targets, grad_outputs=mu, allow_unused=True
                )
        finally:
            write_head.train(was_training_bw)

        grad_x_seq = grads[0]
        grad_params = grads[1 : 1 + len(params)]
        grad_init_state = grads[1 + len(params)] if need_init_grad else None

        for p, g in zip(params, grad_params):
            if g is not None:
                p.grad = g if p.grad is None else p.grad + g

        return None, grad_x_seq, grad_init_state, None, None


# ==========================================================================
# 1. DEERParallelDNC
# ==========================================================================
class DEERParallelDNC(MambaDNC):
    """`MambaDNC` (mamba_controller.py, Step 1) with an additional
    `use_deer=True` forward path that solves the whole-sequence recurrence
    in parallel via quasi-DEER Newton iteration
    (newton_associative_scan.py) instead of the inherited O(T) sequential
    Python loop. `use_deer=False` (the default) is byte-for-byte the
    inherited `MambaDNC.forward` -- i.e. `DEERParallelDNC` is a strict
    drop-in superset, the same "every new knob defaults to reproducing
    prior behavior" convention `Alter_PHASE3_mamba_step2.py` already
    established for `CONTROLLER_TYPE`/`CHUNK_SIZE`.

    Only supports `num_layers == 1` and `share_memory_between_layers ==
    True` -- the only configuration ever constructed in this program's
    training script; asserted in `__init__`.
    """

    def __init__(
        self,
        *args,
        deer_max_newton_iters: int = 50,
        deer_tol: Optional[float] = None,
        deer_damping: float = 0.0,
        deer_max_jac_diag_abs: Optional[float] = None,
        deer_jac_chunk_size: int = 1,  # v3/v4 -- see class docstring
        deer_jac_sample_batch_size: int = 1,  # v4 -- see class docstring
        deer_step_sample_batch_size: Optional[int] = None,  # v5 -- see class docstring
        deer_use_analytic_diag: bool = False,  # Alternate Phase 3, Step 2, Option 3
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if self.num_layers != 1:
            raise NotImplementedError(
                "DEERParallelDNC only supports num_layers=1 (every model "
                "constructed by this program's training script uses "
                "num_layers=1); extending the state-pytree packing in "
                "this file to a genuine multi-layer stack is "
                "unimplemented."
            )
        if not self.share_memory_between_layers:
            raise NotImplementedError(
                "DEERParallelDNC only supports share_memory_between_layers"
                "=True (the only setting this program's training script "
                "ever uses)."
            )
        if self.rnn_type.lower() != "mamba":
            raise NotImplementedError(
                "DEERParallelDNC requires rnn_type='mamba'. mamba_controller.py's "
                "MambaDNC state (a list of per-block (conv_state, ssm_state) "
                "tuples) is already batch-first at every leaf, which is what "
                "make_state_spec/pack_state (n_leading_dims=1) assume. For "
                "rnn_type in {'lstm','gru','rnn'}, dnc.DNC._init_hidden returns "
                "controller state shaped (num_hidden_layers, B, hidden) -- an "
                "extra leading dim before batch that would be silently "
                "mis-packed as if it were the batch dimension, corrupting the "
                "Newton solve without raising an error. Per the roadmap, "
                "MambaDNC is the only nonlinear model DEER is specified to "
                "linearize around; run with --controller mamba."
            )
        if not hasattr(self, "_layer_forward"):
            raise RuntimeError(
                "DEERParallelDNC requires the base dnc.DNC class to "
                "expose _layer_forward(input, layer, hx, "
                "pass_through_memory=True) -- see pytorch-dnc's "
                "dnc/dnc.py. This DNC build does not have it."
            )

        self.deer_max_newton_iters = deer_max_newton_iters
        self.deer_tol = deer_tol
        self.deer_damping = deer_damping
        self.deer_max_jac_diag_abs = deer_max_jac_diag_abs
        self.deer_jac_chunk_size = deer_jac_chunk_size  # v3
        self.deer_jac_sample_batch_size = deer_jac_sample_batch_size  # v4
        self.deer_step_sample_batch_size = deer_step_sample_batch_size  # v5
        self.deer_use_analytic_diag = deer_use_analytic_diag  # Option 3
        # Populated after every use_deer=True forward call: the
        # `newton_iters`/`final_max_abs_delta`/`converged`/`tol` dict
        # deer_quasi_newton_solve returns, for the training script to log.
        self._last_deer_diagnostics: Optional[dict] = None

    def _find_write_head(self) -> StochasticWriteHead:
        return _find_stochastic_write_head(self)

    def forward(
        self,
        input,
        hx=(None, None, None),
        reset_experience: bool = False,
        pass_through_memory: bool = True,
        use_deer: bool = False,
        eps_seq: Optional[torch.Tensor] = None,
    ):
        if not use_deer:
            # Byte-for-byte the inherited sequential path -- see class
            # docstring. eps_seq is silently ignored here (it's a
            # DEER-only override), matching this file's convention of
            # every new knob defaulting to reproducing prior behavior.
            return super().forward(input, hx, reset_experience, pass_through_memory)
        if not pass_through_memory:
            raise NotImplementedError(
                "DEER wiring assumes pass_through_memory=True (the only "
                "setting this program's training script ever uses)."
            )
        return self._forward_deer(input, hx, reset_experience, eps_seq)

    # ---------------------------------------------------------------- #
    def _forward_deer(self, input, hx, reset_experience, eps_seq):
        if not self.batch_first:
            raise NotImplementedError(
                "DEERParallelDNC assumes batch_first=True (the only mode "
                "this program's training script ever constructs models "
                "with)."
            )
        B, T, _ = input.shape
        device, dtype = input.device, input.dtype

        write_head = self._find_write_head()
        cell_size = write_head.mu_transform.out_features

        # ---- initial state s_0 (fixed, given -- see deer_quasi_newton_
        # solve's docstring) -------------------------------------------
        chx0_all, mhx0_all, last_read0 = self._init_hidden(hx, B, reset_experience)
        chx0 = chx0_all[0]   # single layer (num_layers == 1, asserted above)
        mhx0 = mhx0_all[0]   # single shared memory (share_memory_between_layers)
        # `output` has no meaning before the first controller call -- it
        # is a pure "carried for later readout" component of the state
        # (see class/module docstring point 1), so its initial value is
        # never consumed by the recurrence; zero is an arbitrary but
        # harmless placeholder.
        output0 = torch.zeros(B, self.output_size, device=device, dtype=dtype)

        example_state = (chx0, mhx0, last_read0, output0)
        spec = make_state_spec(example_state, n_leading_dims=1)
        init_state_vec = pack_state(example_state, spec)   # (B, D)

        # ---- pre-sample the write noise for the whole sequence --------
        # See _inject_fixed_noise's docstring for why this must be fixed
        # up front rather than sampled fresh inside per_sample_step.
        if eps_seq is None:
            eps_seq = torch.randn(B, T, cell_size, device=device, dtype=dtype)
        elif eps_seq.shape != (B, T, cell_size):
            raise ValueError(
                f"_forward_deer: eps_seq shape {tuple(eps_seq.shape)} does "
                f"not match the expected (B, T, cell_size) = "
                f"({B}, {T}, {cell_size})."
            )

        x_seq = torch.cat([input, eps_seq], dim=-1)   # (B, T, input_size + cell_size)
        raw_input_dim = self.input_size

        model = self  # avoid shadowing inside the closure below
        analytic_diag_fn = (
            build_analytic_diag_fn(model, spec, raw_input_dim, _inject_fixed_noise)
            if self.deer_use_analytic_diag
            else None
        )

        def per_sample_step(state_vec, x_vec_aug):
            x_vec = x_vec_aug[:raw_input_dim]
            eps_vec = x_vec_aug[raw_input_dim:]

            chx, mhx, last_read, _prev_output = unpack_state(state_vec, spec)
            chx_b = _tree_unsqueeze0(chx)
            mhx_b = _tree_unsqueeze0(mhx)
            last_read_b = last_read.unsqueeze(0)
            x_b = x_vec.unsqueeze(0)

            # `_layer_forward` is the STOCK pytorch-dnc method: it does not
            # concatenate the read vector into the controller input itself
            # (the caller must do that, exactly as `dnc.DNC.forward`'s own
            # per-timestep loop does via `torch.cat([input[time], last_read],
            # 1)` before calling `_layer_forward`), and -- contrary to the
            # note this comment used to make -- its `hx` argument is a
            # 3-tuple `(chx, mhx, _)`, not a plain `(chx, mhx)` pair: the
            # installed pytorch-dnc's `_layer_forward` unconditionally does
            # `(chx, mhx, _) = hx` (dnc/dnc.py), so calling it with only two
            # elements raises `ValueError: not enough values to unpack
            # (expected 3, got 2)`. The third slot is read positionally but
            # never used inside `_layer_forward` itself (the fresh read
            # vectors are computed from `mhx` and returned separately as
            # `read_vectors`), so any placeholder in that slot is fine --
            # `last_read_b` is passed for it purely for readability/
            # consistency with `dnc.DNC.forward`'s own call shape, not
            # because `_layer_forward` reads it. Build the concatenated
            # controller input here and pass/unpack accordingly (this is
            # what was missing before: the raw, un-concatenated `x_b` was
            # being fed straight to the controller, which expects
            # `input_size + read_vectors_size` features, not just
            # `input_size`).
            controller_input_b = torch.cat([x_b, last_read_b], dim=-1)

            # NOTE (2nd pass): `_layer_forward` returns a 2-tuple overall --
            # `output, (chx, mhx, read_vectors)` -- NOT three separate
            # top-level return values. The first attempt at this fix got
            # the *input* hx shape right but mis-unpacked the *return*
            # value as `out, (chx, mhx, _), last_read = ...` (4 things from
            # a 2-tuple), which is what produced the very same "not enough
            # values to unpack (expected 3, got 2)" error one call frame
            # further down. Unpack the inner 3-tuple explicitly instead.
            with _inject_fixed_noise(lambda: eps_vec.unsqueeze(0)):
                new_output_b, (new_chx_b, new_mhx_b, new_last_read_b) = model._layer_forward(
                    controller_input_b, 0, (chx_b, mhx_b, last_read_b)
                )

            new_state = (
                _tree_squeeze0(new_chx_b),
                _tree_squeeze0(new_mhx_b),
                new_last_read_b.squeeze(0),
                new_output_b.squeeze(0),
            )
            return pack_state(new_state, spec)

        # ---- run the Newton solve with write-head bookkeeping silenced
        # -----------------------------------------------------------------
        # Every Newton round (plus every jacrev call inside it) re-
        # evaluates per_sample_step for EVERY timestep -- leaving the
        # write head's bookkeeping on would append many spurious,
        # non-final entries into _kl_terms/_recent_writes per real write
        # timestep. .eval() only silences that bookkeeping -- it does NOT
        # change what v is sampled as (that depends only on self.sample,
        # never self.training), so the dynamics being solved for are
        # unaffected. See module docstring point 4.
        was_training = write_head.training
        write_head.eval()
        solver_kwargs = dict(
            max_newton_iters=self.deer_max_newton_iters,
            tol=self.deer_tol,
            damping=self.deer_damping,
            max_jac_diag_abs=self.deer_max_jac_diag_abs,
            analytic_diag_fn=analytic_diag_fn,  # Option 3 -- None unless deer_use_analytic_diag
            jac_chunk_size=self.deer_jac_chunk_size,  # v3 -- ignored when analytic_diag_fn is set
            jac_sample_batch_size=self.deer_jac_sample_batch_size,  # v4
            step_sample_batch_size=self.deer_step_sample_batch_size,  # v5
        )
        try:
            y = _DEERImplicitSolve.apply(
                per_sample_step, x_seq, init_state_vec, model, solver_kwargs
            )
        finally:
            write_head.train(was_training)

        diagnostics = self._last_deer_diagnostics

        # ---- parallel output reconstruction ----------------------------
        # output/last_read live inside the converged trajectory `y`
        # itself (state point 1/3) -- no sequential replay needed here.
        _, _, last_read_traj, output_traj = unpack_state(y, spec)   # (B, T, ...)
        final_output = self.output(torch.cat([output_traj, last_read_traj], dim=-1))

        # ---- one vectorized bookkeeping pass over the write head -------
        # Recomputes mu/logvar/v from the (already-converged) output_traj
        # with the SAME eps already baked into x_seq, so the recorded v
        # is bit-identical to what the converged trajectory actually used
        # (deterministic function, identical inputs). Flattened to
        # (B*T, cell_size) -- the same 2D (N, cell_size) shape convention
        # every per-timestep call already produces, so pop_kl()/
        # update_prior_snapshot()'s later torch.cat(dim=0) sees a
        # consistent shape regardless of whether it came from this one
        # call or from T separate per-timestep calls. write_head.training
        # is already restored to `was_training` at this point, so
        # forward()'s own `if self.training and self.sample:` guard
        # reproduces exactly the bookkeeping a real sequential pass in the
        # caller's actual mode would have produced.
        output_flat = output_traj.reshape(B * T, self.output_size)
        eps_flat = eps_seq.reshape(B * T, cell_size)
        with _inject_fixed_noise(lambda: eps_flat):
            write_head(output_flat)

        # ---- updated hx for the caller (chaining / multi-episode calls) -
        chx_final, mhx_final, last_read_final, _ = unpack_state(y[:, -1, :], spec)
        new_hx = ([chx_final], [mhx_final], last_read_final)

        return final_output, new_hx


# ==========================================================================
# 2. Correctness self-check against the sequential rollout
# ==========================================================================
def _sequential_reference_forward(
    model,
    input_seq: torch.Tensor,
    hx=(None, None, None),
    reset_experience: bool = True,
    eps_seq: Optional[torch.Tensor] = None,
):
    """Ordinary O(T) sequential rollout through the SAME `_layer_forward`
    code path `DEERParallelDNC._forward_deer`'s `per_sample_step` wraps,
    used only as the ground-truth reference for
    `deer_vs_sequential_max_abs_error`. Never used in the training loop --
    DEER's whole point is to avoid this loop. Works on any `MambaDNC`
    instance (including a plain `DEERParallelDNC` with use_deer=False, or
    `MambaDNC` itself), not just through the DEER wrapper's own forward.
    """
    if not model.batch_first:
        raise NotImplementedError("assumes batch_first=True")
    if model.num_layers != 1 or not model.share_memory_between_layers:
        raise NotImplementedError(
            "assumes num_layers=1, share_memory_between_layers=True"
        )

    B, T, _ = input_seq.shape
    device, dtype = input_seq.device, input_seq.dtype
    write_head = _find_stochastic_write_head(model)
    cell_size = write_head.mu_transform.out_features

    chx_all, mhx_all, last_read = model._init_hidden(hx, B, reset_experience)
    chx, mhx = chx_all[0], mhx_all[0]

    if eps_seq is None:
        eps_seq = torch.randn(B, T, cell_size, device=device, dtype=dtype)

    outputs = []
    for t in range(T):
        # Same fix as `per_sample_step` above: `_layer_forward` needs the
        # read vector concatenated into its `input` argument (it does not
        # do this itself), and its `hx` is a 3-tuple `(chx, mhx, _)` --
        # NOT a plain `(chx, mhx)` pair (see the longer explanation at the
        # `per_sample_step` call site above) -- with the new read vector
        # coming back as a separate third return value rather than folded
        # into `hx`.
        # `_layer_forward` returns a 2-tuple overall -- `output,
        # (chx, mhx, read_vectors)` -- so unpack the inner 3-tuple
        # explicitly rather than treating it as three top-level returns
        # (see the longer note at the `per_sample_step` call site above).
        controller_input = torch.cat([input_seq[:, t, :], last_read], dim=-1)
        with _inject_fixed_noise(lambda t=t: eps_seq[:, t, :]):
            out_t, (chx, mhx, last_read) = model._layer_forward(
                controller_input, 0, (chx, mhx, last_read)
            )
        outputs.append(model.output(torch.cat([out_t, last_read], dim=-1)))

    final_output = torch.stack(outputs, dim=1)
    new_hx = ([chx], [mhx], last_read)
    return final_output, new_hx, eps_seq


def deer_vs_sequential_max_abs_error(
    model: DEERParallelDNC,
    input_seq: torch.Tensor,
    hx=(None, None, None),
    reset_experience: bool = True,
    seed: Optional[int] = None,
    max_batch_size: int = 1,
) -> dict:
    """Correctness gate for this option, per the roadmap: "run DEER at
    enough Newton rounds to fully converge and confirm it reproduces
    MambaDNC's sequential rollout on a fixed batch/seed to near machine
    precision" -- the same max-abs-error style check dnc_parallel_scan.py
    used for its own scan primitive, and the same convergence-tolerance
    reasoning newton_associative_scan.py's own module docstring lays out
    (1e-4 float32 / 1e-7 float64, following Lim et al.'s reference
    implementation). Both rollouts are driven with the identical
    `eps_seq` so the write head's sampled noise cannot itself be the
    source of any discrepancy -- see `_inject_fixed_noise`'s docstring.

    v2: `seed` (when given) only determines `eps_seq` for THIS call -- the
    global RNG state is saved before seeding and restored afterward, so
    calling this function does not perturb whatever random stream the
    caller is mid-way through (e.g. the training loop's RNG, if this is
    called once at startup as Alter_PHASE3_mamba_step2.py does). Without
    this, `torch.manual_seed(seed)` would rewind the caller's RNG back to
    the same point `run()` seeded it at, silently repeating early
    post-seed draws for whatever comes after this check.

    Returns a dict: `max_abs_error` (float, the quantity to compare
    against the tolerance), `newton_diagnostics` (the
    `deer_quasi_newton_solve` diagnostics dict from the DEER call, so a
    large error can be cross-checked against whether Newton actually
    reported `converged=True`), and `eps_seq` (so the same noise can be
    reused for a follow-up call, e.g. at a tighter `deer_tol`).
    """
    cpu_rng_state = None
    cuda_rng_state = None
    if seed is not None:
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        torch.manual_seed(seed)

    try:
        # Wrap the whole diagnostic check in torch.no_grad() to eliminate 
        # unwanted backward graph retention across Newton iterations.
        with torch.no_grad():
            if max_batch_size is not None and max_batch_size > 0:
                input_seq = input_seq[: max_batch_size]
                if hx[0] is not None:
                    hx = (
                        [h[: max_batch_size] for h in hx[0]],
                        [h[: max_batch_size] for h in hx[1]],
                        hx[2][: max_batch_size] if hx[2] is not None else None,
                    )

            B, T, _ = input_seq.shape
            write_head = _find_stochastic_write_head(model)
            cell_size = write_head.mu_transform.out_features
            eps_seq = torch.randn(B, T, cell_size, device=input_seq.device, dtype=input_seq.dtype)

            seq_output, _seq_hx, _ = _sequential_reference_forward(
                model, input_seq, hx=hx, reset_experience=reset_experience, eps_seq=eps_seq
            )
            # Move the reference off-GPU and drop seq-side state before DEER:
            # JVP-based diag_jac needs headroom on ~16 GiB GPUs and the seq pass
            # plus (B,T,D) trajectory buffers already consume most of it.
            seq_output_cpu = seq_output.detach().cpu()
            del seq_output, _seq_hx
            if input_seq.is_cuda:
                torch.cuda.empty_cache()
            deer_output, _deer_hx = model(
                input_seq, hx, reset_experience, use_deer=True, eps_seq=eps_seq
            )
            del _deer_hx
            max_abs_error = (seq_output_cpu - deer_output.detach().cpu()).abs().max().item()
            del deer_output
            return {
                "max_abs_error": max_abs_error,
                "newton_diagnostics": model._last_deer_diagnostics,
                "eps_seq": eps_seq,
            }
    finally:
        if cpu_rng_state is not None:
            torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)