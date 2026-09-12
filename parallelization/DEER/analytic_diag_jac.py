"""
analytic_diag_jac.py -- v1 (new file)

Alternate Phase 3, Step 2, Option 3: replaces newton_associative_scan.py's
brute-force per-element JVP loop (`_diag_jac_via_jvp`) with a closed-form,
architecture-aware diagonal Jacobian for one MambaDNC step, per Gonzalez et
al. 2025 Section 4.1 / Appendix B.1.3's own recommendation ("derive the
diagonal entries of the Jacobian for the architecture of interest" instead
of looping per-element), and Farsang et al. 2025's general thesis (build/
derive the diagonal directly rather than approximate it via autodiff).

--------------------------------------------------------------------------
Why this is necessary, not just an optimization
--------------------------------------------------------------------------
This run's packed per-timestep state D is dominated by DNC's own memory
tensors (link_matrix alone is nr_cells^2 = 65,536 elements at this run's
config: nr_cells=256, cell_size=192). `_diag_jac_via_jvp` needs D separate
forward/backward evaluations of the ENTIRE per-timestep step (Mamba
controller + Memory.forward) to build the diagonal -- i.e. it costs
O(D) times the cost of ONE ordinary step, and one ordinary step itself
touches O(D) work (it writes the whole memory matrix), so the total cost
is ~O(D^2) per (batch, time) sample per Newton round. That is what
produced the observed hang (stuck mid-way through iteration 1's
diagonal-Jacobian pass, not stuck iterating Newton rounds -- the
pre-training correctness check earlier in the same log converged in 6
rounds just fine).

Both DEER's damped/undamped Newton iteration AND Gonzalez et al.'s
ELK/quasi-ELK need EXACTLY this same diagonal Jacobian at EXACTLY this
same asymptotic cost (Gonzalez et al., Table 1: quasi-DEER and quasi-ELK
are both O(T*D) per round -- ELK only changes how the linear solve inside
one round is stabilized, not how the diagonal Jacobian itself is
computed). So switching to (quasi-)ELK on top of the current jvp-loop
diagonal-Jacobian would NOT fix this hang -- it would still pay for the
same O(D) jvp loop per round, just with a Kalman-filter linear solve
layered on top instead of the plain affine scan. This file is the actual
fix; `deer_quasi_newton_solve`'s existing `damping` knob (a "scale-ELK"
style shrink, per that paper's own Appendix A.4) is the cheap stabilizer
to reach for LATER, only if Newton itself needs it once this is in place.

--------------------------------------------------------------------------
The derivation -- why every leaf of MambaDNC's state has a diagonal (or
exactly-zero) self-Jacobian, no autodiff required
--------------------------------------------------------------------------
Quasi-DEER only keeps d(state_t[i])/d(state_{t-1}[i]) for i indexing THE
SAME flattened position -- cross-leaf terms (e.g. d(read_weights_t)/
d(memory_{t-1})) are off this diagonal by construction, regardless of
their true value, because read_weights and memory occupy disjoint index
ranges in the packed state vector. So each leaf only needs its OWN
self-derivative:

  mamba controller state (chx = list of (conv_state, ssm_state), one pair
  per stacked block -- see mamba_controller.py, MambaControllerCell.step):
    - conv_state is a pure ROLL:
        new_conv_state = torch.cat([conv_state[:, :, 1:], x.unsqueeze(-1)], dim=-1)
      new_conv_state[...,k] = conv_state[...,k+1] for k < d_conv-1, and the
      last column comes from x (not from conv_state at all) -- so
      d(new_conv_state[...,k])/d(conv_state[...,k]) is EXACTLY 0 for every
      k, by construction, not an approximation.
    - ssm_state:
        new_ssm_state = ssm_state * dA + x_conv.unsqueeze(-1) * dB
      is EXACTLY elementwise in ssm_state, and dA = exp(einsum("bd,dn->bdn",
      dt, A)) does not depend on ssm_state_{t-1} at all (dt is a function
      of x_t and conv_state only) -- so d(new_ssm_state)/d(ssm_state) = dA
      EXACTLY. `MambaControllerCell.step` already computes dA every call;
      this file just reads it back out (stashed on the cell instance --
      see the one-line addition to mamba_controller.py below) instead of
      re-deriving it via autodiff.

  DNC memory state (mhx) -- standard Graves et al. 2016 equations, which
  pytorch-dnc's `dnc.memory.Memory` is a direct port of:
    - memory M_t[i,j] = M_{t-1}[i,j] * (1 - w_t[i]*e_t[j]) + w_t[i]*v_t[j]
      => diag = 1 - w_t[i]*e_t[j]           (elementwise in M, exact)
    - usage u_t[i] = (u_{t-1}[i] + w_t[i] - u_{t-1}[i]*w_t[i]) * psi_t[i]
      => diag = (1 - w_t[i]) * psi_t[i]     (elementwise in u, exact)
    - precedence p_t[i] = (1 - sum_j w_t[j]) * p_{t-1}[i] + w_t[i]
      => diag = (1 - sum_j w_t[j])          (same scalar for every i, exact)
    - link matrix L_t[i,j] = (1 - w_t[i] - w_t[j]) * L_{t-1}[i,j]
                              + w_t[i]*p_{t-1}[j]   (i != j; L_t[i,i] == 0)
      => diag = 1 - w_t[i] - w_t[j]          (elementwise in L, exact)
    - write_weights_t: recomputed FRESH every step from content/allocation
      addressing against M_{t-1} and the current interface vector -- no
      term of the form "w_{t-1}[i] appears in the formula for w_t[i]"
      exists, so diag = 0 EXACTLY (not approximated away -- it genuinely
      isn't there).
    - read_weights_t[i] = pi1*backward_i + pi2*content_i + pi3*forward_i,
      forward_i = (L_t @ read_weights_{t-1})[i], backward_i =
      (L_t^T @ read_weights_{t-1})[i] -- the coefficient of
      read_weights_{t-1}[i] in either term is L_t[i,i], which the
      link-matrix update above sets to exactly 0 every step (DNC never
      links a memory cell to itself) => diag = 0 EXACTLY.
    - last_read / output: both are recomputed FRESH each step from the
      CURRENT M_t/read_weights_t (last_read) or the current controller
      call (output) -- neither has a term referencing its own previous
      value => diag = 0 EXACTLY.

The only genuinely approximate pieces below are `e_t` (erase vector, used
only in the M diagonal) and `psi_t` (retention vector, used only in the u
diagonal): both require interface-vector sub-transforms internal to
`dnc.memory.Memory`, whose exact attribute names could not be confirmed
here (that package is a pip dependency -- `from dnc.memory import Memory`,
mamba_controller.py -- not vendored in this repo, and not installed in the
environment this file was written in). Per Gonzalez et al.'s own
Proposition-1 corollary ("the fixed-point iterations will converge ... in
at most T iterations ... even if the Jacobians ... are replaced by
arbitrary matrices" -- ANY consistent replacement preserves Newton's
GLOBAL CONVERGENCE; it can only affect ITERATION COUNT, never
correctness), this file defaults e_t and psi_t to their theoretical maxima
(e_t=1, psi_t=1, i.e. "assume full erase, assume nothing gets freed") as a
safe, always-in-[0,1] placeholder. Correctness is unaffected --
`deer_vs_sequential_max_abs_error` (already in deer_parallel_dnc.py) still
checks exact agreement with the sequential rollout regardless of this
simplification -- only convergence SPEED is at stake. If Newton needs
close to `deer_max_newton_iters` rounds after switching to this file, wire
in the real e_t/psi_t (search "erase vector defaulted" / "retention vector
defaulted" below) rather than reverting to the JVP loop.

Any `mhx` key this file doesn't recognize defaults to an all-zeros
diagonal (always valid, per the corollary above) and prints a ONE-TIME
warning naming the key, instead of crashing or silently mishandling it.
If `mhx` turns out not to be a dict at all (some pytorch-dnc versions may
differ), this raises immediately with a clear message -- run:

    chx, mhx, last_read = model._init_hidden((None, None, None), 1, True)
    print(type(mhx[0]), mhx[0].keys() if isinstance(mhx[0], dict) else mhx[0])

and send me the output so the key names below can be corrected.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from DEER.newton_associative_scan import pack_state, unpack_state

__all__ = ["build_analytic_diag_fn"]


# ==========================================================================
# Small tree helpers, duplicated (not imported) from deer_parallel_dnc.py
# to avoid a circular import (deer_parallel_dnc.py imports THIS file).
# ==========================================================================
def _tree_unsqueeze0(tree: Any) -> Any:
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


def _tree_squeeze0(tree: Any) -> Any:
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


_WARNED_UNKNOWN_MHX_KEYS: set[str] = set()


def _dnc_memory_diag(new_mhx: Any) -> Any:
    """Closed-form diagonal of d(mhx_t)/d(mhx_{t-1}); see module docstring
    for the derivation of every branch. `new_mhx` is the (batch-of-1)
    dict `model._layer_forward` just returned as the updated memory
    state."""
    if not isinstance(new_mhx, dict):
        raise TypeError(
            "_dnc_memory_diag assumes dnc.memory.Memory's hidden state is "
            f"a dict (pytorch-dnc's usual convention); got {type(new_mhx)} "
            "instead. See this module's docstring for a one-line snippet "
            "to print the real structure, then adjust this function."
        )

    try:
        w = new_mhx["write_weights"]  # (B, num_writes=1, N) -- verified at
        # runtime: write_weights is (1, 1, 256), i.e. the singleton
        # "num_writes" axis sits at dim=1 (DeepMind's own DNC convention,
        # matching read_weights' (B, read_heads, N) layout), NOT at dim=-1.
    except KeyError as e:
        raise KeyError(
            "_dnc_memory_diag: expected a 'write_weights' key in mhx; "
            f"found keys {list(new_mhx.keys())} instead. Rename the "
            "lookup in this function to match your installed `dnc` "
            "package's actual key for the write weighting."
        ) from e
    if w.dim() == 3:
        # Squeeze whichever axis is actually the singleton one, rather than
        # assuming a fixed position -- dim=1 is what this run's pytorch-dnc
        # build uses; dim=-1 is kept as a fallback for other ports/versions.
        if w.shape[1] == 1:
            w = w.squeeze(1)    # (B, 1, N) -> (B, N)
        elif w.shape[-1] == 1:
            w = w.squeeze(-1)   # (B, N, 1) -> (B, N)
        else:
            raise RuntimeError(
                f"_dnc_memory_diag: write_weights has shape {tuple(w.shape)} "
                "with no singleton axis at dim=1 or dim=-1 to squeeze -- "
                "unrecognized layout, adjust this function."
            )
    elif w.dim() != 2:
        raise RuntimeError(
            f"_dnc_memory_diag: expected write_weights to be 2-D or 3-D; "
            f"got shape {tuple(w.shape)}."
        )

    def _match_ndim(coeff: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Insert singleton dims into `coeff` right after its batch dim
        (dim=1) until it has as many dims as `target`, then broadcast-
        expand to target's exact shape. Needed because pytorch-dnc's
        `dnc.memory.Memory` keeps some hidden-state tensors (`memory`,
        and likely `link_matrix`/`precedence_weighting`) with an extra
        singleton axis (a leftover "num_writes" axis of size 1) that a
        directly-derived per-cell coefficient doesn't carry -- e.g.
        `memory` is (B, 1, N, W) here, not (B, N, W)."""
        while coeff.dim() < target.dim():
            coeff = coeff.unsqueeze(1)
        return coeff.expand_as(target).clone()
    
    diag: dict[str, torch.Tensor] = {}
    for key, tensor in new_mhx.items():
        if key == "memory":
            # M_t[i,j] = M_{t-1}[i,j]*(1 - w[i]*e[j]) + w[i]*v[j].
            # erase vector e_t defaulted to 1 (full erase) -- see module
            # docstring "genuinely approximate pieces".
            row_coeff = (1.0 - w).unsqueeze(-1)  # (B, N, 1) -- broadcasts over W
            diag[key] = _match_ndim(row_coeff, tensor)
        elif key == "usage_vector":
            # u_t[i] = (u_{t-1}[i] + w[i] - u_{t-1}[i]*w[i]) * psi[i].
            # retention vector psi_t defaulted to 1 -- see module docstring.
            diag[key] = _match_ndim((1.0 - w), tensor)
        elif key == "precedence":
            # p_t = (1 - sum(w)) * p_{t-1} + w  -- same scalar broadcast to
            # every cell index.
            scalar = 1.0 - w.sum(dim=-1, keepdim=True)  # (B, 1)
            diag[key] = _match_ndim(scalar, tensor)
        elif key == "link_matrix":
            # L_t[i,j] = (1 - w[i] - w[j]) * L_{t-1}[i,j] + w[i]*p_{t-1}[j]
            pair_coeff = 1.0 - w.unsqueeze(-1) - w.unsqueeze(-2)  # (B, N, N)
            diag[key] = _match_ndim(pair_coeff, tensor)
        elif key in ("write_weights", "read_weights"):
            # Recomputed fresh from addressing every step -- no self-term
            # (read_weights: the coefficient is L_t[i,i], which is always
            # exactly 0 -- DNC never links a cell to itself).
            diag[key] = torch.zeros_like(tensor)
        else:
            if key not in _WARNED_UNKNOWN_MHX_KEYS:
                _WARNED_UNKNOWN_MHX_KEYS.add(key)
                print(
                    f"[analytic_diag_jac] WARNING: mhx key {key!r} has no "
                    "closed-form diagonal wired up -- defaulting to zeros "
                    "(safe -- see Gonzalez et al. Prop. 1 corollary in the "
                    "module docstring -- but check whether this key "
                    "deserves a real formula in _dnc_memory_diag)."
                )
            diag[key] = torch.zeros_like(tensor)
    return diag


def build_analytic_diag_fn(
    model,
    spec: Any,
    raw_input_dim: int,
    inject_fixed_noise: Callable,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Returns `diag_fn(state_vec, x_vec_aug) -> diag_vec`, matching the
    (state_vec: (D,), x_vec: (X,)) -> (D,) contract
    `deer_quasi_newton_solve`'s `analytic_diag_fn` kwarg expects (same
    unbatched-per-sample contract as `per_sample_step` -- see
    newton_associative_scan.py). Costs exactly ONE extra `_layer_forward`
    call per (b,t) sample per Newton round -- the SAME cost as computing
    f_out itself -- not one call per state dimension like
    `_diag_jac_via_jvp`.

    `model`, `spec`, `raw_input_dim` and `inject_fixed_noise` are exactly
    the objects `deer_parallel_dnc.py`'s `_forward_deer` already has in
    scope when it builds `per_sample_step` -- see that call site.
    """

    def diag_fn(state_vec: torch.Tensor, x_vec_aug: torch.Tensor) -> torch.Tensor:
        x_vec = x_vec_aug[:raw_input_dim]
        eps_vec = x_vec_aug[raw_input_dim:]

        chx, mhx, last_read, _prev_output = unpack_state(state_vec, spec)
        chx_b = _tree_unsqueeze0(chx)
        mhx_b = _tree_unsqueeze0(mhx)
        last_read_b = last_read.unsqueeze(0)
        x_b = x_vec.unsqueeze(0)
        controller_input_b = torch.cat([x_b, last_read_b], dim=-1)

        with inject_fixed_noise(lambda: eps_vec.unsqueeze(0)):
            new_output_b, (new_chx_b, new_mhx_b, new_last_read_b) = model._layer_forward(
                controller_input_b, 0, (chx_b, mhx_b, last_read_b)
            )

        # ---- chx (Mamba controller) diagonal -- EXACT, no approximation --
        chx_diag = []
        for block, (conv_state, ssm_state) in zip(model.rnns[0].blocks, chx_b):
            conv_diag = torch.zeros_like(conv_state)  # pure roll -- see module docstring
            dA = getattr(block.cell, "_last_dA", None)
            ssm_diag = dA if dA is not None else torch.zeros_like(ssm_state)
            chx_diag.append((conv_diag, ssm_diag))

        # ---- mhx (DNC memory) diagonal -- standard DNC equations --------
        mhx_diag = _dnc_memory_diag(new_mhx_b)

        last_read_diag = torch.zeros_like(new_last_read_b)
        output_diag = torch.zeros_like(new_output_b)

        diag_state = (
            _tree_squeeze0(chx_diag),
            _tree_squeeze0(mhx_diag),
            last_read_diag.squeeze(0),
            output_diag.squeeze(0),
        )
        return pack_state(diag_state, spec)

    return diag_fn
