"""
newton_associative_scan.py -- v1 (new file)

Alternate Phase 3, Step 2, Option 3 (see Experiment-Roadmap.md, "Attempt to
modify DNC in a way that it can in one way or another simulate SSMs
parallelism" -> "3. Iterative fixed-point parallelization of the full
nonlinear recurrence"): a generic, architecture-agnostic quasi-DEER
Newton-iteration engine.

This file contains ONLY the generic numerical machinery -- it has no
knowledge of DNC, Mamba, or pytorch-dnc. It operates purely on:
  - a per-step transition function  f(state_vec, input_vec) -> new_state_vec
    (state_vec / new_state_vec are 1-D tensors of a fixed dimension D,
    input_vec is whatever fixed-shape 1-D tensor the caller's `f` expects),
  - a generic "state pytree" pack/unpack utility, so the caller never has
    to flatten/reassemble its own nested (dict/list/tuple/Tensor) state by
    hand.

DNC-specific wiring (building the transition function out of
`MambaDNC._layer_forward`, handling the stochastic write head, the
correctness self-check against the sequential rollout, etc.) lives in the
sibling file `deer_parallel_dnc.py`, which imports this module. Splitting
the two apart keeps this file reusable and independently testable, and
keeps `deer_parallel_dnc.py` free of low-level tensor-plumbing detail.

--------------------------------------------------------------------------
Why "quasi"-DEER, and why that specific choice is not a shortcut here --
it is the ONLY choice consistent with reusing dnc_parallel_scan.py as-is
--------------------------------------------------------------------------
Lim et al. 2024 ("Parallelizing Non-linear Sequential Models over the
Sequence Length") parallelize a nonlinear recurrence s_t = f(s_{t-1}, x_t)
by running Newton's method on the whole-sequence fixed-point equation,
where each Newton iteration solves a LINEAR recurrence
    Delta_s_t = A_t @ Delta_s_{t-1} + b_t ,   A_t = df/ds_{t-1}
via a parallel (associative) scan. In their original formulation A_t is
the FULL, dense Jacobian, and the scan's combine operator is therefore
MATRIX multiplication: combine((A1,B1),(A2,B2)) = (A2 @ A1, A2 @ B1 + B2).

Gonzalez et al. 2025 ("Towards Scalable and Stable Parallelization of
Nonlinear RNNs") introduce "quasi-DEER": replace A_t with diag(df/ds_{t-1})
(just the diagonal of the Jacobian). For a DIAGONAL matrix, matrix
multiplication reduces EXACTLY to elementwise (Hadamard) multiplication --
diag(A2) * diag(A1) == diag(A2 @ A1) whenever A1, A2 are both diagonal --
so quasi-DEER's linear solve is an elementwise-affine associative scan,
which is EXACTLY the operator `dnc_parallel_scan.py`'s `associative_scan_affine`
/ `selective_scan_chunk` already implement (that file was written for
Mamba's own diagonal S6 recurrence, h_t = a_t*h_{t-1} + b_t elementwise).

This is not an incidental simplification: it is the precise reason
Experiment-Roadmap.md instructs "keep `associative_scan_affine` ... as-is
-- it becomes the linear solver called once per Newton round, not
something you need to rebuild" for this exact option. Reusing that file
UNCHANGED (as instructed) is only mathematically valid for the diagonal
(quasi-DEER) case; a full dense-Jacobian DEER would need a genuinely
different (matrix-valued) combine operator and is NOT implemented here.

Cost profile (Gonzalez et al., Table 1): O(T*D) memory and work per Newton
round for quasi-DEER, vs. O(T*D^2) / O(T*D^3) for full DEER -- an
additional, independent reason this is the right default at the (small)
memory-matrix sizes this program's experiments use.

Diagonal-Jacobian computation: for simplicity and correctness we take the
"standard, not-especially-memory-efficient" route Gonzalez et al. describe
in their own Appendix B.1.3 -- compute the full per-sample Jacobian via
`torch.func.jacrev` and then take its diagonal -- rather than hand-deriving
a closed-form diagonal for this specific architecture. This costs the same
asymptotic compute as computing the full Jacobian (no work saved on the
forward differentiation itself), but is exactly what that paper says is
fine "for experiments where memory capacity is not a problem" -- which
matches this program's small memory-matrix sizes (nr_cells ~ 5-10).

--------------------------------------------------------------------------
Numerical safeguard: damping / trust-region fallback
--------------------------------------------------------------------------
Experiment-Roadmap.md explicitly flags the risk this file's Newton solver
has to guard against: "DNC's cosine-similarity addressing has flat,
low-gradient regions ... a poorly-conditioned Jacobian there could make
Newton's method converge slowly or unstably." Two independent, cheap
safeguards are implemented, both directly justified by material already in
this program's corpus (not invented ad hoc):

  1. `damping` (float in [0, 1)): a "scale-ELK" style shrink of the
     diagonal-Jacobian entries used in the LINEAR SOLVE only (never of the
     nonlinear residual itself), a_t <- (1-damping) * diag(df/ds_{t-1}).
     This is the cheap, matrix-free variant of the Levenberg-Marquardt /
     Kalman-filter trust region Gonzalez et al. derive in their Appendix
     A.4 ("Scale-ELK"): they show shrinking the transition matrix used in
     the linear recurrence attenuates its eigenvalues by exactly the
     (1-damping) factor, trading Newton's quadratic convergence rate for
     stability when the true Jacobian is large/ill-conditioned. damping=0
     recovers plain quasi-DEER exactly (their own "for lambda=0, ELK
     specializes to DEER" remark, adapted to the diagonal/scale case).
  2. `max_jac_diag_abs` (optional float): a hard elementwise clamp on the
     diagonal-Jacobian magnitude before it enters the scan, as an
     additional, independent numerical floor/ceiling against outright
     blow-up (e.g. from a near-singular addressing softmax) that a fixed
     multiplicative shrink alone would not catch.

Neither mechanism is enabled by default (damping=0.0, max_jac_diag_abs=None
-> exact quasi-DEER); both are op-in per the `deer_quasi_newton_solve`
call, so the plain, undamped algorithm is always what runs unless the
caller has evidence it needs the safety valve.

--------------------------------------------------------------------------
Convergence criterion
--------------------------------------------------------------------------
Matches Lim et al.'s own reference implementation (their Appendix B.1
`deer_iteration`, reproduced in their paper): stop when
`max(abs(y_new - y_old))` drops below a tolerance, defaulting to 1e-4 for
float32 and 1e-7 for float64 (their own stated defaults, and their
Appendix C.1 shows the exact tolerance value barely affects iteration
count as long as it isn't too close to the dtype's own numerical-precision
floor) -- not a fixed iteration count, per Experiment-Roadmap.md's explicit
instruction ("a residual-based convergence check (not a fixed round
count)").
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch.func import jacrev, jvp, vmap

from dnc_parallel_scan import selective_scan_chunk

__all__ = [
    "LeafSpec",
    "make_state_spec",
    "pack_state",
    "unpack_state",
    "deer_quasi_newton_solve",
    "deer_adjoint_scan",
]


# ==========================================================================
# 1. Generic, architecture-agnostic state-pytree pack/unpack
# ==========================================================================
class LeafSpec:
    """Records one Tensor leaf's per-example shape (i.e. its shape with all
    leading "batch-like" dimensions stripped off). Built once from a real
    example state (see `make_state_spec`), then reused to pack/unpack every
    other state with the identical structure without re-deriving shapes."""

    __slots__ = ("shape", "numel")

    def __init__(self, shape: tuple[int, ...]):
        self.shape = tuple(int(s) for s in shape)
        n = 1
        for s in self.shape:
            n *= s
        self.numel = n


def make_state_spec(example_tree: Any, n_leading_dims: int = 1) -> Any:
    """Build a pack/unpack spec from one real example state pytree.

    `example_tree` may be an arbitrary nesting of dict / list / tuple /
    torch.Tensor / None -- e.g. a DNC layer's `(chx, mhx, last_read,
    output)` tuple, whatever its internal structure actually is. This
    function does not need to know that structure in advance: it walks
    whatever it is given. `n_leading_dims` is how many of the example
    tensors' leading dimensions are "batch-like" (not part of the
    per-example shape) -- normally 1 (a plain batch dimension), matching
    how the example state is captured (see deer_parallel_dnc.py).
    """
    if example_tree is None:
        return None
    if isinstance(example_tree, torch.Tensor):
        if example_tree.dim() < n_leading_dims:
            raise ValueError(
                f"make_state_spec: tensor of shape {tuple(example_tree.shape)} "
                f"has fewer dims than n_leading_dims={n_leading_dims}"
            )
        return LeafSpec(example_tree.shape[n_leading_dims:])
    if isinstance(example_tree, dict):
        return {k: make_state_spec(example_tree[k], n_leading_dims) for k in example_tree}
    if isinstance(example_tree, (list, tuple)):
        made = [make_state_spec(v, n_leading_dims) for v in example_tree]
        return tuple(made) if isinstance(example_tree, tuple) else made
    raise TypeError(
        f"make_state_spec: unsupported leaf type {type(example_tree)!r}; "
        "state pytrees may only contain dict/list/tuple/Tensor/None."
    )


def _spec_leaves_in_order(spec: Any) -> list[LeafSpec]:
    """Deterministic left-to-right traversal of `spec`, yielding its
    LeafSpec leaves in the SAME order `_collect_tensors` below walks the
    corresponding real tree -- this shared ordering is what makes pack/
    unpack mutually inverse."""
    if spec is None:
        return []
    if isinstance(spec, LeafSpec):
        return [spec]
    if isinstance(spec, dict):
        out: list[LeafSpec] = []
        for k in spec:
            out.extend(_spec_leaves_in_order(spec[k]))
        return out
    if isinstance(spec, (list, tuple)):
        out = []
        for v in spec:
            out.extend(_spec_leaves_in_order(v))
        return out
    raise TypeError(f"corrupt spec: unexpected node type {type(spec)!r}")


def _collect_tensors_in_order(tree: Any) -> list[torch.Tensor]:
    """Same traversal order as `_spec_leaves_in_order`, applied to a REAL
    tree of tensors instead of a spec of shapes."""
    if tree is None:
        return []
    if isinstance(tree, torch.Tensor):
        return [tree]
    if isinstance(tree, dict):
        out: list[torch.Tensor] = []
        for k in tree:
            out.extend(_collect_tensors_in_order(tree[k]))
        return out
    if isinstance(tree, (list, tuple)):
        out = []
        for v in tree:
            out.extend(_collect_tensors_in_order(v))
        return out
    raise TypeError(
        f"pack_state: unsupported leaf type {type(tree)!r}; state pytrees "
        "may only contain dict/list/tuple/Tensor/None, and must match the "
        "structure `spec` was built from."
    )


def _rebuild_from_spec(spec: Any, leaves_iter) -> Any:
    if spec is None:
        return None
    if isinstance(spec, LeafSpec):
        return next(leaves_iter)
    if isinstance(spec, dict):
        return {k: _rebuild_from_spec(spec[k], leaves_iter) for k in spec}
    if isinstance(spec, (list, tuple)):
        rebuilt = [_rebuild_from_spec(v, leaves_iter) for v in spec]
        return tuple(rebuilt) if isinstance(spec, tuple) else rebuilt
    raise TypeError(f"corrupt spec: unexpected node type {type(spec)!r}")


def pack_state(tree: Any, spec: Any) -> torch.Tensor:
    """Flatten a state pytree (leaves shaped (*leading, *per_example_shape))
    into a single (*leading, D) tensor, D = sum of every leaf's `numel`
    (per `spec`). `leading` may be zero-, one-, or multi-dimensional (e.g.
    () for a single unbatched example inside a vmapped function, (B,) for
    one real timestep across a batch, or (B, T) for a whole trajectory) --
    inferred per-leaf from how many trailing dims `spec` says are "leaf"
    dims, so the same `spec` built once from a single-timestep example
    packs states of any leading shape.
    """
    tensors = _collect_tensors_in_order(tree)
    leaves = _spec_leaves_in_order(spec)
    if len(tensors) != len(leaves):
        raise ValueError(
            f"pack_state: tree has {len(tensors)} tensor leaves but spec "
            f"describes {len(leaves)}; tree does not match the structure "
            "`spec` was built from."
        )
    flat_parts = []
    leading_shape: Optional[tuple[int, ...]] = None
    for t, leaf in zip(tensors, leaves):
        k = len(leaf.shape)
        this_leading = tuple(t.shape[: t.dim() - k]) if k > 0 else tuple(t.shape)
        if leading_shape is None:
            leading_shape = this_leading
        elif this_leading != leading_shape:
            raise ValueError(
                "pack_state: inconsistent leading (batch/time) shape across "
                f"leaves -- got {this_leading} and {leading_shape}"
            )
        flat_parts.append(t.reshape(*this_leading, -1))
    if not flat_parts:
        return torch.zeros(*(leading_shape or ()), 0)
    return torch.cat(flat_parts, dim=-1)


def unpack_state(vec: torch.Tensor, spec: Any) -> Any:
    """Inverse of `pack_state`: reshape a (*leading, D) tensor back into
    the nested pytree `spec` describes. `leading` is inferred as
    `vec.shape[:-1]` -- works uniformly whether `vec` is unbatched (D,),
    batched (B, D), or a full trajectory (B, T, D)."""
    leading = vec.shape[:-1]
    leaves = _spec_leaves_in_order(spec)
    tensors = []
    offset = 0
    for leaf in leaves:
        n = leaf.numel
        chunk = vec[..., offset : offset + n].reshape(*leading, *leaf.shape)
        tensors.append(chunk)
        offset += n
    if offset != vec.shape[-1]:
        raise ValueError(
            f"unpack_state: spec describes {offset} total elements but "
            f"vec's last dim is {vec.shape[-1]}"
        )
    return _rebuild_from_spec(spec, iter(tensors))


# ==========================================================================
# 2. Quasi-DEER Newton solve
# ==========================================================================
def _default_tol(dtype: torch.dtype) -> float:
    if dtype == torch.float64:
        return 1e-7
    return 1e-4


def _diag_jac_via_jvp(
    per_sample_step: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    s: torch.Tensor,
    x: torch.Tensor,
    *,
    jac_chunk_size: int,
) -> torch.Tensor:
    """Diagonal of df/ds via JVPs.

    jac_chunk_size == 1 (default): one unit-direction JVP at a time -- lowest
    peak memory, safe on ~16 GiB GPUs with large Mamba+DNC states.

    jac_chunk_size > 1: vmap that many directions together (faster, but
    multiplies autodiff memory by jac_chunk_size; can OOM when nested inside
    an outer vmap over batch*time).
    """
    D = s.shape[-1]
    if jac_chunk_size <= 0:
        raise ValueError(f"jac_chunk_size must be positive; got {jac_chunk_size}")

    diag = s.new_empty(D)
    f = lambda ss: per_sample_step(ss, x)

    if jac_chunk_size == 1:
        e = torch.zeros_like(s)
        for i in range(D):
            e.zero_()
            e[i] = 1.0
            _, out = jvp(f, (s,), (e,))
            diag[i] = out[i]
        return diag

    for start in range(0, D, jac_chunk_size):
        print(f"  [diag_jac] round {start // jac_chunk_size + 1}/"
              f"{-(-D // jac_chunk_size)} (dims {start}:{end if 'end' in dir() else start+jac_chunk_size})", end="\r")
        end = min(start + jac_chunk_size, D)
        cs = end - start
        tangents = torch.zeros(cs, D, device=s.device, dtype=s.dtype)
        rows = torch.arange(cs, device=s.device)
        cols = torch.arange(start, end, device=s.device)
        tangents[rows, cols] = 1.0

        def _jvp_row(tangent: torch.Tensor) -> torch.Tensor:
            _, out = jvp(f, (s,), (tangent,))
            return out

        jvp_out = vmap(_jvp_row, in_dims=0)(tangents)  # (cs, D)
        diag[start:end] = jvp_out[rows, cols]
    return diag


def _batched_step_fn(
    per_sample_step: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    flat_prev: torch.Tensor,
    flat_x: torch.Tensor,
    *,
    step_sample_batch_size: int,
) -> torch.Tensor:
    """Evaluate per_sample_step over flat (N, D)/(N, X) rows without vmap over
    the full batch*time axis at once.

    A single vmap over all N = B*T pairs makes _layer_forward see an effective
    batch of N inside Mamba (each per_sample_step unsqueezes a batch-of-1 dim
    that vmap stacks), which dominates GPU memory on ~16 GiB cards long before
    the Jacobian pass starts.
    """
    if step_sample_batch_size <= 0:
        raise ValueError(
            f"step_sample_batch_size must be positive; got {step_sample_batch_size}"
        )

    n = flat_prev.shape[0]
    parts: list[torch.Tensor] = []
    batched_step = vmap(per_sample_step, in_dims=(0, 0))
    for start in range(0, n, step_sample_batch_size):
        end = min(start + step_sample_batch_size, n)
        chunk_prev = flat_prev[start:end]
        chunk_x = flat_x[start:end]
        if end - start == 1:
            parts.append(per_sample_step(chunk_prev[0], chunk_x[0]).unsqueeze(0))
        else:
            parts.append(batched_step(chunk_prev, chunk_x))
    return torch.cat(parts, dim=0)


def _batched_diag_jac_fn(
    diag_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    flat_prev: torch.Tensor,
    flat_x: torch.Tensor,
    *,
    jac_sample_batch_size: int,
) -> torch.Tensor:
    """Compute per-row diag(df/ds) without vmap over the full batch*time axis."""
    if jac_sample_batch_size <= 0:
        raise ValueError(
            f"jac_sample_batch_size must be positive; got {jac_sample_batch_size}"
        )

    n = flat_prev.shape[0]
    parts: list[torch.Tensor] = []
    for start in range(0, n, jac_sample_batch_size):
        end = min(start + jac_sample_batch_size, n)
        chunk_prev = flat_prev[start:end]
        chunk_x = flat_x[start:end]
        if end - start == 1:
            parts.append(diag_fn(chunk_prev[0], chunk_x[0]).unsqueeze(0))
        else:
            parts.append(vmap(diag_fn, in_dims=(0, 0))(chunk_prev, chunk_x))
    return torch.cat(parts, dim=0)


def deer_quasi_newton_solve(
    per_sample_step: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_seq: torch.Tensor,
    init_state_vec: torch.Tensor,
    *,
    max_newton_iters: int = 50,
    tol: Optional[float] = None,
    damping: float = 0.0,
    max_jac_diag_abs: Optional[float] = None,
    y_init_guess: Optional[torch.Tensor] = None,
    jac_chunk_size: int = 1,
    jac_sample_batch_size: int = 1,
    step_sample_batch_size: Optional[int] = None,
) -> tuple[torch.Tensor, dict]:
    """Solve, in parallel over the sequence length, the Markovian recurrence

        s_0 = init_state_vec[b]                       (fixed, given)
        s_t = per_sample_step(s_{t-1}, x_seq[b, t])    t = 1..T

    for every batch element b, via quasi-DEER (diagonal-Jacobian Newton
    iteration, see module docstring): each round linearizes
    `per_sample_step` around the current trajectory guess, solves the
    resulting elementwise-affine recurrence for the whole sequence at once
    via `dnc_parallel_scan.py`'s `selective_scan_chunk`, and updates the
    guess -- repeating until the trajectory stops changing (or the round
    budget is exhausted).

    Args:
        per_sample_step: (state_vec: (D,), x_vec: (X,)) -> new_state_vec:
            (D,) -- a SINGLE example's transition, with NO batch dimension.
            This function must be `torch.func.vmap`/`jacrev`-transformable
            (pure tensor ops, no data-dependent Python control flow on
            tensor VALUES). It is vmapped internally over every (t, b) pair
            at once -- do not vmap or batch it yourself.
        x_seq: (B, T, X) -- the raw per-timestep inputs, batch-major.
        init_state_vec: (B, D) -- the true, fixed initial state s_0 (NOT a
            free variable -- constant across every Newton round).
        max_newton_iters: hard cap on Newton rounds (Lim et al. prove at
            most T rounds are needed for EXACT convergence of undamped
            Newton on a length-T sequence; quasi-DEER's diagonal
            approximation forfeits that exact bound, so this is a budget,
            not a guarantee -- check `diagnostics["converged"]`).
        tol: convergence threshold on max(abs(y_new - y_old)); defaults to
            1e-4 (float32) / 1e-7 (float64), matching Lim et al.'s own
            reference implementation.
        damping: scale-ELK style shrink factor in [0, 1) applied to the
            diagonal Jacobian used in the linear solve only (see module
            docstring); 0.0 = plain quasi-DEER.
        max_jac_diag_abs: optional hard clamp on |diagonal Jacobian
            entries| before they enter the scan (extra numerical safety
            valve, see module docstring).
        y_init_guess: optional (B, T, D) initial trajectory guess; defaults
            to all-zeros (Lim et al.'s own choice for their GRU
            experiments, in the absence of a better warm start).
        jac_chunk_size: unit tangent directions to differentiate together
            when extracting diag(df/ds) via JVPs. 1 = serial (low memory);
            >1 vmaps that many directions (fast but memory-hungry). Set <= 0
            to fall back to dense `jacrev` (only safe for small D).
        jac_sample_batch_size: how many (batch, time) pairs to vmap over
            when computing diag_jac. 1 = serial over samples (safest); raise
            on GPUs with headroom for speed.
        step_sample_batch_size: how many (batch, time) pairs to vmap over when
            evaluating f(s_{t-1}, x_t) each Newton round. Defaults to
            jac_sample_batch_size. 1 = serial (safest on ~16 GiB GPUs).

    Returns:
        (y, diagnostics) where `y` is (B, T, D) -- the converged (or
        best-effort, if the iteration budget ran out first) state
        trajectory s_1..s_T -- and `diagnostics` is a dict with
        `newton_iters`, `final_max_abs_delta`, `converged`, `tol`.
    """
    if x_seq.dim() != 3:
        raise ValueError(f"x_seq must be (B, T, X); got shape {tuple(x_seq.shape)}")
    if init_state_vec.dim() != 2:
        raise ValueError(f"init_state_vec must be (B, D); got shape {tuple(init_state_vec.shape)}")

    B, T, X = x_seq.shape
    D = init_state_vec.shape[-1]
    device, dtype = init_state_vec.device, init_state_vec.dtype
    if tol is None:
        tol = _default_tol(dtype)

    y = (
        torch.zeros(B, T, D, device=device, dtype=dtype)
        if y_init_guess is None
        else y_init_guess.to(device=device, dtype=dtype).clone()
    )

    if step_sample_batch_size is None:
        step_sample_batch_size = jac_sample_batch_size

    newton_iters = 0
    final_err = float("inf")
    # No-grad guard for the whole forward solve: gradients are obtained
    # afterward via implicit differentiation (deer_adjoint_scan below),
    # not by differentiating through the Newton loop / JVP calls here.
    # Entered/exited manually (rather than a `with` block) so the loop
    # body below doesn't need to be re-indented.
    _no_grad_guard = torch.no_grad()
    _no_grad_guard.__enter__()
    for it in range(max_newton_iters):
        print(f"[Newton] iter {it+1}/{max_newton_iters}")
        # Predecessor state for every t=1..T: s_0 (fixed) followed by the
        # current guess's own s_1..s_{T-1} (i.e. shift-by-one along time).
        y_prev = torch.cat([init_state_vec.unsqueeze(1), y[:, :-1, :]], dim=1)  # (B, T, D)

        flat_prev = y_prev.reshape(B * T, D)
        flat_x = x_seq.reshape(B * T, X)

        f_out = _batched_step_fn(
            per_sample_step,
            flat_prev,
            flat_x,
            step_sample_batch_size=step_sample_batch_size,
        ).reshape(B, T, D)

        if jac_chunk_size <= 0:
            def _diag_fn(s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                jac = jacrev(lambda ss: per_sample_step(ss, x))(s)
                return torch.diagonal(jac)
        else:
            def _diag_fn(s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                return _diag_jac_via_jvp(
                    per_sample_step, s, x, jac_chunk_size=jac_chunk_size
                )
        
        diag_jac = _batched_diag_jac_fn(
            _diag_fn,
            flat_prev.detach(),
            flat_x.detach(),
            jac_sample_batch_size=jac_sample_batch_size,
        ).reshape(B, T, D)

        if max_jac_diag_abs is not None:
            diag_jac = diag_jac.clamp(min=-max_jac_diag_abs, max=max_jac_diag_abs)
        if damping:
            diag_jac = (1.0 - damping) * diag_jac

        residual = y - f_out  # r_t(s^(i)), t = 1..T

        delta = selective_scan_chunk(
            dA=diag_jac,
            dBx=-residual,
            h0=torch.zeros(B, D, device=device, dtype=dtype),
            time_dim=1,
        )
        y_new = y + delta

        err = (y_new - y).abs().max().item()
        y = y_new
        newton_iters = it + 1
        final_err = err
        if err < tol:
            break

    diagnostics = {
        "newton_iters": newton_iters,
        "final_max_abs_delta": final_err,
        "converged": final_err < tol,
        "tol": tol,
    }
    _no_grad_guard.__exit__(None, None, None)
    return y, diag_jac, diagnostics



def deer_adjoint_scan(diag_jac: torch.Tensor, grad_y: torch.Tensor) -> torch.Tensor:
    """Reverse-time affine scan computing the adjoint state

        mu_t = grad_y_t + diag_jac_{t+1} * mu_{t+1}     (mu_T := grad_y_T)

    Implements the backward half of Lim et al. 2024 eq. 6-7: gradients
    through the converged fixed point come from ONE application of the
    (transposed) linear operator, not from differentiating through the
    Newton iteration. A diagonal Jacobian is its own transpose, so this
    adjoint recurrence has the exact same elementwise-affine shape as the
    forward recurrence -- just walking backward in time with the
    coefficients shifted by one step -- so it reuses `selective_scan_chunk`
    unchanged (same O(log T) cost as the forward linear solve).

    Args:
        diag_jac: (B, T, D) -- diagonal Jacobian at the CONVERGED
            trajectory (diag_jac[:, t, :] = diag(df/ds_{t-1})).
        grad_y: (B, T, D) -- dL/dy at the converged trajectory.

    Returns:
        mu: (B, T, D).
    """
    B, T, D = diag_jac.shape
    zeros_pad = torch.zeros_like(diag_jac[:, :1, :])
    a_rev = torch.cat([zeros_pad, torch.flip(diag_jac[:, 1:, :], dims=[1])], dim=1)
    b_rev = torch.flip(grad_y, dims=[1])
    nu = selective_scan_chunk(
        dA=a_rev,
        dBx=b_rev,
        h0=torch.zeros(B, D, device=diag_jac.device, dtype=diag_jac.dtype),
        time_dim=1,
    )
    return torch.flip(nu, dims=[1])
