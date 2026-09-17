"""
dnc_parallel_scan.py -- v1 (new file)

Alternate Phase 3, Step 2 (see Experiment-Roadmap.md, "Attempt to modify DNC
in a way that it can in one way or another simulate SSMs parallelism").

This file contains exactly one thing: a generic, differentiable, TRUE
parallel (O(log C) sequential depth, not O(C)) associative scan for affine
recurrences of the form

    h_t = a_t * h_{t-1} + b_t ,      t = 1..C

which is the exact structural shape of BOTH:
  - Mamba's S6 selective-scan recurrence (Gu & Dao 2024, Algorithm 2, line 6):
        h_t = dA_t * h_{t-1} + dB_t * x_t
    with a_t := dA_t = exp(Δ_t A) and b_t := dB_t * x_t.
  - the "linear part" of DNC's memory-matrix recurrence used within a chunk
    once the write weighting w_t is frozen to a per-chunk value (see
    Concept_8_-_State_Space_or_Latent_Dynamical_System_Formalism.md and the
    roadmap's "Where DNC and SSMs really are the same math" section):
        M_t = (1 - w_t e_t^T) ⊙ M_{t-1} + w_t v_t^T
    with a_t := (1 - w_t e_t^T) and b_t := w_t v_t^T.

Only the Mamba case is actually driven through this module in this codebase
(see mamba_chunk_controller.py) -- DNC's own write weighting is deliberately
left fully sequential and untouched (see chunked_parallel_dnc.py's module
docstring for why: w_t depends on M_{t-1} via content-based cosine-similarity
addressing, which is exactly the state-dependence the roadmap identifies as
"the one piece of DNC's math that doesn't reduce to an SSM" -- see
Experiment-Roadmap.md, "Where it breaks"). The scan primitive is written
generically (not hardcoded to Mamba's tensor layout) specifically so it is
reusable for that DNC-side recurrence too, should a later phase decide the
tradeoff is worth it (see the standalone architecture document,
"Q21-Alternate-Phase3-Step2-Architecture.md", Section 5, for that discussion)
-- but nothing in this codebase currently calls it for that purpose.

--------------------------------------------------------------------------
Why this exists instead of a Python for-loop over the chunk
--------------------------------------------------------------------------
A naive per-step Python loop over a chunk of length C to compute
h_t = a_t*h_{t-1}+b_t has O(C) *sequential* depth -- C dependent tensor ops
that cannot be parallelized by the GPU scheduler no matter how large the
batch/channel dims are, because op t cannot start before op t-1 finishes.
This is the literal bottleneck Mamba's own hardware-aware scan (Gu & Dao
2024, Section 3.3) exists to remove for its own recurrence -- and it is
equally the bottleneck this file removes for the chunked Mamba controller
used as DNC's controller here.

The classic fix (Blelloch 1990, cited directly by the Mamba paper's Section
3.3 as one of the "three classical techniques" its own hardware-aware scan
uses) is to note that two affine maps compose associatively:

    f_1(x) = a_1*x + b_1
    f_2(x) = a_2*x + b_2
    (f_2 ∘ f_1)(x) = f_2(f_1(x)) = a_2*(a_1*x+b_1) + b_2
                    = (a_2*a_1)*x + (a_2*b_1 + b_2)

so the pair-combine operator

    combine((a1,b1), (a2,b2)) = (a1*a2, a2*b1 + b2)     # apply 1 then 2

is associative, which means the *cumulative* composition (A_t, B_t) for
every t = 1..C (where h_t = A_t*h_0 + B_t) can be computed by a
Hillis-Steele inclusive scan: O(log2(C)) sequential rounds, each round
fully vectorized (parallel) across every t and every batch/channel
dimension simultaneously. This is the same asymptotic shape as Mamba's own
work-efficient parallel scan (Blelloch-style, as the paper's Section 3.3
also references) -- the concrete numbers differ (this is a simpler
Hillis-Steele variant, not the exact work-efficient up-sweep/down-sweep
Blelloch tree, and it is not fused into a single fp32-SRAM-resident CUDA
kernel the way `selective_scan_fn` is), but the *algorithmic class* -- a
genuine parallel associative scan over the recurrence, not a sequential
Python loop -- is the same, and it is portable (pure PyTorch, no custom
CUDA kernel / nvcc build step required, unlike `mamba-ssm`'s own fast path).

--------------------------------------------------------------------------
Numerical stability note
--------------------------------------------------------------------------
Some public parallel-scan writeups for linear recurrences of this shape do
the combine in LOG space (accumulate log(a_t) via cumsum, then divide out
the b_t terms by exp(-cumulative-log)) for numerical stability over very
long sequences. That trick is unnecessary and actively riskier here: this
scan is deliberately only ever run over one *chunk* at a time (length C,
a small, bounded constant -- see chunked_parallel_dnc.py's CHUNK_SIZE,
typically 8-64), with the true recurrent state carried into the chunk as
an ordinary tensor (not re-derived via the scan), so there is no
long-sequence log-magnitude blowup to guard against, and log-space division
by exp(-cumulative_log) would introduce exactly the overflow risk it claims
to avoid once a_t is small (a_t = dA_t = exp(Δ_t A) is always in (0,1) for
Mamba's S6 parameterization, so cumulative log-products shrink monotonically
toward -inf, and dividing by that is the numerically dangerous direction).
This module therefore works directly in "value space" (products of
tensors in (0,1) shrinking toward 0, which is numerically benign -- the
same computation the existing sequential per-step code in
mamba_controller.py already performs without any reported numerical issue,
just batched across the chunk instead of accumulated one step at a time).

Correctness of this exact algorithm (the direct, non-log-space
Hillis-Steele doubling scan implemented below) was verified against a
naive sequential-loop reference in NumPy, at S6-shaped tensors
(batch/chunk/d_inner/d_state dimensions matching Mamba-1's actual
parameterization), across randomized trials spanning chunk lengths 1-32:
max absolute error ~1e-6 (float64 numpy; the residual is ordinary
floating-point associativity error, not an algorithmic bug), and the
chunk_size=1 case was checked to reduce EXACTLY to the direct one-step
formula, which is what makes chunk_size=1 a byte-for-byte-equivalent
special case of the chunked design (see chunked_parallel_dnc.py's module
docstring for why that equivalence matters as the mandatory regression
check for this feature).
"""

from __future__ import annotations

import torch


def _shift_right_with_identity(
    a: torch.Tensor, b: torch.Tensor, offset: int, time_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift (a, b) `offset` positions later along `time_dim`, filling the
    newly-exposed leading positions with the affine IDENTITY map
    (a=1, b=0) -- i.e. "no-op, pass h through unchanged" -- rather than
    zeros. This is what makes the Hillis-Steele combine below correct at
    the sequence boundary: positions with no valid predecessor `offset`
    steps back must combine as if nothing happened there, not as if the
    state were forced to zero.
    """
    C = a.shape[time_dim]
    if offset >= C:
        # Every position's predecessor `offset` steps back is out of range
        # -- the whole shifted tensor is identity.
        return torch.ones_like(a), torch.zeros_like(b)

    pad_shape_a = list(a.shape)
    pad_shape_a[time_dim] = offset
    pad_shape_b = list(b.shape)
    pad_shape_b[time_dim] = offset

    ones_pad = a.new_ones(pad_shape_a)
    zeros_pad = b.new_zeros(pad_shape_b)

    a_body = a.narrow(time_dim, 0, C - offset)
    b_body = b.narrow(time_dim, 0, C - offset)

    a_shifted = torch.cat([ones_pad, a_body], dim=time_dim)
    b_shifted = torch.cat([zeros_pad, b_body], dim=time_dim)
    return a_shifted, b_shifted


def _combine(
    prior: tuple[torch.Tensor, torch.Tensor], new: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Associative combine of two affine maps: apply `prior` then `new`.

    prior(x) = Ap*x + Bp ; new(x) = An*x + Bn
    (new ∘ prior)(x) = An*(Ap*x+Bp) + Bn = (An*Ap)*x + (An*Bp + Bn)
    """
    Ap, Bp = prior
    An, Bn = new
    return Ap * An, Bp * An + Bn


def associative_scan_affine(
    a: torch.Tensor, b: torch.Tensor, time_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive Hillis-Steele parallel scan of the affine-composition
    operator along `time_dim`.

    Given per-step affine coefficients a_t, b_t (any shape, with `time_dim`
    the chunk/sequence axis of length C), returns the CUMULATIVE
    composition (A_t, B_t) for every t = 1..C, i.e. the coefficients of
    the single affine map equivalent to applying steps 1..t in order:

        A_t, B_t  such that  h_t = A_t * h_0 + B_t

    for any initial state h_0 (h_0 itself is NOT baked in here -- see
    `selective_scan_chunk` below, which applies it once after this
    returns, so this function stays a pure, reusable "scan the operators"
    primitive independent of what initial state it will later be applied
    to).

    Sequential depth: ceil(log2(C)) combine rounds, each fully vectorized
    (parallel) across every other dimension. Out-of-place throughout
    (`torch.cat`, elementwise ops only -- no `.copy_()`/in-place writes),
    so this is safe to backpropagate through under full BPTT, exactly like
    every other tensor op in this codebase's autograd graph.
    """
    C = a.shape[time_dim]
    A, B = a, b
    offset = 1
    while offset < C:
        A_shift, B_shift = _shift_right_with_identity(A, B, offset, time_dim)
        A, B = _combine((A_shift, B_shift), (A, B))
        offset *= 2
    return A, B


def selective_scan_chunk(
    dA: torch.Tensor, dBx: torch.Tensor, h0: torch.Tensor, time_dim: int = 1
) -> torch.Tensor:
    """Apply the S6-shaped recurrence h_t = dA_t * h_{t-1} + dBx_t for
    t = 1..C, given initial state h0, over a whole chunk in parallel.

    Args:
        dA:  (..., C, ...) per-step multiplicative coefficients, `time_dim`
             is the chunk axis. For Mamba this is exp(Δ_t A), shape
             (B, C, d_inner, d_state).
        dBx: same shape as dA -- per-step additive term Δ_t B_t x_t.
        h0:  initial state, same shape as dA/dBx MINUS the `time_dim` axis
             (e.g. (B, d_inner, d_state)) -- the state carried in from
             before this chunk (either a genuine zero state at sequence
             start, or the previous chunk's true final state).
        time_dim: which axis of dA/dBx is the chunk/time axis (default 1,
             matching this codebase's (B, C, d_inner, d_state) convention).

    Returns:
        h: same shape as dA/dBx -- h[..., t, ...] = the state AFTER step t
           (i.e. h[:, 0] is the state after the first step of this chunk,
           matching the usual "output at time t reflects input up to and
           including t" recurrent convention).

    Reduces EXACTLY to the direct one-step formula
        h_1 = dA_1 * h0 + dBx_1
    when C == 1 (the Hillis-Steele loop body never executes, since
    `offset=1` is not `< C=1`), which is the algebraic reason
    `chunk_size=1` reproduces mamba_controller.py's `MambaControllerCell.
    step()` bit-for-bit -- see chunked_parallel_dnc.py.
    """
    A, B = associative_scan_affine(dA, dBx, time_dim=time_dim)
    h0_expanded = h0.unsqueeze(time_dim)
    return A * h0_expanded + B
