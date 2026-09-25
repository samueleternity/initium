"""
file: stochastic_write_head_v2.py
Phase 1 patch: stochastic (Gaussian) write vector for pytorch-dnc.

Design, per Phase 1 instructions:
  - Do NOT touch the controller, the write mechanism's addressing
    (content/allocation-based), or any other component.
  - The write head emits distribution parameters (mu_t, sigma_t) of a
    diagonal Gaussian q(v_t) = N(mu_t, diag(sigma_t^2)) instead of a
    deterministic write vector.
  - The value actually written to memory is sampled via the
    reparameterization trick: v_t = mu_t + sigma_t * eps, eps ~ N(0, I).
  - L_KL(t) = KL(q(v_t) || p0) is computed in closed form, using ONLY this
    timestep's write-head output (no scan of the memory matrix, no other
    timestep, no other component) -> unambiguously local.
  - Prior p0 = N(0, I), fixed, non-learned, isotropic, in the memory-row
    (= write-vector) embedding space.

How the patch works
--------------------
dnc.memory.Memory (independent_linears=True, the pytorch-dnc default) builds
the write vector as:

    write_vector = self.write_vector_transform(interface_input)   # nn.Linear

where `interface_input` is the same per-timestep tensor fed to every other
head transform (read keys, erase vector, gates, ...). `write_vector_transform`
is called with a single (BATCH, input_size) tensor and returns a single
(BATCH, cell_size) tensor -- nothing more.

StochasticWriteHead has the exact same call signature: forward(x) -> tensor
of shape (BATCH, cell_size). We install it by literally replacing the
`write_vector_transform` attribute on the Memory submodule(s) found on the
model, after model construction and before optimizer construction. Because
it's just a submodule swap, DNC's forward pass, `_layer_forward`, and
`Memory.write()` need zero modification -- they keep calling
`self.write_vector_transform(x)` exactly as before, unaware that the return
value is now sampled rather than deterministic.

Nothing about content-addressing, allocation, erase vectors, gates, or the
controller is touched. Only the mapping from interface_input -> write_vector
changes.

--- v2: learned, periodically-snapshotted prior (Phase 2) ------------------
Phase 1's prior p0 = N(0,I) was fixed and non-stateful, so there was nothing
to log or checkpoint about it. Phase 2 replaces p0 with p0 = N(mu_g, Sigma_g)
(diagonal), a global prior fit from the write head's own recent output and
updated on a slow, periodic schedule -- per the roadmap's Phase 2 design.
Everything below is additive; nothing about the Phase 1 code path changes
if the new snapshot-update method is never called (mu_g/logvar_g buffers
stay at their zero-init, i.e. N(0,I), reproducing Phase 1 bit-for-bit).

1. StochasticWriteHead now owns two extra registered buffers, `prior_mu` and
   `prior_logvar` (shape (cell_size,), zero-initialized). Buffers, not
   Parameters: they must NOT receive a gradient from the per-write KL term
   (that would make the "prior" just another thing being learned by the
   same backward pass the KL is supposed to regularize against, which is
   exactly the abandon-locality-via-continuous-coupling failure mode the
   roadmap's snapshot design exists to avoid). They are written to only by
   `update_prior_snapshot()`, under `torch.no_grad()`, on the training
   script's own periodic cadence -- never by `.backward()`.
2. `forward()` now also accumulates a detached copy of every sampled write
   v_t into `self._recent_writes` (only while `self.training and
   self.sample`, matching the existing KL-accumulation guard). This is the
   "recent writes" buffer `update_prior_snapshot()` fits mu_g/Sigma_g from.
3. The per-write KL closed form is generalized from KL(q(v_t)||N(0,I)) to
   KL(q(v_t)||N(mu_g,Sigma_g)) -- reduces to the exact Phase 1 formula when
   mu_g=0, Sigma_g=I (the buffers' initial state), so this is a strict
   generalization, not a behavior change, for any run that never calls
   update_prior_snapshot(). The computation still reads only this call's
   (mu_t, logvar_t) plus the current frozen (prior_mu, prior_logvar)
   buffers -- no scan of accumulated writes, no other timestep -- so the
   per-write locality claim is unchanged from Phase 1: the only non-local
   computation is the periodic snapshot update itself, isolated to
   `update_prior_snapshot()`.
4. `update_prior_snapshot(step, ...)`: computes mu_g <- mean and Sigma_g
   (diagonal) <- variance of the accumulated `_recent_writes` samples,
   clamps the resulting log-variance to [min_logvar, max_logvar] as a
   numerical floor/ceiling (prevents a silent Sigma_g -> 0 collapse, which
   would not show up in the existing q(v_t) KL-collapse diagnostic at all,
   since that diagnostic only ever looked at q, never at the prior), copies
   the result into the `prior_mu`/`prior_logvar` buffers under
   `torch.no_grad()`, records `last_snapshot_step = step`, clears the
   buffer, and returns a diagnostics dict (mu_g norm, diag(Sigma_g)
   mean/min/max, trace, sample count) for the training script to log.
5. `snapshot_diagnostics()`: same summary dict as above, read-only, for
   checkpointing / ad-hoc logging without mutating anything.
6. `pop_kl()`'s diagnostics dict now also carries `snapshot_step` (=
   `last_snapshot_step`), so every per-write-KL-window log line can be
   tagged with exactly which frozen (mu_g, Sigma_g) snapshot that window's
   KL was computed against -- the audit trail the roadmap's Phase 2 spec
   calls for.
7. `pop_total_kl()` now also returns the (min, across heads -- normally a
   single head) `snapshot_step` in its merged diagnostics.
8. New module-level helpers `get_prior_state()` / `load_prior_state()` for
   the training script's checkpoint save/resume path, so `mu_g, Sigma_g`
   (and the step they were last updated) are captured in the checkpoint
   itself rather than living only in the in-memory module buffers -- see
   the training script's v5 header for why this matters.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StochasticWriteHead(nn.Module):
    """Drop-in replacement for `Memory.write_vector_transform`.

    Emits a sampled write vector v_t = mu_t + sigma_t * eps (reparameterization
    trick) instead of a deterministic vector, and accumulates the closed-form
    per-timestep KL(q(v_t) || p0) -- p0 = N(0,I) in Phase 1, or the
    periodically-snapshotted N(prior_mu, diag(exp(prior_logvar))) once
    `update_prior_snapshot()` has been called at least once (Phase 2) -- so
    the training loop can retrieve it after the sequence forward pass and
    add beta * L_KL to the task loss.

    Parameters
    ----------
    in_features, out_features:
        Same as the nn.Linear it replaces (in_features = Memory.input_size,
        out_features = Memory.cell_size).
    device:
        Passed through so the new parameters end up on the right device;
        mirrors Memory's own `.to(device)` handling.
    sample:
        If False, forward() always returns mu (no noise added, no KL signal
        needed) -- used for the beta=0 sanity-check run, which should
        recover Phase 0 (deterministic write vector) exactly rather than
        merely "on average" as beta -> 0 pressure removes noise over
        training. mu_transform is initialized from the original Linear's
        weights (see `install_stochastic_write_heads`), so beta=0 with
        sample=False is bit-for-bit Phase 0's write head at step 0, and
        stays deterministic throughout that run.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        sample: bool = True,
    ):
        super().__init__()
        self.mu_transform = nn.Linear(in_features, out_features)
        self.logvar_transform = nn.Linear(in_features, out_features)

        torch.nn.init.kaiming_uniform_(self.mu_transform.weight)
        # Zero-init the logvar head so sigma_t starts at 1 everywhere, i.e.
        # q(v_t) starts equal to the prior p0 = N(0,I) -- standard VAE-style
        # init that avoids an immediate large KL spike / early instability.
        nn.init.zeros_(self.logvar_transform.weight)
        nn.init.constant_(self.logvar_transform.bias, -4.0)  # sigma ~0.135 at init

        self.sample = sample
        self._kl_terms: list[torch.Tensor] = []  # per-call (BATCH, cell_size) KL, fp32
        self._clamp_terms: list[torch.Tensor] = []  # per-call bool mask, fp32

        # v2 (Phase 2): stateful, periodically-snapshotted prior. Buffers
        # (not Parameters) -- see module docstring point 1 for why: they
        # must never receive a gradient from the per-write KL term, only
        # ever be written by update_prior_snapshot() under no_grad().
        # Zero-init reproduces Phase 1's fixed p0 = N(0,I) exactly for any
        # run that never calls update_prior_snapshot().
        self.register_buffer("prior_mu", torch.zeros(out_features))
        self.register_buffer("prior_logvar", torch.zeros(out_features))
        self.last_snapshot_step: int = 0
        self._recent_writes: list[torch.Tensor] = []  # detached v_t samples since last snapshot

        if device is not None and getattr(device, "type", None) == "cuda":
            self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = self.mu_transform(x)
        logvar = self.logvar_transform(x)

        # [NaN-TRACE] Bisects "this head's input was already corrupted"
        # from "this head's own Linear layers are where it starts" --
        # can't tell these apart from outside the module. Silent unless
        # something's actually wrong.
        if self.training and self.sample:
            x_ok = torch.isfinite(x).all()
            mu_ok = torch.isfinite(mu).all()
            logvar_ok = torch.isfinite(logvar).all()
            if not (x_ok and mu_ok and logvar_ok):
                print(
                    f"[NaN-TRACE] StochasticWriteHead.forward: "
                    f"x_finite={bool(x_ok)} mu_finite={bool(mu_ok)} "
                    f"logvar_finite={bool(logvar_ok)} | "
                    f"x.abs().max()={x.abs().max().item() if x_ok else float('nan')} "
                    f"mu.abs().max()={mu.abs().max().item() if mu_ok else float('nan')} "
                    f"logvar.abs().max()={logvar.abs().max().item() if logvar_ok else float('nan')}"
                )

        if self.sample:
            # Clamp for numerical stability (matters under AMP/fp16 autocast).
            logvar_c = torch.clamp(logvar, min=-10.0, max=10.0)
            if self.training:
                self._clamp_terms.append((logvar != logvar_c).float())
            std = torch.exp(0.5 * logvar_c)
            eps = torch.randn_like(std)
            v = mu + std * eps
        else:
            logvar_c = logvar
            v = mu

        if self.training and self.sample:
            # Closed-form per-dimension KL(q(v_t)||p0), computed in fp32
            # regardless of the ambient autocast dtype. p0 is either the
            # fixed N(0,I) (Phase 1: prior_mu/prior_logvar buffers still at
            # their zero-init) or the current frozen snapshot
            # N(prior_mu, diag(exp(prior_logvar))) (Phase 2). Either way
            # this uses ONLY this call's (mu, logvar) plus the *frozen*
            # prior buffers -- i.e. only this timestep's write-head output
            # and a constant -- nothing else is read, so the per-write
            # locality claim is identical to Phase 1's.
            mu32 = mu.float()
            logvar32 = logvar_c.float()
            prior_mu32 = self.prior_mu.float()
            prior_logvar32 = self.prior_logvar.float()
            # General diagonal-Gaussian KL(N(mu,var) || N(prior_mu,prior_var)):
            #   0.5 * [ prior_logvar - logvar + (exp(logvar) + (mu-prior_mu)^2)/exp(prior_logvar) - 1 ]
            # Reduces exactly to Phase 1's 0.5*(exp(logvar)+mu^2-1-logvar) when
            # prior_mu=0, prior_logvar=0.
            kl_per_dim = 0.5 * (
                prior_logvar32
                - logvar32
                + (logvar32.exp() + (mu32 - prior_mu32).pow(2)) / prior_logvar32.exp()
                - 1.0
            )
            self._kl_terms.append(kl_per_dim)
            # v2: accumulate the actual sampled write for the next periodic
            # prior-snapshot fit. Detached -- this must never carry a graph
            # edge back into this step's backward pass; it is read only by
            # update_prior_snapshot(), itself always called under no_grad().
            self._recent_writes.append(v.detach().float())

        return v

    def pop_kl(self, free_bits: float = 0.0) -> tuple[torch.Tensor, dict]:
        """Consume and clear accumulated per-timestep KL terms.

        Returns (loss, diagnostics):
          loss: scalar tensor, mean over (timesteps * batch) of the
                free-bits-floored, dim-summed per-timestep KL. This is
                L_KL for one sequence forward pass, "summed over dims,
                averaged over timesteps" per the Phase 1 spec.
          diagnostics: dict of detached scalars (pre-free-bits) describing
                the raw KL magnitude distribution this call, for collapse
                monitoring (posterior_collapse if mean/max ~ 0 early and
                stays there). Also carries `snapshot_step` (v2/Phase 2):
                the step at which the (prior_mu, prior_logvar) snapshot
                this KL was computed against was last updated (0 if the
                prior has never been snapshotted, i.e. still N(0,I)).
        """
        if not self._kl_terms:
            zero = torch.zeros((), device=self.mu_transform.weight.device)
            return zero, {
                "kl_mean": 0.0,
                "kl_max": 0.0,
                "kl_min": 0.0,
                "kl_std": 0.0,
                "clamp_frac": 0.0,
                "floor_frac": 0.0,
                "snapshot_step": self.last_snapshot_step,
            }

        kl_stack = torch.cat(
            [t.reshape(-1, t.shape[-1]) for t in self._kl_terms], dim=0
        )  # (T*B, cell_size), fp32, raw (pre-free-bits)

        diagnostics = {
            "kl_mean": kl_stack.mean().item(),
            "kl_max": kl_stack.max().item(),
            "kl_min": kl_stack.min().item(),
            "kl_std": kl_stack.std().item(),
            "clamp_frac": torch.cat([t.flatten() for t in self._clamp_terms]).mean().item()
            if self._clamp_terms
            else 0.0,
            "floor_frac": (kl_stack <= free_bits).float().mean().item() if free_bits > 0 else 0.0,
            "snapshot_step": self.last_snapshot_step,  # v2: audit tag, see module docstring point 6
        }

        if free_bits > 0:
            per_step_kl = kl_stack.sum(dim=-1)  # (T*B,) — sum over dims, pre-clamp
            free_bits_total = (
                free_bits * kl_stack.shape[-1]
            )  # scale per-dim threshold to the summed budget
            per_step_kl = torch.clamp(per_step_kl, min=free_bits_total)
            loss = per_step_kl.mean()
        else:
            loss = kl_stack.sum(dim=-1).mean()

        self._kl_terms = []
        self._clamp_terms = []
        return loss, diagnostics

    def reset_kl(self) -> None:
        self._kl_terms = []

    # ---- v2 (Phase 2): periodic prior snapshot -----------------------------
    def update_prior_snapshot(
        self,
        step: int,
        min_logvar: float = -6.0,
        max_logvar: float = 6.0,
        eps: float = 1e-6,
    ) -> dict:
        """Refit (prior_mu, prior_logvar) from the writes accumulated in
        `self._recent_writes` since the last call, then clear that buffer.

        This is the ONE place in this module where the prior's non-locality
        lives: it looks at (up to) every write since the last snapshot, not
        just one timestep's output. It is meant to be called on a slow,
        periodic cadence (every K training steps) by the training script,
        under no_grad (enforced here regardless of caller), and NEVER from
        inside the per-step backward pass -- that separation is what keeps
        the per-write KL term itself local (see forward()).

        min_logvar/max_logvar: numerical floor/ceiling on the fitted
        log-variance. The floor exists specifically so a collapsing
        Sigma_g -> 0 (a real failure mode a naive fit-from-samples estimator
        can hit, e.g. if the write head briefly degenerates to a near-
        constant output) produces a clamped, loggable, non-degenerate value
        instead of silently propagating a near-zero variance into the next
        K steps' KL denominator. This is the prior-collapse check the
        roadmap flags as invisible to the existing q(v_t) KL diagnostic.

        Returns a diagnostics dict (see snapshot_diagnostics()); additionally
        includes `n_samples`, the number of write-vector samples the fit was
        computed from (0 if `_recent_writes` was empty, in which case the
        previous snapshot is left unchanged and only `n_samples=0` is
        returned -- this can legitimately happen for e.g. sample=False heads,
        or a snapshot cadence shorter than one training step's writes).
        """
        if not self._recent_writes:
            diag = self.snapshot_diagnostics()
            diag["n_samples"] = 0
            diag["raw_var_mean"] = diag["raw_var_max"] = diag["raw_hi_frac"] = 0.0
            return diag

        with torch.no_grad():
            stacked = torch.cat(self._recent_writes, dim=0)  # (N, cell_size)
            n_samples = stacked.shape[0]
            new_mu = stacked.mean(dim=0)
            new_var = stacked.var(dim=0, unbiased=False).clamp(min=eps)
            raw_var_mean = new_var.mean().item()
            raw_var_max = new_var.max().item()
            raw_hi_frac = (new_var.log() > max_logvar).float().mean().item()
            new_logvar = new_var.log().clamp(min=min_logvar, max=max_logvar)
            self.prior_mu.copy_(new_mu)
            self.prior_logvar.copy_(new_logvar)

        self.last_snapshot_step = step
        self._recent_writes = []

        diag = self.snapshot_diagnostics()
        diag["n_samples"] = n_samples
        diag["raw_var_mean"] = raw_var_mean
        diag["raw_var_max"] = raw_var_max
        diag["raw_hi_frac"] = raw_hi_frac
        return diag

    def snapshot_diagnostics(self) -> dict:
        """Read-only summary of the current (prior_mu, prior_logvar)
        snapshot: ||mu_g||, diag(Sigma_g) mean/min/max, trace(Sigma_g), and
        the step it was last updated. Does not mutate anything, does not
        touch `_recent_writes` -- safe to call from checkpointing or ad-hoc
        logging without interfering with the next scheduled snapshot.
        """
        with torch.no_grad():
            sigma_g = self.prior_logvar.exp()
            return {
                "mu_g_norm": self.prior_mu.norm().item(),
                "sigma_g_mean": sigma_g.mean().item(),
                "sigma_g_min": sigma_g.min().item(),
                "sigma_g_max": sigma_g.max().item(),
                "trace_sigma_g": sigma_g.sum().item(),
                "snapshot_step": self.last_snapshot_step,
            }


def install_stochastic_write_heads(
    model: nn.Module,
    device: torch.device | None = None,
    sample: bool = True,
) -> list[StochasticWriteHead]:
    """Replace every Memory.write_vector_transform found on `model` with a
    StochasticWriteHead, initializing mu_transform from the original Linear's
    weights (so, before any training, the sampled mean matches Phase 0's
    deterministic write vector exactly).

    Call this AFTER model construction (and after model.to(device)) and
    BEFORE constructing the optimizer, so the new parameters are included in
    optimizer.parameters(). (v2 note: the new prior_mu/prior_logvar buffers
    are buffers, not parameters, so this ordering requirement is unaffected
    by them -- they're never in optimizer.parameters() regardless.)

    Raises if no dnc.memory.Memory submodule is found, or if a Memory module
    doesn't have write_vector_transform (i.e. was built with
    independent_linears=False) -- this patch specifically targets the
    independent_linears=True path, which is pytorch-dnc's default and what
    the training script uses.
    """
    from dnc.memory import Memory  # local import: only required when patching

    heads: list[StochasticWriteHead] = []
    for name, module in model.named_modules():
        if isinstance(module, Memory):
            if not hasattr(module, "write_vector_transform"):
                raise RuntimeError(
                    f"Memory submodule '{name}' has no write_vector_transform "
                    "(likely built with independent_linears=False). This patch "
                    "requires independent_linears=True, the DNC default."
                )
            orig: nn.Linear = module.write_vector_transform
            stochastic = StochasticWriteHead(
                orig.in_features, orig.out_features, device=device, sample=sample
            )
            with torch.no_grad():
                stochastic.mu_transform.weight.copy_(orig.weight)
                stochastic.mu_transform.bias.copy_(orig.bias)
            module.write_vector_transform = stochastic  # submodule swap
            heads.append(stochastic)

    if not heads:
        raise RuntimeError(
            "No dnc.memory.Memory submodule found on model; cannot install "
            "stochastic write head. Is `model` really a dnc.DNC instance?"
        )
    return heads


def pop_total_kl(heads: list[StochasticWriteHead], free_bits: float = 0.0):
    """Sum pop_kl() across all installed heads (normally just one, since
    num_layers=1 / share_memory_between_layers means a single Memory
    instance). Returns (total_loss, merged_diagnostics).

    v2: merged_diagnostics also carries `snapshot_step` -- the minimum
    across heads (normally there is exactly one head, so this is simply
    that head's last_snapshot_step; the min is a defensive choice for the
    multi-head case so the reported tag never overstates how current every
    head's snapshot is).
    """
    total = None
    merged = {
        "kl_mean": [],
        "kl_max": [],
        "kl_min": [],
        "kl_std": [],
        "clamp_frac": [],
        "floor_frac": [],
    }
    snapshot_steps = []
    for h in heads:
        loss, diag = h.pop_kl(free_bits=free_bits)
        total = loss if total is None else total + loss
        for k in merged:
            merged[k].append(diag[k])
        snapshot_steps.append(diag["snapshot_step"])
    merged = {k: (sum(v) / len(v) if v else 0.0) for k, v in merged.items()}
    merged["snapshot_step"] = min(snapshot_steps) if snapshot_steps else 0
    if total is None:
        total = torch.zeros(())
    return total, merged


# ---- v2 (Phase 2): prior snapshot update / checkpoint helpers --------------
def update_all_prior_snapshots(
    heads: list[StochasticWriteHead],
    step: int,
    min_logvar: float = -6.0,
    max_logvar: float = 6.0,
) -> dict:
    """Call update_prior_snapshot(step, ...) on every head and return a
    single merged diagnostics dict (mean across heads for the summary
    scalars; normally there is exactly one head, so this is just that
    head's own diagnostics). Intended to be called by the training script
    on its PRIOR_SNAPSHOT_EVERY cadence, outside of any per-step backward
    pass.
    """
    per_head = [
        h.update_prior_snapshot(step, min_logvar=min_logvar, max_logvar=max_logvar) for h in heads
    ]
    keys = [
        "mu_g_norm",
        "sigma_g_mean",
        "sigma_g_min",
        "sigma_g_max",
        "trace_sigma_g",
        "raw_var_mean",
        "raw_var_max",
        "raw_hi_frac",
    ]
    merged = {k: sum(d[k] for d in per_head) / len(per_head) for k in keys}
    merged["n_samples"] = sum(d["n_samples"] for d in per_head)
    merged["snapshot_step"] = step
    return merged


def get_prior_state(heads: list[StochasticWriteHead]) -> list[dict]:
    """Snapshot (prior_mu, prior_logvar, last_snapshot_step) for every head,
    as plain CPU tensors/ints, suitable for `torch.save`. This is what lets
    a checkpoint be resumed or re-evaluated against the EXACT prior state it
    was trained/evaluated with, rather than whatever happens to be sitting
    in the (freshly re-initialized, zero) buffers of a newly-constructed
    head -- the same class of bug the write_mode deterministic/sampled
    mismatch check in eval_from_checkpoint.py already exists to catch.
    """
    return [
        {
            "prior_mu": h.prior_mu.detach().cpu().clone(),
            "prior_logvar": h.prior_logvar.detach().cpu().clone(),
            "last_snapshot_step": h.last_snapshot_step,
        }
        for h in heads
    ]


def load_prior_state(heads: list[StochasticWriteHead], state: list[dict]) -> None:
    """Inverse of get_prior_state(): restore (prior_mu, prior_logvar,
    last_snapshot_step) onto each head in-place, in order. Raises if the
    number of heads doesn't match the saved state, since a mismatch here
    means the checkpoint doesn't belong to this model config at all.
    """
    if len(heads) != len(state):
        raise RuntimeError(
            f"load_prior_state: {len(heads)} heads on the model but "
            f"{len(state)} entries in the saved prior state."
        )
    with torch.no_grad():
        for h, s in zip(heads, state):
            h.prior_mu.copy_(s["prior_mu"].to(h.prior_mu.device))
            h.prior_logvar.copy_(s["prior_logvar"].to(h.prior_logvar.device))
            h.last_snapshot_step = s["last_snapshot_step"]
