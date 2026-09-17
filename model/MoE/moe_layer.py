"""
file: moe_layer.py

Reusable Switch-style (top-1) Mixture-of-Experts feed-forward layer,
architecture-agnostic controller "add-on" block. Designed per
Experiment-Roadmap.md, "Alternative Phase 3 - Step 2, Option 4" and the
corpus's MoE-Mamba-specific and general-MoE failure-mode tables:

  - EXTERNAL placement only: this module is meant to be interleaved
    BETWEEN controller blocks (Mamba blocks, or eventually Transformer
    blocks), never spliced inside a block's own internal projections.
    MoE-Mamba's own ablation (Dead-End #43) found every internal-placement
    variant underperforms external interleaving -- this module's whole
    API shape (a drop-in residual feed-forward sublayer operating on
    (..., d_model) token representations) exists so it can ONLY be used
    externally, by construction: it has no notion of any specific
    controller's internal gates/projections, so there's nothing to splice
    it into.
  - Switch-style top-1 routing (Fedus, Zoph & Shazeer, "Switch
    Transformers"), not Shazeer et al. 2017's noisy top-k -- MoE-Mamba
    itself builds on Switch's k=1 simplification specifically because it
    reduces router computation, at-least-halves expert capacity, and
    simplifies communication, with no quality loss over top-k>1 in
    Switch's own ablations. Reusing that choice here.
  - Expert-count floor: default num_experts=8, with num_experts<4
    rejected outright in __init__, per Dead-End #45 -- a single-expert
    MoE layer measurably underperforms no-MoE-at-all in MoE-Mamba's own
    ablation.
  - Auxiliary load-balancing loss, Switch's single differentiable term
    (Fedus et al. Eq. 4-6): loss = alpha * N * sum_i f_i * P_i. Switch
    simplified Shazeer et al. 2017's separate importance + load losses
    into this one term; MoE-Mamba follows Switch, so this file does too.
    Default alpha=0.01, Switch's own tuned value (they swept 1e-1..1e-5
    and found 1e-2 balanced load quickly without hurting the primary
    objective).
  - Controller-agnostic call signature: forward(x) where x is
    (..., d_model) -- works whether the caller passes one timestep
    (B, d_model), as mamba_controller.py's MambaControllerWrapper does
    (DNC drives its controller one timestep at a time, interleaved with
    memory read/write), or a whole sequence (B, T, d_model) at once, as a
    future Transformer controller could (it doesn't need DNC's
    per-timestep interleaving the same way). Token routing is computed
    per-token regardless of T, so no controller-specific branching is
    needed inside this module -- this is the concrete mechanism that
    makes Option 4 reusable across controllers, per the roadmap's
    explicit requirement.

Sized per the corpus's dexpert scaling note (MoE-Mamba, Section 3.4 /
Appendix B): to keep active-parameters-per-token roughly comparable to a
dense feed-forward layer at a given Mamba:MoE active-parameter ratio,
MoE-Mamba defaults to dexpert = 3 * d_model at their own "3:3" ratio
(their own reported sweet spot -- Figure 5 -- gains become marginal past
this ratio and higher ratios are impractical due to routing overhead).
This file mirrors that default (expert_dim = 3 * d_model) but leaves it
fully overridable.

capacity_factor default is 1.5, not Switch's own 1.0-2.0 range midpoint of
1.25: mamba_controller.py's MambaControllerWrapper calls this module once
PER TIMESTEP (num_tokens == batch_size only, e.g. 16), not once per whole
padded sequence the way a Transformer FFN pass would. With that few
tokens per call, a low capacity factor drops a much larger fraction of
tokens on ordinary jitter than it would at Switch's own token-per-batch
scale -- 1.5 gives more headroom for that per-call regime while still
being a real (not effectively-infinite) capacity constraint.

Failure mode this file's docstrings exist specifically to flag but cannot
itself check (see Experiment-Roadmap.md, "Option 4" failure-mode table):
  - ANOM-151 (MoE-Mamba's own unresolved anomaly): added MoE capacity does
    NOT substitute for content-based retrieval/copying capacity -- lower
    perplexity but lower accuracy was observed against dense
    Transformer-MoE in the source paper, with the authors' own untested
    conjecture being that a fixed-size state limits exactly the
    induction-head-style copying capacity content-addressed retrieval
    depends on. This module's diagnostics (below) cover routing/load
    health only. The mandatory check is the CALLER's job: keep running
    the training script's existing memory-dependency ablation
    (evaluate_traversal(..., ablate_memory=True)) and hop-count breakdown
    before/after enabling MoE, and treat a task-loss-only comparison as
    insufficient.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """One feed-forward expert: Linear -> ReLU -> Linear, matching
    MoE-Mamba's/Switch's own expert architecture (a single hidden layer,
    ReLU-activated) rather than anything more exotic -- keeping this
    minimal is itself corpus-consistent (Dead-End #43: fancier internal
    placements underperformed the plain external design)."""

    def __init__(self, d_model: int, expert_dim: int):
        super().__init__()
        self.w_in = nn.Linear(d_model, expert_dim)
        self.w_out = nn.Linear(expert_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(F.relu(self.w_in(x)))


class SwitchMoE(nn.Module):
    """Switch-style (top-1) sparsely-gated MoE feed-forward sublayer.
    Controller-agnostic: operates on the last dimension of whatever shape
    is passed in (..., d_model), and returns the same shape -- it does NOT
    apply its own residual connection (see MoEBlock below for that), so it
    composes cleanly regardless of what pre-norm/residual convention the
    calling controller block already uses.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        expert_dim: int | None = None,
        capacity_factor: float = 1.5,
        router_noise_eps: float = 1e-2,
        load_balance_alpha: float = 0.01,
    ):
        super().__init__()
        if num_experts < 4:
            raise ValueError(
                f"SwitchMoE: num_experts={num_experts} < 4 -- Dead-End #45 "
                "found a single-expert (and, by the same logic, very-few-"
                "expert) MoE layer measurably underperforms no-MoE-at-all. "
                "This floor is enforced here rather than left as a silent "
                "footgun; pass num_experts>=4 (8+ preferred, per the roadmap)."
            )
        self.d_model = d_model
        self.num_experts = num_experts
        expert_dim = expert_dim if expert_dim is not None else 3 * d_model
        self.capacity_factor = capacity_factor
        self.router_noise_eps = router_noise_eps
        self.load_balance_alpha = load_balance_alpha

        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d_model, expert_dim) for _ in range(num_experts)])

        self._aux_losses: list[torch.Tensor] = []
        self._last_diag: dict = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        d_model = orig_shape[-1]
        assert d_model == self.d_model, (
            f"SwitchMoE: input last dim {d_model} != configured d_model {self.d_model}"
        )
        flat = x.reshape(-1, d_model)  # (num_tokens, d_model)
        num_tokens = flat.shape[0]

        logits32 = self.router(flat).float()  # router kept in fp32 (Switch's own selective-precision fix)
        if self.training and self.router_noise_eps > 0:
            # Switch's own exploration mechanism (Appendix C): multiplicative
            # jitter noise. We apply it directly to the router logits here
            # (a commonly-used equivalent of jittering the router input) for
            # simplicity.
            noise = torch.empty_like(logits32).uniform_(1.0 - self.router_noise_eps, 1.0 + self.router_noise_eps)
            logits32 = logits32 * noise
        probs = torch.softmax(logits32, dim=-1)  # (num_tokens, num_experts)

        top1_prob, top1_idx = probs.max(dim=-1)  # (num_tokens,), (num_tokens,)

        if self.training:
            # Switch/Shazeer-style load-balancing auxiliary loss (Fedus et
            # al. Eq. 4-6): loss = alpha * N * sum_i f_i * P_i, where f_i is
            # the fraction of tokens routed (argmax) to expert i and P_i is
            # the fraction of router probability mass assigned to expert i
            # across the batch. f is non-differentiable (argmax); P carries
            # the gradient back into the router. Computed here on only the
            # tokens THIS forward() call sees -- no scan of anything outside
            # this one call, matching Phase 1/2's "local to this call" bar.
            one_hot = F.one_hot(top1_idx, num_classes=self.num_experts).float()
            f_i = one_hot.mean(dim=0)          # (num_experts,)
            P_i = probs.mean(dim=0)            # (num_experts,)
            aux_loss = self.load_balance_alpha * self.num_experts * (f_i * P_i).sum()
            self._aux_losses.append(aux_loss)

            with torch.no_grad():
                cv_importance = (P_i.std() / P_i.mean().clamp(min=1e-8)).item()
                cv_load = (f_i.std() / f_i.mean().clamp(min=1e-8)).item()
                self._last_diag = {
                    "cv_importance": cv_importance,
                    "cv_load": cv_load,
                    "max_load_frac": f_i.max().item(),
                }

        # Expert capacity (Switch Eq. 3): buffer above the even split so
        # token routing doesn't collapse under minor imbalance. Tokens
        # beyond an expert's capacity are dropped -- they get zero expert
        # contribution and pass through via whatever residual the CALLER
        # applies (this module returns only the expert contribution, not
        # x + expert(x)).
        capacity = max(1, int((num_tokens / self.num_experts) * self.capacity_factor))

        """
        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token_mask = top1_idx == expert_id
            token_indices = token_mask.nonzero(as_tuple=True)[0]
            if token_indices.numel() == 0:
                continue
            if token_indices.numel() > capacity:
                # Drop overflow tokens (Switch's own documented behavior).
                token_indices = token_indices[:capacity]
            expert_out = expert(flat[token_indices])
            gate = top1_prob[token_indices].unsqueeze(-1).to(expert_out.dtype)
            output[token_indices] = (expert_out * gate).to(output.dtype)

        return output.reshape(orig_shape)
        """
    

        # Vectorized dispatch: at this project's scale (num_tokens ==
        # batch_size, e.g. 16, called once per DNC timestep), the old
        # per-expert Python loop launched one tiny (<=capacity-row) matmul
        # per expert per call -- kernel-launch-overhead-bound, not
        # compute-bound, at capacity~3. Replacing it with two batched
        # einsum calls (every expert applied to every token at once)
        # trades a few extra FLOPs (cheap at this token count) for a
        # constant number of kernel launches regardless of num_experts.
        # Capacity-based dropping is preserved exactly -- same
        # first-N-tokens-in-order-per-expert semantics as the loop
        # version -- computed via a cumulative count instead of
        # per-expert slicing.
        w_in = torch.stack([e.w_in.weight for e in self.experts], dim=0)    # (E, expert_dim, d_model)
        b_in = torch.stack([e.w_in.bias for e in self.experts], dim=0)      # (E, expert_dim)
        w_out = torch.stack([e.w_out.weight for e in self.experts], dim=0)  # (E, d_model, expert_dim)
        b_out = torch.stack([e.w_out.bias for e in self.experts], dim=0)    # (E, d_model)

        hidden = torch.einsum('td,exd->tex', flat, w_in) + b_in            # (T, E, expert_dim)
        hidden = F.relu(hidden)
        expert_out_all = torch.einsum('tex,edx->ted', hidden, w_out) + b_out  # (T, E, d_model)

        # Every token only ever uses its top-1 expert's output.
        expert_out = expert_out_all[torch.arange(num_tokens, device=flat.device), top1_idx]  # (T, d_model)

        # Capacity mask: keep only the first `capacity` tokens (in
        # original token order) routed to each expert -- identical drop
        # behavior to the old token_indices[:capacity] slicing.
        one_hot = F.one_hot(top1_idx, num_classes=self.num_experts)  # (T, E)
        rank_in_expert = (
            one_hot.cumsum(dim=0).gather(1, top1_idx.unsqueeze(1)).squeeze(1) - 1
        )  # (T,) -- this token's position among tokens routed to the same expert
        keep = rank_in_expert < capacity  # (T,) bool

        gate = (top1_prob * keep.to(top1_prob.dtype)).unsqueeze(-1).to(expert_out.dtype)
        output = (expert_out * gate).to(flat.dtype)

        # One caveat: this computes all 8 experts for all tokens rather than skipping unused ones, 
        # so if you ever scale num_tokens way up (e.g. via BATCH_SIZE) the FLOPs cost of this dense approach grows 
        # faster than the old sparse loop's would. At your current scale that's irrelevant; 
        # if you ever do increase batch size significantly later, worth re-benchmarking which approach 
        # wins at that point.
        return output.reshape(orig_shape)

    def pop_aux_loss(self) -> torch.Tensor:
        """Consume and clear the accumulated load-balancing loss (mirrors
        StochasticWriteHead.pop_kl()'s accumulate-then-pop convention in
        stochastic_write_head_v2.py, so the training loop's pattern for
        combining an auxiliary loss with the task loss stays consistent
        across both mechanisms)."""
        if not self._aux_losses:
            return torch.zeros((), device=self.router.weight.device)
        total = torch.stack(self._aux_losses).sum()
        self._aux_losses = []
        return total

    def last_diagnostics(self) -> dict:
        """Read-only routing-health diagnostics from the most recent
        training forward() call: CV(Importance), CV(Load) (Shazeer et al.
        2017 Appendix A's own balance metrics -- lower is better balanced),
        and the single most-loaded expert's fraction of tokens. Does NOT
        include anything about retrieval/copying accuracy -- ANOM-151
        requires the training script's own memory-ablation check for that,
        not a routing-internal metric."""
        return dict(self._last_diag)


class MoEBlock(nn.Module):
    """Pre-norm residual wrapper around a SwitchMoE sublayer -- the same
    Add->LN->Mixer residual pattern any Transformer/Mamba sub-block uses
    (see mamba_controller.py's MambaControllerBlock for the mixer-side
    version of the identical pattern), so this can be interleaved after
    ANY controller's per-block output, Mamba or (future) Transformer,
    without either controller needing its own bespoke MoE wrapper. This is
    the concrete reuse point: a future Transformer controller instantiates
    this exact class the same way mamba_controller.py does.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        expert_dim: int | None = None,
        capacity_factor: float = 1.5,
        router_noise_eps: float = 1e-2,
        load_balance_alpha: float = 0.01,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.moe = SwitchMoE(
            d_model, num_experts=num_experts, expert_dim=expert_dim,
            capacity_factor=capacity_factor, router_noise_eps=router_noise_eps,
            load_balance_alpha=load_balance_alpha,
        )
        if device is not None and getattr(device, "type", None) == "cuda":
            self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.moe(self.norm(x))

    def pop_aux_loss(self) -> torch.Tensor:
        return self.moe.pop_aux_loss()

    def last_diagnostics(self) -> dict:
        return self.moe.last_diagnostics()


def pop_total_moe_aux_loss(moe_layers: list) -> tuple:
    """Sum pop_aux_loss() across all installed MoEBlock/SwitchMoE layers and
    merge their diagnostics (mean across layers for CV metrics, max for the
    worst-case load fraction), mirroring stochastic_write_head_v2's
    pop_total_kl() pattern exactly, so the training loop combines this
    auxiliary loss the same way it already combines the KL loss."""
    total = None
    cv_imp, cv_load, max_load = [], [], []
    for layer in moe_layers:
        loss = layer.pop_aux_loss()
        total = loss if total is None else total + loss
        diag = layer.last_diagnostics()
        if diag:
            cv_imp.append(diag["cv_importance"])
            cv_load.append(diag["cv_load"])
            max_load.append(diag["max_load_frac"])
    if total is None:
        total = torch.zeros(())
    merged = {
        "moe_cv_importance": sum(cv_imp) / len(cv_imp) if cv_imp else 0.0,
        "moe_cv_load": sum(cv_load) / len(cv_load) if cv_load else 0.0,
        "moe_max_load_frac": max(max_load) if max_load else 0.0,
    }
    return total, merged