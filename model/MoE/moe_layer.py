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
        top_k: int = 1,
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
        if not (1 <= top_k <= num_experts):
            raise ValueError(f"SwitchMoE: top_k={top_k} must be in [1, num_experts={num_experts}]")
        self.top_k = top_k

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

        # Top-K routing (Shazeer et al. 2017 / GShard), generalizing Switch's
        # own k=1 special case: each token selects its top_k highest-prob
        # experts, gate weights renormalized to sum to 1 across just those
        # k slots -- top_k=1 reduces to the original Switch gate exactly.
        # Every expert is computed ONCE for the WHOLE token batch (dense,
        # single pair of batched einsums below) regardless of top_k, so the
        # k selected experts for a given token are genuinely active
        # SIMULTANEOUSLY (one fused kernel launch covers all E experts x
        # all T tokens), not run as k sequential passes -- this is what
        # "multiple experts active at once" means at this project's scale.
        top_k = self.top_k
        topk_prob, topk_idx = probs.topk(top_k, dim=-1)              # (T, k) each
        gate_weights = topk_prob / topk_prob.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        # f_i/P_i are cheap to compute regardless of mode; only the aux LOSS
        # accumulation (needed for backward()) is training-gated below.
        # Computing diagnostics unconditionally is what lets inference
        # (model.eval()) report routing health -- previously last_diagnostics()
        # stayed empty forever outside training, which made it impossible to
        # verify MoE routing on real held-out data at inference time.
        one_hot_k = F.one_hot(topk_idx, num_classes=self.num_experts).float()  # (T, k, E)
        f_i = one_hot_k.sum(dim=(0, 1)) / max(num_tokens * top_k, 1)
        P_i = probs.mean(dim=0)

        if self.training:
            aux_loss = self.load_balance_alpha * self.num_experts * (f_i * P_i).sum()
            self._aux_losses.append(aux_loss)

        with torch.no_grad():
            self._last_diag = {
                "cv_importance": (P_i.std() / P_i.mean().clamp(min=1e-8)).item(),
                "cv_load": (f_i.std() / f_i.mean().clamp(min=1e-8)).item(),
                "max_load_frac": f_i.max().item(),
            }

        # Expert capacity, scaled by top_k (each slot competes for the same
        # per-expert buffer).
        capacity = max(1, int((num_tokens * top_k / self.num_experts) * self.capacity_factor))

        w_in = torch.stack([e.w_in.weight for e in self.experts], dim=0)    # (E, expert_dim, d_model)
        b_in = torch.stack([e.w_in.bias for e in self.experts], dim=0)      # (E, expert_dim)
        w_out = torch.stack([e.w_out.weight for e in self.experts], dim=0)  # (E, d_model, expert_dim)
        b_out = torch.stack([e.w_out.bias for e in self.experts], dim=0)    # (E, d_model)

        # Dense pass over EVERY expert for EVERY token -- the single fused
        # computation that makes the top-k selected experts per token
        # simultaneous rather than sequential.
        hidden = torch.einsum('td,exd->tex', flat, w_in) + b_in            # (T, E, expert_dim)
        hidden = F.relu(hidden)
        expert_out_all = torch.einsum('tex,edx->ted', hidden, w_out) + b_out  # (T, E, d_model)

        gathered = torch.gather(
            expert_out_all, 1, topk_idx.unsqueeze(-1).expand(-1, -1, d_model)
        )  # (T, k, d_model)

        # Capacity mask: keep only the first `capacity` (token,slot) pairs,
        # in original order, routed to each expert -- same drop semantics
        # as the prior top-1 implementation, generalized over the
        # flattened (T*k) slot order.
        flat_idx = topk_idx.reshape(-1)
        one_hot_flat = F.one_hot(flat_idx, num_classes=self.num_experts)
        rank_in_expert = one_hot_flat.cumsum(dim=0).gather(1, flat_idx.unsqueeze(1)).squeeze(1) - 1
        keep = (rank_in_expert < capacity).reshape(num_tokens, top_k)

        gate = (gate_weights * keep.to(gate_weights.dtype)).unsqueeze(-1).to(gathered.dtype)
        output = (gathered * gate).sum(dim=1).to(flat.dtype)  # (T, d_model)

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
        top_k: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.moe = SwitchMoE(
            d_model, num_experts=num_experts, expert_dim=expert_dim,
            capacity_factor=capacity_factor, router_noise_eps=router_noise_eps,
            load_balance_alpha=load_balance_alpha, top_k=top_k,
        )
        if device is not None and getattr(device, "type", None) == "cuda":
            self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.moe(self.norm(x))

    def pop_aux_loss(self) -> torch.Tensor:
        return self.moe.pop_aux_loss()

    def last_diagnostics(self) -> dict:
        return self.moe.last_diagnostics()

class SourceEmbedding(nn.Module):
    """Learned per-source bias added to a token's representation before
    routing, so a Top-K router can specialize by SOURCE identity (e.g.
    "parallel backbone output" vs "previous memory read vector" in
    SplitGraphDNC's controller combiner) in addition to content. One
    embedding row per declared source; ADDED, not concatenated, so it never
    changes d_model. Zero-init: routing starts purely content-based and the
    model has to learn any source specialization, matching this project's
    "start equal to the simpler baseline" convention (see
    stochastic_write_head_v2.py's zero-init logvar head)."""

    def __init__(self, num_sources: int, d_model: int, device=None, dtype=None):
        super().__init__()
        self.embedding = nn.Embedding(num_sources, d_model, device=device, dtype=dtype)
        nn.init.zeros_(self.embedding.weight)

    def forward(self, x: torch.Tensor, source_id: int) -> torch.Tensor:
        idx = torch.full((x.shape[0],), source_id, dtype=torch.long, device=x.device)
        return x + self.embedding(idx).to(x.dtype)


class MultiSourceMoEBlock(nn.Module):
    """CfC-oriented MoE sublayer: accepts a VARIABLE-length list of
    per-source token tensors -- each (B, d_model) -- and returns the same
    number of outputs, one per source, each individually routed and gated
    through ONE shared bank of Top-K experts (SwitchMoE.forward -- every
    expert still runs exactly once per call, dense and simultaneous).

    This is what lets a CfC controller/combiner keep its "many inputs in,
    many outputs out, with per-input specialization" property once MoE
    sits in front of/inside it: the router sees BOTH a token's content AND
    a learned source embedding, so it can genuinely learn "route source A's
    tokens toward experts {2,5}, source B's toward {1,7}" -- a per-source
    specialization on top of ordinary within-source content routing -- the
    concrete mechanism behind "an internal router analyzes the type of
    input and allocates it to specific experts."

    All sources are concatenated into one (sum(B_i), d_model) token batch
    and pushed through a SINGLE SwitchMoE call, so adding sources costs
    router/gather overhead only, never extra kernel launches.
    """

    def __init__(self, d_model: int, num_sources: int, num_experts: int = 8,
                 expert_dim: int | None = None, top_k: int = 1,
                 capacity_factor: float = 1.5, router_noise_eps: float = 1e-2,
                 load_balance_alpha: float = 0.01, device=None, dtype=None):
        super().__init__()
        if num_sources < 1:
            raise ValueError(f"MultiSourceMoEBlock: num_sources must be >= 1, got {num_sources}")
        self.num_sources = num_sources
        self.norm = nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.source_embed = SourceEmbedding(num_sources, d_model, device=device, dtype=dtype)
        self.moe = SwitchMoE(d_model, num_experts=num_experts, expert_dim=expert_dim, top_k=top_k,
                             capacity_factor=capacity_factor, router_noise_eps=router_noise_eps,
                             load_balance_alpha=load_balance_alpha)
        if device is not None and getattr(device, "type", None) == "cuda":
            self.to(device)

    def forward(self, sources: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(sources) != self.num_sources:
            raise ValueError(f"MultiSourceMoEBlock: expected {self.num_sources} source tensors, "
                             f"got {len(sources)}")
        batch_sizes = [s.shape[0] for s in sources]
        tagged = torch.cat([self.source_embed(self.norm(s), i) for i, s in enumerate(sources)], dim=0)
        routed = self.moe(tagged)
        outs, offset = [], 0
        for i, b in enumerate(batch_sizes):
            outs.append(sources[i] + routed[offset:offset + b])  # per-source residual
            offset += b
        return outs

    def pop_aux_loss(self) -> torch.Tensor:
        return self.moe.pop_aux_loss()

    def last_diagnostics(self) -> dict:
        return self.moe.last_diagnostics()


class MoERNNWrapper(nn.Module):
    """Minimal external-MoE add-on for the stock nn.LSTM/GRU/RNN controller
    path (dnc.DNC's own rnn_type in {'lstm','gru','rnn'}). The plain
    baseline controller doesn't need per-block/per-source specialization --
    only a single Top-K MoE sublayer applied to its own per-step output --
    so this stays deliberately simpler than the Mamba/CfC wrappers'
    external-per-block interleaving.

    Preserves the exact `module(x.unsqueeze(1), hx) -> (out.unsqueeze(1),
    new_hx)` calling convention pytorch-dnc's `DNC._layer_forward` expects,
    so it substitutes for `self.rnns[layer]` with zero changes anywhere
    else in dnc.DNC's forward path.
    """

    def __init__(self, rnn: nn.Module, d_model: int, num_experts: int = 8,
                 expert_dim: int | None = None, top_k: int = 1,
                 capacity_factor: float = 1.5, load_balance_alpha: float = 0.01,
                 device=None, dtype=None):
        super().__init__()
        self.rnn = rnn
        self.d_model = d_model
        self.moe_enabled = True
        self.moe_blocks = nn.ModuleList([
            MoEBlock(d_model, num_experts=num_experts, expert_dim=expert_dim, top_k=top_k,
                    capacity_factor=capacity_factor, load_balance_alpha=load_balance_alpha,
                    device=device, dtype=dtype)
        ])

    def forward(self, input: torch.Tensor, hx):
        out, new_hx = self.rnn(input, hx)
        out = self.moe_blocks[0](out)
        return out, new_hx

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