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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """One feed-forward expert: Linear -> ReLU -> Linear, matching
    MoE-Mamba's/Switch's own expert architecture (a single hidden layer,
    ReLU-activated) rather than anything more exotic -- keeping this
    minimal is itself corpus-consistent (Dead-End #43: fancier internal
    placements underperformed the plain external design)."""

    w_in: nn.Linear
    w_out: nn.Linear

    def __init__(self, d_model: int, expert_dim: int):
        super().__init__()
        self.w_in = nn.Linear(d_model, expert_dim)
        self.w_out = nn.Linear(expert_dim, d_model)
        # Make inserting an MoE residual a no-op at initialization.  The
        # output projections still receive task gradients on the first
        # update; their hidden projections begin learning as soon as those
        # output weights move away from zero.  This avoids a randomly
        # initialized expert bank changing the controller's signal scale
        # before routing/expert specialization has learned anything.
        nn.init.zeros_(self.w_out.weight)
        nn.init.zeros_(self.w_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(F.relu(self.w_in(x)))


class SwitchMoE(nn.Module):
    """Switch-style top-k sparsely-gated MoE feed-forward sublayer.
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
        self._diag_accum: dict[str, torch.Tensor] = {}
        self._diag_calls = 0
        self._last_topk_idx: torch.Tensor | None = (
            None  # (T, k), detached -- for per-source breakdowns
        )
        self._last_accepted: torch.Tensor | None = None  # (T, k), detached

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_details: bool = False,
        token_mask: torch.Tensor | None = None,
    ):
        orig_shape = x.shape
        d_model = orig_shape[-1]
        assert d_model == self.d_model, (
            f"SwitchMoE: input last dim {d_model} != configured d_model {self.d_model}"
        )
        flat = x.reshape(-1, d_model)  # (num_tokens, d_model)
        num_tokens = flat.shape[0]
        if token_mask is not None:
            if tuple(token_mask.shape) != tuple(orig_shape[:-1]):
                raise ValueError(
                    "SwitchMoE: token_mask must match the leading input dimensions "
                    f"{tuple(orig_shape[:-1])}, got {tuple(token_mask.shape)}"
                )
            token_mask_flat = token_mask.reshape(-1).to(device=x.device, dtype=torch.bool)
        else:
            token_mask_flat = None

        # ``.float()`` after Linear is too late under autocast: CUDA AMP may
        # execute the router matmul in fp16 and only cast its already-rounded
        # result back to fp32.  Switch's selective-precision rule requires the
        # projection itself to run in fp32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            flat32 = flat.float()
            logits32 = F.linear(
                flat32,
                self.router.weight.float(),
                None if self.router.bias is None else self.router.bias.float(),
            )
        if self.training and self.router_noise_eps > 0:
            # Switch's own exploration mechanism (Appendix C): multiplicative
            # jitter noise. We apply it directly to the router logits here
            # (a commonly-used equivalent of jittering the router input) for
            # simplicity.
            noise = torch.empty_like(logits32).uniform_(
                1.0 - self.router_noise_eps, 1.0 + self.router_noise_eps
            )
            logits32 = logits32 * noise
        probs = torch.softmax(logits32, dim=-1)  # (num_tokens, num_experts)

        # Top-K routing (Shazeer et al. 2017 / GShard), generalizing Switch's
        # own k=1 special case: top-1 retains its selected probability from
        # the full expert softmax (so the task objective can train the
        # router); top-k>1 normalizes among the selected experts.
        # Normal training dispatches selected, capacity-accepted routes into
        # bounded per-expert buffers. The optional all-expert result is
        # reserved for the workspace specialist path, which explicitly needs
        # every specialist candidate. This keeps top-k routing sparse in
        # compute and avoids expanding expert weights per route.
        top_k = self.top_k
        topk_prob, topk_idx = probs.topk(top_k, dim=-1)  # (T, k) each
        if top_k == 1:
            # Switch's top-1 gate multiplies the expert output by its
            # probability in the full expert softmax. Renormalizing a
            # one-element selected set would make every gate exactly 1 and
            # cancel the softmax derivative, leaving the task loss unable
            # to train the router at all.
            gate_weights = topk_prob
        else:
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
            self._last_topk_idx = topk_idx.detach()

        # Expert capacity, scaled by top_k (each slot competes for the same
        # per-expert buffer).
        capacity = max(1, int((num_tokens * top_k / self.num_experts) * self.capacity_factor))

        # Capacity is assigned by router confidence, not flattened token
        # order.  In recurrent multi-source calls the source batches are
        # concatenated (backbone first, memory read second); first-come
        # capacity therefore silently favored the first source whenever an
        # expert overflowed.  Confidence-priority acceptance is independent
        # of source ordering and preserves the strongest routes.
        flat_idx = topk_idx.reshape(-1)
        flat_priority = topk_prob.reshape(-1)
        priority_order = torch.argsort(flat_priority, descending=True, stable=True)
        expert_in_priority_order = flat_idx.index_select(0, priority_order)
        one_hot_priority = F.one_hot(
            expert_in_priority_order, num_classes=self.num_experts
        )
        ranks_in_priority_order = one_hot_priority.cumsum(dim=0).gather(
            1, expert_in_priority_order.unsqueeze(1)
        ).squeeze(1) - 1
        rank_in_expert = torch.empty_like(ranks_in_priority_order)
        rank_in_expert.scatter_(0, priority_order, ranks_in_priority_order)
        keep = (rank_in_expert < capacity).reshape(num_tokens, top_k)
        accepted_gates = gate_weights * keep.to(gate_weights.dtype)
        if top_k > 1:
            accepted_gates = accepted_gates / accepted_gates.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-9)

        with torch.autocast(device_type=x.device.type, enabled=False):
            # Dispatch activations into fixed-capacity expert buffers. The
            # previous batched-weight implementation expanded each expert's
            # (D, hidden) matrices once per token/route; autograd retained
            # those copies for every recurrent call (96 MiB for just one
            # 32-token, 512x1536 projection). Running each expert's Linear on
            # its capacity buffer shares the original Parameter storage and
            # limits saved activations to dispatched tokens.
            route_slots = rank_in_expert.clamp(max=capacity - 1)
            route_tokens = torch.arange(num_tokens, device=x.device).repeat_interleave(top_k)
            route_keep = keep.reshape(-1)
            route_inputs = flat32.index_select(0, route_tokens)
            route_inputs = route_inputs * route_keep.unsqueeze(-1).to(route_inputs.dtype)
            dispatch_indices = flat_idx * capacity + route_slots
            dispatch = flat32.new_zeros((self.num_experts * capacity, d_model))
            dispatch = dispatch.index_add(0, dispatch_indices, route_inputs)
            dispatch = dispatch.reshape(self.num_experts, capacity, d_model)

            expert_outputs = []
            for expert_index, expert in enumerate(self.experts):
                expert_input = dispatch[expert_index]
                expert_hidden = F.linear(
                    expert_input, expert.w_in.weight.float(), expert.w_in.bias.float()
                ).relu()
                expert_outputs.append(
                    F.linear(
                        expert_hidden,
                        expert.w_out.weight.float(),
                        expert.w_out.bias.float(),
                    )
                )
            expert_outputs = torch.stack(expert_outputs, dim=0)

            # Both advanced indices must have identical (T, top_k) shapes.
            # Leaving flat_idx flat (T*top_k,) makes PyTorch broadcast it
            # against the (T, top_k) slot index. For top_k=1 this silently
            # creates a (T, T, D) selection, mixing unrelated tokens and
            # summing T expert outputs per token; the resulting oversized
            # MoE delta caused the immediate forward-scale and NaN-gradient
            # failure in the Phase 1 trace.
            route_experts = flat_idx.reshape(num_tokens, top_k)
            route_slots_2d = route_slots.reshape(num_tokens, top_k)
            selected_outputs = expert_outputs[route_experts, route_slots_2d]
            output32 = (selected_outputs * accepted_gates.unsqueeze(-1)).sum(dim=1)
            accepted_outputs = selected_outputs * keep.unsqueeze(-1).to(selected_outputs.dtype)

            expert_out_all = None
            if return_details:
                # Workspace broadcasting is the one caller that needs all
                # specialist candidates, including experts not selected by
                # the sparse route. Keep this expensive activation path
                # explicit, while still using each expert's shared weights.
                expert_out_all = torch.stack(
                    [
                        F.linear(
                            F.relu(
                                F.linear(
                                    flat32,
                                    expert.w_in.weight.float(),
                                    expert.w_in.bias.float(),
                                )
                            ),
                            expert.w_out.weight.float(),
                            expert.w_out.bias.float(),
                        )
                        for expert in self.experts
                    ],
                    dim=1,
                )

        # Keep the result in fp32 when the input was autocast to fp16/bf16.
        # Casting a large but finite fp32 expert result back to fp16 can
        # create an Inf before the residual or DNC path has a chance to
        # stabilize it. The surrounding residual naturally promotes to fp32.
        output_dtype = (
            torch.float32
            if flat.dtype in (torch.float16, torch.bfloat16)
            else flat.dtype
        )
        output = output32.to(output_dtype)

        with torch.no_grad():
            self._last_accepted = keep.detach()
            self._last_diag = {
                "cv_importance": (P_i.std() / P_i.mean().clamp(min=1e-8)).detach(),
                "cv_load": (f_i.std() / f_i.mean().clamp(min=1e-8)).detach(),
                "max_load_frac": f_i.max().detach(),
                "expert_load_frac": f_i.detach(),
                "router_importance_frac": P_i.detach(),
                "router_entropy": (
                    -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
                ).detach(),
                "router_max_prob": probs.max(dim=-1).values.mean().detach(),
                "topk_gate_mean": topk_prob.mean().detach(),
                "input_rms": flat32.square().mean().sqrt().detach(),
                "input_absmax": flat32.abs().amax().detach(),
                "expert_output_rms": accepted_outputs.square().mean().sqrt().detach(),
                "expert_output_absmax": accepted_outputs.abs().amax().detach(),
                "routed_output_rms": output32.square().mean().sqrt().detach(),
                "routed_output_absmax": output32.abs().amax().detach(),
                "capacity_drop_frac": (1.0 - keep.float().mean()).detach(),
                "accepted_gate_mass": accepted_gates.sum(dim=-1).mean().detach(),
            }
            if token_mask_flat is not None:
                valid = token_mask_flat
                padded = ~valid
                per_token_entropy = -(
                    probs * probs.clamp_min(1e-9).log()
                ).sum(dim=-1)
                per_token_max_prob = probs.max(dim=-1).values
                per_token_drop = 1.0 - keep.float().mean(dim=-1)

                def subset_routing_stats(subset: torch.Tensor, prefix: str) -> None:
                    subset_weight = subset.to(probs.dtype)
                    count = subset_weight.sum()
                    denom = count.clamp_min(1.0)
                    subset_load = (
                        one_hot_k
                        * subset_weight[:, None, None]
                    ).sum(dim=(0, 1)) / (denom * top_k)
                    subset_importance = (
                        probs * subset_weight[:, None]
                    ).sum(dim=0) / denom
                    entropy = (per_token_entropy * subset_weight).sum() / denom
                    max_prob = (per_token_max_prob * subset_weight).sum() / denom
                    drop_frac = (per_token_drop * subset_weight).sum() / denom
                    self._last_diag[f"{prefix}_token_count"] = count.detach()
                    self._last_diag[f"{prefix}_expert_load_frac"] = subset_load.detach()
                    self._last_diag[f"{prefix}_router_importance_frac"] = (
                        subset_importance.detach()
                    )
                    self._last_diag[f"{prefix}_router_entropy"] = entropy.detach()
                    self._last_diag[f"{prefix}_router_max_prob"] = max_prob.detach()
                    self._last_diag[f"{prefix}_capacity_drop_frac"] = drop_frac.detach()

                subset_routing_stats(valid, "valid")
                subset_routing_stats(padded, "padded")
            if self.training:
                for key, value in self._last_diag.items():
                    self._diag_accum[key] = (
                        self._diag_accum[key] + value
                        if key in self._diag_accum
                        else value.clone()
                    )
                self._diag_calls += 1

        output = output.reshape(orig_shape)
        if return_details:
            expert_shape = (*orig_shape[:-1], self.num_experts, d_model)
            route_shape = (*orig_shape[:-1], top_k)
            return (
                output,
                expert_out_all.to(output_dtype).reshape(expert_shape),
                topk_idx.reshape(route_shape),
                accepted_gates.reshape(route_shape),
            )
        return output

    def pop_aux_loss(self) -> torch.Tensor:
        """Consume and clear the accumulated load-balancing loss (mirrors
        StochasticWriteHead.pop_kl()'s accumulate-then-pop convention in
        stochastic_write_head_v2.py, so the training loop's pattern for
        combining an auxiliary loss with the task loss stays consistent
        across both mechanisms)."""
        if not self._aux_losses:
            return torch.zeros((), device=self.router.weight.device)
        # A recurrent controller invokes this layer once per timestep.  The
        # Switch objective is per routing batch, so sum() would silently
        # multiply its strength by episode length.  Average calls within this
        # layer; pop_total_moe_aux_loss still sums distinct model layers.
        total = torch.stack(self._aux_losses).mean()
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

    def pop_diagnostics(self) -> dict:
        """Consume mean training diagnostics accumulated since the last pop."""
        if not self._diag_calls:
            return {}
        result = {key: value / self._diag_calls for key, value in self._diag_accum.items()}
        self._diag_accum = {}
        self._diag_calls = 0
        return result

    def last_routing(self) -> torch.Tensor | None:
        """(num_tokens, top_k) expert indices chosen on the most recent
        forward() call, or None before any call. Callers that know the
        token-batch's composition (e.g. MultiSourceMoEBlock, which knows
        which token rows came from which source) slice this to compute
        per-slice routing breakdowns that this pooled-over-all-tokens
        class has no notion of on its own."""
        return self._last_topk_idx

    def last_accepted_routing(self) -> torch.Tensor | None:
        """Capacity acceptance mask ``(num_tokens, top_k)`` for the most
        recent call, or ``None`` before the first call."""
        return self._last_accepted


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
        residual_scale: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if not math.isfinite(residual_scale) or residual_scale < 0.0:
            raise ValueError(
                f"MoEBlock residual_scale must be finite and >= 0, got {residual_scale!r}"
            )
        self.residual_scale = float(residual_scale)
        self.norm = nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.moe = SwitchMoE(
            d_model,
            num_experts=num_experts,
            expert_dim=expert_dim,
            capacity_factor=capacity_factor,
            router_noise_eps=router_noise_eps,
            load_balance_alpha=load_balance_alpha,
            top_k=top_k,
        )
        self._cumulative_source_counts: list[torch.Tensor] | None = (
            None  # per-source, (num_experts,)
        )
        if device is not None and getattr(device, "type", None) == "cuda":
            self.to(device)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_details: bool = False,
        token_mask: torch.Tensor | None = None,
    ):
        if not return_details:
            routed = self.moe(self.norm(x), token_mask=token_mask)
            return x + self.residual_scale * routed
        routed, experts, indices, weights = self.moe(
            self.norm(x), return_details=True, token_mask=token_mask
        )
        return x + self.residual_scale * routed, experts, indices, weights

    def pop_aux_loss(self) -> torch.Tensor:
        return self.moe.pop_aux_loss()

    def last_diagnostics(self) -> dict:
        return self.moe.last_diagnostics()

    def pop_diagnostics(self) -> dict:
        return self.moe.pop_diagnostics()


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
    through ONE shared bank of Top-K experts (SwitchMoE.forward computes
    only selected, capacity-accepted token/expert pairs).

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

    _cumulative_source_counts: list[torch.Tensor] | None
    _cumulative_source_accepted: list[torch.Tensor] | None
    _cumulative_source_routes: list[torch.Tensor] | None

    def __init__(
        self,
        d_model: int,
        num_sources: int,
        num_experts: int = 8,
        expert_dim: int | None = None,
        top_k: int = 1,
        capacity_factor: float = 1.5,
        router_noise_eps: float = 1e-2,
        load_balance_alpha: float = 0.01,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if num_sources < 1:
            raise ValueError(f"MultiSourceMoEBlock: num_sources must be >= 1, got {num_sources}")
        self.num_sources = num_sources
        self.norm = nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.source_embed = SourceEmbedding(num_sources, d_model, device=device, dtype=dtype)
        self.moe = SwitchMoE(
            d_model,
            num_experts=num_experts,
            expert_dim=expert_dim,
            top_k=top_k,
            capacity_factor=capacity_factor,
            router_noise_eps=router_noise_eps,
            load_balance_alpha=load_balance_alpha,
        )
        self._cumulative_source_counts = None
        self._cumulative_source_accepted = None
        self._cumulative_source_routes = None
        if device is not None and getattr(device, "type", None) == "cuda":
            self.to(device)

    def forward(self, sources: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(sources) != self.num_sources:
            raise ValueError(
                f"MultiSourceMoEBlock: expected {self.num_sources} source tensors, "
                f"got {len(sources)}"
            )
        batch_sizes = [s.shape[0] for s in sources]
        tagged = torch.cat(
            [self.source_embed(self.norm(s), i) for i, s in enumerate(sources)], dim=0
        )
        routed = self.moe(tagged)

        # Per-source routing breakdown: which experts did THIS source's rows
        # actually go to, independent of the pooled cv_load/cv_importance
        # SwitchMoE.last_diagnostics() reports over the whole concatenated
        # batch. This is the direct evidence for (or against) genuine
        # per-source specialization -- e.g. "source 0 (backbone output)
        # concentrates on experts {2,5}, source 1 (read vector) on {1,7}"
        # -- rather than every source routing near-identically by chance.
        topk_idx = self.moe.last_routing()  # (sum(batch_sizes), top_k) or None
        accepted = self.moe.last_accepted_routing()
        if topk_idx is not None and accepted is not None:
            num_experts = self.moe.num_experts
            if self._cumulative_source_counts is None:
                self._cumulative_source_counts = [
                    torch.zeros(num_experts, device=topk_idx.device) for _ in batch_sizes
                ]
                self._cumulative_source_accepted = [
                    torch.zeros(num_experts, device=topk_idx.device) for _ in batch_sizes
                ]
                self._cumulative_source_routes = [
                    torch.zeros((), device=topk_idx.device) for _ in batch_sizes
                ]
            offset = 0
            for i, b in enumerate(batch_sizes):
                idx_slice = topk_idx[offset : offset + b].reshape(-1)
                counts = torch.bincount(idx_slice, minlength=num_experts).float()
                accepted_slice = accepted[offset : offset + b].reshape(-1)
                accepted_counts = torch.bincount(
                    idx_slice,
                    weights=accepted_slice.to(dtype=torch.float32),
                    minlength=num_experts,
                )
                # Keep routing counters on device and defer CPU synchronization
                # until inference explicitly requests a report. This block runs
                # once per CfC timestep during training.
                self._cumulative_source_counts[i] += counts
                assert self._cumulative_source_accepted is not None
                assert self._cumulative_source_routes is not None
                self._cumulative_source_accepted[i] += accepted_counts
                self._cumulative_source_routes[i] += accepted_slice.sum()
                offset += b

        outs, offset = [], 0
        for i, b in enumerate(batch_sizes):
            outs.append(sources[i] + routed[offset : offset + b])  # per-source residual
            offset += b
        return outs

    def pop_aux_loss(self) -> torch.Tensor:
        return self.moe.pop_aux_loss()

    def last_diagnostics(self) -> dict:
        return self.moe.last_diagnostics()

    def pop_diagnostics(self) -> dict:
        return self.moe.pop_diagnostics()

    def last_source_diagnostics(self) -> list[dict] | None:
        """Per-source expert-usage breakdown from the most recent forward()
        call: one dict per source (same order as the `sources` list passed
        in), each with `expert_frac` (this source's rows' distribution over
        experts), `top_expert`, and `top_expert_frac`. None before any call.
        Comparing this ACROSS sources is the concrete test of whether the
        router is genuinely specializing by input identity rather than
        routing every source near-identically."""
        if self._cumulative_source_counts is None:
            return None
        source_diag = []
        assert self._cumulative_source_accepted is not None
        assert self._cumulative_source_routes is not None
        for counts, accepted_counts, accepted_routes in zip(
            self._cumulative_source_counts,
            self._cumulative_source_accepted,
            self._cumulative_source_routes,
        ):
            counts_cpu = counts.detach().float().cpu()
            accepted_cpu = accepted_counts.detach().float().cpu()
            fractions = counts_cpu / counts_cpu.sum().clamp_min(1.0)
            top_expert = int(counts_cpu.argmax().item())
            total_routes = float(counts_cpu.sum().item())
            accepted_total = float(accepted_cpu.sum().item())
            source_diag.append(
                {
                    "expert_frac": fractions.tolist(),
                    "top_expert": top_expert,
                    "top_expert_frac": float(fractions[top_expert]),
                    "accepted_expert_frac": (
                        accepted_cpu / accepted_cpu.sum().clamp_min(1.0)
                    ).tolist(),
                    "accepted_route_frac": accepted_total / max(total_routes, 1.0),
                    "capacity_drop_frac": 1.0 - accepted_total / max(total_routes, 1.0),
                    "n_routes_accumulated": int(total_routes),
                    "n_accepted_accumulated": int(accepted_routes.detach().item()),
                }
            )
        return source_diag

    def pop_source_diagnostics(self) -> list[dict] | None:
        """Return and reset per-source routing/capacity counters.

        This is intended for periodic training logs, where source-level
        overflow must be compared over the same window as the pooled MoE
        diagnostics. ``last_source_diagnostics`` remains cumulative for
        inference callers.
        """
        report = self.last_source_diagnostics()
        if report is not None:
            assert self._cumulative_source_counts is not None
            assert self._cumulative_source_accepted is not None
            assert self._cumulative_source_routes is not None
            for counts, accepted, routes in zip(
                self._cumulative_source_counts,
                self._cumulative_source_accepted,
                self._cumulative_source_routes,
            ):
                counts.zero_()
                accepted.zero_()
                routes.zero_()
        return report


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

    def __init__(
        self,
        rnn: nn.Module,
        d_model: int,
        num_experts: int = 8,
        expert_dim: int | None = None,
        top_k: int = 1,
        capacity_factor: float = 1.5,
        load_balance_alpha: float = 0.01,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.rnn = rnn
        self.d_model = d_model
        self.moe_enabled = True
        self.moe_blocks = nn.ModuleList(
            [
                MoEBlock(
                    d_model,
                    num_experts=num_experts,
                    expert_dim=expert_dim,
                    top_k=top_k,
                    capacity_factor=capacity_factor,
                    load_balance_alpha=load_balance_alpha,
                    device=device,
                    dtype=dtype,
                )
            ]
        )

    def forward(self, input: torch.Tensor, hx):
        out, new_hx = self.rnn(input, hx)
        out = self.moe_blocks[0](out)
        return out, new_hx


def pop_total_moe_aux_loss(moe_layers: list, *, include_diagnostics: bool = True) -> tuple:
    """Sum pop_aux_loss() across all installed MoEBlock/SwitchMoE layers and
    merge their diagnostics (mean across layers for CV metrics, max for the
    worst-case load fraction), mirroring stochastic_write_head_v2's
    pop_total_kl() pattern exactly, so the training loop combines this
    auxiliary loss the same way it already combines the KL loss."""
    total = None
    cv_imp, cv_load, max_load, capacity_drop = [], [], [], []
    layer_reports = []
    for layer in moe_layers:
        loss = layer.pop_aux_loss()
        total = loss if total is None else total + loss
        if include_diagnostics:
            pop_diagnostics = getattr(layer, "pop_diagnostics", None)
            diag = pop_diagnostics() if pop_diagnostics is not None else layer.last_diagnostics()
        else:
            diag = {}
        if diag:
            source_report = None
            if include_diagnostics:
                pop_source_diagnostics = getattr(layer, "pop_source_diagnostics", None)
                if pop_source_diagnostics is not None:
                    source_report = pop_source_diagnostics()
            cv_imp.append(float(diag["cv_importance"]))
            cv_load.append(float(diag["cv_load"]))
            max_load.append(float(diag["max_load_frac"]))
            capacity_drop.append(float(diag.get("capacity_drop_frac", 0.0)))
            layer_reports.append(
                {
                    "name": getattr(layer, "diagnostic_name", type(layer).__name__),
                    **{
                        key: float(diag[key])
                        for key in (
                            "cv_importance",
                            "cv_load",
                            "max_load_frac",
                            "router_entropy",
                            "router_max_prob",
                            "topk_gate_mean",
                            "input_rms",
                            "input_absmax",
                            "expert_output_rms",
                            "expert_output_absmax",
                            "routed_output_rms",
                            "routed_output_absmax",
                            "capacity_drop_frac",
                            "accepted_gate_mass",
                            "valid_token_count",
                            "padded_token_count",
                            "valid_router_entropy",
                            "valid_router_max_prob",
                            "valid_capacity_drop_frac",
                            "padded_router_entropy",
                            "padded_router_max_prob",
                            "padded_capacity_drop_frac",
                        )
                        if key in diag
                    },
                    **{
                        key: diag[key].detach().float().cpu().tolist()
                        for key in (
                            "expert_load_frac",
                            "router_importance_frac",
                            "valid_expert_load_frac",
                            "valid_router_importance_frac",
                            "padded_expert_load_frac",
                            "padded_router_importance_frac",
                        )
                        if key in diag
                    },
                    **({"source_routes": source_report} if source_report is not None else {}),
                }
            )
    if total is None:
        total = torch.zeros(())
    merged = {
        "moe_cv_importance": sum(cv_imp) / len(cv_imp) if cv_imp else 0.0,
        "moe_cv_load": sum(cv_load) / len(cv_load) if cv_load else 0.0,
        "moe_max_load_frac": max(max_load) if max_load else 0.0,
        "moe_capacity_drop_frac": (
            sum(capacity_drop) / len(capacity_drop) if capacity_drop else 0.0
        ),
        "moe_layers": layer_reports,
    }
    return total, merged
