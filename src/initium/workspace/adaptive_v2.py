"""Adaptive workspace skills with per-skill output subspaces and MI routing."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveWorkspaceSkillsV2(nn.Module):
    """Learn distinct low-rank skill pathways without task labels or heads.

    Compared with :class:`AdaptiveWorkspaceSkills`, every skill owns both its
    bottleneck and output projection. A small mutual-information objective
    encourages confident context-to-skill assignments while retaining broad
    skill use across a batch. The ordinary task loss remains the source of
    functional learning.
    """

    _INITIAL_OUTPUT_SCALE = 0.05
    _OUTPUT_WEIGHT_STD = 0.002

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        num_skills: int = 8,
        rank: int = 32,
        balance_loss_weight: float = 0.01,
        mutual_information_loss_weight: float = 0.01,
    ) -> None:
        super().__init__()
        if d_model < 1 or num_experts < 1 or num_skills < 2 or rank < 1:
            raise ValueError(
                "d_model, num_experts, and rank must be positive; num_skills must be at least two"
            )
        if not math.isfinite(balance_loss_weight) or balance_loss_weight < 0:
            raise ValueError("balance_loss_weight must be finite and non-negative")
        if not math.isfinite(mutual_information_loss_weight) or mutual_information_loss_weight < 0:
            raise ValueError("mutual_information_loss_weight must be finite and non-negative")

        self.d_model = int(d_model)
        self.num_experts = int(num_experts)
        self.num_skills = int(num_skills)
        self.rank = int(rank)
        self.balance_loss_weight = float(balance_loss_weight)
        self.mutual_information_loss_weight = float(mutual_information_loss_weight)
        self.context_width = 3 * self.d_model + self.num_experts

        self.context_norm = nn.LayerNorm(self.context_width)
        self.router = nn.Linear(self.context_width, self.num_skills, bias=False)
        self.skill_down = nn.Parameter(
            torch.empty(self.num_skills, self.context_width, self.rank)
        )
        self.skill_up = nn.Parameter(torch.empty(self.num_skills, self.rank, self.d_model))
        self.skill_bias = nn.Parameter(torch.zeros(self.num_skills, self.rank))
        initial_scale_logit = math.log(
            self._INITIAL_OUTPUT_SCALE / (1.0 - self._INITIAL_OUTPUT_SCALE)
        )
        self.output_scale_logit = nn.Parameter(
            torch.full((self.num_skills,), initial_scale_logit)
        )
        nn.init.xavier_uniform_(self.skill_down)
        nn.init.normal_(self.skill_up, mean=0.0, std=self._OUTPUT_WEIGHT_STD)

        self._last_aux_loss: torch.Tensor | None = None
        self._last_diagnostics: dict[str, torch.Tensor] = {}
        self._last_training_diagnostics: dict[str, torch.Tensor] = {}
        self._last_route_assignments: torch.Tensor | None = None
        self.eval_bypass = False

    def forward(
        self,
        backbone_features: torch.Tensor,
        specialist_deltas: torch.Tensor,
        route_indices: torch.Tensor,
        route_weights: torch.Tensor,
        workspace_broadcast: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        selected_specialists_only: bool = False,
    ) -> torch.Tensor:
        expected_prefix = backbone_features.shape[:2]
        expected_specialists = (
            route_indices.shape[-1] if selected_specialists_only else self.num_experts
        )
        if (
            backbone_features.ndim != 3
            or backbone_features.shape[-1] != self.d_model
            or specialist_deltas.ndim != 4
            or specialist_deltas.shape[:2] != expected_prefix
            or specialist_deltas.shape[2] != expected_specialists
            or specialist_deltas.shape[-1] != self.d_model
            or workspace_broadcast.shape != backbone_features.shape
            or route_indices.ndim != 3
            or route_indices.shape[:2] != expected_prefix
            or route_weights.shape != route_indices.shape
            or (valid_mask is not None and valid_mask.shape != expected_prefix)
        ):
            raise ValueError(
                "adaptive workspace v2 inputs have incompatible shapes: "
                f"backbone={tuple(backbone_features.shape)}, "
                f"specialists={tuple(specialist_deltas.shape)}, "
                f"routes={tuple(route_indices.shape)}, "
                f"route_weights={tuple(route_weights.shape)}, "
                f"broadcast={tuple(workspace_broadcast.shape)}, "
                f"valid_mask={tuple(valid_mask.shape) if valid_mask is not None else None}, "
                f"selected_only={selected_specialists_only}"
            )
        if self.eval_bypass:
            self._last_aux_loss = backbone_features.new_zeros(())
            self._last_diagnostics = {}
            self._last_route_assignments = None
            return torch.zeros_like(backbone_features)

        selected_deltas = (
            specialist_deltas
            if selected_specialists_only
            else specialist_deltas.gather(
                2,
                route_indices.long().unsqueeze(-1).expand(-1, -1, -1, self.d_model),
            )
        )
        routed_context = (
            selected_deltas * route_weights.to(selected_deltas.dtype).unsqueeze(-1)
        ).sum(dim=2)
        route_code = torch.zeros(
            (*expected_prefix, self.num_experts),
            device=route_indices.device,
            dtype=torch.float32,
        ).scatter_add_(2, route_indices.long(), route_weights.float())
        context = self.context_norm(
            torch.cat(
                (backbone_features, routed_context, workspace_broadcast, route_code), dim=-1
            ).float()
        )

        probabilities = self.router(context).float().softmax(dim=-1)
        selected_skill = probabilities.argmax(dim=-1)
        hard_routes = F.one_hot(selected_skill, num_classes=self.num_skills).to(
            probabilities.dtype
        )
        token_mask = (
            torch.ones(expected_prefix, device=context.device, dtype=torch.float32)
            if valid_mask is None
            else valid_mask.to(device=context.device, dtype=torch.float32)
        )
        token_count = token_mask.sum().clamp_min(1.0)

        # Accumulate per-skill projections one at a time. This avoids a
        # (B,T,skills,D) output tensor while preserving straight-through
        # gradients from the soft router mixture.
        hard_delta = context.new_zeros((*expected_prefix, self.d_model))
        router_surrogate_delta = torch.zeros_like(hard_delta)
        hard_raw_delta = torch.zeros_like(hard_delta)
        scales = self.output_scale_logit.sigmoid()
        for skill_index in range(self.num_skills):
            hidden = F.linear(
                context,
                self.skill_down[skill_index].transpose(0, 1),
                self.skill_bias[skill_index],
            ).tanh()
            skill_output = F.linear(
                hidden,
                self.skill_up[skill_index].transpose(0, 1),
            )
            scale = scales[skill_index]
            hard_weight = hard_routes[..., skill_index : skill_index + 1]
            soft_weight = probabilities[..., skill_index : skill_index + 1]
            hard_raw_delta = hard_raw_delta + skill_output * hard_weight
            hard_delta = hard_delta + skill_output * scale * hard_weight
            router_surrogate_delta = router_surrogate_delta + (
                skill_output * scale
            ).detach() * soft_weight

        adaptive_delta = (
            hard_delta + router_surrogate_delta - router_surrogate_delta.detach()
        )

        hard_load = (hard_routes * token_mask.unsqueeze(-1)).sum(dim=(0, 1)) / token_count
        importance = (probabilities * token_mask.unsqueeze(-1)).sum(dim=(0, 1)) / token_count
        balance_loss = self.num_skills * (hard_load * importance).sum() - 1.0
        conditional_entropy = (
            -(probabilities * probabilities.clamp_min(1e-8).log())
            .sum(dim=-1)
            .mul(token_mask)
            .sum()
            .div(token_count)
        )
        marginal_entropy = -(
            importance * importance.clamp_min(1e-8).log()
        ).sum()
        mutual_information = (marginal_entropy - conditional_entropy).clamp_min(0.0)
        self._last_aux_loss = (
            balance_loss * self.balance_loss_weight
            - mutual_information * self.mutual_information_loss_weight
        )
        self._last_route_assignments = selected_skill.detach()
        self._last_diagnostics = {
            "route_frequency": hard_load.detach(),
            "route_probability": importance.detach(),
            "route_entropy": conditional_entropy.detach(),
            "route_marginal_entropy": marginal_entropy.detach(),
            "routing_mutual_information": mutual_information.detach(),
            "skill_delta_rms": (
                (hard_raw_delta.detach().float().square() * token_mask.unsqueeze(-1))
                .sum()
                .div(token_count * self.d_model)
                .clamp_min(1e-12)
                .sqrt()
            ),
            "adaptive_delta_rms": (
                (adaptive_delta.detach().float().square() * token_mask.unsqueeze(-1))
                .sum()
                .div(token_count * self.d_model)
                .clamp_min(1e-12)
                .sqrt()
            ),
            "output_scale": scales.detach(),
            "balance_loss": balance_loss.detach(),
            "mutual_information_loss": (-mutual_information).detach(),
        }
        if self.training:
            self._last_training_diagnostics = self._last_diagnostics
        return adaptive_delta.to(dtype=backbone_features.dtype)

    def pop_aux_loss(self) -> torch.Tensor | None:
        """Return and clear the latest routing objective."""
        loss = self._last_aux_loss
        self._last_aux_loss = None
        return loss

    def last_diagnostics(self) -> dict[str, torch.Tensor]:
        """Return measurements from the latest training batch."""
        return self._last_training_diagnostics

    def last_route_assignments(self) -> torch.Tensor | None:
        """Return per-token selected pathways for task-stratified evaluation."""
        return self._last_route_assignments
