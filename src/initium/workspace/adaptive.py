"""Task-loss-trained functional specialization for the shared workspace.

The skill bank has fixed capacity, but its router and low-rank pathways are
learned jointly from the ordinary task loss. It receives backbone, routed
specialist, and workspace-broadcast context without task IDs or task-specific
decoders. Sparse routing lets the bank organize into useful pathways during
training while retaining a small activation and parameter footprint.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveWorkspaceSkills(nn.Module):
    """Learn sparse low-rank skill pathways conditioned on global workspace context.

    Args:
        d_model: Width of backbone and specialist representations.
        num_experts: Number of backbone MoE experts available to the workspace.
        num_skills: Number of initially available skill pathways.
        rank: Bottleneck width of each pathway.
        balance_loss_weight: Weight applied to the router load-balance objective.

    Raises:
        ValueError: If dimensions are non-positive or the balance weight is invalid.
    """

    _INITIAL_OUTPUT_SCALE = 0.05
    _OUTPUT_WEIGHT_STD = 0.002

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        num_skills: int = 4,
        rank: int = 32,
        balance_loss_weight: float = 0.01,
    ) -> None:
        super().__init__()
        if d_model < 1 or num_experts < 1 or num_skills < 2 or rank < 1:
            raise ValueError(
                "d_model, num_experts, and rank must be positive; num_skills must be at least two"
            )
        if not math.isfinite(balance_loss_weight) or balance_loss_weight < 0:
            raise ValueError("balance_loss_weight must be finite and non-negative")
        self.d_model = int(d_model)
        self.num_experts = int(num_experts)
        self.num_skills = int(num_skills)
        self.rank = int(rank)
        self.balance_loss_weight = float(balance_loss_weight)
        context_width = 3 * self.d_model + self.num_experts
        self.context_norm = nn.LayerNorm(context_width)
        self.router = nn.Linear(context_width, self.num_skills, bias=False)
        self.skill_down = nn.Parameter(torch.empty(self.num_skills, context_width, self.rank))
        self.skill_up = nn.Parameter(torch.empty(self.rank, self.d_model))
        self.skill_bias = nn.Parameter(torch.zeros(self.num_skills, self.rank))
        self.output_scale_logit = nn.Parameter(
            torch.tensor(math.log(self._INITIAL_OUTPUT_SCALE / (1.0 - self._INITIAL_OUTPUT_SCALE)))
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
        """Return a task-trained sparse skill residual for ``(batch, time, width)`` inputs.

        Args:
            backbone_features: Pre-workspace backbone output, shaped ``(B,T,D)``.
            specialist_deltas: MoE candidate outputs, shaped ``(B,T,E,D)`` when
                all experts are present, or ``(B,T,K,D)`` when only routed
                experts are present in route order.
            route_indices: Selected expert indices, shaped ``(B,T,K)``.
            route_weights: Selected expert weights, shaped ``(B,T,K)``.
            workspace_broadcast: Workspace-enriched MoE residual, shaped ``(B,T,D)``.
            valid_mask: Optional token mask shaped ``(B,T)``; padding is excluded
                from routing balance and activation statistics.
            selected_specialists_only: Whether ``specialist_deltas`` contains
                only routed candidates, aligned positionally with route indices.

        Returns:
            A scaled adaptive residual shaped ``(B,T,D)``.

        Raises:
            ValueError: If the batch, time, or feature dimensions do not agree.
        """
        expected_prefix = backbone_features.shape[:2]
        if (
            backbone_features.ndim != 3
            or backbone_features.shape[-1] != self.d_model
            or specialist_deltas.ndim != 4
            or specialist_deltas.shape[:2] != expected_prefix
            or specialist_deltas.shape[2]
            != (route_indices.shape[-1] if selected_specialists_only else self.num_experts)
            or specialist_deltas.shape[-1] != self.d_model
            or workspace_broadcast.shape != backbone_features.shape
            or route_indices.shape[:2] != expected_prefix
            or route_weights.shape != route_indices.shape
            or (valid_mask is not None and valid_mask.shape != expected_prefix)
        ):
            raise ValueError(
                "adaptive workspace inputs have incompatible shapes: "
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
        routed_context = (selected_deltas * route_weights.to(selected_deltas.dtype).unsqueeze(-1)).sum(
            dim=2
        )
        route_code = torch.zeros(
            (*expected_prefix, self.num_experts),
            device=route_indices.device,
            dtype=torch.float32,
        ).scatter_add_(2, route_indices.long(), route_weights.float())
        context = torch.cat(
            (backbone_features, routed_context, workspace_broadcast, route_code), dim=-1
        )
        context = self.context_norm(context.float())
        router_probabilities = self.router(context).float().softmax(dim=-1)
        top_skill = router_probabilities.argmax(dim=-1)
        hard_routes = F.one_hot(top_skill, num_classes=self.num_skills).to(router_probabilities.dtype)
        straight_through_routes = hard_routes + router_probabilities - router_probabilities.detach()
        token_mask = (
            torch.ones(expected_prefix, device=backbone_features.device, dtype=torch.float32)
            if valid_mask is None
            else valid_mask.to(device=backbone_features.device, dtype=torch.float32)
        )
        valid_token_count = token_mask.sum().clamp_min(1.0)

        skill_hidden = torch.einsum("bti,sir->btsr", context, self.skill_down)
        skill_hidden = (skill_hidden + self.skill_bias.view(1, 1, self.num_skills, self.rank)).tanh()
        routed_skill_state = (skill_hidden * straight_through_routes.unsqueeze(-1)).sum(dim=2)
        skill_residual = F.linear(routed_skill_state, self.skill_up.transpose(0, 1))
        output_scale = self.output_scale_logit.sigmoid()
        adaptive_delta = skill_residual.to(backbone_features.dtype) * output_scale.to(
            backbone_features.dtype
        )

        hard_load = (hard_routes * token_mask.unsqueeze(-1)).sum(dim=(0, 1)) / valid_token_count
        importance = (router_probabilities * token_mask.unsqueeze(-1)).sum(dim=(0, 1)) / valid_token_count
        balance_loss = self.num_skills * (hard_load * importance).sum() - 1.0
        self._last_aux_loss = balance_loss * self.balance_loss_weight
        self._last_route_assignments = top_skill.detach()
        self._last_diagnostics = {
            "route_frequency": hard_load.detach(),
            "route_probability": importance.detach(),
            "route_entropy": (
                -(router_probabilities * router_probabilities.clamp_min(1e-8).log())
                .sum(dim=-1)
                .mul(token_mask)
                .sum()
                .div(valid_token_count)
                .detach()
            ),
            "skill_delta_rms": (
                (skill_residual.detach().float().square() * token_mask.unsqueeze(-1))
                .sum()
                .div(valid_token_count * self.d_model)
                .clamp_min(1e-12)
                .sqrt()
            ),
            "adaptive_delta_rms": (
                (adaptive_delta.detach().float().square() * token_mask.unsqueeze(-1))
                .sum()
                .div(valid_token_count * self.d_model)
                .clamp_min(1e-12)
                .sqrt()
            ),
            "output_scale": output_scale.detach(),
            "balance_loss": balance_loss.detach(),
        }
        if self.training:
            self._last_training_diagnostics = self._last_diagnostics
        return adaptive_delta

    def pop_aux_loss(self) -> torch.Tensor | None:
        """Return and clear the latest router load-balance objective."""
        loss = self._last_aux_loss
        self._last_aux_loss = None
        return loss

    def last_diagnostics(self) -> dict[str, torch.Tensor]:
        """Return measurements from the most recent training batch."""
        return self._last_training_diagnostics

    def last_route_assignments(self) -> torch.Tensor | None:
        """Return per-token selected pathways for task-stratified evaluation."""
        return self._last_route_assignments
