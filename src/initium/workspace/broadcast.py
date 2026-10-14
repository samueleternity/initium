"""Persistent shared-object write and broadcast read for MoE specialists.

The module owns no episode state. Callers pass the slot matrix between steps,
which keeps resets explicit and makes continuation/checkpointing possible.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from initium.workspace.bistability import BistableWriteGate


class WorkspaceBroadcast(nn.Module):
    """Competitive MoE-selected writes and all-expert shared-slot readback.

    Inputs are specialist deltas ``(B, E, D)`` and sparse MoE routes
    ``(B, K)``/weights ``(B, K)``. The router's selection determines which
    specialists write. Every specialist independently queries all slots and
    receives the resulting context before the selected deltas are combined.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        num_slots: int = 5,
        *,
        gate_variant: str = "continuous",
        bistable_selection_scale: float = 1.0,
        bistable_noise_scale: float = 0.75,
        bistable_gradient_mode: str = "truncated",
    ):
        super().__init__()
        if d_model < 1 or num_experts < 1:
            raise ValueError("d_model and num_experts must be positive")
        if num_slots < 0:
            raise ValueError("num_slots must be >= 0")
        if gate_variant not in ("continuous", "bistable"):
            raise ValueError("gate_variant must be 'continuous' or 'bistable'")
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_slots = num_slots
        self.gate_variant = gate_variant
        self.bistable_selection_scale = float(bistable_selection_scale)
        self.bistable_noise_scale = float(bistable_noise_scale)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.write_gate = nn.Linear(2 * d_model, d_model)
        self.read_gate = nn.Parameter(torch.zeros(()))
        self.write_gate_closed = False
        self.bistable_gate = (
            BistableWriteGate(
                selection_scale=bistable_selection_scale,
                noise_scale=bistable_noise_scale,
                gradient_mode=bistable_gradient_mode,
            )
            if gate_variant == "bistable"
            else None
        )
        self.diagnostics_enabled = True
        self._last_step_diag: dict[str, torch.Tensor] = {}
        self._last_diag: dict[str, torch.Tensor] = {}
        self._bistable_diagnostic_totals: dict[str, torch.Tensor] = {}
        self._bistable_diagnostic_calls = 0
        nn.init.zeros_(self.write_gate.weight)
        nn.init.zeros_(self.write_gate.bias)

    def init_state(self, batch: int, *, device, dtype) -> torch.Tensor:
        return torch.zeros(batch, self.num_slots, self.d_model, device=device, dtype=dtype)

    def forward_step(
        self,
        specialist_deltas: torch.Tensor,
        route_indices: torch.Tensor,
        route_weights: torch.Tensor,
        state: torch.Tensor,
        *,
        no_selection: bool = False,
        reset_each_step: bool = False,
        bypass_broadcast: bool = False,
        bistable_noise_scale: float | None = None,
        bistable_selection_scale: float | None = None,
        deterministic_gate: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Keep the zero-slot/closed-gate bypass identical to the selection-only
        # MoE path. Active workspace attention and recurrent slot updates use
        # fp32 even when the surrounding controller is under CUDA autocast.
        if self.num_slots == 0 or self.write_gate_closed:
            return self._forward_step(
                specialist_deltas,
                route_indices,
                route_weights,
                state,
                no_selection=no_selection,
                reset_each_step=reset_each_step,
                bypass_broadcast=bypass_broadcast,
                bistable_noise_scale=bistable_noise_scale,
                bistable_selection_scale=bistable_selection_scale,
                deterministic_gate=deterministic_gate,
            )
        if specialist_deltas.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                return self._forward_step(
                    specialist_deltas.float(),
                    route_indices,
                    route_weights.float(),
                    state.float(),
                    no_selection=no_selection,
                    reset_each_step=reset_each_step,
                    bypass_broadcast=bypass_broadcast,
                    bistable_noise_scale=bistable_noise_scale,
                    bistable_selection_scale=bistable_selection_scale,
                    deterministic_gate=deterministic_gate,
                )
        return self._forward_step(
            specialist_deltas.float(),
            route_indices,
            route_weights.float(),
            state.float(),
            no_selection=no_selection,
            reset_each_step=reset_each_step,
            bypass_broadcast=bypass_broadcast,
            bistable_noise_scale=bistable_noise_scale,
            bistable_selection_scale=bistable_selection_scale,
            deterministic_gate=deterministic_gate,
        )

    def _forward_step(
        self,
        specialist_deltas: torch.Tensor,
        route_indices: torch.Tensor,
        route_weights: torch.Tensor,
        state: torch.Tensor,
        *,
        no_selection: bool = False,
        reset_each_step: bool = False,
        bypass_broadcast: bool = False,
        bistable_noise_scale: float | None = None,
        bistable_selection_scale: float | None = None,
        deterministic_gate: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_slots == 0 or self.write_gate_closed:
            selected = specialist_deltas.gather(
                1, route_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
            )
            if self.diagnostics_enabled:
                zero = specialist_deltas.new_zeros((), dtype=torch.float32)
                self._last_step_diag = {
                    "write_mass": zero,
                    "active_write_frac": zero,
                    "write_gate_mean": zero,
                    "slot_rms": zero,
                    "broadcast_rms": zero,
                    "specialist_rms": specialist_deltas.detach().float().square().mean().sqrt(),
                }
            return (selected * route_weights.unsqueeze(-1)).sum(dim=1), state

        batch, experts, width = specialist_deltas.shape
        if (experts, width) != (self.num_experts, self.d_model):
            raise ValueError("specialist delta shape does not match workspace configuration")
        if state.shape != (batch, self.num_slots, self.d_model):
            raise ValueError("workspace state has an incompatible batch/slot/feature shape")

        if reset_each_step:
            state = torch.zeros_like(state)
        selected_deltas = specialist_deltas.gather(
            1, route_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
        )
        if no_selection:
            # The random-write control intentionally samples across every
            # specialist, so it must keep the full expert dimension here.
            write_weights = torch.rand(
                batch, experts, device=specialist_deltas.device, dtype=torch.float32
            )
            write_weights = write_weights / write_weights.sum(-1, keepdim=True).clamp_min(1e-8)
            write_candidates = specialist_deltas
        else:
            # Only routed experts can contribute to the weighted write.
            # Keeping the top-k axis avoids projecting and scoring all E
            # specialists on every sequential timestep. Duplicate routes,
            # if supplied by a caller, retain their original additive effect.
            write_weights = route_weights.float()
            write_candidates = selected_deltas

        # Candidate keys compete over slots using content-based attention;
        # slot writes are weighted by the existing sparse MoE selection.
        keys = self.key(write_candidates)
        scores = torch.einsum("bed,bsd->bes", keys.float(), self.query(state).float())
        scores = scores / math.sqrt(self.d_model)
        slot_attention = scores.softmax(dim=-1).to(specialist_deltas.dtype)
        values = self.value(write_candidates)
        slot_proposal = torch.einsum(
            "be,bes,bed->bsd", write_weights.to(values.dtype), slot_attention, values
        )
        mass = torch.einsum("be,bes->bs", write_weights.to(values.dtype), slot_attention)
        slot_proposal = slot_proposal / mass.unsqueeze(-1).clamp_min(1e-6)
        slot_inputs = slot_proposal
        old_inputs = state
        update = torch.sigmoid(self.write_gate(torch.cat([old_inputs, slot_inputs], dim=-1)))
        has_writer = write_weights.sum(dim=-1, keepdim=True).unsqueeze(-1) > 0
        update = update * has_writer.to(update.dtype)
        bistable_diag = {}
        if self.bistable_gate is not None:
            committed, _ = self.bistable_gate(
                mass,
                selection_confidence=write_weights.sum(dim=-1),
                candidate_mask=has_writer.reshape(batch),
                noise_scale=bistable_noise_scale,
                selection_scale=bistable_selection_scale,
                deterministic=deterministic_gate,
                collect_diagnostics=self.diagnostics_enabled,
            )
            update = update * committed.unsqueeze(-1)
            bistable_diag = self.bistable_gate.last_diagnostics()
        if self.diagnostics_enabled:
            writer_weights = torch.zeros(
                batch, self.num_experts, device=route_indices.device, dtype=torch.float32
            ).scatter_add_(1, route_indices, route_weights.float())
            writer_frequency = writer_weights.mean(dim=0).detach()
        next_state = update.mul(slot_inputs).add((1.0 - update).mul(old_inputs))

        if bypass_broadcast:
            # Evaluation counterfactual: preserve the selected MoE residual
            # while preventing workspace readback from influencing it. The
            # workspace write path still executes, so this isolates the
            # broadcast contribution without changing routing or the model
            # parameters loaded from the checkpoint.
            if self.diagnostics_enabled:
                self._last_step_diag = {
                    "write_mass": write_weights.detach().sum(dim=-1).mean(),
                    "active_write_frac": (write_weights.detach().sum(dim=-1) > 0)
                    .float()
                    .mean(),
                    "write_gate_mean": update.detach().float().mean(),
                    "slot_rms": next_state.detach().float().square().mean().sqrt(),
                    "broadcast_rms": next_state.new_zeros((), dtype=torch.float32),
                    "specialist_rms": specialist_deltas.detach().float().square().mean().sqrt(),
                    **bistable_diag,
                    **({"writer_frequency": writer_frequency} if self.diagnostics_enabled else {}),
                }
            return (selected_deltas * route_weights.unsqueeze(-1)).sum(dim=1), next_state

        # Only routed experts are combined into the backbone residual, so
        # computing readback for unselected experts is dead work.
        read_queries = self.query(selected_deltas)
        read_scores = torch.einsum(
            "bed,bsd->bes", read_queries.float(), self.key(next_state).float()
        )
        read_scores = read_scores / math.sqrt(self.d_model)
        read_weights = read_scores.softmax(dim=-1).to(specialist_deltas.dtype)
        read_values = self.value(next_state)
        broadcast = torch.einsum("bes,bsd->bed", read_weights, read_values)
        read_scale = torch.sigmoid(self.read_gate).to(specialist_deltas.dtype)
        enriched = selected_deltas + read_scale * broadcast
        if self.diagnostics_enabled:
            self._last_step_diag = {
                "write_mass": write_weights.detach().sum(dim=-1).mean(),
                "active_write_frac": (write_weights.detach().sum(dim=-1) > 0).float().mean(),
                "write_gate_mean": update.detach().float().mean(),
                "slot_rms": next_state.detach().float().square().mean().sqrt(),
                "broadcast_rms": (
                    read_scale.detach().float() * broadcast.detach().float()
                ).square().mean().sqrt(),
                "specialist_rms": specialist_deltas.detach().float().square().mean().sqrt(),
                **bistable_diag,
                **({"writer_frequency": writer_frequency} if self.diagnostics_enabled else {}),
            }
        return (enriched * route_weights.unsqueeze(-1)).sum(dim=1), next_state

    def forward(
        self,
        specialist_deltas: torch.Tensor,
        route_indices: torch.Tensor,
        route_weights: torch.Tensor,
        state: torch.Tensor,
        *,
        no_selection: bool = False,
        reset_each_step: bool = False,
        bypass_broadcast: bool = False,
        bistable_noise_scale: float | None = None,
        bistable_selection_scale: float | None = None,
        deterministic_gate: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process time-major specialists ``(B,T,E,D)`` sequentially."""
        outputs = []
        step_diags = [] if self.diagnostics_enabled else None
        for step in range(specialist_deltas.shape[1]):
            out, state = self.forward_step(
                specialist_deltas[:, step], route_indices[:, step], route_weights[:, step],
                state,
                no_selection=no_selection,
                reset_each_step=reset_each_step,
                bypass_broadcast=bypass_broadcast,
                bistable_noise_scale=bistable_noise_scale,
                bistable_selection_scale=bistable_selection_scale,
                deterministic_gate=deterministic_gate,
            )
            outputs.append(out)
            if step_diags is not None:
                step_diags.append(self._last_step_diag)
        if step_diags:
            self._last_diag = {
                key: torch.stack([diag[key] for diag in step_diags]).mean().detach()
                for key in step_diags[0]
            }
            self._last_diag["read_gate"] = torch.sigmoid(self.read_gate.detach()).float()
            self._last_diag["broadcast_to_specialist_ratio"] = (
                self._last_diag["broadcast_rms"]
                / self._last_diag["specialist_rms"].clamp_min(1e-8)
            )
            if "writer_frequency" in step_diags[0]:
                self._last_diag["writer_frequency"] = torch.stack(
                    [diag["writer_frequency"] for diag in step_diags]
                ).mean(dim=0).detach()
            if "gate_candidate_count" in step_diags[0]:
                candidate_counts = torch.stack(
                    [diag["gate_candidate_count"] for diag in step_diags]
                )
                candidate_total = candidate_counts.sum()
                candidate_denom = candidate_total.clamp_min(1.0)
                self._last_diag["gate_candidate_count"] = candidate_total.detach()
                for key in (
                    "gate_probability_mean",
                    "gate_low_frac",
                    "gate_mid_frac",
                    "gate_high_frac",
                    "write_frequency",
                    "selection_strength_mean",
                ):
                    values = torch.stack([diag[key] for diag in step_diags])
                    self._last_diag[key] = (
                        (values * candidate_counts).sum() / candidate_denom
                    ).detach()
                histograms = torch.stack(
                    [diag["gate_histogram"] for diag in step_diags]
                )
                self._last_diag["gate_histogram"] = (
                    (histograms * candidate_counts.unsqueeze(-1)).sum(dim=0)
                    / candidate_denom
                ).detach()
            self._accumulate_bistable_diagnostics(self._last_diag)
        return torch.stack(outputs, dim=1), state

    def last_diagnostics(self) -> dict[str, torch.Tensor]:
        """Detached activity metrics for the most recent sequence pass."""
        return dict(self._last_diag)

    def set_diagnostics_enabled(self, enabled: bool) -> None:
        """Enable or skip optional per-timestep workspace health reductions."""
        self.diagnostics_enabled = bool(enabled)
        if not self.diagnostics_enabled:
            self._last_step_diag = {}
            self._last_diag = {}
            self.reset_bistable_diagnostics()

    def _accumulate_bistable_diagnostics(self, diagnostics: dict[str, torch.Tensor]) -> None:
        if "gate_histogram" not in diagnostics:
            return
        candidate_count = diagnostics.get("gate_candidate_count")
        if candidate_count is None:
            return
        keys = (
            "gate_probability_mean",
            "gate_histogram",
            "gate_low_frac",
            "gate_mid_frac",
            "gate_high_frac",
            "write_frequency",
            "selection_strength_mean",
            "writer_frequency",
        )
        for key in keys:
            value = diagnostics.get(key)
            if value is not None:
                weight = (
                    candidate_count / self.num_slots
                    if key == "writer_frequency" or key == "selection_strength_mean"
                    else candidate_count
                )
                self._bistable_diagnostic_totals[key] = (
                    self._bistable_diagnostic_totals[key] + value.detach() * weight
                    if key in self._bistable_diagnostic_totals
                    else value.detach().clone() * weight
                )
        self._bistable_diagnostic_totals["gate_candidate_count"] = (
            self._bistable_diagnostic_totals.get(
                "gate_candidate_count", candidate_count.detach().new_zeros(())
            )
            + candidate_count.detach()
        )
        self._bistable_diagnostic_calls += 1

    def consume_bistable_diagnostics(self) -> dict[str, torch.Tensor]:
        """Return candidate diagnostics averaged across forwards since reset."""
        if self._bistable_diagnostic_calls == 0:
            return {}
        candidate_count = self._bistable_diagnostic_totals["gate_candidate_count"]
        denominator = candidate_count.clamp_min(1.0)
        result = {
            key: value
            / (
                denominator / self.num_slots
                if key in ("writer_frequency", "selection_strength_mean")
                else denominator
            )
            for key, value in self._bistable_diagnostic_totals.items()
            if key != "gate_candidate_count"
        }
        result["gate_candidate_count"] = candidate_count
        self.reset_bistable_diagnostics()
        return result

    def reset_bistable_diagnostics(self) -> None:
        self._bistable_diagnostic_totals.clear()
        self._bistable_diagnostic_calls = 0

    def bistable_gate_parameters(self):
        """Return experimental gate parameters, or an empty iterator."""
        if self.bistable_gate is None:
            return iter(())
        return self.bistable_gate.parameters()
