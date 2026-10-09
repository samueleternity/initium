"""Persistent shared-object write and broadcast read for MoE specialists.

The module owns no episode state. Callers pass the slot matrix between steps,
which keeps resets explicit and makes continuation/checkpointing possible.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class WorkspaceBroadcast(nn.Module):
    """Competitive MoE-selected writes and all-expert shared-slot readback.

    Inputs are specialist deltas ``(B, E, D)`` and sparse MoE routes
    ``(B, K)``/weights ``(B, K)``. The router's selection determines which
    specialists write. Every specialist independently queries all slots and
    receives the resulting context before the selected deltas are combined.
    """

    def __init__(self, d_model: int, num_experts: int, num_slots: int = 5):
        super().__init__()
        if d_model < 1 or num_experts < 1:
            raise ValueError("d_model and num_experts must be positive")
        if num_slots < 0:
            raise ValueError("num_slots must be >= 0")
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_slots = num_slots
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.write_gate = nn.Linear(2 * d_model, d_model)
        self.read_gate = nn.Parameter(torch.zeros(()))
        self.write_gate_closed = False
        self._last_step_diag: dict[str, torch.Tensor] = {}
        self._last_diag: dict[str, torch.Tensor] = {}
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_slots == 0 or self.write_gate_closed:
            selected = specialist_deltas.gather(
                1, route_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
            )
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
        if no_selection:
            write_weights = torch.rand(
                batch, experts, device=specialist_deltas.device, dtype=torch.float32
            )
            write_weights = write_weights / write_weights.sum(-1, keepdim=True).clamp_min(1e-8)
        else:
            write_weights = torch.zeros(
                batch, experts, device=specialist_deltas.device, dtype=torch.float32
            )
            write_weights.scatter_add_(1, route_indices, route_weights.float())

        # Candidate keys compete over slots using content-based attention;
        # slot writes are weighted by the existing sparse MoE selection.
        keys = self.key(specialist_deltas)
        scores = torch.einsum("bed,bsd->bes", keys.float(), self.query(state).float())
        scores = scores / math.sqrt(self.d_model)
        slot_attention = scores.softmax(dim=-1).to(specialist_deltas.dtype)
        values = self.value(specialist_deltas)
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
        next_state = update.mul(slot_inputs).add((1.0 - update).mul(old_inputs))

        # All experts query the shared object, including those not selected
        # for this step's write.
        read_queries = self.query(specialist_deltas)
        read_scores = torch.einsum(
            "bed,bsd->bes", read_queries.float(), self.key(next_state).float()
        )
        read_scores = read_scores / math.sqrt(self.d_model)
        read_weights = read_scores.softmax(dim=-1).to(specialist_deltas.dtype)
        read_values = self.value(next_state)
        broadcast = torch.einsum("bes,bsd->bed", read_weights, read_values)
        read_scale = torch.sigmoid(self.read_gate).to(specialist_deltas.dtype)
        enriched = specialist_deltas + read_scale * broadcast
        selected = enriched.gather(1, route_indices.unsqueeze(-1).expand(-1, -1, self.d_model))
        self._last_step_diag = {
            "write_mass": write_weights.detach().sum(dim=-1).mean(),
            "active_write_frac": (write_weights.detach().sum(dim=-1) > 0).float().mean(),
            "write_gate_mean": update.detach().float().mean(),
            "slot_rms": next_state.detach().float().square().mean().sqrt(),
            "broadcast_rms": (
                read_scale.detach().float() * broadcast.detach().float()
            ).square().mean().sqrt(),
            "specialist_rms": specialist_deltas.detach().float().square().mean().sqrt(),
        }
        return (selected * route_weights.unsqueeze(-1)).sum(dim=1), next_state

    def forward(
        self,
        specialist_deltas: torch.Tensor,
        route_indices: torch.Tensor,
        route_weights: torch.Tensor,
        state: torch.Tensor,
        *,
        no_selection: bool = False,
        reset_each_step: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process time-major specialists ``(B,T,E,D)`` sequentially."""
        outputs = []
        step_diags = []
        for step in range(specialist_deltas.shape[1]):
            out, state = self.forward_step(
                specialist_deltas[:, step], route_indices[:, step], route_weights[:, step],
                state,
                no_selection=no_selection,
                reset_each_step=reset_each_step,
            )
            outputs.append(out)
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
        return torch.stack(outputs, dim=1), state

    def last_diagnostics(self) -> dict[str, torch.Tensor]:
        """Detached activity metrics for the most recent sequence pass."""
        return dict(self._last_diag)
