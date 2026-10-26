"""Persistent shared-object write and broadcast read for MoE specialists.

The module owns no episode state. Callers pass the slot matrix between steps,
which keeps resets explicit and makes continuation/checkpointing possible.
"""

from __future__ import annotations

import math
import warnings

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

    _RECONSIDERATION_ERROR_EMA_DECAY = 0.9

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
        bptt_horizon: int = 0,
        reconsideration_enabled: bool = False,
        reconsideration_surprise_threshold: float = 0.5,
        reconsideration_trigger_mode: str = "innovation",
        prediction_error_salience_enabled: bool = False,
        compile_recurrence: bool = False,
    ):
        super().__init__()
        if d_model < 1 or num_experts < 1:
            raise ValueError("d_model and num_experts must be positive")
        if num_slots < 0:
            raise ValueError("num_slots must be >= 0")
        if compile_recurrence and num_slots == 0:
            raise ValueError("compiled workspace recurrence requires at least one slot")
        if gate_variant not in ("continuous", "bistable"):
            raise ValueError("gate_variant must be 'continuous' or 'bistable'")
        if bptt_horizon < 0:
            raise ValueError("bptt_horizon must be >= 0 (0 disables truncation)")
        if not math.isfinite(reconsideration_surprise_threshold) or not (
            0.0 <= reconsideration_surprise_threshold <= 1.0
        ):
            raise ValueError("reconsideration_surprise_threshold must be finite and in [0, 1]")
        if reconsideration_trigger_mode not in (
            "innovation",
            "random_rate_matched",
            "prediction_error",
            "prediction_error_random_rate_matched",
        ):
            raise ValueError("unknown workspace reconsideration trigger mode")
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_slots = num_slots
        self.gate_variant = gate_variant
        self.bistable_selection_scale = float(bistable_selection_scale)
        self.bistable_noise_scale = float(bistable_noise_scale)
        self.bptt_horizon = int(bptt_horizon)
        self.reconsideration_enabled = bool(reconsideration_enabled)
        self.reconsideration_surprise_threshold = float(reconsideration_surprise_threshold)
        self.reconsideration_trigger_mode = reconsideration_trigger_mode
        self.prediction_error_salience_enabled = bool(prediction_error_salience_enabled)
        self.compile_recurrence = bool(compile_recurrence)
        self._compiled_recurrence_step = None
        self._compiled_recurrence_failed = False
        self._compiled_fallback_warned = False
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.write_gate = nn.Linear(2 * d_model, d_model)
        self.prediction_error_to_write_gate = (
            nn.Linear(1, d_model, bias=False) if self.prediction_error_salience_enabled else None
        )
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
        if self.prediction_error_to_write_gate is not None:
            nn.init.zeros_(self.prediction_error_to_write_gate.weight)

    def init_state(self, batch: int, *, device, dtype) -> torch.Tensor:
        return torch.zeros(batch, self.num_slots, self.d_model, device=device, dtype=dtype)

    def _compiled_recurrence_step_impl(
        self,
        selected_deltas: torch.Tensor,
        route_weights: torch.Tensor,
        route_indices: torch.Tensor,
        write_keys: torch.Tensor,
        write_values: torch.Tensor,
        read_queries: torch.Tensor,
        state: torch.Tensor,
        prediction_error: torch.Tensor,
        reconsideration_error: torch.Tensor,
        reconsideration_random_scores: torch.Tensor,
    ):
        """Pure tensor step suitable for torch.compile and autograd."""
        scores = torch.einsum(
            "bed,bsd->bes", write_keys.float(), self.query(state).float()
        ) / math.sqrt(self.d_model)
        attention = scores.softmax(dim=-1).to(selected_deltas.dtype)
        weights = route_weights.float()
        proposal = torch.einsum(
            "be,bes,bed->bsd", weights, attention, write_values
        )
        mass = torch.einsum("be,bes->bs", weights, attention)
        proposal = proposal / mass.unsqueeze(-1).clamp_min(1e-6)
        update_logits = self.write_gate(torch.cat([state, proposal], dim=-1))
        if self.prediction_error_to_write_gate is not None:
            error_value = prediction_error.reshape(state.shape[0], 1).float().detach()
            update_logits = update_logits + self.prediction_error_to_write_gate(
                error_value
            ).unsqueeze(1)
        update = torch.sigmoid(update_logits)
        has_writer = weights.sum(dim=-1, keepdim=True).unsqueeze(-1) > 0
        update = update * has_writer.to(update.dtype)
        reconsideration_diagnostics = ()
        if self.reconsideration_enabled:
            old_probe = state.detach().float()
            # Spell norms as primitive reductions instead of linalg.vector_norm.
            # Some Colab Triton/Inductor runtimes fail lowering a fused kernel
            # containing vector_norm (PassManager::run failed).
            old_norm_sq = old_probe.square().sum(dim=-1)
            eligible = (old_norm_sq > 1e-12) & has_writer.squeeze(-1).squeeze(-1).unsqueeze(-1)
            if self.reconsideration_trigger_mode in (
                "prediction_error",
                "prediction_error_random_rate_matched",
            ):
                signal = reconsideration_error.reshape(state.shape[0], 1).float()
                surprise = signal.detach().clamp(0.0, 1.0).expand_as(eligible)
            else:
                proposal_probe = proposal.detach().float()
                old_norm = old_norm_sq.sqrt()
                proposal_norm = proposal_probe.square().sum(dim=-1).sqrt()
                innovation = (proposal_probe - old_probe).square().sum(dim=-1).sqrt()
                surprise = innovation / (old_norm + proposal_norm).clamp_min(1e-6)
            innovation_trigger = eligible & (
                surprise >= self.reconsideration_surprise_threshold
            )
            if self.reconsideration_trigger_mode in (
                "random_rate_matched",
                "prediction_error_random_rate_matched",
            ):
                eligible_flat = eligible.reshape(-1)
                random_scores_flat = reconsideration_random_scores.reshape(-1).masked_fill(
                    ~eligible_flat, -1.0
                )
                order = torch.argsort(random_scores_flat, descending=True)
                ranks = torch.empty_like(order)
                ranks.scatter_(0, order, torch.arange(order.numel(), device=order.device))
                target_count = innovation_trigger.sum()
                reconsideration_trigger = eligible & (
                    ranks.reshape_as(eligible) < target_count
                )
            else:
                reconsideration_trigger = innovation_trigger
            update = torch.where(
                reconsideration_trigger.unsqueeze(-1),
                torch.ones_like(update),
                update,
            )
        else:
            reconsideration_trigger = None
        next_state = update.mul(proposal).add((1.0 - update).mul(state))

        if self.reconsideration_enabled and self.diagnostics_enabled:
            eligible_count_raw = eligible.sum().float()
            eligible_count = eligible_count_raw.clamp_min(1.0)
            triggered_count = reconsideration_trigger.sum()
            change = (next_state - state).float()
            actual_change = change.square().sum(dim=-1).sqrt()
            changed_count = (reconsideration_trigger & (actual_change > 1e-6)).sum().float()
            state_delta_sq_sum = (actual_change * reconsideration_trigger).square().sum()
            reconsideration_diagnostics = (
                eligible.float().mean().detach(),
                eligible_count_raw.detach(),
                triggered_count.float().detach(),
                ((surprise * eligible).sum() / eligible_count).detach(),
                (triggered_count.float() / eligible_count).detach(),
                ((surprise * eligible).sum() / eligible_count).detach(),
                (changed_count / triggered_count.float().clamp_min(1.0)).detach(),
                changed_count.detach(),
                state_delta_sq_sum.detach(),
                (state_delta_sq_sum / triggered_count.float().clamp_min(1.0)).sqrt().detach(),
            )
        prediction_error_diagnostics = ()
        if self.prediction_error_to_write_gate is not None and self.diagnostics_enabled:
            prediction_error_diagnostics = (
                prediction_error.detach().float().mean(),
                self.prediction_error_to_write_gate.weight.detach()
                .float()
                .square()
                .mean()
                .sqrt(),
            )

        read_scores = torch.einsum(
            "bed,bsd->bes", read_queries.float(), self.key(next_state).float()
        ) / math.sqrt(self.d_model)
        read_weights = read_scores.softmax(dim=-1).to(selected_deltas.dtype)
        read_values = self.value(next_state)
        broadcast = torch.einsum("bes,bsd->bed", read_weights, read_values)
        read_scale = torch.sigmoid(self.read_gate).to(selected_deltas.dtype)
        output = (
            (selected_deltas + read_scale * broadcast) * route_weights.unsqueeze(-1)
        ).sum(dim=1)

        if self.diagnostics_enabled:
            write_mass = weights.sum(dim=-1).mean().detach()
            active_write = (weights.sum(dim=-1) > 0).float().mean().detach()
            write_gate_mean = update.detach().float().mean()
            slot_rms = next_state.detach().float().square().mean().sqrt()
            broadcast_rms = (
                read_scale.detach().float() * broadcast.detach().float()
            ).square().mean().sqrt()
            specialist_rms = selected_deltas.detach().float().square().mean().sqrt()
            writer_frequency = torch.zeros(
                self.num_experts, device=route_indices.device, dtype=torch.float32
            ).scatter_add_(
                0,
                route_indices.reshape(-1),
                route_weights.detach().float().reshape(-1),
            )
            writer_frequency = writer_frequency / max(route_indices.shape[0], 1)
            diagnostics = (
                write_mass,
                active_write,
                write_gate_mean,
                slot_rms,
                broadcast_rms,
                specialist_rms,
                writer_frequency,
            ) + reconsideration_diagnostics + prediction_error_diagnostics
        else:
            diagnostics = ()
        return output, next_state, diagnostics

    def _get_compiled_recurrence_step(self):
        if self._compiled_recurrence_step is None:
            compiler = getattr(torch, "compile", None)
            if compiler is None:
                raise RuntimeError(
                    "--workspace-compile-recurrence requires PyTorch 2.4 or newer"
                )
            self._compiled_recurrence_step = compiler(
                self._compiled_recurrence_step_impl,
                # The compiled transition is called once per sequence token,
                # and all calls retain autograd state until the sequence
                # backward pass. CUDA-graph reduce-overhead mode is designed
                # for completed invocation lifetimes and warns/falls back when
                # many forwards are pending backward in this recurrence.
                mode="default",
                fullgraph=True,
                dynamic=False,
                # Respect the lightweight default path even if the Colab
                # process has globally enabled max-autotune. This recurrence
                # is a small per-token graph; exhaustive GEMM tuning adds
                # compile-time work and can select unsupported Triton paths
                # on the small SM-count GPUs commonly assigned by Colab.
                options={"max_autotune": False},
            )
        return self._compiled_recurrence_step

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
        prediction_error: torch.Tensor | None = None,
        reconsideration_error: torch.Tensor | None = None,
        reconsideration_random_scores: torch.Tensor | None = None,
        selected_specialists_only: bool = False,
        projected_write_keys: torch.Tensor | None = None,
        projected_write_values: torch.Tensor | None = None,
        projected_read_queries: torch.Tensor | None = None,
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
                prediction_error=prediction_error,
                reconsideration_error=reconsideration_error,
                reconsideration_random_scores=reconsideration_random_scores,
                selected_specialists_only=selected_specialists_only,
                projected_write_keys=projected_write_keys,
                projected_write_values=projected_write_values,
                projected_read_queries=projected_read_queries,
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
                    prediction_error=prediction_error,
                    reconsideration_error=reconsideration_error,
                    reconsideration_random_scores=reconsideration_random_scores,
                    selected_specialists_only=selected_specialists_only,
                    projected_write_keys=projected_write_keys,
                    projected_write_values=projected_write_values,
                    projected_read_queries=projected_read_queries,
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
            prediction_error=prediction_error,
            reconsideration_error=reconsideration_error,
            reconsideration_random_scores=reconsideration_random_scores,
            selected_specialists_only=selected_specialists_only,
            projected_write_keys=projected_write_keys,
            projected_write_values=projected_write_values,
            projected_read_queries=projected_read_queries,
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
        prediction_error: torch.Tensor | None = None,
        reconsideration_error: torch.Tensor | None = None,
        reconsideration_random_scores: torch.Tensor | None = None,
        selected_specialists_only: bool = False,
        projected_write_keys: torch.Tensor | None = None,
        projected_write_values: torch.Tensor | None = None,
        projected_read_queries: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_slots == 0 or self.write_gate_closed:
            selected = (
                specialist_deltas
                if selected_specialists_only
                else specialist_deltas.gather(
                    1, route_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
                )
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
        expected_experts = route_indices.shape[-1] if selected_specialists_only else self.num_experts
        if (experts, width) != (expected_experts, self.d_model):
            raise ValueError("specialist delta shape does not match workspace configuration")
        if selected_specialists_only and no_selection:
            raise ValueError("random workspace selection requires all expert candidates")
        if state.shape != (batch, self.num_slots, self.d_model):
            raise ValueError("workspace state has an incompatible batch/slot/feature shape")

        if reset_each_step:
            state = torch.zeros_like(state)
        selected_deltas = (
            specialist_deltas
            if selected_specialists_only
            else specialist_deltas.gather(
                1, route_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
            )
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
        keys = (
            projected_write_keys
            if projected_write_keys is not None
            else self.key(write_candidates)
        )
        scores = torch.einsum("bed,bsd->bes", keys.float(), self.query(state).float())
        scores = scores / math.sqrt(self.d_model)
        slot_attention = scores.softmax(dim=-1).to(specialist_deltas.dtype)
        values = (
            projected_write_values
            if projected_write_values is not None
            else self.value(write_candidates)
        )
        slot_proposal = torch.einsum(
            "be,bes,bed->bsd", write_weights.to(values.dtype), slot_attention, values
        )
        mass = torch.einsum("be,bes->bs", write_weights.to(values.dtype), slot_attention)
        slot_proposal = slot_proposal / mass.unsqueeze(-1).clamp_min(1e-6)
        slot_inputs = slot_proposal
        old_inputs = state
        update_logits = self.write_gate(torch.cat([old_inputs, slot_inputs], dim=-1))
        error_value = None
        if prediction_error is not None:
            if prediction_error.numel() not in (1, batch):
                raise ValueError("prediction_error must be scalar or have one value per batch item")
            error_value = prediction_error.reshape(batch, 1).float().detach()
            if self.prediction_error_to_write_gate is not None:
                update_logits = update_logits + self.prediction_error_to_write_gate(
                    error_value
                ).unsqueeze(1)
        elif self.prediction_error_salience_enabled:
            raise ValueError("workspace prediction-error salience requires prediction_error")
        update = torch.sigmoid(update_logits)
        has_writer = write_weights.sum(dim=-1, keepdim=True).unsqueeze(-1) > 0
        update = update * has_writer.to(update.dtype)
        reconsideration_diag = {}
        reconsideration_trigger = None
        if self.reconsideration_enabled:
            # Operational surprise is normalized innovation between the
            # incoming slot proposal and its current committed content. It is
            # bounded in [0, 1], uses no target labels/future tokens, and only
            # applies to occupied slots with a current routed writer.
            old_probe = old_inputs.detach().float()
            proposal_probe = slot_inputs.detach().float()
            old_norm = torch.linalg.vector_norm(old_probe, dim=-1)
            proposal_norm = torch.linalg.vector_norm(proposal_probe, dim=-1)
            innovation = torch.linalg.vector_norm(proposal_probe - old_probe, dim=-1)
            surprise = innovation / (old_norm + proposal_norm).clamp_min(1e-6)
            eligible = (old_norm > 1e-6) & has_writer.squeeze(-1).squeeze(-1).unsqueeze(-1)
            if self.reconsideration_trigger_mode in (
                "prediction_error",
                "prediction_error_random_rate_matched",
            ):
                if reconsideration_error is None:
                    raise ValueError(
                        "prediction-error reconsideration requires a per-step prediction_error"
                    )
                signal = reconsideration_error.reshape(batch, 1).float().detach().clamp(0.0, 1.0)
                surprise = signal.expand_as(surprise)
                innovation_trigger = eligible & (
                    surprise >= self.reconsideration_surprise_threshold
                )
            else:
                innovation_trigger = eligible & (
                    surprise >= self.reconsideration_surprise_threshold
                )
            # Draw a fixed-shape random tape in both arms. This gives the
            # rate-matched null an independent choice among eligible slots while
            # keeping global RNG advancement aligned between matched runs.
            random_scores = (
                reconsideration_random_scores
                if reconsideration_random_scores is not None
                else torch.rand_like(surprise)
            )
            if self.reconsideration_trigger_mode in (
                "random_rate_matched",
                "prediction_error_random_rate_matched",
            ):
                eligible_flat = eligible.reshape(-1)
                random_scores_flat = random_scores.reshape(-1).masked_fill(~eligible_flat, -1.0)
                order = torch.argsort(random_scores_flat, descending=True)
                ranks = torch.empty_like(order)
                ranks.scatter_(
                    0,
                    order,
                    torch.arange(order.numel(), device=order.device),
                )
                target_count = innovation_trigger.sum()
                reconsideration_trigger = eligible & (ranks.reshape_as(eligible) < target_count)
            else:
                reconsideration_trigger = innovation_trigger
            if self.diagnostics_enabled:
                eligible_count_raw = eligible.sum().float()
                eligible_count = eligible_count_raw.clamp_min(1.0)
                triggered_count = reconsideration_trigger.sum()
                reconsideration_diag = {
                    "reconsideration_eligible_frac": eligible.float().mean().detach(),
                    "reconsideration_eligible_count": eligible_count_raw.detach(),
                    "reconsideration_trigger_count": triggered_count.float().detach(),
                    "reconsideration_surprise_mean": (
                        (surprise * eligible).sum() / eligible_count
                    ).detach(),
                    "reconsideration_trigger_frac": (
                        triggered_count.float() / eligible_count
                    ).detach(),
                    "reconsideration_prediction_error_mean": (
                        (surprise * eligible).sum() / eligible_count
                    ).detach(),
                }
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
        if reconsideration_trigger is not None:
            # A triggered reconsideration reopens the ordinary write gate for
            # that slot, replacing its old commitment with the new proposal.
            # Apply this after the optional bistable gate so the Phase 5
            # override retains its meaning in a later combined experiment.
            update = torch.where(
                reconsideration_trigger.unsqueeze(-1),
                torch.ones_like(update),
                update,
            )
        if self.diagnostics_enabled:
            writer_weights = torch.zeros(
                batch, self.num_experts, device=route_indices.device, dtype=torch.float32
            ).scatter_add_(1, route_indices, route_weights.float())
            writer_frequency = writer_weights.mean(dim=0).detach()
        next_state = update.mul(slot_inputs).add((1.0 - update).mul(old_inputs))
        if self.diagnostics_enabled and reconsideration_trigger is not None:
            triggered_count_raw = reconsideration_trigger.sum().float()
            triggered_count = triggered_count_raw.clamp_min(1.0)
            actual_change = torch.linalg.vector_norm((next_state - old_inputs).float(), dim=-1)
            changed_count = (reconsideration_trigger & (actual_change > 1e-6)).sum().float()
            state_delta_sq_sum = (actual_change * reconsideration_trigger).square().sum()
            reconsideration_diag["reconsideration_changed_frac"] = (
                changed_count / triggered_count
            ).detach()
            reconsideration_diag["reconsideration_changed_count"] = changed_count.detach()
            reconsideration_diag["reconsideration_state_delta_sq_sum"] = state_delta_sq_sum.detach()
            reconsideration_diag["reconsideration_state_delta_rms"] = (
                (state_delta_sq_sum / triggered_count).sqrt().detach()
            )

        if bypass_broadcast:
            # Evaluation counterfactual: preserve the selected MoE residual
            # while preventing workspace readback from influencing it. The
            # workspace write path still executes, so this isolates the
            # broadcast contribution without changing routing or the model
            # parameters loaded from the checkpoint.
            if self.diagnostics_enabled:
                self._last_step_diag = {
                    "write_mass": write_weights.detach().sum(dim=-1).mean(),
                    "active_write_frac": (write_weights.detach().sum(dim=-1) > 0).float().mean(),
                    "write_gate_mean": update.detach().float().mean(),
                    "slot_rms": next_state.detach().float().square().mean().sqrt(),
                    "broadcast_rms": next_state.new_zeros((), dtype=torch.float32),
                    "specialist_rms": specialist_deltas.detach().float().square().mean().sqrt(),
                    **reconsideration_diag,
                    **bistable_diag,
                    **(
                        {"prediction_error_mean": error_value.mean().detach()}
                        if error_value is not None
                        else {}
                    ),
                    **(
                        {
                            "prediction_error_write_scale_rms": (
                                self.prediction_error_to_write_gate.weight.detach()
                                .float()
                                .square()
                                .mean()
                                .sqrt()
                            )
                        }
                        if self.prediction_error_to_write_gate is not None
                        else {}
                    ),
                    **({"writer_frequency": writer_frequency} if self.diagnostics_enabled else {}),
                }
            return (selected_deltas * route_weights.unsqueeze(-1)).sum(dim=1), next_state

        # Only routed experts are combined into the backbone residual, so
        # computing readback for unselected experts is dead work.
        read_queries = (
            projected_read_queries
            if projected_read_queries is not None
            else self.query(selected_deltas)
        )
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
                "broadcast_rms": (read_scale.detach().float() * broadcast.detach().float())
                .square()
                .mean()
                .sqrt(),
                "specialist_rms": specialist_deltas.detach().float().square().mean().sqrt(),
                **reconsideration_diag,
                **bistable_diag,
                **(
                    {"prediction_error_mean": error_value.mean().detach()}
                    if error_value is not None
                    else {}
                ),
                **(
                    {
                        "prediction_error_write_scale_rms": (
                            self.prediction_error_to_write_gate.weight.detach()
                            .float()
                            .square()
                            .mean()
                            .sqrt()
                        )
                    }
                    if self.prediction_error_to_write_gate is not None
                    else {}
                ),
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
        prediction_error: torch.Tensor | None = None,
        selected_specialists_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process time-major specialists ``(B,T,E,D)`` sequentially."""
        outputs: list[torch.Tensor] = []
        step_diags: list[dict[str, torch.Tensor]] | None = [] if self.diagnostics_enabled else None
        state_observer = getattr(self, "_initium_recurrent_state_observer", None)
        smoothed_prediction_error = None
        projected_write_keys = None
        projected_write_values = None
        projected_read_queries = None
        time_major_deltas = None
        time_major_route_indices = None
        time_major_route_weights = None
        if self.num_slots > 0 and not self.write_gate_closed:
            # These projections depend only on this token's routed expert
            # outputs, so batch them over time before the causal slot loop.
            # The recurrence itself remains sequential and uses the exact
            # per-token state-dependent query, proposal, gate, and update.
            # Use time-major contiguous tensors for recurrent slicing. The
            # batch stride of ``x[:, step]`` depends on total sequence length;
            # changing curriculum lengths would otherwise produce a new
            # Dynamo stride guard/compiled variant for the same recurrence.
            time_major_deltas = specialist_deltas.transpose(0, 1).contiguous()
            time_major_route_indices = route_indices.transpose(0, 1).contiguous()
            time_major_route_weights = route_weights.transpose(0, 1).contiguous()
            selected_deltas = (
                time_major_deltas
                if selected_specialists_only
                else time_major_deltas.gather(
                    2,
                    time_major_route_indices.unsqueeze(-1).expand(
                        -1, -1, -1, self.d_model
                    ),
                )
            )
            write_candidates = time_major_deltas if no_selection else selected_deltas
            projection_context = (
                torch.autocast(device_type="cuda", enabled=False)
                if specialist_deltas.is_cuda
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with projection_context:
                projected_write_keys = self.key(write_candidates.float())
                projected_write_values = self.value(write_candidates.float())
                if not bypass_broadcast:
                    projected_read_queries = self.query(selected_deltas.float())
        compiled_step = None
        # The fused recurrence is a training optimization. Curriculum and
        # periodic evaluations use variable/partial batch sizes, which would
        # otherwise create new fullgraph specializations for each batch shape.
        # Keep evaluation on the eager path; its state transition is identical.
        if (
            self.compile_recurrence
            and self.training
            and torch.is_grad_enabled()
            and not self._compiled_recurrence_failed
        ):
            if not specialist_deltas.is_cuda:
                raise RuntimeError(
                    "--workspace-compile-recurrence requires a CUDA device"
                )
            if (
                no_selection
                or not selected_specialists_only
                or self.bistable_gate is not None
            ):
                raise RuntimeError(
                    "compiled workspace recurrence supports routed writes with the "
                    "continuous gate, reconsideration, and prediction-error write salience; "
                    "random-write and bistable modes require the eager path"
                )
            if (
                self.reconsideration_enabled
                and self.reconsideration_trigger_mode.startswith("prediction_error")
                and prediction_error is None
            ):
                raise ValueError(
                    "prediction-error reconsideration requires a per-step prediction_error"
                )
            if self.prediction_error_to_write_gate is not None and prediction_error is None:
                raise ValueError("workspace prediction-error salience requires prediction_error")
            if bypass_broadcast or self.write_gate_closed:
                if not self._compiled_fallback_warned:
                    warnings.warn(
                        "compiled workspace recurrence is bypassed for this forward; "
                        "using the eager workspace path",
                        stacklevel=2,
                    )
                    self._compiled_fallback_warned = True
            else:
                compiled_step = self._get_compiled_recurrence_step()
        compiled_zero_reconsideration_error = None
        compiled_zero_reconsideration_scores = None
        if compiled_step is not None:
            compiled_zero_reconsideration_error = state.new_zeros(
                (state.shape[0],), dtype=torch.float32
            )
            compiled_zero_reconsideration_scores = state.new_zeros(
                state.shape[:2], dtype=torch.float32
            )
        for step in range(specialist_deltas.shape[1]):
            # Truncate only the gradient carried by persistent slots. Detach
            # preserves the exact forward state and therefore the full write /
            # read trajectory; zero keeps the historical full-sequence BPTT.
            if self.bptt_horizon and step and step % self.bptt_horizon == 0:
                state = state.detach()
            step_prediction_error = (
                prediction_error[:, step] if prediction_error is not None else None
            )
            if step_prediction_error is not None:
                current_error = step_prediction_error.float().detach().clamp(0.0, 1.0)
                if smoothed_prediction_error is None:
                    smoothed_prediction_error = current_error
                else:
                    decay = self._RECONSIDERATION_ERROR_EMA_DECAY
                    smoothed_prediction_error = (
                        decay * smoothed_prediction_error + (1.0 - decay) * current_error
                    )
            reconsideration_random_scores = None
            if compiled_step is not None:
                if reset_each_step:
                    state = torch.zeros_like(state)
                # The first workspace state is usually a fresh zero tensor
                # with requires_grad=False; subsequent recurrent states
                # require gradients through the slot update. Dynamo guards
                # on this property and otherwise recompiles the fullgraph at
                # the first transition (and after each BPTT detach/reset).
                if self.training and torch.is_grad_enabled() and not state.requires_grad:
                    state = state.detach().requires_grad_(True)
                if self.reconsideration_enabled:
                    # The eager path advances RNG once per token even for a
                    # deterministic innovation trigger. Keep that draw outside
                    # the graph and pass the random tape into the recurrence.
                    reconsideration_random_scores = torch.rand_like(
                        state[..., 0], memory_format=torch.contiguous_format
                    )
                else:
                    reconsideration_random_scores = compiled_zero_reconsideration_scores
                compiled_reconsideration_error = (
                    smoothed_prediction_error
                    if smoothed_prediction_error is not None
                    else compiled_zero_reconsideration_error
                )
                compiled_prediction_error = (
                    step_prediction_error
                    if step_prediction_error is not None
                    else compiled_zero_reconsideration_error
                )
                try:
                    with torch.autocast(device_type="cuda", enabled=False):
                        out, state, compiled_diagnostics = compiled_step(
                            time_major_deltas[step].float(),
                            time_major_route_weights[step].float(),
                            time_major_route_indices[step],
                            projected_write_keys[step].float(),
                            projected_write_values[step].float(),
                            projected_read_queries[step].float(),
                            state.float(),
                            compiled_prediction_error.float(),
                            compiled_reconsideration_error.float(),
                            reconsideration_random_scores,
                        )
                except Exception as exc:
                    compile_error_names = {
                        "BackendCompilerFailed",
                        "InductorError",
                        "SubprocException",
                        "TritonError",
                    }
                    cause = exc
                    is_compile_error = False
                    while cause is not None:
                        if type(cause).__name__ in compile_error_names:
                            is_compile_error = True
                            break
                        cause = cause.__cause__ or cause.__context__
                    if not is_compile_error:
                        raise
                    self._compiled_recurrence_failed = True
                    self._compiled_recurrence_step = None
                    compiled_step = None
                    compiler_details = []
                    cause = exc
                    seen_exceptions = set()
                    while cause is not None and id(cause) not in seen_exceptions:
                        seen_exceptions.add(id(cause))
                        message_lines = [
                            line.strip()
                            for line in str(cause).splitlines()
                            if line.strip()
                        ]
                        diagnostic_lines = [
                            line
                            for line in message_lines
                            if "Name=" in line
                            or "RuntimeError:" in line
                            or "error:" in line.lower()
                        ]
                        summary_lines = diagnostic_lines[-3:] or message_lines[-1:]
                        detail = " | ".join(summary_lines)
                        if detail:
                            compiler_details.append(
                                f"{type(cause).__name__}: {detail[:700]}"
                            )
                        cause = cause.__cause__ or cause.__context__
                    detail_text = " <- ".join(compiler_details)
                    warnings.warn(
                        "torch.compile could not build the workspace recurrence "
                        f"({type(exc).__name__}: {str(exc).splitlines()[0]}; "
                        f"compiler chain: {detail_text or 'details unavailable'}). "
                        "Disabling the compiled path for this model and continuing "
                        "with the mathematically equivalent eager recurrence.",
                        stacklevel=2,
                    )
            if compiled_step is None:
                # A backend compile failure is safe to recover from here: the
                # recurrence is pure, the parameters/state have not been
                # mutated, and the same pre-drawn RNG tape is reused by eager.
                out, state = self.forward_step(
                    specialist_deltas[:, step],
                    route_indices[:, step],
                    route_weights[:, step],
                    state,
                    no_selection=no_selection,
                    reset_each_step=reset_each_step,
                    bypass_broadcast=bypass_broadcast,
                    bistable_noise_scale=bistable_noise_scale,
                    bistable_selection_scale=bistable_selection_scale,
                    deterministic_gate=deterministic_gate,
                    prediction_error=step_prediction_error,
                    reconsideration_error=smoothed_prediction_error,
                    reconsideration_random_scores=reconsideration_random_scores,
                    selected_specialists_only=selected_specialists_only,
                    projected_write_keys=(
                        projected_write_keys[step]
                        if projected_write_keys is not None
                        else None
                    ),
                    projected_write_values=(
                        projected_write_values[step]
                        if projected_write_values is not None
                        else None
                    ),
                    projected_read_queries=(
                        projected_read_queries[step]
                        if projected_read_queries is not None
                        else None
                    ),
                )
            else:
                if self.diagnostics_enabled:
                    (
                        write_mass,
                        active_write_frac,
                        write_gate_mean,
                        slot_rms,
                        broadcast_rms,
                        specialist_rms,
                        writer_frequency,
                    ) = compiled_diagnostics[:7]
                    self._last_step_diag = {
                        "write_mass": write_mass,
                        "active_write_frac": active_write_frac,
                        "write_gate_mean": write_gate_mean,
                        "slot_rms": slot_rms,
                        "broadcast_rms": broadcast_rms,
                        "specialist_rms": specialist_rms,
                        "writer_frequency": writer_frequency,
                    }
                    if self.reconsideration_enabled:
                        (
                            eligible_frac,
                            eligible_count,
                            trigger_count,
                            surprise_mean,
                            trigger_frac,
                            prediction_error_mean,
                            changed_frac,
                            changed_count,
                            state_delta_sq_sum,
                            state_delta_rms,
                        ) = compiled_diagnostics[7:]
                        self._last_step_diag.update(
                            {
                                "reconsideration_eligible_frac": eligible_frac,
                                "reconsideration_eligible_count": eligible_count,
                                "reconsideration_trigger_count": trigger_count,
                                "reconsideration_surprise_mean": surprise_mean,
                                "reconsideration_trigger_frac": trigger_frac,
                                "reconsideration_prediction_error_mean": prediction_error_mean,
                                "reconsideration_changed_frac": changed_frac,
                                "reconsideration_changed_count": changed_count,
                                "reconsideration_state_delta_sq_sum": state_delta_sq_sum,
                                "reconsideration_state_delta_rms": state_delta_rms,
                            }
                        )
                    diagnostic_offset = 7 + (10 if self.reconsideration_enabled else 0)
                    if self.prediction_error_to_write_gate is not None:
                        error_mean, write_scale_rms = compiled_diagnostics[
                            diagnostic_offset : diagnostic_offset + 2
                        ]
                        self._last_step_diag.update(
                            {
                                "prediction_error_mean": error_mean,
                                "prediction_error_write_scale_rms": write_scale_rms,
                            }
                        )
                else:
                    self._last_step_diag = {}
            if state_observer is not None:
                state_observer(state, call_index=step)
            outputs.append(out)
            if step_diags is not None:
                step_diags.append(self._last_step_diag)
        if step_diags:
            self._last_diag = {}
            candidate_counts = None
            if "gate_candidate_count" in step_diags[0]:
                candidate_counts = torch.stack(
                    [diag["gate_candidate_count"] for diag in step_diags]
                )
                candidate_weights = candidate_counts / candidate_counts.sum().clamp_min(1.0)
            for key in step_diags[0]:
                values = torch.stack([diag[key] for diag in step_diags])
                if key.endswith("_count") or key.endswith("_sq_sum"):
                    aggregate = values.sum()
                elif key.endswith("_min"):
                    aggregate = values.min()
                elif key.endswith("_max") or key.endswith("_absmax"):
                    aggregate = values.max()
                elif key.startswith("debug_") and candidate_counts is not None:
                    aggregate = (values * candidate_weights).sum()
                else:
                    aggregate = values.mean()
                self._last_diag[key] = aggregate.detach()
            if "reconsideration_eligible_count" in self._last_diag:
                eligible_count = self._last_diag["reconsideration_eligible_count"]
                triggered_count = self._last_diag["reconsideration_trigger_count"]
                changed_count = self._last_diag.get(
                    "reconsideration_changed_count", triggered_count.new_zeros(())
                )
                self._last_diag["reconsideration_trigger_frac"] = (
                    triggered_count / eligible_count.clamp_min(1.0)
                ).detach()
                self._last_diag["reconsideration_changed_frac"] = (
                    changed_count / triggered_count.clamp_min(1.0)
                ).detach()
                if "reconsideration_state_delta_sq_sum" in self._last_diag:
                    self._last_diag["reconsideration_state_delta_rms"] = (
                        (
                            self._last_diag["reconsideration_state_delta_sq_sum"]
                            / triggered_count.clamp_min(1.0)
                        )
                        .sqrt()
                        .detach()
                    )
            self._last_diag["read_gate"] = torch.sigmoid(self.read_gate.detach()).float()
            self._last_diag["broadcast_to_specialist_ratio"] = self._last_diag[
                "broadcast_rms"
            ] / self._last_diag["specialist_rms"].clamp_min(1e-8)
            if "writer_frequency" in step_diags[0]:
                self._last_diag["writer_frequency"] = (
                    torch.stack([diag["writer_frequency"] for diag in step_diags])
                    .mean(dim=0)
                    .detach()
                )
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
                histograms = torch.stack([diag["gate_histogram"] for diag in step_diags])
                self._last_diag["gate_histogram"] = (
                    (histograms * candidate_counts.unsqueeze(-1)).sum(dim=0) / candidate_denom
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
