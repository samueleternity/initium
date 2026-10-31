"""Persistent shared-object write and broadcast read for MoE specialists.

The module owns no episode state. Callers pass the slot matrix between steps,
which keeps resets explicit and makes continuation/checkpointing possible.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from initium.workspace.bistability import BistableWriteGate
from initium.workspace.nth_farthest import select_nth_farthest


def _normalize_address(value: torch.Tensor) -> torch.Tensor:
    """Unit-normalize addresses with bounded gradients and compiler-friendly ops.

    Routed MoE deltas can be extremely small after residual scaling. Ordinary
    L2 normalization then amplifies gradients by the inverse vector norm, which
    made the recurrent address path numerically unstable. Preserve exact unit
    vectors in the forward pass while scaling the normalization Jacobian so its
    local gain is at most 1. The straight-through scale is detached and uses
    only reductions supported by the fused Colab recurrence backend.
    """
    value = value.float()
    norm_sq = value.square().sum(dim=-1, keepdim=True)
    norm = norm_sq.clamp_min(1e-12).sqrt()
    normalized = value / norm
    backward_scale = norm.clamp(max=1.0).detach()
    return normalized.detach() + (normalized - normalized.detach()) * backward_scale


def _condition_address_input(value: torch.Tensor) -> torch.Tensor:
    """Scale tiny address inputs in forward while keeping source gradients tame."""
    scaled = value * WorkspaceBroadcast._ADDRESS_PROJECTION_INPUT_SCALE
    return value + (scaled - value).detach()


def _cap_vector_rms(value: torch.Tensor, max_rms: float) -> torch.Tensor:
    """Bound each vector's RMS while preserving direction and smaller values."""
    value_float = value.float()
    rms_sq = value_float.square().mean(dim=-1, keepdim=True).clamp_min(1e-12)
    rms = rms_sq.sqrt()
    scale = (float(max_rms) / rms).clamp(max=1.0)
    return value * scale.to(value.dtype)


def _cap_readback_to_reference(
    readback: torch.Tensor, reference: torch.Tensor, max_ratio: float
) -> torch.Tensor:
    """Keep semantic-memory residuals bounded relative to their current writer."""
    reference_rms_sq = (
        reference.detach().float().square().mean(dim=-1, keepdim=True).clamp_min(1e-12)
    )
    readback_rms_sq = readback.float().square().mean(dim=-1, keepdim=True).clamp_min(1e-12)
    reference_rms = reference_rms_sq.sqrt()
    readback_rms = readback_rms_sq.sqrt()
    scale = (reference_rms * float(max_ratio) / readback_rms).clamp(max=1.0)
    return readback * scale.to(readback.dtype)


class WorkspaceBroadcast(nn.Module):
    """Competitive MoE-selected writes and all-expert shared-slot readback.

    Inputs are specialist deltas ``(B, E, D)`` and sparse MoE routes
    ``(B, K)``/weights ``(B, K)``. The router's selection determines which
    specialists write. By default writes/readback use shared content slots.
    Learned entity-keyed mode stores learned address/value pairs; semantic
    entity mode accepts stable token-ID addresses and an event write mask while
    retaining learned specialist values.
    """

    _RECONSIDERATION_ERROR_EMA_DECAY = 0.9
    _SEMANTIC_ENTITY_COUNT = 64
    _SEMANTIC_POSITION_COUNT = 16
    _SEMANTIC_CONTENT_RMS_LIMIT = 1.0
    _SEMANTIC_READBACK_MAX_RATIO = 1.0
    # MoE residuals may be scaled to ~1e-5 before entering workspace. This
    # homogeneous scale improves address-projection gradient conditioning;
    # address normalization leaves forward similarities unchanged.
    _ADDRESS_PROJECTION_INPUT_SCALE = 10_000.0

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
        entity_keyed_writes: bool = False,
        semantic_entity_addressing: bool = False,
        permutation_transactional: bool = False,
        nth_farthest_exact_readout: bool = False,
        retrieval_exact_readout: bool = False,
        key_match_threshold: float = 0.5,
        diagnostics_enabled: bool = True,
    ):
        super().__init__()
        if d_model < 1 or num_experts < 1:
            raise ValueError("d_model and num_experts must be positive")
        if num_slots < 0:
            raise ValueError("num_slots must be >= 0")
        if compile_recurrence and num_slots == 0:
            raise ValueError("compiled workspace recurrence requires at least one slot")
        if not math.isfinite(key_match_threshold) or not -1.0 <= key_match_threshold <= 1.0:
            raise ValueError("key_match_threshold must be finite and in [-1, 1]")
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
        self.entity_keyed_writes = bool(entity_keyed_writes)
        self.semantic_entity_addressing = bool(semantic_entity_addressing)
        # Semantic IDs are exact one-hot keys over the 64-entity vocabulary.
        # Keeping padded d_model-wide key vectors through every recurrent
        # state duplicates 448 guaranteed-zero features per slot and token.
        # Store only that address code; non-semantic learned addresses retain
        # their full width and their existing behavior.
        self._address_dim = (
            self._SEMANTIC_ENTITY_COUNT if self.semantic_entity_addressing else d_model
        )
        self.permutation_transactional = bool(permutation_transactional)
        self.nth_farthest_exact_readout = bool(nth_farthest_exact_readout)
        self.retrieval_exact_readout = bool(retrieval_exact_readout)
        if self.nth_farthest_exact_readout and not self.semantic_entity_addressing:
            raise ValueError("Nth-farthest exact readout requires semantic entity addressing")
        if self.retrieval_exact_readout and not self.semantic_entity_addressing:
            raise ValueError("retrieval exact readout requires semantic entity addressing")
        if self.permutation_transactional and not self.semantic_entity_addressing:
            raise ValueError("permutation transactional mode requires semantic entity addressing")
        if self.semantic_entity_addressing and not self.entity_keyed_writes:
            raise ValueError("semantic entity addressing requires entity-keyed workspace writes")
        if self.semantic_entity_addressing and d_model < self._SEMANTIC_ENTITY_COUNT:
            raise ValueError("semantic entity addressing requires workspace width >= 64")
        if self.semantic_entity_addressing and key_match_threshold <= 0:
            raise ValueError("semantic entity addressing requires a positive key-match threshold")
        self.key_match_threshold = float(key_match_threshold)
        self.semantic_state_decoder = (
            nn.Linear(d_model, self._SEMANTIC_POSITION_COUNT)
            if self.semantic_entity_addressing
            else None
        )
        self.semantic_vector_decoder = (
            nn.Linear(d_model, 6 * 16) if self.nth_farthest_exact_readout else None
        )
        self.semantic_retrieval_value_decoder = (
            nn.Linear(d_model, 16) if self.retrieval_exact_readout else None
        )
        self._last_semantic_state_predictions: list[tuple[int, torch.Tensor]] = []
        self._last_semantic_query_predictions: list[
            tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        self._last_semantic_pair_commits: list[tuple[int, torch.Tensor]] = []
        self._last_nth_farthest_vector_predictions: list[tuple[int, torch.Tensor]] = []
        self._last_nth_farthest_query_predictions: list[
            tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        self._last_retrieval_value_predictions: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        self._last_retrieval_query_predictions: list[
            tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        self._compiled_recurrence_step = None
        self._compiled_recurrence_failed = False
        self._compiled_fallback_warned = False
        # Optional evaluation-only observer. Kept unset during training so it
        # adds no synchronization or work to the recurrence.
        self._initium_address_assignment_observer = None
        self._last_address_assignment_indices = None
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.address_write = (
            nn.Linear(d_model, d_model, bias=False)
            if self.entity_keyed_writes and not self.semantic_entity_addressing
            else None
        )
        self.address_read = (
            nn.Linear(d_model, d_model, bias=False) if self.entity_keyed_writes else None
        )
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
        self.diagnostics_enabled = bool(diagnostics_enabled)
        self._last_step_diag: dict[str, torch.Tensor] = {}
        self._last_diag: dict[str, torch.Tensor] = {}
        self._bistable_diagnostic_totals: dict[str, torch.Tensor] = {}
        self._bistable_diagnostic_calls = 0
        nn.init.zeros_(self.write_gate.weight)
        nn.init.zeros_(self.write_gate.bias)
        if self.prediction_error_to_write_gate is not None:
            nn.init.zeros_(self.prediction_error_to_write_gate.weight)

    def pop_semantic_state_predictions(self) -> list[tuple[int, torch.Tensor]]:
        """Return and clear auxiliary predictions from the most recent call."""
        predictions = self._last_semantic_state_predictions
        self._last_semantic_state_predictions = []
        return predictions

    def pop_semantic_query_predictions(
        self,
    ) -> list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        predictions = self._last_semantic_query_predictions
        self._last_semantic_query_predictions = []
        return predictions

    def pop_semantic_pair_commits(self) -> list[tuple[int, torch.Tensor]]:
        commits = self._last_semantic_pair_commits
        self._last_semantic_pair_commits = []
        return commits

    def pop_nth_farthest_vector_predictions(self) -> list[tuple[int, torch.Tensor]]:
        predictions = self._last_nth_farthest_vector_predictions
        self._last_nth_farthest_vector_predictions = []
        return predictions

    def pop_nth_farthest_query_predictions(
        self,
    ) -> list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
        predictions = self._last_nth_farthest_query_predictions
        self._last_nth_farthest_query_predictions = []
        return predictions

    def _semantic_vector_logits(
        self, state: torch.Tensor, entity_ids: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.semantic_vector_decoder is None:
            raise RuntimeError("Nth-farthest vector decoding is not enabled")
        address_state, content_state = torch.split(state, (self._address_dim, self.d_model), dim=-1)
        safe_ids = entity_ids.clamp(min=0, max=self._SEMANTIC_ENTITY_COUNT - 1)
        queries = F.one_hot(safe_ids, num_classes=self._SEMANTIC_ENTITY_COUNT).to(
            device=state.device, dtype=torch.float32
        )
        address_state = address_state.float()
        occupied = address_state.square().sum(dim=-1) > 1e-12
        scores = torch.einsum("bqd,bsd->bqs", queries, _normalize_address(address_state))
        scores = scores.masked_fill(~occupied.unsqueeze(1), -2.0)
        best_scores, slot_indices = scores.max(dim=-1)
        found = valid_mask & best_scores.ge(self.key_match_threshold)
        read_values = content_state.gather(
            1, slot_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
        ).float()
        read_values = read_values * found.unsqueeze(-1).to(read_values.dtype)
        return self.semantic_vector_decoder(read_values).reshape(*read_values.shape[:-1], 6, 16)

    def pop_retrieval_value_predictions(
        self,
    ) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
        predictions = self._last_retrieval_value_predictions
        self._last_retrieval_value_predictions = []
        return predictions

    def pop_retrieval_query_predictions(
        self,
    ) -> list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
        predictions = self._last_retrieval_query_predictions
        self._last_retrieval_query_predictions = []
        return predictions

    def _semantic_retrieval_value_logits(
        self, state: torch.Tensor, entity_ids: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read scalar values from exact semantic entity addresses."""
        if self.semantic_retrieval_value_decoder is None:
            raise RuntimeError("retrieval exact readout is not enabled")
        address_state, content_state = torch.split(state, (self._address_dim, self.d_model), dim=-1)
        safe_ids = entity_ids.clamp(min=0, max=self._SEMANTIC_ENTITY_COUNT - 1)
        queries = F.one_hot(safe_ids, num_classes=self._SEMANTIC_ENTITY_COUNT).to(
            device=state.device, dtype=torch.float32
        )
        address_state = address_state.float()
        occupied = address_state.square().sum(dim=-1) > 1e-12
        scores = torch.einsum("bqd,bsd->bqs", queries, _normalize_address(address_state))
        scores = scores.masked_fill(~occupied.unsqueeze(1), -2.0)
        best_scores, slot_indices = scores.max(dim=-1)
        found = valid_mask & best_scores.ge(self.key_match_threshold)
        read_values = content_state.gather(
            1, slot_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
        ).float()
        read_values = read_values * found.unsqueeze(-1).to(read_values.dtype)
        return self.semantic_retrieval_value_decoder(read_values), found

    def _nth_farthest_query_logits(
        self,
        state: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
        query_vector: torch.Tensor,
        rank: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vector_logits = self._semantic_vector_logits(state, candidate_ids, candidate_mask)
        decoded_vectors = vector_logits.argmax(dim=-1).float()
        dimensions = query_vector.ne(-1).sum(dim=-1).clamp(min=1, max=6)
        dimension_mask = torch.arange(6, device=state.device).unsqueeze(0) < dimensions.unsqueeze(1)
        expected_candidates = candidate_mask & candidate_ids.ge(0)
        address_state = torch.split(state, (self._address_dim, self.d_model), dim=-1)[0].float()
        address_queries = F.one_hot(
            candidate_ids.clamp(min=0, max=self._SEMANTIC_ENTITY_COUNT - 1),
            num_classes=self._SEMANTIC_ENTITY_COUNT,
        ).to(device=state.device, dtype=torch.float32)
        occupied = address_state.square().sum(dim=-1) > 1e-12
        address_scores = torch.einsum(
            "bqd,bsd->bqs", address_queries, _normalize_address(address_state)
        ).masked_fill(~occupied.unsqueeze(1), -2.0)
        found = address_scores.max(dim=-1).values.ge(self.key_match_threshold)
        candidate_mask = expected_candidates & found
        active = expected_candidates.any(dim=-1)
        all_candidates_found = (~expected_candidates | found).all(dim=-1)
        candidate_counts = candidate_mask.sum(dim=-1)
        rank_valid = rank.ge(0) & rank.lt(candidate_counts)
        query_valid = active & all_candidates_found & rank_valid
        safe_mask = candidate_mask.clone()
        safe_ids = candidate_ids.clone()
        safe_vectors = decoded_vectors * dimension_mask.unsqueeze(1)
        safe_query = query_vector.clamp(min=0).float() * dimension_mask
        safe_rank = torch.minimum(rank.clamp_min(0), candidate_counts.clamp_min(1) - 1)
        safe_mask[:, 0] = torch.where(active, safe_mask[:, 0], True)
        safe_ids[:, 0] = torch.where(active, safe_ids[:, 0], 0)
        safe_vectors[:, 0] = torch.where(active.unsqueeze(-1), safe_vectors[:, 0], 0)
        safe_query = torch.where(active.unsqueeze(-1), safe_query, 0)
        safe_rank = torch.where(active, safe_rank, 0)
        selected, distances = select_nth_farthest(
            safe_vectors,
            safe_query,
            safe_ids,
            safe_rank,
            safe_mask,
        )
        selected = torch.where(query_valid, selected, torch.full_like(selected, -1))
        logits = state.new_full(
            (state.shape[0], self._SEMANTIC_ENTITY_COUNT), -20.0, dtype=torch.float32
        )
        logits.scatter_(1, selected.clamp_min(0).unsqueeze(1), 20.0)
        logits = torch.where(query_valid.unsqueeze(-1), logits, -20.0)
        return logits, selected, distances, active

    def _apply_semantic_pair_swap(
        self, state: torch.Tensor, entity_ids: torch.Tensor, active: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exchange two addressed payloads from the same pre-commit state."""
        address_state, content_state = torch.split(state, (self._address_dim, self.d_model), dim=-1)
        safe_ids = entity_ids.clamp(min=0, max=self._SEMANTIC_ENTITY_COUNT - 1)
        keys = F.one_hot(safe_ids, num_classes=self._SEMANTIC_ENTITY_COUNT).to(
            device=state.device, dtype=torch.float32
        )
        scores = torch.einsum("bpd,bsd->bps", keys, _normalize_address(address_state.float()))
        occupied = address_state.detach().float().square().sum(dim=-1) > 1e-12
        scores = scores.masked_fill(~occupied.unsqueeze(1), -2.0)
        best_scores, slot_ids = scores.max(dim=-1)
        valid = (
            active
            & entity_ids.ge(0).all(dim=-1)
            & entity_ids[:, 0].ne(entity_ids[:, 1])
            & best_scores.ge(self.key_match_threshold).all(dim=-1)
            & slot_ids[:, 0].ne(slot_ids[:, 1])
        )
        left_content = content_state.gather(
            1, slot_ids[:, 0, None, None].expand(-1, 1, self.d_model)
        ).squeeze(1)
        right_content = content_state.gather(
            1, slot_ids[:, 1, None, None].expand(-1, 1, self.d_model)
        ).squeeze(1)
        left_slot = F.one_hot(slot_ids[:, 0], num_classes=self.num_slots).bool()
        right_slot = F.one_hot(slot_ids[:, 1], num_classes=self.num_slots).bool()
        left_slot = left_slot & valid.unsqueeze(-1)
        right_slot = right_slot & valid.unsqueeze(-1)
        exchanged = torch.where(left_slot.unsqueeze(-1), right_content.unsqueeze(1), content_state)
        exchanged = torch.where(right_slot.unsqueeze(-1), left_content.unsqueeze(1), exchanged)
        return torch.cat((address_state, exchanged), dim=-1), valid

    def _semantic_query_class_logits(
        self, state: torch.Tensor, query_kind: torch.Tensor, query_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode object->position and position->object answers from slot values."""
        if self.semantic_state_decoder is None:
            raise RuntimeError("semantic query readout requires semantic entity addressing")
        address_state, content_state = torch.split(state, (self._address_dim, self.d_model), dim=-1)
        occupied = address_state.detach().float().square().sum(dim=-1) > 1e-12
        entity_to_position = self._semantic_state_logits(
            state, query_ids[:, None], query_kind.eq(1)[:, None]
        ).squeeze(1)

        slot_position_logits = self.semantic_state_decoder(content_state.float())
        slot_position_log_probs = torch.log_softmax(slot_position_logits.float(), dim=-1)
        safe_positions = query_ids.clamp(min=0, max=self._SEMANTIC_POSITION_COUNT - 1)
        entity_position_scores = slot_position_log_probs.gather(
            -1,
            safe_positions[:, None, None].expand(-1, self.num_slots, 1),
        ).squeeze(-1)
        address_ids = address_state[..., : self._SEMANTIC_ENTITY_COUNT].argmax(dim=-1)
        entity_keys = F.one_hot(
            address_ids, num_classes=self._SEMANTIC_ENTITY_COUNT
        ).bool() & occupied.unsqueeze(-1)
        entity_to_slot = entity_keys.transpose(1, 2)
        entity_scores = torch.where(
            entity_to_slot,
            entity_position_scores[:, None, :],
            torch.full_like(entity_position_scores[:, None, :], -1e4),
        ).amax(dim=-1)
        return entity_to_position, entity_scores

    def _semantic_state_logits(
        self, state: torch.Tensor, entity_ids: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.semantic_state_decoder is None:
            raise RuntimeError("semantic state decoding requires semantic entity addressing")
        address_state, content_state = torch.split(state, (self._address_dim, self.d_model), dim=-1)
        safe_ids = entity_ids.clamp(min=0, max=self._SEMANTIC_ENTITY_COUNT - 1)
        queries = F.one_hot(safe_ids, num_classes=self._SEMANTIC_ENTITY_COUNT).to(
            device=state.device, dtype=torch.float32
        )
        address_state = address_state.float()
        occupied = address_state.square().sum(dim=-1) > 1e-12
        scores = torch.einsum("bqd,bsd->bqs", queries, _normalize_address(address_state))
        scores = scores.masked_fill(~occupied.unsqueeze(1), -1e4)
        read_weights = torch.softmax(scores, dim=-1)
        read_weights = read_weights * occupied.unsqueeze(1).to(read_weights.dtype)
        read_weights = read_weights / read_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        read_weights = read_weights * valid_mask.unsqueeze(-1).to(read_weights.dtype)
        read_values = torch.einsum("bqs,bsd->bqd", read_weights, content_state.float())
        return self.semantic_state_decoder(read_values)

    def init_state(self, batch: int, *, device, dtype) -> torch.Tensor:
        state_width = self._address_dim + self.d_model if self.entity_keyed_writes else self.d_model
        return torch.zeros(batch, self.num_slots, state_width, device=device, dtype=dtype)

    def _entity_keyed_write_proposals(
        self,
        write_candidates: torch.Tensor,
        write_weights: torch.Tensor,
        address_state: torch.Tensor,
        candidate_addresses: torch.Tensor | None = None,
        projected_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Route each candidate to a matching address or allocate a free slot.

        Assignment is discrete (nearest normalized address); gradients still
        flow through the address projections and the selected key/value update.
        When several new candidates arrive together, their rank assigns them
        distinct free slots before any slot is overwritten.
        """
        if candidate_addresses is None:
            if self.address_write is None:
                raise ValueError("semantic entity addressing requires explicit entity IDs")
            candidate_addresses = self.address_write(_condition_address_input(write_candidates))
        candidate_addresses = _normalize_address(candidate_addresses)
        stored_addresses = _normalize_address(address_state)
        occupied = address_state.detach().float().square().sum(dim=-1) > 1e-12
        working_addresses = stored_addresses
        active = write_weights > 0
        assignments = []
        assignment_indices = []
        matched_candidates = []
        allocated_candidates = []
        overflow_candidates = []
        # Top-k is typically one, so this short loop makes same-token duplicate
        # entity proposals observe earlier allocations without expanding a
        # (batch, experts, experts, slots) collision tensor.
        for candidate_index in range(write_candidates.shape[1]):
            candidate = candidate_addresses[:, candidate_index]
            candidate_active = active[:, candidate_index]
            similarity = torch.einsum("bd,bsd->bs", candidate, working_addresses)
            similarity = similarity.masked_fill(~occupied, -2.0)
            best_similarity, best_slot = similarity.max(dim=-1)
            needs_allocation = candidate_active & (best_similarity < self.key_match_threshold)
            free = ~occupied
            can_allocate = needs_allocation & free.any(dim=-1)
            capacity_overflow = needs_allocation & ~can_allocate
            free_slot = free.to(torch.int64).argmax(dim=-1)
            destination = torch.where(can_allocate, free_slot, best_slot)
            assignment_active = candidate_active
            if self.semantic_entity_addressing:
                # Never overwrite a different semantic identity when all slots
                # are occupied. Dropping that write preserves existing bindings.
                assignment_active = candidate_active & ~capacity_overflow
            assignment = F.one_hot(destination, num_classes=self.num_slots).to(
                candidate_addresses.dtype
            )
            assignment = assignment * assignment_active.unsqueeze(-1).to(assignment.dtype)
            assignments.append(assignment)
            assignment_indices.append(
                torch.where(assignment_active, destination, torch.full_like(destination, -1))
            )
            matched_candidates.append(candidate_active & ~needs_allocation)
            allocated_candidates.append(can_allocate)
            overflow_candidates.append(capacity_overflow)
            new_address_mask = F.one_hot(
                destination, num_classes=self.num_slots
            ).bool() & can_allocate.unsqueeze(-1)
            working_addresses = torch.where(
                new_address_mask.unsqueeze(-1), candidate.unsqueeze(1), working_addresses
            )
            occupied = occupied | (
                F.one_hot(destination, num_classes=self.num_slots).bool()
                & assignment_active.unsqueeze(-1)
            )
        assignment = torch.stack(assignments, dim=1)
        observer = self._initium_address_assignment_observer
        if observer is not None:
            assignment_index_tensor = torch.stack(assignment_indices, dim=1)
            self._last_address_assignment_indices = torch.where(
                active,
                assignment_index_tensor,
                torch.full_like(assignment_index_tensor, -1),
            )
        weighted_assignment = assignment * write_weights.float().unsqueeze(-1)
        mass = weighted_assignment.sum(dim=1)
        denominator = mass.clamp_min(1e-6)

        values = projected_values if projected_values is not None else self.value(write_candidates)
        value_proposal = torch.einsum(
            "bes,bed->bsd", weighted_assignment.to(values.dtype), values
        ) / denominator.to(values.dtype).unsqueeze(-1)
        if self.semantic_entity_addressing:
            value_proposal = _cap_vector_rms(value_proposal, self._SEMANTIC_CONTENT_RMS_LIMIT)
        address_proposal = torch.einsum(
            "bes,bed->bsd", weighted_assignment, candidate_addresses
        ) / denominator.unsqueeze(-1)
        address_proposal = _normalize_address(address_proposal)
        active_count = active.sum().float().clamp_min(1.0)
        address_diagnostics = {
            "address_match_frac": torch.stack(matched_candidates, dim=1).sum().float()
            / active_count,
            "address_allocation_frac": torch.stack(allocated_candidates, dim=1).sum().float()
            / active_count,
            "address_capacity_overflow_frac": torch.stack(overflow_candidates, dim=1).sum().float()
            / active_count,
            "address_occupied_frac": occupied.float().mean(),
        }
        return value_proposal, mass, address_proposal, address_diagnostics

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
        entity_write_mask: torch.Tensor | None,
    ):
        """Pure tensor step suitable for torch.compile and autograd."""
        weights = route_weights.float()
        write_weights = weights
        if entity_write_mask is not None:
            write_weights = write_weights * entity_write_mask.reshape(-1, 1).to(write_weights.dtype)
        if self.entity_keyed_writes:
            address_state, content_state = torch.split(
                state, (self._address_dim, self.d_model), dim=-1
            )
            proposal, mass, address_proposal, address_diagnostics = (
                self._entity_keyed_write_proposals(
                    selected_deltas,
                    write_weights,
                    address_state,
                    candidate_addresses=write_keys,
                    projected_values=write_values,
                )
            )
        else:
            address_state = None
            address_proposal = None
            address_diagnostics = None
            content_state = state
            scores = torch.einsum(
                "bed,bsd->bes", write_keys.float(), self.query(state).float()
            ) / math.sqrt(self.d_model)
            attention = scores.softmax(dim=-1).to(selected_deltas.dtype)
            proposal = torch.einsum("be,bes,bed->bsd", write_weights, attention, write_values)
            mass = torch.einsum("be,bes->bs", write_weights, attention)
            proposal = proposal / mass.unsqueeze(-1).clamp_min(1e-6)
        update_logits = self.write_gate(torch.cat([content_state, proposal], dim=-1))
        if self.prediction_error_to_write_gate is not None:
            error_value = prediction_error.reshape(state.shape[0], 1).float().detach()
            update_logits = update_logits + self.prediction_error_to_write_gate(
                error_value
            ).unsqueeze(1)
        update = torch.sigmoid(update_logits)
        # A write proposal belongs to an individual slot. Masking only by
        # whether the batch item has any writer lets unrelated slots decay
        # toward their zero proposal on every token, corrupting persistent
        # values and potentially erasing their semantic addresses.
        has_writer = mass > 0
        update = update * has_writer.unsqueeze(-1).to(update.dtype)
        eligible = None
        surprise = None
        reconsideration_trigger = None
        if self.reconsideration_enabled:
            old_probe = content_state.detach().float()
            # Spell norms as primitive reductions instead of linalg.vector_norm.
            # Some Colab Triton/Inductor runtimes fail lowering a fused kernel
            # containing vector_norm (PassManager::run failed).
            old_norm_sq = old_probe.square().sum(dim=-1)
            eligible = (old_norm_sq > 1e-12) & has_writer
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
            innovation_trigger = eligible & (surprise >= self.reconsideration_surprise_threshold)
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
                reconsideration_trigger = eligible & (ranks.reshape_as(eligible) < target_count)
            else:
                reconsideration_trigger = innovation_trigger
            update = torch.where(
                reconsideration_trigger.unsqueeze(-1),
                torch.ones_like(update),
                update,
            )
        next_content = update.mul(proposal).add((1.0 - update).mul(content_state))
        if self.semantic_entity_addressing:
            next_content = _cap_vector_rms(next_content, self._SEMANTIC_CONTENT_RMS_LIMIT)
        if self.entity_keyed_writes:
            address_update = update[..., : self._address_dim]
            next_address = address_update.mul(address_proposal).add(
                (1.0 - address_update).mul(address_state)
            )
            next_address = _normalize_address(next_address)
            next_state = torch.cat((next_address, next_content), dim=-1)
            occupied = next_address.detach().float().square().sum(dim=-1) > 1e-12
            read_scores = torch.einsum(
                "bed,bsd->bes",
                _normalize_address(read_queries)[..., : self._address_dim],
                _normalize_address(next_address),
            )
            read_scores = read_scores.masked_fill(~occupied.unsqueeze(1), -1e4)
            read_weights = read_scores.softmax(dim=-1).to(selected_deltas.dtype)
            read_weights = read_weights * occupied.unsqueeze(1).to(read_weights.dtype)
            read_weights = read_weights / read_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            address_match_frac = address_diagnostics["address_match_frac"]
            address_allocation_frac = address_diagnostics["address_allocation_frac"]
            address_capacity_overflow_frac = address_diagnostics["address_capacity_overflow_frac"]
            address_occupied_frac = address_diagnostics["address_occupied_frac"]
        else:
            next_state = next_content
            read_scores = torch.einsum(
                "bed,bsd->bes", read_queries.float(), self.key(next_state).float()
            ) / math.sqrt(self.d_model)
            read_weights = read_scores.softmax(dim=-1).to(selected_deltas.dtype)
            address_match_frac = weights.new_zeros(())
            address_allocation_frac = weights.new_zeros(())
            address_capacity_overflow_frac = weights.new_zeros(())
            address_occupied_frac = weights.new_zeros(())
        read_values = self.value(next_content)
        broadcast = torch.einsum("bes,bsd->bed", read_weights, read_values)
        if self.semantic_entity_addressing:
            broadcast = _cap_readback_to_reference(
                broadcast, selected_deltas, self._SEMANTIC_READBACK_MAX_RATIO
            )
        read_scale = torch.sigmoid(self.read_gate).to(selected_deltas.dtype)
        output = ((selected_deltas + read_scale * broadcast) * route_weights.unsqueeze(-1)).sum(
            dim=1
        )

        # Keep logging reductions and indexed accumulation outside the compiled
        # transition. Triton can fail lowering their fused reduction graph on
        # some Colab builds; the raw tensors below are already produced by the
        # transition and let the caller compute exactly the same telemetry in
        # eager mode without changing state/update math.
        return (
            output,
            next_state,
            update,
            broadcast,
            eligible,
            surprise,
            reconsideration_trigger,
            address_match_frac,
            address_allocation_frac,
            address_capacity_overflow_frac,
            address_occupied_frac,
        )

    def _get_compiled_recurrence_step(self):
        if self._compiled_recurrence_step is None:
            compiler = getattr(torch, "compile", None)
            if compiler is None:
                raise RuntimeError("--workspace-compile-recurrence requires PyTorch 2.4 or newer")
            self._compiled_recurrence_step = compiler(
                self._compiled_recurrence_step_impl,
                # The compiled transition is called once per sequence token,
                # and all calls retain autograd state until the sequence
                # backward pass. Use the options argument without a mode
                # argument: current PyTorch rejects specifying both.
                fullgraph=True,
                dynamic=False,
                # Respect the lightweight default path even if the Colab
                # process has globally enabled max-autotune. This recurrence
                # is a small per-token graph; exhaustive GEMM tuning adds
                # compile-time work and can select unsupported Triton paths
                # on the small SM-count GPUs commonly assigned by Colab.
                # CUDA graph replay is intentionally disabled for this inner
                # recurrence: many token calls retain autograd state before
                # the sequence-level backward, preventing CUDAGraph Trees from
                # reclaiming the prior call's graph pool. Keep the compiled
                # Triton transition while avoiding that ineffective fast path.
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
        shuffle_readback_across_batch: bool = False,
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
        entity_write_mask: torch.Tensor | None = None,
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
                shuffle_readback_across_batch=shuffle_readback_across_batch,
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
                entity_write_mask=entity_write_mask,
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
                    shuffle_readback_across_batch=shuffle_readback_across_batch,
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
                    entity_write_mask=entity_write_mask,
                )
        return self._forward_step(
            specialist_deltas.float(),
            route_indices,
            route_weights.float(),
            state.float(),
            no_selection=no_selection,
            reset_each_step=reset_each_step,
            bypass_broadcast=bypass_broadcast,
            shuffle_readback_across_batch=shuffle_readback_across_batch,
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
            entity_write_mask=entity_write_mask,
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
        shuffle_readback_across_batch: bool = False,
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
        entity_write_mask: torch.Tensor | None = None,
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
        expected_experts = (
            route_indices.shape[-1] if selected_specialists_only else self.num_experts
        )
        if (experts, width) != (expected_experts, self.d_model):
            raise ValueError("specialist delta shape does not match workspace configuration")
        if selected_specialists_only and no_selection:
            raise ValueError("random workspace selection requires all expert candidates")
        state_width = self._address_dim + self.d_model if self.entity_keyed_writes else self.d_model
        if state.shape != (batch, self.num_slots, state_width):
            raise ValueError("workspace state has an incompatible batch/slot/feature shape")

        if reset_each_step:
            state = torch.zeros_like(state)
        if self.entity_keyed_writes:
            address_state, content_state = torch.split(
                state, (self._address_dim, self.d_model), dim=-1
            )
        else:
            address_state, content_state = None, state
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
        if entity_write_mask is not None:
            if not self.semantic_entity_addressing:
                raise ValueError("entity_write_mask requires semantic entity addressing")
            if entity_write_mask.shape != (batch,):
                raise ValueError("entity_write_mask must have shape (batch,)")
            write_weights = write_weights * entity_write_mask.reshape(-1, 1).to(write_weights.dtype)

        # Shared mode softly pools selected candidates into every slot.
        # Entity-keyed mode stores a separate address per slot and routes each
        # candidate to its nearest matching address, allocating free slots for
        # novel addresses while capacity remains.
        address_proposal = None
        address_diagnostics = {}
        if self.entity_keyed_writes:
            assert address_state is not None
            (
                slot_proposal,
                mass,
                address_proposal,
                address_diagnostics,
            ) = self._entity_keyed_write_proposals(
                write_candidates,
                write_weights,
                address_state,
                candidate_addresses=projected_write_keys,
                projected_values=projected_write_values,
            )
        else:
            keys = (
                projected_write_keys
                if projected_write_keys is not None
                else self.key(write_candidates)
            )
            scores = torch.einsum("bed,bsd->bes", keys.float(), self.query(content_state).float())
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
        old_inputs = content_state
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
        # Keep the compiled transition aligned with the eager per-slot mask.
        has_writer = mass > 0
        update = update * has_writer.unsqueeze(-1).to(update.dtype)
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
            eligible = (old_norm > 1e-6) & has_writer
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
                candidate_mask=has_writer.any(dim=-1),
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
        next_content = update.mul(slot_inputs).add((1.0 - update).mul(old_inputs))
        if self.semantic_entity_addressing:
            next_content = _cap_vector_rms(next_content, self._SEMANTIC_CONTENT_RMS_LIMIT)
        if self.entity_keyed_writes:
            assert address_state is not None and address_proposal is not None
            address_update = update[..., : self._address_dim]
            next_address = address_update.mul(address_proposal).add(
                (1.0 - address_update).mul(address_state)
            )
            next_address = _normalize_address(next_address)
            next_state = torch.cat((next_address, next_content), dim=-1)
        else:
            next_state = next_content
        if self.diagnostics_enabled and reconsideration_trigger is not None:
            triggered_count_raw = reconsideration_trigger.sum().float()
            triggered_count = triggered_count_raw.clamp_min(1.0)
            # Reconsideration telemetry measures the committed slot content
            # change. Entity-keyed state also contains an address vector, so
            # comparing the full concatenated state (2 * d_model) against
            # old_inputs (d_model) both misreports the metric and fails during
            # eager evaluation. Keep this consistent with the compiled path.
            actual_change = torch.linalg.vector_norm((next_content - old_inputs).float(), dim=-1)
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
                    "slot_rms": next_content.detach().float().square().mean().sqrt(),
                    "broadcast_rms": next_state.new_zeros((), dtype=torch.float32),
                    "specialist_rms": specialist_deltas.detach().float().square().mean().sqrt(),
                    **reconsideration_diag,
                    **bistable_diag,
                    **address_diagnostics,
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
        if self.entity_keyed_writes:
            assert self.address_read is not None
            read_queries = (
                projected_read_queries
                if projected_read_queries is not None
                else self.address_read(_condition_address_input(selected_deltas))
            )
        else:
            read_queries = (
                projected_read_queries
                if projected_read_queries is not None
                else self.query(selected_deltas)
            )
        read_state = (
            torch.roll(next_state, shifts=1, dims=0)
            if shuffle_readback_across_batch and next_state.shape[0] > 1
            else next_state
        )
        if self.entity_keyed_writes:
            read_addresses, read_content = torch.split(
                read_state, (self._address_dim, self.d_model), dim=-1
            )
            read_scores = torch.einsum(
                "bed,bsd->bes",
                _normalize_address(read_queries)[..., : self._address_dim],
                _normalize_address(read_addresses),
            )
            occupied = read_addresses.detach().float().square().sum(dim=-1) > 1e-12
            read_scores = read_scores.masked_fill(~occupied.unsqueeze(1), -1e4)
            read_weights = read_scores.softmax(dim=-1).to(specialist_deltas.dtype)
            read_weights = read_weights * occupied.unsqueeze(1).to(read_weights.dtype)
            read_weights = read_weights / read_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            read_content = read_state
            read_scores = torch.einsum(
                "bed,bsd->bes", read_queries.float(), self.key(read_content).float()
            )
            read_scores = read_scores / math.sqrt(self.d_model)
            read_weights = read_scores.softmax(dim=-1).to(specialist_deltas.dtype)
        read_values = self.value(read_content)
        broadcast = torch.einsum("bes,bsd->bed", read_weights, read_values)
        if self.semantic_entity_addressing:
            broadcast = _cap_readback_to_reference(
                broadcast, selected_deltas, self._SEMANTIC_READBACK_MAX_RATIO
            )
        read_scale = torch.sigmoid(self.read_gate).to(specialist_deltas.dtype)
        enriched = selected_deltas + read_scale * broadcast
        if self.diagnostics_enabled:
            self._last_step_diag = {
                "write_mass": write_weights.detach().sum(dim=-1).mean(),
                "active_write_frac": (write_weights.detach().sum(dim=-1) > 0).float().mean(),
                "write_gate_mean": update.detach().float().mean(),
                "slot_rms": next_content.detach().float().square().mean().sqrt(),
                "broadcast_rms": (read_scale.detach().float() * broadcast.detach().float())
                .square()
                .mean()
                .sqrt(),
                "specialist_rms": specialist_deltas.detach().float().square().mean().sqrt(),
                **reconsideration_diag,
                **bistable_diag,
                **address_diagnostics,
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
        shuffle_readback_across_batch: bool = False,
        bistable_noise_scale: float | None = None,
        bistable_selection_scale: float | None = None,
        deterministic_gate: bool = False,
        prediction_error: torch.Tensor | None = None,
        selected_specialists_only: bool = False,
        entity_ids: torch.Tensor | None = None,
        entity_write_mask: torch.Tensor | None = None,
        pair_swap_entity_ids: torch.Tensor | None = None,
        pair_swap_mask: torch.Tensor | None = None,
        pair_swap_steps: tuple[int, ...] | None = None,
        permutation_query_kind: torch.Tensor | None = None,
        permutation_query_ids: torch.Tensor | None = None,
        permutation_query_steps: tuple[int, ...] | None = None,
        state_supervision_entity_ids: torch.Tensor | None = None,
        state_supervision_mask: torch.Tensor | None = None,
        state_supervision_steps: tuple[int, ...] | None = None,
        nth_farthest_vector_entity_ids: torch.Tensor | None = None,
        nth_farthest_vector_mask: torch.Tensor | None = None,
        nth_farthest_vector_steps: tuple[int, ...] | None = None,
        nth_farthest_query_candidate_ids: torch.Tensor | None = None,
        nth_farthest_query_candidate_mask: torch.Tensor | None = None,
        nth_farthest_query_vectors: torch.Tensor | None = None,
        nth_farthest_query_ranks: torch.Tensor | None = None,
        nth_farthest_query_steps: tuple[int, ...] | None = None,
        retrieval_value_entity_ids: torch.Tensor | None = None,
        retrieval_value_mask: torch.Tensor | None = None,
        retrieval_value_steps: tuple[int, ...] | None = None,
        retrieval_query_entity_ids: torch.Tensor | None = None,
        retrieval_query_mask: torch.Tensor | None = None,
        retrieval_query_operation: torch.Tensor | None = None,
        retrieval_query_steps: tuple[int, ...] | None = None,
        collect_kv_records: bool = False,
    ):
        """Process time-major specialists ``(B,T,E,D)`` sequentially."""
        outputs: list[torch.Tensor] = []
        kv_records: list[torch.Tensor] = []
        kv_commit_masks: list[torch.Tensor] = []
        self._last_semantic_state_predictions = []
        self._last_semantic_query_predictions = []
        self._last_semantic_pair_commits = []
        self._last_nth_farthest_vector_predictions = []
        self._last_nth_farthest_query_predictions = []
        self._last_retrieval_value_predictions = []
        self._last_retrieval_query_predictions = []
        step_diags: list[dict[str, torch.Tensor]] | None = [] if self.diagnostics_enabled else None
        state_observer = getattr(self, "_initium_recurrent_state_observer", None)
        smoothed_prediction_error = None
        projected_write_keys = None
        projected_write_values = None
        projected_read_queries = None
        time_major_deltas = None
        time_major_route_indices = None
        time_major_route_weights = None
        time_major_entity_write_mask = None
        if self.semantic_entity_addressing:
            if entity_ids is None or entity_write_mask is None:
                raise ValueError(
                    "semantic entity addressing requires entity_ids and entity_write_mask"
                )
            expected_shape = specialist_deltas.shape[:2]
            if entity_ids.shape != expected_shape or entity_write_mask.shape != expected_shape:
                raise ValueError(
                    "entity_ids and entity_write_mask must match the specialist (batch, time) axes"
                )
            if entity_ids.dtype not in (torch.int32, torch.int64):
                raise TypeError("entity_ids must use an integer tensor dtype")
            if entity_write_mask.dtype != torch.bool:
                raise TypeError("entity_write_mask must use torch.bool")
        elif entity_ids is not None or entity_write_mask is not None:
            raise ValueError(
                "entity metadata is only accepted when semantic entity addressing is enabled"
            )
        if self.permutation_transactional:
            expected_pair_shape = (*specialist_deltas.shape[:2], 2)
            expected_metadata_shape = specialist_deltas.shape[:2]
            if (
                pair_swap_entity_ids is None
                or pair_swap_mask is None
                or pair_swap_entity_ids.shape != expected_pair_shape
                or pair_swap_mask.shape != expected_metadata_shape
            ):
                raise ValueError(
                    "transactional permutation mode requires pair swap IDs/mask "
                    "with shapes (batch,time,2) and (batch,time)"
                )
            if (
                permutation_query_kind is None
                or permutation_query_ids is None
                or permutation_query_kind.shape != expected_metadata_shape
                or permutation_query_ids.shape != expected_metadata_shape
            ):
                raise ValueError(
                    "transactional permutation mode requires query kind/ID tensors "
                    "with shape (batch,time)"
                )
            if pair_swap_entity_ids.dtype not in (torch.int32, torch.int64):
                raise TypeError("pair swap entity IDs must use an integer tensor dtype")
            if pair_swap_mask.dtype != torch.bool:
                raise TypeError("pair swap mask must use torch.bool")
            if permutation_query_kind.dtype not in (torch.int32, torch.int64):
                raise TypeError("permutation query kinds must use an integer tensor dtype")
            if permutation_query_ids.dtype not in (torch.int32, torch.int64):
                raise TypeError("permutation query IDs must use an integer tensor dtype")
            pair_swap_steps = tuple(sorted(set(pair_swap_steps or ())))
            permutation_query_steps = tuple(sorted(set(permutation_query_steps or ())))
            max_steps = specialist_deltas.shape[1]
            if any(step < 0 or step >= max_steps for step in pair_swap_steps):
                raise ValueError("pair swap step is outside the current sequence")
            if any(step < 0 or step >= max_steps for step in permutation_query_steps):
                raise ValueError("permutation query step is outside the current sequence")
        elif any(
            value is not None
            for value in (
                pair_swap_entity_ids,
                pair_swap_mask,
                pair_swap_steps,
                permutation_query_kind,
                permutation_query_ids,
                permutation_query_steps,
            )
        ):
            raise ValueError("transactional metadata requires permutation_transactional mode")
        if state_supervision_entity_ids is not None or state_supervision_mask is not None:
            if not self.semantic_entity_addressing:
                raise ValueError("state supervision requires semantic entity addressing")
            expected = (specialist_deltas.shape[0], specialist_deltas.shape[1], 16)
            if (
                state_supervision_entity_ids is None
                or state_supervision_mask is None
                or state_supervision_entity_ids.shape != expected
                or state_supervision_mask.shape != expected
            ):
                raise ValueError("state supervision IDs and mask must have shape (batch, time, 16)")
            if state_supervision_entity_ids.dtype not in (torch.int32, torch.int64):
                raise TypeError("state supervision entity IDs must use an integer tensor dtype")
            if state_supervision_mask.dtype != torch.bool:
                raise TypeError("state supervision mask must use torch.bool")
            state_supervision_steps = tuple(sorted(set(state_supervision_steps or ())))
            if any(
                step < 0 or step >= specialist_deltas.shape[1] for step in state_supervision_steps
            ):
                raise ValueError("state supervision step is outside the current sequence")
        elif state_supervision_steps:
            raise ValueError("state supervision steps require state supervision IDs and mask")
        if self.nth_farthest_exact_readout:
            expected_ids = (*specialist_deltas.shape[:2], 16)
            expected_vectors = (*specialist_deltas.shape[:2], 6)
            if (
                nth_farthest_vector_entity_ids is None
                or nth_farthest_vector_mask is None
                or nth_farthest_query_candidate_ids is None
                or nth_farthest_query_candidate_mask is None
                or nth_farthest_query_vectors is None
                or nth_farthest_query_ranks is None
            ):
                raise ValueError(
                    "Nth-farthest exact readout requires complete supervision/query metadata"
                )
            if (
                nth_farthest_vector_entity_ids.shape != expected_ids
                or nth_farthest_vector_mask.shape != expected_ids
                or nth_farthest_query_candidate_ids.shape != expected_ids
                or nth_farthest_query_candidate_mask.shape != expected_ids
                or nth_farthest_query_vectors.shape != expected_vectors
                or nth_farthest_query_ranks.shape != specialist_deltas.shape[:2]
            ):
                raise ValueError("Nth-farthest metadata shapes must match (batch,time,16[,6])")
            if (
                nth_farthest_vector_mask.dtype != torch.bool
                or nth_farthest_query_candidate_mask.dtype != torch.bool
            ):
                raise TypeError("Nth-farthest candidate masks must use bool dtype")
            integer_tensors = (
                nth_farthest_vector_entity_ids,
                nth_farthest_query_candidate_ids,
                nth_farthest_query_vectors,
                nth_farthest_query_ranks,
            )
            if any(value.dtype not in (torch.int32, torch.int64) for value in integer_tensors):
                raise TypeError("Nth-farthest IDs, coordinates, and ranks must use integer tensors")
            nth_farthest_vector_steps = tuple(sorted(set(nth_farthest_vector_steps or ())))
            nth_farthest_query_steps = tuple(sorted(set(nth_farthest_query_steps or ())))
            max_steps = specialist_deltas.shape[1]
            if any(
                step < 0 or step >= max_steps
                for step in (*nth_farthest_vector_steps, *nth_farthest_query_steps)
            ):
                raise ValueError("Nth-farthest event step is outside the current sequence")
        elif any(
            value is not None
            for value in (
                nth_farthest_vector_entity_ids,
                nth_farthest_vector_mask,
                nth_farthest_vector_steps,
                nth_farthest_query_candidate_ids,
                nth_farthest_query_candidate_mask,
                nth_farthest_query_vectors,
                nth_farthest_query_ranks,
                nth_farthest_query_steps,
            )
        ):
            raise ValueError("Nth-farthest metadata requires nth_farthest_exact_readout mode")
        if self.retrieval_exact_readout:
            expected_query_shape = (*specialist_deltas.shape[:2], 8)
            if (
                retrieval_value_entity_ids is None
                or retrieval_value_mask is None
                or retrieval_query_entity_ids is None
                or retrieval_query_mask is None
                or retrieval_query_operation is None
                or retrieval_query_entity_ids.shape != expected_query_shape
                or retrieval_query_mask.shape != expected_query_shape
                or retrieval_query_operation.shape != specialist_deltas.shape[:2]
                or retrieval_value_entity_ids.shape != specialist_deltas.shape[:2]
                or retrieval_value_mask.shape != specialist_deltas.shape[:2]
            ):
                raise ValueError(
                    "retrieval exact readout requires query IDs/mask shaped (batch,time,8) "
                    "and operations shaped (batch,time)"
                )
            if retrieval_query_mask.dtype != torch.bool or retrieval_value_mask.dtype != torch.bool:
                raise TypeError("retrieval masks must use torch.bool")
            if any(
                value.dtype not in (torch.int32, torch.int64)
                for value in (
                    retrieval_value_entity_ids,
                    retrieval_query_entity_ids,
                    retrieval_query_operation,
                )
            ):
                raise TypeError("retrieval query IDs and operations must be integer tensors")
            retrieval_value_steps = tuple(sorted(set(retrieval_value_steps or ())))
            retrieval_query_steps = tuple(sorted(set(retrieval_query_steps or ())))
            max_steps = specialist_deltas.shape[1]
            if any(
                step < 0 or step >= max_steps
                for step in (*retrieval_value_steps, *retrieval_query_steps)
            ):
                raise ValueError("retrieval event/query step is outside the current sequence")
        elif any(
            value is not None
            for value in (
                retrieval_value_steps,
                retrieval_value_entity_ids,
                retrieval_value_mask,
                retrieval_query_entity_ids,
                retrieval_query_mask,
                retrieval_query_operation,
                retrieval_query_steps,
            )
        ):
            raise ValueError("retrieval metadata requires retrieval_exact_readout mode")
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
            time_major_entity_write_mask = (
                entity_write_mask.transpose(0, 1).contiguous()
                if entity_write_mask is not None
                else None
            )
            selected_deltas = (
                time_major_deltas
                if selected_specialists_only
                else time_major_deltas.gather(
                    2,
                    time_major_route_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_model),
                )
            )
            write_candidates = time_major_deltas if no_selection else selected_deltas
            projection_context = (
                torch.autocast(device_type="cuda", enabled=False)
                if specialist_deltas.is_cuda
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with projection_context:
                if self.entity_keyed_writes:
                    assert self.address_read is not None
                    if self.semantic_entity_addressing:
                        assert entity_ids is not None
                        entity_ids_time_major = entity_ids.transpose(0, 1).contiguous()
                        safe_entity_ids = entity_ids_time_major.clamp(
                            min=0, max=self._SEMANTIC_ENTITY_COUNT - 1
                        )
                        identity_keys_time_major = F.one_hot(
                            safe_entity_ids, num_classes=self._SEMANTIC_ENTITY_COUNT
                        ).to(dtype=torch.float32)
                        # Keep the batch stride of each time slice stable as
                        # the curriculum changes sequence length. The expert
                        # axis remains a zero-copy expansion.
                        identity_keys_time_major = identity_keys_time_major * (
                            entity_ids_time_major.ge(0).unsqueeze(-1)
                        )
                        projected_write_keys = identity_keys_time_major.unsqueeze(2).expand(
                            -1, -1, write_candidates.shape[2], -1
                        )
                    else:
                        assert self.address_write is not None
                        projected_write_keys = self.address_write(
                            _condition_address_input(write_candidates.float())
                        )
                    if not bypass_broadcast:
                        learned_queries = self.address_read(
                            _condition_address_input(selected_deltas.float())
                        )
                        if self.semantic_entity_addressing:
                            assert entity_ids is not None
                            # The read projection remains d_model-wide because
                            # non-entity tokens use learned queries. Keep that
                            # interface while storing recurrent semantic keys
                            # compactly at _address_dim.
                            direct_queries = (
                                F.pad(
                                    identity_keys_time_major,
                                    (0, self.d_model - self._address_dim),
                                )
                                .unsqueeze(2)
                                .expand(-1, -1, selected_deltas.shape[2], -1)
                            )
                            semantic_query_mask = entity_ids.ge(0).transpose(0, 1)[:, :, None, None]
                            projected_read_queries = torch.where(
                                semantic_query_mask, direct_queries, learned_queries
                            )
                        else:
                            projected_read_queries = learned_queries
                else:
                    projected_write_keys = self.key(write_candidates.float())
                    if not bypass_broadcast:
                        projected_read_queries = self.query(selected_deltas.float())
                projected_write_values = self.value(write_candidates.float()).contiguous()
                if projected_read_queries is not None:
                    # Canonicalize the time-major read tensor before the
                    # recurrent compiler sees one token slice at a time. Its
                    # per-step batch strides must not depend on sequence length.
                    projected_read_queries = projected_read_queries.contiguous()
        compiled_step = None
        # The fused recurrence is a training optimization. Curriculum and
        # periodic evaluations use variable/partial batch sizes, which would
        # otherwise create new fullgraph specializations for each batch shape.
        # Keep evaluation on the eager path; its state transition is identical.
        if (
            self.compile_recurrence
            and self.training
            and torch.is_grad_enabled()
            and not shuffle_readback_across_batch
            and not self._compiled_recurrence_failed
        ):
            if not specialist_deltas.is_cuda:
                raise RuntimeError("--workspace-compile-recurrence requires a CUDA device")
            if no_selection or not selected_specialists_only or self.bistable_gate is not None:
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
            state_before_step = torch.zeros_like(state) if reset_each_step else state
            step_prediction_error = (
                prediction_error[:, step] if prediction_error is not None else None
            )
            step_entity_write_mask = (
                time_major_entity_write_mask[step]
                if time_major_entity_write_mask is not None
                else None
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
                previous_state = state
                try:
                    with torch.autocast(device_type="cuda", enabled=False):
                        (
                            out,
                            state,
                            compiled_update,
                            compiled_broadcast,
                            compiled_eligible,
                            compiled_surprise,
                            compiled_trigger,
                            compiled_address_match_frac,
                            compiled_address_allocation_frac,
                            compiled_address_capacity_overflow_frac,
                            compiled_address_occupied_frac,
                        ) = compiled_step(
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
                            step_entity_write_mask,
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
                            line.strip() for line in str(cause).splitlines() if line.strip()
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
                            compiler_details.append(f"{type(cause).__name__}: {detail[:700]}")
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
                    shuffle_readback_across_batch=shuffle_readback_across_batch,
                    bistable_noise_scale=bistable_noise_scale,
                    bistable_selection_scale=bistable_selection_scale,
                    deterministic_gate=deterministic_gate,
                    prediction_error=step_prediction_error,
                    reconsideration_error=smoothed_prediction_error,
                    reconsideration_random_scores=reconsideration_random_scores,
                    selected_specialists_only=selected_specialists_only,
                    projected_write_keys=(
                        projected_write_keys[step] if projected_write_keys is not None else None
                    ),
                    projected_write_values=(
                        projected_write_values[step] if projected_write_values is not None else None
                    ),
                    projected_read_queries=(
                        projected_read_queries[step] if projected_read_queries is not None else None
                    ),
                    entity_write_mask=step_entity_write_mask,
                )
                observer = self._initium_address_assignment_observer
                assignments = self._last_address_assignment_indices
                if observer is not None and assignments is not None:
                    observer(step, assignments.detach())
            else:
                if self.diagnostics_enabled:
                    step_weights = time_major_route_weights[step].detach().float()
                    step_routes = time_major_route_indices[step]
                    write_mass = step_weights.sum(dim=-1).mean()
                    active_write_frac = (step_weights.sum(dim=-1) > 0).float().mean()
                    write_gate_mean = compiled_update.detach().float().mean()
                    diagnostic_state = (
                        torch.split(state, (self._address_dim, self.d_model), dim=-1)[1]
                        if self.entity_keyed_writes
                        else state
                    )
                    slot_rms = diagnostic_state.detach().float().square().mean().sqrt()
                    broadcast_rms = (
                        (
                            torch.sigmoid(self.read_gate.detach()).float()
                            * compiled_broadcast.detach().float()
                        )
                        .square()
                        .mean()
                        .sqrt()
                    )
                    specialist_rms = time_major_deltas[step].detach().float().square().mean().sqrt()
                    writer_frequency = torch.zeros(
                        self.num_experts,
                        device=step_routes.device,
                        dtype=torch.float32,
                    ).scatter_add_(
                        0,
                        step_routes.reshape(-1),
                        step_weights.reshape(-1),
                    ) / max(step_routes.shape[0], 1)
                    self._last_step_diag = {
                        "write_mass": write_mass,
                        "active_write_frac": active_write_frac,
                        "write_gate_mean": write_gate_mean,
                        "slot_rms": slot_rms,
                        "broadcast_rms": broadcast_rms,
                        "specialist_rms": specialist_rms,
                        "writer_frequency": writer_frequency,
                    }
                    if self.entity_keyed_writes:
                        self._last_step_diag.update(
                            {
                                "address_match_frac": compiled_address_match_frac,
                                "address_allocation_frac": compiled_address_allocation_frac,
                                "address_capacity_overflow_frac": (
                                    compiled_address_capacity_overflow_frac
                                ),
                                "address_occupied_frac": compiled_address_occupied_frac,
                            }
                        )
                    if self.reconsideration_enabled:
                        eligible = compiled_eligible.detach()
                        surprise = compiled_surprise.detach()
                        trigger = compiled_trigger.detach()
                        eligible_count_raw = eligible.sum().float()
                        eligible_count = eligible_count_raw.clamp_min(1.0)
                        trigger_count = trigger.sum().float()
                        surprise_mean = (surprise * eligible).sum() / eligible_count
                        if self.entity_keyed_writes:
                            state_content = torch.split(
                                state, (self._address_dim, self.d_model), dim=-1
                            )[1]
                            previous_content = torch.split(
                                previous_state, (self._address_dim, self.d_model), dim=-1
                            )[1]
                            actual_change = state_content.detach() - previous_content.detach()
                        else:
                            actual_change = state.detach() - previous_state.detach()
                        actual_change = actual_change.float()
                        actual_change = actual_change.square().sum(dim=-1).sqrt()
                        changed_count = (trigger & (actual_change > 1e-6)).sum().float()
                        state_delta_sq_sum = (actual_change * trigger).square().sum()
                        eligible_frac = eligible.float().mean()
                        trigger_frac = trigger_count / eligible_count
                        changed_frac = changed_count / trigger_count.clamp_min(1.0)
                        state_delta_rms = (state_delta_sq_sum / trigger_count.clamp_min(1.0)).sqrt()
                        self._last_step_diag.update(
                            {
                                "reconsideration_eligible_frac": eligible_frac,
                                "reconsideration_eligible_count": eligible_count_raw,
                                "reconsideration_trigger_count": trigger_count,
                                "reconsideration_surprise_mean": surprise_mean,
                                "reconsideration_trigger_frac": trigger_frac,
                                "reconsideration_prediction_error_mean": surprise_mean,
                                "reconsideration_changed_frac": changed_frac,
                                "reconsideration_changed_count": changed_count,
                                "reconsideration_state_delta_sq_sum": state_delta_sq_sum,
                                "reconsideration_state_delta_rms": state_delta_rms,
                            }
                        )
                    if self.prediction_error_to_write_gate is not None:
                        error_mean = compiled_prediction_error.detach().float().mean()
                        write_scale_rms = (
                            self.prediction_error_to_write_gate.weight.detach()
                            .float()
                            .square()
                            .mean()
                            .sqrt()
                        )
                        self._last_step_diag.update(
                            {
                                "prediction_error_mean": error_mean,
                                "prediction_error_write_scale_rms": write_scale_rms,
                            }
                        )
                else:
                    self._last_step_diag = {}
            if self.permutation_transactional and pair_swap_steps and step in pair_swap_steps:
                assert pair_swap_entity_ids is not None and pair_swap_mask is not None
                state, committed = self._apply_semantic_pair_swap(
                    state,
                    pair_swap_entity_ids[:, step],
                    pair_swap_mask[:, step],
                )
                if not self.training:
                    self._last_semantic_pair_commits.append((step, committed))
            if collect_kv_records:
                old_content = (
                    torch.split(state_before_step, (self._address_dim, self.d_model), dim=-1)[1]
                    if self.entity_keyed_writes
                    else state_before_step
                )
                new_content = (
                    torch.split(state, (self._address_dim, self.d_model), dim=-1)[1]
                    if self.entity_keyed_writes
                    else state
                )
                change = torch.linalg.vector_norm(
                    (new_content - old_content).float(), dim=-1
                ).detach()
                accepted = change > 1e-6
                weights = change * accepted.to(change.dtype)
                if self.entity_keyed_writes and self.semantic_entity_addressing:
                    record_address, record_content = torch.split(
                        state, (self._address_dim, self.d_model), dim=-1
                    )
                    record_payload = torch.cat(
                        (
                            F.pad(record_address, (0, self.d_model - self._address_dim)),
                            record_content,
                        ),
                        dim=-1,
                    )
                else:
                    record_payload = state if self.entity_keyed_writes else new_content
                record_count = min(2, record_payload.shape[1])
                top_weights, top_slots = weights.topk(record_count, dim=1)
                record = record_payload.gather(
                    1,
                    top_slots.unsqueeze(-1).expand(-1, -1, record_payload.shape[-1]),
                )
                kv_records.append(record.to(dtype=specialist_deltas.dtype))
                kv_commit_masks.append(top_weights > 1e-6)
            if state_observer is not None:
                state_observer(state, call_index=step)
            if (
                self.permutation_transactional
                and permutation_query_steps
                and step in permutation_query_steps
            ):
                assert permutation_query_kind is not None and permutation_query_ids is not None
                position_logits, entity_logits = self._semantic_query_class_logits(
                    state,
                    permutation_query_kind[:, step],
                    permutation_query_ids[:, step],
                )
                self._last_semantic_query_predictions.append(
                    (
                        step,
                        position_logits,
                        entity_logits,
                        permutation_query_kind[:, step],
                        permutation_query_ids[:, step],
                    )
                )
            if (
                state_supervision_steps
                and step in state_supervision_steps
                and state_supervision_entity_ids is not None
                and state_supervision_mask is not None
            ):
                self._last_semantic_state_predictions.append(
                    (
                        step,
                        self._semantic_state_logits(
                            state,
                            state_supervision_entity_ids[:, step],
                            state_supervision_mask[:, step],
                        ),
                    )
                )
            if (
                nth_farthest_vector_steps
                and step in nth_farthest_vector_steps
                and nth_farthest_vector_entity_ids is not None
                and nth_farthest_vector_mask is not None
            ):
                self._last_nth_farthest_vector_predictions.append(
                    (
                        step,
                        self._semantic_vector_logits(
                            state,
                            nth_farthest_vector_entity_ids[:, step],
                            nth_farthest_vector_mask[:, step],
                        ),
                    )
                )
            if nth_farthest_query_steps and step in nth_farthest_query_steps:
                assert nth_farthest_query_candidate_ids is not None
                assert nth_farthest_query_candidate_mask is not None
                assert nth_farthest_query_vectors is not None
                assert nth_farthest_query_ranks is not None
                query_logits, selected_entities, _distances, active_queries = (
                    self._nth_farthest_query_logits(
                        state,
                        nth_farthest_query_candidate_ids[:, step],
                        nth_farthest_query_candidate_mask[:, step],
                        nth_farthest_query_vectors[:, step],
                        nth_farthest_query_ranks[:, step],
                    )
                )
                self._last_nth_farthest_query_predictions.append(
                    (step, query_logits, selected_entities, active_queries)
                )
            if retrieval_value_steps and step in retrieval_value_steps:
                assert retrieval_value_entity_ids is not None and retrieval_value_mask is not None
                value_logits, found = self._semantic_retrieval_value_logits(
                    state,
                    retrieval_value_entity_ids[:, step, None],
                    retrieval_value_mask[:, step, None],
                )
                self._last_retrieval_value_predictions.append(
                    (step, value_logits.squeeze(1), found.squeeze(1))
                )
            if retrieval_query_steps and step in retrieval_query_steps:
                assert retrieval_query_entity_ids is not None
                assert retrieval_query_mask is not None
                query_logits, found = self._semantic_retrieval_value_logits(
                    state,
                    retrieval_query_entity_ids[:, step],
                    retrieval_query_mask[:, step],
                )
                self._last_retrieval_query_predictions.append(
                    (
                        step,
                        query_logits.detach(),
                        found.detach(),
                        retrieval_query_mask[:, step].detach(),
                    )
                )
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
        result = (torch.stack(outputs, dim=1), state)
        if collect_kv_records:
            records = (
                torch.stack(kv_records, dim=1)
                if kv_records
                else specialist_deltas.new_zeros(
                    specialist_deltas.shape[0],
                    0,
                    min(2, self.num_slots),
                    self.d_model * (2 if self.entity_keyed_writes else 1),
                )
            )
            commit_mask = (
                torch.stack(kv_commit_masks, dim=1)
                if kv_commit_masks
                else torch.zeros(
                    specialist_deltas.shape[0],
                    0,
                    min(2, self.num_slots),
                    device=specialist_deltas.device,
                    dtype=torch.bool,
                )
            )
            return (*result, records, commit_mask)
        return result

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
