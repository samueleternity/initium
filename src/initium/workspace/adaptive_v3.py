"""Learned, address-aware workspace operators for adaptive workspace V3."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveWorkspaceOperatorBank(nn.Module):
    """Select, bind, execute, and commit learned workspace operations.

    The controller receives only recurrent features and workspace state. Teacher
    records are consumed exclusively by the auxiliary imitation losses below;
    they are never concatenated into the controller input or used to select an
    operation, operand, or destination at inference time.
    """

    _OPERATOR_COUNT = 6
    _COPY = 1
    _ADD = 2
    _COMPARE = 3
    _SELECT = 4
    _RANK = 5
    _SORT = 6

    def __init__(
        self,
        d_model: int,
        address_dim: int,
        *,
        rank: int = 32,
        usage_floor_weight: float = 0.0,
        readback_consistency_weight: float = 0.0,
        attention_temperature: float = 12.0,
        temperature_enabled: bool = False,
        straight_through_binding: bool = False,
        entity_destinations: bool = False,
        split_context_norm: bool = False,
        use_backbone_context: bool = False,
        teacher_forced_operator_outputs: bool = False,
        class_balanced_operator_loss: bool = False,
    ) -> None:
        super().__init__()
        if d_model < 1 or address_dim < 1 or rank < 1:
            raise ValueError("adaptive operator dimensions must be positive")
        if not math.isfinite(usage_floor_weight) or usage_floor_weight < 0:
            raise ValueError("usage_floor_weight must be finite and non-negative")
        if not math.isfinite(readback_consistency_weight) or readback_consistency_weight < 0:
            raise ValueError("readback_consistency_weight must be finite and non-negative")
        if not math.isfinite(attention_temperature) or attention_temperature <= 0:
            raise ValueError("attention_temperature must be finite and positive")
        self.d_model = int(d_model)
        self.address_dim = int(address_dim)
        self.rank = int(rank)
        self.usage_floor_weight = float(usage_floor_weight)
        self.readback_consistency_weight = float(readback_consistency_weight)
        self.temperature_enabled = bool(temperature_enabled)
        if self.temperature_enabled:
            self.log_inverse_temperature = nn.Parameter(
                torch.tensor(math.log(float(attention_temperature)), dtype=torch.float32)
            )
        else:
            self.register_parameter("log_inverse_temperature", None)
        self.straight_through_binding = bool(straight_through_binding)
        self.entity_destinations = bool(entity_destinations)
        self.split_context_norm = bool(split_context_norm)
        self.use_backbone_context = bool(use_backbone_context)
        self.teacher_forced_operator_outputs = bool(teacher_forced_operator_outputs)
        self.class_balanced_operator_loss = bool(class_balanced_operator_loss)
        self.context_parts = 2 + int(self.use_backbone_context)
        self.context_width = self.context_parts * self.d_model

        self.controller_norm = nn.LayerNorm(self.context_width)
        self.routed_context_norm = nn.LayerNorm(self.d_model) if self.split_context_norm else None
        self.slot_context_norm = nn.LayerNorm(self.d_model) if self.split_context_norm else None
        self.backbone_context_norm = (
            nn.LayerNorm(self.d_model)
            if self.split_context_norm and self.use_backbone_context
            else None
        )
        self.operation_router = nn.Linear(self.context_width, self._OPERATOR_COUNT + 1)
        self.binding_queries = nn.Linear(self.context_width, 3 * self.address_dim, bias=False)
        self.register_buffer("semantic_slot_prototypes", torch.eye(self.address_dim))
        self.operator_down = nn.Parameter(
            torch.empty(self._OPERATOR_COUNT, self.context_width + 2 * self.d_model, rank)
        )
        self.operator_up = nn.Parameter(torch.empty(self._OPERATOR_COUNT, rank, 2 * self.d_model))
        self.operator_bias = nn.Parameter(torch.zeros(self._OPERATOR_COUNT, rank))

        nn.init.xavier_uniform_(self.binding_queries.weight)
        nn.init.xavier_uniform_(self.operator_down)
        nn.init.normal_(self.operator_up, mean=0.0, std=0.01)
        with torch.no_grad():
            self.operation_router.bias.zero_()
            self.operation_router.bias[0] = 3.0

        self.eval_bypass = False
        self._last_aux_loss: torch.Tensor | None = None
        self._last_training_diagnostics: dict[str, torch.Tensor] = {}
        self._last_training_diagnostic_counts: dict[str, torch.Tensor] = {}
        self._last_route_steps: list[tuple[int, torch.Tensor]] = []

    def begin_sequence(self) -> None:
        self._last_aux_loss = None
        self._last_training_diagnostics = {}
        self._last_training_diagnostic_counts = {}
        self._last_route_steps = []

    def last_route_steps(self) -> tuple[tuple[int, torch.Tensor], ...]:
        return tuple(self._last_route_steps)

    def pop_aux_loss(self) -> torch.Tensor | None:
        loss = self._last_aux_loss
        self._last_aux_loss = None
        return loss

    def last_diagnostics(self) -> dict[str, torch.Tensor]:
        return {
            **self._last_training_diagnostics,
            **{
                name: value
                for name, value in self._last_training_diagnostic_counts.items()
                if not name.endswith("_samples")
            },
        }

    def _accumulate_diagnostics(self, diagnostics: dict[str, torch.Tensor]) -> None:
        for name, value in diagnostics.items():
            if name.endswith("_count"):
                previous = self._last_training_diagnostic_counts.get(name)
                self._last_training_diagnostic_counts[name] = (
                    value.detach() if previous is None else previous + value.detach()
                )
                continue
            count_name = f"{name}_count"
            count = diagnostics.get(count_name)
            if count is None:
                count = value.new_ones(())
            count = count.detach().float()
            previous_count = self._last_training_diagnostic_counts.get(
                f"{name}_samples", value.new_zeros(())
            )
            previous_value = self._last_training_diagnostics.get(name)
            total_count = previous_count + count
            total = value.detach().float() * count
            if previous_value is not None:
                total = total + previous_value * previous_count
            self._last_training_diagnostics[name] = total / total_count.clamp_min(1.0)
            self._last_training_diagnostic_counts[f"{name}_samples"] = total_count

    def _temperature(self) -> torch.Tensor | float:
        if self.log_inverse_temperature is None:
            return 1.0
        return self.log_inverse_temperature

    @staticmethod
    def _slot_attention(
        query: torch.Tensor,
        address_state: torch.Tensor,
        occupied: torch.Tensor,
        *,
        temperature: torch.Tensor | float = 1.0,
        straight_through: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.einsum(
            "bd,bsd->bs", F.normalize(query.float(), dim=-1), F.normalize(address_state.float(), dim=-1)
        )
        scores = scores.masked_fill(~occupied, -1e4)
        scale = temperature.exp().clamp(1.0, 32.0) if isinstance(temperature, torch.Tensor) else float(temperature)
        weights = torch.softmax(scores * scale, dim=-1) * occupied.to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if straight_through:
            hard = F.one_hot(weights.argmax(dim=-1), num_classes=weights.shape[-1]).to(weights.dtype)
            hard = hard * occupied.to(hard.dtype)
            weights = hard + weights - weights.detach()
        return scores, weights

    @staticmethod
    def _entity_destination_weights(
        entity_logits: torch.Tensor,
        address_state: torch.Tensor,
        occupied: torch.Tensor,
        *,
        temperature: torch.Tensor | float,
        straight_through: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Choose a semantic entity then resolve it with first-free allocation.

        Returns slot weights, entity IDs, selected slots, and an allocation mask.
        Existing entity addresses are never rewritten; an address is installed
        only when the entity is assigned to a previously empty slot.
        """
        batch, slots, address_dim = address_state.shape
        if entity_logits.shape != (batch, address_dim):
            raise ValueError("entity destination logits do not match the address vocabulary")
        entity_ids_by_slot = address_state.float().argmax(dim=-1)
        entity_to_slot = F.one_hot(
            entity_ids_by_slot, num_classes=address_dim
        ).permute(0, 2, 1).bool() & occupied.unsqueeze(1)
        entity_found = entity_to_slot.any(dim=-1)
        free = ~occupied
        has_free = free.any(dim=-1)
        first_free = free.float().argmax(dim=-1)
        fallback_slot = first_free.unsqueeze(-1).expand(-1, address_dim)
        fallback_map = F.one_hot(fallback_slot, num_classes=slots).bool()
        fallback_map = fallback_map & has_free[:, None, None]
        entity_map = torch.where(entity_found.unsqueeze(-1), entity_to_slot, fallback_map)

        scale = temperature.exp().clamp(1.0, 32.0) if isinstance(temperature, torch.Tensor) else float(temperature)
        probabilities = torch.softmax(entity_logits.float() * scale, dim=-1)
        soft_slots = torch.einsum("ba,bas->bs", probabilities, entity_map.float())
        predicted_entity = entity_logits.argmax(dim=-1)
        matched = entity_found.gather(1, predicted_entity.unsqueeze(-1)).squeeze(-1)
        selected_slot = torch.where(
            matched,
            entity_to_slot.float().argmax(dim=-1).gather(1, predicted_entity.unsqueeze(-1)).squeeze(-1),
            first_free,
        )
        valid = matched | has_free
        hard_slots = F.one_hot(selected_slot, num_classes=slots).to(soft_slots.dtype)
        hard_slots = hard_slots * valid.unsqueeze(-1).to(hard_slots.dtype)
        if straight_through:
            slot_weights = hard_slots + soft_slots - soft_slots.detach()
        else:
            slot_weights = hard_slots
        allocate = valid & ~matched
        return slot_weights, predicted_entity, selected_slot, allocate

    @staticmethod
    def _target_slot_indices(
        address_state: torch.Tensor,
        occupied: torch.Tensor,
        entity_ids: torch.Tensor,
        *,
        allow_empty_fallback: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        address_ids = address_state.float().argmax(dim=-1)
        matches = occupied & address_ids.eq(entity_ids.unsqueeze(-1))
        found = matches.any(dim=-1)
        selected = matches.float().argmax(dim=-1)
        # Semantic addresses have a fixed entity-to-slot mapping even before a
        # write has occupied the slot. This lets the learned destination head
        # acquire a new entity slot rather than being restricted to old values.
        fallback = entity_ids.clamp(min=0, max=address_state.shape[1] - 1)
        if allow_empty_fallback:
            selected = torch.where(found, selected, fallback)
            found = found | (entity_ids.ge(0) & entity_ids.lt(address_state.shape[1]))
        return selected, found

    def _teacher_losses(
        self,
        *,
        operation_logits: torch.Tensor,
        left_scores: torch.Tensor,
        right_scores: torch.Tensor,
        destination0_scores: torch.Tensor,
        destination1_scores: torch.Tensor,
        output0_logits: torch.Tensor,
        teacher_forced_output0_logits: torch.Tensor | None,
        output1_logits: torch.Tensor,
        operand0_logits: torch.Tensor,
        operand1_logits: torch.Tensor,
        address_state: torch.Tensor,
        occupied: torch.Tensor,
        teacher_targets: torch.Tensor | None,
        teacher_weight: float,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        if teacher_targets is None or teacher_weight <= 0:
            return None, {}
        if teacher_targets.ndim != 2 or teacher_targets.shape[-1] != 12:
            raise ValueError("adaptive V3 teacher targets must have shape (batch, 12)")

        op_code = teacher_targets[:, 0].long()
        valid = op_code.ge(0)
        feature_off = valid & teacher_targets[:, 11].eq(1)
        op_class = torch.zeros_like(op_code)
        op_class[op_code.eq(1) | op_code.eq(2)] = self._COPY
        op_class[op_code.eq(3)] = self._ADD
        op_class[op_code.eq(5)] = self._COMPARE
        op_class[op_code.eq(6)] = self._SELECT
        op_class[op_code.eq(7)] = self._RANK
        op_class[op_code.eq(4)] = self._SORT

        per_token_op = F.cross_entropy(
            operation_logits.float(),
            op_class.clamp(min=0, max=self._OPERATOR_COUNT),
            reduction="none",
        )
        positive = valid & op_code.gt(0)
        negative = valid & op_code.eq(0)
        if self.class_balanced_operator_loss:
            class_losses = []
            class_present = []
            for class_index in range(1, self._OPERATOR_COUNT + 1):
                class_mask = positive & op_class.eq(class_index)
                count = class_mask.sum()
                class_present.append(count.gt(0).to(per_token_op.dtype))
                class_losses.append((per_token_op * class_mask).sum() / count.clamp_min(1))
            present = torch.stack(class_present)
            active_loss = (torch.stack(class_losses) * present).sum() / present.sum().clamp_min(1)
            op_loss = 0.5 * active_loss + 0.5 * (
                (per_token_op * negative).sum() / negative.sum().clamp_min(1)
            )
        else:
            op_loss = 0.5 * (per_token_op * positive).sum() / positive.sum().clamp_min(1)
            op_loss = op_loss + 0.5 * (per_token_op * negative).sum() / negative.sum().clamp_min(1)

        losses = [op_loss]
        diagnostics: dict[str, torch.Tensor] = {
            "teacher_operation_loss": op_loss.detach(),
            "teacher_operation_loss_count": valid.sum().detach(),
            "teacher_operation_accuracy": (
                operation_logits.argmax(dim=-1).eq(op_class) & valid
            ).sum().float().div(valid.sum().clamp_min(1)).detach(),
            "teacher_operation_accuracy_count": valid.sum().detach(),
            "teacher_active_operation_accuracy": (
                operation_logits.argmax(dim=-1).eq(op_class) & positive
            ).sum().float().div(positive.sum().clamp_min(1)).detach(),
            "teacher_active_operation_accuracy_count": positive.sum().detach(),
            "teacher_noop_accuracy": (
                operation_logits.argmax(dim=-1).eq(0) & negative
            ).sum().float().div(negative.sum().clamp_min(1)).detach(),
            "teacher_noop_accuracy_count": negative.sum().detach(),
            "teacher_feature_off_operation_accuracy": (
                operation_logits.argmax(dim=-1).eq(op_class) & feature_off
            ).sum().float().div(feature_off.sum().clamp_min(1)).detach(),
            "teacher_feature_off_operation_accuracy_count": feature_off.sum().detach(),
            "teacher_feature_off_active_operation_accuracy": (
                operation_logits.argmax(dim=-1).eq(op_class) & positive & feature_off
            ).sum().float().div((positive & feature_off).sum().clamp_min(1)).detach(),
            "teacher_feature_off_active_operation_accuracy_count": (
                positive & feature_off
            ).sum().detach(),
        }
        operator_names = ("noop", "copy", "add_mod", "compare", "select", "rank", "sort")
        diagnostics["teacher_events_noop_count"] = negative.sum().detach()
        diagnostics["teacher_op_loss_noop"] = (
            (per_token_op * negative).sum() / negative.sum().clamp_min(1)
        ).detach()
        diagnostics["teacher_op_loss_noop_count"] = negative.sum().detach()
        for class_index, name in enumerate(operator_names[1:], start=1):
            operator_valid = positive & op_class.eq(class_index)
            diagnostics[f"teacher_events_{name}_count"] = operator_valid.sum().detach()
            diagnostics[f"teacher_op_loss_{name}"] = (
                (per_token_op * operator_valid).sum()
                / operator_valid.sum().clamp_min(1)
            ).detach()
            diagnostics[f"teacher_op_loss_{name}_count"] = operator_valid.sum().detach()
        for column, value_logits in ((7, operand0_logits), (8, operand1_logits)):
            target = teacher_targets[:, column].long()
            value_valid = positive & target.ge(0)
            safe_target = target.masked_fill(~value_valid, -100)
            per_item = F.cross_entropy(
                value_logits.float(), safe_target, reduction="none", ignore_index=-100
            )
            value_loss = per_item.sum() / value_valid.sum().clamp_min(1)
            losses.append(value_loss)
            diagnostics[f"teacher_operand{column - 7}_loss"] = value_loss.detach()
            diagnostics[f"teacher_operand{column - 7}_loss_count"] = value_valid.sum().detach()
            diagnostics[f"teacher_operand{column - 7}_accuracy"] = (
                value_logits.argmax(dim=-1).eq(target) & value_valid
            ).sum().float().div(value_valid.sum().clamp_min(1)).detach()
            diagnostics[f"teacher_operand{column - 7}_accuracy_count"] = value_valid.sum().detach()
            feature_off_value_valid = value_valid & feature_off
            diagnostics[f"teacher_feature_off_operand{column - 7}_accuracy"] = (
                value_logits.argmax(dim=-1).eq(target) & feature_off_value_valid
            ).sum().float().div(feature_off_value_valid.sum().clamp_min(1)).detach()
            diagnostics[
                f"teacher_feature_off_operand{column - 7}_accuracy_count"
            ] = feature_off_value_valid.sum().detach()
        output0_target = teacher_targets[:, 9].long()
        valid_output0 = positive & output0_target.ge(0)
        output0_safe_target = output0_target.masked_fill(~valid_output0, -100)
        supervised_output0_logits = (
            teacher_forced_output0_logits
            if self.teacher_forced_operator_outputs and teacher_forced_output0_logits is not None
            else output0_logits
        )
        output0_per_item = F.cross_entropy(
            supervised_output0_logits.float(),
            output0_safe_target,
            reduction="none",
            ignore_index=-100,
        )
        output0_loss = output0_per_item.sum() / valid_output0.sum().clamp_min(1)
        losses.append(output0_loss)
        diagnostics["teacher_output_loss"] = output0_loss.detach()
        diagnostics["teacher_output_loss_count"] = valid_output0.sum().detach()
        diagnostics["teacher_output_accuracy"] = (
            supervised_output0_logits.argmax(dim=-1).eq(output0_target) & valid_output0
        ).sum().float().div(valid_output0.sum().clamp_min(1)).detach()
        diagnostics["teacher_output_accuracy_count"] = valid_output0.sum().detach()
        feature_off_output0 = valid_output0 & feature_off
        diagnostics["teacher_feature_off_output_accuracy"] = (
            supervised_output0_logits.argmax(dim=-1).eq(output0_target) & feature_off_output0
        ).sum().float().div(feature_off_output0.sum().clamp_min(1)).detach()
        diagnostics["teacher_feature_off_output_accuracy_count"] = feature_off_output0.sum().detach()
        if self.teacher_forced_operator_outputs and teacher_forced_output0_logits is not None:
            for class_index, name in enumerate(
                ("copy", "add_mod", "compare", "select", "rank", "sort"), start=1
            ):
                operator_output_valid = valid_output0 & op_class.eq(class_index)
                diagnostics[f"teacher_operator_output_loss_{name}"] = (
                    (output0_per_item * operator_output_valid).sum()
                    / operator_output_valid.sum().clamp_min(1)
                ).detach()
                diagnostics[f"teacher_operator_output_loss_{name}_count"] = (
                    operator_output_valid.sum().detach()
                )
                diagnostics[f"teacher_operator_output_accuracy_{name}"] = (
                    (
                        teacher_forced_output0_logits.argmax(dim=-1).eq(output0_target)
                        & operator_output_valid
                    ).sum().float()
                    / operator_output_valid.sum().clamp_min(1)
                ).detach()
                diagnostics[f"teacher_operator_output_accuracy_{name}_count"] = (
                    operator_output_valid.sum().detach()
                )
        output1_target = teacher_targets[:, 10].long()
        valid_output1 = positive & output1_target.ge(0)
        output1_safe_target = output1_target.masked_fill(~valid_output1, -100)
        output1_per_item = F.cross_entropy(
            output1_logits.float(), output1_safe_target, reduction="none", ignore_index=-100
        )
        output1_loss = output1_per_item.sum() / valid_output1.sum().clamp_min(1)
        losses.append(output1_loss)
        diagnostics["teacher_output1_loss"] = output1_loss.detach()
        diagnostics["teacher_output1_loss_count"] = valid_output1.sum().detach()
        diagnostics["teacher_output1_accuracy"] = (
            output1_logits.argmax(dim=-1).eq(output1_target) & valid_output1
        ).sum().float().div(valid_output1.sum().clamp_min(1)).detach()
        diagnostics["teacher_output1_accuracy_count"] = valid_output1.sum().detach()
        feature_off_output1 = valid_output1 & feature_off
        diagnostics["teacher_feature_off_output1_accuracy"] = (
            output1_logits.argmax(dim=-1).eq(output1_target) & feature_off_output1
        ).sum().float().div(feature_off_output1.sum().clamp_min(1)).detach()
        diagnostics["teacher_feature_off_output1_accuracy_count"] = feature_off_output1.sum().detach()

        for column, scores in (
            (4, left_scores),
            (6, right_scores),
            (1, destination0_scores),
            (2, destination1_scores),
        ):
            entity_ids = teacher_targets[:, column].long()
            kind_valid = (
                teacher_targets[:, 3].eq(1)
                if column == 4
                else teacher_targets[:, 5].eq(1)
                if column == 6
                else entity_ids.ge(0)
            )
            selected = positive & kind_valid & entity_ids.ge(0)
            if self.entity_destinations and column in (1, 2):
                target_slots = entity_ids.clamp(min=0, max=self.address_dim - 1)
                found = entity_ids.ge(0) & entity_ids.lt(self.address_dim)
            else:
                target_slots, found = self._target_slot_indices(
                    address_state,
                    occupied,
                    entity_ids.clamp(min=0, max=self.address_dim - 1),
                    allow_empty_fallback=column in (1, 2),
                )
            binding_valid = selected & found
            safe_target_slots = target_slots.masked_fill(~binding_valid, -100)
            per_item_binding_loss = F.cross_entropy(
                scores.float() * (
                    self.log_inverse_temperature.exp().clamp(1.0, 32.0)
                    if self.log_inverse_temperature is not None else 1.0
                ),
                safe_target_slots,
                reduction="none",
                ignore_index=-100,
            )
            binding_loss = per_item_binding_loss.sum() / binding_valid.sum().clamp_min(1)
            losses.append(binding_loss)
            binding_name = {
                4: "operand0",
                6: "operand1",
                1: "destination0",
                2: "destination1",
            }[column]
            key = f"teacher_{binding_name}_binding"
            diagnostics[f"{key}_loss"] = binding_loss.detach()
            diagnostics[f"{key}_loss_count"] = binding_valid.sum().detach()
            diagnostics[f"{key}_accuracy"] = (
                scores.argmax(dim=-1).eq(target_slots) & binding_valid
            ).sum().float().div(binding_valid.sum().clamp_min(1)).detach()
            diagnostics[f"{key}_accuracy_count"] = binding_valid.sum().detach()
            feature_off_binding_valid = binding_valid & feature_off
            feature_off_key = f"teacher_feature_off_{binding_name}_binding"
            diagnostics[f"{feature_off_key}_accuracy"] = (
                scores.argmax(dim=-1).eq(target_slots) & feature_off_binding_valid
            ).sum().float().div(feature_off_binding_valid.sum().clamp_min(1)).detach()
            diagnostics[f"{feature_off_key}_accuracy_count"] = feature_off_binding_valid.sum().detach()

        teacher_loss = torch.stack([loss.float() for loss in losses]).mean()
        return teacher_loss * float(teacher_weight), diagnostics

    def step(
        self,
        *,
        state: torch.Tensor,
        context: torch.Tensor,
        backbone_context: torch.Tensor | None,
        semantic_value_decoder: nn.Linear,
        active_mask: torch.Tensor,
        execution_mask: torch.Tensor,
        teacher_targets: torch.Tensor | None,
        teacher_weight: float,
        step_index: int,
    ) -> torch.Tensor:
        """Run one candidate event; selector inputs exclude teacher metadata."""
        if self.eval_bypass:
            return state
        if (
            state.ndim != 3
            or context.ndim != 2
            or context.shape != (state.shape[0], 2 * self.d_model)
            or (self.use_backbone_context and (
                backbone_context is None or backbone_context.shape != (state.shape[0], self.d_model)
            ))
            or semantic_value_decoder.in_features != self.d_model
            or state.shape[-1] != self.address_dim + self.d_model
            or active_mask.shape != (state.shape[0],)
            or execution_mask.shape != (state.shape[0],)
        ):
            raise ValueError("adaptive V3 operator inputs have incompatible shapes")
        address_state, content_state = torch.split(
            state, [self.address_dim, self.d_model], dim=-1
        )
        occupied = address_state.float().square().sum(dim=-1).gt(1e-12)
        context_parts = list(torch.split(context.float(), self.d_model, dim=-1))
        if self.use_backbone_context:
            assert backbone_context is not None
            context_parts.append(backbone_context.float())
        if self.split_context_norm:
            assert self.routed_context_norm is not None and self.slot_context_norm is not None
            normalized_parts = [
                self.routed_context_norm(context_parts[0]),
                self.slot_context_norm(context_parts[1]),
            ]
            if self.use_backbone_context:
                assert self.backbone_context_norm is not None
                normalized_parts.append(self.backbone_context_norm(context_parts[2]))
            normalized_context = torch.cat(normalized_parts, dim=-1)
        else:
            normalized_context = self.controller_norm(torch.cat(context_parts, dim=-1))
        operation_logits = self.operation_router(normalized_context).float()
        operation_probabilities = operation_logits.softmax(dim=-1)
        binding_queries = self.binding_queries(normalized_context).view(
            state.shape[0], 3, self.address_dim
        )
        left_scores, left_weights = self._slot_attention(
            binding_queries[:, 0], address_state, occupied,
            temperature=self._temperature(),
            straight_through=self.straight_through_binding,
        )
        right_scores, right_weights = self._slot_attention(
            binding_queries[:, 1], address_state, occupied,
            temperature=self._temperature(),
            straight_through=self.straight_through_binding,
        )
        destination_slots = self.semantic_slot_prototypes.to(
            device=state.device, dtype=address_state.dtype
        )
        if state.shape[1] != self.address_dim:
            destination_slots = destination_slots[
                torch.arange(state.shape[1], device=state.device).remainder(self.address_dim)
            ]
        destination_slots = destination_slots.unsqueeze(0).expand(state.shape[0], -1, -1)
        destination0_allocate = destination1_allocate = None
        destination0_slot = None
        if self.entity_destinations:
            normalized_prototypes = F.normalize(
                self.semantic_slot_prototypes.to(device=state.device, dtype=torch.float32),
                dim=-1,
            )
            destination0_scores = torch.einsum(
                "bd,sd->bs", F.normalize(binding_queries[:, 2].float(), dim=-1),
                normalized_prototypes,
            )
            destination0_weights, destination0_entity, destination0_slot, destination0_allocate = (
                self._entity_destination_weights(
                    destination0_scores, address_state, occupied,
                    temperature=self._temperature(),
                    straight_through=self.straight_through_binding,
                )
            )
        else:
            destination0_scores, destination0_weights = self._slot_attention(
                binding_queries[:, 2], destination_slots, torch.ones_like(occupied),
                temperature=self._temperature(),
                straight_through=self.straight_through_binding,
            )
            destination0_entity = destination0_scores.argmax(dim=-1)
        # The second result of the sort operator uses an independent learned
        # binding query derived from the same local state, with a disjoint
        # projection slice so the two write heads can learn distinct addresses.
        destination1_query = binding_queries[:, 2] + binding_queries[:, 0]
        if self.entity_destinations:
            destination1_address_state = address_state
            destination1_occupied = occupied
            assert destination0_slot is not None and destination0_allocate is not None
            destination0_new_mask_for_resolve = (
                destination0_allocate & active_mask.bool() & execution_mask.bool()
            )
            destination0_new_addresses = F.one_hot(
                destination0_slot, num_classes=state.shape[1]
            ).to(address_state.dtype).unsqueeze(-1) * F.one_hot(
                destination0_entity, num_classes=self.address_dim
            ).to(address_state.dtype).unsqueeze(1)
            destination1_address_state = torch.where(
                destination0_new_mask_for_resolve[:, None, None],
                destination0_new_addresses,
                address_state,
            )
            destination1_occupied = destination1_address_state.float().square().sum(dim=-1).gt(1e-12)
            destination1_scores = torch.einsum(
                "bd,sd->bs", F.normalize(destination1_query.float(), dim=-1),
                normalized_prototypes,
            )
            destination1_weights, destination1_entity, _destination1_slot, destination1_allocate = (
                self._entity_destination_weights(
                    destination1_scores, destination1_address_state, destination1_occupied,
                    temperature=self._temperature(),
                    straight_through=self.straight_through_binding,
                )
            )
        else:
            destination1_scores, destination1_weights = self._slot_attention(
                destination1_query, destination_slots, torch.ones_like(occupied),
                temperature=self._temperature(),
                straight_through=self.straight_through_binding,
            )
            destination1_entity = destination1_scores.argmax(dim=-1)
        left_value = torch.einsum("bs,bsd->bd", left_weights, content_state.float())
        right_value = torch.einsum("bs,bsd->bd", right_weights, content_state.float())
        operator_input = torch.cat((normalized_context, left_value, right_value), dim=-1)

        operator_hidden = torch.einsum(
            "bi,kir->bkr", operator_input, self.operator_down
        )
        operator_hidden = torch.tanh(operator_hidden + self.operator_bias.unsqueeze(0))
        raw_outputs = torch.einsum(
            "bkr,krd->bkd", operator_hidden, self.operator_up
        ).view(state.shape[0], self._OPERATOR_COUNT, 2, self.d_model)
        active_probabilities = operation_probabilities[:, 1:].unsqueeze(-1).unsqueeze(-1)
        mixed_outputs = (raw_outputs * active_probabilities).sum(dim=1)
        output0, output1 = mixed_outputs.unbind(dim=1)

        teacher_active = teacher_targets is not None and teacher_weight > 0
        output0_logits = (
            semantic_value_decoder(output0)
            if teacher_active or self.readback_consistency_weight > 0
            else None
        )
        output1_logits = semantic_value_decoder(output1) if teacher_active else None
        teacher_loss = None
        teacher_diagnostics = {}
        if teacher_active:
            assert output0_logits is not None and output1_logits is not None
            teacher_forced_output0_logits = None
            if self.teacher_forced_operator_outputs:
                teacher_op_class = teacher_targets[:, 0].long()
                teacher_operator_index = torch.where(
                    teacher_op_class.eq(1) | teacher_op_class.eq(2), self._COPY,
                    torch.where(
                        teacher_op_class.eq(3), self._ADD,
                        torch.where(
                            teacher_op_class.eq(5), self._COMPARE,
                            torch.where(
                                teacher_op_class.eq(6), self._SELECT,
                                torch.where(teacher_op_class.eq(7), self._RANK, self._SORT),
                            ),
                        ),
                    ),
                ) - 1
                selected_operator_output = raw_outputs[
                    torch.arange(state.shape[0], device=state.device),
                    teacher_operator_index.clamp(min=0, max=self._OPERATOR_COUNT - 1),
                    0,
                ]
                teacher_forced_output0_logits = semantic_value_decoder(selected_operator_output)
            operand0_logits = semantic_value_decoder(left_value)
            operand1_logits = semantic_value_decoder(right_value)
            teacher_loss, teacher_diagnostics = self._teacher_losses(
                operation_logits=operation_logits,
                left_scores=left_scores,
                right_scores=right_scores,
                destination0_scores=destination0_scores,
                destination1_scores=destination1_scores,
                output0_logits=output0_logits,
                teacher_forced_output0_logits=teacher_forced_output0_logits,
                output1_logits=output1_logits,
                operand0_logits=operand0_logits,
                operand1_logits=operand1_logits,
                address_state=address_state,
                occupied=occupied,
                teacher_targets=teacher_targets,
                teacher_weight=teacher_weight,
            )
        if teacher_loss is not None:
            self._last_aux_loss = (
                teacher_loss if self._last_aux_loss is None else self._last_aux_loss + teacher_loss
            )
            if self.training:
                self._accumulate_diagnostics(teacher_diagnostics)

        if self.usage_floor_weight > 0 and self.training:
            candidate = active_mask.to(operation_probabilities.dtype)
            marginal = (
                operation_probabilities[:, 1:] * candidate.unsqueeze(-1)
            ).sum(dim=0) / candidate.sum().clamp_min(1)
            floor = 0.05 / self._OPERATOR_COUNT
            usage_loss = F.relu(floor - marginal).mean() * self.usage_floor_weight
            self._last_aux_loss = (
                usage_loss if self._last_aux_loss is None else self._last_aux_loss + usage_loss
            )
            self._accumulate_diagnostics(
                {"operator_usage_floor_loss": usage_loss.detach()}
            )

        if self.training:
            previous = self._last_training_diagnostics.get("operation_probability")
            current = operation_probabilities.detach().mean(dim=0)
            self._last_training_diagnostics["operation_probability"] = (
                current if previous is None else 0.5 * (previous + current)
            )
            self._accumulate_diagnostics(
                {
                    "controller_routed_context_rms": context_parts[0].detach().float().square().mean().sqrt(),
                    "controller_slot_context_rms": context_parts[1].detach().float().square().mean().sqrt(),
                    **(
                        {"controller_backbone_context_rms": context_parts[2].detach().float().square().mean().sqrt()}
                        if self.use_backbone_context
                        else {}
                    ),
                    "binding_inverse_temperature": (
                        self.log_inverse_temperature.detach().exp().clamp(1.0, 32.0)
                        if self.log_inverse_temperature is not None
                        else context.new_ones(())
                    ),
                }
            )
        if not self.training:
            self._last_route_steps.append(
                (int(step_index), operation_probabilities.argmax(dim=-1).detach())
            )

        active = active_mask.bool() & execution_mask.bool()
        active_weight = (1.0 - operation_probabilities[:, 0]) * active.to(
            operation_probabilities.dtype
        )
        sort_weight = operation_probabilities[:, self._SORT] * active.to(
            operation_probabilities.dtype
        )
        destination0_mass = active_weight.unsqueeze(-1) * destination0_weights
        destination1_mass = sort_weight.unsqueeze(-1) * destination1_weights
        content = content_state.float()
        content = content * (1.0 - destination0_mass.unsqueeze(-1)) + (
            output0.unsqueeze(1) * destination0_mass.unsqueeze(-1)
        )
        content = content * (1.0 - destination1_mass.unsqueeze(-1)) + (
            output1.unsqueeze(1) * destination1_mass.unsqueeze(-1)
        )
        addresses = destination_slots
        if self.entity_destinations:
            address = address_state.float()
        else:
            address = address_state.float() * (1.0 - destination0_mass.unsqueeze(-1)) + (
                addresses * destination0_mass.unsqueeze(-1)
            )
            address = address * (1.0 - destination1_mass.unsqueeze(-1)) + (
                addresses * destination1_mass.unsqueeze(-1)
            )
        if self.entity_destinations:
            assert destination0_allocate is not None and destination1_allocate is not None
            destination0_new = F.one_hot(
                destination0_entity, num_classes=self.address_dim
            ).to(address.dtype)
            destination1_new = F.one_hot(
                destination1_entity, num_classes=self.address_dim
            ).to(address.dtype)
            destination0_new_mask = (
                destination0_allocate & active
            ).unsqueeze(-1).unsqueeze(-1) & destination0_weights.unsqueeze(-1).gt(0)
            destination1_new_mask = (
                destination1_allocate & active & (operation_probabilities[:, self._SORT].gt(0))
            ).unsqueeze(-1).unsqueeze(-1) & destination1_weights.unsqueeze(-1).gt(0)
            address = torch.where(
                destination0_new_mask,
                destination0_new.unsqueeze(1),
                address,
            )
            address = torch.where(
                destination1_new_mask,
                destination1_new.unsqueeze(1),
                address,
            )
        updated_state = torch.cat((address, content), dim=-1)

        if self.readback_consistency_weight > 0:
            assert output0_logits is not None
            readback = torch.einsum("bs,bsd->bd", destination0_weights, content)
            predicted = output0_logits.float().softmax(dim=-1).detach()
            readback_logits = semantic_value_decoder(readback).float()
            readback_probabilities = readback_logits.softmax(dim=-1)
            mixture = 0.5 * (predicted + readback_probabilities)
            js = 0.5 * (
                predicted
                * (predicted.clamp_min(1e-8).log() - mixture.clamp_min(1e-8).log())
            ).sum(dim=-1) + 0.5 * (
                readback_probabilities
                * (
                    readback_probabilities.clamp_min(1e-8).log()
                    - mixture.clamp_min(1e-8).log()
                )
            ).sum(dim=-1)
            consistency = (
                (js * active.to(js.dtype)).sum() / active.sum().clamp_min(1)
            ) * self.readback_consistency_weight
            self._last_aux_loss = (
                consistency
                if self._last_aux_loss is None
                else self._last_aux_loss + consistency
            )
            if self.training:
                self._accumulate_diagnostics(
                    {"readback_consistency_loss": consistency.detach()}
                )
        return updated_state.to(dtype=state.dtype)
