"""Low-synchronization activation and gradient summaries for model segments.

Hooks are selected by module type rather than a particular model topology,
so the same collector covers split-graph, standalone Mamba/CfC, chained
Mamba+CfC, recurrent DNC, and workspace models.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, tuple | list):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _all_tensors(value: Any) -> list[torch.Tensor]:
    """Return distinct tensor leaves in output order for structured modules."""
    tensors: list[torch.Tensor] = []
    seen: set[int] = set()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            identity = id(item)
            if identity not in seen:
                seen.add(identity)
                tensors.append(item)
        elif isinstance(item, tuple | list):
            for child in item:
                visit(child)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)

    visit(value)
    return tensors


def _stable_sum_squares(value: torch.Tensor) -> torch.Tensor:
    """Sum squares without overflowing the input dtype's accumulator."""
    if not value.numel():
        return torch.zeros((), device=value.device, dtype=torch.float64)
    scale = value.abs().amax().clamp_min(1.0)
    scaled_sum = (value / scale).square().sum()
    result = scaled_sum.to(torch.float64) * scale.to(torch.float64).square()
    return torch.where(
        torch.isfinite(scale),
        result,
        torch.full_like(result, float("inf")),
    )


def _segment_label(path: str, module: nn.Module) -> str | None:
    class_name = type(module).__name__
    lower_name = class_name.lower()
    leaf = path or "model"
    if class_name in {"MambaBackboneParallel", "CfCBackboneParallel"}:
        return f"backbone.{class_name.removesuffix('BackboneParallel').lower()}:{leaf}"
    if class_name in {"_ParallelBlock", "_CfCParallelBlock"}:
        kind = "mamba" if class_name == "_ParallelBlock" else "cfc"
        return f"backbone.{kind}_block:{leaf}"
    if class_name in {"MoEBlock", "MultiSourceMoEBlock"}:
        return f"moe_residual:{leaf}"
    if class_name == "SplitGraphDNC":
        return f"split_graph:{leaf}"
    if class_name == "ChainedControllerWrapper":
        return f"controller_chain:{leaf}"
    if class_name == "CfCControllerWrapper":
        return f"controller.cfc:{leaf}"
    if class_name == "CfCControllerBlock":
        return f"controller.cfc_block:{leaf}"
    if class_name in {
        "MambaControllerWrapper",
        "Mamba2ControllerWrapper",
        "Mamba3ControllerWrapper",
    }:
        return f"controller.{class_name.removesuffix('ControllerWrapper').lower()}:{leaf}"
    if class_name in {
        "MambaControllerBlock",
        "Mamba2ControllerBlock",
        "Mamba3ControllerBlock",
    }:
        return f"controller.mamba_block:{leaf}"
    if class_name in {"MambaControllerCell", "Mamba2ControllerCell", "Mamba3ControllerCell"}:
        return f"controller.mamba_cell:{leaf}"
    if class_name == "WorkspaceBroadcast":
        return f"workspace:{leaf}"
    if class_name == "BistableWriteGate":
        return f"workspace_bistable_gate:{leaf}"
    if class_name == "Memory" or "memory" in path.lower() and "memory" in lower_name:
        return f"dnc_memory:{leaf}"
    if isinstance(module, nn.LSTM):
        return f"controller.lstm:{leaf}"
    if isinstance(module, nn.GRU):
        return f"controller.gru:{leaf}"
    if isinstance(module, nn.RNN):
        return f"controller.rnn:{leaf}"
    return None


class SegmentDiagnostics:
    """Aggregate segment input/output health and hierarchical gradient norms.

    Scalar reductions remain detached on their source device. The only
    device-to-host transfer occurs when ``emit`` is called at a log/eval
    boundary or when a numerical failure needs an immediate trace.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        enabled: bool = True,
        activation_absmax_limit: float | None = None,
    ):
        self.model = model
        self.enabled = bool(enabled)
        self.activation_absmax_limit = activation_absmax_limit
        self.channel: str | None = "train"
        self._current_step: int | None = None
        self._step_call_counts: dict[str, int] = {}
        self._records: dict[str, OrderedDict[str, dict[str, Any]]] = {}
        self._segments: list[tuple[str, str]] = []
        self._external_modules: list[tuple[str, nn.Module]] = []
        self._handles: list[RemovableHandle] = []
        self._nonfinite_trace_enabled = False
        self._nonfinite_trace_reported = False
        if not self.enabled:
            return
        for path, module in model.named_modules():
            label = _segment_label(path, module)
            if label is not None:
                self._register(module, path, label)

    def _register(self, module: nn.Module, path: str, label: str) -> None:
        self._segments.append((path, label))

        def hook(_module, inputs, output):
            if self.channel is None:
                return
            channel = self.channel
            input_tensor = _first_tensor(inputs)
            tracks_secondary_outputs = label.startswith(
                ("workspace:", "workspace_bistable_gate:", "controller.cfc_block:")
            )
            tracks_cfc_state = label.startswith("controller.cfc_block:")
            output_tensors = (
                _all_tensors(output)
                if tracks_secondary_outputs
                else [tensor for tensor in (_first_tensor(output),) if tensor is not None]
            )
            output_tensor = output_tensors[0] if output_tensors else None
            record = self._records.setdefault(channel, OrderedDict()).setdefault(
                label, {"path": path, "calls": 0}
            )
            record["calls"] += 1
            call_index = self._step_call_counts.get(label, 0)
            self._step_call_counts[label] = call_index + 1
            location = (
                (self._current_step, call_index)
                if self._current_step is not None and channel == "train"
                else None
            )
            input_context = None
            output_context = None
            if (
                location is not None
                and label.startswith("workspace_bistable_gate:")
                and input_tensor is not None
                and input_tensor.requires_grad
            ):
                selection = input_tensor.detach().float().clamp_min(0.0)
                normalized = selection / selection.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                selection_min, selection_max = torch.aminmax(normalized)
                input_context = {
                    "selection_min": selection_min,
                    "selection_max": selection_max,
                }
            if (
                location is not None
                and label.startswith("workspace_bistable_gate:")
                and isinstance(output, tuple | list)
                and len(output) > 1
                and torch.is_tensor(output[1])
                and output_tensor is not None
                and output_tensor.requires_grad
            ):
                probability = output[1].detach().float()
                probability_min, probability_max = torch.aminmax(probability)
                output_context = {
                    "prob_min": probability_min,
                    "prob_max": probability_max,
                }
            if input_tensor is not None:
                self._accumulate_tensor(record, "in", input_tensor, location=location)
                if input_tensor.requires_grad:
                    input_tensor.register_hook(
                        lambda grad,
                        target=record,
                        call_location=location,
                        context=input_context: self._accumulate_tensor(
                            target,
                            "grad_in",
                            grad,
                            location=call_location,
                            context=context,
                        )
                    )
            if output_tensor is not None:
                self._accumulate_tensor(record, "out", output_tensor, location=location)
                if output_tensor.requires_grad:
                    output_tensor.register_hook(
                        lambda grad,
                        target=record,
                        call_location=location,
                        context=output_context: self._accumulate_tensor(
                            target,
                            "grad_out",
                            grad,
                            location=call_location,
                            context=context,
                        )
                    )

            # CfC's recurrent state is a separate input/output path. Keep its
            # hidden-state gradients visible so BPTT spikes are not hidden by
            # the block's ordinary output gradient.
            if tracks_cfc_state:
                for state_index, state_tensor in enumerate(_all_tensors(inputs[1:]), start=1):
                    if state_tensor.requires_grad:
                        state_tensor.register_hook(
                            lambda grad,
                            target=record,
                            call_location=location,
                            prefix=f"grad_in_{state_index}": self._accumulate_tensor(
                                target, prefix, grad, location=call_location
                            )
                        )

            # Structured recurrent modules can return a large persistent state
            # alongside their visible output. Keep these gradients separate so
            # a state-path spike cannot be hidden by the first returned tensor.
            secondary_outputs = output_tensors[1:] if tracks_secondary_outputs else ()
            for output_index, tensor in enumerate(secondary_outputs, start=1):
                output_prefix = f"out_{output_index}"
                gradient_prefix = f"grad_out_{output_index}"
                self._accumulate_tensor(record, output_prefix, tensor, location=location)
                if tensor.requires_grad:
                    tensor.register_hook(
                        lambda grad,
                        target=record,
                        call_location=location,
                        prefix=gradient_prefix,
                        context=output_context
                        if output_index == 1
                        else None: self._accumulate_tensor(
                            target,
                            prefix,
                            grad,
                            location=call_location,
                            context=context,
                        )
                    )

        self._handles.append(module.register_forward_hook(hook))

    def _accumulate_tensor(
        self,
        record: dict[str, Any],
        prefix: str,
        tensor: torch.Tensor,
        *,
        location: tuple[int, int] | None = None,
        context: dict[str, torch.Tensor] | None = None,
    ) -> None:
        value = tensor.detach().float().reshape(-1)
        finite = torch.isfinite(value)
        if (
            prefix.startswith("grad")
            and self._nonfinite_trace_enabled
            and not self._nonfinite_trace_reported
        ):
            bad_count = (~finite).sum()
            if bool(bad_count):
                finite_values = value[finite]
                bad_fraction = float((bad_count.float() / max(1, value.numel())).cpu())
                finite_max = (
                    float(finite_values.abs().max().cpu())
                    if finite_values.numel()
                    else float("nan")
                )
                call_location = f" step={location[0]} call={location[1]}" if location else ""
                print(
                    f"[segment-first-nonfinite] {record['path']} {prefix}"
                    f"{call_location} bad={bad_fraction:.3%} "
                    f"finite_absmax={finite_max:.4g} shape={tuple(tensor.shape)}"
                )
                self._nonfinite_trace_reported = True
        safe = torch.where(finite, value, torch.zeros_like(value))
        values = {
            f"{prefix}_sum_sq": _stable_sum_squares(safe),
            f"{prefix}_max": safe.abs().amax() if safe.numel() else safe.new_zeros(()),
            f"{prefix}_bad": (~finite).sum().float(),
        }
        for key, item in values.items():
            if key in record and key.endswith("_max"):
                previous = record[key]
                is_new_max = item > previous
                record[key] = torch.maximum(record[key], item)
                if location is not None:
                    step_key = f"{key}_step"
                    call_key = f"{key}_call"
                    old_step = record.get(step_key, torch.full_like(item, -1, dtype=torch.long))
                    old_call = record.get(call_key, torch.full_like(item, -1, dtype=torch.long))
                    record[step_key] = torch.where(
                        is_new_max, torch.full_like(old_step, location[0]), old_step
                    )
                    record[call_key] = torch.where(
                        is_new_max, torch.full_like(old_call, location[1]), old_call
                    )
                    for context_name, context_value in (context or {}).items():
                        context_key = f"{key}_context_{context_name}"
                        previous_context = record.get(context_key, torch.zeros_like(context_value))
                        record[context_key] = torch.where(
                            is_new_max, context_value.to(previous_context), previous_context
                        )
            else:
                record[key] = record[key] + item if key in record else item
                if key.endswith("_max") and location is not None:
                    record[f"{key}_step"] = torch.full_like(item, location[0], dtype=torch.long)
                    record[f"{key}_call"] = torch.full_like(item, location[1], dtype=torch.long)
                    for context_name, context_value in (context or {}).items():
                        record[f"{key}_context_{context_name}"] = context_value.detach().clone()
        record[f"{prefix}_numel"] = record.get(f"{prefix}_numel", 0) + value.numel()

    def set_nonfinite_trace(self, enabled: bool) -> None:
        """Print the first bad segment-boundary gradient for one diagnostic step."""
        self._nonfinite_trace_enabled = bool(enabled)
        self._nonfinite_trace_reported = False

    def add_module(self, module: nn.Module, label: str) -> None:
        """Add a caller-named module such as the task output projection."""
        if not self.enabled:
            return
        self._register(module, label, label)
        self._external_modules.append((label, module))

    def set_channel(self, channel: str | None) -> None:
        self.channel = channel

    def set_step(self, step: int | None) -> None:
        """Set the outer training step used to locate recurrent call maxima."""
        self._current_step = step
        self._step_call_counts.clear()

    def observe_tensor(self, label: str, tensor: torch.Tensor) -> None:
        """Record a named tensor at a call site that has no module boundary.

        This keeps diagnostics on-device and lets recurrent models expose
        exact per-step values without adding synchronization to their loop.
        """
        if not self.enabled or self.channel is None:
            return
        record = self._records.setdefault(self.channel, OrderedDict()).setdefault(
            label, {"path": label, "calls": 0}
        )
        record["calls"] += 1
        self._accumulate_tensor(record, "out", tensor)

    def observe_recurrent_state(self, tensor: torch.Tensor, *, call_index: int) -> None:
        """Track forward and backward values for an internal recurrent state."""
        if not self.enabled or self.channel != "train":
            return
        label = "workspace.slot_state_step"
        record = self._records.setdefault(self.channel, OrderedDict()).setdefault(
            label, {"path": label, "calls": 0}
        )
        record["calls"] += 1
        location = (self._current_step, call_index) if self._current_step is not None else None
        self._accumulate_tensor(record, "out", tensor, location=location)
        if tensor.requires_grad:
            tensor.register_hook(
                lambda grad, target=record, call_location=location: self._accumulate_tensor(
                    target, "grad_out", grad, location=call_location
                )
            )

    def _gradient_records(self) -> OrderedDict[str, tuple[torch.Tensor, int]]:
        records: OrderedDict[str, list[tuple[torch.Tensor, int]]] = OrderedDict()
        unassigned: list[tuple[torch.Tensor, int]] = []
        for name, parameter in self.model.named_parameters():
            grad = parameter.grad
            if grad is None:
                continue
            value = grad.detach().float()
            contribution = _stable_sum_squares(value)
            matched = False
            for path, label in self._segments:
                if path and (name == path or name.startswith(path + ".")):
                    records.setdefault(label, []).append((contribution, parameter.numel()))
                    matched = True
            if not matched:
                unassigned.append((contribution, parameter.numel()))
        for label, module in self._external_modules:
            for parameter in module.parameters():
                grad = parameter.grad
                if grad is not None:
                    value = grad.detach().float()
                    records.setdefault(label, []).append(
                        (_stable_sum_squares(value), parameter.numel())
                    )
        if unassigned:
            records["other"] = unassigned
        result = OrderedDict()
        for label, tensors in records.items():
            result[label] = (
                torch.stack([value for value, _ in tensors]).sum().sqrt(),
                sum(numel for _, numel in tensors),
            )
        return result

    def emit(self, channel: str, *, include_gradients: bool = False) -> None:
        if not self.enabled:
            return
        records = self._records.pop(channel, None)
        gradient_records = self._gradient_records() if include_gradients else OrderedDict()
        if not records and not gradient_records:
            return

        scalar_tensors: list[torch.Tensor] = []
        record_shapes: list[
            tuple[
                str,
                int,
                list[tuple[str, int, str]],
                dict[str, tuple[int, int, dict[str, float]]],
            ]
        ] = []
        for label, record in (records or {}).items():
            locations = {}
            gradient_prefixes = ["grad_in", "grad_out"]
            secondary_output_indices = sorted(
                int(key[4:-6])
                for key in record
                if key.startswith("out_") and key.endswith("_numel") and key[4:-6].isdigit()
            )
            secondary_input_indices = sorted(
                int(key[8:-6])
                for key in record
                if key.startswith("grad_in_") and key.endswith("_numel") and key[8:-6].isdigit()
            )
            gradient_prefixes.extend(
                f"grad_in_{input_index}" for input_index in secondary_input_indices
            )
            gradient_prefixes.extend(
                f"grad_out_{output_index}" for output_index in secondary_output_indices
            )
            location_prefixes = ["in", "out", *gradient_prefixes]
            location_prefixes.extend(
                f"out_{output_index}" for output_index in secondary_output_indices
            )
            for prefix in location_prefixes:
                maximum_key = f"{prefix}_max"
                step_value = record.get(f"{maximum_key}_step")
                call_value = record.get(f"{maximum_key}_call")
                if step_value is not None and call_value is not None:
                    context = {
                        name.removeprefix(f"{maximum_key}_context_"): float(value.cpu())
                        for name, value in record.items()
                        if name.startswith(f"{maximum_key}_context_")
                    }
                    locations[prefix] = (int(step_value.cpu()), int(call_value.cpu()), context)
            metrics = [
                ("in", record.get("in_numel", 0), "in"),
                ("out", record.get("out_numel", 0), "out"),
                ("grad_in", record.get("grad_in_numel", 0), "g_in"),
                ("grad_out", record.get("grad_out_numel", 0), "g_out"),
            ]
            metrics.extend(
                (
                    f"grad_in_{input_index}",
                    record.get(f"grad_in_{input_index}_numel", 0),
                    f"g_in{input_index}",
                )
                for input_index in secondary_input_indices
            )
            for output_index in secondary_output_indices:
                metrics.extend(
                    (
                        (
                            f"out_{output_index}",
                            record.get(f"out_{output_index}_numel", 0),
                            f"out{output_index}",
                        ),
                        (
                            f"grad_out_{output_index}",
                            record.get(f"grad_out_{output_index}_numel", 0),
                            f"g_out{output_index}",
                        ),
                    )
                )
            record_shapes.append((label, record["calls"], metrics, locations))
            for prefix, _, _ in metrics:
                for suffix in ("sum_sq", "max", "bad"):
                    item = record.get(f"{prefix}_{suffix}")
                    if item is None:
                        item = torch.zeros((), device=next(self.model.parameters()).device)
                    scalar_tensors.append(item.detach().float().reshape(()))
        gradient_shapes = list(gradient_records.items())
        scalar_tensors.extend(
            value.detach().float().reshape(()) for value, _ in gradient_records.values()
        )
        if scalar_tensors:
            values = torch.stack(scalar_tensors).cpu().tolist()
        else:
            values = []

        parts = []
        index = 0
        for label, calls, metrics, locations in record_shapes:
            summaries = []
            for prefix, numel, display in metrics:
                sum_sq, maximum, bad = values[index : index + 3]
                index += 3
                if numel:
                    location_text = ""
                    if prefix in locations:
                        location_step, location_call, context = locations[prefix]
                        context_text = ",".join(
                            f"{name}={value:.3g}" for name, value in context.items()
                        )
                        context_suffix = f",{context_text}" if context_text else ""
                        location_text = f"@step{location_step}/call{location_call}{context_suffix}"
                    summaries.append(
                        f"{display}:rms={(sum_sq / numel) ** 0.5:.3g},"
                        f"max={maximum:.3g}{location_text},nonfinite={bad / numel:.2%}"
                    )
            parts.append(f"{label}[calls={calls} {' '.join(summaries)}]")
        if gradient_shapes:
            gradients = []
            for label, (_, parameter_count) in gradient_shapes:
                norm = values[index]
                index += 1
                gradients.append(f"{label}={norm:.3g}/n{parameter_count}")
            parts.append("grad_norm[" + "; ".join(gradients) + "]")
        print(f"[segment-diag:{channel}] " + " | ".join(parts))

    def activation_absmax_exceeded(self, channel: str) -> tuple[bool, float, str | None]:
        """Check captured forward maxima against the optional safety limit."""
        if not self.enabled or self.activation_absmax_limit is None:
            return False, 0.0, None
        records: Mapping[str, dict[str, Any]] = self._records.get(channel, {})
        candidates: list[tuple[str, torch.Tensor]] = []
        for label, record in records.items():
            secondary_output_maxima = (
                key for key in record if key.startswith("out_") and key.endswith("_max")
            )
            for key in ("in_max", "out_max", *secondary_output_maxima):
                value = record.get(key)
                if torch.is_tensor(value):
                    candidates.append((label, value))
        if not candidates:
            return False, 0.0, None
        values = torch.stack([value.detach().float() for _, value in candidates])
        maximum, index = values.max(dim=0)
        maximum_value = float(maximum.cpu())
        label = candidates[int(index.cpu())][0]
        return maximum_value > self.activation_absmax_limit, maximum_value, label

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
