"""Low-synchronization activation and gradient summaries for model segments.

Hooks are selected by module type rather than a particular model topology,
so the same collector covers split-graph, standalone Mamba/CfC, chained
Mamba+CfC, recurrent DNC, and workspace models.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import nn


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
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
    if class_name in {"MambaControllerWrapper", "Mamba2ControllerWrapper", "Mamba3ControllerWrapper"}:
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

    def __init__(self, model: nn.Module):
        self.model = model
        self.channel: str | None = "train"
        self._records: dict[str, OrderedDict[str, dict[str, Any]]] = {}
        self._segments: list[tuple[str, str]] = []
        self._external_modules: list[tuple[str, nn.Module]] = []
        self._handles = []
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
            output_tensor = _first_tensor(output)
            record = self._records.setdefault(channel, OrderedDict()).setdefault(
                label, {"path": path, "calls": 0}
            )
            record["calls"] += 1
            if input_tensor is not None:
                self._accumulate_tensor(record, "in", input_tensor)
                if input_tensor.requires_grad:
                    input_tensor.register_hook(
                        lambda grad, target=record: self._accumulate_tensor(
                            target, "grad_in", grad
                        )
                    )
            if output_tensor is not None:
                self._accumulate_tensor(record, "out", output_tensor)
                if output_tensor.requires_grad:
                    output_tensor.register_hook(
                        lambda grad, target=record: self._accumulate_tensor(
                            target, "grad_out", grad
                        )
                    )

        self._handles.append(module.register_forward_hook(hook))

    @staticmethod
    def _accumulate_tensor(record: dict[str, Any], prefix: str, tensor: torch.Tensor) -> None:
        value = tensor.detach().float().reshape(-1)
        finite = torch.isfinite(value)
        safe = torch.where(finite, value, torch.zeros_like(value))
        values = {
            f"{prefix}_sum_sq": safe.square().sum(),
            f"{prefix}_max": safe.abs().amax() if safe.numel() else safe.new_zeros(()),
            f"{prefix}_bad": (~finite).sum().float(),
        }
        for key, item in values.items():
            record[key] = record[key] + item if key in record else item
        record[f"{prefix}_numel"] = record.get(f"{prefix}_numel", 0) + value.numel()

    def add_module(self, module: nn.Module, label: str) -> None:
        """Add a caller-named module such as the task output projection."""
        self._register(module, label, label)
        self._external_modules.append((label, module))

    def set_channel(self, channel: str | None) -> None:
        self.channel = channel

    def _gradient_records(self) -> OrderedDict[str, tuple[torch.Tensor, int]]:
        records: OrderedDict[str, list[tuple[torch.Tensor, int]]] = OrderedDict()
        unassigned: list[tuple[torch.Tensor, int]] = []
        for name, parameter in self.model.named_parameters():
            grad = parameter.grad
            if grad is None:
                continue
            value = grad.detach().float()
            contribution = value.square().sum()
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
                    records.setdefault(label, []).append(
                        (grad.detach().float().square().sum(), parameter.numel())
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
        records = self._records.pop(channel, None)
        gradient_records = self._gradient_records() if include_gradients else OrderedDict()
        if not records and not gradient_records:
            return

        scalar_tensors: list[torch.Tensor] = []
        record_shapes: list[tuple[str, int, int, int, int, int]] = []
        for label, record in (records or {}).items():
            record_shapes.append(
                (
                    label,
                    record["calls"],
                    record.get("in_numel", 0),
                    record.get("out_numel", 0),
                    record.get("grad_in_numel", 0),
                    record.get("grad_out_numel", 0),
                )
            )
            for prefix in ("in", "out", "grad_in", "grad_out"):
                for suffix in ("sum_sq", "max", "bad"):
                    item = record.get(f"{prefix}_{suffix}")
                    if item is None:
                        item = torch.zeros((), device=next(self.model.parameters()).device)
                    scalar_tensors.append(item.detach().float().reshape(()))
        gradient_shapes = list(gradient_records.items())
        scalar_tensors.extend(value.detach().float().reshape(()) for value, _ in gradient_records.values())
        if scalar_tensors:
            values = torch.stack(scalar_tensors).cpu().tolist()
        else:
            values = []

        parts = []
        index = 0
        for label, calls, in_numel, out_numel, grad_in_numel, grad_out_numel in record_shapes:
            summaries = []
            for prefix, numel, display in (
                ("in", in_numel, "in"),
                ("out", out_numel, "out"),
                ("grad_in", grad_in_numel, "g_in"),
                ("grad_out", grad_out_numel, "g_out"),
            ):
                sum_sq, maximum, bad = values[index : index + 3]
                index += 3
                if numel:
                    summaries.append(
                        f"{display}:rms={(sum_sq / numel) ** 0.5:.3g},"
                        f"max={maximum:.3g},nonfinite={bad / numel:.2%}"
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

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
