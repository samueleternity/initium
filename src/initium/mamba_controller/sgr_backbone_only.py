"""DNC-free split-graph backbone for SGR Phase 2 experiments."""

from __future__ import annotations

import weakref

import torch
import torch.nn as nn

from initium.mamba_controller.mamba_backbone_parallel import MambaBackboneParallel


class _MemoryMetadata:
    """Compatibility metadata for generic trainer reporting; owns no weights/state."""

    def __init__(self, nr_cells: int):
        self.nr_cells = int(nr_cells)

    def begin_stage_diagnostics(self) -> None:
        pass

    def finish_stage_diagnostics(self) -> dict[str, float]:
        return {}


class SurpriseRecallBackboneOnly(nn.Module):
    """Two-block Mamba model with no DNC memory or recurrent combiner.

    The small ``memories`` metadata list exists only because the shared trainer
    uses its memory capacity for checkpoint/run descriptions. It is not an
    ``nn.Module`` and contributes no parameters or computation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        nr_cells_metadata: int,
        variant: str = "mamba1",
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        delta_error_gating: bool = True,
        device: torch.device | None = None,
    ):
        super().__init__()
        if variant != "mamba1":
            raise ValueError("SGR Phase 2 currently requires the Mamba-1 backbone")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size
        self.nn_output_size = hidden_size
        self.nr_cells = int(nr_cells_metadata)
        self.variant = variant
        self.backbone = MambaBackboneParallel(
            in_dim=input_size,
            d_model=hidden_size,
            num_blocks=2,
            variant=variant,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            sgr_delta_error_gating=delta_error_gating,
            device=device,
        )
        self.memories = [_MemoryMetadata(nr_cells_metadata)]
        self.moe_enabled = False
        self.moe_layers: list[nn.Module] = []
        self.phase2_arm = "content"
        self.phase2_signal_mode = "excess-surprise"
        self.gate_eval_override = "none"
        self._prediction_decoder_ref = None
        self.last_sgr_diagnostics: dict[str, torch.Tensor] = {}
        self.last_sgr_signal_diagnostics: dict[str, torch.Tensor] = {}

    def set_prediction_decoder(self, decoder: nn.Module | None) -> None:
        # Keep the external decoder out of this module's state_dict and named
        # parameters; core_training owns its optimizer/checkpoint lifecycle.
        object.__setattr__(
            self,
            "_prediction_decoder_ref",
            weakref.ref(decoder) if decoder is not None else None,
        )

    def forward(
        self,
        input: torch.Tensor,
        hx=(None, None, None),
        reset_experience: bool = False,
        pass_through_memory: bool = True,
        *,
        return_backbone_features: bool = False,
        return_prediction_logits: bool = False,
        oracle_flags: torch.Tensor | None = None,
        stream_lengths: list[int] | None = None,
        **_unused,
    ):
        del reset_experience, pass_through_memory
        decoder_ref = self._prediction_decoder_ref
        decoder = decoder_ref() if decoder_ref is not None else None
        features, first_features, prediction_logits, diagnostics = self.backbone.forward_sgr(
            input,
            arm=self.phase2_arm,
            decoder=decoder,
            signal_mode=self.phase2_signal_mode,
            stream_lengths=stream_lengths,
            oracle_flags=oracle_flags,
            gate_override=self.gate_eval_override if not self.training else "none",
        )
        self.last_sgr_diagnostics = diagnostics
        signal = diagnostics["prediction_error"]
        valid_positions = torch.arange(signal.shape[1], device=signal.device).unsqueeze(0)
        lengths = (
            torch.as_tensor(stream_lengths, device=signal.device)
            if stream_lengths is not None
            else torch.where(
                input.argmax(dim=-1).eq(20),
                valid_positions,
                signal.shape[1],
            ).amin(dim=1)
        )
        valid_mask = valid_positions < lengths.unsqueeze(1)
        valid_error = signal[valid_mask]
        valid_gate = diagnostics["gate_signal"][valid_mask]
        valid_count = valid_mask.sum().clamp_min(1)
        self.last_sgr_signal_diagnostics = {
            "mean": valid_error.sum() / valid_count,
            "abs_mean": valid_error.abs().sum() / valid_count,
            "positive_frac": valid_error.gt(0).float().sum() / valid_count,
            "gate_mean": valid_gate.sum() / valid_count,
        }
        output = features.transpose(0, 1)
        state = (None, None, None)
        if return_backbone_features and return_prediction_logits:
            return output, state, first_features, prediction_logits
        if return_backbone_features:
            return output, state, first_features
        if return_prediction_logits:
            return output, state, prediction_logits
        return output, state
