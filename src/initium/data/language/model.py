"""Model assembly and causal language adapter for the selected architecture."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from initium.data.language.config import ArchitectureConfig
from initium.data.language.heads import LanguageObjectiveHeads
from initium.mamba_controller.split_graph_dnc import SplitGraphDNC


@dataclass
class LanguageModelOutput:
    logits: torch.Tensor
    hidden: torch.Tensor
    state: object


class LanguageWorldModel(nn.Module):
    """Split-graph Mamba language model with the requested parallel workspace."""

    def __init__(self, vocab_size: int, *, hidden_size: int = 512,
                 architecture: ArchitectureConfig = ArchitectureConfig(),
                 action_vocab_size: int | None = None):
        super().__init__()
        if vocab_size < 320:
            raise ValueError("vocabulary must be at least 320 for token-ID input")
        if not architecture.split_graph or architecture.split_graph_variant != "mamba1":
            raise ValueError("this language model assembly requires split-graph Mamba1")
        self.vocab_size = vocab_size
        self.backbone = SplitGraphDNC(
            input_size=vocab_size,
            hidden_size=hidden_size,
            nr_cells=256,
            cell_size=64,
            read_heads=4,
            num_backbone_blocks=architecture.split_graph_num_blocks,
            mamba_variant=architecture.split_graph_variant,
            mamba_d_state=architecture.mamba_d_state,
            mamba_d_conv=architecture.mamba_d_conv,
            mamba_expand=architecture.mamba_expand,
            dnc_enabled=False,
            memory_type=architecture.memory_type,
            workspace_token_id_input=architecture.workspace_token_id_input,
            workspace_parallel_dense=architecture.workspace_parallel_dense,
            workspace_block_index=architecture.workspace_block_index,
            workspace_num_slots=architecture.workspace_num_slots,
            workspace_key_dim=architecture.workspace_key_dim,
            workspace_scan_chunk=architecture.workspace_scan_chunk,
            modulation_enabled=architecture.modulation,
            modulation_targets=architecture.modulation_targets,
            modulation_blocks=architecture.modulation_blocks,
            modulation_source=architecture.modulation_source,
            modulation_rank=architecture.modulation_rank,
            modulation_alphas=architecture.modulation_alphas,
            modulation_lambda=architecture.modulation_lambda,
            modulation_temporal_lambda=architecture.modulation_temporal_lambda,
            modulation_meta_responsiveness="responsiveness" in architecture.effective_neuromod_arms,
        )
        self.output_projection = nn.Linear(hidden_size, vocab_size, bias=False)
        self.objective_heads = LanguageObjectiveHeads(hidden_size, action_vocab_size or vocab_size)

    def encode(self, token_ids: torch.Tensor, *, reset_state: bool = True):
        if token_ids.ndim != 2:
            raise ValueError("token ids must have shape (batch, time)")
        valid = token_ids.ne(0)
        time_major, state = self.backbone(
            token_ids,
            (None, None, None),
            reset_experience=reset_state,
            sequence_valid_mask=valid,
            prediction_valid_mask=valid,
        )
        hidden = time_major.transpose(0, 1)
        return hidden, state

    def forward(self, token_ids: torch.Tensor, *, reset_state: bool = True) -> LanguageModelOutput:
        hidden, state = self.encode(token_ids, reset_state=reset_state)
        return LanguageModelOutput(self.output_projection(hidden), hidden, state)

    def pop_modulation_aux_loss(self):
        head = getattr(self.backbone, "modulation_head", None)
        return head.pop_aux_loss() if head is not None else None


def pool_sequence(hidden: torch.Tensor, attention_mask: torch.Tensor | None = None):
    """Masked mean pool token features into a state representation."""
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
