"""
file: data/common/digit_codec.py

Shared digit-based label encoding/decoding + loss/diversity, factored out of
data/graph_traversal/graph_traversal.py's LABEL_DIGITS/DIGIT_BASE/encode_label/
label_to_digits/digit_loss/prediction_diversity so every new modality dataset
(text/audio/video/multimodal) can reuse the exact same "predict N digits per
field, one-hot over DIGIT_BASE" convention instead of re-deriving it.
graph_traversal.py itself is left untouched (it predates this module and is a
validated Phase-2+ dataset not worth touching) - this is purely for NEW
dataset modules to import from.
"""

from __future__ import annotations

import torch


class DigitCodec:
    def __init__(self, num_digits: int = 3, digit_base: int = 10):
        self.num_digits = num_digits
        self.digit_base = digit_base
        self.label_dim = num_digits * digit_base
        self.label_range = digit_base**num_digits

    def encode_label(self, label) -> torch.Tensor:
        vec = torch.zeros(self.label_dim)
        if label is None:
            return vec
        for pos, d in enumerate(self._digits(label)):
            vec[pos * self.digit_base + d] = 1.0
        return vec

    def _digits(self, label: int):
        s, out = label, []
        for _ in range(self.num_digits):
            out.append(s % self.digit_base)
            s //= self.digit_base
        return list(reversed(out))

    def label_to_digit_targets(self, label: int):
        return self._digits(label)

    def decode_field(self, logits_flat: torch.Tensor) -> int:
        """logits_flat: (num_digits * digit_base,) -> int label."""
        logits = logits_flat.view(self.num_digits, self.digit_base)
        val = 0
        for d in logits.argmax(dim=-1).tolist():
            val = val * self.digit_base + d
        return val


def digit_field_loss(
    output: torch.Tensor,
    target_digits: torch.Tensor,
    answer_mask: torch.Tensor,
    num_fields: int,
    digit_base: int,
) -> torch.Tensor:
    """Generalizes graph_traversal.digit_loss to an arbitrary field count
    (graph uses num_fields=3 [src,edge,dst]; the chain-task family uses 2
    [value,cumsum])."""
    B, T, _ = output.shape
    num_digits_total = target_digits.shape[-1]
    logits = output.view(B, T, num_digits_total, digit_base)
    log_probs = torch.log_softmax(logits, dim=-1)
    gathered = torch.gather(log_probs, -1, target_digits.unsqueeze(-1)).squeeze(-1)
    per_step_loss = -gathered.sum(dim=-1)
    mask = answer_mask.float()
    total = (per_step_loss * mask).sum()
    denom = mask.sum().clamp(min=1.0)
    return total / denom


def digit_field_diversity(
    output: torch.Tensor, answer_mask: torch.Tensor, num_digits_total: int, digit_base: int
) -> float:
    mask = answer_mask.bool()
    if mask.sum() == 0:
        return 0.0
    logits = output.view(*output.shape[:2], num_digits_total, digit_base)
    preds = logits.argmax(dim=-1)[mask]
    if preds.numel() == 0:
        return 0.0
    return sum(preds[:, d].unique().numel() for d in range(num_digits_total)) / num_digits_total
