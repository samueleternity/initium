"""Small prediction paths over a shared language/workspace representation."""

from __future__ import annotations

import torch
from torch import nn


class StateProjection(nn.Module):
    def __init__(self, hidden_size: int, projection_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, projection_size),
                                 nn.GELU(), nn.Linear(projection_size, projection_size))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class ActionConditionedPredictor(nn.Module):
    """Predict next-state latent from current-state latent and action embedding."""

    def __init__(self, state_size: int, action_vocab_size: int, action_size: int = 128):
        super().__init__()
        self.action_embedding = nn.Embedding(action_vocab_size, action_size)
        self.predict = nn.Sequential(nn.Linear(state_size + action_size, state_size * 2),
                                     nn.GELU(), nn.Linear(state_size * 2, state_size))

    def forward(self, current_state: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
        action = self.action_embedding(action_ids)
        return self.predict(torch.cat((current_state, action), dim=-1))


class LogicHead(nn.Module):
    def __init__(self, hidden_size: int, labels: int = 8):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size),
                                 nn.GELU(), nn.Linear(hidden_size, labels))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class LanguageObjectiveHeads(nn.Module):
    """Auxiliary heads: same-state projector, transition predictor, and logic."""

    def __init__(self, hidden_size: int, action_vocab_size: int, projection_size: int = 256,
                 logic_labels: int = 8):
        super().__init__()
        self.state = StateProjection(hidden_size, projection_size)
        self.transition = ActionConditionedPredictor(projection_size, action_vocab_size)
        self.logic = LogicHead(hidden_size, logic_labels)

    def paired_state(self, first_hidden: torch.Tensor, second_hidden: torch.Tensor):
        return self.state(first_hidden), self.state(second_hidden)

    def future_state(self, current_hidden: torch.Tensor, action_ids: torch.Tensor):
        return self.transition(self.state(current_hidden), action_ids)
