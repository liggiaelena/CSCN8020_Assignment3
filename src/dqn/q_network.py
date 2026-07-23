from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    """
    Maps a 4-value observation to 3 unconstrained action-value
    estimates (one per discrete action).

    Architecture (assignment-required minimum):
        Input            : 4
        Hidden layer 1   : 64 units, ReLU
        Hidden layer 2   : 64 units, ReLU
        Output           : 3 Q-values, no activation
    """

    def __init__(
        self,
        observation_dim: int = 4,
        action_dim: int = 3,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)
