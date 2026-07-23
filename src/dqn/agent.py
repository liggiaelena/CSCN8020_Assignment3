from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .q_network import QNetwork
from .replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Online/target Q-network pair plus the optimization step that
    turns sampled transitions into a Bellman-target update.
    """

    def __init__(
        self,
        observation_dim: int = 4,
        action_dim: int = 3,
        hidden_size: int = 64,
        gamma: float = 0.95,
        learning_rate: float = 0.001,
        target_update_interval: int = 250,
        grad_clip_norm: float = 10.0,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.grad_clip_norm = grad_clip_norm

        self.online_network = QNetwork(
            observation_dim,
            action_dim,
            hidden_size,
        ).to(self.device)

        self.target_network = QNetwork(
            observation_dim,
            action_dim,
            hidden_size,
        ).to(self.device)

        # Target network starts as an exact copy of the online
        # network and is never trained directly by the optimizer.
        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=learning_rate,
        )

        # Huber loss: quadratic near zero, linear for large errors.
        # Less sensitive to the occasional large TD-error spike than
        # plain MSE, which helps stability early in training when
        # Q-value estimates are still poor.
        self.loss_fn = nn.SmoothL1Loss()

        self.optimization_steps = 0

    def select_action(
        self,
        observation: np.ndarray,
        epsilon: float,
        rng: np.random.Generator,
    ) -> int:
        """
        Epsilon-greedy action selection.

        With probability ``epsilon`` a uniformly random action is
        returned (exploration). Otherwise the action with the
        highest online-network Q-value is returned (exploitation).
        Pass ``epsilon=0.0`` for deterministic greedy evaluation.
        """
        if rng.random() < epsilon:
            return int(rng.integers(0, self.action_dim))

        with torch.no_grad():
            state = torch.as_tensor(
                observation,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            q_values = self.online_network(state)

            return int(torch.argmax(q_values, dim=1).item())

    def optimize_model(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int,
    ) -> float | None:
        """
        Sample one mini-batch and run one gradient step.

        Returns the scalar loss, or ``None`` if the buffer does not
        yet hold enough transitions for a full batch (the required
        warm-up period).
        """
        if len(replay_buffer) < batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            terminated,
        ) = replay_buffer.sample(batch_size, device=self.device)

        # Q(s, a) for the action that was actually taken.
        predicted_q_values = self.online_network(states).gather(
            1,
            actions,
        )

        with torch.no_grad():
            # max_a' Q_target(s', a'), evaluated on the target
            # network so bootstrapped targets stay fixed for
            # `target_update_interval` steps -- this is what
            # prevents the moving-target instability of using the
            # online network on both sides of the TD error.
            next_q_values = self.target_network(next_states).max(
                dim=1,
                keepdim=True,
            ).values

            # (1 - terminated) zeroes the bootstrap term exactly
            # when the episode ended for a real reason (the elbow
            # held the goal). A truncated-but-not-terminated
            # transition keeps terminated == 0, so it still
            # bootstraps from next_state -- truncation cuts the
            # episode short but does not mean "no future return."
            targets = rewards + self.gamma * next_q_values * (
                1.0 - terminated
            )

        loss = self.loss_fn(predicted_q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            max_norm=self.grad_clip_norm,
        )

        self.optimizer.step()

        self.optimization_steps += 1

        if self.optimization_steps % self.target_update_interval == 0:
            self.update_target_network()

        return float(loss.item())

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "online_network": self.online_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "optimization_steps": self.optimization_steps,
            },
            path,
        )

    def load_checkpoint(
        self,
        path: str | Path,
        map_location: torch.device | str | None = None,
    ) -> None:
        checkpoint = torch.load(
            path,
            map_location=map_location or self.device,
        )

        self.online_network.load_state_dict(
            checkpoint["online_network"]
        )
        self.target_network.load_state_dict(
            checkpoint["target_network"]
        )
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.optimization_steps = checkpoint.get(
            "optimization_steps",
            0,
        )
