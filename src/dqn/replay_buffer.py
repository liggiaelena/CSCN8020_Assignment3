from __future__ import annotations

import random
from collections import deque, namedtuple

import numpy as np
import torch

Transition = namedtuple(
    "Transition",
    ["state", "action", "reward", "next_state", "terminated"],
)


class ReplayBuffer:
    """
    Fixed-capacity experience replay buffer.

    Stores ``terminated`` rather than a combined ``done`` flag. A
    time-limit truncation is NOT a true terminal state, so its
    transitions must still bootstrap from ``next_state`` in the
    Bellman target. Only genuine termination (the elbow held the
    goal long enough) should stop bootstrapping. See
    ``DQNAgent.optimize_model`` for where this mask is used.
    """

    def __init__(self, capacity: int) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        self.buffer.append(
            Transition(
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(terminated),
            )
        )

    def sample(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch = random.sample(self.buffer, batch_size)

        states = torch.as_tensor(
            np.stack([t.state for t in batch]),
            dtype=torch.float32,
            device=device,
        )

        actions = torch.as_tensor(
            [t.action for t in batch],
            dtype=torch.int64,
            device=device,
        ).unsqueeze(1)

        rewards = torch.as_tensor(
            [t.reward for t in batch],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)

        next_states = torch.as_tensor(
            np.stack([t.next_state for t in batch]),
            dtype=torch.float32,
            device=device,
        )

        terminated = torch.as_tensor(
            [t.terminated for t in batch],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)

        return states, actions, rewards, next_states, terminated

    def __len__(self) -> int:
        return len(self.buffer)
