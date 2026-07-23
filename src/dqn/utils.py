from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """
    Seed every source of randomness the DQN pipeline touches.

    Gymnasium's own RNG is seeded separately, per call, via
    ``env.reset(seed=seed)`` -- it is not global state.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
