from __future__ import annotations

import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomCartPolePolicy(ActorCriticPolicy):
    """Starter PPO policy users can edit directly inside their project."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("net_arch", [64, 64])
        kwargs.setdefault("activation_fn", nn.Tanh)
        super().__init__(*args, **kwargs)
