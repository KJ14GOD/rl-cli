from __future__ import annotations

from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.envs.registration import register, registry

ENV_ID = "RLXCustomCartPole-v0"


class CustomCartPoleEnv(CartPoleEnv):
    """Starter custom environment with a small angle penalty for reward shaping."""

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        angle_penalty = abs(float(obs[2])) * 0.05
        shaped_reward = float(reward) - angle_penalty
        info["angle_penalty"] = angle_penalty
        return obs, shaped_reward, terminated, truncated, info


if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="envs.custom_env:CustomCartPoleEnv",
        max_episode_steps=500,
    )
