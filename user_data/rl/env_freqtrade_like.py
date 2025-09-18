"""Simplified reinforcement-learning environment approximating Freqtrade fills."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional dependency
    gym = None
    spaces = None


@dataclass
class EnvConfig:
    feature_columns: Sequence[str]
    max_position_size: float = 1.0
    fee: float = 0.001
    slippage: float = 0.0005


class FreqtradeLikeEnv(gym.Env if gym else object):
    """Iterates through trade feature rows and learns fractional sizing decisions."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, data: pd.DataFrame, config: EnvConfig) -> None:
        if gym is None or spaces is None:  # pragma: no cover - fallback when gym missing
            raise ImportError("gymnasium is required to use FreqtradeLikeEnv")
        if data.empty:
            raise ValueError("Environment requires non-empty dataset")
        self.data = data.reset_index(drop=True)
        self.config = config
        self.index = 0

        self.action_space = spaces.Box(low=0.0, high=config.max_position_size, shape=(1,), dtype=np.float32)
        obs_dim = len(config.feature_columns)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        super().reset(seed=seed)
        self.index = 0
        return self._get_observation(), {}

    def step(self, action):  # type: ignore[override]
        size = float(np.clip(action[0], 0.0, self.config.max_position_size))
        row = self.data.loc[self.index]
        reward = self._calculate_reward(size, row)

        self.index += 1
        terminated = self.index >= len(self.data)
        obs = self._get_observation() if not terminated else np.zeros_like(self.observation_space.sample())
        info = {"size": size, "profit_ratio": row.get("profit_ratio", 0.0)}
        return obs, reward, terminated, False, info

    def _get_observation(self) -> np.ndarray:
        row = self.data.loc[self.index]
        obs = row[self.config.feature_columns].to_numpy(dtype=np.float32)
        return obs

    def _calculate_reward(self, size: float, row: pd.Series) -> float:
        gross = row.get("profit_ratio", 0.0) * size
        costs = (self.config.fee + self.config.slippage) * abs(size)
        return gross - costs

    def render(self):  # pragma: no cover - not required for training
        print(f"Step {self.index}")

    def close(self):  # pragma: no cover
        pass
