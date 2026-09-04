"""Gymnasium adapter: one snake game as a standard ``gym.Env`` (``SnakeBatch`` with ``n=1``).

Global context: used wherever a single classic environment is expected - the env checkers,
``Monitor``/``make_vec_env`` evaluation envs, human play and the DummyVecEnv benchmark row.
Training uses the batched adapter in ``vec_env.py``; both share the rules in ``physics.py``.

Local notes: Gymnasium API (``reset -> (obs, info)``, ``step -> 5-tuple``) from
https://gymnasium.farama.org/api/env/ ; ``action_masks()`` is the method name sb3-contrib looks
for (``EXPECTED_METHOD_NAME`` in
https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/maskable/utils.py),
and ``DummyVecEnv.env_method`` finds it through wrappers such as ``Monitor``.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from snake4d import render
from snake4d.config import Config
from snake4d.physics import SnakeBatch


class SnakeEnv(gym.Env):
    """N-dimensional snake; observation = flat float32 vector, action = one of 2*ndim moves."""

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 10}

    def __init__(self, cfg: Config | None = None, render_mode: str | None = None) -> None:
        self.cfg = cfg or Config()
        self.render_mode = render_mode
        self.sim = SnakeBatch(self.cfg, n=1)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.sim.obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.cfg.n_actions)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Gymnasium reset: seeds both the base-class RNG and the simulator's generator."""
        super().reset(seed=seed)
        if seed is not None:
            self.sim.seed(seed)
        self.sim.reset()
        return self.sim.observe()[0], self.sim.infos()[0]

    def step(self, action: int):
        """One move; returns ``obs, reward, terminated, truncated, info``."""
        reward, terminated, truncated = self.sim.step(np.array([action]))
        return (
            self.sim.observe()[0],
            float(reward[0]),
            bool(terminated[0]),
            bool(truncated[0]),
            self.sim.infos()[0],
        )

    def action_masks(self) -> np.ndarray:
        """Legal-action mask for MaskablePPO (shape ``(2*ndim,)``)."""
        return self.sim.action_masks()[0]

    def set_curriculum(self, hi: int, window: int, p_true_start: float) -> None:
        """Forward the Backplay frontier to the simulator (called through ``env_method``)."""
        self.sim.set_curriculum(hi, window, p_true_start)

    def render(self):
        """``ansi`` -> text montage of 2D slices; ``rgb_array`` -> uint8 image of the montage."""
        board = self.sim.board(0)
        if self.render_mode == "rgb_array":
            return render.to_rgb(board)
        return render.ascii(board)
