"""Batched stable-baselines3 VecEnv adapter: all training snakes live in one ``SnakeBatch``.

Global context: PPO throughput on a cheap environment is dominated by per-env Python overhead and
inter-process communication (SB3 docs on DummyVecEnv vs SubprocVecEnv; Gymnasium paper Fig. 1:
custom numpy vectorisation beats both).  ``SnakeVecEnv`` therefore steps ``N`` boards with the
vectorised numpy rules of ``physics.py`` inside a single process; ``make_env`` wraps it in SB3's
``VecMonitor`` so episode statistics come from library code.  Evaluation and the training callback
use the same factory.

Local notes (the custom-VecEnv contract of SB3 v2.9.0,
https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/base_vec_env.py):
* exactly the eight abstract methods are implemented: reset, step_async, step_wait, close,
  get_attr, set_attr, env_method, env_is_wrapped;
* ``VecEnv.__init__`` calls ``self.get_attr("render_mode")`` before this subclass finishes
  initialising, so ``render_mode``/``num_envs`` exist beforehand;
* ``get_attr`` raises ``AttributeError`` for unknown names so the base ``has_attr`` (used by
  sb3-contrib's ``is_masking_supported``) works; ``env_method("action_masks")`` returns one mask
  row per env, which ``get_action_masks`` stacks;
* ``step_wait`` reproduces DummyVecEnv's auto-reset, ``terminal_observation`` and
  ``TimeLimit.truncated`` (https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html).
"""

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv, VecMonitor

from snake4d.config import Config
from snake4d.physics import SnakeBatch

INFO_KEYS = ("is_success", "fill", "start_len")  # copied into info["episode"] by VecMonitor


class SnakeVecEnv(VecEnv):
    """``n_envs`` snake games stepped together; observation/action spaces match ``SnakeEnv``."""

    render_mode = None  # read by VecEnv.__init__ through get_attr before __init__ completes
    EXPOSED = ("render_mode", "action_masks", "set_curriculum", "cfg", "sim")

    def __init__(self, cfg: Config, n_envs: int, seed: int | None = None) -> None:
        self.cfg = cfg
        self.sim = SnakeBatch(cfg, n_envs, seed)
        self.num_envs = n_envs
        self._actions = np.zeros(n_envs, dtype=np.int64)
        super().__init__(
            n_envs,
            spaces.Box(0.0, 1.0, shape=(self.sim.obs_size,), dtype=np.float32),
            spaces.Discrete(cfg.n_actions),
        )

    # --- the eight abstract methods -----------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Reset every board; honours a pending ``seed()`` like DummyVecEnv does."""
        if self._seeds[0] is not None:
            self.sim.seed(self._seeds[0])
        self._reset_seeds()
        self.sim.reset()
        self.reset_infos = self.sim.infos()
        return self.sim.observe()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, dtype=np.int64)

    def step_wait(self):
        """Step all boards, then auto-reset finished ones (keeping their terminal observation)."""
        reward, terminated, truncated = self.sim.step(self._actions)
        done = terminated | truncated
        obs = self.sim.observe()
        infos = self.sim.infos()  # computed BEFORE resets: fill/is_success of the finished episode
        done_rows = np.flatnonzero(done)
        for i in done_rows:
            infos[i]["terminal_observation"] = obs[i].copy()  # a view would see the reset below
            infos[i]["TimeLimit.truncated"] = bool(truncated[i] and not terminated[i])
        if done_rows.size:
            self.sim.reset(done_rows)
            obs[done_rows] = self.sim.observe(done_rows)
        return obs, reward, done, infos

    def close(self) -> None:
        """Nothing to release: no subprocesses, no window."""

    def get_attr(self, attr_name: str, indices=None) -> list:
        if attr_name not in self.EXPOSED:
            raise AttributeError(attr_name)  # lets VecEnv.has_attr return False for anything else
        return [getattr(self, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> list:
        """``action_masks`` (one row per env) and ``set_curriculum`` (whole batch)."""
        rows = list(self._get_indices(indices))
        if method_name == "action_masks":
            masks = self.sim.action_masks()
            return [masks[i] for i in rows]
        if method_name == "set_curriculum":
            self.set_curriculum(*args, **kwargs)
            return [None] * len(rows)
        raise AttributeError(method_name)

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False for _ in self._get_indices(indices)]

    # --- extras used by sb3-contrib and the curriculum ----------------------------------------
    def action_masks(self) -> np.ndarray:
        """``(n_envs, 2*ndim)`` legal-action mask."""
        return self.sim.action_masks()

    def set_curriculum(self, hi: int, window: int, p_true_start: float) -> None:
        """Backplay frontier for every board in the batch."""
        self.sim.set_curriculum(hi, window, p_true_start)


def make_env(cfg: Config, n_envs: int, seed: int | None = None,
             monitor_path: str | None = None) -> VecMonitor:
    """The one environment factory for training and evaluation: batched env + VecMonitor."""
    return VecMonitor(SnakeVecEnv(cfg, n_envs, seed), filename=monitor_path,
                      info_keywords=INFO_KEYS)
