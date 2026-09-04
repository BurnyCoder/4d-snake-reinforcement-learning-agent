"""Scripted policies that plug into the same evaluation pipeline as trained models.

Global context: the Hamiltonian route follower proves that a board is completable and gives the
perfect-play step count to beat (twanvl/snake, MIT: a fixed cycle wins 100 % of games); the masked
random policy is the floor.  Both duck-type ``MaskablePPO.predict`` so sb3-contrib's
``evaluate_policy`` drives them exactly like a trained agent (``evaluation.py``).

Local notes: ``predict(observation, state, episode_start, deterministic, action_masks)`` is the
signature sb3-contrib calls
(https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/maskable/evaluation.py);
the head cell is decoded from the head one-hot block of the observation (``physics.observe``).
"""

import numpy as np

from snake4d.config import Config
from snake4d.grid import neighbour_table
from snake4d.hamilton import route_actions, route_for


class ScriptedPolicy:
    """Base: batched ``predict`` over observations + masks, one ``_act`` call per row."""

    def __init__(self, cfg: Config, seed: int | None = None) -> None:
        self.cfg, self.n_cells, self.n_actions = cfg, cfg.n_cells, cfg.n_actions
        self.rng = np.random.default_rng(seed)

    def set_random_seed(self, seed: int | None) -> None:
        """Same hook as ``BaseAlgorithm.set_random_seed`` so evaluation seeds both alike."""
        self.rng = np.random.default_rng(seed)

    def predict(self, observation, state=None, episode_start=None, deterministic=True,
                action_masks=None):
        """Return ``(actions, state)`` for a batch of observations (MaskablePPO.predict API)."""
        obs = np.atleast_2d(observation)
        masks = (
            np.ones((len(obs), self.n_actions), dtype=bool)
            if action_masks is None
            else np.atleast_2d(action_masks)
        )
        actions = np.array([self._act(o, m) for o, m in zip(obs, masks, strict=True)])
        return actions, state

    def head_cell(self, obs: np.ndarray) -> int:
        """Flat head index from the head one-hot block ``obs[2C:3C]``."""
        return int(np.argmax(obs[2 * self.n_cells : 3 * self.n_cells]))

    def _act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        raise NotImplementedError


class RandomMaskedPolicy(ScriptedPolicy):
    """Uniformly random legal action: the floor every learned agent must beat."""

    def _act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        return int(self.rng.choice(np.flatnonzero(mask)))


class RoutePolicy(ScriptedPolicy):
    """Follow the precomputed Hamiltonian route; fall back to the first legal move if blocked."""

    def __init__(self, cfg: Config, seed: int | None = None) -> None:
        super().__init__(cfg, seed)
        route, closed = route_for(cfg.size, cfg.ndim)
        self.actions = route_actions(route, closed, neighbour_table(cfg.size, cfg.ndim))
        self.fallbacks = 0  # how often the route was blocked (never on a closed cycle)

    def _act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        action = int(self.actions[self.head_cell(obs)])
        if action < 0 or not mask[action]:
            self.fallbacks += 1
            action = int(np.flatnonzero(mask)[0])
        return action


POLICIES = {"route": RoutePolicy, "random": RandomMaskedPolicy}
