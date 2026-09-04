"""Scripted policies that plug into the same evaluation pipeline as trained models.

Global context: the Hamiltonian route follower proves that a board is completable and gives the
perfect-play step count to beat (twanvl/snake, MIT, https://github.com/twanvl/snake: a fixed
cycle wins 100 % of its 30x30 games); the masked
random policy is the floor.  Both duck-type ``MaskablePPO.predict`` so sb3-contrib's
``evaluate_policy`` drives them exactly like a trained agent (``evaluation.py``).

Local notes:
* ``predict(observation, state, episode_start, deterministic, action_masks)`` is the signature
  sb3-contrib calls
  (https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/maskable/evaluation.py);
  head, body and food are decoded from the observation blocks of ``physics.observe``.
* On odd boards the route is a cycle over every cell but the corner (``hamilton``).  The follower
  enters the corner only when the food is there and the skipped arc of the cycle up to the exit
  cell is free, so the head can never run into its own tail (the perturbed-Hamiltonian-cycle
  rule "the head must never overtake the tail" from johnflux's Nokia-snake write-up,
  https://johnflux.com/2015/05/02/nokia-6110-part-3-algorithms/, applied to a one-cell detour).
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
        self.route, closed = route_for(cfg.size, cfg.ndim)
        self.neigh = neighbour_table(cfg.size, cfg.ndim)
        self.actions = route_actions(self.route, closed, self.neigh)
        self.pos = np.full(cfg.n_cells, -1, dtype=np.int64)  # position of each cell on the route
        self.pos[self.route] = np.arange(len(self.route))
        off_route = np.flatnonzero(self.pos < 0)
        self.corner = int(off_route[0]) if off_route.size else -1  # the skipped corner (odd boards)
        self.fallbacks = 0  # how often the route was blocked (never on an even board)

    def _act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        head = self.head_cell(obs)
        detour = self._corner_detour(obs, mask, head) if self.corner >= 0 else None
        if detour is not None:
            return detour
        action = int(self.actions[head])
        if action < 0 or not mask[action]:
            self.fallbacks += 1
            action = int(np.flatnonzero(mask)[0])
        return action

    # --- odd boards: the one cell off the cycle ------------------------------------------------
    def _corner_detour(self, obs: np.ndarray, mask: np.ndarray, head: int) -> int | None:
        """Enter the corner when the food is there and it is safe; leave it the step after.

        Safety (visit-order argument): after the detour the head re-enters the cycle ``d`` cells
        ahead of where it left, so it skips ``d - 1`` cells; those skipped cells are simply visited
        next lap, but the cells after the exit were last visited ``d - 1`` visits more recently than
        usual, so they are free only if ``d - 1 <= free cycle cells``.
        """
        n = self.n_cells
        body, food = obs[:n] > 0, int(np.argmax(obs[3 * n : 4 * n]))
        free = len(self.route) - int(body.sum()) + int(body[self.corner])  # free cycle cells
        if head == self.corner:  # leaving: the neck is the body cell with the highest age
            ages = np.where(body, obs[n : 2 * n], -1.0)
            ages[head] = -1.0
            exit_cell = self._exit_cell(int(np.argmax(ages)), free)
            action = None if exit_cell is None else self._action_to(head, exit_cell)
            return action if action is not None and mask[action] else None
        if food == self.corner and (free == 0 or self._exit_cell(head, free) is not None):
            action = self._action_to(head, self.corner)  # None unless the head is adjacent
            return action if action is not None and mask[action] else None
        return None

    def _exit_cell(self, entry: int, free: int) -> int | None:
        """Nearest corner neighbour ahead of ``entry`` that skips at most ``free`` cycle cells."""
        best: tuple[int, int] | None = None
        for candidate in self.neigh[self.corner]:
            if candidate < 0:
                continue
            distance = int((self.pos[candidate] - self.pos[entry]) % len(self.route))
            if 0 < distance <= free + 1 and (best is None or distance < best[0]):
                best = (distance, int(candidate))
        return None if best is None else best[1]

    def _action_to(self, cell: int, target: int) -> int | None:
        """The action moving from ``cell`` to the adjacent ``target`` (None if not adjacent)."""
        hits = np.flatnonzero(self.neigh[cell] == target)
        return int(hits[0]) if hits.size else None


POLICIES = {"route": RoutePolicy, "random": RandomMaskedPolicy}
