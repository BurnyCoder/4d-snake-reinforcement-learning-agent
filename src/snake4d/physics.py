"""The one implementation of the N-dimensional snake rules, batched over ``n`` boards in numpy.

Global context: ``SnakeBatch`` holds every rule (movement, growth, collisions, food, win, caps,
action masks, observation, rewards, curriculum starts).  ``env.py`` wraps ``SnakeBatch(n=1)`` as a
Gymnasium env and ``vec_env.py`` wraps ``SnakeBatch(n=N)`` as a stable-baselines3 VecEnv, so the
rules exist exactly once.

Local notes:
* State is a body-age grid ``age[row, cell]``: 0 = empty, 1 = tail, ``length`` = head.  Moving the
  tail is "decrement every positive age"; collision is "age at the target cell > 0".  This is the
  representation of PufferLib's snake (MIT,
  https://github.com/PufferAI/PufferLib/blob/main/ocean/snake/snake.h), re-expressed in numpy.
* Order of one step (post-tail-move occupancy): target cell -> wall? -> eating? -> tail moves for
  non-eating rows -> body collision on the moved grid (following your own tail is legal) -> head is
  written with age = length -> win when ``length == C`` is checked BEFORE food is spawned.
* Food is drawn uniformly over free cells with masked random scores + argmax (no rejection loop),
  the numpy form of the Jumanji/mapox spawn.
* Starvation (``idle > idle_cap``) and the absolute cap (``t >= C*C``) are truncations with reward
  0, never a death: Gymnasium's terminated/truncated split,
  https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
* Optional potential-based shaping ``gamma*Phi(s') - Phi(s)`` with ``Phi(terminal) = 0`` keeps the
  optimal policy unchanged (Ng, Harada & Russell 1999, restated in https://arxiv.org/pdf/2501.00989).
"""

import numpy as np

from snake4d.config import Config
from snake4d.grid import coords_of, neighbour_table
from snake4d.hamilton import route_for

EMPTY, BODY, HEAD, FOOD = 0, 1, 2, 3  # cell codes used by render.py


class SnakeBatch:
    """``n`` independent snake games advanced together with vectorised numpy operations."""

    def __init__(self, cfg: Config, n: int, seed: int | None = None) -> None:
        self.cfg, self.n, self.n_cells = cfg, n, cfg.n_cells
        self.neigh = neighbour_table(cfg.size, cfg.ndim)  # (C, 2*ndim), -1 at walls
        self.coords = coords_of(np.arange(self.n_cells), cfg.size, cfg.ndim)  # for L1 distances
        self.route, self.closed = route_for(cfg.size, cfg.ndim)  # curriculum demonstration
        self.route_len = len(self.route)  # C, or C-1 on odd boards (cycle minus the corner)
        self.rng = np.random.default_rng(seed)
        self.rows = np.arange(n)
        self.age = np.zeros((n, self.n_cells), dtype=np.int16)
        self.head = np.zeros(n, dtype=np.int64)
        self.food = np.zeros(n, dtype=np.int64)
        self.length = np.zeros(n, dtype=np.int32)
        self.idle = np.zeros(n, dtype=np.int32)
        self.t = np.zeros(n, dtype=np.int32)
        self.start_len = np.ones(n, dtype=np.int32)
        self.cur_hi, self.cur_window, self.p_true_start = 1, cfg.curriculum_window, cfg.p_true_start
        self.obs_size = 4 * self.n_cells + 2

    # --- seeding / curriculum ----------------------------------------------------------------
    def seed(self, seed: int | None) -> None:
        """Re-create the generator (Gymnasium: seed once right after construction)."""
        self.rng = np.random.default_rng(seed)

    def set_curriculum(self, hi: int, window: int, p_true_start: float) -> None:
        """Backplay frontier: resets start at length in [hi - window, hi]; hi <= 1 turns it off."""
        self.cur_hi = int(min(hi, self.route_len - 1))
        self.cur_window, self.p_true_start = int(window), float(p_true_start)

    # --- reset ---------------------------------------------------------------------------------
    def reset(self, rows: np.ndarray | None = None) -> None:
        """Start fresh episodes on ``rows`` (all rows when ``None``)."""
        rows = self.rows if rows is None else np.asarray(rows)
        self.age[rows] = 0
        lengths = np.ones(len(rows), dtype=np.int32)
        if self.cur_hi > 1:  # curriculum: most resets begin as a prefix of the demonstration
            from_route = self.rng.random(len(rows)) >= self.p_true_start
            lo = max(1, self.cur_hi - self.cur_window)
            lengths[from_route] = self.rng.integers(lo, self.cur_hi + 1, size=int(from_route.sum()))
        for row, length in zip(rows, lengths, strict=True):
            if length == 1:
                self.head[row] = self.rng.integers(self.n_cells)
                self.age[row, self.head[row]] = 1
            else:  # body = route[offset : offset+length] with tail age 1 ... head age length
                max_offset = self.route_len if self.closed else self.route_len - length + 1
                offset = self.rng.integers(max_offset)  # any rotation of a cycle is a route
                cells = self.route[(offset + np.arange(length)) % self.route_len]
                self.age[row, cells] = np.arange(1, length + 1)
                self.head[row] = cells[-1]
        self.length[rows], self.start_len[rows] = lengths, lengths
        self.idle[rows] = self.t[rows] = 0
        self._spawn_food(rows)

    def _spawn_food(self, rows: np.ndarray) -> None:
        """Uniform food over the free cells of each row: random scores, body cells masked out."""
        scores = self.rng.random((len(rows), self.n_cells))
        scores[self.age[rows] > 0] = -1.0
        self.food[rows] = scores.argmax(axis=1)

    # --- step ----------------------------------------------------------------------------------
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance every row by one action; returns ``(reward, terminated, truncated)`` arrays."""
        cfg, rows = self.cfg, self.rows
        phi_old = self._phi() if cfg.shaping_coef else None
        target = self.neigh[self.head, np.asarray(actions, dtype=np.int64)]
        oob = target < 0
        safe = np.where(oob, self.head, target)  # index-safe stand-in for wall moves
        ate = ~oob & (safe == self.food)
        self.age[(self.age > 0) & ~ate[:, None]] -= 1  # tail moves unless eating
        hit = ~oob & (self.age[rows, safe] > 0)  # body collision on the moved grid
        dead = oob | hit
        alive = ~dead
        self.length[alive & ate] += 1
        self.age[rows[alive], safe[alive]] = self.length[alive]
        self.head[alive] = safe[alive]
        won = alive & ate & (self.length == self.n_cells)  # win checked before spawning
        respawn = np.flatnonzero(alive & ate & ~won)
        if respawn.size:
            self._spawn_food(respawn)
        self.idle = np.where(ate, 0, self.idle + 1)
        self.t += 1
        terminated = dead | won
        truncated = ((self.idle > cfg.idle_cap) | (self.t >= cfg.max_steps)) & ~terminated
        reward = (
            cfg.r_food * ate + cfg.r_death * dead + cfg.r_win * won + cfg.r_step * (alive & ~won)
        ).astype(np.float32)
        if phi_old is not None:
            phi_new = np.where(terminated, 0.0, self._phi())  # Phi(terminal) = 0
            reward += (cfg.shaping_coef * (cfg.gamma * phi_new - phi_old)).astype(np.float32)
        return reward, terminated, truncated

    def _phi(self) -> np.ndarray:
        """Shaping potential: minus the normalised L1 distance from head to food."""
        dist = np.abs(self.coords[self.head] - self.coords[self.food]).sum(axis=1)
        return -dist / (self.cfg.ndim * self.cfg.size)

    # --- views ---------------------------------------------------------------------------------
    def action_masks(self) -> np.ndarray:
        """``(n, 2*ndim)`` bool: legal = inside, free once the tail has moved, and not the neck."""
        target = self.neigh[self.head]  # (n, A)
        inside = target >= 0
        age_at = np.take_along_axis(self.age, np.where(inside, target, 0), axis=1)
        neck = (self.length[:, None] == 2) & (age_at == 1)  # forbid the length-2 flip
        mask = inside & (age_at <= 1) & ~neck  # age 1 = tail, which vacates this step
        boxed = ~mask.any(axis=1)
        mask[boxed] = inside[boxed]  # no legal move: let the agent pick any in-bounds move and die
        return mask

    def observe(self, rows: np.ndarray | None = None) -> np.ndarray:
        """``(len(rows), 4*C+2)`` float32 in [0, 1]: body, time-to-vacate, head, food, scalars."""
        rows = self.rows if rows is None else np.asarray(rows)
        k = len(rows)
        body = self.age[rows] > 0
        to_vacate = self.age[rows] / np.maximum(self.length[rows], 1)[:, None]  # tail 1/L, head 1
        head = np.zeros((k, self.n_cells), dtype=np.float32)
        head[np.arange(k), self.head[rows]] = 1.0
        food = np.zeros((k, self.n_cells), dtype=np.float32)
        food[np.arange(k), self.food[rows]] = 1.0
        hunger = np.minimum(self.idle[rows] / self.cfg.idle_cap, 1.0)  # clipped: stays in the Box
        scalars = np.stack([self.length[rows] / self.n_cells, hunger], axis=1)
        return np.concatenate([body, to_vacate, head, food, scalars], axis=1).astype(np.float32)

    def infos(self, rows: np.ndarray | None = None) -> list[dict]:
        """Per-row info dicts; ``is_success``/``fill``/``start_len`` are present on every step."""
        rows = self.rows if rows is None else np.asarray(rows)
        return [
            {
                "fill": float(self.length[r]) / self.n_cells,
                "is_success": bool(self.length[r] == self.n_cells),
                "start_len": int(self.start_len[r]),
            }
            for r in rows
        ]

    def board(self, row: int = 0) -> np.ndarray:
        """Cell codes (EMPTY/BODY/HEAD/FOOD) as a ``(size,)*ndim`` array for rendering."""
        codes = (self.age[row] > 0).astype(np.int8) * BODY
        codes[self.head[row]] = HEAD
        if self.length[row] < self.n_cells:
            codes[self.food[row]] = FOOD
        return codes.reshape((self.cfg.size,) * self.cfg.ndim)
