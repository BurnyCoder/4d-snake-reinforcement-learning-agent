"""Tests for snake4d.physics: batched rules vs a plain deque oracle, plus targeted rule checks."""

from collections import deque

import numpy as np
import pytest

from snake4d.config import Config
from snake4d.grid import cell_of, neighbour_table
from snake4d.hamilton import route_actions
from snake4d.physics import FOOD, HEAD, SnakeBatch


class DequeOracle:
    """Classic single snake as a deque of cells (head last); food is copied from the batched sim."""

    def __init__(self, size: int, ndim: int, head: int, food: int) -> None:
        self.n_cells = size**ndim
        self.neigh = neighbour_table(size, ndim)
        self.body, self.food, self.dead = deque([head]), food, False

    def step(self, action: int) -> bool:
        nxt = int(self.neigh[self.body[-1], action])
        if nxt < 0:
            self.dead = True
            return False
        ate = nxt == self.food
        if not ate:
            self.body.popleft()  # tail moves first, so following the tail is legal
        if nxt in self.body:
            self.dead = True
            return False
        self.body.append(nxt)
        return ate

    def ages(self) -> np.ndarray:
        ages = np.zeros(self.n_cells, dtype=np.int16)
        for k, cell in enumerate(self.body):
            ages[cell] = k + 1
        return ages


@pytest.mark.parametrize(("size", "ndim"), [(4, 2), (3, 3), (3, 4)])
def test_batched_physics_matches_the_deque_oracle(size, ndim):
    cfg = Config(size=size, ndim=ndim)
    sim = SnakeBatch(cfg, n=6, seed=1)
    sim.reset()
    rng = np.random.default_rng(2)
    oracles = [DequeOracle(size, ndim, int(sim.head[i]), int(sim.food[i])) for i in range(6)]
    for _ in range(600):
        actions = np.array([rng.choice(np.flatnonzero(m)) for m in sim.action_masks()])
        reward, terminated, truncated = sim.step(actions)
        for i, oracle in enumerate(oracles):
            oracle.step(int(actions[i]))
            if oracle.dead:
                assert terminated[i] and reward[i] == cfg.r_death
            else:
                np.testing.assert_array_equal(sim.age[i], oracle.ages())
                assert sim.head[i] == oracle.body[-1] and sim.length[i] == len(oracle.body)
                if len(oracle.body) == cfg.n_cells:
                    assert terminated[i] and reward[i] == cfg.r_food + cfg.r_win
                else:
                    assert sim.age[i, sim.food[i]] == 0  # food always on a free cell
                    oracle.food = int(sim.food[i])
            if terminated[i] or truncated[i]:
                sim.reset(np.array([i]))
                oracles[i] = DequeOracle(size, ndim, int(sim.head[i]), int(sim.food[i]))


def test_following_the_cycle_fills_the_board_and_wins_before_spawning():
    cfg = Config(size=2, ndim=2)  # 4 cells
    sim = SnakeBatch(cfg, n=1, seed=0)
    sim.reset()
    actions = route_actions(sim.route, sim.closed, sim.neigh)
    for _ in range(cfg.max_steps):
        reward, terminated, truncated = sim.step(actions[sim.head])
        assert not truncated[0]
        if terminated[0]:
            break
    assert sim.length[0] == cfg.n_cells and reward[0] == cfg.r_food + cfg.r_win
    assert sim.infos()[0]["is_success"] and sim.infos()[0]["fill"] == 1.0
    assert FOOD not in sim.board(0) and (sim.board(0) == HEAD).sum() == 1


def _oscillate(sim: SnakeBatch, steps: int):
    """Move a length-1 snake back and forth between two non-food cells."""
    head = int(sim.head[0])
    action = next(a for a in range(sim.cfg.n_actions)
                  if sim.neigh[head, a] >= 0 and sim.neigh[head, a] != sim.food[0])
    results = []
    for k in range(steps):
        results.append(sim.step(np.array([action ^ (k % 2)])))
    return results


def test_starvation_is_a_truncation_with_zero_penalty():
    cfg = Config(size=2, ndim=2, idle_mult=2)  # idle cap = 8 steps without food
    sim = SnakeBatch(cfg, n=1, seed=3)
    sim.reset()
    results = _oscillate(sim, cfg.idle_cap + 1)
    for reward, terminated, truncated in results[:-1]:
        assert not terminated[0] and not truncated[0] and reward[0] == cfg.r_step
    reward, terminated, truncated = results[-1]
    assert truncated[0] and not terminated[0] and reward[0] == cfg.r_step


def test_absolute_step_cap_truncates():
    cfg = Config(size=2, ndim=2, idle_mult=100)  # idle cap 400 > max_steps 16
    sim = SnakeBatch(cfg, n=1, seed=3)
    sim.reset()
    results = _oscillate(sim, cfg.max_steps)
    assert not any(t[0] for _, _, t in results[:-1]) and results[-1][2][0]


def test_wall_and_self_collision_are_terminal_deaths():
    cfg = Config(size=3, ndim=2)
    sim = SnakeBatch(cfg, n=1, seed=0)
    sim.reset()
    _place(sim, [(0, 0)], food=(2, 2))
    reward, terminated, _ = sim.step(np.array([1]))  # x-1 from x=0 is a wall
    assert terminated[0] and reward[0] == cfg.r_death
    sim.reset()
    _place(sim, [(0, 0), (0, 1), (1, 1), (1, 0)], food=(2, 2))  # head (1,0), neck (1,1)
    reward, terminated, _ = sim.step(np.array([2]))  # y+1 into the neck
    assert terminated[0] and reward[0] == cfg.r_death


def _place(sim: SnakeBatch, cells_tail_to_head, food):
    """Write an explicit snake (coordinate tuples, tail first) and food into row 0."""
    size, ndim = sim.cfg.size, sim.cfg.ndim
    sim.age[0] = 0
    for k, coord in enumerate(cells_tail_to_head):
        sim.age[0, cell_of(coord, size, ndim)] = k + 1
    sim.head[0] = cell_of(cells_tail_to_head[-1], size, ndim)
    sim.length[0] = len(cells_tail_to_head)
    sim.food[0] = cell_of(food, size, ndim)


def test_action_masks_rules_on_a_3x3_board():
    cfg = Config(size=3, ndim=2)  # actions: 0 x+1, 1 x-1, 2 y+1, 3 y-1
    sim = SnakeBatch(cfg, n=1, seed=0)
    sim.reset()
    _place(sim, [(0, 0), (0, 1), (1, 1), (1, 0)], food=(2, 2))
    assert sim.action_masks()[0].tolist() == [True, True, False, False]  # tail-follow legal
    _place(sim, [(0, 0), (1, 0)], food=(2, 2))
    assert sim.action_masks()[0].tolist() == [True, False, True, False]  # length-2 flip illegal
    _place(sim, [(1, 2), (0, 2), (0, 1), (1, 1), (1, 0), (0, 0)], food=(2, 2))
    assert sim.action_masks()[0].tolist() == [True, False, True, False]  # boxed -> in-bounds
    _place(sim, [(1, 1)], food=(2, 2))
    assert sim.action_masks()[0].all()  # length 1: every in-bounds move is legal


def test_food_is_uniform_over_free_cells():
    cfg = Config(size=2, ndim=2)
    sim = SnakeBatch(cfg, n=4000, seed=5)
    sim.reset()
    counts = np.bincount(sim.food, minlength=4)
    assert (sim.age[np.arange(4000), sim.food] == 0).all()
    assert counts.min() > 800  # ~1000 each; the head cell is excluded per row


def test_curriculum_resets_lay_the_snake_along_the_route():
    cfg = Config(size=4, ndim=4, p_true_start=0.0)
    sim = SnakeBatch(cfg, n=64, seed=7)
    sim.set_curriculum(hi=200, window=8, p_true_start=0.0)
    sim.reset()
    assert ((sim.length >= 192) & (sim.length <= 200)).all() and (sim.start_len == sim.length).all()
    order = np.argsort(sim.route)  # position of each cell on the route
    for i in range(64):
        cells = np.flatnonzero(sim.age[i])
        ages = sim.age[i, cells]
        positions = order[cells[np.argsort(ages)]]  # route positions from tail to head
        steps = np.diff(positions) % cfg.n_cells
        assert (steps == 1).all() and sim.head[i] == cells[np.argmax(ages)]
        assert sim.age[i, sim.food[i]] == 0
    sim.set_curriculum(hi=1, window=8, p_true_start=0.0)
    sim.reset()
    assert (sim.length == 1).all()


def test_shaping_is_potential_based_and_zero_at_terminals():
    cfg = Config(size=3, ndim=2, shaping_coef=1.0, r_food=0.0, r_step=0.0)
    sim = SnakeBatch(cfg, n=1, seed=0)
    sim.reset()
    _place(sim, [(0, 0)], food=(2, 2))
    phi0 = sim._phi()[0]
    reward, _, _ = sim.step(np.array([0]))  # (0,0) -> (1,0): one step closer
    assert reward[0] == pytest.approx(cfg.gamma * sim._phi()[0] - phi0)
    _place(sim, [(0, 0)], food=(2, 2))
    phi0 = sim._phi()[0]
    reward, terminated, _ = sim.step(np.array([1]))  # wall: Phi(terminal) = 0
    assert terminated[0] and reward[0] == pytest.approx(cfg.r_death - phi0)
