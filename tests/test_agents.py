"""Tests for snake4d.agents: scripted policies respect masks and the route follower completes."""

import numpy as np
import pytest

from snake4d.agents import POLICIES, RandomMaskedPolicy, RoutePolicy
from snake4d.config import Config
from snake4d.physics import SnakeBatch


def _rollout(policy, sim: SnakeBatch, max_steps: int):
    """Play one episode per row (finished rows are reset); return (won, finished) per row."""
    sim.reset()
    obs = sim.observe()
    won = np.zeros(sim.n, dtype=bool)
    finished = np.zeros(sim.n, dtype=bool)
    for _ in range(max_steps):
        actions, _ = policy.predict(obs, action_masks=sim.action_masks())
        assert ((actions >= 0) & (actions < sim.cfg.n_actions)).all()
        _, terminated, truncated = sim.step(actions)
        done = terminated | truncated
        first = done & ~finished
        won[first] = sim.length[first] == sim.n_cells
        finished |= done
        if finished.all():
            break
        if done.any():
            sim.reset(np.flatnonzero(done))
        obs = sim.observe()
    return won, finished


@pytest.mark.parametrize(("size", "ndim"), [(2, 4), (4, 2), (2, 2), (3, 2), (3, 3), (5, 2)])
def test_route_policy_fills_even_and_odd_boards(size, ndim):
    cfg = Config(size=size, ndim=ndim)
    sim = SnakeBatch(cfg, n=16, seed=0)
    policy = RoutePolicy(cfg)
    won, finished = _rollout(policy, sim, cfg.max_steps)
    assert finished.all() and won.all() and policy.fallbacks == 0


def test_route_policy_fills_the_3x4_debug_board():
    cfg = Config(size=3, ndim=4)  # 81 cells, cycle over 80 + guarded corner detour
    sim = SnakeBatch(cfg, n=8, seed=1)
    policy = RoutePolicy(cfg)
    won, finished = _rollout(policy, sim, cfg.max_steps)
    assert finished.all() and won.all() and policy.corner == 0 and policy.fallbacks == 0


def test_random_policy_only_picks_legal_actions():
    cfg = Config(size=3, ndim=2)
    policy = RandomMaskedPolicy(cfg, seed=0)
    obs = np.zeros((5, cfg.n_cells * 4 + 2), dtype=np.float32)
    masks = np.zeros((5, 4), dtype=bool)
    masks[:, 2] = True
    actions, _ = policy.predict(obs, action_masks=masks)
    assert (actions == 2).all()


def test_policy_registry_and_seed_hook():
    assert set(POLICIES) == {"route", "random"}
    policy = POLICIES["random"](Config(size=2, ndim=2))
    policy.set_random_seed(3)
    a = policy.predict(np.zeros(18), action_masks=np.ones(4, bool))[0]
    policy.set_random_seed(3)
    assert (a == policy.predict(np.zeros(18), action_masks=np.ones(4, bool))[0]).all()
