"""Tests for snake4d.agents: scripted policies respect masks and the route follower completes."""

import numpy as np

from snake4d.agents import POLICIES, RandomMaskedPolicy, RoutePolicy
from snake4d.config import Config
from snake4d.physics import SnakeBatch


def _rollout(policy, sim: SnakeBatch, max_steps: int):
    """Drive a SnakeBatch with a scripted policy until every row is done; return (won, steps)."""
    sim.reset()
    obs = sim.observe()
    steps = np.zeros(sim.n, dtype=int)
    done = np.zeros(sim.n, dtype=bool)
    for _ in range(max_steps):
        actions, _ = policy.predict(obs, action_masks=sim.action_masks())
        assert ((actions >= 0) & (actions < sim.cfg.n_actions)).all()
        _, terminated, truncated = sim.step(actions)
        steps[~done] += 1
        done |= terminated | truncated
        if done.all():
            break
        obs = sim.observe()
    return sim.length == sim.n_cells, steps


def test_route_policy_fills_even_boards():
    for size, ndim in [(2, 4), (4, 2), (2, 2)]:
        cfg = Config(size=size, ndim=ndim)
        sim = SnakeBatch(cfg, n=16, seed=0)
        won, steps = _rollout(RoutePolicy(cfg), sim, cfg.max_steps)
        assert won.all() and (steps <= cfg.max_steps).all()


def test_route_policy_never_leaves_the_action_range_on_odd_boards():
    cfg = Config(size=3, ndim=3)  # open Gray path: the follower must fall back at the path end
    sim = SnakeBatch(cfg, n=8, seed=1)
    policy = RoutePolicy(cfg)
    _rollout(policy, sim, 2000)
    assert policy.fallbacks > 0


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
