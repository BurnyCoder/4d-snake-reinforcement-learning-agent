"""Tests for snake4d.env: Gymnasium/SB3 env checkers and the single-env contract."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from snake4d.config import Config
from snake4d.env import SnakeEnv


@pytest.mark.parametrize(("size", "ndim"), [(3, 2), (3, 3), (2, 4), (3, 4)])
def test_env_passes_both_checkers(size, ndim):
    env = SnakeEnv(Config(size=size, ndim=ndim))
    sb3_check_env(env, warn=True, skip_render_check=True)
    gym_check_env(env, skip_render_check=True)


def test_reset_is_deterministic_for_a_seed():
    env = SnakeEnv(Config(size=3, ndim=4))
    obs_a, _ = env.reset(seed=42)
    obs_b, _ = env.reset(seed=42)
    np.testing.assert_array_equal(obs_a, obs_b)
    assert obs_a.dtype == np.float32 and obs_a.shape == (4 * 81 + 2,)


def test_step_contract_and_info_keys():
    env = SnakeEnv(Config(size=3, ndim=2))
    obs, info = env.reset(seed=0)
    assert {"fill", "is_success", "start_len"} <= info.keys()
    action = int(np.flatnonzero(env.action_masks())[0])
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs) and isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert env.action_masks().shape == (4,)


def test_observation_stays_in_bounds_when_starving():
    env = SnakeEnv(Config(size=2, ndim=2, idle_mult=2))
    env.reset(seed=1)
    head, food = int(env.sim.head[0]), int(env.sim.food[0])
    action = next(a for a in range(4) if env.sim.neigh[head, a] not in (-1, food))
    for k in range(env.cfg.idle_cap + 1):
        obs, _, terminated, truncated, _ = env.step(action ^ (k % 2))
        assert env.observation_space.contains(obs)
    assert truncated and not terminated


def test_render_modes():
    board_env = SnakeEnv(Config(size=3, ndim=4), render_mode="ansi")
    board_env.reset(seed=0)
    text = board_env.render()
    assert "@" in text and "*" in text
    rgb_env = SnakeEnv(Config(size=3, ndim=4), render_mode="rgb_array")
    rgb_env.reset(seed=0)
    assert rgb_env.render().shape == (9, 9, 3)


def test_set_curriculum_reaches_the_simulator():
    env = SnakeEnv(Config(size=3, ndim=3))
    env.set_curriculum(hi=20, window=4, p_true_start=0.0)
    env.reset(seed=0)
    assert 16 <= env.sim.length[0] <= 20
