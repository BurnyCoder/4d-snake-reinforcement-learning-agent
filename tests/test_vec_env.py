"""Tests for snake4d.vec_env: one test per rule of the custom-VecEnv contract, plus PPO smoke."""

import numpy as np
import pytest
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

from snake4d.config import Config
from snake4d.env import SnakeEnv
from snake4d.grid import cell_of
from snake4d.vec_env import INFO_KEYS, SnakeVecEnv, make_env


def test_instantiates_with_render_mode_none_and_spaces():
    env = SnakeVecEnv(Config(size=3, ndim=4), n_envs=4)  # TypeError if a method is missing
    assert env.render_mode is None and env.num_envs == 4
    assert env.observation_space.shape == (4 * 81 + 2,) and env.action_space.n == 8


def test_masking_is_supported_through_vecmonitor():
    env = make_env(Config(size=3, ndim=2), n_envs=5)
    assert isinstance(env, VecMonitor) and is_masking_supported(env)
    assert not env.has_attr("does_not_exist")
    env.reset()
    masks = get_action_masks(env)
    assert masks.shape == (5, 4) and masks.dtype == bool
    np.testing.assert_array_equal(masks, env.venv.sim.action_masks())


def test_set_curriculum_via_env_method_on_both_env_types():
    batched = make_env(Config(size=3, ndim=3), n_envs=3)
    batched.env_method("set_curriculum", 20, 4, 0.0)
    assert batched.venv.sim.cur_hi == 20
    single = make_vec_env(SnakeEnv, n_envs=2, env_kwargs={"cfg": Config(size=3, ndim=3)},
                          monitor_kwargs={"info_keywords": INFO_KEYS})
    single.env_method("set_curriculum", 20, 4, 0.0)
    assert all(e.unwrapped.sim.cur_hi == 20 for e in single.envs)


def test_step_wait_auto_resets_with_terminal_observation():
    cfg = Config(size=3, ndim=2)
    env = SnakeVecEnv(cfg, n_envs=2, seed=0)
    env.reset()
    sim = env.sim
    sim.age[:] = 0
    sim.age[0, cell_of((0, 0), 3, 2)] = 1  # row 0 at the corner (0,0)
    sim.age[1, cell_of((1, 1), 3, 2)] = 1  # row 1 in the middle
    sim.head[:] = [cell_of((0, 0), 3, 2), cell_of((1, 1), 3, 2)]
    sim.length[:] = 1
    sim.food[:] = cell_of((2, 2), 3, 2)
    obs, reward, done, infos = env.step(np.array([1, 0]))  # row 0 walks into the x=-1 wall
    assert done[0] and not done[1] and reward[0] == cfg.r_death
    terminal = infos[0]["terminal_observation"]
    assert terminal.shape == obs.shape[1:] and not np.shares_memory(terminal, obs)
    assert terminal[2 * 9 : 3 * 9].argmax() == cell_of((0, 0), 3, 2)  # the pre-reset head
    assert infos[0]["TimeLimit.truncated"] is False and infos[0]["is_success"] is False
    assert sim.length[0] == 1 and sim.t[0] == 0  # row 0 was reset
    assert obs[0][2 * 9 : 3 * 9].argmax() == sim.head[0]  # fresh observation after the reset


def test_truncation_flag_and_vecmonitor_episode_info():
    cfg = Config(size=2, ndim=2, idle_mult=1)  # starve after 4 idle steps
    env = make_env(cfg, n_envs=1, seed=0)
    env.reset()
    sim = env.venv.sim
    head, food = int(sim.head[0]), int(sim.food[0])
    action = next(a for a in range(4) if sim.neigh[head, a] not in (-1, food))
    for k in range(cfg.idle_cap + 1):
        obs, reward, done, infos = env.step(np.array([action ^ (k % 2)]))
    assert done[0] and infos[0]["TimeLimit.truncated"] is True
    episode = infos[0]["episode"]
    assert episode["l"] == cfg.idle_cap + 1 and set(INFO_KEYS) <= episode.keys()
    assert episode["is_success"] is False and episode["fill"] == 0.25


def test_seed_is_consumed_at_reset_like_dummy_vec_env():
    cfg = Config(size=3, ndim=3)
    a, b = SnakeVecEnv(cfg, 4), SnakeVecEnv(cfg, 4)
    a.seed(11)
    b.seed(11)
    np.testing.assert_array_equal(a.reset(), b.reset())
    assert a._seeds == [None] * 4


def test_random_legal_rollout_keeps_invariants():
    cfg = Config(size=2, ndim=4)  # 16 cells: random masked play ends episodes quickly
    env = make_env(cfg, n_envs=32, seed=1)
    obs = env.reset()
    rng = np.random.default_rng(0)
    episodes = 0
    for _ in range(300):
        masks = get_action_masks(env)
        actions = np.where(masks, rng.random(masks.shape), -1.0).argmax(axis=1)
        obs, reward, done, infos = env.step(actions)
        assert obs.dtype == np.float32 and obs.min() >= 0.0 and obs.max() <= 1.0
        episodes += int(done.sum())
        assert all({"fill", "is_success", "start_len"} <= info.keys() for info in infos)
    assert episodes > 0


@pytest.mark.parametrize("device", ["cpu"])
def test_maskable_ppo_learns_a_few_rollouts_on_the_batched_env(device):
    from snake4d.train import build_model

    cfg = Config(size=2, ndim=2, n_envs=8, n_steps=32, batch_size=64, eval_every=256)
    env = make_env(cfg, cfg.n_envs, seed=0)
    model = build_model(cfg, env, device=device)
    model.learn(total_timesteps=1024)
    assert model.num_timesteps >= 1024
    assert len(model.ep_info_buffer) > 0 and len(model.ep_success_buffer) > 0
    assert "fill" in model.ep_info_buffer[0]
