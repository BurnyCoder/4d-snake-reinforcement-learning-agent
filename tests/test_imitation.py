"""Tests for snake4d.imitation: expert data collection and behaviour cloning on a tiny board."""

from snake4d.agents import RoutePolicy
from snake4d.config import Config
from snake4d.evaluation import evaluate
from snake4d.imitation import behaviour_clone, collect, run
from snake4d.train import build_model
from snake4d.vec_env import make_env


def test_collect_returns_expert_actions_on_all_lengths():
    cfg = Config(size=2, ndim=2)
    obs, actions, masks = collect(cfg, n_envs=16, steps=8, seed=0)
    assert obs.shape == (128, 4 * 4 + 2) and actions.shape == (128,) and masks.shape == (128, 4)
    assert masks[range(128), actions].all()  # the expert only picks legal actions
    lengths = (obs[:, :4] > 0).sum(axis=1)
    assert lengths.min() == 1 and lengths.max() >= 3  # curriculum starts cover several lengths


def test_behaviour_clone_reaches_high_accuracy_and_completes(tmp_path):
    cfg = Config(size=2, ndim=2, device="cpu", net_width=64, n_envs=64, n_steps=16, batch_size=256,
                 eval_episodes=10, eval_seeds="0", runs_dir=str(tmp_path))
    obs, actions, masks = collect(cfg, cfg.n_envs, cfg.n_steps, seed=0)
    model = build_model(cfg, make_env(cfg, cfg.n_envs, 0))
    accuracy = behaviour_clone(model, obs, actions, masks, epochs=30, batch_size=cfg.batch_size,
                               lr=cfg.bc_lr)
    assert accuracy > 0.9
    summary, _ = evaluate(model, cfg, seeds=(0,))
    assert summary["modes"]["deterministic=True"]["success_rate_mean"] > 0.5


def test_imitate_phase_saves_a_model(tmp_path):
    cfg = Config(size=2, ndim=2, device="cpu", net_width=32, n_envs=16, n_steps=8, batch_size=64,
                 bc_epochs=2, runs_dir=str(tmp_path))
    run_dir = run(cfg)
    assert (run_dir / "bc_model.zip").exists()
    assert "expert-action accuracy" in (run_dir / "run.log").read_text(encoding="utf-8")
    assert RoutePolicy(cfg).corner == -1  # even board: the expert is the plain cycle follower
