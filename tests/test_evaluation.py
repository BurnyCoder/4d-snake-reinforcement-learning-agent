"""Tests for snake4d.evaluation: masked evaluation of scripted policies and result files."""

import json

import pytest
from sb3_contrib.common.maskable.utils import is_masking_supported

from snake4d.agents import RandomMaskedPolicy, RoutePolicy
from snake4d.config import Config
from snake4d.evaluation import evaluate, make_eval_env, run, summarize


def test_eval_env_supports_masking_and_seeds_reset():
    cfg = Config(size=2, ndim=2)
    env = make_eval_env(cfg, n_envs=3, seed=7)
    assert is_masking_supported(env)
    first = env.reset()
    assert first.shape == (3, 4 * 4 + 2)
    again = make_eval_env(cfg, n_envs=3, seed=7).reset()
    assert (first == again).all()


def test_route_policy_completes_2x4_under_the_full_protocol():
    cfg = Config(size=2, ndim=4, eval_episodes=20)
    summary, records = evaluate(RoutePolicy(cfg), cfg, seeds=(0, 1))
    mode = summary["modes"]["deterministic=True"]
    assert mode["success_rate_mean"] == 1.0 and mode["success_rate_std"] == 0.0
    assert len(records) == 40 and all(r["fill"] == 1.0 for r in records)
    assert mode["steps_to_complete_mean"] <= cfg.max_steps
    assert mode["win_within"]["C2/2"] == 1.0


def test_random_policy_rarely_completes_but_reports_fill():
    cfg = Config(size=3, ndim=2, eval_episodes=10)
    summary, _ = evaluate(RandomMaskedPolicy(cfg), cfg, seeds=(0,))
    mode = summary["modes"]["deterministic=True"]
    assert 0.0 <= mode["fill_mean"] <= 1.0 and mode["episodes"] == 10


def test_summarize_handles_no_wins():
    cfg = Config(size=2, ndim=2)
    stats = summarize([{"success": 0, "fill": 0.5, "length": 3, "return": -1.0}], cfg)
    assert stats["steps_to_complete_mean"] is None and stats["win_within"]["C"] == 0.0


def test_run_phase_writes_summary_and_csv(tmp_path):
    cfg = Config(size=2, ndim=2, eval_episodes=5, eval_seeds="0", runs_dir=str(tmp_path),
                 policy="route")
    summary = run(cfg)
    (run_dir,) = tmp_path.iterdir()
    assert (run_dir / "summary.json").exists() and (run_dir / "eval_episodes.csv").exists()
    assert json.loads((run_dir / "summary.json").read_text())["board"] == "2^2"
    assert summary["modes"]["deterministic=True"]["success_rate_mean"] == 1.0
    assert "evaluating route" in (run_dir / "log.txt").read_text(encoding="utf-8")


def test_unknown_policy_name_fails_fast(tmp_path):
    with pytest.raises(KeyError):
        run(Config(size=2, ndim=2, policy="nope", runs_dir=str(tmp_path)))
