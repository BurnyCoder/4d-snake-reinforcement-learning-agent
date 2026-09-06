"""Tests for snake4d.train and snake4d.callbacks: a tiny end-to-end training run and its files."""

import numpy as np
import pandas as pd
import pytest

from snake4d.callbacks import Backplay, FillLogger, finished_episodes
from snake4d.config import Config
from snake4d.evaluation import run as evaluate_run
from snake4d.train import run


def tiny_config(tmp_path, **overrides) -> Config:
    """A 2^2 board with a few hundred steps of PPO; every cadence fits the small budget."""
    return Config(size=2, ndim=2, device="cpu", net_width=32, n_envs=8, n_steps=32, batch_size=64,
                  total_timesteps=2048,
                  eval_every=512, ckpt_every=1024, eval_episodes=4, eval_seeds="0",
                  curriculum_min_eps=5, runs_dir=str(tmp_path), **overrides)


def test_finished_episodes_filters_infos():
    infos = [{"fill": 0.1}, {"episode": {"r": 1.0, "l": 3, "fill": 0.5, "start_len": 1,
                                         "is_success": False}}]
    assert finished_episodes(infos) == [infos[1]["episode"]]


def test_train_phase_writes_every_artifact_and_evaluates(tmp_path):
    cfg = tiny_config(tmp_path, curriculum=1)
    run_dir = run(cfg)
    names = {p.name for p in run_dir.iterdir()}
    assert {"run.log", "log.txt", "progress.csv", "config.json", "versions.json",
            "final_model.zip", "best_model.zip", "eval", "checkpoints"} <= names
    progress = pd.read_csv(run_dir / "progress.csv")
    for column in ("rollout/fill_mean", "rollout/episodes", "curriculum/frontier",
                   "eval/success_rate", "time/fps", "train/loss"):
        assert column in progress.columns, column
    assert progress["curriculum/frontier"].dropna().is_monotonic_decreasing  # eval rows are NaN
    assert progress["curriculum/frontier"].iloc[0] <= cfg.n_cells - 1
    evaluations = np.load(run_dir / "eval" / "evaluations.npz")
    assert "successes" in evaluations.files and len(evaluations["timesteps"]) >= 1
    assert any((run_dir / "checkpoints").glob("model_*_steps.zip"))
    assert "training 2^2" in (run_dir / "run.log").read_text(encoding="utf-8")
    resumed = Config(**{**cfg.__dict__, "model_path": str(run_dir / "final_model.zip")})
    summary = evaluate_run(resumed)
    assert set(summary["modes"]) == {"deterministic=True", "deterministic=False"}


def test_train_without_curriculum_has_no_frontier_column(tmp_path):
    run_dir = run(tiny_config(tmp_path, curriculum=0))
    progress = pd.read_csv(run_dir / "progress.csv")
    assert "curriculum/frontier" not in progress.columns
    assert "rollout/fill_mean_true_start" in progress.columns


def test_callbacks_are_constructible_without_a_model():
    cfg = Config(size=3, ndim=3)
    assert Backplay(cfg).hi == cfg.n_cells - 1
    assert FillLogger().fills == []


@pytest.mark.slow
def test_resume_from_a_saved_model(tmp_path):
    first = run(tiny_config(tmp_path))
    second = run(tiny_config(tmp_path, model_path="run", run_name="resume"))  # first's run name
    assert (second / "final_model.zip").exists()
    assert str(first / "best_model.zip") in (second / "run.log").read_text(encoding="utf-8")
