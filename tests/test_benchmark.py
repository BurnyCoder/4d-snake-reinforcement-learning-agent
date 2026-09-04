"""Tests for snake4d.benchmark: env-only timing, a tiny PPO grid and the markdown/json outputs."""

import json
from pathlib import Path

from snake4d.benchmark import env_steps_per_second, recommend, run
from snake4d.config import Config


def test_env_only_throughput_is_positive():
    assert env_steps_per_second(Config(size=2, ndim=2), n_envs=8, steps=20) > 0


def test_recommend_picks_the_fastest_row_and_a_sane_eval_cadence():
    results = [
        {"env": "dummy", "n_envs": 16, "device": "cpu", "threads": 1, "n_steps": 512,
         "batch_size": 2048, "fps": 3000},
        {"env": "batched", "n_envs": 1024, "device": "cuda", "threads": 1, "n_steps": 64,
         "batch_size": 2048, "fps": 90000},
        {"env": "subproc", "n_envs": 16, "device": "cpu", "threads": 1, "error": "boom"},
    ]
    rec = recommend(results)
    assert rec["SNAKE_N_ENVS"] == 1024 and rec["SNAKE_DEVICE"] == "cuda"
    assert rec["SNAKE_EVAL_EVERY"] % (64 * 1024) == 0 and rec["SNAKE_EVAL_EVERY"] >= 64 * 1024
    assert rec["SNAKE_CKPT_EVERY"] == 4 * rec["SNAKE_EVAL_EVERY"]


def test_run_writes_json_and_markdown(tmp_path):
    cfg = Config(size=2, ndim=2, n_steps=16, batch_size=64, runs_dir=str(tmp_path / "runs"))
    docs = tmp_path / "docs" / "benchmark.md"
    rec = run(cfg, rows=[{"env": "batched", "n_envs": 8}, {"env": "dummy", "n_envs": 2}],
              devices=["cpu"], threads=[1], timesteps=128, docs_path=docs)
    assert rec["SNAKE_N_ENVS"] in (8, 2)
    (run_dir,) = Path(tmp_path / "runs").iterdir()
    data = json.loads((run_dir / "benchmark.json").read_text(encoding="utf-8"))
    assert len(data["results"]) == 2 and all("fps" in r for r in data["results"])
    text = docs.read_text(encoding="utf-8")
    assert "| batched | 8 | cpu | 1 |" in text and "SNAKE_N_ENVS" in text
