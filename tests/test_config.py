"""Tests for snake4d.config: defaults, environment overrides, derived values and guards."""

import dataclasses
from pathlib import Path

import pytest

from snake4d.config import Config


def test_defaults_are_the_headline_board():
    cfg = Config()
    assert (cfg.size, cfg.ndim, cfg.n_cells, cfg.n_actions) == (4, 4, 256, 8)
    assert cfg.idle_cap == 4 * 256 and cfg.max_steps == 256 * 256
    assert cfg.curriculum_step == 4 and cfg.seeds == (0, 1, 2)


def test_env_example_documents_every_config_field():
    text = (Path(__file__).resolve().parents[1] / ".env.example").read_text(encoding="utf-8")
    missing = [f.name for f in dataclasses.fields(Config) if f"SNAKE_{f.name.upper()}=" not in text]
    assert missing == []


def test_field_types_are_castable_from_strings():
    # int/float/str only: f.type(value) must be an exact cast (bool("False") would be True)
    assert {f.type for f in dataclasses.fields(Config)} <= {int, float, str}


def test_env_and_set_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("SNAKE_SIZE", "3")
    env_file = tmp_path / "exp.env"
    env_file.write_text("SNAKE_NDIM=3\nSNAKE_GAMMA=0.95\n", encoding="utf-8")
    cfg = Config.from_env(str(env_file), ("run_name=abc", "SNAKE_N_ENVS=8", "batch_size=64"))
    assert (cfg.size, cfg.ndim, cfg.gamma, cfg.run_name, cfg.n_envs) == (3, 3, 0.95, "abc", 8)
    assert isinstance(cfg.gamma, float) and isinstance(cfg.n_envs, int)


def test_unknown_override_field_is_an_error():
    with pytest.raises(ValueError, match="unknown config field"):
        Config.from_env(None, ("sizee=3",))


def test_missing_env_file_is_an_error():
    with pytest.raises(ValueError, match="env file not found"):
        Config.from_env("does/not/exist.env")


def test_step_penalty_vs_gamma_guard():
    with pytest.raises(ValueError, match="r_death"):
        Config(gamma=0.999, r_step=-0.001)  # 0.001 / 0.001 == 1.0 == |r_death| -> rejected
    Config(gamma=0.999, r_step=-0.0005)  # the sweep arm must lower r_step


def test_rollout_and_eval_guards():
    with pytest.raises(ValueError, match="batch_size"):
        Config(n_envs=8, n_steps=64, batch_size=100)
    with pytest.raises(ValueError, match="eval_every"):
        Config(n_envs=1024, n_steps=64, batch_size=2048, eval_every=1000)


def test_to_json_roundtrip(tmp_path):
    path = tmp_path / "config.json"
    Config(size=3).to_json(path)
    assert '"size": 3' in path.read_text(encoding="utf-8")
