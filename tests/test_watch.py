"""Tests for snake4d.watch: board inference, key controls, the headless loop and GIF recording."""

import json
import os

import pygame
import pytest
from gymnasium import spaces

from snake4d.agents import RoutePolicy
from snake4d.config import Config
from snake4d.train import build_model
from snake4d.vec_env import make_env

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame for tests

from snake4d import watch  # noqa: E402  (must follow the SDL driver setting)


class _Spaces:
    """Stand-in for a loaded MaskablePPO: only the two spaces ``board_of`` reads."""

    def __init__(self, n_inputs: int, n_actions: int) -> None:
        self.observation_space = spaces.Box(0.0, 1.0, shape=(n_inputs,))
        self.action_space = spaces.Discrete(n_actions)


def test_board_is_inferred_from_the_checkpoint_spaces():
    cfg = watch.board_of(_Spaces(4 * 16 + 2, 8), Config())  # a 2^4 checkpoint, 4^4 defaults
    assert (cfg.size, cfg.ndim, cfg.n_cells) == (2, 4, 16)
    assert watch.board_of(_Spaces(4 * 27 + 2, 6), Config()).size == 3  # 3^3
    with pytest.raises(ValueError, match="no board"):
        watch.board_of(_Spaces(67, 8), Config())
    scripted = Config(size=3, ndim=2)
    assert watch.board_of(RoutePolicy(scripted), scripted) is scripted


def test_keys_pause_step_speed_and_restart():
    controls = watch.Controls(speed=8)
    watch.apply_watch_key(pygame.K_SPACE, controls)
    assert controls.paused
    watch.apply_watch_key(pygame.K_n, controls)
    assert controls.pending == 1
    for key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
        watch.apply_watch_key(key, controls)
    assert controls.speed == 64
    for _ in range(20):
        watch.apply_watch_key(pygame.K_MINUS, controls)
    assert controls.speed == 1  # clamped at one move per second
    for _ in range(20):
        watch.apply_watch_key(pygame.K_KP_MINUS, controls)
        watch.apply_watch_key(pygame.K_KP_PLUS, controls)
        watch.apply_watch_key(pygame.K_KP_PLUS, controls)
    assert controls.speed == watch.MAX_SPEED
    watch.apply_watch_key(pygame.K_r, controls)
    assert controls.restart


def test_headless_route_policy_plays_logs_and_records_a_gif(tmp_path):
    cfg = Config(size=2, ndim=2, policy="route", runs_dir=str(tmp_path), watch_speed=200,
                 watch_gif=1)
    session = watch.run(cfg, max_frames=80)  # the route needs 3 moves on 4 cells
    assert session.done and session.status == "WON" and session.game == 1
    (run_dir,) = tmp_path.iterdir()
    log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "watching route on 2^2" in log and "game 1: WON after" in log
    assert "recorded game 1" in log
    image = pytest.importorskip("PIL.Image").open(run_dir / "game.gif")
    assert image.n_frames == session.steps + 1  # the start position plus one frame per move


def test_gif_recording_stops_at_the_frame_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(watch, "GIF_MAX_FRAMES", 2)
    cfg = Config(size=2, ndim=2, policy="route", runs_dir=str(tmp_path), watch_speed=200,
                 watch_gif=1)
    session = watch.run(cfg, max_frames=80)
    assert session.steps > 2  # the game went on after the recording stopped
    (run_dir,) = tmp_path.iterdir()
    assert "recorded game 1 (2 frames, capped)" in (run_dir / "run.log").read_text(encoding="utf-8")
    assert pytest.importorskip("PIL.Image").open(run_dir / "game.gif").n_frames == 2


def test_saved_model_drives_the_window_on_its_own_board(tmp_path):
    small = Config(size=2, ndim=2, device="cpu", net_width=16, n_envs=4, n_steps=16, batch_size=32)
    build_model(small, make_env(small, 4, 0)).save(tmp_path / "model.zip")  # untrained is fine
    cfg = Config(model_path=str(tmp_path / "model.zip"), runs_dir=str(tmp_path / "runs"),
                 device="cpu", watch_speed=500)  # 4^4 defaults; the checkpoint says 2^2
    watch.run(cfg, max_frames=10)
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert "on 2^2 (4 cells)" in (run_dir / "run.log").read_text(encoding="utf-8")
    assert json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["size"] == 2
