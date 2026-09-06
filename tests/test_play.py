"""Tests for snake4d.play: key mapping, the neck rule, key handling and the headless loop."""

import os

import numpy as np
import pygame

from snake4d.config import Config
from snake4d.env import SnakeEnv
from snake4d.grid import cell_of

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame for tests

from snake4d import play  # noqa: E402  (must follow the SDL driver setting)


def _env_with_snake(cells_tail_to_head, food, size=3, ndim=2) -> SnakeEnv:
    env = SnakeEnv(Config(size=size, ndim=ndim))
    env.reset(seed=0)
    sim = env.sim
    sim.age[0] = 0
    for k, coord in enumerate(cells_tail_to_head):
        sim.age[0, cell_of(coord, size, ndim)] = k + 1
    sim.head[0] = cell_of(cells_tail_to_head[-1], size, ndim)
    sim.length[0] = len(cells_tail_to_head)
    sim.food[0] = cell_of(food, size, ndim)
    return env


def test_keys_cover_all_eight_4d_actions():
    assert sorted(play.KEYS.values()) == list(range(8))
    assert play.KEYS[pygame.K_d] == 2 and play.KEYS[pygame.K_k] == 4 and play.KEYS[pygame.K_l] == 6


def test_neck_key_is_ignored_and_other_keys_step():
    env = _env_with_snake([(0, 0), (1, 0)], food=(2, 2))  # head (1,0), neck (0,0) via W (x-1)
    session = play.Session()
    play.apply_key(env, pygame.K_w, session)  # reversal into the neck: ignored
    assert session.steps == 0 and env.sim.head[0] == cell_of((1, 0), 3, 2)
    play.apply_key(env, pygame.K_d, session)  # y+1 -> (1,1)
    assert session.steps == 1 and env.sim.head[0] == cell_of((1, 1), 3, 2)
    play.apply_key(env, pygame.K_l, session)  # w-axis key does not exist in 2D: ignored
    assert session.steps == 1


def test_death_then_any_key_restarts():
    env = _env_with_snake([(0, 0)], food=(2, 2))
    session = play.Session()
    play.apply_key(env, pygame.K_w, session)  # x-1 from x=0: wall
    assert session.done and session.status == "dead"
    play.apply_key(env, pygame.K_d, session)
    assert not session.done and session.steps == 0 and env.sim.length[0] == 1


def test_window_scales_cells_labels_the_tiles_and_fits_two_hud_lines():
    cfg = Config(size=2, ndim=4)
    window = play.Window(cfg, "test")
    assert window.cell == play.WINDOW_PX // 4  # the 2^4 montage is 4 cells wide
    width, height = window.screen.get_size()
    assert width > play.WINDOW_PX and height > width  # labels margin; HUD below the board
    env = SnakeEnv(cfg)
    env.reset(seed=0)
    window.draw(env, (play.hud_text(env, play.Session()), play.PLAY_HELP))
    frame = window.frame()
    assert frame.shape == (height, width, 3) and frame.dtype == np.uint8
    assert frame.max() > 200  # the head is drawn (bright yellow), so the montage is on screen
    assert window.font.size(play.PLAY_HELP)[0] < width  # the help line fits on one line
    window.close()


def test_headless_window_loop(tmp_path):
    cfg = Config(size=3, ndim=4, runs_dir=str(tmp_path), seed=0)
    pygame.init()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    session = play.run(cfg, max_frames=5)
    assert session.steps == 0
    assert "play 3^4" in (next(tmp_path.iterdir()) / "run.log").read_text(encoding="utf-8")
