"""Watch phase: a saved MaskablePPO checkpoint (or a scripted policy) plays in the pygame window.

Global context: the "show me" half of evaluation.  ``evaluate`` reports success rates over 300
games; this phase plays single games of the same policy in ``play.Window`` so a human can see how
the network moves.  It is the rl-baselines3-zoo ``enjoy`` loop - predict, step, render
(https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/enjoy.py) - with sb3-contrib's
masked ``model.predict(obs, action_masks=action_masks)``
(https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html).

Local notes:
* The board is inferred from the checkpoint's spaces (``4C + 2`` inputs, ``2 * ndim`` actions), so
  no ``--set size/ndim`` is needed and a mismatch cannot crash inside torch.
* Keys: Space pause/resume, N one move, +/- double/halve the speed, R restart, Esc quit
  (constants: https://pyga.me/docs/ref/key.html).  A finished game stays on screen for ``HOLD_S``
  seconds, then the next one starts.
* ``watch_gif=1`` records the first game (the whole window, HUD included; at most
  ``GIF_MAX_FRAMES`` moves, then the file is written) as an animated GIF with Pillow, a required
  runtime dependency of matplotlib (https://matplotlib.org/stable/install/dependencies.html):
  ``save_all``, ``append_images``, ``duration`` in ms per frame, ``loop=0`` means forever -
  https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
"""

import dataclasses
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame

from snake4d.config import Config
from snake4d.env import SnakeEnv
from snake4d.evaluation import load_policy
from snake4d.logging_utils import make_run_dir, setup_logging
from snake4d.play import Session, Window, hud_text

MAX_SPEED, HOLD_S, GIF_MAX_FRAMES = 1024, 1.5, 1000
FASTER = (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS)  # the '=' key too: no Shift needed
SLOWER = (pygame.K_MINUS, pygame.K_KP_MINUS)
HELP = "Space pause   N one move   +/- speed ({} moves/s)   R restart   Esc quit"
log = logging.getLogger("snake4d.watch")


@dataclass
class Controls:
    """What the viewer asked for: playback speed, pause, single moves and a restart."""

    speed: int
    paused: bool = False
    pending: int = 0  # single moves requested with N (played even while paused)
    restart: bool = False


def apply_watch_key(key: int, controls: Controls) -> None:
    """Space pause/resume, N one move, +/- double or halve the speed (1..MAX_SPEED), R restart."""
    if key == pygame.K_SPACE:
        controls.paused = not controls.paused
    elif key == pygame.K_n:
        controls.pending += 1
    elif key in FASTER:
        controls.speed = min(MAX_SPEED, controls.speed * 2)
    elif key in SLOWER:
        controls.speed = max(1, controls.speed // 2)
    elif key == pygame.K_r:
        controls.restart = True


def board_of(policy, cfg: Config) -> Config:
    """``cfg`` with the board a checkpoint was trained on (scripted policies keep ``cfg``)."""
    obs_space, act_space = (getattr(policy, name, None)
                            for name in ("observation_space", "action_space"))
    if obs_space is None or act_space is None:
        return cfg
    n_inputs, ndim = obs_space.shape[0], int(act_space.n) // 2  # 4C + 2 inputs, 2*ndim actions
    cells = (n_inputs - 2) // 4
    size = round(cells ** (1 / ndim))
    if 4 * cells + 2 != n_inputs or size**ndim != cells:
        raise ValueError(f"no board matches a checkpoint with {n_inputs} inputs and "
                         f"{2 * ndim} actions")
    return dataclasses.replace(cfg, size=size, ndim=ndim)


def advance(policy, env: SnakeEnv, obs: np.ndarray, session: Session, cfg: Config) -> np.ndarray:
    """One move: masked predict, step, session bookkeeping.

    ``deterministic=True`` returns the distribution's mode, the argmax of the masked
    probabilities (``MaskableCategoricalDistribution.mode``,
    https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/maskable/distributions.py);
    ``False`` samples from them.
    """
    actions, _ = policy.predict(obs[None], action_masks=env.action_masks()[None],
                                deterministic=bool(cfg.deterministic))
    obs, _, terminated, truncated, info = env.step(int(np.asarray(actions).reshape(-1)[0]))
    session.finish(terminated, truncated, info)
    return obs


def save_gif(frames: list[np.ndarray], path: Path, speed: int) -> Path:
    """Animated GIF of the recorded frames at ``speed`` frames per second, looping forever."""
    from PIL import Image  # lazy: only the recording path needs Pillow

    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=1000 // speed, loop=0)
    return path


def run(cfg: Config, max_frames: int | None = None) -> Session:
    """Phase entry point: the policy plays game after game until Esc/close (or ``max_frames``)."""
    policy, _ = load_policy(cfg)
    cfg = board_of(policy, cfg)  # before make_run_dir, so config.json records the board played
    run_dir = make_run_dir(cfg, "watch")
    logger = setup_logging(run_dir)
    name = cfg.model_path or cfg.policy
    logger.info("watching %s on %d^%d (%d cells): %s, %d moves/s", name, cfg.size, cfg.ndim,
                cfg.n_cells, "argmax" if cfg.deterministic else "sampling", cfg.watch_speed)
    env = SnakeEnv(cfg)
    obs, _ = env.reset(seed=cfg.seed)
    policy.set_random_seed(cfg.seed)
    session, controls = Session(), Controls(speed=cfg.watch_speed)
    window = Window(cfg, f"snake4d watch: {Path(name).name}")
    frames, hold_until = 0, 0.0
    window.draw(env, (hud_text(env, session), HELP.format(controls.speed)))
    recording: list[np.ndarray] | None = [window.frame()] if cfg.watch_gif else None  # game 1
    while max_frames is None or frames < max_frames:
        keys = window.keys()
        if keys is None:
            break
        for key in keys:
            apply_watch_key(key, controls)
        moved = False
        if controls.restart or (session.done and time.monotonic() >= hold_until):
            obs, _ = env.reset()
            session.restart()
            controls.restart = False
        elif not session.done and (not controls.paused or controls.pending):
            obs = advance(policy, env, obs, session, cfg)
            controls.pending, moved = max(0, controls.pending - 1), True
            if session.done:
                hold_until = time.monotonic() + HOLD_S
        window.draw(env, (hud_text(env, session), HELP.format(controls.speed)))
        if recording is not None and moved:  # the start position, then one frame per move
            recording.append(window.frame())
            if session.done or len(recording) >= GIF_MAX_FRAMES:  # cap: frames are 1-2 MB each
                gif = save_gif(recording, run_dir / "game.gif", controls.speed)
                logger.info("recorded game 1 (%d frames%s) to %s", len(recording),
                            "" if session.done else ", capped", gif)
                recording = None
        window.tick(controls.speed)
        frames += 1
    window.close()
    return session
