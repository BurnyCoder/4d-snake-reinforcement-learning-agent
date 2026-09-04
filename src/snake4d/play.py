"""Human play: the N-dimensional board in a pygame-ce window, one move per key press.

Global context: the "game" half of the project.  The same ``SnakeEnv`` the agent trains on is
driven by the keyboard; the board is shown as the slice montage from ``render.py`` (a z-by-w grid of
x-by-y tiles for 4D), so every cell is visible at once.

Local notes:
* Keys: W/S move along x (up/down inside a tile), A/D along y (left/right inside a tile),
  I/K along z (tile row up/down), J/L along w (tile column left/right) - the WASD + IJKL
  convention of Pella86/Snake4d and the itch.io 4D snakes.  Keys beyond ``2*ndim`` are ignored.
* A key that targets the neck is ignored (classic snake refuses reversals); walls and body kill.
* Rendering is one ``pygame.surfarray.make_surface`` of the RGB montage
  (https://pyga.me/docs/ref/surfarray.html), no per-cell draw loop.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pygame

from snake4d.config import Config
from snake4d.env import SnakeEnv
from snake4d.logging_utils import make_run_dir, setup_logging
from snake4d.physics import SnakeBatch
from snake4d.render import to_rgb

# key -> action (axis = action >> 1, even = +1): x rows, y cols, z tile rows, w tile cols
KEYS = {
    pygame.K_s: 0, pygame.K_w: 1,  # x+1 (down), x-1 (up)
    pygame.K_d: 2, pygame.K_a: 3,  # y+1 (right), y-1 (left)
    pygame.K_k: 4, pygame.K_i: 5,  # z+1 (tile row down), z-1 (tile row up)
    pygame.K_l: 6, pygame.K_j: 7,  # w+1 (tile column right), w-1 (tile column left)
}
CELL_PX, LINE_PX, HUD_PX, FPS = 24, 3, 28, 30
log = logging.getLogger("snake4d.play")


@dataclass
class Session:
    """Mutable state of one sitting: step counter and episode status."""

    steps: int = 0
    status: str = "playing"
    done: bool = False


def targets_neck(sim: SnakeBatch, action: int) -> bool:
    """True when ``action`` would move the head into its own neck (the classic illegal reversal)."""
    target = sim.neigh[sim.head[0], action]
    return bool(sim.length[0] >= 2 and target >= 0 and sim.age[0, target] == sim.length[0] - 1)


def apply_key(env: SnakeEnv, key: int, session: Session) -> None:
    """Translate one key press into a reset, an ignored input, or one environment step."""
    if key == pygame.K_r or (session.done and key in KEYS):
        env.reset()
        session.steps, session.status, session.done = 0, "playing", False
        return
    if session.done or key not in KEYS or KEYS[key] >= env.cfg.n_actions:
        return
    action = KEYS[key]
    if targets_neck(env.sim, action):
        return  # ignored, like every classic snake
    _, _, terminated, truncated, info = env.step(action)
    session.steps += 1
    session.done = terminated or truncated
    if session.done:
        session.status = "WON" if info["is_success"] else ("dead" if terminated else "starved")
        log.info("episode over: %s after %d steps, fill %.2f", session.status, session.steps,
                 info["fill"])


def frame_surface(board: np.ndarray) -> pygame.Surface:
    """RGB montage -> pygame surface (surfarray expects (width, height, 3), hence the swap)."""
    rgb = to_rgb(board, cell=CELL_PX, line=LINE_PX)
    return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))


def hud_text(env: SnakeEnv, session: Session) -> str:
    """One-line status: length / cells, fill, steps and the episode state."""
    sim = env.sim
    return (f"length {int(sim.length[0])}/{sim.n_cells}  fill {sim.length[0] / sim.n_cells:.2f}  "
            f"steps {session.steps}  {session.status}   WASD=x/y IJKL=z/w  R=restart  Esc=quit")


def run(cfg: Config, max_frames: int | None = None) -> Session:
    """Open the window and play until Esc/close (or ``max_frames`` frames, for headless tests)."""
    run_dir = make_run_dir(cfg, "play")
    logger = setup_logging(run_dir)
    env = SnakeEnv(cfg, render_mode="rgb_array")
    env.reset(seed=cfg.seed)
    session = Session()
    pygame.init()
    frame = frame_surface(env.sim.board(0))
    screen = pygame.display.set_mode((frame.get_width(), frame.get_height() + HUD_PX))
    pygame.display.set_caption(f"snake4d {cfg.size}^{cfg.ndim}")
    font = pygame.font.SysFont(None, 22)
    clock = pygame.time.Clock()
    logger.info("play %d^%d: %d cells, %d actions", cfg.size, cfg.ndim, cfg.n_cells, cfg.n_actions)
    frames = 0
    while max_frames is None or frames < max_frames:
        for event in pygame.event.get():
            quit_key = event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            if event.type == pygame.QUIT or quit_key:
                pygame.quit()
                return session
            if event.type == pygame.KEYDOWN:
                apply_key(env, event.key, session)
        screen.blit(frame_surface(env.sim.board(0)), (0, 0))
        screen.blit(font.render(hud_text(env, session), True, (230, 230, 230)),
                    (6, frame.get_height() + 6))
        pygame.display.flip()
        clock.tick(FPS)
        frames += 1
    pygame.quit()
    return session
