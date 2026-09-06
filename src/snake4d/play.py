"""Human play: the N-dimensional board in a pygame-ce window, one move per key press.

Global context: the "game" half of the project.  The same ``SnakeEnv`` the agent trains on is
driven by the keyboard; the board is shown as the slice montage from ``render.py`` (a z-by-w grid of
x-by-y tiles for 4D), so every cell is visible at once.  ``Window`` and ``Session`` are shared with
``watch.py``, where a saved network chooses the moves instead of the keyboard.

Local notes:
* Keys: W/S move along x (up/down inside a tile), A/D along y (left/right inside a tile),
  I/K along z (tile row up/down), J/L along w (tile column left/right) - the WASD + IJKL
  convention of Pella86/Snake4d (https://github.com/Pella86/Snake4d/blob/master/main.py).
  Keys beyond ``2*ndim`` are ignored.
* A key that targets the neck is ignored (classic snake refuses reversals); walls and body kill.
* Cells are scaled so the montage is about ``WINDOW_PX`` wide on every board size, and the tile
  rows/columns carry ``z``/``w`` labels.  Rendering is one ``pygame.surfarray.make_surface`` of the
  RGB montage (https://pyga.me/docs/ref/surfarray.html), no per-cell draw loop; text uses the
  built-in font (``SysFont(None, size)``, https://pyga.me/docs/ref/font.html); ``Clock.tick(fps)``
  paces the loop (https://pyga.me/docs/ref/time.html#pygame.time.Clock.tick).
"""

import logging
from dataclasses import dataclass

import numpy as np
import pygame

from snake4d.config import Config
from snake4d.env import SnakeEnv
from snake4d.logging_utils import make_run_dir, setup_logging
from snake4d.physics import SnakeBatch
from snake4d.render import montage, to_rgb

# key -> action (axis = action >> 1, even = +1): x rows, y cols, z tile rows, w tile cols
KEYS = {
    pygame.K_s: 0, pygame.K_w: 1,  # x+1 (down), x-1 (up)
    pygame.K_d: 2, pygame.K_a: 3,  # y+1 (right), y-1 (left)
    pygame.K_k: 4, pygame.K_i: 5,  # z+1 (tile row down), z-1 (tile row up)
    pygame.K_l: 6, pygame.K_j: 7,  # w+1 (tile column right), w-1 (tile column left)
}
WINDOW_PX, MIN_CELL_PX, LINE_PX, LABEL_PX, FONT_PX, FPS = 640, 12, 3, 18, 20, 30
HUD_PX = 2 * FONT_PX + 12  # two text lines under the board
TEXT = (230, 230, 230)
PLAY_HELP = ("WASD = x (rows) / y (cols)   IJKL = z (tile rows) / w (tile cols)   "
             "R restart   Esc quit")
log = logging.getLogger("snake4d.play")


@dataclass
class Session:
    """Mutable state of one sitting: game counter, step counter and episode status."""

    game: int = 1
    steps: int = 0
    status: str = "playing"
    done: bool = False

    def restart(self) -> None:
        """Next game: counters back to zero (the caller resets the environment)."""
        self.game, self.steps, self.status, self.done = self.game + 1, 0, "playing", False

    def finish(self, terminated: bool, truncated: bool, info: dict) -> None:
        """Count one move; when it ended the game, set the status and log the outcome."""
        self.steps += 1
        self.done = terminated or truncated
        if self.done:
            self.status = "WON" if info["is_success"] else ("dead" if terminated else "starved")
            log.info("game %d: %s after %d steps, fill %.2f", self.game, self.status, self.steps,
                     info["fill"])


def targets_neck(sim: SnakeBatch, action: int) -> bool:
    """True when ``action`` would move the head into its own neck (the classic illegal reversal)."""
    target = sim.neigh[sim.head[0], action]
    return bool(sim.length[0] >= 2 and target >= 0 and sim.age[0, target] == sim.length[0] - 1)


def apply_key(env: SnakeEnv, key: int, session: Session) -> None:
    """Translate one key press into a reset, an ignored input, or one environment step."""
    if key == pygame.K_r or (session.done and key in KEYS):
        env.reset()
        session.restart()
        return
    if session.done or key not in KEYS or KEYS[key] >= env.cfg.n_actions:
        return
    action = KEYS[key]
    if targets_neck(env.sim, action):
        return  # ignored, like every classic snake
    _, _, terminated, truncated, info = env.step(action)
    session.finish(terminated, truncated, info)


def hud_text(env: SnakeEnv, session: Session) -> str:
    """One-line status: game number, length / cells, fill, steps and the episode state."""
    sim = env.sim
    return (f"game {session.game}   length {int(sim.length[0])}/{sim.n_cells}   "
            f"fill {sim.length[0] / sim.n_cells:.2f}   steps {session.steps}   {session.status}")


class Window:
    """The pygame window: the scaled slice montage with tile labels above two HUD text lines."""

    def __init__(self, cfg: Config, caption: str) -> None:
        pygame.init()
        self.cfg = cfg
        blank = np.zeros((cfg.size,) * cfg.ndim, dtype=np.int8)
        rows, cols = montage(blank).shape  # the montage in cells: a 4D board is size^2 x size^2
        self.cell = max(MIN_CELL_PX, min(WINDOW_PX // rows, WINDOW_PX // cols))
        self.margin = LABEL_PX if cfg.ndim >= 3 else 0  # room for the z / w tile labels
        height, width = to_rgb(blank, self.cell, LINE_PX).shape[:2]
        self.screen = pygame.display.set_mode((width + self.margin, height + self.margin + HUD_PX))
        pygame.display.set_caption(caption)
        self.font = pygame.font.SysFont(None, FONT_PX)
        self.clock = pygame.time.Clock()

    def keys(self) -> list[int] | None:
        """Keys pressed since the last frame, or ``None`` once the window closed / Esc was hit."""
        pressed = []
        for event in pygame.event.get():
            quit_key = event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            if event.type == pygame.QUIT or quit_key:
                return None
            if event.type == pygame.KEYDOWN:
                pressed.append(event.key)
        return pressed

    def draw(self, env: SnakeEnv, lines: tuple[str, ...]) -> None:
        """Blit the montage (surfarray wants (width, height, 3), hence the swap), labels and HUD."""
        board, sim = env.sim.board(0), env.sim
        shade = (sim.age[0] / max(int(sim.length[0]), 1)).reshape(board.shape)  # tail dark
        rgb = to_rgb(board, cell=self.cell, line=LINE_PX, shade=shade)
        self.screen.fill((0, 0, 0))
        self.screen.blit(pygame.surfarray.make_surface(rgb.swapaxes(0, 1)),
                         (self.margin, self.margin))
        pitch = self.cfg.size * self.cell + LINE_PX  # pixels from one tile row/column to the next
        for k in range(self.cfg.size if self.cfg.ndim >= 3 else 0):  # 3D: z rows; 4D: + w columns
            self.screen.blit(self.font.render(f"z{k}", True, TEXT),
                             (2, self.margin + k * pitch + 2))
            if self.cfg.ndim == 4:
                self.screen.blit(self.font.render(f"w{k}", True, TEXT),
                                 (self.margin + k * pitch + 2, 2))
        for i, text in enumerate(lines):
            self.screen.blit(self.font.render(text, True, TEXT),
                             (6, self.margin + rgb.shape[0] + 6 + i * FONT_PX))
        pygame.display.flip()

    def frame(self) -> np.ndarray:
        """The whole window as an ``(H, W, 3)`` uint8 array (``array3d`` returns ``(W, H, 3)``)."""
        return pygame.surfarray.array3d(self.screen).swapaxes(0, 1)

    def tick(self, fps: int) -> None:
        """Sleep so the loop runs at most ``fps`` iterations per second."""
        self.clock.tick(fps)

    def close(self) -> None:
        """Close the window and release pygame."""
        pygame.quit()


def run(cfg: Config, max_frames: int | None = None) -> Session:
    """Open the window and play until Esc/close (or ``max_frames`` frames, for headless tests)."""
    run_dir = make_run_dir(cfg, "play")
    logger = setup_logging(run_dir)
    env = SnakeEnv(cfg)
    env.reset(seed=cfg.seed)
    session, window = Session(), Window(cfg, f"snake4d {cfg.size}^{cfg.ndim}")
    logger.info("play %d^%d: %d cells, %d actions", cfg.size, cfg.ndim, cfg.n_cells, cfg.n_actions)
    frames = 0
    while max_frames is None or frames < max_frames:
        keys = window.keys()
        if keys is None:
            break
        for key in keys:
            apply_key(env, key, session)
        window.draw(env, (hud_text(env, session), PLAY_HELP))
        window.tick(FPS)
        frames += 1
    window.close()
    return session
