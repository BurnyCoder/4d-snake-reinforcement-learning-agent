"""Rendering an N-dimensional board for humans: a 2D montage of 2D slices.

Global context: used by ``SnakeEnv.render`` (ansi / rgb_array), the pygame window in ``play.py``
and report figures.  The 4D board ``(x, y, z, w)`` becomes a ``z``-by-``w`` grid of ``x``-by-``y``
tiles - the layout of the 4D minesweeper by Checkmate6659 ("a 2D grid of 2D layers"); 4D->3D
perspective projections are deliberately avoided (eugeneko/Snake4D's README says its projected
board becomes unplayable beyond a certain snake length, https://github.com/eugeneko/Snake4D).

Local notes: the montage is one numpy transpose+reshape
(https://numpy.org/doc/stable/reference/generated/numpy.transpose.html); the index identity
``img[z*X + x, w*Y + y] == board[x, y, z, w]`` is asserted in tests/test_render.py.
"""

import numpy as np

from snake4d.physics import BODY, EMPTY, FOOD, HEAD

CHARS = {EMPTY: ".", BODY: "o", HEAD: "@", FOOD: "*"}
PALETTE = np.array(
    [[30, 30, 30], [60, 160, 60], [240, 240, 60], [220, 60, 60]], dtype=np.uint8
)  # EMPTY, BODY, HEAD, FOOD
GRID_LINE = np.array([90, 90, 90], dtype=np.uint8)


def montage(board: np.ndarray) -> np.ndarray:
    """Fold a 1D-4D board into a 2D array of tiles (rows = z-tiles of x, cols = w-tiles of y)."""
    s = board.shape[0]
    if board.ndim == 1:
        return board[None, :]
    if board.ndim == 2:
        return board
    if board.ndim == 3:  # (x, y, z) -> z tiles stacked vertically
        return board.transpose(2, 0, 1).reshape(s * s, s)
    if board.ndim == 4:  # (x, y, z, w) -> z-by-w grid of x-by-y tiles
        return board.transpose(2, 0, 3, 1).reshape(s * s, s * s)
    raise NotImplementedError("montage supports ndim <= 4")


def ascii(board: np.ndarray) -> str:
    """Text montage with a blank line/column between tiles; head '@', body 'o', food '*'."""
    img, s = montage(board), board.shape[0]
    lines = []
    for r, row in enumerate(img):
        if r and r % s == 0:
            lines.append("")
        chars = [CHARS[int(v)] + (" " if (c + 1) % s == 0 else "") for c, v in enumerate(row)]
        lines.append("".join(chars).rstrip())
    return "\n".join(lines)


def _with_lines(rgb: np.ndarray, period: int, width: int, axis: int) -> np.ndarray:
    """Insert a grey strip of ``width`` pixels after every ``period`` pixels along ``axis``."""
    chunks = np.split(rgb, range(period, rgb.shape[axis], period), axis=axis)
    shape = list(rgb.shape)
    shape[axis] = width
    strip = np.broadcast_to(GRID_LINE, shape)
    pieces = [piece for chunk in chunks for piece in (chunk, strip)][:-1]
    return np.concatenate(pieces, axis=axis)


def to_rgb(board: np.ndarray, cell: int = 1, line: int = 0) -> np.ndarray:
    """``(H, W, 3)`` uint8 image of the montage: ``cell`` px per cell, ``line`` px between tiles."""
    img, s = montage(board), board.shape[0]
    rgb = PALETTE[img].repeat(cell, axis=0).repeat(cell, axis=1)  # palette lookup + upscale
    if line:
        rgb = _with_lines(_with_lines(rgb, s * cell, line, axis=0), s * cell, line, axis=1)
    return np.ascontiguousarray(rgb)


def save_png(board: np.ndarray, path, cell: int = 16) -> None:
    """Write the montage as a PNG with matplotlib's ``imsave`` (no figure/axes needed)."""
    import matplotlib.pyplot as plt  # lazy: keeps the game core free of matplotlib

    plt.imsave(path, to_rgb(board, cell=cell, line=2))
