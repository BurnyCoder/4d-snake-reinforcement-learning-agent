"""Grid geometry for an N-dimensional board addressed by flat cell indices.

Global context: the whole game (physics, masks, routes, rendering) works on flat indices
``0..C-1`` with ``C = size ** ndim``; this module is the only place that converts between flat
indices and coordinates, so no other file does coordinate arithmetic.

Local notes: action ``a`` moves along axis ``a >> 1`` by ``+1`` if ``a`` is even else ``-1`` - the
2*ndim direction scheme of Pella86/Snake4d (https://github.com/Pella86/Snake4d/blob/master/src/snake.py,
MIT).  Conversions use numpy's C-order helpers
https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html and
https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html.
"""

import numpy as np


def shape_of(size: int, ndim: int) -> tuple[int, ...]:
    """Board shape ``(size,) * ndim``."""
    return (size,) * ndim


def coords_of(cells: np.ndarray | int, size: int, ndim: int) -> np.ndarray:
    """Flat index -> integer coordinates, shape ``(..., ndim)``."""
    return np.stack(np.unravel_index(cells, shape_of(size, ndim)), axis=-1)


def cell_of(coord: tuple[int, ...] | np.ndarray, size: int, ndim: int) -> int:
    """Integer coordinates -> flat index."""
    return int(np.ravel_multi_index(tuple(int(c) for c in coord), shape_of(size, ndim)))


def neighbour_table(size: int, ndim: int) -> np.ndarray:
    """``NEIGH[cell, a]`` = flat index reached from ``cell`` by action ``a``, or ``-1`` at a wall.

    Built once per board (shape ``(C, 2*ndim)``); every move, wall test and action mask is then a
    single gather ``NEIGH[head, a]`` with no per-step coordinate arithmetic.
    """
    n_cells = size**ndim
    coords = coords_of(np.arange(n_cells), size, ndim)  # (C, ndim)
    table = np.full((n_cells, 2 * ndim), -1, dtype=np.int64)
    for action in range(2 * ndim):
        axis, delta = action >> 1, 1 - 2 * (action & 1)  # even action = +1, odd action = -1
        moved = coords.copy()
        moved[:, axis] += delta
        inside = (moved[:, axis] >= 0) & (moved[:, axis] < size)
        table[inside, action] = np.ravel_multi_index(moved[inside].T, shape_of(size, ndim))
    return table


def parity_split(size: int, ndim: int) -> tuple[int, int]:
    """Sizes of the two colour classes of the grid graph (cells with even / odd coordinate sum).

    Grid graphs are bipartite by coordinate-sum parity
    (https://mathworld.wolfram.com/GridGraph.html); a Hamiltonian cycle needs equal classes, so
    ``3**4 -> (41, 40)`` has none and ``4**4 -> (128, 128)`` can have one.
    """
    coords = coords_of(np.arange(size**ndim), size, ndim)
    even = int((coords.sum(axis=1) % 2 == 0).sum())
    return even, size**ndim - even


def l1_distance(a: np.ndarray, b: np.ndarray, size: int, ndim: int) -> np.ndarray:
    """Manhattan distance between flat cells ``a`` and ``b`` (element-wise)."""
    return np.abs(coords_of(a, size, ndim) - coords_of(b, size, ndim)).sum(axis=-1)
