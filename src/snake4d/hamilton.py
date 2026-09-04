"""Hamiltonian routes through the board: the scripted baseline and the curriculum's demonstration.

Global context: a route visiting every cell exactly once proves the board is completable, gives
the perfect-play ceiling (twanvl/snake, MIT, https://github.com/twanvl/snake) and provides the
demonstration trajectory for the Backplay reverse curriculum in ``physics.py``.

Local notes:
* ``gray_path`` is the reflected n-ary Gray code (boustrophedon): consecutive words differ by +-1
  in one digit, i.e. one grid edge - https://en.wikipedia.org/wiki/Gray_code (validated against
  the ternary example ``00 01 02 12 11 10 20 21 22``).  Works for every size and dimension.
* ``ham_cycle`` needs an even size (bipartite parity, see ``grid.parity_split``); existence for
  d >= 3 with an even dimension is Ruskey & Sawada 2003,
  https://www.combinatorics.org/ojs/index.php/eljc/article/view/v10i1r1.  The recursive
  fibre construction below is our own, so ``check_route`` asserts every property.
"""

import numpy as np

from snake4d.grid import cell_of, neighbour_table


def gray_path(size: int, ndim: int) -> list[tuple[int, ...]]:
    """Hamiltonian path as coordinate tuples (reflected n-ary Gray code), any size/ndim."""
    seq: list[tuple[int, ...]] = [()]
    for _ in range(ndim):
        seq = [
            (value,) + tail
            for i, value in enumerate(range(size))
            for tail in (seq if i % 2 == 0 else seq[::-1])  # reflect every other block
        ]
    return seq


def _ham_cycle_2d(size: int) -> list[tuple[int, int]]:
    """Even-size 2D cycle: a spine along y=0, then columns swept boustrophedon back to x=0."""
    cycle = [(x, 0) for x in range(size)]
    for j, x in enumerate(range(size - 1, -1, -1)):
        rows = range(1, size) if j % 2 == 0 else range(size - 1, 0, -1)
        cycle += [(x, y) for y in rows]
    return cycle


def ham_cycle(size: int, ndim: int) -> list[tuple[int, ...]]:
    """Hamiltonian cycle for even ``size`` and ``ndim >= 2`` (recursive fibre construction).

    Each cell of the (ndim-1)-dimensional cycle is expanded into a full sweep of the new axis,
    alternating direction; the inner cycle has even length so the sweep ends where it started.
    """
    if size % 2 or ndim < 2:
        raise ValueError("a Hamiltonian cycle needs an even size and ndim >= 2")
    if ndim == 2:
        return _ham_cycle_2d(size)
    inner = ham_cycle(size, ndim - 1)
    out: list[tuple[int, ...]] = []
    for i, cell in enumerate(inner):
        levels = range(size) if i % 2 == 0 else range(size - 1, -1, -1)
        out += [cell + (level,) for level in levels]
    return out


def check_route(route: np.ndarray, size: int, ndim: int, closed: bool) -> None:
    """Raise ``ValueError`` unless ``route`` visits every cell once with unit steps (and closes)."""
    n_cells = size**ndim
    neigh = neighbour_table(size, ndim)
    if len(route) != n_cells or len(set(route.tolist())) != n_cells:
        raise ValueError("route must visit every cell exactly once")
    successors = np.roll(route, -1) if closed else route[1:]
    pairs = zip(route[: len(successors)], successors, strict=True)
    for a, b in pairs:
        if b not in neigh[a]:
            raise ValueError(f"cells {a} and {b} are not adjacent")


def route_for(size: int, ndim: int) -> tuple[np.ndarray, bool]:
    """Flat-index route for the board: a cycle when possible, otherwise a Gray-code path.

    Returns ``(route, closed)``; ``route[k]`` is the k-th cell, ``closed`` tells whether the last
    cell is adjacent to the first (so any rotation of the route is also a route).
    """
    closed = size % 2 == 0 and ndim >= 2
    coords = ham_cycle(size, ndim) if closed else gray_path(size, ndim)
    route = np.array([cell_of(c, size, ndim) for c in coords], dtype=np.int64)
    check_route(route, size, ndim, closed)
    return route, closed


def route_actions(route: np.ndarray, closed: bool, neigh: np.ndarray) -> np.ndarray:
    """``actions[cell]`` = the action that moves from ``cell`` to its successor on the route.

    ``-1`` marks the last cell of an open path (no successor).
    """
    actions = np.full(len(route), -1, dtype=np.int64)
    successors = np.roll(route, -1) if closed else np.append(route[1:], -1)
    for cell, nxt in zip(route, successors, strict=True):
        if nxt >= 0:
            actions[cell] = int(np.flatnonzero(neigh[cell] == nxt)[0])
    return actions
