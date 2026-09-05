"""Hamiltonian routes through the board: the scripted baseline and the curriculum's demonstration.

Global context: a route visiting every cell exactly once proves the board is completable, gives
the perfect-play ceiling (twanvl/snake, MIT, https://github.com/twanvl/snake) and provides the
demonstration trajectory for the Backplay reverse curriculum in ``physics.py``.

Local notes:
* ``gray_path`` is the reflected n-ary Gray code (boustrophedon): consecutive words differ by +-1
  in one digit, i.e. one grid edge - https://en.wikipedia.org/wiki/Gray_code (validated against
  the ternary example ``00 01 02 12 11 10 20 21 22``).  Works for every size and dimension.
* ``ham_cycle`` needs an even size: a 2D grid graph with both sides >= 2 is Hamiltonian when one
  side is even (Skiena 1990, via https://mathworld.wolfram.com/GridGraph.html), and an even
  cell count is necessary because grid graphs are bipartite (see ``grid.parity_split``; Itai,
  Papadimitriou & Szwarcfiter 1982, https://doi.org/10.1137/0211056).  The recursive fibre
  construction below lifts the 2D cycle to higher dimensions; it is our own, so
  ``check_route`` asserts every property.
* Odd sizes have no Hamiltonian cycle (odd cell count), but the board minus the corner
  ``(0, ..., 0)`` has an even count and ``cycle_minus_corner`` builds one: an explicit 2D
  construction, then per extra dimension the same fibre sweep with the corner's own column
  spliced in as a 2-by-(n-1) "ladder" next to a neighbouring column.  The corner is only entered
  by a guarded detour (``agents.RoutePolicy``), which is what makes odd boards completable.
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


def _cycle_minus_corner_2d(size: int) -> list[tuple[int, int]]:
    """Odd-size 2D cycle over every cell except (0, 0): rows 0-1 zig-zag, rows 2.. boustrophedon,
    then column x=0 from the top back down to (0, 1), which closes onto the start (1, 1)."""
    cycle = [(1, 1), (1, 0)]
    for x in range(2, size):
        cycle += [(x, 0), (x, 1)] if x % 2 == 0 else [(x, 1), (x, 0)]
    for y in range(2, size):
        xs = range(size - 1, 0, -1) if y % 2 == 0 else range(1, size)
        cycle += [(x, y) for x in xs]
    cycle += [(0, y) for y in range(size - 1, 0, -1)]
    return cycle


def _ladder(z: tuple[int, ...], corner: tuple[int, ...], size: int, up: bool) -> list:
    """Sweep of column ``z`` with the corner's column (levels 1..size-1) woven in as a ladder."""
    cells = [z + (0,)]
    for level in range(1, size - 1, 2):
        cells += [z + (level,), corner + (level,), corner + (level + 1,), z + (level + 1,)]
    return cells if up else cells[::-1]


def cycle_minus_corner(size: int, ndim: int) -> list[tuple[int, ...]]:
    """Hamiltonian cycle over all cells except ``(0,)*ndim`` for odd ``size`` and ``ndim >= 2``."""
    if size % 2 == 0 or ndim < 2:
        raise ValueError("cycle_minus_corner needs an odd size and ndim >= 2")
    if ndim == 2:
        return _cycle_minus_corner_2d(size)
    inner = cycle_minus_corner(size, ndim - 1)  # even length; misses the inner corner
    corner = (0,) * (ndim - 1)
    z = (1,) + (0,) * (ndim - 2)  # a neighbour of the corner, always on the inner cycle
    out: list[tuple[int, ...]] = []
    for i, cell in enumerate(inner):
        up = i % 2 == 0
        if cell == z:
            out += _ladder(z, corner, size, up)
        else:
            levels = range(size) if up else range(size - 1, -1, -1)
            out += [cell + (level,) for level in levels]
    return out


def check_route(route: np.ndarray, size: int, ndim: int, closed: bool) -> None:
    """Raise ``ValueError`` unless ``route`` visits every cell once with unit steps (and closes).

    A closed route on an odd board may omit exactly the corner cell 0 (see ``cycle_minus_corner``).
    """
    n_cells = size**ndim
    expected = n_cells - 1 if (closed and size % 2) else n_cells
    neigh = neighbour_table(size, ndim)
    if len(route) != expected or len(set(route.tolist())) != expected:
        raise ValueError("route must visit every cell exactly once")
    if expected < n_cells and 0 in route:
        raise ValueError("an odd-board cycle must leave out the corner cell 0")
    successors = np.roll(route, -1) if closed else route[1:]
    for a, b in zip(route[: len(successors)], successors, strict=True):
        if b not in neigh[a]:
            raise ValueError(f"cells {a} and {b} are not adjacent")


def route_for(size: int, ndim: int) -> tuple[np.ndarray, bool]:
    """Flat-index route for the board: a cycle when ``ndim >= 2`` (minus the corner on odd sizes),
    otherwise the Gray-code path.  Returns ``(route, closed)``."""
    if ndim < 2:
        coords, closed = gray_path(size, ndim), False
    elif size % 2 == 0:
        coords, closed = ham_cycle(size, ndim), True
    else:
        coords, closed = cycle_minus_corner(size, ndim), True
    route = np.array([cell_of(c, size, ndim) for c in coords], dtype=np.int64)
    check_route(route, size, ndim, closed)
    return route, closed


def route_actions(route: np.ndarray, closed: bool, neigh: np.ndarray) -> np.ndarray:
    """``actions[cell]`` = the action that moves from ``cell`` to its successor on the route.

    ``-1`` marks cells with no successor (the end of an open path, or a cell off the route).
    """
    actions = np.full(len(neigh), -1, dtype=np.int64)
    successors = np.roll(route, -1) if closed else np.append(route[1:], -1)
    for cell, nxt in zip(route, successors, strict=True):
        if nxt >= 0:
            actions[cell] = int(np.flatnonzero(neigh[cell] == nxt)[0])
    return actions
