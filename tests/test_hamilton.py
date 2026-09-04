"""Tests for snake4d.hamilton: Gray-code paths, even-size cycles and route validation."""

import numpy as np
import pytest

from snake4d.grid import neighbour_table
from snake4d.hamilton import check_route, gray_path, ham_cycle, route_actions, route_for


@pytest.mark.parametrize(
    ("size", "ndim"), [(2, 4), (3, 4), (4, 4), (4, 2), (6, 3), (3, 2), (2, 1), (5, 3)]
)
def test_route_for_visits_every_cell_with_unit_steps(size, ndim):
    route, closed = route_for(size, ndim)
    assert len(route) == size**ndim and closed == (size % 2 == 0 and ndim >= 2)
    check_route(route, size, ndim, closed)  # raises on any defect


def test_gray_path_matches_the_wikipedia_ternary_example():
    words = ["".join(map(str, t)) for t in gray_path(3, 2)]
    assert words == "00 01 02 12 11 10 20 21 22".split()


def test_ham_cycle_rejects_odd_sizes_and_1d():
    with pytest.raises(ValueError):
        ham_cycle(3, 2)
    with pytest.raises(ValueError):
        ham_cycle(4, 1)


def test_check_route_rejects_broken_routes():
    route, closed = route_for(4, 2)
    swapped = route.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    with pytest.raises(ValueError, match="adjacent"):
        check_route(swapped, 4, 2, closed)
    duplicated = route.copy()
    duplicated[1] = duplicated[0]
    with pytest.raises(ValueError, match="exactly once"):
        check_route(duplicated, 4, 2, closed)


@pytest.mark.parametrize(("size", "ndim"), [(4, 4), (3, 4)])
def test_route_actions_step_along_the_route(size, ndim):
    route, closed = route_for(size, ndim)
    neigh = neighbour_table(size, ndim)
    actions = route_actions(route, closed, neigh)
    n_cells = size**ndim
    for k in range(n_cells - (0 if closed else 1)):
        assert neigh[route[k], actions[route[k]]] == route[(k + 1) % n_cells]
    if not closed:
        assert actions[route[-1]] == -1
    assert np.all(actions[route[: n_cells - 1]] >= 0)
