"""Tests for snake4d.grid: neighbour table, parity classes and distances."""

import numpy as np
import pytest

from snake4d.grid import cell_of, coords_of, l1_distance, neighbour_table, parity_split


@pytest.mark.parametrize(("size", "ndim"), [(2, 2), (3, 3), (4, 4), (5, 2)])
def test_neighbour_table_matches_coordinate_arithmetic(size, ndim):
    neigh = neighbour_table(size, ndim)
    coords = coords_of(np.arange(size**ndim), size, ndim)
    assert neigh.shape == (size**ndim, 2 * ndim)
    for cell in range(size**ndim):
        for action in range(2 * ndim):
            moved = coords[cell].copy()
            moved[action >> 1] += 1 if action % 2 == 0 else -1
            inside = 0 <= moved[action >> 1] < size
            assert neigh[cell, action] == (cell_of(moved, size, ndim) if inside else -1)


def test_opposite_action_is_the_bitwise_complement():
    neigh = neighbour_table(4, 4)
    for cell in range(256):
        for action in range(8):
            if neigh[cell, action] >= 0:
                assert neigh[neigh[cell, action], action ^ 1] == cell


def test_parity_split_decides_hamiltonian_cycles():
    assert parity_split(3, 4) == (41, 40)  # odd side: no Hamiltonian cycle exists
    assert parity_split(4, 4) == (128, 128)  # even side: cycle possible
    assert parity_split(2, 2) == (2, 2)


def test_l1_distance_across_the_4d_board():
    assert l1_distance(np.array([0]), np.array([255]), 4, 4)[0] == 12  # (0,0,0,0) -> (3,3,3,3)
