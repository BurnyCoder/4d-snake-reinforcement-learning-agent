"""Tests for snake4d.render: montage index identity, ASCII and RGB output."""

import numpy as np

from snake4d.physics import BODY, FOOD, HEAD
from snake4d.render import ascii, montage, to_rgb


def test_4d_montage_index_identity():
    board = np.arange(4**4).reshape(4, 4, 4, 4)
    img = montage(board)
    assert img.shape == (16, 16)
    for x, y, z, w in np.ndindex(4, 4, 4, 4):
        assert img[z * 4 + x, w * 4 + y] == board[x, y, z, w]


def test_lower_dimensional_montages():
    assert montage(np.zeros(3)).shape == (1, 3)
    assert montage(np.zeros((3, 3))).shape == (3, 3)
    assert montage(np.arange(27).reshape(3, 3, 3)).shape == (9, 3)


def test_ascii_and_rgb_show_every_entity():
    board = np.zeros((3, 3, 3, 3), dtype=np.int8)
    board[0, 0, 0, 0], board[1, 0, 0, 0], board[2, 2, 2, 2] = HEAD, BODY, FOOD
    text = ascii(board)
    assert text.count("@") == 1 and text.count("o") == 1 and text.count("*") == 1
    rgb = to_rgb(board, cell=2, line=1)
    assert rgb.shape == (9 * 2 + 2, 9 * 2 + 2, 3) and rgb.dtype == np.uint8


def test_shade_darkens_the_body_towards_the_tail_only():
    board = np.zeros((2, 2), dtype=np.int8)
    board[0, 0], board[0, 1], board[1, 1] = BODY, BODY, HEAD  # tail (0,0), neck (0,1), head
    shade = np.array([[1 / 3, 2 / 3], [0.0, 1.0]])  # age / length
    plain, shaded = to_rgb(board), to_rgb(board, shade=shade)
    assert (shaded[0, 0] < shaded[0, 1]).all() and (shaded[0, 1] < plain[0, 1]).all()
    assert (shaded[1, 1] == plain[1, 1]).all()  # the head keeps its full colour
