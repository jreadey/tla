import random

import pytest

from tla.ai.hexfield import HexField, HexFieldGeometry
from tla.board import Board
from tla.hexgrid import AxialCoord, hexes_in_range
from tla.tile import Tile, TerrainType


def _brute_force_mass_near(field: HexField, coord: AxialCoord, radius: int) -> float:
    """The pre-vectorization implementation, kept here only as an
    independent reference to check the real (now numpy-vectorized)
    `HexField.mass_near` against -- see that method's own docstring for
    why it was rewritten (a real profiling find: 85% of a scored-move
    turn's time on a large board/fleet, almost entirely this walking
    `hexes_in_range` one hex at a time in pure Python over an array that
    was already numpy underneath)."""
    total = 0.0
    for c in hexes_in_range(coord, radius):
        row, col = field.geometry.to_index(c)
        if field.geometry.in_bounds(row, col):
            total += float(field.values[row, col])
    return total


def _sea_board(width: int = 25, height: int = 20) -> Board:
    board = Board(width=width, height=height)
    for q in range(-5, width - 5):
        for r in range(-5, height - 5):
            board.tiles[AxialCoord(q, r)] = Tile(coord=AxialCoord(q, r), terrain=TerrainType.SEA)
    return board


def test_mass_near_matches_brute_force_across_random_points_and_radii():
    board = _sea_board()
    field = HexField(HexFieldGeometry.from_board(board))
    rng = random.Random(42)
    for _ in range(200):
        coord = AxialCoord(rng.randint(-5, 19), rng.randint(-5, 14))
        if coord in board.tiles:
            field.add_point_mass(coord, rng.random() * 5)

    for _ in range(300):
        coord = AxialCoord(rng.randint(-8, 22), rng.randint(-8, 17))  # some outside the board entirely
        radius = rng.choice([0, 1, 2, 3, 4, 6])
        # approx: numpy's vectorized sum and the brute-force sequential
        # Python sum accumulate the same float terms in a different
        # order, so they can differ in the last bit or two -- not a
        # correctness issue.
        assert field.mass_near(coord, radius) == pytest.approx(_brute_force_mass_near(field, coord, radius))


def test_mass_near_radius_zero_is_just_the_one_cell():
    board = _sea_board()
    field = HexField(HexFieldGeometry.from_board(board))
    coord = AxialCoord(3, 2)
    field.add_point_mass(coord, 4.5)
    field.add_point_mass(AxialCoord(4, 2), 1.0)  # one hex away -- must not count at radius 0

    assert field.mass_near(coord, 0) == 4.5


def test_mass_near_handles_a_query_point_entirely_off_the_board():
    board = _sea_board()
    field = HexField(HexFieldGeometry.from_board(board))
    field.add_point_mass(AxialCoord(3, 2), 4.5)

    assert field.mass_near(AxialCoord(1000, 1000), 3) == 0.0


def test_mass_near_clips_correctly_at_a_board_edge():
    # A window that runs off the board on one side must still sum exactly
    # the in-bounds portion, not silently include out-of-bounds cells as
    # zero-but-shifted (which would misalign the kernel against the real
    # array) or crash.
    board = _sea_board()
    field = HexField(HexFieldGeometry.from_board(board))
    corner = AxialCoord(-5, -5)  # the board's own q0, r0 corner
    field.add_point_mass(corner, 2.0)

    assert field.mass_near(corner, 2) == pytest.approx(_brute_force_mass_near(field, corner, 2))
    assert field.mass_near(corner, 2) > 0.0
