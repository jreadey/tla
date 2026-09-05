"""Axial-to-pixel conversion and hex/board drawing (flat-top orientation)."""

from __future__ import annotations

import math

import arcade

from tla.board import Board
from tla.hexgrid import AxialCoord
from tla.tile import PLAYER_A, PLAYER_B, TerrainType

TERRAIN_COLORS = {
    TerrainType.LAND: (86, 125, 70),
    TerrainType.SEA: (43, 92, 138),
    TerrainType.SHORE: (194, 178, 128),
}
OUTLINE_COLOR = (30, 30, 30, 140)
PORT_COLORS = {
    PLAYER_A: (220, 60, 60),
    PLAYER_B: (60, 100, 230),
}


def axial_to_pixel(coord: AxialCoord, hex_size: float) -> tuple[float, float]:
    x = hex_size * 1.5 * coord.q
    y = hex_size * math.sqrt(3) * (coord.r + coord.q / 2)
    return x, y


def hex_corners(center: tuple[float, float], hex_size: float) -> list[tuple[float, float]]:
    cx, cy = center
    return [
        (cx + hex_size * math.cos(math.radians(60 * i)), cy + hex_size * math.sin(math.radians(60 * i)))
        for i in range(6)
    ]


def board_pixel_bounds(board: Board, hex_size: float) -> tuple[float, float, float, float]:
    """(min_x, min_y, max_x, max_y) of the board's hex centers, padded by one hex."""
    xs: list[float] = []
    ys: list[float] = []
    for coord in board.tiles:
        x, y = axial_to_pixel(coord, hex_size)
        xs.append(x)
        ys.append(y)
    pad = hex_size * 2
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad


def draw_board(board: Board, hex_size: float, origin_x: float, origin_y: float) -> None:
    for coord, tile in board.tiles.items():
        x, y = axial_to_pixel(coord, hex_size)
        center = (x + origin_x, y + origin_y)
        corners = hex_corners(center, hex_size * 0.98)
        arcade.draw_polygon_filled(corners, TERRAIN_COLORS[tile.terrain])
        arcade.draw_polygon_outline(corners, OUTLINE_COLOR, 1)
        if tile.is_port:
            arcade.draw_circle_filled(
                center[0], center[1], hex_size * 0.35, PORT_COLORS[tile.port_owner]
            )
