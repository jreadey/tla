"""Hex/board drawing, plus the coastline contour overlay (flat-top orientation)."""

from __future__ import annotations

import math
from pathlib import Path

import arcade

from tla.board import Board
from tla.elevation import Segment
from tla.hexgrid import axial_to_pixel
from tla.tile import PLAYER_A, PLAYER_B, TerrainType

TERRAIN_COLORS = {
    TerrainType.LAND: (86, 125, 70),
    TerrainType.SEA: (43, 92, 138),
}
OUTLINE_COLOR = (30, 30, 30, 140)
PORT_COLORS = {
    PLAYER_A: (220, 60, 60),
    PLAYER_B: (60, 100, 230),
}
CONTOUR_COLOR = (20, 20, 20, 200)
CONTOUR_WIDTH = 2.0

# Anchor icon is stored white-on-transparent so it can be tinted per player
# via draw_texture_rect's color multiplier, rather than keeping one texture
# per player color.
_ANCHOR_ICON_PATH = Path(__file__).parent / "assets" / "anchor.png"
_anchor_texture: arcade.Texture | None = None


def _anchor_texture_cached() -> arcade.Texture:
    global _anchor_texture
    if _anchor_texture is None:
        _anchor_texture = arcade.load_texture(_ANCHOR_ICON_PATH)
    return _anchor_texture


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


def draw_anchor(center: tuple[float, float], size: float, color: tuple[int, int, int]) -> None:
    """Draw the anchor icon centered at `center`, tinted to `color`."""
    cx, cy = center
    rect = arcade.XYWH(cx, cy, size, size)
    arcade.draw_texture_rect(_anchor_texture_cached(), rect, color=arcade.types.Color(*color))


def draw_board(board: Board, hex_size: float, origin_x: float, origin_y: float) -> None:
    for coord, tile in board.tiles.items():
        x, y = axial_to_pixel(coord, hex_size)
        center = (x + origin_x, y + origin_y)
        corners = hex_corners(center, hex_size * 0.98)
        arcade.draw_polygon_filled(corners, TERRAIN_COLORS[tile.terrain])
        arcade.draw_polygon_outline(corners, OUTLINE_COLOR, 1)
        if tile.is_port:
            draw_anchor(center, hex_size * 0.85, PORT_COLORS[tile.port_owner])


def draw_contour(segments: list[Segment], origin_x: float, origin_y: float) -> None:
    """Draw precomputed coastline segments (see tla.elevation.marching_squares_segments)."""
    if not segments:
        return
    points: list[tuple[float, float]] = []
    for (x0, y0), (x1, y1) in segments:
        points.append((x0 + origin_x, y0 + origin_y))
        points.append((x1 + origin_x, y1 + origin_y))
    arcade.draw_lines(points, CONTOUR_COLOR, CONTOUR_WIDTH)
