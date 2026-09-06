"""Hex/board drawing, plus the coastline contour overlay (flat-top orientation)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import arcade

from tla.board import Board
from tla.elevation import Segment
from tla.hexgrid import AxialCoord, axial_to_pixel
from tla.rendering.ship_glyphs import draw_ship_glyph
from tla.ship import Ship, ShipKind
from tla.tile import PLAYER_A, PLAYER_B, PlayerId, TerrainType

TERRAIN_COLORS = {
    TerrainType.LAND: (86, 125, 70),
    TerrainType.SEA: (43, 92, 138),
}
OUTLINE_COLOR = (30, 30, 30, 140)
PLAYER_COLORS = {
    PLAYER_A: (220, 60, 60),
    PLAYER_B: (60, 100, 230),
}


def _lighten(color: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    r, g, b = color
    return (
        round(r + (255 - r) * amount),
        round(g + (255 - g) * amount),
        round(b + (255 - b) * amount),
    )


def _dim(color: tuple[int, int, int], amount: float = 0.55) -> tuple[int, int, int]:
    """Blend `color` toward neutral gray -- used for a current-player ship
    that has used up its movement this turn."""
    gray = 120
    r, g, b = color
    return (
        round(r + (gray - r) * amount),
        round(g + (gray - g) * amount),
        round(b + (gray - b) * amount),
    )


# Port hexes are tinted a lighter shade of their owner's color, so friendly
# vs. enemy ports are distinguishable at a glance, not just by the anchor icon.
PORT_COLORS = {player: _lighten(color, 0.55) for player, color in PLAYER_COLORS.items()}
CONTOUR_COLOR = (20, 20, 20, 200)
CONTOUR_WIDTH = 2.0
# Background hint of a ship's total range while dragging out a move (subtle
# -- the drawn path itself, not this, is what actually gets committed).
RANGE_PREVIEW_COLOR = (255, 255, 255, 55)
# The exact route drawn so far this drag.
PATH_HIGHLIGHT_COLOR = (255, 165, 0, 140)
PATH_LINE_COLOR = (255, 165, 0, 255)

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


def draw_board(board: Board, hex_size: float) -> None:
    """Draws in raw world space -- an active camera handles panning/viewport."""
    for coord, tile in board.tiles.items():
        center = axial_to_pixel(coord, hex_size)
        corners = hex_corners(center, hex_size * 0.98)
        fill_color = PORT_COLORS[tile.port_owner] if tile.is_port else TERRAIN_COLORS[tile.terrain]
        arcade.draw_polygon_filled(corners, fill_color)
        arcade.draw_polygon_outline(corners, OUTLINE_COLOR, 1)
        if tile.is_port:
            draw_anchor(center, hex_size * 0.85, PLAYER_COLORS[tile.port_owner])


def draw_ships(
    ships: Iterable[Ship], hex_size: float, current_player: PlayerId | None = None
) -> None:
    """Draws in raw world space -- an active camera handles panning/viewport.

    A `current_player` ship with no movement left this turn is dimmed, as a
    visual indicator of which of the active player's ships can still move.
    """
    for ship in ships:
        center = axial_to_pixel(ship.position, hex_size)
        submerged = ship.kind == ShipKind.SUBMARINE and not ship.surfaced
        color = PLAYER_COLORS[ship.owner]
        if ship.owner == current_player and ship.movement_remaining <= 0:
            color = _dim(color)
        draw_ship_glyph(center, hex_size, ship.kind, color, submerged=submerged)


def draw_hex_highlight(coord: AxialCoord, hex_size: float, color: tuple[int, int, int, int]) -> None:
    """Draws in raw world space -- an active camera handles panning/viewport."""
    center = axial_to_pixel(coord, hex_size)
    corners = hex_corners(center, hex_size * 0.98)
    arcade.draw_polygon_filled(corners, color)


def draw_contour(segments: list[Segment]) -> None:
    """Draw precomputed coastline segments (see tla.elevation.marching_squares_segments).

    Draws in raw world space -- an active camera handles panning/viewport.
    """
    if not segments:
        return
    points: list[tuple[float, float]] = []
    for (x0, y0), (x1, y1) in segments:
        points.append((x0, y0))
        points.append((x1, y1))
    arcade.draw_lines(points, CONTOUR_COLOR, CONTOUR_WIDTH)
