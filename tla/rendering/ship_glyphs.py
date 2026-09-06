"""Simple top-down ship silhouettes, built from plain polygons (no image assets).

Each ship kind gets a distinct hull outline (built from a bow-to-stern
half-width profile, mirrored to form a closed polygon) plus 0-2 small accent
shapes, so the six kinds stay visually distinguishable at small hex-icon
scale by both size and silhouette:

- Battleship: long, narrow, pointed bow, three turret roundels.
- Aircraft Carrier: widest, blunt bow/stern (flat deck), offset island.
- Cruiser: medium, pointed bow, one turret roundel (fewer than battleship).
- Destroyer: small and sleek, pointed at both ends.
- Submarine: narrow rounded capsule, with a conning-tower sail.
- Patrol Boat: smallest, a simple wedge.

All coordinates are local unit-space (bow at +y, beam along x); a single
scale factor converts to pixels at draw time. No Arcade objects are built at
import time, only plain point lists, so this stays easy to unit test.
"""

from __future__ import annotations

from dataclasses import dataclass

import arcade

from tla.ship import ShipKind

Point = tuple[float, float]


@dataclass(frozen=True)
class Turret:
    center: Point
    radius: float

# Ships are drawn noticeably larger than the anchor icon relative to a hex,
# matching how counters are usually drawn oversized for legibility.
SHIP_SCALE = 1.4
OUTLINE_COLOR = (20, 20, 20, 220)
OUTLINE_WIDTH = 1.5


def _hull_from_profile(profile: list[tuple[float, float]]) -> list[Point]:
    """Build a closed hull polygon from a bow->stern (y, half_width) profile."""
    right_side = [(hw, y) for y, hw in profile]
    left_side = [(-hw, y) for y, hw in reversed(profile)]
    return right_side + left_side


_HULL_PROFILES: dict[ShipKind, list[tuple[float, float]]] = {
    ShipKind.BATTLESHIP: [
        (0.50, 0.00),
        (0.38, 0.11),
        (0.10, 0.15),
        (-0.20, 0.15),
        (-0.42, 0.09),
        (-0.50, 0.04),
    ],
    ShipKind.CARRIER: [
        (0.475, 0.19),
        (0.30, 0.21),
        (-0.30, 0.21),
        (-0.475, 0.17),
    ],
    ShipKind.CRUISER: [
        (0.375, 0.00),
        (0.20, 0.10),
        (0.00, 0.13),
        (-0.25, 0.11),
        (-0.375, 0.05),
    ],
    ShipKind.DESTROYER: [
        (0.275, 0.00),
        (0.10, 0.08),
        (-0.10, 0.08),
        (-0.275, 0.00),
    ],
    ShipKind.SUBMARINE: [
        (0.325, 0.00),
        (0.28, 0.045),
        (0.15, 0.06),
        (-0.15, 0.06),
        (-0.28, 0.045),
        (-0.325, 0.00),
    ],
    ShipKind.PATROL_BOAT: [
        (0.175, 0.00),
        (-0.05, 0.08),
        (-0.175, 0.06),
    ],
}

# Turrets are drawn as small circles on the centerline -- battleship gets
# three (its main-gun turrets), cruiser just one, so the two aren't
# distinguishable by size alone.
_TURRETS: dict[ShipKind, list[Turret]] = {
    ShipKind.BATTLESHIP: [
        Turret((0.0, 0.30), 0.045),
        Turret((0.0, 0.02), 0.045),
        Turret((0.0, -0.28), 0.045),
    ],
    ShipKind.CRUISER: [Turret((0.0, 0.05), 0.04)],
    ShipKind.CARRIER: [],
    ShipKind.DESTROYER: [],
    ShipKind.SUBMARINE: [],
    ShipKind.PATROL_BOAT: [],
}

_STRUCTURE_ACCENTS: dict[ShipKind, list[list[Point]]] = {
    ShipKind.CARRIER: [
        # Island superstructure, offset to one side toward the stern.
        [(0.085, -0.13), (0.175, -0.13), (0.175, 0.03), (0.085, 0.03)]
    ],
    ShipKind.SUBMARINE: [
        # Conning tower / sail, on the centerline just aft of amidships.
        [(-0.03, -0.11), (0.03, -0.11), (0.03, 0.01), (-0.03, 0.01)]
    ],
    ShipKind.BATTLESHIP: [],
    ShipKind.CRUISER: [],
    ShipKind.DESTROYER: [],
    ShipKind.PATROL_BOAT: [],
}

SHIP_HULLS: dict[ShipKind, list[Point]] = {
    kind: _hull_from_profile(profile) for kind, profile in _HULL_PROFILES.items()
}


def draw_ship_glyph(
    center: tuple[float, float],
    hex_size: float,
    kind: ShipKind,
    color: tuple[int, int, int],
    *,
    submerged: bool = False,
) -> None:
    """Draw the top-down silhouette for `kind`, centered at `center`.

    `submerged` (meaningful only for a submarine) draws a hollow "ghost"
    outline instead of a solid fill, with the conning-tower sail hidden
    since it wouldn't be visible from above once submerged. A fade/alpha
    treatment was tried and rejected: blending the player color over the
    sea tile shifts its hue into a muddy blend rather than reading as
    "faded", especially at small hex-icon scale.
    """
    cx, cy = center
    scale = hex_size * SHIP_SCALE
    filled = not submerged

    def to_pixels(points: list[Point]) -> list[Point]:
        return [(cx + x * scale, cy + y * scale) for x, y in points]

    hull = to_pixels(SHIP_HULLS[kind])
    if filled:
        arcade.draw_polygon_filled(hull, color)
        arcade.draw_polygon_outline(hull, OUTLINE_COLOR, OUTLINE_WIDTH)
    else:
        arcade.draw_polygon_outline(hull, color, max(OUTLINE_WIDTH, 2.0))

    if submerged:
        return

    for accent in _STRUCTURE_ACCENTS[kind]:
        pixels = to_pixels(accent)
        arcade.draw_polygon_filled(pixels, color)
        arcade.draw_polygon_outline(pixels, OUTLINE_COLOR, OUTLINE_WIDTH)

    for turret in _TURRETS[kind]:
        tx, ty = turret.center
        px, py = cx + tx * scale, cy + ty * scale
        radius = turret.radius * scale
        arcade.draw_circle_filled(px, py, radius, color)
        arcade.draw_circle_outline(px, py, radius, OUTLINE_COLOR, OUTLINE_WIDTH)
