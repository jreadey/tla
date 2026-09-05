"""Axial hex-coordinate math for a flat-top, odd-q offset hex grid.

No dependency on Arcade or any rendering concern lives here — this module is
pure geometry so the rules engine can be unit tested headless.
"""

from __future__ import annotations

from typing import NamedTuple


class AxialCoord(NamedTuple):
    """A hex position in axial coordinates (q, r)."""

    q: int
    r: int

    def __add__(self, other: "AxialCoord") -> "AxialCoord":
        return AxialCoord(self.q + other.q, self.r + other.r)


# The six axial neighbor directions. Orientation (flat-top vs pointy-top) only
# affects pixel conversion, not this adjacency math.
_DIRECTIONS: tuple[AxialCoord, ...] = (
    AxialCoord(1, 0),
    AxialCoord(1, -1),
    AxialCoord(0, -1),
    AxialCoord(-1, 0),
    AxialCoord(-1, 1),
    AxialCoord(0, 1),
)


def neighbors(coord: AxialCoord) -> list[AxialCoord]:
    """Return the 6 hexes adjacent to coord, in no particular guaranteed order."""
    return [coord + d for d in _DIRECTIONS]


def distance(a: AxialCoord, b: AxialCoord) -> int:
    """Hex distance between two axial coordinates."""
    dq = a.q - b.q
    dr = a.r - b.r
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def hexes_in_range(center: AxialCoord, radius: int) -> set[AxialCoord]:
    """All hexes within `radius` steps of center (inclusive), including center."""
    if radius < 0:
        return set()
    results: set[AxialCoord] = set()
    for dq in range(-radius, radius + 1):
        r_min = max(-radius, -dq - radius)
        r_max = min(radius, -dq + radius)
        for dr in range(r_min, r_max + 1):
            results.add(AxialCoord(center.q + dq, center.r + dr))
    return results


def offset_to_axial(col: int, row: int) -> AxialCoord:
    """Convert odd-q vertical offset coordinates (col, row) to axial."""
    q = col
    r = row - (col - (col & 1)) // 2
    return AxialCoord(q, r)


def axial_to_offset(coord: AxialCoord) -> tuple[int, int]:
    """Convert axial coordinates to odd-q vertical offset (col, row)."""
    col = coord.q
    row = coord.r + (coord.q - (coord.q & 1)) // 2
    return col, row
