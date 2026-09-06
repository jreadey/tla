"""Fine-grained elevation raster and coastline contour extraction.

The raster is sampled at a finer resolution than the hex grid (see
`MapConfig.elevation_supersample`) so that hex terrain classification and the
drawn coastline both derive from one consistent continuous elevation field,
with sea level fixed at elevation 0. No Arcade dependency -- this is pure
math over a 2D array, testable headless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from noise import pnoise2

Point = tuple[float, float]
Segment = tuple[Point, Point]


@dataclass
class ElevationGrid:
    origin_x: float
    origin_y: float
    cell_size: float
    cols: int
    rows: int
    noise_scale_px: float
    octaves: int
    sea_level: float
    noise_base: int
    values: list[list[float]] = field(default_factory=list)

    def sample(self, x: float, y: float) -> float:
        """Continuous elevation at an arbitrary pixel point (not raster-snapped)."""
        n = pnoise2(x / self.noise_scale_px, y / self.noise_scale_px, octaves=self.octaves, base=self.noise_base)
        return n - self.sea_level


def build_elevation_grid(
    bounds: tuple[float, float, float, float],
    hex_pixel_size: float,
    supersample: int,
    noise_scale: float,
    octaves: int,
    sea_level: float,
    noise_base: int,
) -> ElevationGrid:
    min_x, min_y, max_x, max_y = bounds
    cell_size = hex_pixel_size / supersample
    cols = max(2, int((max_x - min_x) / cell_size) + 1)
    rows = max(2, int((max_y - min_y) / cell_size) + 1)

    grid = ElevationGrid(
        origin_x=min_x,
        origin_y=min_y,
        cell_size=cell_size,
        cols=cols,
        rows=rows,
        noise_scale_px=noise_scale * hex_pixel_size,
        octaves=octaves,
        sea_level=sea_level,
        noise_base=noise_base,
    )
    grid.values = [
        [grid.sample(min_x + c * cell_size, min_y + r * cell_size) for c in range(cols)]
        for r in range(rows)
    ]
    return grid


def marching_squares_segments(grid: ElevationGrid) -> list[Segment]:
    """Line segments (in the grid's pixel space) tracing the elevation == 0 contour."""

    def interp(v1: float, p1: Point, v2: float, p2: Point) -> Point:
        t = 0.5 if v1 == v2 else v1 / (v1 - v2)
        return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))

    segments: list[Segment] = []
    values = grid.values
    for r in range(grid.rows - 1):
        y0 = grid.origin_y + r * grid.cell_size
        y1 = y0 + grid.cell_size
        for c in range(grid.cols - 1):
            x0 = grid.origin_x + c * grid.cell_size
            x1 = x0 + grid.cell_size

            # Corners of this raster cell, taken in a cycle A->B->C->D->A.
            v_a, v_b, v_c, v_d = values[r][c], values[r][c + 1], values[r + 1][c + 1], values[r + 1][c]
            p_a, p_b, p_c, p_d = (x0, y0), (x1, y0), (x1, y1), (x0, y1)

            case = (
                (1 if v_a > 0 else 0)
                | (2 if v_b > 0 else 0)
                | (4 if v_c > 0 else 0)
                | (8 if v_d > 0 else 0)
            )
            if case in (0, 15):
                continue

            def e_ab() -> Point:
                return interp(v_a, p_a, v_b, p_b)

            def e_bc() -> Point:
                return interp(v_b, p_b, v_c, p_c)

            def e_cd() -> Point:
                return interp(v_c, p_c, v_d, p_d)

            def e_da() -> Point:
                return interp(v_d, p_d, v_a, p_a)

            if case in (1, 14):
                segments.append((e_da(), e_ab()))
            elif case in (2, 13):
                segments.append((e_ab(), e_bc()))
            elif case in (3, 12):
                segments.append((e_da(), e_bc()))
            elif case in (4, 11):
                segments.append((e_bc(), e_cd()))
            elif case in (6, 9):
                segments.append((e_ab(), e_cd()))
            elif case in (7, 8):
                segments.append((e_cd(), e_da()))
            else:  # 5, 10: saddle -- both diagonally-opposite corners agree, draw both crossings
                segments.append((e_da(), e_ab()))
                segments.append((e_bc(), e_cd()))
    return segments
