"""Generic axial-indexed dense numpy scalar fields over a hex board.

Pure geometry/array math, no enemy-model or other AI-domain semantics --
`tla.ai.enemy_model` builds its position-belief fields on top of this, and
any other spatial-field need (a future threat-density map, say) can reuse it
directly.

Indexed in raw axial (q, r) via a fixed additive origin offset, not
`tla.board.Board.axial_to_offset`'s odd-q rectangular grid -- that grid's
neighbor deltas depend on column parity, which would break the uniform
array shifts diffusion relies on. Axial neighbor deltas are the same
everywhere, so a uniform offset is sufficient and keeps shifts simple.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from tla.board import Board
from tla.hexgrid import AxialCoord, neighbors

# The 6 axial neighbor deltas, in a fixed order shared by every geometry --
# derived from the public `neighbors()` helper (origin's own neighbors are
# exactly the direction vectors) rather than reaching into hexgrid's
# private `_DIRECTIONS`.
_DIRECTIONS: tuple[AxialCoord, ...] = tuple(neighbors(AxialCoord(0, 0)))

# Sentinel distance for a cell `HexFieldGeometry.distance_grid_to` can't
# reach at all (out of bounds, non-sea, or a genuinely disconnected sea
# pocket) -- larger than any real board's diameter could ever produce,
# mirroring `tla.ai.task_force._UNREACHABLE_SEA_DISTANCE`'s spirit without
# sharing its exact value (this module has no dependency on task_force.py
# -- see the module docstring above -- so the two are independent).
_UNREACHABLE_DISTANCE = 10_000


@lru_cache(maxsize=None)
def _distance_kernel(radius: int) -> np.ndarray:
    """bool[(2*radius+1), (2*radius+1)] -- True at (dr+radius, dq+radius)
    iff the axial hex distance from the center (dq=0, dr=0) is <=
    `radius`. `HexFieldGeometry.to_index` maps a coord's own (q, r) to
    (row, col) via a fixed additive offset (row = r - r0, col = q - q0),
    so a *difference* in (row, col) space is exactly the same as a
    difference in (q, r) space -- this one kernel, keyed only by radius,
    applies at any board position, letting `HexField.mass_near` slice it
    against `self.values` directly instead of walking `hexes_in_range` in
    Python per call (see that function's own docstring for why -- a real
    profiling find: this was 85% of a scored-move-selection turn's total
    time on a large board/fleet, almost all of it in pure-Python looping
    over an array that was already numpy underneath). A small, fixed set
    of radii repeat constantly (`EnemyModel`'s own radius parameters),
    so caching per radius means this is built once, ever, per radius
    actually used."""
    size = 2 * radius + 1
    delta = np.arange(-radius, radius + 1)
    dr, dq = np.meshgrid(delta, delta, indexing="ij")
    axial_distance = (np.abs(dq) + np.abs(dr) + np.abs(dq + dr)) // 2
    return axial_distance <= radius


def _shift(arr: np.ndarray, dr: int, dq: int) -> np.ndarray:
    """Move `arr`'s content by (dr, dq) -- out[r, c] = arr[r - dr, c - dq]
    where that source is in bounds, else 0. Unlike `np.roll`, content
    shifted off an edge is dropped rather than wrapping around."""
    out = np.zeros_like(arr)
    rows, cols = arr.shape
    r_src_start, r_src_end = max(0, -dr), min(rows, rows - dr)
    c_src_start, c_src_end = max(0, -dq), min(cols, cols - dq)
    if r_src_start >= r_src_end or c_src_start >= c_src_end:
        return out
    r_dst_start, r_dst_end = r_src_start + dr, r_src_end + dr
    c_dst_start, c_dst_end = c_src_start + dq, c_src_end + dq
    out[r_dst_start:r_dst_end, c_dst_start:c_dst_end] = arr[r_src_start:r_src_end, c_src_start:c_src_end]
    return out


@dataclass(frozen=True)
class HexFieldGeometry:
    """Everything about one board's shape a `HexField` needs to index and
    diffuse over it, computed once and shared read-only across every field
    over that same board (rebuilding this per field would repeat the same
    board scan for no reason)."""

    rows: int
    cols: int
    q0: int
    r0: int
    sea_mask: np.ndarray  # bool[rows, cols] -- True where a ship could actually be
    # 6x bool[rows, cols], aligned with _DIRECTIONS: valid_neighbor_masks[d][r, c]
    # is True iff cell (r, c) itself is sea and its neighbor in direction d is too.
    valid_neighbor_masks: tuple[np.ndarray, ...]
    stay_denom: np.ndarray  # float[rows, cols] = 1 + this cell's count of sea neighbors

    @classmethod
    def from_board(cls, board: Board) -> "HexFieldGeometry":
        coords = list(board.tiles.keys())
        qs = [c.q for c in coords]
        rs = [c.r for c in coords]
        q0, r0 = min(qs), min(rs)
        cols = max(qs) - q0 + 1
        rows = max(rs) - r0 + 1

        sea_mask = np.zeros((rows, cols), dtype=bool)
        for c in coords:
            if board.is_occupiable(c):
                sea_mask[c.r - r0, c.q - q0] = True

        valid_neighbor_masks = tuple(
            sea_mask & _shift(sea_mask, -d.r, -d.q) for d in _DIRECTIONS
        )

        stay_denom = np.ones((rows, cols), dtype=float)
        for mask in valid_neighbor_masks:
            stay_denom += mask

        return cls(
            rows=rows,
            cols=cols,
            q0=q0,
            r0=r0,
            sea_mask=sea_mask,
            valid_neighbor_masks=valid_neighbor_masks,
            stay_denom=stay_denom,
        )

    def to_index(self, coord: AxialCoord) -> tuple[int, int]:
        return coord.r - self.r0, coord.q - self.q0

    def to_coord(self, row: int, col: int) -> AxialCoord:
        return AxialCoord(col + self.q0, row + self.r0)

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def distance_grid_to(self, target: AxialCoord) -> np.ndarray:
        """int[rows, cols] hex-step distance to `target`, via plain BFS over
        this geometry's own `sea_mask`/`valid_neighbor_masks` -- the exact
        connectivity `HexField.diffuse_step` already walks, not a second,
        independently-derived notion of "sea" (see `tla.ai.task_force.
        sea_distance_field`'s own separate `TerrainType.SEA`-only BFS,
        deliberately not reused here to avoid both a new cross-module
        dependency and any risk of the two disagreeing about which hexes
        count). A port hex is already `True` in `sea_mask` (`Tile.
        occupiable` is `terrain == SEA or is_port`), so targeting a port
        needs no special-casing. Every unreached cell (out of bounds,
        non-sea, or a genuinely disconnected sea pocket) gets
        `_UNREACHABLE_DISTANCE` -- see `HexField.diffuse_step`'s
        `bias_distance_grid` for how that sentinel is meant to be
        consumed (always falls back to ordinary isotropic diffusion,
        never raises or produces a nonsensical bias)."""
        dist = np.full((self.rows, self.cols), _UNREACHABLE_DISTANCE, dtype=int)
        row0, col0 = self.to_index(target)
        if not self.in_bounds(row0, col0) or not self.sea_mask[row0, col0]:
            return dist
        dist[row0, col0] = 0
        frontier = [(row0, col0)]
        while frontier:
            next_frontier: list[tuple[int, int]] = []
            for r, c in frontier:
                d = dist[r, c]
                for mask, delta in zip(self.valid_neighbor_masks, _DIRECTIONS):
                    if not mask[r, c]:
                        continue
                    nr, nc = r + delta.r, c + delta.q
                    if dist[nr, nc] > d + 1:
                        dist[nr, nc] = d + 1
                        next_frontier.append((nr, nc))
            frontier = next_frontier
        return dist


class HexField:
    """One nonnegative scalar "mass" per sea hex of a board. The caller
    decides what mass represents (a probability, an expected ship count,
    ...); this class only guarantees mass never sits on land and that
    `diffuse_step` conserves `total_mass` exactly."""

    def __init__(self, geometry: HexFieldGeometry) -> None:
        self.geometry = geometry
        self.values = np.zeros((geometry.rows, geometry.cols), dtype=float)

    def set_point_mass(self, coord: AxialCoord, mass: float = 1.0) -> None:
        self.values[:, :] = 0.0
        row, col = self.geometry.to_index(coord)
        self.values[row, col] = mass

    def add_point_mass(self, coord: AxialCoord, mass: float) -> None:
        row, col = self.geometry.to_index(coord)
        self.values[row, col] += mass

    def diffuse_step(self, bias_distance_grid: np.ndarray | None = None) -> None:
        """Advance belief by one movement point. With `bias_distance_grid`
        left `None` (the default): each cell's mass splits into `1/(1 + k)`
        shares (k = its count of sea neighbors) -- one share stays, one
        goes to each valid sea-neighbor direction, zero toward a land
        neighbor (so a coastal cell's blocked shares simply fall back into
        its own "stay" bucket). Conserves `total_mass` exactly every step,
        with no renormalization needed.

        Given `bias_distance_grid` (a per-cell distance-to-some-target grid
        -- see `HexFieldGeometry.distance_grid_to`), diffusion becomes
        directed instead of isotropic: for each cell, whichever of its
        valid sea-neighbors have a *strictly smaller* distance-to-target
        than the cell itself ("progress" directions) evenly split the
        cell's *entire* mass -- no stay share, no share toward a
        non-progress direction, modeling "keeps moving toward the target
        at max speed" (see `tla.ai.enemy_model.EnemyModel._diffuse_all`,
        the only caller that ever passes this -- a never-individually-
        sighted ship assumed to be heading somewhere on purpose, not
        wandering). A cell with no progress direction at all (a dead end,
        the target's own cell, or the target unreachable from there --
        `distance_grid_to`'s `_UNREACHABLE_DISTANCE` sentinel naturally
        produces this) falls back to the plain isotropic split above for
        that cell only, so this is always well-defined. `total_mass` stays
        exactly conserved either way: every cell's mass is fully
        redistributed by exactly one of the two rules (selected per cell,
        never split between them), and each rule's shares sum back to that
        cell's own mass by construction."""
        geo = self.geometry
        if bias_distance_grid is None:
            stay_share = np.where(geo.sea_mask, self.values / geo.stay_denom, 0.0)
            new = stay_share.copy()
            for mask, d in zip(geo.valid_neighbor_masks, _DIRECTIONS):
                outgoing = np.where(mask, stay_share, 0.0)
                new += _shift(outgoing, d.r, d.q)
            self.values = new
            return

        progress_masks = [
            mask & (_shift(bias_distance_grid, -d.r, -d.q) < bias_distance_grid)
            for mask, d in zip(geo.valid_neighbor_masks, _DIRECTIONS)
        ]
        num_progress = sum(progress_masks)  # int[rows, cols], 0..6
        has_progress = num_progress > 0

        directed_share = np.where(has_progress, self.values / np.where(has_progress, num_progress, 1), 0.0)
        iso_stay_share = np.where(geo.sea_mask, self.values / geo.stay_denom, 0.0)

        new = np.where(has_progress, 0.0, iso_stay_share)
        for progress_mask, mask, d in zip(progress_masks, geo.valid_neighbor_masks, _DIRECTIONS):
            directed_outgoing = np.where(progress_mask, directed_share, 0.0)
            iso_outgoing = np.where(mask & ~has_progress, iso_stay_share, 0.0)
            new += _shift(directed_outgoing + iso_outgoing, d.r, d.q)
        self.values = new

    def total_mass(self) -> float:
        return float(self.values.sum())

    def add_field(self, other: "HexField") -> None:
        """Merge another field's mass into this one, cell for cell --
        both must share the same geometry (always true for two fields over
        the same board). Used when folding a stale per-ship field's belief
        back into its kind's shared pool field."""
        self.values += other.values

    def renormalize_to(self, target_mass: float) -> None:
        """Rescale so `total_mass()` becomes exactly `target_mass`, keeping
        the current distribution's shape. A no-op if there's currently no
        mass at all to redistribute (nothing to scale up from zero)."""
        current = self.total_mass()
        if current <= 0:
            return
        self.values *= target_mass / current

    def exclude_and_renormalize(self, excluded_indices: list[tuple[int, int]]) -> None:
        """Zero mass at each `(row, col)` in `excluded_indices`, then
        rescale the remainder back up to the total mass this field had
        just before -- for cells that are now *known* impossible (not
        merely less likely), e.g. hexes within a tracker's own current
        vision where an undetected ship genuinely cannot be (see
        `tla.ai.enemy_model.EnemyModel._diffuse_all`). Conserves
        `total_mass` exactly, the same guarantee `diffuse_step` gives --
        except in the degenerate case where every last bit of mass
        happened to fall in `excluded_indices`, which leaves the field at
        zero (nothing left to redistribute; `renormalize_to` already
        no-ops rather than raising)."""
        pre_total = self.total_mass()
        for row, col in excluded_indices:
            self.values[row, col] = 0.0
        self.renormalize_to(pre_total)

    def mass_near(self, coord: AxialCoord, radius: int) -> float:
        """Total mass within `radius` hexes of `coord` -- vectorized via
        `_distance_kernel` rather than walking `hexes_in_range` one hex at
        a time (see that function's own docstring for why). Slices both
        `self.values` and the kernel to their mutual overlap first, so a
        `coord` near the board edge (whose full (2r+1)x(2r+1) window would
        otherwise run off either side) still sums exactly the in-bounds
        cells the old per-hex loop did -- `in_bounds` there and this
        clipping here are the same boundary check, just batched."""
        row, col = self.geometry.to_index(coord)
        kernel = _distance_kernel(radius)
        rows, cols = self.values.shape

        row_start, row_stop = row - radius, row + radius + 1
        col_start, col_stop = col - radius, col + radius + 1
        array_row_start, array_row_stop = max(0, row_start), min(rows, row_stop)
        array_col_start, array_col_stop = max(0, col_start), min(cols, col_stop)
        if array_row_start >= array_row_stop or array_col_start >= array_col_stop:
            return 0.0

        kernel_row_start = array_row_start - row_start
        kernel_row_stop = kernel_row_start + (array_row_stop - array_row_start)
        kernel_col_start = array_col_start - col_start
        kernel_col_stop = kernel_col_start + (array_col_stop - array_col_start)

        values_window = self.values[array_row_start:array_row_stop, array_col_start:array_col_stop]
        kernel_window = kernel[kernel_row_start:kernel_row_stop, kernel_col_start:kernel_col_stop]
        return float(np.sum(values_window, where=kernel_window))

    def most_likely_hex(self) -> AxialCoord | None:
        if self.total_mass() <= 0:
            return None
        row, col = np.unravel_index(np.argmax(self.values), self.values.shape)
        return self.geometry.to_coord(int(row), int(col))

    def as_dict(self) -> dict[AxialCoord, float]:
        """Sparse view of every hex with nonzero mass -- inspection/tests
        only, not meant for hot-path use."""
        result: dict[AxialCoord, float] = {}
        rows, cols = self.values.shape
        for r in range(rows):
            for c in range(cols):
                value = float(self.values[r, c])
                if value > 0:
                    result[self.geometry.to_coord(r, c)] = value
        return result
