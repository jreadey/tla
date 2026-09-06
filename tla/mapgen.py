"""Elevation-driven map generation.

Builds a fine elevation raster (see tla.elevation), then classifies each hex
by the fraction of its raster samples above sea level: more than
`land_area_threshold` -> LAND, otherwise -> SEA. There is no separate shore
terrain; ports are placed on coastal LAND hexes (see `_place_ports`). Pure
function of config + seed, no reliance on global RNG state, so results are
deterministic and easy to unit test.
"""

from __future__ import annotations

import random

from tla.board import Board
from tla.config import MapConfig, PortConfig
from tla.elevation import ElevationGrid, build_elevation_grid
from tla.hexgrid import AxialCoord, axial_to_pixel, distance, neighbors, offset_to_axial, pixel_to_axial
from tla.tile import PLAYER_A, PLAYER_B, Tile, TerrainType


def generate_map(
    map_config: MapConfig, port_config: PortConfig, seed: int | None = None
) -> Board:
    if seed is None:
        seed = map_config.seed if map_config.seed is not None else random.randrange(1_000_000)
    noise_base = seed % 256
    hex_size = map_config.hex_pixel_size

    coords = [
        offset_to_axial(col, row)
        for col in range(map_config.width)
        for row in range(map_config.height)
    ]
    coord_set = set(coords)

    pixels = [axial_to_pixel(c, hex_size) for c in coords]
    pad = hex_size
    bounds = (
        min(x for x, _ in pixels) - pad,
        min(y for _, y in pixels) - pad,
        max(x for x, _ in pixels) + pad,
        max(y for _, y in pixels) + pad,
    )

    elevation = build_elevation_grid(
        bounds=bounds,
        hex_pixel_size=hex_size,
        supersample=map_config.elevation_supersample,
        noise_scale=map_config.noise_scale,
        octaves=map_config.octaves,
        sea_level=map_config.sea_level,
        noise_base=noise_base,
    )

    hex_samples: dict[AxialCoord, list[float]] = {}
    for row_idx in range(elevation.rows):
        y = elevation.origin_y + row_idx * elevation.cell_size
        for col_idx in range(elevation.cols):
            x = elevation.origin_x + col_idx * elevation.cell_size
            coord = pixel_to_axial(x, y, hex_size)
            if coord not in coord_set:
                continue
            hex_samples.setdefault(coord, []).append(elevation.values[row_idx][col_idx])

    board = Board(
        width=map_config.width,
        height=map_config.height,
        hex_pixel_size=hex_size,
        elevation=elevation,
    )
    for coord in coords:
        values = hex_samples.get(coord)
        if not values:
            # Rare: a hex too small to catch any raster sample point. Fall
            # back to the continuous value at its own center.
            x, y = axial_to_pixel(coord, hex_size)
            values = [elevation.sample(x, y)]
        board.tiles[coord] = Tile(
            coord=coord, terrain=_classify(values, map_config.land_area_threshold)
        )

    _place_ports(board, port_config, seed)
    return board


def _classify(values: list[float], land_area_threshold: float) -> TerrainType:
    land_fraction = sum(1 for v in values if v > 0) / len(values)
    return TerrainType.LAND if land_fraction > land_area_threshold else TerrainType.SEA


def _largest_sea_component(board: Board) -> set[AxialCoord]:
    """The biggest connected body of SEA tiles (flood fill over sea-sea
    adjacency). Small enclosed ponds end up as separate, smaller components,
    so ports can be required to border this one instead -- otherwise a port
    could open onto a landlocked puddle with no way for ships to reach the
    open ocean, or for the enemy to ever besiege it."""
    sea_tiles = {c for c, t in board.tiles.items() if t.terrain == TerrainType.SEA}
    seen: set[AxialCoord] = set()
    largest: set[AxialCoord] = set()
    for start in sea_tiles:
        if start in seen:
            continue
        component: set[AxialCoord] = set()
        stack = [start]
        while stack:
            coord = stack.pop()
            if coord in component:
                continue
            component.add(coord)
            for n in neighbors(coord):
                if n in sea_tiles and n not in component:
                    stack.append(n)
        seen |= component
        if len(component) > len(largest):
            largest = component
    return largest


def _place_ports(board: Board, port_config: PortConfig, seed: int) -> None:
    rng = random.Random(seed)
    main_sea = _largest_sea_component(board)

    def is_coastal_land(coord: AxialCoord) -> bool:
        tile = board.tiles[coord]
        if tile.terrain != TerrainType.LAND:
            return False
        return any(n in main_sea for n in neighbors(coord))

    coastal_coords = [c for c in board.tiles if is_coastal_land(c)]
    needed_per_player = port_config.ports_per_player

    if len(coastal_coords) < needed_per_player * 2:
        raise RuntimeError(
            f"Only {len(coastal_coords)} coastal land tiles available, need "
            f"{needed_per_player * 2} for {needed_per_player} ports/player. Adjust "
            "map size or sea_level/land_area_threshold."
        )

    # Two seed hexes as far apart as possible, so each player's ports cluster
    # near their own seed and away from the other player's -- this is what
    # keeps a port's average distance to friendly ports below its average
    # distance to enemy ports.
    shuffled = coastal_coords[:]
    rng.shuffle(shuffled)
    seed_a = shuffled[0]
    seed_b = max(coastal_coords, key=lambda c: distance(c, seed_a))

    cluster1 = _cluster_near(seed_a, coastal_coords, needed_per_player, port_config.min_port_spacing)
    remaining = [c for c in coastal_coords if c not in cluster1]
    cluster2 = _cluster_near(seed_b, remaining, needed_per_player, port_config.min_port_spacing)

    # Player A is on the west (smaller q) side of the map, Player B on the
    # east -- a stable, predictable left/right layout rather than whichever
    # cluster happened to get the randomly-picked first seed.
    if sum(c.q for c in cluster1) <= sum(c.q for c in cluster2):
        ports_a, ports_b = cluster1, cluster2
    else:
        ports_a, ports_b = cluster2, cluster1

    for coord in ports_a:
        tile = board.tiles[coord]
        tile.is_port = True
        tile.port_owner = PLAYER_A
    for coord in ports_b:
        tile = board.tiles[coord]
        tile.is_port = True
        tile.port_owner = PLAYER_B


def _cluster_near(
    anchor: AxialCoord, pool: list[AxialCoord], count: int, spacing: int
) -> list[AxialCoord]:
    """Pick `count` hexes from `pool`, closest to `anchor` first, spread apart
    by at least `spacing` where the pool allows it."""
    by_distance = sorted(pool, key=lambda c: distance(c, anchor))
    chosen: list[AxialCoord] = []
    for coord in by_distance:
        if len(chosen) >= count:
            break
        if all(distance(coord, other) >= spacing for other in chosen):
            chosen.append(coord)
    if len(chosen) < count:
        remaining = [c for c in by_distance if c not in chosen]
        chosen.extend(remaining[: count - len(chosen)])
    return chosen
