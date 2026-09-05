"""Perlin-noise-driven map generation.

Pure function: given config and a seed, produces a `Board`. No Arcade
dependency, no reliance on global RNG state, so results are deterministic and
easy to unit test.
"""

from __future__ import annotations

import random

from noise import pnoise2

from tla.board import Board
from tla.config import MapConfig, PortConfig
from tla.hexgrid import AxialCoord, distance, offset_to_axial
from tla.tile import PLAYER_A, PLAYER_B, Tile, TerrainType


def generate_map(
    map_config: MapConfig, port_config: PortConfig, seed: int | None = None
) -> Board:
    if seed is None:
        seed = map_config.seed if map_config.seed is not None else random.randrange(1_000_000)
    noise_base = seed % 256

    board = Board(width=map_config.width, height=map_config.height)
    for col in range(map_config.width):
        for row in range(map_config.height):
            n = pnoise2(
                col / map_config.noise_scale,
                row / map_config.noise_scale,
                octaves=map_config.octaves,
                base=noise_base,
            )
            coord = offset_to_axial(col, row)
            board.tiles[coord] = Tile(coord=coord, terrain=_bucket_terrain(n, map_config))

    _place_ports(board, port_config, seed)
    return board


def _bucket_terrain(n: float, map_config: MapConfig) -> TerrainType:
    if n > map_config.land_threshold:
        return TerrainType.LAND
    if n > map_config.shore_threshold:
        return TerrainType.SHORE
    return TerrainType.SEA


def _place_ports(board: Board, port_config: PortConfig, seed: int) -> None:
    rng = random.Random(seed)
    shore_coords = [c for c, t in board.tiles.items() if t.terrain == TerrainType.SHORE]
    rng.shuffle(shore_coords)

    needed = port_config.ports_per_player * 2

    chosen: list[AxialCoord] = []
    for coord in shore_coords:
        if len(chosen) >= needed:
            break
        if all(distance(coord, other) >= port_config.min_port_spacing for other in chosen):
            chosen.append(coord)

    if len(chosen) < needed:
        remaining = [c for c in shore_coords if c not in chosen]
        chosen.extend(remaining[: needed - len(chosen)])

    if len(chosen) < needed:
        raise RuntimeError(
            f"Only {len(shore_coords)} shore tiles available, need {needed} for "
            f"{port_config.ports_per_player} ports/player. Adjust map size or "
            "land/shore thresholds."
        )

    # Split by longitude (q) so each player's ports cluster on one side of the
    # map -- a simple fairness heuristic for opposite-side starts.
    chosen.sort(key=lambda c: c.q)
    half = port_config.ports_per_player
    for coord in chosen[:half]:
        tile = board.tiles[coord]
        tile.is_port = True
        tile.port_owner = PLAYER_A
    for coord in chosen[half:needed]:
        tile = board.tiles[coord]
        tile.is_port = True
        tile.port_owner = PLAYER_B
