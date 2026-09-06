"""Random initial fleet placement, near each player's own ports."""

from __future__ import annotations

import random

from tla.board import Board
from tla.config import Config
from tla.hexgrid import AxialCoord, hexes_in_range
from tla.ship import Ship
from tla.tile import PLAYER_A, PLAYER_B


def place_initial_fleets(board: Board, config: Config, seed: int) -> dict[int, Ship]:
    rng = random.Random(seed)
    ships: dict[int, Ship] = {}
    next_id = 1

    for player in (PLAYER_A, PLAYER_B):
        ports = board.ports_for(player)
        occupied: set[AxialCoord] = set()
        for kind, count in config.fleet.counts.items():
            stats = config.ship_stats.stats[kind]
            for _ in range(count):
                coord = _pick_start_hex(board, ports, occupied, rng)
                occupied.add(coord)
                ships[next_id] = Ship(
                    id=next_id,
                    kind=kind,
                    owner=player,
                    position=coord,
                    current_hp=stats.hp,
                    movement_remaining=stats.movement,
                )
                next_id += 1

    return ships


def _pick_start_hex(
    board: Board, ports: list[AxialCoord], occupied: set[AxialCoord], rng: random.Random
) -> AxialCoord:
    max_radius = board.width + board.height
    radius = 1
    while radius <= max_radius:
        candidates: set[AxialCoord] = set()
        for port in ports:
            candidates.update(hexes_in_range(port, radius))
        valid = [c for c in candidates if board.is_occupiable(c) and c not in occupied]
        if valid:
            return rng.choice(valid)
        radius += 1
    raise RuntimeError("Could not find enough sea room near a port to place the starting fleet.")
