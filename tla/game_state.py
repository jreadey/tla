"""Aggregate game state: the map, all ships, and turn/phase tracking. No
Arcade dependency."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from tla.board import Board
from tla.config import Config
from tla.fleet_setup import place_initial_fleets
from tla.hexgrid import AxialCoord
from tla.mapgen import generate_map
from tla.ship import Ship
from tla.tile import PLAYER_A, PlayerId


class TurnPhase(Enum):
    """Phase 3 scope: only the two movement phases exist so far. Production
    and after-action reporting are added in later phases."""

    MOVE_A = "move_a"
    MOVE_B = "move_b"


@dataclass
class GameState:
    config: Config
    board: Board
    ships: dict[int, Ship] = field(default_factory=dict)
    current_player: PlayerId = PLAYER_A
    phase: TurnPhase = TurnPhase.MOVE_A
    turn_number: int = 1

    def ship_at(self, coord: AxialCoord) -> Ship | None:
        for ship in self.ships.values():
            if ship.position == coord:
                return ship
        return None

    def ships_for(self, player: PlayerId) -> list[Ship]:
        return [ship for ship in self.ships.values() if ship.owner == player]


def new_game(config: Config, seed: int) -> GameState:
    board = generate_map(config.map, config.ports, seed=seed)
    ships = place_initial_fleets(board, config, seed)
    return GameState(config=config, board=board, ships=ships)
