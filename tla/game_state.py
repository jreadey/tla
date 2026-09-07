"""Aggregate game state: the map, all ships, turn/phase tracking, and each
player's production economy. No Arcade dependency."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from tla.board import Board
from tla.config import Config
from tla.fleet_setup import place_initial_fleets
from tla.hexgrid import AxialCoord
from tla.mapgen import generate_map
from tla.ship import Ship, ShipKind
from tla.tile import PLAYER_A, PLAYER_B, PlayerId


class TurnPhase(Enum):
    MOVE_A = "move_a"
    MOVE_B = "move_b"


@dataclass
class PortProduction:
    """One port's own build queue and the points banked toward the order at
    its front. Points persist even while the port is occupied by a friendly
    ship and can't spawn -- see tla.production.run_production -- but the
    entire queue is wiped the instant an enemy ship occupies the port; see
    tla.production.handle_port_capture."""

    orders: list[ShipKind] = field(default_factory=list)
    points: int = 0


@dataclass
class PlayerState:
    # Each of the player's ports keeps its own queue -- see PortProduction
    # and tla.production. A port with no entry here has never had anything
    # queued (equivalent to an empty queue).
    port_production: dict[AxialCoord, PortProduction] = field(default_factory=dict)
    # Consecutive turn-ends this player's ports have all been enemy-held;
    # see tla.win_condition.
    siege_streak: int = 0


@dataclass
class GameState:
    config: Config
    board: Board
    ships: dict[int, Ship] = field(default_factory=dict)
    current_player: PlayerId = PLAYER_A
    phase: TurnPhase = TurnPhase.MOVE_A
    turn_number: int = 1
    players: dict[PlayerId, PlayerState] = field(
        default_factory=lambda: {PLAYER_A: PlayerState(), PLAYER_B: PlayerState()}
    )
    next_ship_id: int = 1
    winner: PlayerId | None = None

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
    next_ship_id = max(ships.keys(), default=0) + 1
    return GameState(config=config, board=board, ships=ships, next_ship_id=next_ship_id)
