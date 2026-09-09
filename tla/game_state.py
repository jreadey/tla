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
from tla.win_condition import check_elimination


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


@dataclass
class TurnStats:
    """One player's running battle tally for the turn currently in
    progress -- both their own and the opponent's movement phases, since
    the after-action report (tla.rendering.game_view) covers the whole
    turn at once. Accumulated by tla.battle.resolve_round/
    apply_battle_outcome as battles happen, and reset by
    tla.turn_manager.end_movement_phase once a new turn actually starts
    (after the report has had a chance to show it)."""

    hp_dealt: int = 0
    hp_taken: int = 0
    ships_lost: list[ShipKind] = field(default_factory=list)


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
    turn_stats: dict[PlayerId, TurnStats] = field(
        default_factory=lambda: {PLAYER_A: TurnStats(), PLAYER_B: TurnStats()}
    )
    # Whoever controlled every port on the map as of the most recent full
    # turn boundary -- None if no single player did. See
    # tla.win_condition.advance_port_control_claim: total port control
    # only wins once the SAME player holds this two boundaries running
    # (i.e. continuously through one full intervening turn), so this is
    # the "clock" that tracks whether that streak is still alive.
    port_control_claimant: PlayerId | None = None

    def ship_at(self, coord: AxialCoord) -> Ship | None:
        for ship in self.ships.values():
            if ship.position == coord:
                return ship
        return None

    def ships_for(self, player: PlayerId) -> list[Ship]:
        return [ship for ship in self.ships.values() if ship.owner == player]

    def refresh_winner(self) -> None:
        """Re-evaluate the elimination win condition and set `winner` if
        it's now met. Never un-sets an already-decided winner, so this is
        safe to call speculatively -- it's meant to be called by the rules
        layer itself (movement, battle) right after any mutation that
        could end the game: a ship sunk, rather than left to whatever's
        driving the game (a human UI or an AI) to remember to check
        afterward. Total port control is deliberately NOT checked here --
        unlike elimination, it doesn't win instantly; see
        tla.win_condition.advance_port_control_claim, called once per full
        turn boundary by tla.turn_manager."""
        if self.winner is not None:
            return
        self.winner = check_elimination(self)


def new_game(config: Config, seed: int) -> GameState:
    board = generate_map(config.map, config.ports, seed=seed)
    ships = place_initial_fleets(board, config, seed)
    next_ship_id = max(ships.keys(), default=0) + 1
    return GameState(config=config, board=board, ships=ships, next_ship_id=next_ship_id)
