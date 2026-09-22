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
    """One port's own progress through `ProductionConfig.build_order` (its
    position in that fixed, shared sequence -- there is no player choice of
    what to build, see tla.production) and the points banked toward the
    kind currently at that position. Points persist even while the port is
    occupied by a friendly ship and can't spawn -- see
    tla.production.run_production -- but both are wiped the instant an
    enemy ship occupies the port, resetting the sequence to the start for
    whoever controls it next; see tla.production.handle_port_capture."""

    next_index: int = 0
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
class BattleLogEntry:
    """One resolved combat round -- appended by `tla.battle.resolve_round`,
    the single chokepoint both the AI's synchronous `run_battle` loop and
    the human UI's manual round-by-round driving
    (`tla.rendering.game_view._resolve_battle_round`) already go through,
    so this is a complete, per-round record of every exchange regardless of
    who's playing. Lives here rather than in `tla.battle` for the same
    reason `TurnStats` does: `tla.battle` imports `GameState` from this
    module, so a type it populates can't also be defined there without a
    circular import.

    One entry per *round*, not per battle -- grouping consecutive entries
    that share the same `attacker_id`/`defender_id` into one printed
    "battle" is left to a reader (see `replay_viewer.py`), not tracked
    here with an explicit id.

    Unlike `TurnStats` (deliberately cumulative across both halves of a
    turn, reset once after `move_b`), this is cleared at the start of
    *every* half-turn (see `tla.turn_manager.TurnManager.
    end_movement_phase`) -- half-turn granularity is what following a
    specific battle actually needs, and a turn-cumulative list would just
    force every consumer to de-duplicate what it already saw last record."""

    attacker_id: int
    attacker_kind: ShipKind
    attacker_owner: PlayerId
    defender_id: int
    defender_kind: ShipKind
    defender_owner: PlayerId
    battle_hex: AxialCoord
    damage_to_defender: int
    damage_to_attacker: int
    # The carrier-bonus component of each side's damage above, broken out
    # -- see `tla.battle.carrier_bonus_for` -- so a reader can compute
    # exact carrier-assist totals instead of estimating them from ship
    # positions after the fact.
    attacker_carrier_bonus: int
    defender_carrier_bonus: int
    defender_hp_after: int
    attacker_hp_after: int
    defender_sunk: bool
    attacker_sunk: bool


@dataclass
class MoveLogEntry:
    """One continuous stretch of hexes a single ship actually traveled --
    appended, in the exact chronological order ships moved, by
    `tla.movement.move_ship`/`move_ship_along_path` and
    `tla.battle.apply_battle_outcome`'s final capture step (a surviving
    attacker's last hex onto a just-sunk defender's position). `path` is
    inclusive of both ends (`path[0]` is where this stretch started,
    `path[-1]` is where it ended), the exact hex-by-hex route -- not just
    origin/destination -- so a reader (the replay viewer) can animate the
    real route instead of a straight-line jump between the two.

    A single ship can appear more than once in one half-turn: an
    engagement's approach is one entry, and (if the attacker wins) the
    final single-hex capture step is a second, separate entry right after
    it. Same half-turn lifecycle as `battle_log` (see `BattleLogEntry`) --
    cleared at both transitions by `tla.turn_manager.TurnManager.
    end_movement_phase`, for the same reason: this is what happened
    *this* half-turn, not a cumulative history."""

    ship_id: int
    kind: ShipKind
    owner: PlayerId
    path: list[AxialCoord]


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
    # Every combat round resolved since the current half-turn began -- see
    # BattleLogEntry for why this has a different (shorter) lifecycle than
    # turn_stats above, despite both being populated by tla.battle.
    battle_log: list[BattleLogEntry] = field(default_factory=list)
    # Every ship movement resolved since the current half-turn began, in the
    # order it actually happened -- see MoveLogEntry for why this has the
    # same half-turn lifecycle as battle_log above.
    move_log: list[MoveLogEntry] = field(default_factory=list)
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
