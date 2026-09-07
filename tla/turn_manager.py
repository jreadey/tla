"""Turn/phase orchestration: movement for both players, then -- automatically,
no player input needed -- one turn of production for each, then (if the game
isn't already won) a new turn's movement."""

from __future__ import annotations

from tla.game_state import GameState, TurnPhase
from tla.production import run_production
from tla.tile import PLAYER_A, PLAYER_B, PlayerId
from tla.win_condition import check_port_siege


def start_movement_phase(game_state: GameState, player: PlayerId) -> None:
    """Reset movement budget and submarine toggle flags for one player's
    ships, at the start of their movement phase."""
    stats_map = game_state.config.ship_stats.stats
    for ship in game_state.ships_for(player):
        stats = stats_map[ship.kind]
        ship.movement_remaining = ship.max_movement(stats)
        ship.toggled_pre_move = False
        ship.toggled_post_move = False


class TurnManager:
    def __init__(self, game_state: GameState) -> None:
        self.game_state = game_state

    def end_movement_phase(self) -> None:
        """Called when the current player is done moving. Advances to the
        other player's movement phase, or -- once both players have moved --
        runs one turn of automatic production for both, checks the
        port-siege win condition, and starts a new turn's movement if
        nobody has just won."""
        gs = self.game_state
        if gs.phase == TurnPhase.MOVE_A:
            gs.phase = TurnPhase.MOVE_B
            gs.current_player = PLAYER_B
            start_movement_phase(gs, PLAYER_B)
        elif gs.phase == TurnPhase.MOVE_B:
            run_production(gs, PLAYER_A)
            run_production(gs, PLAYER_B)
            winner = check_port_siege(gs)
            if winner is not None:
                gs.winner = winner
                return
            gs.phase = TurnPhase.MOVE_A
            gs.current_player = PLAYER_A
            gs.turn_number += 1
            start_movement_phase(gs, PLAYER_A)
