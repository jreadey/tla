"""Turn/phase orchestration.

Phase 3 scope: cycling between the two movement phases and resetting each
player's ships (movement budget, submarine toggle flags) at the start of
their phase. After-action reporting and production are added in later
phases.
"""

from __future__ import annotations

from tla.game_state import GameState, TurnPhase
from tla.tile import PLAYER_A, PLAYER_B, PlayerId


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
        other player's movement phase, or starts a new turn if both players
        have now moved."""
        gs = self.game_state
        if gs.phase == TurnPhase.MOVE_A:
            gs.phase = TurnPhase.MOVE_B
            gs.current_player = PLAYER_B
        else:
            gs.phase = TurnPhase.MOVE_A
            gs.current_player = PLAYER_A
            gs.turn_number += 1
        start_movement_phase(gs, gs.current_player)
