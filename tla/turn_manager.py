"""Turn/phase orchestration: movement for both players, then -- automatically,
no player input needed -- one turn of production for each, then a new
turn's movement. Elimination is checked immediately as it happens (see
tla.rendering.game_view), not at a turn boundary, so by the time this runs
the game is never already won that way -- ended input is blocked well
before end_movement_phase would be called again. Total port control is
different: it's only ever (re-)evaluated here, once per full turn
boundary, since holding every port has to survive a whole intervening
turn before it wins -- see tla.win_condition.advance_port_control_claim."""

from __future__ import annotations

from tla.game_state import GameState, TurnPhase, TurnStats
from tla.production import run_production
from tla.tile import PLAYER_A, PLAYER_B, PlayerId
from tla.win_condition import advance_port_control_claim


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
        runs one turn of automatic production for both and starts a new
        turn's movement."""
        gs = self.game_state
        if gs.phase == TurnPhase.MOVE_A:
            gs.phase = TurnPhase.MOVE_B
            gs.current_player = PLAYER_B
            start_movement_phase(gs, PLAYER_B)
        elif gs.phase == TurnPhase.MOVE_B:
            # The turn boundary: both players have now moved, so this is
            # exactly the once-per-turn checkpoint total port control gets
            # evaluated at -- see advance_port_control_claim. Checked
            # before production/reset below, though order doesn't actually
            # matter here since neither touches port control.
            advance_port_control_claim(gs)
            run_production(gs, PLAYER_A)
            run_production(gs, PLAYER_B)
            gs.phase = TurnPhase.MOVE_A
            gs.current_player = PLAYER_A
            gs.turn_number += 1
            # Reset now that the turn that just ended is fully behind us --
            # a caller wanting to show an after-action report for it (see
            # tla.rendering.game_view) must read gs.turn_stats before
            # calling this, since it won't reflect that turn afterward.
            gs.turn_stats = {PLAYER_A: TurnStats(), PLAYER_B: TurnStats()}
            start_movement_phase(gs, PLAYER_A)
