"""Win conditions: total fleet elimination, or controlling every port on
the map."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tla.tile import PLAYER_A, PLAYER_B, PlayerId

if TYPE_CHECKING:
    # Only needed for type hints -- GameState.refresh_winner() imports this
    # module at runtime, so importing GameState back here for real would be
    # circular.
    from tla.game_state import GameState


def check_elimination(game_state: GameState) -> PlayerId | None:
    """The other player wins the moment a side has zero ships left."""
    for player, enemy in ((PLAYER_A, PLAYER_B), (PLAYER_B, PLAYER_A)):
        if not game_state.ships_for(player):
            return enemy
    return None


def check_port_control(game_state: GameState) -> PlayerId | None:
    """Whether a single player currently displays as controlling every
    port on the map -- both their own and every one they've captured from
    the opponent (see `Tile.port_display_owner`, which is sticky: a
    captured port stays captured even once the capturing ship moves on).
    A pure snapshot query, not a win check by itself -- see
    `advance_port_control_claim` for how this actually decides the game.
    None if there are no ports at all, or control is currently split
    between the two sides."""
    ports = [tile for tile in game_state.board.tiles.values() if tile.is_port]
    if not ports:
        return None
    controllers = {tile.port_display_owner for tile in ports}
    if len(controllers) == 1:
        return next(iter(controllers))
    return None


def advance_port_control_claim(game_state: GameState) -> None:
    """Call once per full turn boundary (both players have moved -- see
    `tla.turn_manager.end_movement_phase`). Total port control only wins
    once the *same* player still holds every port at the *next* boundary
    after first achieving it -- i.e. having controlled the whole map
    continuously through one entire intervening turn, not just for an
    instant. Gaining full control starts the clock (records the claimant);
    losing it at any boundary check -- even briefly, then regaining it
    later -- resets the clock, since only consecutive boundary snapshots
    are compared. Never overrides an already-decided winner (e.g.
    elimination)."""
    if game_state.winner is not None:
        return
    controller = check_port_control(game_state)
    if controller is not None and controller == game_state.port_control_claimant:
        game_state.winner = controller
    else:
        game_state.port_control_claimant = controller
