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
    """A player wins the moment every port on the map -- both their own and
    every one they've captured from the opponent -- displays as theirs (see
    `Tile.port_display_owner`, which is sticky: a captured port stays
    captured even once the capturing ship moves on). None if there are no
    ports at all, or control is currently split between the two sides."""
    ports = [tile for tile in game_state.board.tiles.values() if tile.is_port]
    if not ports:
        return None
    controllers = {tile.port_display_owner for tile in ports}
    if len(controllers) == 1:
        return next(iter(controllers))
    return None
