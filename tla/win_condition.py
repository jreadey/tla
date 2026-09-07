"""Win conditions: total fleet elimination, or besieging every one of the
opponent's ports through one full uncontested turn boundary."""

from __future__ import annotations

from tla.game_state import GameState
from tla.tile import PLAYER_A, PLAYER_B, PlayerId

SIEGE_STREAK_TO_WIN = 2


def check_elimination(game_state: GameState) -> PlayerId | None:
    """The other player wins the moment a side has zero ships left."""
    for player, enemy in ((PLAYER_A, PLAYER_B), (PLAYER_B, PLAYER_A)):
        if not game_state.ships_for(player):
            return enemy
    return None


def check_port_siege(game_state: GameState) -> PlayerId | None:
    """Call once per full turn boundary (after both productions). Updates
    each player's siege streak and returns a winner if one has just been
    besieged -- all of their ports simultaneously enemy-occupied -- for
    `SIEGE_STREAK_TO_WIN` consecutive turn-ends in a row. Any turn-end
    where the siege isn't total resets that player's streak to zero, so a
    port retaken mid-turn breaks it."""
    winner: PlayerId | None = None
    for player, enemy in ((PLAYER_A, PLAYER_B), (PLAYER_B, PLAYER_A)):
        ports = game_state.board.ports_for(player)
        player_state = game_state.players[player]
        besieged = bool(ports) and all(
            (occupant := game_state.ship_at(p)) is not None and occupant.owner == enemy
            for p in ports
        )
        if besieged:
            player_state.siege_streak += 1
            if player_state.siege_streak >= SIEGE_STREAK_TO_WIN:
                winner = enemy
        else:
            player_state.siege_streak = 0
    return winner
