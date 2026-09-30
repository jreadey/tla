"""What an AI player currently knows about the enemy fleet.

The only fog-of-war seam the rest of tla.ai touches -- every heuristic in
tla.ai.scoring and tla.ai.policy consumes `enemy_ships_visible_to`'s result
exclusively, never `game_state.ships` directly for enemy-owned ships, so the
AI can never react to a ship it wouldn't actually have vision into.
"""

from __future__ import annotations

from tla.fow import is_hidden, visible_hexes_for
from tla.game_state import GameState
from tla.ship import Ship
from tla.tile import PlayerId


def enemy_ships_visible_to(game_state: GameState, player: PlayerId) -> dict[int, Ship]:
    """Every enemy ship `player`'s AI currently knows about. If fog of war
    is disabled (`config.fow.enabled` is False), this is simply every enemy
    ship in the game -- there's nothing to hide. If enabled, this is exactly
    what a human in the same seat would see: filtered through
    `fow.visible_hexes_for`/`fow.is_hidden`, so a submerged enemy submarine
    outside direct combat is invisible to the AI too."""
    enemy_ships = {
        ship_id: ship for ship_id, ship in game_state.ships.items() if ship.owner != player
    }
    if not game_state.config.fow.enabled:
        return enemy_ships
    visible_hexes = visible_hexes_for(game_state, player)
    return {
        ship_id: ship
        for ship_id, ship in enemy_ships.items()
        if not is_hidden(player, ship, visible_hexes)
    }
