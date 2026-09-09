"""Fog of war: which hexes a player currently has vision over.

Only ship *positions* are hidden by fog of war -- the map itself (terrain,
coastline, who owns/controls which port) is always fully known to both
players. This is a pure query with no Arcade dependency; the rendering
layer (tla.rendering.game_view) is what actually hides enemy ship glyphs
and highlights visible hexes.
"""

from __future__ import annotations

from tla.game_state import GameState
from tla.hexgrid import AxialCoord, hexes_in_range
from tla.ship import Ship, ShipKind
from tla.tile import PlayerId


def visible_hexes_for(game_state: GameState, player: PlayerId) -> set[AxialCoord]:
    """Every hex `player` can currently see: within `ship_visibility_radius`
    of any of their own ships, or within `port_and_carrier_visibility_radius`
    of a port they control or one of their own aircraft carriers (which see
    further than that on their own, stacking with the base ship radius)."""
    fow = game_state.config.fow
    visible: set[AxialCoord] = set()
    for ship in game_state.ships_for(player):
        visible.update(hexes_in_range(ship.position, fow.ship_visibility_radius))
        if ship.kind == ShipKind.CARRIER:
            visible.update(hexes_in_range(ship.position, fow.port_and_carrier_visibility_radius))
    for port in game_state.board.controlled_ports_for(player):
        visible.update(hexes_in_range(port, fow.port_and_carrier_visibility_radius))
    return visible


def is_hidden(viewer: PlayerId, ship: Ship, visible_hexes: set[AxialCoord]) -> bool:
    """Whether `ship` should be hidden from `viewer` by fog of war. Always
    False for `viewer`'s own ships. A submerged submarine stays hidden even
    within `viewer`'s normal vision -- stealth beats ordinary detection;
    the only way to spot one is direct combat contact, which the renderer
    handles as a separate, explicit exception rather than a visibility rule
    (see tla.rendering.game_view)."""
    if ship.owner == viewer:
        return False
    if ship.position not in visible_hexes:
        return True
    return ship.kind == ShipKind.SUBMARINE and not ship.surfaced
