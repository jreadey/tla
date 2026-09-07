"""Automatic, per-port production: each port keeps its own build queue and
independently earns `ProductionConfig.points_per_turn` points every turn --
not a shared budget split across ports, so a player's total production
scales with how many ports they control."""

from __future__ import annotations

from tla.game_state import GameState, PortProduction
from tla.hexgrid import AxialCoord
from tla.ship import Ship, ShipKind
from tla.tile import PlayerId


def order(game_state: GameState, player: PlayerId, port: AxialCoord, kind: ShipKind) -> None:
    """Queue one `kind` at `port` -- unlimited, no cost check up front. It's
    paid for gradually by run_production, whenever it reaches the front of
    that port's own queue."""
    progress = game_state.players[player].port_production.setdefault(port, PortProduction())
    progress.orders.append(kind)


def handle_port_capture(game_state: GameState, coord: AxialCoord) -> None:
    """Call after any ship's movement (plain move or a battle's approach/
    occupation) ends on `coord`. If `coord` is a port and the ship now
    sitting there belongs to someone other than whoever currently controls
    it (the port's owner, or a previous capturer -- see
    `Tile.port_display_owner`), control changes hands: the previous
    controller's production there -- the order in progress, its banked
    points, and everything else still waiting in its queue -- is lost
    outright, and the new occupant's side becomes the controller, gaining
    the ability to build from this port themselves (see
    `Board.controlled_ports_for`) until someone takes it back. Ownership of
    the port itself (`port_owner`) never changes -- that's fixed at map
    generation and is used only for starting-fleet placement -- only who
    can currently use and build from it changes, which is also what
    `tla.win_condition.check_port_control` tracks.

    Nothing happens if the occupant already is the current controller
    (including a friendly ship simply parking on its own still-controlled
    port), and control does NOT revert just because the ship that captured
    it later leaves -- it stays captured until the other side retakes it.
    Refreshes `game_state.winner` afterward, since a flip here can complete
    total port control.
    """
    tile = game_state.board.get_tile(coord)
    if tile is None or not tile.is_port or tile.port_owner is None:
        return
    occupant = game_state.ship_at(coord)
    if occupant is None:
        return
    previous_controller = tile.port_display_owner
    if occupant.owner != previous_controller:
        game_state.players[previous_controller].port_production.pop(coord, None)
        tile.port_controller = occupant.owner
        game_state.refresh_winner()


def run_production(game_state: GameState, player: PlayerId) -> None:
    """Run one turn of `player`'s production: every port they currently
    control (their own unflipped ports plus any of the opponent's they've
    captured -- see `Board.controlled_ports_for`) that's unoccupied banks
    `ProductionConfig.points_per_turn` points independently -- controlling
    more ports means more total production, not a thinner split of a fixed
    budget. A port occupied by either side's ship earns nothing this turn.
    Points bank even at a port with nothing queued, ready the moment an
    order arrives; a port with an order spawns a ship as soon as its banked
    points cover the cost of the order at the front of its queue -- one
    spawn per port per turn (the hex becomes occupied) -- and any leftover
    carries toward its next order."""
    player_state = game_state.players[player]
    progress = player_state.port_production
    stats = game_state.config.ship_stats.stats
    points_per_turn = game_state.config.production.points_per_turn

    active_ports = [
        p for p in game_state.board.controlled_ports_for(player) if game_state.ship_at(p) is None
    ]
    for port in active_ports:
        state = progress.setdefault(port, PortProduction())
        state.points += points_per_turn
        if not state.orders:
            continue
        kind = state.orders[0]
        if state.points >= stats[kind].cost:
            state.points -= stats[kind].cost
            _spawn_ship(game_state, player, port, kind)
            state.orders.pop(0)


def _spawn_ship(game_state: GameState, player: PlayerId, port: AxialCoord, kind: ShipKind) -> None:
    stats = game_state.config.ship_stats.stats[kind]
    ship_id = game_state.next_ship_id
    game_state.next_ship_id += 1
    game_state.ships[ship_id] = Ship(
        id=ship_id,
        kind=kind,
        owner=player,
        position=port,
        current_hp=stats.hp,
        movement_remaining=0,  # movement phases for this turn are already over
    )
