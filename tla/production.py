"""Automatic, per-port production: each port keeps its own build queue, and
each turn's production budget is split evenly across a player's currently
unoccupied ports and invested in whatever each one is building."""

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
    sitting there belongs to someone other than the port's owner, that
    port's entire production -- the order in progress, its banked points,
    and everything else still waiting in its queue -- is lost outright.
    Occupying a port disrupts it completely rather than merely pausing it;
    ownership of the port itself doesn't change, so it resumes from empty
    once free again."""
    tile = game_state.board.get_tile(coord)
    if tile is None or not tile.is_port or tile.port_owner is None:
        return
    occupant = game_state.ship_at(coord)
    if occupant is not None and occupant.owner != tile.port_owner:
        game_state.players[tile.port_owner].port_production.pop(coord, None)


def run_production(game_state: GameState, player: PlayerId) -> None:
    """Run one turn of `player`'s production: split this turn's point
    budget evenly across their currently-unoccupied ports that have
    something queued, and spawn a ship at any port whose banked points now
    cover the cost of the order at the front of its own queue. A port
    occupied by either side's ship is skipped for this turn's share
    entirely -- its points simply aren't distributed rather than being
    wasted on a port that couldn't use them anyway. A port that just
    spawned a ship stops for the turn (the hex is now occupied); any
    leftover points stay banked toward its next order."""
    player_state = game_state.players[player]
    progress = player_state.port_production
    stats = game_state.config.ship_stats.stats

    active_ports = [p for p in game_state.board.ports_for(player) if game_state.ship_at(p) is None]
    building_ports = [p for p in active_ports if progress.get(p) and progress[p].orders]
    if not building_ports:
        return

    budget = game_state.config.production.points_per_turn
    share, remainder = divmod(budget, len(building_ports))

    for i, port in enumerate(building_ports):
        state = progress[port]
        state.points += share + (1 if i < remainder else 0)
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
