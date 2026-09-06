"""Movement legality and submarine surface/submerge toggling.

An enemy-occupied hex is reachable only as the very last hex of a move --
entering it triggers a battle (see tla.battle) instead of just occupying
the hex, so it can't be passed through on the way to somewhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tla.game_state import GameState
from tla.hexgrid import AxialCoord, neighbors
from tla.ship import Ship, ShipKind, ShipStats
from tla.tile import PlayerId, TerrainType

StepKind = Literal["blocked", "open", "enemy"]


def _classify_step(
    game_state: GameState, mover_owner: PlayerId, to_coord: AxialCoord, leaving_origin_port: bool
) -> StepKind:
    """"blocked": can't go there at all. "open": empty, can continue past.
    "enemy": occupied by the other side -- legal only as the final step of a
    move, since arriving there triggers a battle rather than occupying it.
    A friendly-occupied hex is always "blocked" -- no stacking, and (for now)
    no passing through either."""
    tile = game_state.board.get_tile(to_coord)
    if tile is None or not tile.occupiable:
        return "blocked"
    if leaving_origin_port and tile.terrain != TerrainType.SEA:
        return "blocked"
    occupant = game_state.ship_at(to_coord)
    if occupant is None:
        return "open"
    if occupant.owner == mover_owner:
        return "blocked"
    return "enemy"


def reachable_hexes(ship: Ship, game_state: GameState) -> dict[AxialCoord, int]:
    """Every hex `ship` could move to this turn, mapped to the number of
    steps (movement points) it costs to get there. Does not include the
    ship's own current hex. See `_classify_step` for per-step legality --
    an enemy-occupied hex is included as a terminal but not expanded
    further."""
    budget = ship.movement_remaining
    origin = ship.position
    origin_tile = game_state.board.get_tile(origin)
    leaving_port = origin_tile is not None and origin_tile.is_port

    reachable: dict[AxialCoord, int] = {}
    visited = {origin}
    frontier: list[tuple[AxialCoord, int]] = [(origin, 0)]
    while frontier:
        coord, cost = frontier.pop(0)
        if cost >= budget:
            continue
        for n in neighbors(coord):
            if n in visited:
                continue
            step = _classify_step(game_state, ship.owner, n, coord == origin and leaving_port)
            if step == "blocked":
                continue
            visited.add(n)
            new_cost = cost + 1
            reachable[n] = new_cost
            if step == "open":
                frontier.append((n, new_cost))
    return reachable


@dataclass
class MoveResult:
    ship: Ship
    origin: AxialCoord
    destination: AxialCoord
    cost: int


def move_ship(ship: Ship, destination: AxialCoord, game_state: GameState) -> MoveResult:
    """Move `ship` to `destination` by the shortest legal route, which must
    be in `reachable_hexes(ship, game_state)`. Deducts that route's cost
    from movement_remaining. Intended for simple single-destination movers
    (e.g. the AI); the player-facing UI instead draws an explicit route via
    `move_ship_along_path`, since the shortest route to a hex isn't always
    the one the player meant (e.g. routing around a threat). Raises if
    `destination` is enemy-occupied -- that's `begin_engagement`'s job."""
    reachable = reachable_hexes(ship, game_state)
    if destination not in reachable:
        raise ValueError(f"{destination} is not reachable by ship {ship.id} this turn")
    if game_state.ship_at(destination) is not None:
        raise ValueError(f"{destination} is enemy-occupied; use begin_engagement to attack it")
    cost = reachable[destination]
    origin = ship.position
    ship.position = destination
    ship.movement_remaining -= cost
    return MoveResult(ship=ship, origin=origin, destination=destination, cost=cost)


def validate_path(ship: Ship, path: list[AxialCoord], game_state: GameState) -> None:
    """Raise ValueError if `path` isn't a legal move for `ship` this turn.

    `path[0]` must be the ship's current position, each consecutive pair
    must be hex neighbors, each step must satisfy `_classify_step` (an
    enemy-occupied hex is legal only as the very last step), and the number
    of steps must fit within movement_remaining. Unlike `reachable_hexes`,
    this validates the exact route given -- an explicit, possibly
    non-shortest path is exactly the point (see `move_ship`).
    """
    if not path or path[0] != ship.position:
        raise ValueError("path must start at the ship's current position")
    steps = len(path) - 1
    if steps > ship.movement_remaining:
        raise ValueError(
            f"path costs {steps} but ship {ship.id} only has {ship.movement_remaining} movement left"
        )
    origin_tile = game_state.board.get_tile(path[0])
    leaving_port = origin_tile is not None and origin_tile.is_port

    for i in range(steps):
        a, b = path[i], path[i + 1]
        if b not in neighbors(a):
            raise ValueError(f"{b} is not adjacent to {a}")
        step = _classify_step(game_state, ship.owner, b, i == 0 and leaving_port)
        if step == "blocked":
            raise ValueError(f"{b} is not a legal step from {a}")
        if step == "enemy" and i != steps - 1:
            raise ValueError(f"{b} is enemy-occupied and can only be the final step of a move")


def move_ship_along_path(ship: Ship, path: list[AxialCoord], game_state: GameState) -> MoveResult:
    """Move `ship` along the exact `path` (see `validate_path`), deducting
    its length in movement points -- not necessarily the shortest possible
    cost to the final hex, since the whole point is an explicit route.
    Raises if the final hex is enemy-occupied; use `begin_engagement` for a
    combat-triggering move instead."""
    validate_path(ship, path, game_state)
    if game_state.ship_at(path[-1]) is not None:
        raise ValueError(f"{path[-1]} is enemy-occupied; use begin_engagement to attack it")
    cost = len(path) - 1
    origin = ship.position
    ship.position = path[-1]
    ship.movement_remaining -= cost
    return MoveResult(ship=ship, origin=origin, destination=path[-1], cost=cost)


def begin_engagement(ship: Ship, path: list[AxialCoord], game_state: GameState) -> Ship:
    """Validate `path`, whose final hex must be enemy-occupied, apply the
    approach portion of the move (everything before that hex), and consume
    the ship's remaining movement for the turn -- entering combat ends a
    ship's movement regardless of outcome. Returns the defending Ship;
    battle resolution itself is `tla.battle.run_battle`.

    If the attacker wins (defender sunk), the caller must move it onto
    `defender.position` afterward -- that hex is only vacated once the
    defender is actually removed from the game.
    """
    validate_path(ship, path, game_state)
    defender = game_state.ship_at(path[-1])
    if defender is None or defender.owner == ship.owner:
        raise ValueError(f"{path[-1]} is not an enemy-occupied hex")
    approach_path = path[:-1]
    if len(approach_path) > 1:
        move_ship_along_path(ship, approach_path, game_state)
    ship.movement_remaining = 0
    return defender


def toggle_submarine_state(ship: Ship, stats: ShipStats) -> None:
    """Flip `ship.surfaced`. A submarine gets two toggle opportunities per
    turn (one before it moves, one after); this consumes whichever hasn't
    been used yet, in order. `stats` must be this ship's own ShipStats.

    A pre-move toggle refreshes movement_remaining to match the new
    surfaced/submerged budget, since nothing has been spent yet. A
    post-move toggle doesn't -- movement for the turn is already done.
    """
    if ship.kind != ShipKind.SUBMARINE:
        raise ValueError(f"Only submarines can surface/submerge, not {ship.kind.value}")
    if not ship.toggled_pre_move:
        ship.toggled_pre_move = True
        ship.surfaced = not ship.surfaced
        ship.movement_remaining = ship.max_movement(stats)
    elif not ship.toggled_post_move:
        ship.toggled_post_move = True
        ship.surfaced = not ship.surfaced
    else:
        raise ValueError(f"Submarine {ship.id} already toggled surfaced/submerged twice this turn")
