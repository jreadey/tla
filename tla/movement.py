"""Movement legality and submarine surface/submerge toggling.

An enemy-occupied hex is reachable only as the very last hex of a move --
entering it triggers a battle (see tla.battle) instead of just occupying
the hex, so it can't be passed through on the way to somewhere else. A
friendly-occupied hex is the mirror image: never a legal place to *stop*
(at most one ship per hex at the end of a turn), but passable in transit
if the mover still has at least 2 movement points left when it gets there
-- 1 to enter, and at least 1 more left over so it's never left stranded
mid-hex on someone else's ship.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tla.game_state import GameState
from tla.hexgrid import AxialCoord, neighbors
from tla.production import handle_port_capture
from tla.ship import Ship, ShipKind, ShipStats
from tla.tile import PlayerId, TerrainType

StepKind = Literal["blocked", "open", "enemy", "passthrough"]


def _classify_step(
    game_state: GameState,
    mover_owner: PlayerId,
    to_coord: AxialCoord,
    leaving_origin_port: bool,
    remaining_before_step: int,
    treat_as_open: frozenset[AxialCoord] = frozenset(),
) -> StepKind:
    """"blocked": can't go there at all. "open": empty, can continue past.
    "enemy": occupied by the other side -- legal only as the final step of a
    move, since arriving there triggers a battle rather than occupying it.
    "passthrough": occupied by a friendly ship -- legal only as a
    non-final step, and only with `remaining_before_step` (the mover's
    movement budget before spending a point on this step) of at least 2, so
    it always has movement left to continue past rather than getting stuck
    stopped on a hex it can't legally occupy.

    `treat_as_open` overrides an otherwise-"enemy" hex to classify as
    "open" instead -- used by the UI to preview a drag as if a hidden
    submerged submarine weren't there, so its exact location can't be
    inferred from where the preview stops short (see
    tla.rendering.game_view). Has no effect on any other classification.
    """
    tile = game_state.board.get_tile(to_coord)
    if tile is None or not tile.occupiable:
        return "blocked"
    if leaving_origin_port and tile.terrain != TerrainType.SEA:
        return "blocked"
    occupant = game_state.ship_at(to_coord)
    if occupant is None:
        return "open"
    if occupant.owner == mover_owner:
        return "passthrough" if remaining_before_step >= 2 else "blocked"
    if to_coord in treat_as_open:
        return "open"
    return "enemy"


def reachable_hexes(
    ship: Ship, game_state: GameState, treat_as_open: frozenset[AxialCoord] = frozenset()
) -> dict[AxialCoord, int]:
    """Every hex `ship` could move to (i.e. legally stop at) this turn,
    mapped to the number of steps (movement points) it costs to get there.
    Does not include the ship's own current hex. See `_classify_step` for
    per-step legality -- an enemy-occupied hex is included as a terminal
    but not expanded further; a friendly-occupied hex is never included
    (can't stop there) but is expanded past if reached with enough budget
    left, so hexes beyond it can still be reachable stopping points.
    `treat_as_open` is passed straight through to `_classify_step`."""
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
        remaining_before_step = budget - cost
        for n in neighbors(coord):
            if n in visited:
                continue
            step = _classify_step(
                game_state,
                ship.owner,
                n,
                coord == origin and leaving_port,
                remaining_before_step,
                treat_as_open,
            )
            if step == "blocked":
                continue
            visited.add(n)
            new_cost = cost + 1
            if step == "passthrough":
                frontier.append((n, new_cost))
                continue
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
    `destination` is enemy-occupied -- that's `begin_engagement`'s job.

    May capture a port at `destination` (see
    `tla.production.handle_port_capture`), exactly like
    `move_ship_along_path` -- callers don't need to check for that
    separately."""
    reachable = reachable_hexes(ship, game_state)
    if destination not in reachable:
        raise ValueError(f"{destination} is not reachable by ship {ship.id} this turn")
    if game_state.ship_at(destination) is not None:
        raise ValueError(f"{destination} is enemy-occupied; use begin_engagement to attack it")
    cost = reachable[destination]
    origin = ship.position
    ship.position = destination
    ship.movement_remaining -= cost
    handle_port_capture(game_state, destination)
    return MoveResult(ship=ship, origin=origin, destination=destination, cost=cost)


def validate_path(
    ship: Ship,
    path: list[AxialCoord],
    game_state: GameState,
    *,
    allow_passthrough_final: bool = False,
    treat_as_open: frozenset[AxialCoord] = frozenset(),
) -> None:
    """Raise ValueError if `path` isn't a legal move for `ship` this turn.

    `path[0]` must be the ship's current position, each consecutive pair
    must be hex neighbors, each step must satisfy `_classify_step` (an
    enemy-occupied hex is legal only as the very last step; a
    friendly-occupied hex is legal only as a non-final step, and only with
    enough movement left to clear it), and the number of steps must fit
    within movement_remaining. Unlike `reachable_hexes`, this validates the
    exact route given -- an explicit, possibly non-shortest path is exactly
    the point (see `move_ship`).

    `allow_passthrough_final` relaxes just the friendly-occupied-hex rule so
    a path is accepted even if it currently *ends* on one -- meant for a
    UI validating an in-progress drag one hex at a time, where the newest
    hex is only provisionally the end and the player may well drag further;
    the actual move (`move_ship_along_path`/`begin_engagement`) always
    validates with the default `False` and still rejects stopping there.

    `treat_as_open` is passed straight through to `_classify_step`, for the
    same drag-preview purpose as in `reachable_hexes`.
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
        remaining_before_step = ship.movement_remaining - i
        step = _classify_step(
            game_state, ship.owner, b, i == 0 and leaving_port, remaining_before_step, treat_as_open
        )
        if step == "blocked":
            raise ValueError(f"{b} is not a legal step from {a}")
        if step == "enemy" and i != steps - 1:
            raise ValueError(f"{b} is enemy-occupied and can only be the final step of a move")
        is_final = i == steps - 1
        if step == "passthrough" and is_final and not allow_passthrough_final:
            raise ValueError(f"{b} is occupied by a friendly ship and can't be the final step of a move")


def move_ship_along_path(ship: Ship, path: list[AxialCoord], game_state: GameState) -> MoveResult:
    """Move `ship` along the exact `path` (see `validate_path`), deducting
    its length in movement points -- not necessarily the shortest possible
    cost to the final hex, since the whole point is an explicit route.
    Raises if the final hex is enemy-occupied; use `begin_engagement` for a
    combat-triggering move instead.

    If the destination is a port, this may capture it (see
    `tla.production.handle_port_capture`) and, if that completes total port
    control, end the game -- callers don't need to check for that
    separately. This applies even to an approach path fed in from
    `begin_engagement`: stopping at an enemy port on the way to attacking
    someone adjacent still counts as occupying it.
    """
    validate_path(ship, path, game_state)
    occupant = game_state.ship_at(path[-1])
    if occupant is not None and occupant.owner != ship.owner:
        raise ValueError(f"{path[-1]} is enemy-occupied; use begin_engagement to attack it")
    cost = len(path) - 1
    origin = ship.position
    ship.position = path[-1]
    ship.movement_remaining -= cost
    handle_port_capture(game_state, path[-1])
    return MoveResult(ship=ship, origin=origin, destination=path[-1], cost=cost)


def begin_engagement(ship: Ship, path: list[AxialCoord], game_state: GameState) -> Ship:
    """Validate `path`, whose final hex must be enemy-occupied, apply the
    approach portion of the move, and charge for the final attack stretch
    at the same per-hex rate as a normal move -- engaging never costs more
    than moving the same distance would. Returns the defending Ship; battle
    resolution itself is `tla.battle.run_battle`.

    The attacker's approach stops at the *last hex before the target that
    it can actually occupy* -- normally that's simply the hex right next
    to the target, but if that hex is a friendly ship merely passed through
    en route (see the "passthrough" rule in `_classify_step`), the attacker
    can't literally rest there alongside it, so it stops one hex earlier
    instead and the final attack is charged for the whole remaining
    stretch (crossing the passthrough hex and then the target), not the
    usual flat 1 point. A retreat afterward returns to no further than
    wherever the attacker actually stopped, for free; if the ship has
    movement left over -- whether it retreats or wins and continues -- it
    can keep moving this turn.

    If the attacker wins (defender sunk), the caller must move it onto
    `defender.position` afterward -- that hex is only vacated once the
    defender is actually removed from the game.
    """
    validate_path(ship, path, game_state)
    defender = game_state.ship_at(path[-1])
    if defender is None or defender.owner == ship.owner:
        raise ValueError(f"{path[-1]} is not an enemy-occupied hex")

    # The last hex before the target that's actually free to occupy --
    # searching backward from the target so the attacker gets as close as
    # it possibly can. path[0] (the ship's own current hex) always
    # qualifies, so this is guaranteed to find a stop.
    stop_index = 0
    for i in range(len(path) - 2, -1, -1):
        if path[i] == ship.position or game_state.ship_at(path[i]) is None:
            stop_index = i
            break

    approach_path = path[: stop_index + 1]
    if len(approach_path) > 1:
        move_ship_along_path(ship, approach_path, game_state)
    attack_cost = len(path) - 1 - stop_index
    ship.movement_remaining -= attack_cost
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
