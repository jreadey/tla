"""Stateless scoring/search helpers shared by tla.ai.policy -- pure
functions of a GameState, reusable by any future, smarter policy."""

from __future__ import annotations

import math

from tla import movement
from tla.ai.task_force import sea_distance_field, sea_route_distance
from tla.battle import carrier_bonus_for
from tla.game_state import GameState
from tla.hexgrid import AxialCoord, distance
from tla.ship import Ship, ShipKind
from tla.tile import PlayerId


def effective_damage(attacker: Ship, defender: Ship, battle_hex: AxialCoord, game_state: GameState) -> int:
    """Damage `attacker` deals `defender` in one round at `battle_hex`,
    `attacker`'s own carrier bonus included -- mirrors `battle.
    _base_damage` + the carrier bonus computed by `resolve_round`, so
    this estimate matches actual combat exactly rather than risking a
    drifting duplicate.

    `battle_hex` is explicit and must be the *real* (always stationary)
    defender's position, even when this is called with the roles
    swapped to get the return-fire direction (see `matchup_score` below)
    -- a real bug, found via replay review, used to default this to
    `defender.position` implicitly, which is correct for the attack
    direction but silently used the *mover's own current position*
    instead of the actual battle location for the return-fire direction,
    systematically undercounting a defender's nearby-carrier bonus
    whenever the attacker approached from outside that radius (`battle.
    resolve_round` itself always uses the one real, fixed `battle_hex =
    defender.position` for both directions -- this now matches that
    exactly). Public -- `tla.ai.tactics` needs this same exact math, same
    reasoning as `tla.ai.task_force.enemy_reachable_next_turn` being
    public: more than one module needs it, never a duplicated copy."""
    stats = game_state.config.ship_stats.stats[attacker.kind]
    base = stats.asw if (defender.kind == ShipKind.SUBMARINE and not defender.surfaced) else stats.damage
    bonus = carrier_bonus_for(game_state, attacker, defender, battle_hex, game_state.config.combat)
    return base + bonus


def matchup_score(attacker: Ship, defender: Ship, game_state: GameState) -> float:
    """How favorable an `attacker`-vs-`defender` engagement looks, using
    each side's current HP and a simple "who kills whom first" race:
    `rounds_to_kill_me - rounds_to_kill_them`. Zero or positive means
    favorable (I win the race or it's a tie -- ties favor the attacker,
    since `battle.resolve_round`'s simultaneous damage means initiative
    never matters). +inf if the defender can't hurt me at all (e.g. an
    asw-0 attacker's target can't out-asw me back -- degenerate but
    handled cleanly); -inf if I can't hurt the defender at all (e.g. my own
    asw is 0 against a submerged submarine); 0.0 if neither side can hurt
    the other (a true stalemate, which still counts as favorable/no-loss
    for the attacker)."""
    battle_hex = defender.position  # the one real, fixed battle location -- see effective_damage's own docstring
    my_dpr = effective_damage(attacker, defender, battle_hex, game_state)
    their_dpr = effective_damage(defender, attacker, battle_hex, game_state)
    if my_dpr <= 0 and their_dpr <= 0:
        return 0.0
    if my_dpr <= 0:
        return -math.inf
    if their_dpr <= 0:
        return math.inf
    rounds_to_kill_them = math.ceil(defender.current_hp / my_dpr)
    rounds_to_kill_me = math.ceil(attacker.current_hp / their_dpr)
    return rounds_to_kill_me - rounds_to_kill_them


def worth_a_tie(ship: Ship, target: Ship, game_state: GameState) -> bool:
    """Whether trading `ship` for `target` at even odds (a `matchup_score`
    tie -- both die) is actually a good deal: `target`'s replacement cost
    must *exceed* `ship`'s own, not just match it -- an even-cost mirror
    matchup (e.g. two identical destroyers) is a neutral trade, still
    declined by default the same as before this existed, requiring
    `tie_tolerant`/`Strategy.AGGRESSIVE` same as any other non-favorable
    tie. Uses `ShipStats.cost` -- the only "replacement value" concept
    already in the game (`tla.production`'s own build economy) -- rather
    than inventing a second, competing notion of ship worth. Only
    meaningful for a genuine tie (`matchup_score(ship, target, game_
    state) == 0`); not used to accept a losing trade."""
    stats = game_state.config.ship_stats.stats
    return stats[target.kind].cost > stats[ship.kind].cost


def nearest_enemy(ship: Ship, enemy_ships: dict[int, Ship], game_state: GameState) -> Ship | None:
    """The closest ship in `enemy_ships` to `ship`, breaking ties by ship
    id for determinism. None if `enemy_ships` is empty."""
    if not enemy_ships:
        return None
    return min(enemy_ships.values(), key=lambda s: (distance(ship.position, s.position), s.id))


def nearest_uncontrolled_port(
    game_state: GameState, player: PlayerId, from_coord: AxialCoord
) -> AxialCoord | None:
    """The closest-by-actual-sea-route (see
    `tla.ai.task_force.sea_distance_field` -- not straight-line hex
    distance, which can rank a landlocked-looking-close port ahead of one
    that's actually a much shorter sail away) port to `from_coord` that
    `player` doesn't currently control -- i.e. worth expanding toward.
    None if `player` already controls every port (which is already a won
    game)."""
    controlled = set(game_state.board.controlled_ports_for(player))
    candidates = [
        coord for coord, tile in game_state.board.tiles.items() if tile.is_port and coord not in controlled
    ]
    if not candidates:
        return None
    field = sea_distance_field(game_state, from_coord)
    return min(candidates, key=lambda c: (sea_route_distance(field, c), c))


def reachable_attack_candidates(
    ship: Ship, game_state: GameState, enemy_ships: dict[int, Ship]
) -> list[tuple[AxialCoord, int]]:
    """Every enemy-occupied hex `ship` could reach and attack this turn,
    paired with its movement cost -- `(coord, cost)` for each. Empty if
    `ship` can't reach any enemy this turn. Factored out of
    `best_reachable_attack` (which still just wants the single best one)
    so `tla.ai.tactics.secure_kills_pass` -- which needs to consider
    *every* reachable target, not just the best -- doesn't duplicate this
    same reachable-hexes-intersected-with-enemy-positions computation."""
    if not enemy_ships:
        return []
    reachable = movement.reachable_hexes(ship, game_state)
    enemy_by_position = {s.position: s for s in enemy_ships.values()}
    return [(coord, cost) for coord, cost in reachable.items() if coord in enemy_by_position]


def best_reachable_attack(
    ship: Ship, game_state: GameState, enemy_ships: dict[int, Ship]
) -> AxialCoord | None:
    """Among the enemy-occupied hexes `ship` could reach and attack this
    turn, the one with the best `matchup_score` against it (ties broken by
    lower movement cost, then hex, for determinism). None if `ship` can't
    reach any enemy this turn."""
    candidates = reachable_attack_candidates(ship, game_state, enemy_ships)
    if not candidates:
        return None
    enemy_by_position = {s.position: s for s in enemy_ships.values()}

    def key(item: tuple[AxialCoord, int]) -> tuple[float, int, AxialCoord]:
        coord, cost = item
        return (-matchup_score(ship, enemy_by_position[coord], game_state), cost, coord)

    return min(candidates, key=key)[0]
