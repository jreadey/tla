"""Task forces: an AI-internal grouping of ships pursuing one shared,
longer-range strategic goal (capture a port, blockade a chokepoint),
distinct from the ship-by-ship tactical decisions in tla.ai.scoring/policy.

Purely a planning tool inside tla.ai.policy.NaivePolicy -- never part of
GameState, never persisted, never visible to the UI or a human player. A
TaskForce is mutated in place over its lifetime (forms, loses members,
completes or abandons its goal, dissolves), the same way a Ship is -- not
an immutable value object like Config.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Callable

from tla import movement
from tla.ai.enemy_model import EnemyModel
from tla.config import AiConfig
from tla.game_state import GameState
from tla.hexgrid import AxialCoord, distance, neighbors, round_to_axial
from tla.ship import Ship, ShipKind
from tla.tile import PLAYER_A, PLAYER_B, PlayerId, TerrainType

if TYPE_CHECKING:
    # Only for the type hint in apply_defensive_port_priority -- never
    # imported at runtime (that function does its own local import instead)
    # since tla.ai.global_strategy imports DANGEROUS_TO_CARRIER_KINDS/
    # outmatched from this module, and a module-level import here would be
    # a cycle.
    from tla.ai.global_strategy import Posture


class GoalKind(Enum):
    CAPTURE_PORT = "capture_port"
    BLOCKADE = "blockade"
    # A DEFENSIVE-posture response to a believed (not necessarily visible)
    # threat near one of the player's own controlled ports -- see
    # apply_defensive_port_priority. Owned and released entirely by that
    # function; update_task_force_goals exempts it from its own stall-clear
    # logic the same way it already exempts RETREAT.
    DEFEND_PORT = "defend_port"
    # Never persisted via assign_goal -- purely an ephemeral value built
    # fresh each turn by tla.ai.policy._choose_destination while a force's
    # retreating is True, shadowing its real goal for movement purposes
    # only. See update_task_force_stance.
    RETREAT = "retreat"


# The kinds a carrier actually needs defending against -- a destroyer or
# patrol boat alone doesn't trigger a carrier's own threat-avoidance
# (tla.ai.policy._choose_carrier_destination) or compute_carrier_defense_
# directives below, matching a human player's own reported tactic of only
# scanning for the "big hitters."
DANGEROUS_TO_CARRIER_KINDS = frozenset({ShipKind.BATTLESHIP, ShipKind.CRUISER, ShipKind.SUBMARINE})


@dataclass
class TaskForceGoal:
    kind: GoalKind
    # a port hex for CAPTURE_PORT (uncontrolled) or DEFEND_PORT (controlled);
    # a chokepoint hex for BLOCKADE
    target: AxialCoord


@dataclass
class TaskForce:
    id: int
    owner: PlayerId
    member_ids: set[int] = field(default_factory=set)
    goal: TaskForceGoal | None = None
    # How many turns in a row the force's best (nearest-member) distance to
    # its own goal.target has failed to improve -- see
    # tla.ai.policy.record_task_force_progress. Reset whenever the goal
    # changes. Used to detect a stalled goal (see tla.ai.policy.
    # update_task_force_goals) rather than pursuing it forever.
    turns_since_progress: int = 0
    best_progress_distance: int | None = None
    # Whether the force is currently in a forced tactical retreat (see
    # update_task_force_stance/is_outnumbered/is_attritting). Does NOT
    # change `goal`, which stays exactly what it was; only shadows it for
    # movement purposes (tla.ai.policy._choose_destination) until the force
    # is strong enough to safely resume -- see retreat_threat_power.
    retreating: bool = False
    # Turns spent in the *current* retreat so far, counting up -- capped by
    # AiConfig.task_force_max_retreat_turns as a safety valve in case the
    # opponent reinforces just as fast and the force would otherwise never
    # catch up. Reset to 0 whenever a retreat starts or ends.
    retreat_turns: int = 0
    # (total hp, total damage) of whatever the force retreated from,
    # snapshotted the instant retreat triggers (see _threat_snapshot) --
    # the bar a retreating force's own current strength must clearly beat
    # (see is_outnumbered's task_force_outnumbered_margin) before it's safe
    # to resume. None whenever the force isn't retreating.
    retreat_threat_power: tuple[int, int] | None = None
    # Retreat destination for the current retreat, snapshotted the instant
    # it triggers (see _pullback_waypoint) -- only set when
    # AiConfig.retreat_pullback_hexes caps a retreat short of the rally
    # port itself. None means retreat all the way to the rally port
    # (tla.ai.task_force.rally_point), same as when not retreating at all.
    retreat_waypoint: AxialCoord | None = None
    # Consecutive turns (this player's own turns, tracked by
    # repair_task_forces) this force has lost at least one member -- reset
    # to 0 the instant a turn passes with no losses, and whenever the goal
    # changes (see assign_goal). See is_attritting: paired with
    # turns_since_progress to catch a losing war of attrition even when no
    # single instant looks clearly outnumbered.
    consecutive_loss_turns: int = 0


def repair_task_forces(game_state: GameState, player: PlayerId, forces: list[TaskForce]) -> None:
    """Drop sunk ships from membership and remove any force left with no
    members at all -- survivors fall into the unassigned pool and get
    picked up again by the next formation pass (see
    tla.ai.policy.form_task_forces). A newly produced ship is never in any
    member_ids set to begin with, so it's automatically unassigned too,
    with no special casing needed. Call once per turn, before formation.

    Also maintains `consecutive_loss_turns`: incremented whenever this
    call actually drops one or more now-sunk members, reset to 0 when it
    doesn't -- this is the only place membership shrinks from casualties,
    so it's the natural place to notice a loss happened at all."""
    owned_ids = {s.id for s in game_state.ships_for(player)}
    for force in forces:
        before = len(force.member_ids)
        force.member_ids &= owned_ids
        if len(force.member_ids) < before:
            force.consecutive_loss_turns += 1
        else:
            force.consecutive_loss_turns = 0
    forces[:] = [f for f in forces if f.member_ids]


def _gather(
    center: AxialCoord,
    exclude_ids: frozenset[int],
    pool: list[Ship],
    gather_radius: int,
    capacity: int,
    *,
    exclude_carriers: bool,
    existing_kinds: Counter[ShipKind] | None = None,
) -> set[int]:
    """Up to `capacity` `pool` ships (excluding `exclude_ids`) within
    `gather_radius` of `center`. Returns only the newly-gathered ids --
    callers union them with whatever's already there (an anchor's own id
    for a fresh force, see `form_task_forces`; or an existing force's
    member_ids for recruitment, see `recruit_into_open_forces`).

    Picks greedily to diversify the force's ship-kind mix rather than
    purely nearest-first: at each step, among candidates already within
    `gather_radius` (diversity is never chased further afield than that),
    prefers whichever kind is currently least represented -- seeded by
    `existing_kinds` (the force's current composition) and updated as
    ships are picked -- breaking ties by distance then id. Without this,
    a force's mix is purely an accident of which ships happened to be
    produced/positioned nearby -- e.g. a cluster of same-kind stragglers
    could fill an entire force with duplicates even when a comparably
    close, different-kind ship was also in reach."""
    kind_counts: Counter[ShipKind] = Counter(existing_kinds) if existing_kinds else Counter()
    remaining = [
        s
        for s in pool
        if s.id not in exclude_ids
        and (not exclude_carriers or s.kind != ShipKind.CARRIER)
        and distance(s.position, center) <= gather_radius
    ]
    picked: list[Ship] = []
    while remaining and len(picked) < capacity:
        remaining.sort(key=lambda s: (kind_counts[s.kind], distance(s.position, center), s.id))
        chosen = remaining.pop(0)
        picked.append(chosen)
        kind_counts[chosen.kind] += 1
    return {s.id for s in picked}


def form_task_forces(
    game_state: GameState,
    player: PlayerId,
    forces: list[TaskForce],
    ai_config: AiConfig,
    alloc_id: Callable[[], int],
) -> None:
    """Opportunistically group `player`'s currently-unassigned ships (not
    already a member of any force in `forces`) into new task forces.
    Additive, not a rewrite: whatever's left unassigned after this simply
    isn't touched, and keeps using the existing per-ship
    `_choose_generic_destination`/`_choose_carrier_destination` logic --
    it's reconsidered for formation again next turn as positions shift.

    One force per carrier (a force's carrier is its anchor -- see
    `tla.ai.policy._choose_carrier_destination`'s `goal` extension), formed
    first and in ascending ship-id order for determinism, pulling in the
    nearest other unassigned non-carrier ships within
    `AiConfig.task_force_gather_radius`, up to `AiConfig.task_force_max_size`
    members total. If `AiConfig.task_force_allow_non_carrier_forces` and
    ships still remain unassigned afterward, one more force is formed the
    same way around the lowest-id remaining ship as an arbitrary anchor --
    but only kept if it reaches `AiConfig.task_force_min_size` members;
    otherwise those ships stay unassigned rather than forming an
    under-strength force with no realistic goal-pursuit capability.

    New forces are appended to `forces` with `goal=None` -- assigning one
    is `assign_goal`'s job, called separately so a fresh force gets a goal
    the same turn it forms rather than sitting idle for one turn."""
    assigned = {i for f in forces for i in f.member_ids}
    pool = [s for s in game_state.ships_for(player) if s.id not in assigned]

    for carrier in [s for s in pool if s.kind == ShipKind.CARRIER]:
        picked = _gather(
            carrier.position,
            frozenset({carrier.id}),
            pool,
            ai_config.task_force_gather_radius,
            ai_config.task_force_max_size - 1,
            exclude_carriers=True,
            existing_kinds=Counter({ShipKind.CARRIER: 1}),
        )
        member_ids = {carrier.id, *picked}
        forces.append(TaskForce(id=alloc_id(), owner=player, member_ids=member_ids))
        pool = [s for s in pool if s.id not in member_ids]

    if ai_config.task_force_allow_non_carrier_forces and pool:
        anchor = min(pool, key=lambda s: s.id)
        picked = _gather(
            anchor.position,
            frozenset({anchor.id}),
            pool,
            ai_config.task_force_gather_radius,
            ai_config.task_force_max_size - 1,
            existing_kinds=Counter({anchor.kind: 1}),
            exclude_carriers=False,
        )
        member_ids = {anchor.id, *picked}
        if len(member_ids) >= ai_config.task_force_min_size:
            forces.append(TaskForce(id=alloc_id(), owner=player, member_ids=member_ids))


def _centroid(coords: list[AxialCoord]) -> AxialCoord:
    mean_q = sum(c.q for c in coords) / len(coords)
    mean_r = sum(c.r for c in coords) / len(coords)
    return round_to_axial(mean_q, mean_r)


def _nearest_in(coord: AxialCoord, sea: set[AxialCoord]) -> AxialCoord:
    """`coord` itself if it's navigable sea, else the closest hex that is
    -- a raw port centroid can easily land on land or between islands."""
    if coord in sea:
        return coord
    return min(sea, key=lambda c: (distance(coord, c), c))


def _shortest_sea_path(start: AxialCoord, goal: AxialCoord, sea: set[AxialCoord]) -> list[AxialCoord] | None:
    """Plain unweighted BFS shortest path from `start` to `goal`, moving
    only through hexes in `sea` -- no movement budget, ship, or turn
    involved, just graph connectivity. None if the two aren't connected
    (e.g. a disconnected map)."""
    if start == goal:
        return [start]
    parents: dict[AxialCoord, AxialCoord] = {}
    visited = {start}
    frontier = [start]
    while frontier:
        next_frontier: list[AxialCoord] = []
        for coord in frontier:
            for n in neighbors(coord):
                if n not in sea or n in visited:
                    continue
                visited.add(n)
                parents[n] = coord
                if n == goal:
                    path = [goal]
                    while path[-1] != start:
                        path.append(parents[path[-1]])
                    path.reverse()
                    return path
                next_frontier.append(n)
        frontier = next_frontier
    return None


_UNREACHABLE_SEA_DISTANCE = 10_000


def sea_distance_field(game_state: GameState, source: AxialCoord) -> dict[AxialCoord, int]:
    """Real sea-route distance (hex steps), from `source` to every sea hex
    reachable from it by water -- unlike straight-line `hexgrid.distance`,
    this correctly reflects a landmass or narrow strait that leaves two
    hexes close as the crow flies but far apart (or entirely unreachable
    from each other) by the only actual sea route between them. `source`
    itself always gets distance 0, whether it's a sea hex (e.g. a
    `BLOCKADE` chokepoint, or a force's own centroid) or land (e.g. a
    `CAPTURE_PORT` target) -- BFS proper then runs outward from `source`'s
    sea-adjacent neighbors, each starting at distance 1. A sea hex with no
    path back to `source` at all (a genuinely disconnected component) is
    simply absent from the result -- see `sea_route_distance` for querying
    a hex, land included, against a field built this way.

    Distance is symmetric, so the same field answers "how far is X from
    `source`" regardless of which endpoint a caller conceptually treats as
    the destination -- used both to rank a ship's candidate destinations
    against a fixed goal (`tla.ai.policy._step_toward`) and to rank
    candidate goals against a fixed current position (`assign_goal`,
    `tla.ai.scoring.nearest_uncontrolled_port`)."""
    sea = {c for c, t in game_state.board.tiles.items() if t.terrain == TerrainType.SEA}
    dist: dict[AxialCoord, int] = {source: 0}
    frontier = [n for n in neighbors(source) if n in sea]
    for n in frontier:
        dist.setdefault(n, 1)
    while frontier:
        next_frontier: list[AxialCoord] = []
        for coord in frontier:
            for n in neighbors(coord):
                if n in sea and n not in dist:
                    dist[n] = dist[coord] + 1
                    next_frontier.append(n)
        frontier = next_frontier
    return dist


def sea_route_distance(field: dict[AxialCoord, int], coord: AxialCoord) -> int:
    """`field`'s (see `sea_distance_field`) real sea-route distance to any
    hex, land included -- a land hex other than the field's own `source`
    (e.g. a port being evaluated as a candidate goal) is never itself a
    member of the underlying sea graph, so its distance is 1 + the best
    among its own sea-adjacent neighbors already in `field`.
    `_UNREACHABLE_SEA_DISTANCE` if `coord` has no sea-adjacent neighbor in
    `field` at all -- genuinely disconnected from the field's source,
    ranked as arbitrarily far rather than raising."""
    if coord in field:
        return field[coord]
    neighbor_distances = [field[n] for n in neighbors(coord) if n in field]
    return min(neighbor_distances) + 1 if neighbor_distances else _UNREACHABLE_SEA_DISTANCE


def find_chokepoint(game_state: GameState) -> AxialCoord | None:
    """A hex on the shortest sea route between the two players' home
    territories (mean position of `board.ports_for(player)` -- permanent
    and stable all game, so this doesn't drift as ports change hands)
    that's the *narrowest* point along that route: the one with the most
    neighboring hexes that aren't navigable sea (land, or off the map
    entirely). 0 such neighbors means open ocean; higher means a tighter
    passage between landmasses -- exactly where a `BLOCKADE` task force
    should stand to intercept traffic between the two sides. Ties broken
    by proximity to the path's own midpoint, then by coordinate, for
    determinism.

    None if there's no sea route between the two territories at all (a
    fully disconnected map) or either side has no ports -- callers must
    treat that as "no blockade goal available."

    Deliberately paths over *every* sea tile, not just
    `mapgen.largest_sea_component` -- snapping both territories onto that
    single, one-and-only "biggest" component would make them trivially
    "connected" via it even when the two territories' own local waters are
    genuinely disconnected from each other, defeating the point of
    checking connectivity at all.
    """
    board = game_state.board
    sea = {c for c, t in board.tiles.items() if t.terrain == TerrainType.SEA}
    a_ports = board.ports_for(PLAYER_A)
    b_ports = board.ports_for(PLAYER_B)
    if not sea or not a_ports or not b_ports:
        return None

    start = _nearest_in(_centroid(a_ports), sea)
    goal = _nearest_in(_centroid(b_ports), sea)
    path = _shortest_sea_path(start, goal, sea)
    if not path:
        return None

    def narrowness(coord: AxialCoord) -> int:
        return sum(1 for n in neighbors(coord) if n not in sea)

    mid = len(path) // 2
    best_index = min(range(len(path)), key=lambda i: (-narrowness(path[i]), abs(i - mid), path[i]))
    return path[best_index]


def assign_goal(
    force: TaskForce,
    game_state: GameState,
    all_forces: list[TaskForce],
    exclude: frozenset[AxialCoord] = frozenset(),
) -> None:
    """Give `force` a fresh goal -- call whenever `force.goal is None`
    (freshly formed, just completed, or just gave up on a stalled one).
    Goals are sticky once assigned (see `update_task_force_goals`), so this
    is the only place a force's target ever changes, and it always resets
    progress tracking for the new goal.

    Prefers the nearest-by-actual-sea-route (see `sea_distance_field`, not
    straight-line hex distance -- a landmass between the force and a
    candidate port can make it far closer as the crow flies than it
    actually is to sail to) uncontrolled port not already claimed by
    another of this player's forces (`all_forces`) or in `exclude` (used
    for a one-shot "don't immediately re-pick what I just gave up on"
    exclusion after a stall -- see `update_task_force_goals`; not a
    permanent blacklist, since the board can change and make a skipped
    port worth retrying later). Falls back to `find_chokepoint` (a
    `BLOCKADE` goal) if every port is already controlled or claimed.
    Leaves `force.goal` as `None` if neither is available (e.g. a
    disconnected map with no chokepoint either) -- the force simply has
    nothing to do this turn and is tried again next turn."""
    members = [game_state.ships[i] for i in force.member_ids if i in game_state.ships]
    if not members:
        return
    centroid = _centroid([m.position for m in members])

    claimed = {
        f.goal.target
        for f in all_forces
        if f is not force and f.goal is not None and f.goal.kind == GoalKind.CAPTURE_PORT
    }
    controlled = set(game_state.board.controlled_ports_for(force.owner))
    candidates = [
        coord
        for coord, tile in game_state.board.tiles.items()
        if tile.is_port and coord not in controlled and coord not in claimed and coord not in exclude
    ]

    if candidates:
        field = sea_distance_field(game_state, centroid)
        target = min(candidates, key=lambda c: (sea_route_distance(field, c), c))
        force.goal = TaskForceGoal(kind=GoalKind.CAPTURE_PORT, target=target)
    else:
        chokepoint = find_chokepoint(game_state)
        force.goal = TaskForceGoal(kind=GoalKind.BLOCKADE, target=chokepoint) if chokepoint is not None else None

    force.turns_since_progress = 0
    force.best_progress_distance = None
    force.retreating = False
    force.retreat_turns = 0
    force.retreat_threat_power = None
    force.retreat_waypoint = None
    force.consecutive_loss_turns = 0


def force_recapture_goal(force: TaskForce, game_state: GameState, lost_ports: set[AxialCoord]) -> None:
    """Override `force`'s real `goal` -- unconditionally, regardless of what
    it was pursuing or how much progress it had made -- to retake the
    nearest (by real sea-route distance, same as `assign_goal`) of
    `lost_ports`: ports this player controlled as of last turn and does not
    anymore. Called once per force, for every force, the instant a loss is
    detected (see `tla.ai.policy.plan_movement`) -- losing a port is
    existential enough to redirect everything at once, not something to
    wait on a stall or completion to notice. Deliberately does NOT touch
    `retreating`/`retreat_threat_power`: an already-fleeing, currently weak
    force isn't forced to charge back in immediately -- only its *real*
    goal changes, so it resumes toward the new recapture target exactly
    when it would otherwise have resumed its old one (see
    `update_task_force_stance`), and that same outnumbered check still
    applies here going forward same as any other goal -- this makes
    recapture the thing the force commits to, it doesn't bypass the AI's
    existing sense not to charge into a clearly-losing fight.

    No-op if `force` has no living members (nothing to redirect) or
    `lost_ports` is empty."""
    if not lost_ports:
        return
    members = [game_state.ships[i] for i in force.member_ids if i in game_state.ships]
    if not members:
        return
    centroid = _centroid([m.position for m in members])
    field = sea_distance_field(game_state, centroid)
    target = min(lost_ports, key=lambda c: (sea_route_distance(field, c), c))

    force.goal = TaskForceGoal(kind=GoalKind.CAPTURE_PORT, target=target)
    force.turns_since_progress = 0
    force.best_progress_distance = None


def apply_defensive_port_priority(
    forces: list[TaskForce],
    game_state: GameState,
    player: PlayerId,
    enemy_model: EnemyModel,
    posture: "Posture",
    ai_config: AiConfig,
) -> None:
    """When `posture` is DEFENSIVE and the player's single most-threatened
    controlled port (`tla.ai.global_strategy.rank_port_threats`, using
    `ai_config.port_defense_trigger_radius` -- the same "how far out is
    this port's business" radius the visible-enemy port-defense directive
    already uses, purely to pick *which* port is worst) is actually
    outmatched by that threat -- `EnemyModel.expected_strength_near`
    (believed dangerous-kind strength within that same radius) against
    `group_power` of `player`'s own ships already within `ai_config.
    port_defense_response_radius` of it, via the same `outmatched`/
    `port_defense_margin` race `compute_port_defense_directives` uses for
    the identical question against *visible* threats -- redirect a force
    to `DEFEND_PORT` that port. A port that already has enough of its own
    ships nearby to handle the threat is left alone, regardless of how
    much raw threat mass is out there.

    Prefers a free force (`goal is None`, `BLOCKADE`, or already
    `DEFEND_PORT` elsewhere) -- no disruption cost. Only if none is
    available does it recall an active `CAPTURE_PORT` force, and only one
    within `ai_config.defend_port_recall_max_distance` sea hexes of the
    port -- abandoning an ongoing offense is never worth it for a force
    that's too far away to actually get back in time. Never a retreating
    force either way.

    Belief-driven and persistent, unlike `compute_port_defense_directives`
    (visible-enemy-only, immediate one-turn ship moves that bypass the
    goal system entirely) -- a different, complementary layer, not a
    replacement; the two can never double-claim the same ship in the same
    turn regardless, since a directive-claimed ship is excluded from the
    goal-driven movement loop before it ever reads a force's goal.

    Stable once assigned: a force already defending the current top-ranked
    port is left alone rather than re-picked every turn. Releases (goal ->
    `None`) any force's `DEFEND_PORT` goal once posture leaves DEFENSIVE or
    the port is no longer outmatched -- `update_task_force_goals`'s own
    goalless-assignment pass (called right after this, same turn) gives a
    released force a fresh `CAPTURE_PORT`/`BLOCKADE` goal immediately, no
    wasted idle turn. Only ever targets the single top-ranked port --
    deliberately narrow scope for a first pass, matching a single dedicated
    defender."""
    # Local import: tla.ai.global_strategy imports DANGEROUS_TO_CARRIER_KINDS/
    # outmatched from this module, so importing it back at module level here
    # would be a cycle.
    from tla.ai.global_strategy import Posture, rank_port_threats

    ranked = rank_port_threats(game_state, player, enemy_model, ai_config.port_defense_trigger_radius)
    target_port = None
    if posture == Posture.DEFENSIVE and ranked:
        top_port, _ = ranked[0]
        top_field = sea_distance_field(game_state, top_port)
        threat_power = enemy_model.expected_strength_near(
            top_port, ai_config.port_defense_trigger_radius, kinds=DANGEROUS_TO_CARRIER_KINDS
        )
        defenders = [
            s
            for s in game_state.ships_for(player)
            if sea_route_distance(top_field, s.position) <= ai_config.port_defense_response_radius
        ]
        defender_power = group_power(defenders, game_state)
        if outmatched(defender_power, threat_power, ai_config.port_defense_margin):
            target_port = top_port

    def _defends(f: TaskForce) -> bool:
        return f.goal is not None and f.goal.kind == GoalKind.DEFEND_PORT

    for force in forces:
        if _defends(force) and force.goal.target != target_port:
            force.goal = None
            force.turns_since_progress = 0
            force.best_progress_distance = None

    if target_port is None:
        return
    if any(_defends(f) and f.goal.target == target_port for f in forces):
        return

    field = sea_distance_field(game_state, target_port)

    def _nearest_member_distance(force: TaskForce) -> int:
        members = [game_state.ships[i] for i in force.member_ids if i in game_state.ships]
        return min(sea_route_distance(field, m.position) for m in members)

    # No force can still have a DEFEND_PORT goal here: the release pass
    # above clears any that don't already target target_port, and the
    # early return just above catches the one that does.
    free_eligible = [
        f
        for f in forces
        if f.member_ids and not f.retreating and (f.goal is None or f.goal.kind == GoalKind.BLOCKADE)
    ]
    chosen = min(free_eligible, key=lambda f: (_nearest_member_distance(f), f.id)) if free_eligible else None
    if chosen is None:
        recallable = [
            f
            for f in forces
            if f.member_ids
            and not f.retreating
            and f.goal is not None
            and f.goal.kind == GoalKind.CAPTURE_PORT
            and _nearest_member_distance(f) <= ai_config.defend_port_recall_max_distance
        ]
        chosen = min(recallable, key=lambda f: (_nearest_member_distance(f), f.id)) if recallable else None
    if chosen is None:
        return

    chosen.goal = TaskForceGoal(kind=GoalKind.DEFEND_PORT, target=target_port)
    chosen.turns_since_progress = 0
    chosen.best_progress_distance = None


def update_task_force_goals(game_state: GameState, player: PlayerId, forces: list[TaskForce]) -> None:
    """Per-turn goal maintenance, called once per player right after
    formation: dissolves a force that's dropped below
    `AiConfig.task_force_min_size` members to casualties (`member_ids`
    cleared -- next turn's `repair_task_forces` drops the now-empty force
    and its survivor(s), if any, fall into the unassigned pool and get
    reconsidered by the next formation pass); clears a force's goal if its
    `CAPTURE_PORT` target is now controlled (done -- no idle turn,
    reassigned immediately below), or if it's stalled
    (`turns_since_progress >= AiConfig.task_force_stall_turns` with no
    strict improvement -- the actual fix for a force camping forever next
    to a fight it can't win); then assigns a fresh goal to every
    still-alive force that now has none. A force currently retreating
    (`retreating`, see `update_task_force_stance`) is skipped entirely
    here -- a temporary tactical pull-back must never look like a stalled
    or abandoned goal. A force with a `DEFEND_PORT` goal is likewise
    skipped entirely: that goal is fully owned by
    `apply_defensive_port_priority` (assigned and released there, not
    here), and a force correctly camped at its already-reached port never
    "improves" its distance-to-target again -- without this exemption it
    would read as stalled after `task_force_stall_turns` and get reassigned
    away from the exact job it's doing."""
    ai_config = game_state.config.ai
    for force in forces:
        if not force.member_ids or len(force.member_ids) >= ai_config.task_force_min_size:
            continue
        # A carrier-anchored force is exempt: formation itself allows a
        # lone carrier with no escorts at all to form its own force (see
        # form_task_forces), so falling to just the carrier alone isn't a
        # "casualty" event for it the way it is for a non-carrier force,
        # which always needed task_force_min_size to form in the first
        # place.
        has_carrier = any(
            game_state.ships[i].kind == ShipKind.CARRIER for i in force.member_ids if i in game_state.ships
        )
        if not has_carrier:
            force.member_ids.clear()

    controlled = set(game_state.board.controlled_ports_for(player))
    stalled_ids: set[int] = set()
    for force in forces:
        if not force.member_ids or force.goal is None or force.retreating:
            continue
        if force.goal.kind == GoalKind.DEFEND_PORT:
            continue
        if force.goal.kind == GoalKind.CAPTURE_PORT and force.goal.target in controlled:
            force.goal = None
        elif force.turns_since_progress >= ai_config.task_force_stall_turns:
            exclude = frozenset({force.goal.target}) if force.goal.kind == GoalKind.CAPTURE_PORT else frozenset()
            force.goal = None
            assign_goal(force, game_state, forces, exclude=exclude)
            stalled_ids.add(force.id)

    for force in forces:
        if force.member_ids and force.goal is None and force.id not in stalled_ids:
            assign_goal(force, game_state, forces)


def record_task_force_progress(game_state: GameState, forces: list[TaskForce]) -> None:
    """Call once, at the end of a player's `plan_movement`, after every one
    of their ships has moved this turn. Tracks each force's best-ever
    (nearest-member) real sea-route distance (see `sea_distance_field` --
    not straight-line hex distance, which can plateau or even fluctuate
    while a force is genuinely, correctly working its way around a
    landmass toward the goal, wrongly reading as "no progress") to its own
    goal -- a ship camped forever at its closest non-attacking approach hex
    never strictly improves this, so `turns_since_progress` climbs every
    turn until `update_task_force_goals`'s stall check fires, instead of
    the force silently pursuing a stuck goal forever. A retreating force
    (`retreating`) is skipped -- moving away from the goal is expected and
    deliberate during a tactical pull-back, not a failure to progress."""
    for force in forces:
        if force.goal is None or force.retreating:
            continue
        members = [game_state.ships[i] for i in force.member_ids if i in game_state.ships]
        if not members:
            continue
        field = sea_distance_field(game_state, force.goal.target)
        current = min(sea_route_distance(field, m.position) for m in members)
        if force.best_progress_distance is None or current < force.best_progress_distance:
            force.best_progress_distance = current
            force.turns_since_progress = 0
        else:
            force.turns_since_progress += 1


def enemy_reachable_next_turn(enemy: Ship, game_state: GameState) -> set[AxialCoord]:
    """Every hex `enemy` could move to -- and so attack, if occupied -- on
    its own next turn, given a full movement budget from its current
    position: a real reachability check (terrain and other ships in the
    way accounted for, via `movement.reachable_hexes`), not a flat hex-
    distance radius. `enemy.movement_remaining` itself is stale for this:
    it's whatever was left over at the end of enemy's *last* turn, not
    the fresh budget its *next* turn actually resets to (see `Ship.
    max_movement`, set by `TurnManager` at the start of each of a
    player's movement phases) -- so a copy with that fresh budget stands
    in instead. Never mutates `enemy` itself, and the copy is never added
    to `game_state.ships`, so nothing else can ever observe or collide
    with it. Shared by a carrier's own threat-avoidance (tla.ai.policy.
    _choose_carrier_destination) and compute_carrier_defense_directives
    below -- both need the same "could this specific enemy actually hit
    me" answer, not just how far away it looks."""
    stats = game_state.config.ship_stats.stats[enemy.kind]
    fresh = replace(enemy, movement_remaining=enemy.max_movement(stats))
    return set(movement.reachable_hexes(fresh, game_state))


def with_repositioned_ships(game_state: GameState, positions: dict[int, AxialCoord]) -> GameState:
    """A hypothetical `GameState` with each ship id in `positions`
    relocated to its given coordinate -- every other field (every other
    ship, `board`, `config`, `battle_log`, etc.) shared by reference with
    `game_state`, never copied or mutated. Safe to build and discard
    freely within one turn's decision-making: a shallow `ships` dict copy
    plus one `dataclasses.replace` per repositioned ship, not a deep copy
    of anything -- nothing this is meant to feed (`battle.
    carrier_bonus_for`, `enemy_reachable_next_turn`, `movement.
    reachable_hexes`) ever writes to `game_state`, so sharing everything
    but `ships` by reference is exactly as safe as it looks.

    Only `position` changes on each repositioned ship -- `movement_
    remaining`/`surfaced`/`current_hp`/etc. are left exactly as the real
    ship has them, since none of the three functions above read anything
    else off a *different* ship's state when answering "what if this ship
    stood here": only its `position` (and its already-unchanged `owner`/
    `kind`/alive-or-sunk) matter to them. A ship id in `positions` that
    isn't (or is no longer) in `game_state.ships` is silently ignored --
    nothing to reposition."""
    if not positions:
        return game_state
    ships = dict(game_state.ships)
    for ship_id, coord in positions.items():
        if ship_id in ships:
            ships[ship_id] = replace(ships[ship_id], position=coord)
    return replace(game_state, ships=ships)


def group_power(ships: list[Ship], game_state: GameState) -> tuple[int, int]:
    """(total current HP, total damage-stat output) for a group of ships
    -- a rough, naive-tier proxy for combined combat strength, shared
    (with `outmatched`) by every group-strength comparison in this module:
    `is_outnumbered`, `is_outmatched_by_target_support`, and
    `compute_port_defense_directives`. Deliberately ignores the
    asw-vs-submerged-sub asymmetry and carrier bonuses (see
    `battle._base_damage`/`carrier_bonus_for`) -- an exact N-vs-M combat
    simulation is out of scope here; this only needs to answer "are we
    clearly overmatched," not predict an exact outcome."""
    stats = game_state.config.ship_stats.stats
    total_hp = sum(s.current_hp for s in ships)
    total_damage = sum(stats[s.kind].damage for s in ships)
    return total_hp, total_damage


def outmatched(power_a: tuple[int, int], power_b: tuple[int, int], margin: int) -> bool:
    """Whether group A (`power_a` = (total hp, total damage)) loses a naive
    toe-to-toe "rounds to kill" race against group B (`power_b`) by more
    than `margin` -- the shared comparison behind `is_outnumbered` and
    `is_outmatched_by_target_support`. Naive and deliberately approximate:
    no formation, focus-fire, or terrain modeling, just "is this clearly a
    bad fight." `margin` requires a clearer disadvantage before tripping,
    to avoid flapping right at the threshold."""
    a_hp, a_damage = power_a
    b_hp, b_damage = power_b
    if a_damage <= 0:
        return b_damage > 0
    if b_damage <= 0:
        return False
    rounds_to_kill_b = math.ceil(b_hp / a_damage)
    rounds_to_kill_a = math.ceil(a_hp / b_damage)
    return rounds_to_kill_a + margin < rounds_to_kill_b


def _nearby_enemies(
    force: TaskForce, game_state: GameState, visible_enemies: dict[int, Ship], ai_config: AiConfig
) -> list[Ship]:
    """Visible enemies currently within `AiConfig.task_force_threat_radius`
    of any of `force`'s living members -- shared by `is_outnumbered` and
    `_threat_snapshot` so both agree on what "nearby" means."""
    members = [game_state.ships[i] for i in force.member_ids if i in game_state.ships]
    if not members:
        return []
    return [
        e
        for e in visible_enemies.values()
        if any(distance(e.position, m.position) <= ai_config.task_force_threat_radius for m in members)
    ]


def is_outnumbered(
    force: TaskForce, game_state: GameState, visible_enemies: dict[int, Ship], ai_config: AiConfig
) -> bool:
    """Whether the enemies currently within `AiConfig.task_force_threat_
    radius` of any of `force`'s members outmatch it (see `outmatched`),
    generalizing the single-ship "rounds to kill" race
    (`tla.ai.scoring.matchup_score`) to the whole group via summed
    HP/damage (`group_power`). False if the force has no living members
    or no enemy is within range."""
    members = [game_state.ships[i] for i in force.member_ids if i in game_state.ships]
    if not members:
        return False
    nearby = _nearby_enemies(force, game_state, visible_enemies, ai_config)
    if not nearby:
        return False
    return outmatched(
        group_power(members, game_state),
        group_power(nearby, game_state),
        ai_config.task_force_outnumbered_margin,
    )


def _threat_snapshot(
    force: TaskForce, game_state: GameState, visible_enemies: dict[int, Ship], ai_config: AiConfig
) -> tuple[int, int]:
    """(total hp, total damage) of whatever `force` is retreating from,
    captured the instant retreat triggers -- see `TaskForce.retreat_
    threat_power`. Prefers the same nearby-within-threat-radius group
    `is_outnumbered` itself used, if any; falls back to every currently
    visible enemy of the force's owner (covers an `is_attritting`-only
    trigger, where the immediate threat radius can be momentarily empty
    even though a fight is genuinely ongoing); `(0, 0)` if nothing is
    visible at all, which `outmatched` already treats correctly -- a
    powerless "threat" is trivially outmatched by anything with positive
    damage, so the force stops retreating almost immediately, which is
    correct since there's nothing left to size up against."""
    nearby = _nearby_enemies(force, game_state, visible_enemies, ai_config)
    group = nearby or list(visible_enemies.values())
    return group_power(group, game_state) if group else (0, 0)


def is_attritting(force: TaskForce, ai_config: AiConfig) -> bool:
    """Whether `force` has been losing members for
    `AiConfig.task_force_attrition_turns` turns in a row while also making
    no progress toward its own goal (`turns_since_progress > 0`) -- a
    losing war of attrition even when no single instant looks clearly
    `is_outnumbered`. An enemy that reinforces piecemeal, rather than ever
    presenting one obviously-superior force at once, can keep a fight
    looking "close" turn after turn while still grinding a force down to
    nothing; this catches that pattern directly from its actual outcome
    (are we shrinking, are we getting anywhere) rather than from a
    snapshot comparison that a well-timed drip of reinforcements can dodge
    indefinitely."""
    return (
        force.consecutive_loss_turns >= ai_config.task_force_attrition_turns
        and force.turns_since_progress > 0
    )


def is_outmatched_by_target_support(
    attacker_group: list[Ship],
    target: Ship,
    game_state: GameState,
    visible_enemies: dict[int, Ship],
    ai_config: AiConfig,
) -> bool:
    """Whether attacking `target` actually means fighting `target` plus
    every other visible enemy within `AiConfig.task_force_threat_radius`
    of it -- backup that could also be drawn into a developing fight --
    outweighing `attacker_group` by the same race `is_outnumbered` uses.
    Guards against a small group (or a lone unassigned ship) picking a
    fight with what looks like an easy 1v1 target that's actually
    screening a much larger force nearby. False if `attacker_group` is
    empty."""
    if not attacker_group:
        return False
    target_group = [
        e
        for e in visible_enemies.values()
        if distance(e.position, target.position) <= ai_config.task_force_threat_radius
    ]
    return outmatched(
        group_power(attacker_group, game_state),
        group_power(target_group, game_state),
        ai_config.task_force_outnumbered_margin,
    )


@dataclass
class PortDefenseDirective:
    """One controlled port's defensive response for this turn -- see
    compute_port_defense_directives. Exactly one of `counterattack` /
    (`block`, `block_hex`) is populated, never both: either every listed
    responder should move to engage `threats` this turn (counterattack),
    or a single ship should occupy `block_hex` to delay them (block)."""

    port: AxialCoord
    threats: list[Ship]
    counterattack: list[Ship] = field(default_factory=list)
    block: Ship | None = None
    block_hex: AxialCoord | None = None


def _choose_blocker(responders: list[Ship]) -> Ship:
    """The delaying picket for an outmatched port defense: a submarine
    first -- its own stealth submerge (already handled by the main loop's
    `_maybe_toggle_submarine`, since a blocker isn't making an attack) is
    what makes blocking actually work, since the enemy can't route around
    a threat it can't see -- then a patrol boat (cheapest to lose), else
    whichever responder sorts first by id, for determinism."""
    return min(
        responders,
        key=lambda s: (0 if s.kind == ShipKind.SUBMARINE else 1, 0 if s.kind == ShipKind.PATROL_BOAT else 1, s.id),
    )


def block_hex_toward(
    threats: list[Ship],
    target: AxialCoord,
    field: dict[AxialCoord, int],
    game_state: GameState,
    prefer_radius: int | None = None,
) -> AxialCoord | None:
    """The hex the threat nearest `target` (by the same sea-route `field`
    used to detect it) would take *next* on its own shortest sea route
    there -- sending a blocker to occupy it forces a detour or a fight
    instead of an unopposed walk-in. `target` is a port for `compute_
    port_defense_directives` (a land hex, snapped to its nearest sea
    neighbor below) or a carrier's own current position for `compute_
    carrier_defense_directives` (already a sea hex, so the snap is a
    no-op there). None if no sea route exists between them at all (a
    disconnected map). Public -- `tla.ai.policy._carrier_screen_
    destination` needs this same "interposing hex" primitive too, same
    reasoning as every other helper here promoted to public the moment a
    second module needs it.

    `prefer_radius`, if given, doesn't just take the very next step on
    the threat's route (`path[1]`) -- it walks the path from the threat's
    end toward `target` and returns the *first* hex already within
    `prefer_radius` of `target`, so a caller that wants a blocking hex
    that ALSO keeps the blocker close to `target` gets one when the path
    actually offers one (a real replay case, game65 turn 2: BB1 four sea
    hexes from an AI carrier, `carrier_heavy_screen_radius`; the plain
    `path[1]` choice put the screening cruiser three hexes from its own
    carrier -- outside `CombatConfig.ac_bonus_radius` -- even though two
    later points on that exact same path, (8,4) and (8,3), block the same
    route just as well while staying within it). Occupying *any* hex on
    the threat's shortest route forces the same detour-or-fight, so
    there's no blocking-strength cost to preferring the one closest to
    the threat (earliest interception) among the radius-compliant
    options, rather than the one closest to `target`. Falls back to the
    plain `path[1]` behavior if no hex on the path satisfies the radius
    at all -- screening priority over coverage, same doctrine already
    documented at every other caller of this tradeoff."""
    sea = {c for c, t in game_state.board.tiles.items() if t.terrain == TerrainType.SEA}
    if not sea:
        return None
    nearest_threat = min(threats, key=lambda t: (sea_route_distance(field, t.position), t.id))
    start = _nearest_in(nearest_threat.position, sea)
    goal = _nearest_in(target, sea)
    path = _shortest_sea_path(start, goal, sea)
    if not path:
        return None
    if prefer_radius is not None:
        # path[-1] is target's own (snapped) hex -- already occupied by
        # whatever this is blocking for, and not itself a position "in
        # between" the threat and target, so it's excluded from
        # consideration even when within radius.
        for hex_ in path[1:-1]:
            if distance(hex_, target) <= prefer_radius:
                return hex_
    return path[1] if len(path) > 1 else path[0]


def compute_port_defense_directives(
    game_state: GameState,
    player: PlayerId,
    visible_enemies: dict[int, Ship],
    ai_config: AiConfig,
) -> list[PortDefenseDirective]:
    """User-specified defensive doctrine, added after self-play showed the
    naive AI committing every ship to offense and letting a small enemy
    raiding party walk into an undefended, still-controlled port: for each
    of `player`'s controlled ports (`Board.controlled_ports_for`) with a
    visible enemy within `AiConfig.port_defense_trigger_radius` sea hexes,
    gather `player`'s own ships within `AiConfig.port_defense_response_
    radius` sea hexes of that port as candidate responders -- anything
    farther is judged too far to arrive in time and is left alone to keep
    doing whatever it's already doing, rather than recalled.

    If those responders collectively are not outmatched by the threat
    (`group_power`/`outmatched`, `AiConfig.port_defense_margin` -- 0 means
    "at least equal strength"), every one of them gets a `counterattack`
    directive. Otherwise, a single responder (`_choose_blocker`) is sent
    to physically block the threat's own route in (`block_hex_toward`)
    -- a deliberate, expected loss that trades one ship to slow the
    takeover rather than losing the port outright while the rest of the
    fleet is still committed elsewhere. See `compute_carrier_defense_
    directives` just below for the same doctrine applied to a threatened
    carrier instead of a threatened port.

    Ports are processed in a fixed (sorted) order and no ship is claimed
    by more than one port's directive in the same turn -- a ship already
    claimed defending one threatened port isn't double-counted as a
    responder for another. Returns one directive per threatened port that
    has at least one candidate responder; an untouched, unthreatened port,
    or a threatened one with nothing close enough to help, contributes
    nothing."""
    ports = sorted(game_state.board.controlled_ports_for(player))
    claimed: set[int] = set()
    directives: list[PortDefenseDirective] = []
    for port in ports:
        field = sea_distance_field(game_state, port)
        threats = [
            e
            for e in visible_enemies.values()
            if sea_route_distance(field, e.position) <= ai_config.port_defense_trigger_radius
        ]
        if not threats:
            continue
        responders = [
            s
            for s in game_state.ships_for(player)
            if s.id not in claimed
            and sea_route_distance(field, s.position) <= ai_config.port_defense_response_radius
        ]
        if not responders:
            continue
        if not outmatched(
            group_power(responders, game_state), group_power(threats, game_state), ai_config.port_defense_margin
        ):
            claimed.update(s.id for s in responders)
            directives.append(PortDefenseDirective(port=port, threats=threats, counterattack=responders))
        else:
            blocker = _choose_blocker(responders)
            block_hex = block_hex_toward(threats, port, field, game_state)
            if block_hex is not None:
                claimed.add(blocker.id)
                directives.append(PortDefenseDirective(port=port, threats=threats, block=blocker, block_hex=block_hex))
    return directives


@dataclass
class CarrierDefenseDirective:
    """One friendly carrier's defensive response for this turn -- see
    compute_carrier_defense_directives. Same shape as PortDefenseDirective:
    exactly one of `counterattack` / (`block`, `block_hex`) is populated."""

    carrier: Ship
    threats: list[Ship]
    counterattack: list[Ship] = field(default_factory=list)
    block: Ship | None = None
    block_hex: AxialCoord | None = None


def compute_carrier_defense_directives(
    game_state: GameState,
    player: PlayerId,
    visible_enemies: dict[int, Ship],
    ai_config: AiConfig,
    already_claimed: frozenset[int] = frozenset(),
) -> list[CarrierDefenseDirective]:
    """User-specified doctrine, added after a real game showed the naive
    AI declining to engage a dangerous enemy converging on one of its own
    carriers -- purely because the fight itself was an exact tie. A
    mutual kill is still far better than losing an unescorted carrier for
    nothing next turn, the same "sometimes a sacrifice is worth it"
    reasoning `compute_port_defense_directives` already applies to a
    threatened port; this mirrors that function almost exactly, just
    keyed on a living carrier's own current position instead of a
    controlled port's fixed one, and with no separate trigger radius of
    its own -- a carrier is "threatened" exactly when `enemy_reachable_
    next_turn` says a `DANGEROUS_TO_CARRIER_KINDS` enemy could reach its
    hex, the same real reachability check its own individual retreat (see
    `tla.ai.policy._choose_carrier_destination`) already uses, rather
    than a second, looser proxy for the same thing.

    For each of `player`'s carriers under threat, gathers `player`'s own
    ships within `AiConfig.carrier_defense_response_radius` sea hexes of
    it as candidate responders -- anything farther is judged too far to
    arrive in time. If those responders collectively are not outmatched
    by the threat (`group_power`/`outmatched`, `AiConfig.carrier_defense_
    margin`), every one of them gets a `counterattack` directive --
    deliberately tolerant of an exact tie (see `tla.ai.policy.
    _carrier_defense_destination`, which accepts `matchup_score >= 0` for
    this directive specifically, not the strictly-favorable `> 0` bar
    ordinary combat uses), since the group-level comparison here already
    decided the trade is worth it. Otherwise, a single responder
    (`_choose_blocker`) physically blocks the threat's own route in
    (`block_hex_toward`), same delaying-sacrifice tactic as port defense.

    Carriers are processed in a fixed (sorted by id) order. `already_
    claimed` seeds the claimed-ship set -- callers run this after `compute_
    port_defense_directives` in the same turn and pass its claims through,
    since losing a controlled port outright is judged worse than losing an
    already-endangered carrier, so port defense gets first call on any
    ship both could use. A carrier that's safe, or has nothing close
    enough to help, contributes nothing."""
    carriers = sorted(
        (s for s in game_state.ships_for(player) if s.kind == ShipKind.CARRIER),
        key=lambda s: s.id,
    )
    claimed: set[int] = set(already_claimed)
    directives: list[CarrierDefenseDirective] = []
    for carrier in carriers:
        threats = [
            e
            for e in visible_enemies.values()
            if e.kind in DANGEROUS_TO_CARRIER_KINDS
            and carrier.position in enemy_reachable_next_turn(e, game_state)
        ]
        if not threats:
            continue
        field = sea_distance_field(game_state, carrier.position)
        responders = [
            s
            for s in game_state.ships_for(player)
            if s.id not in claimed
            and s.id != carrier.id
            and s.kind != ShipKind.CARRIER
            and sea_route_distance(field, s.position) <= ai_config.carrier_defense_response_radius
        ]
        if not responders:
            continue
        if not outmatched(
            group_power(responders, game_state), group_power(threats, game_state), ai_config.carrier_defense_margin
        ):
            claimed.update(s.id for s in responders)
            directives.append(CarrierDefenseDirective(carrier=carrier, threats=threats, counterattack=responders))
        else:
            blocker = _choose_blocker(responders)
            block_hex = block_hex_toward(threats, carrier.position, field, game_state)
            if block_hex is not None:
                claimed.add(blocker.id)
                directives.append(
                    CarrierDefenseDirective(carrier=carrier, threats=threats, block=blocker, block_hex=block_hex)
                )
    return directives


def rally_point(force: TaskForce, game_state: GameState) -> AxialCoord | None:
    """Nearest currently-controlled port to `force`'s centroid -- where a
    retreating force falls back to. Public so
    `tla.ai.policy._choose_destination` can use it to build the ephemeral
    `RETREAT` goal override. `None` if the force's owner controls no ports
    at all (a near-defeat edge case) -- callers then just leave the force
    pursuing its real goal, since there's nowhere better to retreat to
    anyway."""
    members = [game_state.ships[i] for i in force.member_ids if i in game_state.ships]
    if not members:
        return None
    controlled = game_state.board.controlled_ports_for(force.owner)
    if not controlled:
        return None
    centroid = _centroid([m.position for m in members])
    return min(controlled, key=lambda p: (distance(centroid, p), p))


def _pullback_waypoint(force: TaskForce, game_state: GameState, pullback_hexes: int) -> AxialCoord | None:
    """A retreat destination only `pullback_hexes` real-sea-route steps
    from `force`'s current centroid toward its rally port (see
    `rally_point`), rather than the port itself -- see
    `AiConfig.retreat_pullback_hexes`. Snapshotted once when retreat
    triggers (see `TaskForce.retreat_waypoint`), not recomputed turn to
    turn, so the force actually holds at a fixed point rather than a
    moving target that recedes as it approaches. Falls through to the
    port itself if the real route there is already no longer than the cap
    (nothing to shorten), or if there's no rally point or living members
    at all -- both match `rally_point`'s own `None`/no-op cases."""
    members = [game_state.ships[i] for i in force.member_ids if i in game_state.ships]
    if not members:
        return None
    port = rally_point(force, game_state)
    if port is None:
        return None
    sea = {c for c, t in game_state.board.tiles.items() if t.terrain == TerrainType.SEA}
    if not sea:
        return port
    centroid = _centroid([m.position for m in members])
    start = _nearest_in(centroid, sea)
    goal = _nearest_in(port, sea)
    path = _shortest_sea_path(start, goal, sea)
    if not path or len(path) <= pullback_hexes + 1:
        return port
    return path[pullback_hexes]


def _near_a_controlled_port(force: TaskForce, game_state: GameState, max_distance: int) -> bool:
    """Whether `force`'s centroid is within `max_distance` real sea-route
    hexes (see `sea_distance_field`) of any port it currently controls --
    used by `update_task_force_stance` to let a retreating force stand and
    fight once close enough to home, instead of only ever stopping once
    strictly stronger than what it fled. False (never "near") if the force
    has no living members or controls no ports at all."""
    members = [game_state.ships[i] for i in force.member_ids if i in game_state.ships]
    if not members:
        return False
    controlled = game_state.board.controlled_ports_for(force.owner)
    if not controlled:
        return False
    centroid = _centroid([m.position for m in members])
    field = sea_distance_field(game_state, centroid)
    return min(sea_route_distance(field, p) for p in controlled) <= max_distance


def update_task_force_stance(
    game_state: GameState,
    player: PlayerId,
    forces: list[TaskForce],
    visible_enemies: dict[int, Ship],
    should_retreat_by_force: dict[int, bool] | None = None,
) -> None:
    """Per-turn retreat-stance maintenance, called once per player right
    after formation and before `update_task_force_goals` -- a force with
    an active goal that's currently outnumbered near its own position
    (see `is_outnumbered`), that's been grinding down in a losing war of
    attrition without either side's advantage ever looking decisive at
    any single instant (see `is_attritting`), or whose own multi-candidate
    tactical evaluation this turn (`tla.ai.policy._pick_strategy_for_force`,
    via `should_retreat_by_force` -- keyed by force id, `True` means that
    evaluation's best-scoring strategy among AGGRESSIVE/ADVANCE/HOLD/RETREAT
    was RETREAT: not just "this turn's default advance is a bad trade," but
    "no candidate this turn -- including holding in place or fighting -- beats
    pulling back") judges retreat as the better call, pulls back toward its
    nearest controlled port rather than continuing to press its goal,
    without abandoning that goal the way a stall does: `force.goal` itself
    is never touched here, only `retreating`, so it's resumed exactly
    where it left off once it's safe. Home isn't safety for its own sake --
    it's just where `recruit_into_open_forces` finds new production to
    absorb (untouched by retreat state, so this happens automatically) --
    so "safe" is measured directly against that: retreat continues until
    the force's own current strength (`group_power`, recomputed fresh
    each turn from its live membership) clearly outmatches (by the same
    `AiConfig.task_force_outnumbered_margin` `is_outnumbered` itself uses)
    a snapshot of whatever it retreated from (`retreat_threat_power`, see
    `_threat_snapshot`), taken once at the moment retreat triggers -- not
    a fixed turn count, and not "no longer outnumbered *right now*" (which
    a retreating force moving out of `task_force_threat_radius` could
    satisfy trivially, without actually having gotten any stronger).

    `AiConfig.task_force_max_retreat_turns` is a safety valve, not the
    normal path to resuming: if the opponent is reinforcing just as fast,
    the force could otherwise retreat forever without ever clearing the
    bar. Hitting it gives up on the goal via the same stall/exclude path
    `update_task_force_goals` uses, instead of retreating indefinitely.

    `AiConfig.retreat_cancel_port_distance`, if set, adds a second,
    independent way to stop retreating: once the force is within that many
    real sea-route hexes of one of its own controlled ports (see
    `_near_a_controlled_port`), it stands and fights right there even if
    it still hasn't cleared `task_force_outnumbered_margin`. Disabled
    (`None`) by default -- see the field's own docstring for why this
    exists at all.

    `AiConfig.reset_progress_on_distance_resume` controls whether a
    distance-based resume (alone, not backed by real strength) also resets
    `turns_since_progress`/`best_progress_distance` the way a
    strength-based resume always does -- see the field's own docstring: a
    real game showed a force cycling through cheap, fast distance-based
    resumes near its own territory, each one erasing genuine progress
    toward a distant goal, trapping the whole war on one side of the map.

    `AiConfig.retreat_pullback_hexes`, if set, caps how far a retreat
    actually travels (see `_pullback_waypoint`/`TaskForce.retreat_
    waypoint`, snapshotted once at trigger time) instead of always heading
    all the way to the rally port -- a force already reasonably placed
    doesn't have to fully withdraw before the resume checks above can even
    apply."""
    ai_config = game_state.config.ai
    should_retreat_by_force = should_retreat_by_force or {}
    for force in forces:
        if not force.member_ids or force.goal is None:
            continue

        if force.retreating:
            force.retreat_turns += 1
            members = [game_state.ships[i] for i in force.member_ids if i in game_state.ships]
            own_power = group_power(members, game_state)
            safe_by_strength = outmatched(
                force.retreat_threat_power, own_power, ai_config.task_force_outnumbered_margin
            )
            safe_by_distance = not safe_by_strength and (
                ai_config.retreat_cancel_port_distance is not None
                and _near_a_controlled_port(force, game_state, ai_config.retreat_cancel_port_distance)
            )
            if safe_by_strength or safe_by_distance:
                # Either clearly stronger than what we retreated from, or
                # close enough to home to stand and fight anyway.
                force.retreating = False
                force.retreat_threat_power = None
                force.retreat_turns = 0
                force.retreat_waypoint = None
                if safe_by_strength or ai_config.reset_progress_on_distance_resume:
                    # A fresh stall-tracking window, not counting the
                    # retreat itself as "no progress" -- earned by actually
                    # getting stronger, or explicitly opted into for a
                    # distance-only resume too.
                    force.turns_since_progress = 0
                    force.best_progress_distance = None
            elif force.retreat_turns >= ai_config.task_force_max_retreat_turns:
                exclude = frozenset({force.goal.target}) if force.goal.kind == GoalKind.CAPTURE_PORT else frozenset()
                force.goal = None
                force.retreating = False
                force.retreat_threat_power = None
                force.retreat_turns = 0
                force.retreat_waypoint = None
                assign_goal(force, game_state, forces, exclude=exclude)
            continue  # still retreating (or just gave up) -- nothing more to decide this turn

        if (
            is_outnumbered(force, game_state, visible_enemies, ai_config)
            or is_attritting(force, ai_config)
            or should_retreat_by_force.get(force.id, False)
        ):
            force.retreating = True
            force.retreat_turns = 0
            force.retreat_threat_power = _threat_snapshot(force, game_state, visible_enemies, ai_config)
            force.retreat_waypoint = (
                _pullback_waypoint(force, game_state, ai_config.retreat_pullback_hexes)
                if ai_config.retreat_pullback_hexes is not None
                else None
            )


def is_open(force: TaskForce, ai_config: AiConfig) -> bool:
    """Whether `force` is under its target minimum size and should
    actively try to absorb new or stray unassigned ships (see
    `recruit_into_open_forces`) instead of leaving them to spin up a
    brand-new, separate force. Purely derived from current membership,
    not stored state, so it's always consistent -- no separate flag to
    keep in sync."""
    return len(force.member_ids) < ai_config.task_force_target_min_size


def recruit_into_open_forces(
    game_state: GameState, player: PlayerId, forces: list[TaskForce], ai_config: AiConfig
) -> None:
    """Let every existing force that's currently `is_open` (under-strength)
    absorb nearby unassigned ships -- newly produced ones, survivors of a
    dissolved force, or any other stray -- growing it back toward
    `AiConfig.task_force_target_max_size`, instead of `form_task_forces`
    (called right after this, on whatever's left over) always spinning up
    a brand-new, separate force from the same pool.

    Processed in ascending force-id order for determinism. A force at or
    above its target minimum, or already at its target maximum, is left
    alone. Never pulls a ship away from a force it's already a member of
    -- only ever from the fully-unassigned pool, the same restriction
    `form_task_forces` already has. Unlike initial formation (`form_task_
    forces`, which keeps each carrier anchoring its own separate force to
    avoid double-booking one mid-pass), recruitment has no such
    restriction: an open force can absorb any number of stray carriers
    over time, which is how multiple carriers actually end up consolidated
    under one force -- gradually, as later ones become unassigned, not all
    at once from turn one."""
    assigned = {i for f in forces for i in f.member_ids}
    pool = [s for s in game_state.ships_for(player) if s.id not in assigned]

    for force in sorted(forces, key=lambda f: f.id):
        if not force.member_ids or not is_open(force, ai_config):
            continue
        capacity = ai_config.task_force_target_max_size - len(force.member_ids)
        if capacity <= 0:
            continue
        members = [game_state.ships[i] for i in force.member_ids if i in game_state.ships]
        if not members:
            continue
        centroid = _centroid([m.position for m in members])
        picked = _gather(
            centroid,
            frozenset(),
            pool,
            ai_config.task_force_recruit_radius,
            capacity,
            exclude_carriers=False,
            existing_kinds=Counter(m.kind for m in members),
        )
        if picked:
            force.member_ids |= picked
            pool = [s for s in pool if s.id not in picked]
