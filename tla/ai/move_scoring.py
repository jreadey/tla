"""Scoring-based task-force movement -- a prototype replacement for the
fixed procedural priority chain `tla.ai.policy.choose_task_force_destination`
normally applies (cohesion > screening > carrier-bonus-cohesion > rearguard
> goal-advance), and for the separate `secure_kills_pass`/carrier-scouting
passes that run ahead of it. See `AiConfig.scored_task_force_movement_enabled`
for the full motivation: whichever procedural pass claims a ship first wins
under the old chain, regardless of how much better a later-checked option
might have been -- a cruiser ordered to screen a carrier never gets to
compare that against a better available multi-kill elsewhere, since
screening is simply checked earlier. This module instead scores every
reachable hex for every candidate ship directly against each other.

Scope, exactly matching `AiConfig.scored_task_force_movement_enabled`'s own
doc-comment: only a force's members, only when its chosen `Strategy` this
turn is `ADVANCE` or `AGGRESSIVE` and it isn't `retreating`. Port/carrier
defense (still unconditional-priority, checked first, by the caller) and
`Strategy.RETREAT`/`HOLD` (still their own dedicated paths) are untouched
either way -- genuinely different kinds of decisions ("respond to an
emergency now" / "disengage"), not what this redesign is about.

Deliberately has no dependency on `tla.ai.policy` (which imports this
module instead) -- `apply_move` is injected by the caller exactly the way
`tla.ai.tactics.secure_kills_pass` already injects `apply_attack`, for the
identical reason: policy.py's own submarine-toggle/`_execute`/strategy-
reevaluation logic can't be imported back into this module without a cycle.
"""

from __future__ import annotations

from typing import Callable, Iterator

from tla import movement
from tla.ai import scoring
from tla.ai.enemy_model import EnemyModel
from tla.ai.tactics import action_value
from tla.ai.task_force import TaskForce, sea_distance_field, sea_route_distance
from tla.config import AiConfig
from tla.game_state import GameState
from tla.hexgrid import AxialCoord, distance, neighbors
from tla.movement import _classify_step
from tla.ship import Ship, ShipKind
from tla.tile import PlayerId

# Round-robin planning order (tla.ai.move_scoring.plan_force_movement_scored):
# capital ships first, patrol boats last. Found necessary via real play-
# testing (game68): a fast/cheap ship (patrol boat, movement 6 -- the
# highest of any kind) can otherwise commit to a huge solo leap before the
# rest of the force has had any chance to plan its own, much more modest,
# advance -- see plan_force_movement_scored's own docstring.
_ROUND_ROBIN_PRIORITY: dict[ShipKind, int] = {
    ShipKind.BATTLESHIP: 0,
    ShipKind.CARRIER: 0,
    ShipKind.CRUISER: 1,
    ShipKind.DESTROYER: 2,
    ShipKind.SUBMARINE: 2,
    ShipKind.PATROL_BOAT: 3,
}

# A ship kind that can meaningfully stand between a carrier and a threat,
# and how much credit it gets for doing so -- mirrors policy.py's
# _HEAVY_SCREEN_KINDS/_LIGHT_SCREEN_KINDS split (battleship/cruiser vs.
# destroyer/submarine) but as a graded multiplier instead of a hard
# category gate, extended to patrol boats at a low weight (the user's own
# example: "more points if it's a BB or CA... less for a DD or SS, less
# for a PB") -- a carrier itself never screens another carrier here, same
# as it never solo-attacks being a matter of its own exposure term
# dominating rather than a hard exclusion (see AiConfig's own doc-comment
# on why this session deliberately stopped hard-coding that rule).
_SCREEN_MULTIPLIER: dict[ShipKind, float] = {
    ShipKind.BATTLESHIP: 1.0,
    ShipKind.CRUISER: 0.8,
    ShipKind.DESTROYER: 0.5,
    ShipKind.SUBMARINE: 0.5,
    ShipKind.PATROL_BOAT: 0.15,
}
# Threats worth screening against at all -- everything but a carrier or
# patrol boat, matching what policy.py's own screening doctrine already
# treats as dangerous enough to matter.
_SCREENABLE_THREAT_KINDS = frozenset(
    {ShipKind.BATTLESHIP, ShipKind.CRUISER, ShipKind.DESTROYER, ShipKind.SUBMARINE}
)
_CAPITAL_KINDS = frozenset({ShipKind.BATTLESHIP, ShipKind.CARRIER})
_ESCORT_KINDS = frozenset({ShipKind.PATROL_BOAT, ShipKind.DESTROYER})


def _goal_progress_term(destination: AxialCoord, goal_field: dict[AxialCoord, int] | None) -> float:
    if goal_field is None:
        return 0.0
    return -float(sea_route_distance(goal_field, destination))


def _attack_term(ship: Ship, destination: AxialCoord, game_state: GameState, ai_config: AiConfig) -> float:
    target = game_state.ship_at(destination)
    if target is None or target.owner == ship.owner:
        return 0.0
    return action_value(ship, target, game_state, ai_config)


def _screen_term(
    ship: Ship,
    destination: AxialCoord,
    game_state: GameState,
    force: TaskForce,
    visible_enemies: dict[int, Ship],
) -> float:
    """Whether `destination` genuinely stands between one of the force's
    own carriers and a visible dangerous threat -- "genuinely" meaning
    `destination` is closer to the threat than the carrier itself is
    (same cheap distance-based detection `_carrier_screen_destination`'s
    own `is_screened` check uses; picking the actual best interposing hex
    via real sea-route pathing, `block_hex_toward`, is that function's
    job, not scoring's -- here we're just judging candidates already on
    the table). Scaled by `_SCREEN_MULTIPLIER[ship.kind]`; 0 for a kind
    not in that mapping (a carrier never screens another carrier)."""
    multiplier = _SCREEN_MULTIPLIER.get(ship.kind, 0.0)
    if multiplier <= 0.0:
        return 0.0
    carriers = [
        game_state.ships[i]
        for i in force.member_ids
        if i in game_state.ships and game_state.ships[i].kind == ShipKind.CARRIER
    ]
    if not carriers:
        return 0.0
    threats = [e for e in visible_enemies.values() if e.kind in _SCREENABLE_THREAT_KINDS]
    if not threats:
        return 0.0
    best = 0.0
    for carrier in carriers:
        for threat in threats:
            carrier_distance = distance(carrier.position, threat.position)
            if distance(destination, threat.position) >= carrier_distance:
                continue  # not actually in front of the threat
            # Credit scales with how much closer than the carrier itself
            # this candidate stands -- a hex right next to the threat is
            # a far better screen than one merely a hair closer than the
            # carrier is.
            best = max(best, multiplier * (carrier_distance - distance(destination, threat.position)))
    return best


def _cohesion_term(
    ship: Ship,
    destination: AxialCoord,
    game_state: GameState,
    force: TaskForce,
    ai_config: AiConfig,
) -> float:
    """Penalty for drifting beyond `task_force_max_separation` from the
    nearest other living force member, plus a smaller bonus for staying
    within `CombatConfig.ac_bonus_radius` of a friendly carrier -- the
    same two distance concepts `_cohesion_destination`/`_carrier_bonus_
    cohesion_destination` enforce as hard corrections today, here just
    graded scoring terms instead."""
    others = [
        game_state.ships[i].position
        for i in force.member_ids
        if i in game_state.ships and i != ship.id
    ]
    penalty = 0.0
    if others and ai_config.task_force_max_separation is not None:
        nearest = min(distance(destination, p) for p in others)
        penalty = -max(0.0, nearest - ai_config.task_force_max_separation)
    bonus = 0.0
    carriers = [
        game_state.ships[i].position
        for i in force.member_ids
        if i in game_state.ships and game_state.ships[i].kind == ShipKind.CARRIER and i != ship.id
    ]
    if carriers:
        nearest_carrier = min(distance(destination, p) for p in carriers)
        if nearest_carrier <= game_state.config.combat.ac_bonus_radius:
            bonus = 1.0
    return penalty + bonus


def _exposure_term(
    ship: Ship,
    destination: AxialCoord,
    game_state: GameState,
    model: EnemyModel,
    ai_config: AiConfig,
) -> float:
    """Expected damage `ship` risks taking next turn by ending at
    `destination`, weighted by what it would cost to replace `ship`
    (`ShipStats.cost`, plus `AiConfig.carrier_assist_value` for a
    carrier -- losing one also costs the fleet its ac-bonus coverage, not
    just the hull). Reads `EnemyModel.expected_strength_near`, not just
    currently-visible enemies -- a believed-but-unconfirmed threat (a
    submerged submarine, most notably) should already discourage a risky
    move even before it's actually sighted. This is the uniform version
    of "risking the carrier should lower the score": every ship's own
    exposure is weighted by its own value, a carrier is not a special
    case.

    If `destination` is itself an attack (enemy-occupied), two corrections
    account for what the attack's own outcome does to `ship`'s survival --
    both found via a real example (a submerged submarine declining a
    submarine-for-battleship trade solely because a second battleship
    happened to be sitting next to the first): the target's own damage is
    subtracted out of the believed total first, since landing the attack
    is exactly what resolves that particular threat before any "next
    turn" could arrive -- scoring it as still-present danger would
    double-count the fight `_attack_term`'s own `action_value` already
    prices in the outcome of. Second, and more fundamental: if this
    attack is a worthwhile tie (`scoring.worth_a_tie`, `matchup_score ==
    0` -- this engine's combat is deterministic, so a tie always means
    mutual destruction, never a maybe), `ship` itself dies in this same
    engagement and exposure is exactly 0 rather than computed as if it
    survived -- there is no future left for a dead ship to be exposed in.
    A clean win (`matchup_score > 0`) does leave `ship` alive, often at
    reduced HP, so real future exposure remains -- but netted against
    `action_value`'s own number for this exact attack, since a battleship
    just secured is worth discounting a good deal of "but now I'm at 1 HP
    next to another battleship" against, not charging in full on top of
    the value already banked."""
    target = game_state.ship_at(destination)
    is_enemy_attack = target is not None and target.owner != ship.owner
    if is_enemy_attack:
        score = scoring.matchup_score(ship, target, game_state)
        if score == 0 and scoring.worth_a_tie(ship, target, game_state):
            return 0.0  # ship dies in this same engagement -- no future to protect

    _hp, believed_damage = model.expected_strength_near(destination, ai_config.move_score_sub_threat_radius)
    if is_enemy_attack:
        target_stats = game_state.config.ship_stats.stats[target.kind]
        believed_damage = max(0.0, believed_damage - target_stats.damage)
    stats = game_state.config.ship_stats.stats[ship.kind]
    value = stats.cost + (ai_config.carrier_assist_value if ship.kind == ShipKind.CARRIER else 0.0)
    raw_exposure = believed_damage * value

    if is_enemy_attack:
        secured = max(0.0, action_value(ship, target, game_state, ai_config))
        raw_exposure = max(0.0, raw_exposure - secured)

    return -raw_exposure


def _vision_term(
    ship: Ship,
    destination: AxialCoord,
    model: EnemyModel,
    ai_config: AiConfig,
) -> float:
    """A flat bonus for a carrier's own move (its materially wider vision
    radius means moving one is the likeliest single move to reveal more
    of the board this turn -- same reasoning `plan_movement`'s "carriers
    first" ordering already used), suppressed to 0 whenever `destination`
    carries meaningful believed-submarine mass (`AiConfig.move_score_sub_
    threat_radius`/`_threshold`) -- the user's "probe first with a cheap
    ASW-capable ship" case, expressed as a scoring suppression rather than
    a separate ordering rule: a risky hex simply isn't worth more to a
    carrier than to anything else once its own exposure term already
    accounts for the danger, so nothing else needs to independently favor
    sending the carrier there first."""
    if ship.kind != ShipKind.CARRIER:
        return 0.0
    sub_mass = model.mass_near(destination, ai_config.move_score_sub_threat_radius, frozenset({ShipKind.SUBMARINE}))
    if sub_mass >= ai_config.move_score_sub_threat_threshold:
        return 0.0
    return 1.0


def _rearguard_term(
    ship: Ship,
    destination: AxialCoord,
    game_state: GameState,
    force: TaskForce,
    goal_field: dict[AxialCoord, int] | None,
) -> float:
    """Small bonus for a destroyer/patrol boat ending behind the force's
    rearmost capital ship (by sea-route progress toward the goal) --
    graded version of `_rearguard_target`'s hard "trail behind" target:
    0 if `ship` isn't an escort kind, there's no living capital ship, or
    `destination` would actually be ahead of the rearmost one; otherwise
    higher the closer `destination` sits to that rearmost ship."""
    if ship.kind not in _ESCORT_KINDS or goal_field is None:
        return 0.0
    capital_ships = [
        game_state.ships[i]
        for i in force.member_ids
        if i in game_state.ships and game_state.ships[i].kind in _CAPITAL_KINDS
    ]
    if not capital_ships:
        return 0.0
    rearmost = max(capital_ships, key=lambda s: (sea_route_distance(goal_field, s.position), s.id))
    if sea_route_distance(goal_field, destination) < sea_route_distance(goal_field, rearmost.position):
        return 0.0  # would be ahead of the rearmost capital ship -- no bonus
    return -float(distance(destination, rearmost.position))


def score_move(
    ship: Ship,
    destination: AxialCoord,
    game_state: GameState,
    force: TaskForce,
    visible_enemies: dict[int, Ship],
    model: EnemyModel,
    goal_field: dict[AxialCoord, int] | None,
    ai_config: AiConfig,
) -> float:
    """`ship` ending its move at `destination` this turn, as one number --
    see this module's own docstring for the full list of terms and what
    each measures. `goal_field` is `sea_distance_field(game_state,
    force.goal.target)` -- computed once per force per outer-loop
    iteration by the caller (`plan_force_movement_scored`) and passed in
    rather than recomputed per candidate hex, since it doesn't change
    just because one ship moved elsewhere in the force; `None` for a
    force with no goal (nothing to advance toward, that term drops out)."""
    return (
        ai_config.move_score_goal_weight * _goal_progress_term(destination, goal_field)
        + ai_config.move_score_attack_weight * _attack_term(ship, destination, game_state, ai_config)
        + ai_config.move_score_screen_weight * _screen_term(ship, destination, game_state, force, visible_enemies)
        + ai_config.move_score_cohesion_weight * _cohesion_term(ship, destination, game_state, force, ai_config)
        + ai_config.move_score_exposure_weight * _exposure_term(ship, destination, game_state, model, ai_config)
        + ai_config.move_score_vision_weight * _vision_term(ship, destination, model, ai_config)
        + ai_config.move_score_rearguard_weight * _rearguard_term(ship, destination, game_state, force, goal_field)
    )


def _round_robin_order(ship_ids: list[int], game_state: GameState) -> list[int]:
    return sorted(
        ship_ids, key=lambda i: (_ROUND_ROBIN_PRIORITY.get(game_state.ships[i].kind, 4), i)
    )


def _plan_moves(
    game_state: GameState,
    force: TaskForce,
    plannable_ids: set[int],
    model: EnemyModel,
    ai_config: AiConfig,
    visible_enemies: dict[int, Ship],
    goal_field: dict[AxialCoord, int] | None,
) -> tuple[dict[int, AxialCoord], list[int]]:
    """Phase 1: builds a mutually-consistent plan (`{ship_id: destination}`)
    for every living member of `plannable_ids`, without physically moving
    anyone for real, plus the order those destinations were actually
    decided in -- the caller must execute in *this* order, not just
    `_round_robin_order` again, since a blocked ship can pull another one
    forward out of its normal turn (see the blocker-resolution paragraph
    below) -- a plan built on "the escort will have already stepped aside
    by the time I get here" is only consistent if that's also the order
    it's really executed in. See `plan_force_movement_scored`'s own
    docstring for the full motivation and `AiConfig.scored_task_force_movement_enabled`'s
    real-example writeup (game68, patrol_boat6).

    Ships take turns proposing **one hex step at a time** (round-robin,
    `_round_robin_order` -- capital ships first, patrol boats last), rather
    than each ship claiming its whole move in one shot: this is what lets
    a fast ship's own cohesion term see where the rest of the force
    actually intends to end up, not just their stale start-of-turn
    position, since by the time it's a fast/cheap ship's turn the heavier
    ships have typically already taken several of their own steps.

    Mutates `game_state.ships[i].position`/`.movement_remaining` for
    `i in plannable_ids` *directly* as the hypothetical planning state --
    safe because nothing else reads `game_state` mid-call (no `yield`
    happens here), and it's what lets every other ship's own `score_move`
    call see the latest hypothetical position via the normal `game_state.
    ships[...]` lookups `_cohesion_term`/`_screen_term`/etc. already use,
    with no separate shadow-state plumbing needed. Restored to each ship's
    real starting position/movement before returning -- this function only
    ever plans, the caller (`plan_force_movement_scored`) does the real
    moving in its own, separate execution phase.

    Each ship's turn, each round: classify its up to 6 neighbors via the
    same `_classify_step` `movement.reachable_hexes`/`shortest_path` use;
    `"open"`/`"enemy"` neighbors are real candidate next-hexes, scored with
    `score_move`. A ship only ever advances to a neighbor that strictly
    beats its *own current* hypothetical score -- so it naturally stops
    the moment nothing nearby helps (no separate "did we converge" check,
    and its final hypothetical position is always its own best by
    construction, since every step actually taken was an improvement over
    the one before it).

    If a ship's would-be best move is blocked by a `"passthrough"` (an
    unmoved, still-`plannable_ids` friendly ship occupying that neighbor),
    that blocker is forced one step toward the force's own goal (see
    `_step_aside`) rather than given its own full scoring-based turn --
    a blocker already scoring well by staying put would otherwise have no
    reason to ever move, indefinitely vetoing the ship behind it (see
    `_step_aside`'s own docstring for the real self-play failure this
    fixes). `_step_aside` only ever considers strictly-`"open"` neighbors
    of its own, never another `"passthrough"`, so this never recurses --
    no cycle-protection bookkeeping is needed for it (the `pending` set
    below exists only for `resolve` itself, which is no longer called
    recursively now that the blocker branch uses `_step_aside` instead).
    No separate fallback for a ship "boxed in"
    by terrain alone: `sea_distance_field` is a pure terrain BFS (no
    occupancy awareness), and by construction every hex with distance d>0
    has at least one neighbor with distance d-1 -- a downhill-toward-goal
    direction always exists purely from terrain, including the first hex
    of a detour around an island, so this greedy walk finds it without
    needing lookahead. The only thing that can make a ship's terrain-
    optimal neighbor unavailable is occupancy, which the blocker-resolution
    above (friendly) or the attack/exposure terms (enemy -- a real
    tactical decision, not an algorithmic gap) already cover."""
    member_ids = [i for i in plannable_ids if i in game_state.ships]
    if not member_ids:
        return {}

    start_position = {i: game_state.ships[i].position for i in member_ids}
    start_remaining = {i: game_state.ships[i].movement_remaining for i in member_ids}
    started_at_port = {
        i: (tile := game_state.board.get_tile(start_position[i])) is not None and tile.is_port
        for i in member_ids
    }
    first_step_taken = {i: False for i in member_ids}
    best_score: dict[int, float] = {}
    best_position: dict[int, AxialCoord] = {}
    # enemy ship id -> id of the plannable ship currently claiming it as an
    # attack target this call, whether via the seed below or a later step
    # in the walk -- prevents two of our own ships from both targeting the
    # same enemy (found via real play-testing: once the first one's real
    # attack kills or occupies it, the second's planned destination is
    # left pointing at a hex it can no longer legally reach at all).
    claimed_by: dict[int, int] = {}
    for i in _round_robin_order(member_ids, game_state):
        ship = game_state.ships[i]
        best_score[i] = score_move(ship, ship.position, game_state, force, visible_enemies, model, goal_field, ai_config)
        best_position[i] = ship.position
        # Seed with any attack reachable this turn's whole movement
        # budget allows -- unlike the goal term (which has a real
        # gradient toward the target via sea_distance_field, so the
        # one-hex-ahead walk below always finds it eventually), an
        # attack's value only appears once actually adjacent to the
        # target, giving the walk nothing to head toward from farther
        # away. Without this seed a ship can walk straight past a
        # securable kill it would have taken under the old whole-
        # destination scoring, simply because none of its immediate
        # neighbors look better than continuing toward its goal.
        #
        # Seeded in round-robin order, and a target already claimed by an
        # earlier-seeded ship this same call is skipped -- otherwise two
        # ships can independently both seed the same juicy target, and
        # once the first one's real attack kills (or occupies) it, the
        # second's planned destination is left pointing at a hex it can
        # no longer legally reach at all (found via real play-testing).
        for coord, _cost in scoring.reachable_attack_candidates(ship, game_state, visible_enemies):
            target = game_state.ship_at(coord)
            if target is not None and target.id in claimed_by:
                continue
            attack_score = score_move(ship, coord, game_state, force, visible_enemies, model, goal_field, ai_config)
            if attack_score > best_score[i]:
                best_score[i] = attack_score
                best_position[i] = coord
        claimed_target = game_state.ship_at(best_position[i])
        if claimed_target is not None and claimed_target.owner != ship.owner:
            claimed_by[claimed_target.id] = i

    done: set[int] = set()
    pending: list[int] = []
    # The order ships actually finished planning in. Now always equal to
    # _round_robin_order (a blocked capital ship's escort is nudged aside
    # via _step_aside, not given an out-of-order resolve() of its own --
    # see that function's docstring), but Phase 2 (plan_force_movement_
    # scored) still must execute in *this* order rather than assume that
    # equivalence, since that's what it's actually for: a plan built
    # assuming the escort has already stepped out of the way is only
    # consistent if it really does move first for real too (found via
    # real self-play: executing in fixed order instead left a later
    # ship's planned destination sitting on a hex an earlier-in-fixed-
    # order/later-in-actual-resolution ship hadn't vacated yet).
    resolution_order: list[int] = []

    def _step_aside(sid: int) -> bool:
        """Forces `sid` one step toward the force's own goal, bypassing
        `score_move` entirely, when it's blocking a higher-priority
        (earlier-round-robin) member that's already decided its own best
        move runs through `sid`'s current hex. The previous behaviour
        (`resolve(sid)` -- the blocker's own full scoring-based decision)
        let a blocker that was already scoring well right where it stood
        veto the ship behind it forever: if staying scored at least as
        well as any open neighbor for `sid` itself, it would never budge,
        and nothing forced it to -- found via self-play (seed 1,
        configs/dev_scored.json vs itself never reaching a winner in 1000
        turns; battle_log and ship counts both went flat around turn 200,
        capital ships included). Escorts exist to serve the formation's
        advance, not to independently veto it, so clearing the way takes
        priority over `sid`'s own preference. Returns whether `sid`
        actually moved (False if out of movement or boxed in itself --
        the caller falls back to today's "genuinely stuck" handling)."""
        ship = game_state.ships[sid]
        if ship.movement_remaining <= 0 or goal_field is None:
            return False
        leaving_port = started_at_port[sid] and not first_step_taken[sid]
        open_candidates = [
            n
            for n in neighbors(ship.position)
            if _classify_step(game_state, ship.owner, n, leaving_port, ship.movement_remaining) == "open"
        ]
        if not open_candidates:
            return False
        best_n = min(open_candidates, key=lambda n: (sea_route_distance(goal_field, n), n))
        ship.position = best_n
        ship.movement_remaining -= 1
        first_step_taken[sid] = True
        best_position[sid] = best_n
        # sid hasn't been through resolve() of its own yet (it's still
        # unresolved -- that's exactly why it was available to block in
        # the first place), so this also re-seeds its own running best
        # score from its new hypothetical position, the same way the
        # Phase 1 seed loop above originally did from its start position.
        best_score[sid] = score_move(ship, best_n, game_state, force, visible_enemies, model, goal_field, ai_config)
        return True

    def resolve(sid: int) -> None:
        if sid in done or sid in pending:
            return
        pending.append(sid)
        while True:
            ship = game_state.ships[sid]
            if ship.movement_remaining <= 0:
                break
            leaving_port = started_at_port[sid] and not first_step_taken[sid]
            open_candidates: list[AxialCoord] = []
            blocker_id: int | None = None
            for n in neighbors(ship.position):
                step = _classify_step(game_state, ship.owner, n, leaving_port, ship.movement_remaining)
                if step == "enemy":
                    occupant = game_state.ship_at(n)
                    if occupant is not None and claimed_by.get(occupant.id, sid) != sid:
                        continue  # already claimed by a different plannable ship this call
                    open_candidates.append(n)
                elif step == "open":
                    open_candidates.append(n)
                elif step == "passthrough" and blocker_id is None:
                    occupant = game_state.ship_at(n)
                    if occupant is not None and occupant.id in member_ids and occupant.id not in done:
                        blocker_id = occupant.id

            scored = [
                (n, score_move(ship, n, game_state, force, visible_enemies, model, goal_field, ai_config))
                for n in open_candidates
            ]
            best_n, best_s = max(scored, key=lambda p: (p[1], p[0])) if scored else (None, None)

            if best_n is not None and best_s > best_score[sid]:
                ship.position = best_n
                ship.movement_remaining -= 1
                first_step_taken[sid] = True
                best_score[sid] = best_s
                best_position[sid] = best_n
                new_target = game_state.ship_at(best_n)
                if new_target is not None and new_target.owner != ship.owner:
                    claimed_by[new_target.id] = sid
                continue

            if blocker_id is not None and blocker_id not in pending and _step_aside(blocker_id):
                continue  # retry now that the blocker has stepped aside

            break  # genuinely stuck: nothing improves, no resolvable blocker
        pending.pop()
        done.add(sid)
        resolution_order.append(sid)

    for sid in _round_robin_order(member_ids, game_state):
        resolve(sid)

    for i in member_ids:
        ship = game_state.ships[i]
        ship.position = start_position[i]
        ship.movement_remaining = start_remaining[i]

    return best_position, resolution_order


def plan_force_movement_scored(
    game_state: GameState,
    player: PlayerId,
    force: TaskForce,
    model: EnemyModel,
    ai_config: AiConfig,
    visible_enemies_fn: Callable[[], dict[int, Ship]],
    apply_move: Callable[[Ship, AxialCoord], None],
) -> Iterator[None]:
    """Moves every still-alive member of `force` this turn, in two phases
    -- see `_plan_moves`'s own docstring for the planning phase in detail,
    and `AiConfig.scored_task_force_movement_enabled` for the real example
    (game68) that motivated this two-phase design over the original
    single-phase "score full destinations, execute the single best one,
    refresh, repeat," which let a fast/cheap ship commit to a solo leap
    before the rest of the force had a chance to plan its own advance.

    Phase 1 (`_plan_moves`) builds a destination for every still-unresolved
    member without physically moving anyone. Phase 2 then actually
    executes that plan, one ship at a time, via `apply_move` -- in the
    *same order Phase 1 actually resolved them in* (not just `_round_
    robin_order` again -- a blocked ship can pull another one forward out
    of its normal turn during planning, see `_plan_moves`'s own
    docstring, and a plan built on "the escort will have already stepped
    aside by the time I get here" is only consistent if that's the real
    execution order too). After each real move, if it revealed an
    enemy that wasn't visible when this phase's plan was built, the
    remaining, not-yet-executed members get a fresh Phase 1 plan built
    from their real current positions and the updated visibility, rather
    than blindly continuing the stale one -- the outer `while unresolved`
    loop is what drives this replan-on-new-sighting behavior.

    `apply_move` is the caller's own closure -- submarine-toggle,
    `_execute`, and `reevaluate_strategy_on_new_sighting` handling all
    live in `tla.ai.policy` and are injected here exactly the way `tla.ai.
    tactics.secure_kills_pass` already injects `apply_attack`, for the
    identical circular-import reason (see this module's own docstring).

    Yields once per ship actually moved, same per-ship pacing every other
    pass in `plan_movement` already uses."""
    unresolved = {i for i in force.member_ids if i in game_state.ships}
    if not unresolved:
        return

    while unresolved:
        visible_enemies = visible_enemies_fn()
        goal_field = sea_distance_field(game_state, force.goal.target) if force.goal is not None else None
        plan, resolution_order = _plan_moves(
            game_state, force, set(unresolved), model, ai_config, visible_enemies, goal_field
        )
        if not plan:
            return  # nobody left who can move at all this turn

        # Execute in resolution_order where possible -- it's usually
        # already a legal order (ships mostly finish planning in roughly
        # the order they'd need to move in for real) -- but not always:
        # a ship recursively pulled forward to unblock another can finish
        # *before* the ship that unblocked it, even though that ship's own
        # destination only became free because a *third* ship vacated it
        # partway through -- an ordering relationship resolution_order
        # doesn't capture, since it only records when each ship's whole
        # walk finished, not the finer-grained sequence of who-vacated-
        # what-when inside it (found via real self-play). So this sweeps
        # resolution_order repeatedly, executing whichever planned ships
        # are *actually* reachable right now and deferring the rest to
        # the next sweep, until nothing more can progress -- self-
        # correcting for that regardless of exactly how it arose, rather
        # than trying to derive the perfect order from planning's own
        # internal recursion.
        pending_sids = list(resolution_order)
        replan = False
        while pending_sids and not replan:
            deferred: list[int] = []
            progressed = False
            for sid in pending_sids:
                if sid not in plan:
                    unresolved.discard(sid)
                    continue
                ship = game_state.ships[sid]
                destination = plan[sid]
                if destination != ship.position and destination not in movement.reachable_hexes(ship, game_state):
                    deferred.append(sid)  # not yet reachable -- still blocked by an unmoved ship
                    continue
                progressed = True
                unresolved.discard(sid)
                before_visible = set(visible_enemies_fn())
                apply_move(ship, destination)
                yield
                if sid not in game_state.ships:
                    continue  # sunk by return fire in its own attack -- nothing more to check
                if set(visible_enemies_fn()) - before_visible:
                    replan = True  # a new sighting -- replan everyone left from scratch, real positions
                    break
            if not progressed:
                # A genuine, unresolvable conflict (shouldn't normally
                # happen -- the reordering in _plan_moves is meant to
                # avoid exactly this) -- give up on whoever's left this
                # turn rather than crashing (or looping forever: they'd
                # otherwise stay in `unresolved`, and the outer `while
                # unresolved` loop would just plan the exact same stuck
                # situation again). They'll get a fresh plan, from
                # wherever they really are, next turn.
                for sid in deferred:
                    unresolved.discard(sid)
                break
            pending_sids = deferred
