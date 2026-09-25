"""All tunable game parameters, in one place.

Everything here has an in-code default matching the game design spec, and can
be partially overridden by a JSON file via `Config.load(path)` -- e.g. a small
dev map + tiny fleet for fast iteration (see configs/dev.json).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from tla.ship import ShipKind, ShipStats
from tla.tile import PLAYER_A, PLAYER_B, PlayerId

DEFAULT_SHIP_STATS: dict[ShipKind, ShipStats] = {
    ShipKind.BATTLESHIP: ShipStats(movement=4, hp=12, damage=4, asw=1, cost=10),
    ShipKind.CARRIER: ShipStats(movement=4, hp=7, damage=2, asw=1, cost=10),
    ShipKind.CRUISER: ShipStats(movement=4, hp=8, damage=3, asw=2, cost=7),
    ShipKind.DESTROYER: ShipStats(movement=3, hp=6, damage=2, asw=3, cost=4),
    ShipKind.SUBMARINE: ShipStats(
        movement=3, movement_submerged=1, hp=4, damage=4, asw=1, cost=4
    ),
    ShipKind.PATROL_BOAT: ShipStats(movement=6, hp=2, damage=1, asw=2, cost=1),
}

DEFAULT_FLEET: dict[ShipKind, int] = {
    ShipKind.BATTLESHIP: 2,
    ShipKind.CARRIER: 2,
    ShipKind.CRUISER: 4,
    ShipKind.DESTROYER: 8,
    ShipKind.SUBMARINE: 8,
    ShipKind.PATROL_BOAT: 8,
}


@dataclass
class MapConfig:
    # Sized to fit on-screen without panning at the default hex_pixel_size on
    # most laptop screens. For a bigger map that requires panning to see the
    # whole thing, see configs/large.json.
    width: int = 50
    height: int = 25
    seed: int | None = None
    noise_scale: float = 7.5
    octaves: int = 4
    # Perlin noise (roughly -1..1) minus sea_level defines a continuous
    # elevation field; elevation > 0 is land, <= 0 is sea -- see tla.elevation.
    sea_level: float = 0.15
    # A hex is LAND if more than this fraction of its elevation-raster
    # samples are above sea level; otherwise it's SEA. There is no separate
    # "shore" terrain -- coastal land hexes (bordering a sea hex) are simply
    # where ports may be placed, see tla.mapgen._place_ports.
    land_area_threshold: float = 0.10
    # Elevation raster resolution, as a multiple of hex_pixel_size -- higher
    # means finer-grained shore detection and a smoother drawn coastline.
    elevation_supersample: int = 4
    # Pixel scale shared by map generation (elevation raster spacing) and
    # rendering (hex drawing size), so both stay in sync.
    hex_pixel_size: float = 18.0
    # A generated map is rejected (see tla.mapgen.generate_map) unless
    # every sea hex is within `max_route_distance_fraction * max(width,
    # height)` real sea-route hexes of one of the (up to two) shortest
    # routes between the two sides' ports -- see tla.mapgen.has_sea_route_
    # coverage. Without this, a map can pass every other playability
    # condition while still burying a large stretch of open ocean nowhere
    # near anywhere the two sides could plausibly ever meet -- user's own
    # diagnosis of a real generated map. A fraction of map size rather than
    # a flat hex count so it scales with the map instead of needing its
    # own per-config retuning -- empirically, the typical (median, across
    # several seeds) real gap on a random candidate map sits at roughly
    # 0.5-0.55x the longer map dimension on both this module's small
    # default map and configs/large.json's much bigger one; 0.6 sits just
    # above that median on both, so it still meaningfully rejects the
    # worst outlier candidates without making an already-rare acceptable
    # candidate (see has_fully_connected_sea) rarer still by compounding
    # with an overly strict threshold.
    max_route_distance_fraction: float = 0.6
    # How many candidate maps generate_map will try (each with a different
    # derived seed) before giving up and raising -- see PortConfig.
    # min_port_sea_neighbors and tla.mapgen.has_wide_enough_sea_passage,
    # two of the several conditions a candidate map/port-placement must
    # all pass -- see tla.mapgen._map_is_playable. 100, not a smaller round
    # number, because has_fully_connected_sea (requiring literally zero
    # stray disconnected ponds anywhere, not just away from a port) can
    # take several dozen attempts to satisfy by chance even at this
    # module's own default map size -- an empirical sweep of 30 seeds at
    # that size found one needing 66 attempts and none needing more.
    max_generation_attempts: int = 100


@dataclass
class PortConfig:
    ports_per_player: int = 4
    min_port_spacing: int = 4
    # A generated map is rejected (see tla.mapgen.generate_map) unless
    # every port has at least this many sea-hex neighbors -- fewer would
    # let a single blockading ship pin a port shut right at its own
    # doorstep, with no other way in or out at all.
    min_port_sea_neighbors: int = 2


@dataclass
class ShipStatsConfig:
    stats: dict[ShipKind, ShipStats] = field(
        default_factory=lambda: dict(DEFAULT_SHIP_STATS)
    )


@dataclass
class FleetConfig:
    counts: dict[ShipKind, int] = field(default_factory=lambda: dict(DEFAULT_FLEET))


@dataclass
class ProductionConfig:
    # Added to *each* controlled, unoccupied port every turn -- not a
    # shared budget split across ports, so a player's total production
    # scales with how many ports they control.
    points_per_turn: int = 5
    # Every port -- either player's, human or AI -- automatically builds
    # from this same fixed sequence, repeating once it reaches the end;
    # there is no player choice of what to build (see tla.production).
    # A captured port's progress through the sequence resets to the start
    # for its new controller (see tla.production.handle_port_capture).
    build_order: list[ShipKind] = field(
        default_factory=lambda: [
            ShipKind.BATTLESHIP,
            ShipKind.CARRIER,
            ShipKind.CRUISER,
            ShipKind.DESTROYER,
            ShipKind.SUBMARINE,
            ShipKind.PATROL_BOAT,
        ]
    )


@dataclass
class CombatConfig:
    ac_bonus_radius: int = 2
    ac_bonus_amount: int = 1


@dataclass
class FowConfig:
    """Fog of war: when enabled, a player can't see the enemy's ships
    unless they're within vision -- see tla.fow.visible_hexes_for."""

    enabled: bool = True
    # Every hex within this many steps of one of the player's own ships.
    ship_visibility_radius: int = 1
    # Every hex within this many steps of a port the player controls, or
    # one of their own aircraft carriers -- both see further than a
    # regular ship does.
    port_and_carrier_visibility_radius: int = 4


@dataclass
class AiConfig:
    """Tuning knobs for tla.ai.NaivePolicy -- see that module for how each
    is used."""

    # Retreat from an otherwise-winning battle once the attacker's own HP
    # fraction drops below this, to avoid a follow-up ambush.
    damaged_withdraw_fraction: float = 0.34
    # A carrier retreats toward its escorts once a visible enemy battleship,
    # cruiser, or submarine -- the kinds that can actually hurt a carrier in
    # a fight, per tla.ai.policy._DANGEROUS_TO_CARRIER_KINDS -- could reach
    # (and so attack) its hex on that enemy's own next turn; a destroyer or
    # patrol boat alone doesn't trigger it. See tla.ai.policy.
    # _enemy_reachable_next_turn for that reachability check -- a real one
    # (terrain/other ships accounted for, each kind's own actual movement
    # stat), not a flat hex-distance radius (an earlier version of this
    # used one; abandoned after self-play showed no single radius working
    # well across map/ship-movement scales -- too tight and it doesn't
    # notice real threats, too loose and it freezes the carrier's advance
    # entirely). Modeled on a human player's own reported tactic: advance
    # a couple hexes at a time, and don't stop somewhere a dangerous enemy
    # could reach and hit you on its own turn.
    #
    # How many of its own movement points a carrier holds back each turn
    # when advancing toward a task-force goal, rather than spending its
    # full budget -- see tla.ai.policy._choose_carrier_destination. Same
    # human tactic as above, described precisely: advance one hex, and if
    # nothing dangerous has come into view, advance one more next turn --
    # never committing more than a single hex per turn while blind, so a
    # threat sitting just past the carrier's own vision (see FowConfig.
    # port_and_carrier_visibility_radius) gets one more chance to be
    # spotted before the carrier closes the rest of the gap itself. 3
    # (leaving 1 of the carrier's 4 movement to actually spend) is that one
    # hex; a real game (see the project's own carrier-formation memory)
    # showed a 2-hex version of this same reserve let a carrier jump clean
    # over a threat sitting one hex past its own vision, straight into
    # that threat's own reach, without either side ever having seen the
    # other first.
    carrier_advance_reserve: int = 3
    # Seconds paced between each AI ship's move, so a human opponent can
    # watch an AI turn unfold instead of it resolving instantly.
    turn_pacing_seconds: float = 0.4
    # Task forces (see tla.ai.task_force) -- an AI-internal grouping of
    # ships pursuing one shared strategic goal, never visible outside the
    # AI itself.
    # Max hex distance a ship can be pulled into a forming force.
    task_force_gather_radius: int = 3
    # Cap on a force's size at the instant it *first forms* (anchor
    # included) -- not how big it can ever grow. A force seeded this small
    # then grows toward task_force_target_max_size afterward, turn by
    # turn, via recruit_into_open_forces -- deliberately kept separate
    # (rather than just raising this instead) so a force never forms
    # already-huge in one shot, only builds up to that size gradually.
    task_force_max_size: int = 4
    # Below this many members after casualties, a force dissolves; also
    # the minimum size to bother forming a non-carrier-anchored force.
    task_force_min_size: int = 2
    task_force_allow_non_carrier_forces: bool = True
    # If set, a non-submarine force member advancing toward a non-RETREAT
    # goal (tla.ai.policy.choose_task_force_destination) won't let its own
    # sea-route progress toward that goal get more than this many hexes
    # ahead of the force's own straggler (its slowest-progressing living
    # non-submarine member) -- it redirects toward the straggler instead
    # of continuing to advance. None (default) leaves cohesion purely
    # emergent from the shared per-turn pace cap (_compute_force_pace),
    # which bounds *speed* but not the position drift that different
    # members' independent path choices (favorable attacks taken, escort
    # rerouting, etc.) can still accumulate turn over turn.
    task_force_max_separation: int | None = None
    # Turns with no strict improvement in distance-to-goal before a force
    # gives up on its current goal and gets reassigned a new one -- the
    # fix for a ship/force camping forever next to a fight it can't win.
    task_force_stall_turns: int = 8
    # Consecutive turns a force can lose a member while also making no
    # progress toward its goal (see is_attritting) before that alone
    # triggers a retreat, same as is_outnumbered would -- an enemy
    # reinforcing piecemeal can keep a fight looking "close" turn after
    # turn without ever presenting one clearly-superior force at a single
    # instant, so a force can bleed to death in a stalled fight that
    # is_outnumbered's own instantaneous group-power check never flags.
    # (Lowering task_force_stall_turns instead -- reassign sooner rather
    # than retreat sooner -- was tried first and reverted: self-play
    # showed forces abandoning goals before they had a real chance to
    # succeed, stalling otherwise-winnable games. This is the version of
    # the fix that held up under the same 20-seed sweep.)
    task_force_attrition_turns: int = 3
    # Radius (from any force member) within which enemies count as
    # "nearby" for the is_outnumbered check.
    task_force_threat_radius: int = 4
    # How much clearer the disadvantage must be than a bare tie before a
    # force retreats -- 0 means retreat the instant the race would go
    # against it. Also the margin used to decide when a *retreating* force
    # has recruited enough to safely resume -- see task_force_max_retreat_
    # turns. A margin of 0 was originally found to be *far* too
    # trigger-happy on its own: self-play showed otherwise-winnable games
    # never concluding, because a retreating force could get stuck unable
    # to clear even a zero margin against its frozen threat snapshot,
    # having also retreated too readily in the first place. That regression
    # is specific to margin 0 with no other way to stop retreating --
    # retesting after adding retreat_cancel_port_distance (an independent,
    # non-strength way to end a retreat) showed the same margin of 0 winning
    # a head-to-head tournament ~62% of the time against the old default (4)
    # with zero non-convergent games across a 20-seed sweep, since a force
    # is no longer solely dependent on clearing this margin to ever resume.
    task_force_outnumbered_margin: int = 0
    # A retreating force keeps retreating toward its rally port -- absorbing
    # nearby unassigned ships via recruit_into_open_forces the same as any
    # other open force -- until its own current strength clearly outmatches
    # (by task_force_outnumbered_margin) whatever it retreated from, not for
    # a fixed number of turns; home isn't safety in itself, it's just where
    # new production shows up to actually change the balance. This caps how
    # long ONE continuous retreat is allowed to chase that condition before
    # giving up instead -- reassigning the goal exactly like a stall does --
    # in case the opponent is reinforcing just as fast and the force would
    # otherwise never catch up. Generous relative to task_force_stall_turns
    # (8) and task_force_attrition_turns (3) since the whole point here is
    # giving real production time to matter.
    task_force_max_retreat_turns: int = 15
    # A retreating force also stops retreating -- regardless of whether it
    # has actually out-grown what it retreated from -- once it comes within
    # this many hexes (real sea route, not straight-line) of one of its own
    # controlled ports. Rationale: retreating helps an attacker advance
    # unopposed more than it helps the defender, since the defender bleeds
    # ships either way and a fight near home still slows the attacker down
    # while staying close to where new production actually arrives -- so
    # once a force is already that close, standing and fighting is worth
    # more than continuing to retreat purely because it hasn't cleared
    # task_force_outnumbered_margin yet. Confirmed via a self-play
    # head-to-head tournament (30 seeds, both sides swapped): every
    # threshold from 1 to 100 hexes beat "disabled" ~60-63% of the time,
    # with results essentially flat across that whole range -- the value
    # itself barely matters (the real fix is having *any* non-strength way
    # to stop retreating), so 8 was picked on the original strategic
    # reasoning above, not because the data preferred it specifically. See
    # project_retreat_cancel_near_port memory for the full experiment.
    # None disables this, leaving the strength check as the only way to
    # stop retreating (the pre-tournament behavior).
    retreat_cancel_port_distance: int | None = 8
    # Whether resuming via retreat_cancel_port_distance (as opposed to
    # actually out-growing the threat) also resets turns_since_progress/
    # best_progress_distance for the force's goal, same as a strength-based
    # resume always does. False (default) keeps the prior progress tracking
    # intact across a distance-based resume instead -- found necessary
    # after a real game showed a force cycling trigger-retreat-resume-
    # repeat near its own territory, each fast, cheap resume erasing
    # hard-won progress toward a distant goal and trapping the whole war
    # near one side of the map. A head-to-head self-play tournament (30
    # seeds, both sides swapped, both a 14x10 and a 20x12 map) came back an
    # even 48-50% against the old (True) behavior either way -- not a
    # measured improvement, but no measured cost either, and it directly
    # fixes a diagnosed, reproducible failure mode the aggregate win-rate
    # metric is apparently too coarse to reward or punish. See
    # project_retreat_cancel_near_port memory for the full experiment,
    # including a second candidate fix (capping how far a retreat travels)
    # that was tried and NOT adopted -- it measurably regressed in one of
    # the two tournament configurations.
    reset_progress_on_distance_resume: bool = False
    # If set, a retreating force's movement target is capped to this many
    # hexes along the real sea route toward its rally port (see
    # rally_point/_pullback_waypoint), not the port itself -- so a force
    # already reasonably safe doesn't have to fully retreat before the
    # resume checks above can apply. None (default) retreats all the way
    # to the rally port, as before.
    retreat_pullback_hexes: int | None = None
    # Below this many members, a force is "open" and actively recruits
    # nearby unassigned ships (see tla.ai.task_force.recruit_into_open_
    # forces) instead of them always spinning up a new, separate force.
    # Distinct from task_force_min_size (the dissolution floor) -- a force
    # can be open without being anywhere near dissolving.
    task_force_target_min_size: int = 12
    # Recruitment stops once a force reaches this many members -- the real
    # ceiling on how big a force ever gets, as opposed to task_force_max_
    # size (only the cap at the moment a force first forms).
    task_force_target_max_size: int = 20
    # How far a stray/new ship can be from an open force's centroid and
    # still be pulled into it. Deliberately larger than
    # task_force_gather_radius (used only at the instant a force first
    # forms, among already-co-located ships) -- a reinforcement (e.g. a
    # newly produced ship spawning at a home port) needs to reach a force
    # that may already be deployed far away.
    task_force_recruit_radius: int = 10
    # Port-defense doctrine (see tla.ai.task_force.compute_port_defense_
    # directives), specified directly by the user from their own playtest
    # strategy: a controlled port is "threatened" once a visible enemy is
    # within this many sea hexes of it.
    port_defense_trigger_radius: int = 4
    # Only a player's own ships within this many sea hexes of a threatened
    # port are considered as candidate responders -- anything farther is
    # judged too far to arrive in time and is left doing whatever it's
    # already doing rather than recalled.
    port_defense_response_radius: int = 8
    # How much stronger the threat must be (see tla.ai.task_force.
    # outmatched) before responders fall back to a single delaying
    # blocker instead of counterattacking as a group. 0 -- the default,
    # matching the user's own stated doctrine -- means "counterattack as
    # long as we're at least as strong," not strictly stronger.
    port_defense_margin: int = 0
    # Carrier-defense doctrine (see tla.ai.task_force.compute_carrier_
    # defense_directives), specified directly by the user after a real
    # game showed the naive AI declining to engage a dangerous enemy
    # converging on one of its own carriers, purely because the fight
    # itself was an exact tie -- a mutual kill is still far better than
    # losing an unescorted carrier for nothing next turn. No separate
    # trigger radius here (unlike port defense's own): a carrier is
    # "threatened" exactly when `tla.ai.task_force.enemy_reachable_next_
    # turn` says a dangerous-kind enemy could reach its hex, the same
    # real reachability check its own individual retreat already uses.
    # Only a player's own ships within this many sea hexes of a
    # threatened carrier are considered as candidate responders --
    # anything farther is judged too far to arrive in time.
    carrier_defense_response_radius: int = 8
    # Same meaning as port_defense_margin, for carrier defense: 0 means
    # "counterattack as long as we're at least as strong as the threat,"
    # not strictly stronger.
    carrier_defense_margin: int = 0
    # tla.ai.enemy_model.EnemyModel: a tracked enemy ship unseen for more
    # than this many turns is folded back into its kind's pool (its
    # individual position belief merged into the shared per-kind field,
    # its own id-level tracking dropped) rather than keeping an
    # ever-more-diffuse per-ship field alive forever. Bounds how many live
    # per-ship fields a long game accumulates; purely a memory/performance
    # knob, not a fairness one -- a folded-back ship's belief mass is
    # preserved, just merged into a coarser bucket.
    enemy_model_stale_turns: int = 10
    # tla.ai.global_strategy.compute_posture: how much clearer an edge (in
    # the same "rounds to kill" units tla.ai.task_force.outmatched already
    # uses) than the default-0 local-tactical margins before the *global*
    # force-balance assessment calls a player aggressive/defensive rather
    # than neutral -- a global posture shift shouldn't flip on the same
    # hair-trigger a single local fight does.
    posture_margin: int = 2
    # tla.ai.global_strategy.posture_adjusted_ai_config: how far
    # task_force_outnumbered_margin/port_defense_margin/
    # carrier_defense_margin move under a non-neutral posture (aggressive
    # raises them, defensive lowers them, by this same amount for all
    # three -- see that function's own docstring for why one shared delta
    # applies uniformly). Deliberately modest to start.
    posture_margin_shift: int = 1


@dataclass
class Config:
    map: MapConfig = field(default_factory=MapConfig)
    ports: PortConfig = field(default_factory=PortConfig)
    ship_stats: ShipStatsConfig = field(default_factory=ShipStatsConfig)
    fleet: FleetConfig = field(default_factory=FleetConfig)
    production: ProductionConfig = field(default_factory=ProductionConfig)
    combat: CombatConfig = field(default_factory=CombatConfig)
    fow: FowConfig = field(default_factory=FowConfig)
    ai: AiConfig = field(default_factory=AiConfig)
    # Which side each player is -- "human" (default) or "ai". Session/launch
    # configuration (who's driving each seat), not persisted game state --
    # see main.py's --ai flag and tla.rendering.game_view's use of this to
    # decide when to drive tla.ai.NaivePolicy instead of waiting on input.
    player_kinds: dict[PlayerId, str] = field(
        default_factory=lambda: {PLAYER_A: "human", PLAYER_B: "human"}
    )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        base = cls()
        if path is None:
            return base
        data = json.loads(Path(path).read_text())
        return _apply_overrides(base, data)


def _apply_overrides(base: Config, data: dict) -> Config:
    map_cfg = replace(base.map, **data.get("map", {}))
    ports_cfg = replace(base.ports, **data.get("ports", {}))

    production_overrides = dict(data.get("production", {}))
    if "build_order" in production_overrides:
        production_overrides["build_order"] = [
            ShipKind(name) for name in production_overrides["build_order"]
        ]
    production_cfg = replace(base.production, **production_overrides)

    combat_cfg = replace(base.combat, **data.get("combat", {}))
    fow_cfg = replace(base.fow, **data.get("fow", {}))
    ai_cfg = replace(base.ai, **data.get("ai", {}))

    player_kinds = dict(base.player_kinds)
    for name, kind in data.get("player_kinds", {}).items():
        player_kinds[{"a": PLAYER_A, "b": PLAYER_B}[name.lower()]] = kind

    fleet_counts = dict(base.fleet.counts)
    for name, count in data.get("fleet", {}).items():
        fleet_counts[ShipKind(name)] = count
    fleet_cfg = FleetConfig(counts=fleet_counts)

    ship_stats = dict(base.ship_stats.stats)
    for name, overrides in data.get("ship_stats", {}).items():
        kind = ShipKind(name)
        ship_stats[kind] = replace(ship_stats[kind], **overrides)
    ship_stats_cfg = ShipStatsConfig(stats=ship_stats)

    return Config(
        map=map_cfg,
        ports=ports_cfg,
        ship_stats=ship_stats_cfg,
        fleet=fleet_cfg,
        production=production_cfg,
        combat=combat_cfg,
        fow=fow_cfg,
        ai=ai_cfg,
        player_kinds=player_kinds,
    )
