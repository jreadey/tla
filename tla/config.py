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


@dataclass
class PortConfig:
    ports_per_player: int = 4
    min_port_spacing: int = 4


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
    # How many orders to keep queued at each controlled port at once.
    queue_depth: int = 1
    # Cycled through (by turn number) to pick what a port's next order is,
    # whenever its queue has room -- repetition IS the weighting, so
    # cheaper/faster ships appear more often than expensive ones.
    production_order: list[ShipKind] = field(
        default_factory=lambda: [
            ShipKind.PATROL_BOAT,
            ShipKind.DESTROYER,
            ShipKind.SUBMARINE,
            ShipKind.CRUISER,
            ShipKind.PATROL_BOAT,
            ShipKind.DESTROYER,
            ShipKind.BATTLESHIP,
            ShipKind.CARRIER,
        ]
    )
    # A carrier retreats toward its escorts once a visible enemy comes
    # within this many hexes of it.
    carrier_threat_radius: int = 3
    # A battleship/carrier's chosen destination is only worth scouting
    # ahead of (see tla.ai.policy._scout_prepass) if it's within this many
    # hexes of a currently visible enemy -- otherwise the whole ocean would
    # need "clearing" the instant any enemy is spotted anywhere, even far
    # from where a capital ship is actually headed.
    scout_trigger_radius: int = 4
    # Seconds paced between each AI ship's move, so a human opponent can
    # watch an AI turn unfold instead of it resolving instantly.
    turn_pacing_seconds: float = 0.4
    # Task forces (see tla.ai.task_force) -- an AI-internal grouping of
    # ships pursuing one shared strategic goal, never visible outside the
    # AI itself.
    # Max hex distance a ship can be pulled into a forming force.
    task_force_gather_radius: int = 3
    # Max members per force, anchor included.
    task_force_max_size: int = 4
    # Below this many members after casualties, a force dissolves; also
    # the minimum size to bother forming a non-carrier-anchored force.
    task_force_min_size: int = 2
    task_force_allow_non_carrier_forces: bool = True
    # Turns with no strict improvement in distance-to-goal before a force
    # gives up on its current goal and gets reassigned a new one -- the
    # fix for a ship/force camping forever next to a fight it can't win.
    task_force_stall_turns: int = 8
    # Radius (from any force member) within which enemies count as
    # "nearby" for the is_outnumbered check.
    task_force_threat_radius: int = 4
    # Minimum turns a force spends retreating once outnumbered triggers,
    # before re-checking whether it's safe to resume its goal.
    task_force_retreat_turns: int = 4
    # How much clearer the disadvantage must be than a bare tie before a
    # force retreats -- 0 means retreat the instant the race would go
    # against it, which in practice is *far* too trigger-happy (confirmed
    # via self-play: a margin of 0 caused otherwise-winnable games to
    # never conclude within a generous turn cap, by making forces retreat
    # from marginal, often-recoverable disadvantages instead of pressing
    # small, real advantages elsewhere). A higher margin requires a more
    # decisive, unambiguous mismatch before backing off.
    task_force_outnumbered_margin: int = 4
    # Retreats for the same goal before it's treated as stalled (reassigned)
    # instead of retreating yet again -- guards against a retreat/resume
    # loop that never accumulates enough consecutive stalled turns to trip
    # task_force_stall_turns on its own.
    task_force_max_retreats: int = 3
    # Below this many members, a force is "open" and actively recruits
    # nearby unassigned ships (see tla.ai.task_force.recruit_into_open_
    # forces) instead of them always spinning up a new, separate force.
    # Distinct from task_force_min_size (the dissolution floor) -- a force
    # can be open without being anywhere near dissolving.
    task_force_target_min_size: int = 6
    # Recruitment stops once a force reaches this many members.
    task_force_target_max_size: int = 12
    # How far a stray/new ship can be from an open force's centroid and
    # still be pulled into it. Deliberately larger than
    # task_force_gather_radius (used only at the instant a force first
    # forms, among already-co-located ships) -- a reinforcement (e.g. a
    # newly produced ship spawning at a home port) needs to reach a force
    # that may already be deployed far away.
    task_force_recruit_radius: int = 10


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
    production_cfg = replace(base.production, **data.get("production", {}))
    combat_cfg = replace(base.combat, **data.get("combat", {}))
    fow_cfg = replace(base.fow, **data.get("fow", {}))

    ai_overrides = dict(data.get("ai", {}))
    if "production_order" in ai_overrides:
        ai_overrides["production_order"] = [
            ShipKind(name) for name in ai_overrides["production_order"]
        ]
    ai_cfg = replace(base.ai, **ai_overrides)

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
