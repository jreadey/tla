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

DEFAULT_SHIP_STATS: dict[ShipKind, ShipStats] = {
    ShipKind.BATTLESHIP: ShipStats(movement=4, hp=12, damage=4, aws=0, cost=10),
    ShipKind.CARRIER: ShipStats(movement=4, hp=7, damage=2, aws=0, cost=10),
    ShipKind.CRUISER: ShipStats(movement=4, hp=8, damage=4, aws=2, cost=7),
    ShipKind.DESTROYER: ShipStats(movement=4, hp=6, damage=2, aws=2, cost=4),
    ShipKind.SUBMARINE: ShipStats(
        movement=3, movement_submerged=1, hp=4, damage=4, aws=0, cost=4
    ),
    ShipKind.PATROL_BOAT: ShipStats(movement=6, hp=2, damage=1, aws=1, cost=1),
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
    width: int = 80
    height: int = 40
    seed: int | None = None
    noise_scale: float = 12.0
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
    points_per_turn: int = 20


@dataclass
class CombatConfig:
    ac_bonus_radius: int = 1
    ac_bonus_amount: int = 1


@dataclass
class Config:
    map: MapConfig = field(default_factory=MapConfig)
    ports: PortConfig = field(default_factory=PortConfig)
    ship_stats: ShipStatsConfig = field(default_factory=ShipStatsConfig)
    fleet: FleetConfig = field(default_factory=FleetConfig)
    production: ProductionConfig = field(default_factory=ProductionConfig)
    combat: CombatConfig = field(default_factory=CombatConfig)

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
    )
