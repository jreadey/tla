"""Terrain and tile data model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tla.hexgrid import AxialCoord

# A player is identified by a plain int for now (exactly two players: 1 and 2).
PlayerId = int
PLAYER_A: PlayerId = 1
PLAYER_B: PlayerId = 2


class TerrainType(Enum):
    LAND = "land"
    SEA = "sea"
    SHORE = "shore"


@dataclass
class Tile:
    coord: AxialCoord
    terrain: TerrainType
    is_port: bool = False
    port_owner: PlayerId | None = None

    @property
    def occupiable(self) -> bool:
        return self.terrain in (TerrainType.SEA, TerrainType.SHORE)
