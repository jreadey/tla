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


@dataclass
class Tile:
    coord: AxialCoord
    terrain: TerrainType
    is_port: bool = False
    port_owner: PlayerId | None = None

    @property
    def occupiable(self) -> bool:
        """Ships may normally only be on sea. A port is the one exception:
        it sits on land but ships can occupy it (see tla.movement, Phase 3,
        for the rule that leaving a port must go straight to an adjacent
        sea hex rather than another land hex)."""
        return self.terrain == TerrainType.SEA or self.is_port
