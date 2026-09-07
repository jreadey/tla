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
    # Which side the port currently *displays* as held by -- distinct from
    # the permanent `port_owner`. It only changes when a ship belonging to
    # someone other than the current controller occupies the port, and
    # stays put (doesn't revert) once that ship leaves again; see
    # tla.production.handle_port_capture. None (the common case: never yet
    # occupied by anyone) means "same as port_owner" -- see
    # tla.tile.Tile.port_display_owner.
    port_controller: PlayerId | None = None

    @property
    def port_display_owner(self) -> PlayerId | None:
        """The side a port should currently be rendered as belonging to:
        `port_controller` once it's been set by an occupation, else
        `port_owner`. None for a non-port tile."""
        if not self.is_port:
            return None
        return self.port_controller if self.port_controller is not None else self.port_owner

    @property
    def occupiable(self) -> bool:
        """Ships may normally only be on sea. A port is the one exception:
        it sits on land but ships can occupy it (see tla.movement, Phase 3,
        for the rule that leaving a port must go straight to an adjacent
        sea hex rather than another land hex)."""
        return self.terrain == TerrainType.SEA or self.is_port
