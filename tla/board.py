"""The game map: a dict of axial coordinate -> Tile plus convenience queries."""

from __future__ import annotations

from dataclasses import dataclass, field

from tla.elevation import ElevationGrid
from tla.hexgrid import AxialCoord, axial_to_offset
from tla.tile import PlayerId, Tile


@dataclass
class Board:
    width: int
    height: int
    hex_pixel_size: float = 18.0
    elevation: ElevationGrid | None = None
    tiles: dict[AxialCoord, Tile] = field(default_factory=dict)

    def get_tile(self, coord: AxialCoord) -> Tile | None:
        return self.tiles.get(coord)

    def in_bounds(self, coord: AxialCoord) -> bool:
        col, row = axial_to_offset(coord)
        return 0 <= col < self.width and 0 <= row < self.height

    def is_occupiable(self, coord: AxialCoord) -> bool:
        tile = self.get_tile(coord)
        return tile is not None and tile.occupiable

    def ports_for(self, player: PlayerId) -> list[AxialCoord]:
        """Ports `player` was permanently assigned at map generation --
        fixed for the whole game. Used for starting-fleet placement and the
        port-siege win condition, which is about holding an opponent's
        original home ports specifically. For "which ports can `player`
        currently build at", see `controlled_ports_for` instead."""
        return [
            coord
            for coord, tile in self.tiles.items()
            if tile.is_port and tile.port_owner == player
        ]

    def controlled_ports_for(self, player: PlayerId) -> list[AxialCoord]:
        """Ports currently *displaying* as `player`'s -- their own
        never-flipped ports, plus any of the opponent's ports `player` has
        captured (see `Tile.port_display_owner`). This is what production
        uses: capturing an enemy port lets you build from it."""
        return [
            coord
            for coord, tile in self.tiles.items()
            if tile.is_port and tile.port_display_owner == player
        ]
