"""The game map: a dict of axial coordinate -> Tile plus convenience queries."""

from __future__ import annotations

from dataclasses import dataclass, field

from tla.hexgrid import AxialCoord, axial_to_offset
from tla.tile import PlayerId, Tile


@dataclass
class Board:
    width: int
    height: int
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
        return [
            coord
            for coord, tile in self.tiles.items()
            if tile.is_port and tile.port_owner == player
        ]
