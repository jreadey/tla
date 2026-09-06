"""Ship kinds, their stat sheet, and the mutable per-instance Ship state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tla.hexgrid import AxialCoord
from tla.tile import PlayerId


class ShipKind(Enum):
    BATTLESHIP = "battleship"
    CARRIER = "carrier"
    CRUISER = "cruiser"
    DESTROYER = "destroyer"
    SUBMARINE = "submarine"
    PATROL_BOAT = "patrol_boat"


@dataclass(frozen=True)
class ShipStats:
    """Stat sheet for one ship kind.

    `movement` is the normal move budget; for submarines it is the surfaced
    budget, and `movement_submerged` gives the (lower) submerged budget.
    `damage` applies to any enemy ship except a submerged submarine, which
    can only be damaged by `asw` (anti-submarine warfare).
    """

    movement: int
    hp: int
    damage: int
    asw: int
    cost: int
    movement_submerged: int | None = None


@dataclass
class Ship:
    """One ship on the board. `surfaced` is only meaningful for submarines --
    every other kind stays `True` and is never toggled.

    A submarine may toggle surfaced/submerged at most twice per turn (once
    before it moves, once after); `toggled_pre_move`/`toggled_post_move`
    track whether each of those two opportunities has been used yet this
    turn (see tla.movement.toggle_submarine_state).
    """

    id: int
    kind: ShipKind
    owner: PlayerId
    position: AxialCoord
    current_hp: int
    surfaced: bool = True
    movement_remaining: int = 0
    toggled_pre_move: bool = False
    toggled_post_move: bool = False

    def max_movement(self, stats: ShipStats) -> int:
        """`stats` must be this ship's own ShipStats. Submerged submarines
        use the lower movement_submerged budget."""
        if self.kind == ShipKind.SUBMARINE and not self.surfaced:
            return stats.movement_submerged if stats.movement_submerged is not None else stats.movement
        return stats.movement
