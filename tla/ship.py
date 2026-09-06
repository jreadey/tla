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
    can only be damaged by `aws`.
    """

    movement: int
    hp: int
    damage: int
    aws: int
    cost: int
    movement_submerged: int | None = None


@dataclass
class Ship:
    """One ship on the board. `surfaced` is only meaningful for submarines --
    every other kind stays `True` and is never toggled (see tla.movement,
    Phase 3)."""

    id: int
    kind: ShipKind
    owner: PlayerId
    position: AxialCoord
    current_hp: int
    surfaced: bool = True
