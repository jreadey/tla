"""Ship kinds and their stat sheet.

The mutable `Ship` instance class (position, current HP, submerged state,
etc.) is added in Phase 2 once movement/fleet placement exist. This module
starts with just the static data needed by `tla.config`'s defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
