"""Battle resolution: what happens when a ship moves into an enemy-occupied
hex.

Damage is simultaneous each round: both ships compute and apply damage the
same round, based on each ship's `damage` stat -- or `asw` if the target is
a submerged submarine, which `damage` cannot touch at all. Each of a side's
own aircraft carriers within `CombatConfig.ac_bonus_radius` of the battle
hex adds `CombatConfig.ac_bonus_amount` to that side's attack, recomputed
fresh every round (a carrier arriving or sinking mid-battle changes it).
After a round where both ships survive, the attacker (the ship that moved
into the hex) chooses to stay for another round or retreat -- this is the
`decision_fn` seam, filled by a human UI prompt or an AI policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from tla.config import CombatConfig
from tla.game_state import GameState
from tla.hexgrid import AxialCoord, distance
from tla.ship import Ship, ShipKind, ShipStats
from tla.tile import PlayerId

Decision = Literal["stay", "retreat"]
DecisionFn = Callable[[Ship, Ship, GameState], Decision]


def _base_damage(attacker_stats: ShipStats, defender: Ship) -> int:
    """`damage` applies to anything except a submerged submarine, which only
    `asw` can hurt -- a submerged sub's own outgoing attack still uses its
    normal `damage`, since this is evaluated per-attacker against the
    *other* ship's state."""
    if defender.kind == ShipKind.SUBMARINE and not defender.surfaced:
        return attacker_stats.asw
    return attacker_stats.damage


def _carrier_bonus(
    game_state: GameState, owner: PlayerId, battle_hex: AxialCoord, combat_config: CombatConfig
) -> int:
    radius = combat_config.ac_bonus_radius
    count = sum(
        1
        for ship in game_state.ships.values()
        if ship.owner == owner
        and ship.kind == ShipKind.CARRIER
        and not ship.is_sunk
        and distance(ship.position, battle_hex) <= radius
    )
    return count * combat_config.ac_bonus_amount


@dataclass
class RoundResult:
    damage_to_defender: int
    damage_to_attacker: int
    defender_hp_after: int
    attacker_hp_after: int
    defender_sunk: bool
    attacker_sunk: bool


def resolve_round(attacker: Ship, defender: Ship, game_state: GameState) -> RoundResult:
    """Apply one simultaneous exchange of damage, mutating both ships'
    current_hp in place. `defender.position` is the battle hex."""
    stats = game_state.config.ship_stats.stats
    combat_config = game_state.config.combat
    battle_hex = defender.position

    damage_to_defender = _base_damage(stats[attacker.kind], defender) + _carrier_bonus(
        game_state, attacker.owner, battle_hex, combat_config
    )
    damage_to_attacker = _base_damage(stats[defender.kind], attacker) + _carrier_bonus(
        game_state, defender.owner, battle_hex, combat_config
    )

    defender.current_hp = max(0, defender.current_hp - damage_to_defender)
    attacker.current_hp = max(0, attacker.current_hp - damage_to_attacker)

    return RoundResult(
        damage_to_defender=damage_to_defender,
        damage_to_attacker=damage_to_attacker,
        defender_hp_after=defender.current_hp,
        attacker_hp_after=attacker.current_hp,
        defender_sunk=defender.is_sunk,
        attacker_sunk=attacker.is_sunk,
    )


@dataclass
class BattleResult:
    attacker: Ship
    defender: Ship
    rounds: list[RoundResult] = field(default_factory=list)
    retreated: bool = False

    @property
    def attacker_sunk(self) -> bool:
        return self.attacker.is_sunk

    @property
    def defender_sunk(self) -> bool:
        return self.defender.is_sunk


def run_battle(attacker: Ship, defender: Ship, game_state: GameState, decision_fn: DecisionFn) -> BattleResult:
    """Resolve rounds until a sink or a retreat. `decision_fn` is asked
    after each round where both ships survive; it is never asked before the
    first round -- contact always causes at least one exchange."""
    result = BattleResult(attacker=attacker, defender=defender)
    while True:
        round_result = resolve_round(attacker, defender, game_state)
        result.rounds.append(round_result)
        if round_result.attacker_sunk or round_result.defender_sunk:
            return result
        if decision_fn(attacker, defender, game_state) == "retreat":
            result.retreated = True
            return result
