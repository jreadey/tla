"""Battle resolution: what happens when a ship moves into an enemy-occupied
hex.

Damage is simultaneous each round: both ships compute and apply damage the
same round, based on each ship's `damage` stat -- or `asw` if the target is
a submerged submarine, which `damage` cannot touch at all. Each of a side's
own aircraft carriers within `CombatConfig.ac_bonus_radius` of the battle
hex adds `CombatConfig.ac_bonus_amount` to that side's attack, recomputed
fresh every round (a carrier arriving or sinking mid-battle changes it) --
but only for the "larger surface ships" (`_AC_BONUS_ELIGIBLE_KINDS`), and
never against a submerged submarine target: air cover doesn't help spot or
track something submerged, regardless of which side has it or which side
the submerged sub itself is fighting from (attacker or defender role).
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
from tla.production import handle_port_capture
from tla.ship import Ship, ShipKind, ShipStats

Decision = Literal["stay", "retreat"]
DecisionFn = Callable[[Ship, Ship, GameState], Decision]

# Only these "larger surface ship" kinds benefit from nearby carrier air
# cover -- a submarine (submerged or not) or patrol boat gets no bonus,
# regardless of how many friendly carriers are nearby.
_AC_BONUS_ELIGIBLE_KINDS = frozenset(
    {ShipKind.CARRIER, ShipKind.BATTLESHIP, ShipKind.CRUISER, ShipKind.DESTROYER}
)


def _base_damage(attacker_stats: ShipStats, defender: Ship) -> int:
    """`damage` applies to anything except a submerged submarine, which only
    `asw` can hurt -- a submerged sub's own outgoing attack still uses its
    normal `damage`, since this is evaluated per-attacker against the
    *other* ship's state."""
    if defender.kind == ShipKind.SUBMARINE and not defender.surfaced:
        return attacker_stats.asw
    return attacker_stats.damage


def carrier_bonus_for(
    game_state: GameState,
    attacking_ship: Ship,
    target: Ship,
    battle_hex: AxialCoord,
    combat_config: CombatConfig,
) -> int:
    """Bonus damage for `attacking_ship`'s side from nearby friendly
    carriers, against `target` -- zero outright if `attacking_ship` isn't
    one of the kinds that benefits (see `_AC_BONUS_ELIGIBLE_KINDS`), or if
    `target` is a submerged submarine: air cover never helps against a
    submerged target, no matter which side has the carrier or which role
    (attacker/defender) the submerged sub is playing in this engagement.
    Public so callers other than `resolve_round` -- e.g.
    `tla.ai.scoring.matchup_score`, estimating an engagement before
    committing to it -- can reuse the exact same combat math rather than
    risking a duplicated, driftable copy of it."""
    if attacking_ship.kind not in _AC_BONUS_ELIGIBLE_KINDS:
        return 0
    if target.kind == ShipKind.SUBMARINE and not target.surfaced:
        return 0
    radius = combat_config.ac_bonus_radius
    count = sum(
        1
        for ship in game_state.ships.values()
        if ship.owner == attacking_ship.owner
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

    damage_to_defender = _base_damage(stats[attacker.kind], defender) + carrier_bonus_for(
        game_state, attacker, defender, battle_hex, combat_config
    )
    damage_to_attacker = _base_damage(stats[defender.kind], attacker) + carrier_bonus_for(
        game_state, defender, attacker, battle_hex, combat_config
    )

    defender.current_hp = max(0, defender.current_hp - damage_to_defender)
    attacker.current_hp = max(0, attacker.current_hp - damage_to_attacker)

    # Tallied for the after-action report (tla.rendering.game_view) --
    # each side's own dealt/taken from this round, attributed by owner so
    # it's correct regardless of which one is nominally the "attacker".
    attacker_stats = game_state.turn_stats[attacker.owner]
    attacker_stats.hp_dealt += damage_to_defender
    attacker_stats.hp_taken += damage_to_attacker
    defender_stats = game_state.turn_stats[defender.owner]
    defender_stats.hp_dealt += damage_to_attacker
    defender_stats.hp_taken += damage_to_defender

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
    first round -- contact always causes at least one exchange. Applies the
    outcome to `game_state` before returning -- see `apply_battle_outcome`."""
    result = BattleResult(attacker=attacker, defender=defender)
    while True:
        round_result = resolve_round(attacker, defender, game_state)
        result.rounds.append(round_result)
        if round_result.attacker_sunk or round_result.defender_sunk:
            break
        if decision_fn(attacker, defender, game_state) == "retreat":
            result.retreated = True
            break
    apply_battle_outcome(game_state, attacker, defender)
    return result


def apply_battle_outcome(game_state: GameState, attacker: Ship, defender: Ship) -> None:
    """Apply a concluded battle's consequences to `game_state`: remove any
    sunk ship(s), move a surviving attacker onto a defeated defender's hex
    (which may capture a port there -- see
    `tla.production.handle_port_capture`), and refresh the winner. A pure
    retreat (both survive) needs no changes here -- the attacker never
    advanced past its approach hex in the first place, per
    `tla.movement.begin_engagement`.

    Called automatically by `run_battle` for a synchronous decision_fn (an
    AI, say); a UI driving rounds one at a time via `resolve_round` directly
    -- to let the player see each round before choosing -- must call this
    itself once the battle actually concludes.
    """
    if defender.is_sunk:
        game_state.turn_stats[defender.owner].ships_lost.append(defender.kind)
        del game_state.ships[defender.id]
        if not attacker.is_sunk:
            attacker.position = defender.position
            handle_port_capture(game_state, attacker.position)
    if attacker.is_sunk:
        game_state.turn_stats[attacker.owner].ships_lost.append(attacker.kind)
        del game_state.ships[attacker.id]
    game_state.refresh_winner()
