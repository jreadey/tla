"""Global-layer AI reasoning: overall force balance -> posture, plus a
port-threat ranking query -- the first, most strategic layer of the
three-level AI design the user outlined (global -> task-force -> tactical).
Built entirely on top of `tla.ai.enemy_model.EnemyModel` (Phase 0) and
`tla.ai.task_force`'s existing `group_power`/`outmatched`, not a
replacement for either -- this module only decides how aggressive or
cautious the AI should be right now and which of its own ports look most
threatened; task-force/tactical decisions still make their own calls,
just under margins this layer may have shifted.

`rank_port_threats` is deliberately query-only for now -- nothing in
`tla.ai` consumes it yet. It's exposed so a later, focused pass can wire it
into a real behavior change (e.g. a belief-driven port-defense trigger)
once the core posture mechanic here has been self-play-validated on its
own.
"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum

from tla.ai.enemy_model import EnemyModel
from tla.ai.task_force import DANGEROUS_TO_CARRIER_KINDS, outmatched
from tla.config import AiConfig
from tla.game_state import GameState
from tla.hexgrid import AxialCoord
from tla.tile import PlayerId


class Posture(Enum):
    AGGRESSIVE = "aggressive"
    NEUTRAL = "neutral"
    DEFENSIVE = "defensive"


def own_strength(game_state: GameState, player: PlayerId) -> tuple[int, int]:
    """(total current HP, total damage-stat output) across `player`'s own
    living ships -- the same `(hp, damage)` shape
    `tla.ai.task_force.group_power` uses, so it can be compared directly
    against `EnemyModel.expected_strength()` via `outmatched`."""
    stats = game_state.config.ship_stats.stats
    ships = game_state.ships_for(player)
    return sum(s.current_hp for s in ships), sum(stats[s.kind].damage for s in ships)


def compute_posture(
    game_state: GameState, player: PlayerId, enemy_model: EnemyModel, ai_config: AiConfig
) -> Posture:
    """AGGRESSIVE if `player` clearly outmatches the enemy's believed
    total strength (`EnemyModel.expected_strength` -- itself a safe upper
    bound on the true enemy strength, see that module's own docstring, so
    this is if anything a conservative call to go aggressive), DEFENSIVE
    if the enemy clearly outmatches `player`, NEUTRAL otherwise. Reuses
    `tla.ai.task_force.outmatched`/`group_power`'s exact `(hp, damage)`
    "rounds to kill" race -- the same comparison already driving
    `is_outnumbered` and port/carrier defense -- so "clearly ahead" means
    the same thing everywhere in this codebase, not a second, separate
    definition of relative strength."""
    ours = own_strength(game_state, player)
    theirs = enemy_model.expected_strength()
    if outmatched(theirs, ours, ai_config.posture_margin):
        return Posture.AGGRESSIVE
    if outmatched(ours, theirs, ai_config.posture_margin):
        return Posture.DEFENSIVE
    return Posture.NEUTRAL


def posture_adjusted_ai_config(ai_config: AiConfig, posture: Posture) -> AiConfig:
    """A copy of `ai_config` with `task_force_outnumbered_margin`/
    `port_defense_margin`/`carrier_defense_margin` uniformly shifted --
    AGGRESSIVE raises them (tolerate worse odds before retreating, or
    counterattack a port/carrier threat instead of just blocking it),
    DEFENSIVE lowers them (retreat/block sooner, even from a roughly even
    fight). NEUTRAL returns `ai_config` unchanged (not a copy). One shared
    delta rather than per-knob tuning: every one of these margins already
    means the same thing -- see `tla.ai.task_force.outmatched`'s own
    docstring -- a higher margin makes "outmatched" harder to trigger for
    the side it's compared in favor of, for all three."""
    if posture == Posture.NEUTRAL:
        return ai_config
    delta = ai_config.posture_margin_shift if posture == Posture.AGGRESSIVE else -ai_config.posture_margin_shift
    return replace(
        ai_config,
        task_force_outnumbered_margin=ai_config.task_force_outnumbered_margin + delta,
        port_defense_margin=ai_config.port_defense_margin + delta,
        carrier_defense_margin=ai_config.carrier_defense_margin + delta,
    )


def rank_port_threats(
    game_state: GameState, player: PlayerId, enemy_model: EnemyModel, radius: int
) -> list[tuple[AxialCoord, float]]:
    """Every port `player` currently controls, ranked most-to-least
    threatened by summed believed enemy mass (`EnemyModel.mass_near`,
    `DANGEROUS_TO_CARRIER_KINDS` only -- the same kinds carrier
    self-defense already worries about, since those are what's actually
    worth worrying about approaching a port) within `radius` sea hexes of
    it. Query-only -- see this module's own docstring for why nothing
    consumes it yet."""
    ports = game_state.board.controlled_ports_for(player)
    ranked = [(port, enemy_model.mass_near(port, radius, kinds=DANGEROUS_TO_CARRIER_KINDS)) for port in ports]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked
