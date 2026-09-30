"""Tactical layer: turn-level coordination across a player's own ships,
the third and final level of the three-level AI design (global posture ->
task-force goals -> tactical). Sits above `tla.ai.scoring` (single-ship,
context-free combat math) the same way `tla.ai.task_force` (group-level)
sits above it too -- this module is what actually coordinates *between*
several of the player's own ships acting in the same turn, which nothing
below it does.

The motivating gap, confirmed by reading the actual code before writing
any of this: `tla.ai.policy.plan_movement` processes a player's ships one
at a time, each independently picking its own best 1v1 target via
`scoring.matchup_score`. Two of a player's battleships facing two enemy
battleships will each pick their own best target and split fire, leaving
both enemies alive to both hit back next turn -- when concentrating fire
to sink one enemy removes its entire future damage output instead. This
module's `secure_kills_pass` finds and commits to exactly that kind of
joint-only kill before the ordinary per-ship loop runs.

Combat in this engine is fully deterministic (see `tla.battle` -- no RNG
anywhere), so every prediction here is exact closed-form arithmetic
mirroring `battle.resolve_round`/`NaivePolicy.decide_battle`'s own real
mechanics, never a simulation or an approximation: `decide_battle` never
breaks off a race it's still winning (or a worthwhile tie, see
`scoring.worth_a_tie`) partway through, so a strictly-favorable
`matchup_score` -- or a tie worth taking -- always runs to completion.
`secures_kill` captures exactly that.

Deliberately scoped to *visible* enemies and to the kill-prioritization
mechanism only -- a later, separate pass may add a post-attack "will an
uninvolved enemy be able to reach and punish me next turn" exposure
penalty; this module doesn't attempt that yet.
"""

from __future__ import annotations

from typing import Callable, Iterator

from tla.ai import scoring
from tla.config import AiConfig
from tla.game_state import GameState
from tla.hexgrid import AxialCoord
from tla.ship import Ship, ShipKind
from tla.tile import PlayerId


def secures_kill(ship: Ship, target: Ship, game_state: GameState) -> bool:
    """True iff `ship` attacking `target` alone, this turn, finishes it
    off: either a *clean* win (`matchup_score > 0`, strictly -- `ship`
    survives) or a *worthwhile* tie (`matchup_score == 0` and `scoring.
    worth_a_tie` says the trade is worth it -- both die, but the kill
    still lands; see `_favorable_attack`'s own docstring for the full
    reasoning). Either way this always completes: `NaivePolicy.
    decide_battle` never breaks off a race it's still winning (or a tie
    it's committed to) partway through, so nothing here needs to predict
    an early bail-out anymore -- no round-by-round simulation needed,
    just the race itself."""
    battle_hex = target.position  # the real, fixed battle location -- see effective_damage's own docstring
    my_dpr = scoring.effective_damage(ship, target, battle_hex, game_state)
    if my_dpr <= 0:
        return False
    score = scoring.matchup_score(ship, target, game_state)
    return score > 0 or (score == 0 and scoring.worth_a_tie(ship, target, game_state))


def would_secure_kill(target: Ship, attackers: list[Ship], game_state: GameState) -> list[Ship] | None:
    """Whether any of `attackers` secures `target`'s kill this turn (see
    `secures_kill`) -- callers should already have checked this per-
    attacker against `target`'s real current state and excluded any that
    do (that's the simpler, already-handled case, not this function's
    job); this exists for the remaining case where none do *yet* but one
    might once the field is narrowed to exactly `target`. Returns a
    one-ship prefix for the best candidate (ordered by `matchup_score`,
    best first), or `None`.

    Never returns more than one ship: with `NaivePolicy.decide_battle`
    never breaking off a race it's winning (or a worthwhile tie) partway
    through, `secures_kill` already means "finishes `target` off, alone,
    this turn" in full -- there's no more partial-damage contribution for
    a second ship to build on, since a favorable race always completes by
    itself and a worthwhile tie always ends in `target`'s death too. A
    genuinely multi-ship chain (one ship weakens `target`, a second
    finishes it) isn't attempted here -- that would mean deliberately
    committing a ship to a fight it doesn't itself judge worth taking,
    a different and riskier doctrine than "take good trades," not part of
    this pass."""
    ordered = sorted(attackers, key=lambda s: (-scoring.matchup_score(s, target, game_state), s.id))
    for ship in ordered:
        if secures_kill(ship, target, game_state):
            return [ship]
    return None


def action_value(ship: Ship, target: Ship, game_state: GameState, ai_config: AiConfig) -> float:
    """`matchup_score`, plus a bonus if this specific attack secures
    `target`'s kill this turn -- weighted by exactly the future
    damage-per-round that kill removes from the opponent
    (`ShipStats.damage` for `target`'s kind), not an arbitrary flat
    constant: finishing off a target is worth exactly the firepower it
    stops being able to bring next turn, no more and no less. Degenerate
    `matchup_score` values (+-inf) pass through unchanged -- the kill
    bonus is always finite, so it never overturns those."""
    value = scoring.matchup_score(ship, target, game_state)
    if ai_config.tactics_enabled and secures_kill(ship, target, game_state):
        value += ai_config.tactics_kill_weight * game_state.config.ship_stats.stats[target.kind].damage
    return value


def secure_kills_pass(
    game_state: GameState,
    player: PlayerId,
    visible_enemies: dict[int, Ship],
    ai_config: AiConfig,
    apply_attack: Callable[[Ship, AxialCoord], None],
) -> Iterator[None]:
    """One pass, run once per `plan_movement` call, *before* the ordinary
    per-ship loop: finds enemy targets no single one of `player`'s ships
    would otherwise pick as its own best target (each ship, run
    independently, might split fire across different targets) but that
    `would_secure_kill` finds a real securing candidate for once judged
    directly against that one target, and commits to it -- calling
    `apply_attack` for the committed ship, yielding once per commit (same
    per-ship pacing the ordinary loop and the port/carrier-defense passes
    already use). Returns (via `return`, so a caller does `resolved |=
    yield from secure_kills_pass(...)`) the set of ship ids committed
    here, so the caller's own `resolved` set -- the same claim-and-skip
    pattern port/carrier defense already establish -- excludes them from
    the ordinary loop afterward.

    No-op (empty, no ships claimed) if `ai_config.tactics_enabled` is
    False or there are no visible enemies at all. `apply_attack` is
    injected rather than imported directly (it's `NaivePolicy`'s own
    `_execute` plus its submarine-stealth-toggle step, in `policy.py`) --
    `tla.ai.policy` imports this module, so this module importing back
    from `policy.py` would be a cycle; the same reasoning already used
    for `tla.battle.run_battle`'s own `decision_fn` seam.

    Bounded, no combinatorial assignment search: for each of `player`'s
    ships, one `scoring.reachable_attack_candidates` call (a single BFS,
    the same cost the ordinary per-ship loop already pays per ship, just
    done once up front); for each remaining joint-only target, `would_
    secure_kill` checks at most `ai_config.tactics_max_focus_fire_group`
    candidate ships, small and capped. This is why a greedy pass suffices
    rather than a real search: combat is deterministic (see this module's
    own docstring), so one closed-form prediction per candidate is exact,
    never needing to be sampled or retried."""
    claimed: set[int] = set()
    if not ai_config.tactics_enabled or not visible_enemies:
        return claimed

    enemy_by_position = {e.position: e for e in visible_enemies.values()}
    # A carrier never deliberately self-initiates an attack, regardless of
    # this pass's own priorities -- see tla.ai.policy._choose_carrier_
    # destination's own explicit doctrine. Excluded here at the source
    # rather than relying on it being incidentally true elsewhere: before
    # would_secure_kill was simplified to a single-candidate search (see
    # its own docstring), a solo-securable target was always filtered out
    # of this pass entirely, so a carrier that could secure a kill alone
    # never reached here in the first place; now that every securable
    # target is considered (not just ones needing coordination), a lone
    # carrier with a favorable matchup would otherwise get swept in too.
    own_ships = [s for s in game_state.ships_for(player) if s.kind != ShipKind.CARRIER]

    # target id -> [(ship, attack_hex), ...] -- every one of player's
    # ships that can reach and attack this target this turn.
    candidates: dict[int, list[tuple[Ship, AxialCoord]]] = {}
    for ship in own_ships:
        for coord, _cost in scoring.reachable_attack_candidates(ship, game_state, visible_enemies):
            target = enemy_by_position[coord]
            candidates.setdefault(target.id, []).append((ship, coord))

    # Highest future damage-output removed first, so this pass doesn't
    # spend ships on a low-value kill and leave a higher-value one
    # uncoordinated -- also what stops every ship's own independent best-
    # target pick (the ordinary per-ship loop, run afterward) from piling
    # onto the same juiciest kill while an equally-securable target goes
    # completely unclaimed. Every target with at least one reachable
    # candidate is ranked here now (not just ones no single candidate
    # could already secure alone) -- with would_secure_kill simplified to
    # "find the best single securing candidate" (see its own docstring),
    # that per-target securing check and this pass's own claim now happen
    # in the same place, rather than needing a separate up-front filter
    # to tell them apart.
    ranked_target_ids = sorted(
        candidates, key=lambda target_id: -game_state.config.ship_stats.stats[visible_enemies[target_id].kind].damage
    )

    for target_id in ranked_target_ids:
        target = game_state.ships.get(target_id)
        if target is None:
            continue  # already sunk earlier this same pass -- e.g. return fire
        group = [ship for ship, _hex in candidates[target_id] if ship.id not in claimed and ship.id in game_state.ships]
        group = group[: ai_config.tactics_max_focus_fire_group]
        if not group:
            continue
        prefix = would_secure_kill(target, group, game_state)
        if prefix is None:
            continue
        hex_for_ship = {ship.id: hexcoord for ship, hexcoord in candidates[target_id]}
        for ship in prefix:
            if ship.id not in game_state.ships or target_id not in game_state.ships:
                break  # this ship or the target itself already resolved mid-sequence
            apply_attack(ship, hex_for_ship[ship.id])
            claimed.add(ship.id)
            yield

    return claimed
