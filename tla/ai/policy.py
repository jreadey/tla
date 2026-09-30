"""NaivePolicy: a first, deliberately simple AI opponent.

Every decision here drives the exact same `tla.movement`/`tla.battle`/
`tla.production` functions a human's input goes through -- never a bypass,
so anything the AI does is exactly as legal as a human move, and
`GameState.refresh_winner()` (called automatically by those functions)
keeps tracking the winner correctly with no extra effort here.

`NaivePolicy` is *mostly* stateless -- `decide_battle` is a pure function of
the `GameState` it's handed -- with a few deliberate exceptions:
`plan_movement` remembers each player's task forces (see
`tla.ai.task_force`), each player's belief about the enemy fleet (see
`tla.ai.enemy_model.EnemyModel`), and the true, unshifted `AiConfig` (see
`tla.ai.global_strategy` -- `plan_movement` overwrites `game_state.config.
ai` with a posture-adjusted copy every call, so the real base has to be
captured once, separately) across turns, keyed by `PlayerId` on the
instance itself where relevant, so a force's goal/stall-progress tracking,
the enemy model's roster/position belief, and posture's own margin shifts
all persist and stay consistent turn to turn. This means, unlike before any
of this existed, **the same instance must keep being reused turn after turn
for a given player** for that continuity to mean anything -- constructing a
fresh `NaivePolicy` mid-game would silently forget every force, reset enemy
belief to "nothing known yet," and lose the true base `AiConfig` (letting
posture's shift compound onto whatever `game_state.config.ai` was last left
holding). `tla.rendering.game_view.GameView` already does this (one
instance for the whole game); so does `tests/test_ai_selfplay.py`. All of
this is purely AI-internal planning state -- never part of `GameState`,
never persisted, never visible to the UI or a human player.
"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from tla import movement
from tla.ai import scoring
from tla.ai.enemy_model import EnemyModel, opponent_of
from tla.ai.global_strategy import Posture, compute_posture, own_strength, posture_adjusted_ai_config
from tla.ai.move_scoring import plan_force_movement_scored
from tla.ai.tactics import action_value, secure_kills_pass, secures_kill, would_secure_kill
from tla.ai.task_force import (
    DANGEROUS_TO_CARRIER_KINDS,
    GoalKind,
    TaskForce,
    TaskForceGoal,
    apply_defensive_port_priority,
    block_hex_toward,
    compute_carrier_defense_directives,
    compute_port_defense_directives,
    enemy_reachable_next_turn,
    force_recapture_goal,
    form_task_forces,
    group_power,
    is_outmatched_by_target_support,
    outmatched,
    rally_point,
    record_task_force_progress,
    recruit_into_open_forces,
    repair_task_forces,
    sea_distance_field,
    sea_route_distance,
    update_task_force_goals,
    update_task_force_stance,
    with_repositioned_ships,
)
from tla.ai.vision import enemy_ships_visible_to
from tla.battle import AC_BONUS_ELIGIBLE_KINDS, Decision, run_battle
from tla.config import AiConfig
from tla.game_state import GameState
from tla.hexgrid import AxialCoord, distance, neighbors
from tla.movement import shortest_path
from tla.ship import Ship, ShipKind, ShipStats
from tla.tile import PlayerId

if TYPE_CHECKING:
    # Only for the type hint below -- never imported at runtime, so
    # NaivePolicy never needs h5py installed unless enemy_belief_path is
    # actually given. See tla.ai.belief_store's own module docstring.
    from tla.ai.belief_store import BeliefStore

# The two kinds that trail behind a task force's capital ships as its
# rearguard -- see _rearguard_target.
_ESCORT_KINDS = frozenset({ShipKind.PATROL_BOAT, ShipKind.DESTROYER})
# High-value, thin-`asw` ships worth escorting -- see _rearguard_target.
_CAPITAL_KINDS = frozenset({ShipKind.BATTLESHIP, ShipKind.CARRIER})
# The two weight classes _carrier_screen_destination screens against,
# each with its own matching own-kind screener and radius (AiConfig.
# carrier_heavy_screen_radius/carrier_light_screen_radius) -- a heavy
# enemy (battleship/cruiser) gets screened by an own battleship/cruiser,
# a light/fast enemy (destroyer/submarine) by an own destroyer/submarine.
_HEAVY_SCREEN_KINDS = frozenset({ShipKind.BATTLESHIP, ShipKind.CRUISER})
_LIGHT_SCREEN_KINDS = frozenset({ShipKind.DESTROYER, ShipKind.SUBMARINE})
_NO_ENEMY_SENTINEL = 10_000


class Strategy(Enum):
    """A task force's per-turn tactical stance, chosen fresh each turn by
    `_pick_strategy_for_force` -- see that function's own docstring for how,
    and `choose_task_force_destination`/`_choose_carrier_destination` for
    what each one actually changes about a member's destination. `ADVANCE`
    is today's long-standing default behavior (attack only a strictly
    favorable fight, otherwise advance in formation); the other three are
    small, targeted relaxations/overrides of that same pipeline, not
    separate algorithms -- see the plan this was built from for why that
    was possible."""

    AGGRESSIVE = "aggressive"
    ADVANCE = "advance"
    HOLD = "hold"
    RETREAT = "retreat"


# Tie-break ordering for _pick_strategy_for_force: when two strategies score
# equally, prefer the least drastic -- i.e. the lowest value here.
_STRATEGY_DRASTICNESS: dict[Strategy, int] = {
    Strategy.ADVANCE: 0,
    Strategy.HOLD: 1,
    Strategy.AGGRESSIVE: 2,
    Strategy.RETREAT: 3,
}


class NaivePolicy:
    def __init__(self, enemy_belief_path: str | Path | None = None) -> None:
        self._task_forces: dict[PlayerId, list[TaskForce]] = {}
        self._next_force_id: int = 1
        # Last turn's controlled_ports_for(player) snapshot -- compared
        # each plan_movement call to detect a just-lost port (see
        # _detect_lost_ports) and never touched otherwise.
        self._controlled_ports_seen: dict[PlayerId, set[AxialCoord]] = {}
        # One EnemyModel per player, built lazily the first time
        # plan_movement runs for them (needs a GameState to seed initial
        # belief from) -- see _enemy_model_for. Same persistence pattern
        # as _task_forces: this must survive across turns for a player, so
        # the same NaivePolicy instance has to keep being reused.
        self._enemy_models: dict[PlayerId, EnemyModel] = {}
        # Optional path for tla.ai.belief_store.BeliefStore -- one shared
        # store (not one per player) since ship ids are globally unique, so
        # both players' EnemyModels can safely write to the same file with
        # no dataset-name collisions. Lazily constructed on first use (only
        # then do we need h5py importable at all -- see _enemy_model_for).
        self._enemy_belief_path = enemy_belief_path
        self._belief_store: "BeliefStore | None" = None
        # Global layer (see tla.ai.global_strategy) -- the true, unshifted
        # AiConfig, captured once the first time plan_movement ever runs
        # for either player (both share one game_state.config). Posture
        # is always computed from this stable copy, never from whatever
        # game_state.config.ai currently holds -- that field gets
        # overwritten with a posture-adjusted copy every plan_movement
        # call (see below), and computing from it instead of this fixed
        # base would let each turn's shift compound onto the last one's.
        self._base_ai_config: AiConfig | None = None
        self._last_posture: dict[PlayerId, Posture] = {}
        # (own, believed-enemy) (hp, damage) totals posture was computed
        # from, captured alongside it purely for diagnostics -- see
        # posture_snapshot_for.
        self._last_posture_inputs: dict[PlayerId, tuple[tuple[int, int], tuple[int, int]]] = {}
        # AiConfig.defense_stall_turns: per-ship consecutive-turns-without-
        # landing-an-attack count while assigned a port/carrier-defense
        # counterattack directive -- see that field's own doc-comment.
        # Rebuilt (pruned) fresh every plan_movement call to only the ship
        # ids actually considered a counterattack candidate that turn --
        # see _update_defense_stall_turns.
        self._defense_stall_turns: dict[PlayerId, dict[int, int]] = {}

    def _enemy_model_for(self, game_state: GameState, player: PlayerId) -> EnemyModel:
        if player not in self._enemy_models:
            if self._enemy_belief_path is not None and self._belief_store is None:
                from tla.ai.belief_store import BeliefStore  # local: only needs h5py if actually used

                self._belief_store = BeliefStore(self._enemy_belief_path)
            self._enemy_models[player] = EnemyModel(
                game_state, player, opponent_of(player, game_state), belief_store=self._belief_store
            )
        return self._enemy_models[player]

    def enemy_model_for(self, player: PlayerId) -> EnemyModel | None:
        """Read-only access to `player`'s current belief about the enemy
        fleet -- for diagnostics/tests only, mirrors `task_forces_for`.
        None if `player` hasn't had `plan_movement` called for them yet
        this game."""
        return self._enemy_models.get(player)

    def close_enemy_belief_store(self) -> None:
        """No-op if `enemy_belief_path` was never given, or already
        closed. Call once the game ends -- mirrors
        `tla.replay.ReplayWriter.write_final`'s own idempotent close."""
        if self._belief_store is not None:
            self._belief_store.close()

    def posture_for(self, player: PlayerId) -> Posture | None:
        """Read-only access to `player`'s current global-layer posture
        (see `tla.ai.global_strategy`) -- for diagnostics/tests only,
        mirrors `task_forces_for`/`enemy_model_for`. None if `player`
        hasn't had `plan_movement` called for them yet this game."""
        return self._last_posture.get(player)

    def posture_snapshot_for(self, player: PlayerId) -> dict | None:
        """Read-only diagnostic snapshot of `player`'s current posture plus
        the exact `(own, believed-enemy)` `(hp, damage)` totals it was
        computed from -- for `tla.replay`'s replay logging, mirrors
        `task_forces_for`/`enemy_model_for`. None if `player` hasn't had
        `plan_movement` called for them yet this game."""
        posture = self._last_posture.get(player)
        if posture is None:
            return None
        own, believed_enemy = self._last_posture_inputs[player]
        return {
            "posture": posture.value,
            "own_hp": own[0],
            "own_damage": own[1],
            "believed_enemy_hp": believed_enemy[0],
            "believed_enemy_damage": believed_enemy[1],
        }

    def _alloc_force_id(self) -> int:
        force_id = self._next_force_id
        self._next_force_id += 1
        return force_id

    def task_forces_for(self, player: PlayerId) -> list[TaskForce]:
        """Read-only view of `player`'s current task forces -- for
        diagnostics only (e.g. `tla.replay`'s replay logging); never
        mutate the returned list or its `TaskForce` objects. Empty if
        `player` isn't AI-controlled, or hasn't had `plan_movement`
        called for them yet this game."""
        return self._task_forces.get(player, [])

    def _detect_lost_ports(self, game_state: GameState, player: PlayerId) -> set[AxialCoord]:
        """Ports `player` controlled as of the last `plan_movement` call
        for them and doesn't anymore -- see `force_recapture_goal`. Always
        updates the stored snapshot for next time, even when nothing was
        lost (ports gained/held are just as relevant to "what did I have
        last time"). Empty, and no false "loss", on a player's very first
        call this game (no prior snapshot to compare against)."""
        current = set(game_state.board.controlled_ports_for(player))
        previous = self._controlled_ports_seen.get(player)
        self._controlled_ports_seen[player] = current
        if previous is None:
            return set()
        return previous - current

    def plan_movement(self, game_state: GameState, player: PlayerId) -> Iterator[None]:
        """Move every one of `player`'s ships this turn, resolving any
        engagement a ship starts synchronously (via `battle.run_battle`,
        given `self.decide_battle` as its `decision_fn`) before moving on
        to the next ship -- `run_battle` loops rounds internally and
        applies the final outcome itself, so this never needs to suspend
        mid-battle waiting on a decision. Yields once after each ship is
        fully resolved, so a caller (e.g. the UI) can pace draining this
        generator one ship at a time; a caller that wants it to run
        instantly can just exhaust it, e.g.
        `list(policy.plan_movement(game_state, player))`."""
        # Advances this player's belief about the enemy fleet by one turn
        # and folds in anything newly known -- must run before anything
        # else, while game_state.battle_log is still guaranteed empty (see
        # EnemyModel.begin_turn). Also must run before the posture
        # computation just below, which reads this same model's belief.
        model = self._enemy_model_for(game_state, player)
        model.begin_turn(game_state)

        # Global layer (see tla.ai.global_strategy): how this player's own
        # strength compares to the enemy's believed strength, applied by
        # overwriting game_state.config.ai with a posture-adjusted copy
        # for the rest of this call -- every task-force/tactical decision
        # below reads AiConfig either via an explicit parameter or
        # directly off game_state.config.ai, so this has to actually
        # replace that field, not just be computed and set aside. Always
        # computed from self._base_ai_config (captured once, below), never
        # from game_state.config.ai itself -- that already holds whatever
        # the *previous* plan_movement call (for either player) left
        # there, and computing from an already-shifted value would let
        # each turn's shift compound onto the last one's.
        if self._base_ai_config is None:
            self._base_ai_config = game_state.config.ai
        posture = compute_posture(game_state, player, model, self._base_ai_config)
        self._last_posture[player] = posture
        self._last_posture_inputs[player] = (own_strength(game_state, player), model.expected_strength())
        game_state.config = replace(
            game_state.config, ai=posture_adjusted_ai_config(self._base_ai_config, posture)
        )

        forces = self._task_forces.setdefault(player, [])
        repair_task_forces(game_state, player, forces)
        recruit_into_open_forces(game_state, player, forces, game_state.config.ai)
        form_task_forces(game_state, player, forces, game_state.config.ai, self._alloc_force_id)

        # Captured now, before anything moves this turn (port defense
        # below included) -- see _compute_force_pace for why it must not
        # be recomputed mid-loop.
        pace_by_force = _compute_force_pace(game_state, forces)

        # A just-lost port overrides every force's real goal, unconditionally
        # -- "throw anything it had" -- before the usual stance/goal
        # maintenance below, which would otherwise leave an in-progress
        # goal untouched for many turns (see force_recapture_goal).
        lost_ports = self._detect_lost_ports(game_state, player)
        if lost_ports:
            for force in forces:
                force_recapture_goal(force, game_state, lost_ports)

        # Global layer feeding back into task-force goals: under DEFENSIVE
        # posture, redirect one force to defend the most (believed-)
        # threatened controlled port -- see apply_defensive_port_priority.
        # Runs after recapture (so a just-lost port is never also "defended"
        # -- it's no longer controlled_ports_for(player) by this point
        # anyway) and before update_task_force_goals below, so a force this
        # releases gets a fresh goal the same turn instead of sitting idle.
        apply_defensive_port_priority(forces, game_state, player, model, posture, game_state.config.ai)

        # A still-*controlled* port that's merely threatened gets the same
        # kind of unconditional priority, one step earlier -- see
        # _port_defense_destinations/compute_port_defense_directives. Ships
        # claimed here are marked resolved so the main loop below doesn't
        # reconsider them this turn.
        #
        # AiConfig.defense_stall_turns: a counterattack-role ship (never a
        # block-role one -- see that field's own doc-comment) that's
        # already given up this "same" defense response (its own count
        # already at the threshold) is skipped entirely here -- left
        # unresolved so it falls through to its ordinary task-force
        # movement below, which has its own, already-tested stall
        # handling. `stall_turns` is this player's persistent tracker
        # (survives across turns on `self`); `stalled_seen` collects every
        # counterattack-role ship id actually considered this turn (skipped
        # or not) so the tracker can be pruned to just those at the end --
        # a ship no longer a candidate at all (its threat resolved, or it
        # drifted out of response range) gets its count dropped rather than
        # silently carried into some later, unrelated threat.
        resolved: set[int] = set()
        stall_turns = self._defense_stall_turns.setdefault(player, {})
        stall_limit = game_state.config.ai.defense_stall_turns
        stalled_seen: set[int] = set()
        port_defense_assignments = _port_defense_assignments(
            game_state, player, enemy_ships_visible_to(game_state, player), game_state.config.ai
        )
        for ship, directive in port_defense_assignments:
            if ship.id not in game_state.ships:
                continue  # sunk earlier this same pass (e.g. return fire on a counterattack)
            is_counterattack = directive.block is None
            if is_counterattack:
                stalled_seen.add(ship.id)
                if stall_turns.get(ship.id, 0) >= stall_limit:
                    continue  # given up -- ordinary movement handles it below instead
            resolved.add(ship.id)
            visible_enemies = enemy_ships_visible_to(game_state, player)
            if ship.kind == ShipKind.SUBMARINE:
                _maybe_toggle_submarine(ship, game_state, visible_enemies)
            destination = _port_defense_destination(ship, directive, game_state, visible_enemies)
            will_attack = destination is not None and game_state.ship_at(destination) is not None
            if destination is not None and destination != ship.position:
                _execute(ship, destination, game_state, self.decide_battle)
            if is_counterattack:
                if will_attack:
                    stall_turns.pop(ship.id, None)
                else:
                    stall_turns[ship.id] = stall_turns.get(ship.id, 0) + 1
            yield

        # Same unconditional priority as port defense just above, for a
        # carrier a dangerous enemy could reach next turn rather than a
        # threatened port -- see compute_carrier_defense_directives for
        # why this exists (a real game showed the naive AI declining to
        # engage purely because the fight itself was an exact tie, when
        # losing the carrier for nothing next turn was clearly worse).
        # `resolved` (port defense's own claims) is passed through so the
        # same ship is never double-booked by both in one turn. Same
        # defense_stall_turns handling as port defense just above.
        carrier_defense_assignments = _carrier_defense_assignments(
            game_state, player, enemy_ships_visible_to(game_state, player), game_state.config.ai, frozenset(resolved)
        )
        for ship, directive in carrier_defense_assignments:
            if ship.id not in game_state.ships:
                continue  # sunk earlier this same pass
            is_counterattack = directive.block is None
            if is_counterattack:
                stalled_seen.add(ship.id)
                if stall_turns.get(ship.id, 0) >= stall_limit:
                    continue
            resolved.add(ship.id)
            visible_enemies = enemy_ships_visible_to(game_state, player)
            if ship.kind == ShipKind.SUBMARINE:
                _maybe_toggle_submarine(ship, game_state, visible_enemies)
            destination = _carrier_defense_destination(ship, directive, game_state, visible_enemies)
            will_attack = destination is not None and game_state.ship_at(destination) is not None
            if destination is not None and destination != ship.position:
                _execute(ship, destination, game_state, self.decide_battle)
            if is_counterattack:
                if will_attack:
                    stall_turns.pop(ship.id, None)
                else:
                    stall_turns[ship.id] = stall_turns.get(ship.id, 0) + 1
            yield

        # AiConfig.defense_stall_turns bookkeeping: drop any tracked ship
        # id not actually considered a counterattack candidate by either
        # pass this turn, so an old count never silently carries over into
        # a later, unrelated threat -- see this player's own port-defense
        # comment above for the full reasoning.
        self._defense_stall_turns[player] = {
            ship_id: count for ship_id, count in stall_turns.items() if ship_id in stalled_seen
        }

        # Multi-candidate tactical strategy: for each not-already-retreating
        # force, pick which of AGGRESSIVE/ADVANCE/HOLD/RETREAT it plays this
        # turn (see _pick_strategy_for_force -- a cheap ADVANCE/no-retreat
        # default when the force is clearly ahead, clearly behind, or has
        # nothing visible; a genuine 4-way dry-run comparison otherwise).
        # The RETREAT verdict feeds update_task_force_stance as its third
        # retreat trigger, alongside its existing is_outnumbered/is_attritting
        # checks -- a force whose verdict is RETREAT has force.retreating
        # flip to True inside that very call, so the main per-ship loop
        # below already sees it via _choose_destination's own retreating
        # check the same turn, same as the prior engagement-value trigger
        # did. strategy_by_force is kept regardless so every other force
        # (AGGRESSIVE/ADVANCE/HOLD) gets its chosen stance passed down too.
        stance_visible_enemies = enemy_ships_visible_to(game_state, player)
        strategy_by_force: dict[int, Strategy] = {}
        should_retreat_by_force: dict[int, bool] = {}
        for f in forces:
            if f.retreating:
                continue
            strategy, should_retreat = _pick_strategy_for_force(
                f, game_state, player, stance_visible_enemies, forces, pace_by_force, model
            )
            strategy_by_force[f.id] = strategy
            should_retreat_by_force[f.id] = should_retreat
        update_task_force_stance(game_state, player, forces, stance_visible_enemies, should_retreat_by_force)
        update_task_force_goals(game_state, player, forces)
        # AiConfig.reevaluate_strategy_on_new_sighting: everything visible
        # as of this point is already accounted for in strategy_by_force/
        # should_retreat_by_force above -- seeded here so the first later
        # check only reacts to something genuinely new this turn, not to
        # this same snapshot again.
        seen_enemy_ids: set[int] = set(stance_visible_enemies)

        # Scored task-force movement (see tla.ai.move_scoring,
        # AiConfig.scored_task_force_movement_enabled): a prototype
        # replacement for the fixed procedural chain below (carrier
        # scouting -> secure_kills_pass -> cohesion/screening/rearguard/
        # goal-advance), scoped to exactly the forces that chain would
        # otherwise handle -- ADVANCE/AGGRESSIVE, non-retreating. Port/
        # carrier defense above and RETREAT/HOLD forces are untouched
        # either way. Claimed ships are added to resolved from inside
        # _apply_scored_move itself (rather than via this generator's own
        # yield value) so every pass below -- already filtering on `s.id
        # not in resolved` -- automatically skips whatever this one
        # already moved, same claim-and-skip pattern every other pass
        # here uses.
        if game_state.config.ai.scored_task_force_movement_enabled:

            def _apply_scored_move(ship: Ship, destination: AxialCoord) -> None:
                visible = enemy_ships_visible_to(game_state, player)
                if ship.kind == ShipKind.SUBMARINE and _toggle_preserves_destination(ship, destination, game_state):
                    _maybe_toggle_submarine(ship, game_state, visible)
                _execute(ship, destination, game_state, self.decide_battle)
                resolved.add(ship.id)
                if game_state.config.ai.reevaluate_strategy_on_new_sighting:
                    _reevaluate_strategy_on_new_sightings(
                        game_state, player, forces, pace_by_force, model,
                        seen_enemy_ids, strategy_by_force, should_retreat_by_force,
                    )

            for f in forces:
                if f.retreating:
                    continue
                if strategy_by_force.get(f.id, Strategy.ADVANCE) not in (Strategy.ADVANCE, Strategy.AGGRESSIVE):
                    continue
                yield from plan_force_movement_scored(
                    game_state,
                    player,
                    f,
                    model,
                    game_state.config.ai,
                    lambda: enemy_ships_visible_to(game_state, player),
                    _apply_scored_move,
                )

        # Carrier scouting (see AiConfig.carrier_scouting_enabled,
        # _carrier_scouting_advance): before the ordinary per-ship loop,
        # let any carrier that would otherwise just take the flat-capped
        # cautious advance (see _carrier_scouting_eligible -- an already-
        # threatened, retreating, cohesion-needing, or goal-less carrier
        # is left untouched, handled exactly as before by the generic
        # pipeline below) instead advance hex-by-hex, re-verifying safety
        # against freshly recomputed vision after each hop. Must run
        # after update_task_force_stance/update_task_force_goals/
        # strategy_by_force just above -- the eligibility gate needs
        # force.retreating/force.goal/this force's chosen Strategy as
        # finalized for this turn, not their pre-update values.
        if game_state.config.ai.carrier_scouting_enabled:
            carriers = [s for s in game_state.ships_for(player) if s.kind == ShipKind.CARRIER and s.id not in resolved]
            for ship in carriers:
                force = _force_for_ship(forces, ship.id)
                strategy = strategy_by_force.get(force.id, Strategy.ADVANCE) if force is not None else Strategy.ADVANCE
                visible_enemies = enemy_ships_visible_to(game_state, player)
                if not _carrier_scouting_eligible(ship, game_state, force, strategy, visible_enemies):
                    continue
                resolved.add(ship.id)
                _carrier_scouting_advance(ship, force, game_state, player, model, pace_by_force, self.decide_battle)
                if game_state.config.ai.reevaluate_strategy_on_new_sighting:
                    _reevaluate_strategy_on_new_sightings(
                        game_state, player, forces, pace_by_force, model,
                        seen_enemy_ids, strategy_by_force, should_retreat_by_force,
                    )
                yield

        # Tactical layer (see tla.ai.tactics): before the ordinary
        # per-ship loop, find enemy targets no single one of player's
        # ships can solo-secure-kill this turn but that 2+ jointly can,
        # and commit to that instead of each independently picking its
        # own best 1v1 target and splitting fire -- a real user complaint
        # (two battleships vs. two battleships should concentrate fire to
        # sink one, not each damage a different one, since a sunk ship
        # deals zero damage next turn while two survivors both still hit
        # back). Claimed ships are added to `resolved`, same
        # claim-and-skip pattern port/carrier defense already establish.
        def _apply_kill_attack(ship: Ship, attack_hex: AxialCoord) -> None:
            visible = enemy_ships_visible_to(game_state, player)
            if ship.kind == ShipKind.SUBMARINE and _toggle_preserves_destination(ship, attack_hex, game_state):
                _maybe_toggle_submarine(ship, game_state, visible)
            _execute(ship, attack_hex, game_state, self.decide_battle)
            if game_state.config.ai.reevaluate_strategy_on_new_sighting:
                _reevaluate_strategy_on_new_sightings(
                    game_state, player, forces, pace_by_force, model,
                    seen_enemy_ids, strategy_by_force, should_retreat_by_force,
                )

        resolved |= yield from secure_kills_pass(
            game_state, player, enemy_ships_visible_to(game_state, player), game_state.config.ai, _apply_kill_attack
        )

        visible_enemies = enemy_ships_visible_to(game_state, player)
        ships = [s for s in game_state.ships_for(player) if s.id not in resolved]
        # Carriers first: their materially wider vision
        # (FowConfig.port_and_carrier_visibility_radius) means moving one
        # is the likeliest single move to reveal more of the board this
        # turn -- visible_enemies is already recomputed fresh for each
        # ship below, so a later ship in this order already benefits from
        # whatever an earlier one's move revealed; this ordering just
        # makes that benefit actually available before it's needed rather
        # than after.
        ships.sort(key=lambda s: (s.kind != ShipKind.CARRIER, _distance_to_nearest_enemy(s, visible_enemies), s.id))

        for ship in ships:
            if ship.id not in game_state.ships:
                continue  # sunk earlier this turn (e.g. defending a scout's contact)
            visible_enemies = enemy_ships_visible_to(game_state, player)
            if ship.kind == ShipKind.SUBMARINE:
                _maybe_toggle_submarine(ship, game_state, visible_enemies)
            ship_force = _force_for_ship(forces, ship.id)
            strategy = Strategy.ADVANCE if ship_force is None else strategy_by_force.get(ship_force.id, Strategy.ADVANCE)
            destination = _choose_destination(
                ship,
                game_state,
                player,
                visible_enemies,
                forces,
                pace_by_force,
                strategy=strategy,
                resolved=frozenset(resolved),
            )
            if destination is not None and destination != ship.position:
                _execute(ship, destination, game_state, self.decide_battle)
                if game_state.config.ai.reevaluate_strategy_on_new_sighting:
                    _reevaluate_strategy_on_new_sightings(
                        game_state, player, forces, pace_by_force, model,
                        seen_enemy_ids, strategy_by_force, should_retreat_by_force,
                    )
            yield

        record_task_force_progress(game_state, forces)

        # Last: game_state.battle_log still holds exactly this half-turn's
        # own battles here (the game loop clears it right after
        # plan_movement returns) -- the one place combat updates belief.
        self._enemy_model_for(game_state, player).end_turn(game_state)
        self._reconcile_defender_belief(game_state, player)

    def _reconcile_defender_belief(self, game_state: GameState, player: PlayerId) -> None:
        """Being attacked reveals the attacker -- a real gap a replay
        review caught: `EnemyModel` only ever reads `game_state.
        battle_log` for entries where *it's* the attacker (see
        `EnemyModel._observe_own_attacks`'s own docstring on why: the
        half-turn battle_log lifecycle means the *defending* side's own
        `EnemyModel` never runs while this half-turn's battle_log still
        exists, so it could never otherwise learn that one of the
        opponent's ships had just revealed itself by attacking). This
        player owns every attacker in `battle_log` right now (nothing
        else could have generated these entries this half-turn), so their
        opponent's own EnemyModel -- a different instance than the one
        `end_turn` just updated above -- gets a chance to observe each
        attacker directly, the same way a real sighting would, before the
        game loop clears battle_log for the next half-turn."""
        opponent = opponent_of(player, game_state)
        opponent_model = self._enemy_model_for(game_state, opponent)
        for entry in game_state.battle_log:
            if entry.attacker_owner != player:
                continue
            if entry.attacker_sunk:
                opponent_model.observe_ship_sunk(entry.attacker_id, entry.attacker_kind)
                continue
            opponent_model.observe_ship_state(
                entry.attacker_id, entry.attacker_kind, entry.battle_hex, entry.attacker_hp_after, game_state
            )

    def decide_battle(self, attacker: Ship, defender: Ship, game_state: GameState) -> Decision:
        """Called by `battle.run_battle` after every round both ships
        survive. Retreats once the race has turned unfavorable
        (`matchup_score < 0`), or once a tie is no longer a good trade by
        relative ship value (see `scoring.worth_a_tie`) -- but never
        breaks off a fight it's still clearly winning just because its
        own HP has dropped low: a strictly-favorable race
        (`matchup_score > 0`) is, by definition of the race, guaranteed
        to secure the kill and survive regardless of how low that leaves
        the attacker, and a damaged-but-alive enemy left behind deals
        exactly as much damage next time as a full-health one -- so
        breaking off early trades a certain kill for nothing (a real
        replay-found gap, game65: the AI backing off a winning fight and
        leaving a fully-functional enemy alive). Replaces the old,
        unconditional `AiConfig.damaged_withdraw_fraction` HP floor,
        removed entirely -- see `tla.ai.tactics.secures_kill`'s own
        matching simplification, since that floor was the only reason a
        strictly-favorable race could ever fail to finish."""
        score = scoring.matchup_score(attacker, defender, game_state)
        if score < 0:
            return "retreat"
        if score == 0 and not scoring.worth_a_tie(attacker, defender, game_state):
            return "retreat"
        return "stay"


def _ship_at(ships_by_id: dict[int, Ship], coord: AxialCoord) -> Ship | None:
    for ship in ships_by_id.values():
        if ship.position == coord:
            return ship
    return None


def _distance_to_nearest_enemy(ship: Ship, visible_enemies: dict[int, Ship]) -> int:
    if not visible_enemies:
        return _NO_ENEMY_SENTINEL
    return min(distance(ship.position, e.position) for e in visible_enemies.values())


def _step_toward(
    ship: Ship,
    game_state: GameState,
    target: AxialCoord,
    *,
    avoid: frozenset[AxialCoord] = frozenset(),
    max_steps: int | None = None,
) -> AxialCoord | None:
    """The reachable hex (this turn) that gets `ship` closest to `target`
    by actual sea route (see `tla.ai.task_force.sea_distance_field` -- not
    straight-line hex distance, which can rank a hex as "closest" even
    when it's on the wrong side of a landmass from `target`, with no
    shorter real route through it at all than through a nominally
    "farther" hex; left unfixed, that mismatch traps a ship oscillating
    along a coastline instead of ever finding the way around), or None if
    `ship` has nowhere it can go. `avoid` excludes specific hexes from
    consideration -- used to reposition toward/past an enemy without
    accidentally landing exactly on it (an enemy-occupied hex is a legal,
    reachable "stop" per `movement.reachable_hexes`, so without this a
    repositioning move could silently turn into the very attack a caller
    just decided *not* to make). `max_steps`, if given, further restricts
    candidates to those reachable within that many movement points -- used
    to pace a task force's advance to its slowest member (see
    `_compute_force_pace`) rather than always moving as far as this ship
    individually can. Both restrictions fall back to the unfiltered set if
    they'd leave no candidates at all, rather than stranding the ship --
    *except* `max_steps=0` specifically, which always means "don't move"
    (see `Strategy.HOLD`): `movement.reachable_hexes` never includes a
    ship's own current hex (a 0-cost "stop"), so without this special case
    the empty-candidates fallback above would silently ignore the cap
    entirely and move the ship at its ordinary, unpaced pace -- the exact
    opposite of what a caller asking for zero movement means."""
    if max_steps == 0:
        return ship.position
    reachable = movement.reachable_hexes(ship, game_state)
    if not reachable:
        return None
    candidates = {h: c for h, c in reachable.items() if h not in avoid} or reachable
    if max_steps is not None:
        candidates = {h: c for h, c in candidates.items() if c <= max_steps} or candidates
    field = sea_distance_field(game_state, target)
    return min(candidates, key=lambda h: (sea_route_distance(field, h), candidates[h], h))


def _most_threatened_port(
    game_state: GameState, player: PlayerId, visible_enemies: dict[int, Ship]
) -> AxialCoord | None:
    """The controlled port nearest to a visible enemy, if any enemy is
    within twice the normal port vision radius -- a simple "is home under
    threat" check, not a full assignment/matching algorithm."""
    threat_radius = 2 * game_state.config.fow.port_and_carrier_visibility_radius
    best: AxialCoord | None = None
    best_dist: int | None = None
    for port in game_state.board.controlled_ports_for(player):
        nearest = min((distance(port, e.position) for e in visible_enemies.values()), default=None)
        if nearest is not None and nearest <= threat_radius and (best_dist is None or nearest < best_dist):
            best, best_dist = port, nearest
    return best


def _favorable_attack(
    ship: Ship,
    game_state: GameState,
    visible_enemies: dict[int, Ship],
    attacker_group: list[Ship],
    *,
    tie_tolerant: bool = False,
) -> AxialCoord | None:
    """A reachable, currently-visible enemy hex worth attacking this turn,
    or None. Among every reachable enemy-occupied hex with a favorable
    `matchup_score` -- strictly (`> 0`), or a tie (`== 0`, since combat
    damage is simultaneous, see tla.battle.resolve_round, so a tied race
    is actually a *mutual kill*, not a clean win) accepted only when it's
    actually worth it: either `tie_tolerant=True` (see `Strategy.
    AGGRESSIVE`, the same relaxation `_carrier_defense_destination`'s own
    tie-tolerant fallback already uses for a different doctrine,
    generalized here into a reusable parameter instead of a second,
    separate copy -- accepts *any* tie, deliberately reckless), or
    `scoring.worth_a_tie` says the target's own replacement cost covers
    this ship's -- trading a cheap ship for an expensive one is a good
    deal even 1-for-1, which a flat "never accept a tie" rule missed (a
    real replay-found gap, game65: cruisers declining clearly-worthwhile
    mutual-kill trades against carriers). Picks the one with the
    best `tla.ai.tactics.action_value` (matchup_score plus a bonus when
    this specific attack would secure the target's kill this turn,
    weighted by the firepower sinking it removes -- see that function's
    own docstring; with `AiConfig.tactics_enabled` off, this is just
    `matchup_score` again, i.e. the prior behavior). The winning
    candidate must also not have enough nearby backup to outmatch
    `attacker_group` (`attacker_group`'s own task force for a member, or
    just `[ship]` for an unassigned one -- see `tla.ai.task_force.
    is_outmatched_by_target_support`), narrowed first to members within
    `AiConfig.task_force_threat_radius` of the target (`ship` itself is
    always kept regardless of distance -- it's the one actually about to
    be there, not backup). Without this narrowing, a straggler still
    hexes away racing to catch up to its own force (see `task_force_max_
    separation`'s own straggler doctrine) counted its full strength as
    "available support" for a fight breaking out right now elsewhere --
    a real replay-found bug (game64 turn 5): a battleship detached alone
    into a 9-ship enemy concentration because its whole 8-member force's
    *aggregate* power looked sufficient, even though only that one
    battleship, isolated, was actually anywhere near the fight. Unaffected
    by `tie_tolerant`, which only relaxes the raw 1v1 race, not this
    backup-force safety check. Shared by `_choose_generic_destination` and
    `choose_task_force_destination` so both agree on when a ship fights
    versus keeps maneuvering -- a force member still takes a favorable
    fight it stumbles across, same as an unassigned ship would."""
    if not visible_enemies:
        return None
    ai_config = game_state.config.ai
    candidates = scoring.reachable_attack_candidates(ship, game_state, visible_enemies)

    def _is_favorable(coord: AxialCoord) -> bool:
        target = _ship_at(visible_enemies, coord)
        score = scoring.matchup_score(ship, target, game_state)
        if score > 0:
            return True
        if score == 0:
            return tie_tolerant or scoring.worth_a_tie(ship, target, game_state)
        return False

    favorable = [(coord, cost) for coord, cost in candidates if _is_favorable(coord)]
    if not favorable:
        return None

    def key(item: tuple[AxialCoord, int]) -> tuple[float, int, AxialCoord]:
        coord, cost = item
        return (-action_value(ship, _ship_at(visible_enemies, coord), game_state, ai_config), cost, coord)

    attack_hex = min(favorable, key=key)[0]
    defender = _ship_at(visible_enemies, attack_hex)
    nearby_group = [
        m
        for m in attacker_group
        if m.id == ship.id or distance(m.position, defender.position) <= ai_config.task_force_threat_radius
    ]
    if is_outmatched_by_target_support(nearby_group, defender, game_state, visible_enemies, ai_config):
        return None
    return attack_hex


def choose_task_force_destination(
    ship: Ship,
    goal: TaskForceGoal,
    game_state: GameState,
    visible_enemies: dict[int, Ship],
    force: TaskForce,
    max_steps: int | None = None,
    strategy: Strategy = Strategy.ADVANCE,
) -> AxialCoord | None:
    """A force member's destination given its force's current `goal` --
    still takes a favorable fight it runs into (see `_favorable_attack`,
    using the rest of `force` as its attacker group), but otherwise
    advances toward the goal instead of chasing the nearest enemy or
    independently expanding toward the nearest uncontrolled port the way
    an unassigned ship (`_choose_generic_destination`) would. `max_steps`
    (see `_compute_force_pace`) paces that advance to the force's slowest
    member instead of always moving this ship as far as it individually
    can -- never applied while retreating (get home without dawdling).

    A `RETREAT` goal never even considers attacking -- "retreat and wait
    for reinforcements" means actually disengaging, not fighting anything
    easy-looking on the way to the rally point. Its target is also a
    *friendly* port (see `tla.ai.task_force.rally_point`) to fall back
    near, not to occupy -- excluded from the reachable candidates so the
    ship settles adjacent to it instead of landing squarely on the hex,
    which would otherwise block that port from producing anything for as
    long as the retreat lasts. Every other goal kind (CAPTURE_PORT,
    BLOCKADE, DEFEND_PORT) targets a hex the ship is actually meant to
    occupy, so it's a legal destination as normal.

    A destroyer/patrol boat (`_ESCORT_KINDS`) advancing toward a non-RETREAT
    goal alongside at least one capital ship (`_CAPITAL_KINDS`) trails
    behind it instead of marching straight for the goal itself -- see
    `_rearguard_target`. If `AiConfig.task_force_max_separation` is set,
    that takes priority over even this: no member (any kind) gets to end
    its move farther than that many hexes from any living non-submarine
    force-mate -- see `_cohesion_destination`. Ahead of the rearguard check
    but behind that general cohesion cap, two more corrections can
    override the ordinary goal-directed advance: first (`AiConfig.
    carrier_screen_enabled`), an eligible battleship/cruiser or destroyer/
    submarine needed as a standing screen between one of the force's
    carriers and a nearby same-weight-class enemy -- see `_carrier_
    screen_destination`, a live threat taking priority over the coverage
    concern just after it; then an AC-bonus-eligible member
    (`AC_BONUS_ELIGIBLE_KINDS`, minus the carrier itself) that's drifted
    outside `CombatConfig.ac_bonus_radius` of every living carrier in its
    own force gets recalled into bonus range instead -- see
    `_carrier_bonus_cohesion_destination`: the general cap alone can't
    guarantee this, since `task_force_max_separation` is typically looser
    than the bonus radius, and a fight this ship is actually in only gets
    the bonus if it's still close enough to a carrier when that fight
    happens.

    `strategy` (see the `Strategy` enum and `_pick_strategy_for_force`)
    lets a caller ask "what would this ship do under a different force-wide
    stance this turn" without touching `force`/`goal` themselves --
    `Strategy.RETREAT` disengages toward `rally_point(force, game_state)`
    even when `goal.kind` isn't actually `RETREAT` yet (the hypothetical
    case: scoring what retreating *would* look like, before `force.
    retreating` is ever set), `Strategy.AGGRESSIVE` relaxes `_favorable_
    attack`'s gate to accept a tied (mutual-kill) trade, and the default
    `Strategy.ADVANCE` is exactly today's behavior. `Strategy.HOLD` needs
    no special case here -- it's expressed entirely by the caller passing
    `max_steps=0`, which the cohesion/rearguard fallback below already
    honors (or ignores, for cohesion, by design -- see above)."""
    enemy_hexes = frozenset(e.position for e in visible_enemies.values())
    if goal.kind == GoalKind.RETREAT:
        return _step_toward(ship, game_state, goal.target, avoid=enemy_hexes | {goal.target})
    if strategy == Strategy.RETREAT:
        retreat_target = rally_point(force, game_state)
        if retreat_target is not None:
            return _step_toward(ship, game_state, retreat_target, avoid=enemy_hexes | {retreat_target})
    attacker_group = [game_state.ships[i] for i in force.member_ids if i in game_state.ships]
    attack_hex = _favorable_attack(
        ship, game_state, visible_enemies, attacker_group, tie_tolerant=strategy == Strategy.AGGRESSIVE
    )
    if attack_hex is not None:
        return attack_hex
    cohesion_destination = _cohesion_destination(ship, force, game_state, goal.target, enemy_hexes)
    if cohesion_destination is not None:
        return cohesion_destination
    screen_destination = _carrier_screen_destination(ship, force, game_state, enemy_hexes, visible_enemies)
    if screen_destination is not None:
        return screen_destination
    bonus_cohesion_destination = _carrier_bonus_cohesion_destination(
        ship, force, game_state, goal.target, enemy_hexes
    )
    if bonus_cohesion_destination is not None:
        return bonus_cohesion_destination
    if ship.kind in _ESCORT_KINDS:
        rear_target = _rearguard_target(ship, force, game_state, goal.target)
        if rear_target is not None:
            return _step_toward(ship, game_state, rear_target, avoid=enemy_hexes, max_steps=max_steps)
    return _step_toward(ship, game_state, goal.target, avoid=enemy_hexes, max_steps=max_steps)


def _established_force_members(members: list[Ship], max_sep: int) -> set[int]:
    """The largest cluster of `members` connected by pairwise distance
    <= `max_sep` (union-find over that "close enough" edge, transitively
    -- so a chain of several such hops still counts as one cluster, not
    just direct pairs) -- these are the ships `_cohesion_destination`
    treats as "already part of the formation." Anyone outside this
    cluster (most commonly a single freshly produced ship still steaming
    to catch up after `tla.ai.task_force`'s open-force recruitment picked
    it up) is a straggler for this turn: still pulled toward the cluster
    via `_cohesion_destination`'s own fallback, but never itself anchors
    an established member backward -- the user's own reported doctrine
    ("new ships steam at max speed to join the TF, but the TF just
    continues on its own plan without waiting for them"), replacing the
    prior symmetric rule where *every* member (a brand-new recruit
    included) equally anchored every other, which could snap an already-
    advanced formation back toward a ship that had only just spawned at a
    home port and joined the force that same turn.

    Ties broken by the cluster containing the lowest ship id, for
    determinism. If no two members are within `max_sep` of *any* other
    member at all (every cluster is a singleton -- nobody's actually
    clustered with anybody yet), returns every id instead of picking an
    arbitrary singleton "winner": degenerates to the prior plain pairwise
    behavior rather than declaring the whole force stragglers with
    nothing to cohere around."""
    ids = [m.id for m in members]
    position = {m.id: m.position for m in members}
    parent = {i: i for i in ids}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if distance(position[a], position[b]) <= max_sep:
                union(a, b)

    clusters: dict[int, list[int]] = {}
    for i in ids:
        clusters.setdefault(find(i), []).append(i)

    if all(len(cluster) <= 1 for cluster in clusters.values()):
        return set(ids)

    best = max(clusters.values(), key=lambda cluster: (len(cluster), -min(cluster)))
    return set(best)


def _cohesion_destination(
    ship: Ship,
    force: TaskForce,
    game_state: GameState,
    goal_target: AxialCoord,
    enemy_hexes: frozenset[AxialCoord],
) -> AxialCoord | None:
    """`ship`'s actual destination this turn if `AiConfig.task_force_max_
    separation` needs enforcing, or None (fall through to the ordinary
    goal-directed logic) if it isn't set, or `ship` has no established
    force-mates to keep pace with.

    Unlike `_rearguard_target` (a *target* fed through `_step_toward`,
    trusting its ordinary "closest reachable hex" pace-respecting
    selection), this picks the actual destination directly out of `ship`'s
    own `reachable_hexes` this turn: every candidate that would leave
    `ship` farther than `task_force_max_separation` from *any* established
    force-mate (`_established_force_members` -- the force's largest
    already-clustered group; a straggler not yet within range of that
    cluster is excluded as an anchor, see that function's own docstring
    for why) is excluded outright, and the best remaining one (by real
    sea-route progress toward `goal_target`) is returned. This has to
    check the *result* of the move, not just the ship's current position
    before moving -- an earlier version only checked "am I already too
    far from someone right now," which let a ship right at the boundary
    use its full movement allowance and end up past the cap by the time
    it actually stopped, since nothing about a pre-move check accounts
    for the move itself.

    If literally no reachable hex keeps every distance within the cap
    (e.g. a straggler too far behind to close the gap in one move), falls
    back to whichever reachable hex minimizes the worst remaining
    distance instead of leaving the group in the ordinary goal direction
    -- a step toward compliance instead of no constraint at all. This is
    also what actually moves a straggler itself toward the cluster: its
    own candidate call still measures against the same established
    anchors (everyone else in the force, cluster or not, is irrelevant to
    *its* computation only in the sense that a fellow straggler can't
    substitute as a target to converge on).

    Checked first and takes priority over `_rearguard_target`, since this
    is a hard cap meant to apply to every established member, not just an
    escort's qualitative "don't get ahead." Excludes any hex a currently-
    visible enemy occupies from consideration -- cohesion should never
    walk `ship` into a fight `_favorable_attack` already declined a moment
    earlier.

    A submarine is now included on both sides of this check (pulled back
    like any other member, and counted as an anchor for its force-mates
    too) when `AiConfig.submarine_task_force_cohesion` is set (the
    default) -- see `_compute_force_pace`'s own docstring for the doctrine
    this serves. `False` restores the prior exemption (a submarine
    neither pulled nor counted as an anchor)."""
    ai_config = game_state.config.ai
    exempt_submarine = not ai_config.submarine_task_force_cohesion
    if ai_config.task_force_max_separation is None or (exempt_submarine and ship.kind == ShipKind.SUBMARINE):
        return None
    max_sep = ai_config.task_force_max_separation
    all_members = [
        game_state.ships[i]
        for i in force.member_ids
        if i in game_state.ships
        and not (exempt_submarine and game_state.ships[i].kind == ShipKind.SUBMARINE)
    ]
    established_ids = _established_force_members(all_members, max_sep)
    others = [m for m in all_members if m.id != ship.id and m.id in established_ids]
    if not others:
        return None
    if all(distance(ship.position, o.position) <= max_sep for o in others):
        return None
    reachable = {h: c for h, c in movement.reachable_hexes(ship, game_state).items() if h not in enemy_hexes}
    if not reachable:
        return None
    in_bounds = [h for h in reachable if all(distance(h, o.position) <= max_sep for o in others)]
    if not in_bounds:
        return min(reachable, key=lambda h: (max(distance(h, o.position) for o in others), h))
    field = sea_distance_field(game_state, goal_target)
    return min(in_bounds, key=lambda h: (sea_route_distance(field, h), h))


def _carrier_bonus_cohesion_destination(
    ship: Ship,
    force: TaskForce,
    game_state: GameState,
    goal_target: AxialCoord,
    enemy_hexes: frozenset[AxialCoord],
) -> AxialCoord | None:
    """`ship`'s actual destination this turn if it's an AC-bonus-eligible
    non-carrier member (`AC_BONUS_ELIGIBLE_KINDS` minus the carrier itself
    -- battleship, cruiser, destroyer) that's drifted outside
    `CombatConfig.ac_bonus_radius` of every living carrier in its own
    force, or that's about to drift outside it via its own ordinary
    goal-directed step this turn -- None (fall through to `_rearguard_
    target`/ordinary goal-directed logic) if `ship` isn't an eligible
    kind, the force has no living carrier, or neither of those applies.

    This exists because `_cohesion_destination`'s general `task_force_max_
    separation` cap is normally set looser than `ac_bonus_radius` (it's
    meant to catch a badly strung-out straggler, not enforce the tight
    radius the carrier bonus actually needs) -- a member can easily sit
    comfortably inside that general cap while still too far from the
    carrier to get any bonus in a fight it's actually in, or to give the
    carrier's own defense any benefit from its presence. This is that
    tighter, mechanic-specific version of the same idea, checked
    afterward so a member already flagged as a genuine straggler gets
    reeled back into the group as a whole first.

    Being *currently* within radius doesn't skip the check outright: a
    real game showed several battleships that started a turn in bonus
    range ending it 3-4 hexes from their own carrier, having drifted out
    during their own unconstrained goal-directed advance with nothing to
    catch it, since an earlier version of this function only ever
    reacted to an *existing* violation. So a ship starting in range still
    gets a cheap trial step (`_step_toward`, unconstrained) to see where
    its ordinary advance would actually land it -- only if that lands
    outside radius (or nowhere) does the constrained logic below kick in;
    otherwise this still returns None; there's nothing to correct.

    Same shape as `_cohesion_destination` throughout, for the same
    reasons: picks the actual destination directly out of `ship`'s own
    `reachable_hexes`, has to check the *result* of each candidate move
    (not just `ship`'s position before moving, which a ship right at the
    boundary could move past), excludes currently-visible-enemy hexes
    (this should never walk `ship` into a fight `_favorable_attack`
    already declined), and falls back to minimizing the worst remaining
    distance if no reachable hex gets back into range in one move."""
    if ship.kind not in AC_BONUS_ELIGIBLE_KINDS or ship.kind == ShipKind.CARRIER:
        return None
    carriers = [
        game_state.ships[i]
        for i in force.member_ids
        if i in game_state.ships and game_state.ships[i].kind == ShipKind.CARRIER
    ]
    if not carriers:
        return None
    radius = game_state.config.combat.ac_bonus_radius

    def in_range(coord: AxialCoord) -> bool:
        return any(distance(coord, c.position) <= radius for c in carriers)

    if in_range(ship.position):
        trial = _step_toward(ship, game_state, goal_target, avoid=enemy_hexes)
        if trial is None or in_range(trial):
            return None
    reachable = {h: c for h, c in movement.reachable_hexes(ship, game_state).items() if h not in enemy_hexes}
    if not reachable:
        return None
    in_bounds = [h for h in reachable if in_range(h)]
    if not in_bounds:
        return min(reachable, key=lambda h: (min(distance(h, c.position) for c in carriers), h))
    field = sea_distance_field(game_state, goal_target)
    return min(in_bounds, key=lambda h: (sea_route_distance(field, h), h))


def _carrier_screen_destination(
    ship: Ship,
    force: TaskForce,
    game_state: GameState,
    enemy_hexes: frozenset[AxialCoord],
    visible_enemies: dict[int, Ship],
) -> AxialCoord | None:
    """`ship`'s actual destination this turn if it's needed as a standing
    screen between one of `force`'s carriers and a nearby enemy of the
    matching weight class -- None (fall through to `_carrier_bonus_
    cohesion_destination`/ordinary logic) if `AiConfig.carrier_screen_
    enabled` is off, `ship` isn't an eligible screening kind, the force
    has no living carrier, or no correction is needed.

    Two independent weight classes, each with its own own-kind screener
    and radius: an own battleship/cruiser (`_HEAVY_SCREEN_KINDS`) screens
    a visible enemy battleship/cruiser within `AiConfig.carrier_heavy_
    screen_radius` of the nearest carrier; an own destroyer/submarine
    (`_LIGHT_SCREEN_KINDS`) screens a visible, *surfaced* enemy destroyer/
    submarine (a submerged enemy sub isn't in `visible_enemies` under FOW
    anyway -- the check mostly documents intent) within `AiConfig.
    carrier_light_screen_radius`. `ship`'s own kind picks which class (if
    either) applies -- a destroyer never screens a battleship threat, a
    battleship never screens a destroyer threat.

    "Already screened" means an own eligible-kind force-mate's plain hex
    distance to that specific threat is strictly less than the target
    carrier's own distance to it -- genuinely standing in front, not just
    somewhere nearby. Cheap raw `distance`, matching `_carrier_bonus_
    cohesion_destination`'s own precedent for range checks right above --
    real sea-route pathing (`block_hex_toward`) is reserved for actually
    picking the destination once a correction is needed, passing
    `CombatConfig.ac_bonus_radius` as `prefer_radius` so the chosen
    blocking hex also keeps the screen in the carrier's own bonus range
    when the threat's route actually offers such a point (see `block_hex_
    toward`'s own docstring -- a real replay case, game65 turn 2, showed
    the plain "next step on the threat's route" choice sending a screen
    three hexes from its carrier when a closer, equally-blocking hex was
    available on the same path). No batch
    assignment: each eligible ship checks for itself when it's its own
    turn to move, and this codebase's real sequential per-ship execution
    (each move actually happens before the next ship is even decided) is
    what stops two ships both redirecting to screen the same gap -- by
    the time a second eligible ship is evaluated, first one's already-
    executed move already counts as "already screened" if it worked.

    Born from a replay review (game65): compute_carrier_defense_
    directives swept nearly the AI's whole fleet into ad hoc "close the
    distance on the threat" duty the instant one enemy battleship came
    within range of a carrier, each ship moving independently with no
    coordination -- a destroyer ended up exposed in front while two
    battleships ended up too far away to matter, and the carriers (moving
    last, since nearly everyone else had already been claimed) got
    caught with no real screen and lost two of three in one turn."""
    ai_config = game_state.config.ai
    if not ai_config.carrier_screen_enabled:
        return None
    if ship.kind in _HEAVY_SCREEN_KINDS:
        screen_kinds = _HEAVY_SCREEN_KINDS
        threat_kinds = _HEAVY_SCREEN_KINDS
        radius = ai_config.carrier_heavy_screen_radius
    elif ship.kind in _LIGHT_SCREEN_KINDS:
        screen_kinds = _LIGHT_SCREEN_KINDS
        threat_kinds = _LIGHT_SCREEN_KINDS
        radius = ai_config.carrier_light_screen_radius
    else:
        return None

    carriers = [
        game_state.ships[i]
        for i in force.member_ids
        if i in game_state.ships and game_state.ships[i].kind == ShipKind.CARRIER
    ]
    if not carriers:
        return None

    def nearest_carrier(pos: AxialCoord) -> Ship:
        return min(carriers, key=lambda c: (distance(pos, c.position), c.id))

    threats = [
        e
        for e in visible_enemies.values()
        if e.kind in threat_kinds
        and (e.kind != ShipKind.SUBMARINE or e.surfaced)
        and distance(e.position, nearest_carrier(e.position).position) <= radius
    ]
    if not threats:
        return None

    own_screens = [
        game_state.ships[i]
        for i in force.member_ids
        if i in game_state.ships and i != ship.id and game_state.ships[i].kind in screen_kinds
    ]

    def is_screened(threat: Ship) -> bool:
        carrier = nearest_carrier(threat.position)
        return any(
            distance(screen.position, threat.position) < distance(carrier.position, threat.position)
            for screen in own_screens
        )

    unscreened = [t for t in threats if not is_screened(t)]
    if not unscreened:
        return None

    nearest_threat = min(unscreened, key=lambda t: distance(t.position, nearest_carrier(t.position).position))
    target_carrier = nearest_carrier(nearest_threat.position)
    field = sea_distance_field(game_state, target_carrier.position)
    block_hex = block_hex_toward(
        [nearest_threat],
        target_carrier.position,
        field,
        game_state,
        prefer_radius=game_state.config.combat.ac_bonus_radius,
    )
    if block_hex is None:
        return None
    return _step_toward(ship, game_state, block_hex, avoid=enemy_hexes)


def _rearguard_target(
    ship: Ship, force: TaskForce, game_state: GameState, goal_target: AxialCoord
) -> AxialCoord | None:
    """Where a destroyer/patrol boat escorting `force`'s capital ships
    should head to trail behind them -- None if there's no living capital
    ship in the force to escort, or if `ship` is already at or behind the
    rearmost one (nothing to trail further; just advance normally).

    "Behind" is simply "whichever capital ship has made the least sea-
    route progress toward the goal" -- targeting *that ship's own current
    position*, not the goal, means the escort is always chasing where the
    capital ship *was*, one step behind, re-aimed fresh every turn as the
    capital ship keeps advancing: a trailing formation with no path-
    prediction or extra state needed. Replaces the old scout-ahead
    mechanism (send a scout to clear a hex before a capital ship enters
    it) -- deemed too cumbersome for what it bought. The idea now: if a
    submerged sub ambushes the lead capital ship, it can retreat (see
    `NaivePolicy.decide_battle`) and the trailing escort -- close by,
    already positioned -- is what actually deals with the now-revealed
    sub, rather than trying to prevent the ambush in the first place."""
    capital_ships = [
        game_state.ships[i]
        for i in force.member_ids
        if i in game_state.ships and game_state.ships[i].kind in _CAPITAL_KINDS
    ]
    if not capital_ships:
        return None
    field = sea_distance_field(game_state, goal_target)
    ship_distance = sea_route_distance(field, ship.position)
    rearmost = max(capital_ships, key=lambda s: (sea_route_distance(field, s.position), s.id))
    if ship_distance >= sea_route_distance(field, rearmost.position):
        return None
    return rearmost.position


def _force_for_ship(forces: list[TaskForce], ship_id: int) -> TaskForce | None:
    for force in forces:
        if ship_id in force.member_ids:
            return force
    return None


def _compute_force_pace(game_state: GameState, forces: list[TaskForce]) -> dict[int, int]:
    """Each force's shared advance pace for this turn: the lowest current
    `movement_remaining` among its members. Call once per player at the
    very top of `plan_movement`, before anything has moved this turn --
    every member's `movement_remaining` is still its full, unspent budget
    for the turn at that point, so the result is stable for the whole
    turn regardless of what order ships end up processed in afterward (an
    already-moved member's now-reduced budget must never leak into this).

    Used so battleships/carriers/etc. don't race ahead of a slower escort
    while advancing toward a goal -- the group arrives together for
    mutual support, and a capital ship's vision/carrier-bonus radius ends
    up actually where the fighting happens instead of trailing behind it.

    A submarine now participates like any other member when `AiConfig.
    submarine_task_force_cohesion` is set (the default) -- `Ship.
    max_movement` already gives it the right budget for whichever state
    it's in, so a surfaced sub (movement comparable to the rest of the
    fleet) rarely bottlenecks this, while a submerged one (a crawl)
    correctly becomes the pace, deliberately slowing the whole force to
    stay with it -- the user's own tested doctrine: an attacking enemy is
    then likely to end up fighting the one ship that barely takes damage,
    not picking off the fleet's real firepower (see `_cohesion_
    destination`, which keeps the sub from drifting out of that formation
    in the first place). `False` restores the prior exemption (excluded
    entirely, for the same reason a submarine is also never capped by the
    result -- see `_choose_destination`) -- found via replay review after
    an unconstrained sub raced ahead of its own force and was picked off
    alone."""
    exempt_submarine = not game_state.config.ai.submarine_task_force_cohesion
    pace: dict[int, int] = {}
    for force in forces:
        members = [
            game_state.ships[i]
            for i in force.member_ids
            if i in game_state.ships and not (exempt_submarine and game_state.ships[i].kind == ShipKind.SUBMARINE)
        ]
        if members:
            pace[force.id] = min(m.movement_remaining for m in members)
    return pace


def _choose_destination(
    ship: Ship,
    game_state: GameState,
    player: PlayerId,
    visible_enemies: dict[int, Ship],
    forces: list[TaskForce],
    pace_by_force: dict[int, int],
    strategy: Strategy = Strategy.ADVANCE,
    resolved: frozenset[int] = frozenset(),
) -> AxialCoord | None:
    """Dispatches to the right per-ship destination logic, task-force
    membership included -- the single per-ship decision point `plan_
    movement`'s main loop calls for every ship not already claimed by
    port defense.

    A force currently retreating (`retreating`, see
    `tla.ai.task_force.update_task_force_stance`) has its real `goal`
    shadowed here with an ephemeral `RETREAT` goal pointed at
    `force.retreat_waypoint` if set (a capped pullback short of the rally
    port -- see `AiConfig.retreat_pullback_hexes`), else `rally_point`
    itself -- `force.goal` itself is never touched, so this is purely a
    per-call override; once the retreat ends the force resumes advancing
    toward its real goal with no memory needed of the interruption. If
    neither is available (owner controls no ports), the real goal is used
    unchanged -- nowhere better to retreat to.

    `pace_by_force` (see `_compute_force_pace`) is only ever handed down
    as `max_steps` for a force member actually advancing toward a real
    goal, never while the shadowed goal is `RETREAT` (get home without
    dawdling, see `choose_task_force_destination`). A submarine is capped
    by it too when `AiConfig.submarine_task_force_cohesion` is set (the
    default -- see `_compute_force_pace`'s own docstring for why); `False`
    exempts it (moves at its own pace regardless), matching the prior,
    unconditional behavior.

    `strategy` (see the `Strategy` enum) is only ever meaningful for a ship
    that belongs to a force -- an unassigned ship always gets `_choose_
    generic_destination`'s ordinary behavior regardless. `Strategy.HOLD`
    zeroes `max_steps` here (before it's handed to either destination
    function below) rather than inside them, since it's the one strategy
    that's purely "the same plan, less movement" and applies identically to
    every member kind; the other three strategies change the fight/retreat
    decision itself, which only `choose_task_force_destination`/`_choose_
    carrier_destination` have enough context to do -- see their own
    docstrings.

    `resolved` (the set of ship ids already claimed and moved for real
    this turn by an earlier pass -- port defense, carrier defense,
    `secure_kills_pass`) is only actually consulted by `_carrier_
    formation_destination` (via `_choose_carrier_destination`), to score
    an escort at its real, already-decided position instead of a
    fictional re-dry-run of a move it will never make. Defaulted to an
    empty set for every caller that doesn't need this (e.g. `_project_
    engagement_value`'s own dry run, which already documents reading live
    state throughout)."""
    force = _force_for_ship(forces, ship.id)
    goal = force.goal if force is not None else None
    if force is not None and force.retreating:
        target = force.retreat_waypoint if force.retreat_waypoint is not None else rally_point(force, game_state)
        if target is not None:
            goal = TaskForceGoal(kind=GoalKind.RETREAT, target=target)
    max_steps = None
    exempt_submarine = not game_state.config.ai.submarine_task_force_cohesion
    if (
        force is not None
        and goal is not None
        and goal.kind != GoalKind.RETREAT
        and not (exempt_submarine and ship.kind == ShipKind.SUBMARINE)
    ):
        max_steps = pace_by_force.get(force.id)
    if force is not None and strategy == Strategy.HOLD and goal is not None and goal.kind != GoalKind.RETREAT:
        max_steps = 0
    if ship.kind == ShipKind.CARRIER:
        return _choose_carrier_destination(
            ship,
            game_state,
            player,
            visible_enemies,
            goal=goal,
            max_steps=max_steps,
            force=force,
            strategy=strategy,
            resolved=resolved,
        )
    if goal is not None and force is not None:
        return choose_task_force_destination(
            ship, goal, game_state, visible_enemies, force, max_steps=max_steps, strategy=strategy
        )
    return _choose_generic_destination(ship, game_state, player, visible_enemies)


def _weighted_damage(ship: Ship, ai_config: AiConfig, stats: dict[ShipKind, ShipStats]) -> float:
    """A ship's value for `_project_engagement_value`'s formula: its own
    damage stat, plus `AiConfig.carrier_assist_value` if it's a carrier
    -- losing one removes the assist bonus it was granting nearby ships
    too, not just its own modest attack stat, and sinking an enemy
    carrier removes that same assist from their side."""
    value = stats[ship.kind].damage
    if ship.kind == ShipKind.CARRIER:
        value += ai_config.carrier_assist_value
    return value


def _project_engagement_value(
    force: TaskForce,
    game_state: GameState,
    player: PlayerId,
    visible_enemies: dict[int, Ship],
    forces: list[TaskForce],
    pace_by_force: dict[int, int],
    model: EnemyModel,
    strategy: Strategy = Strategy.ADVANCE,
) -> float:
    """(carrier-weighted damage this force could secure-kill this turn,
    from current positions -- reachability already reflects "if we
    advance/engage as normal") minus (carrier-weighted damage this
    force's own members are exposed to losing next turn, from their
    *projected* end-of-turn positions) -- as a fraction of this force's
    own total carrier-weighted firepower. 0.0 (a no-op, never triggers
    `tla.ai.task_force.update_task_force_stance`'s new retreat trigger)
    if the force has no members or no enemies are visible at all --
    nothing to project.

    Projected positions come from dry-running `_choose_destination` for
    each member -- the exact same logic that will really move them,
    called without ever executing the result via `_execute`, so this is
    always "what the AI would actually do," never a separate, driftable
    approximation of it. A known simplification: carrier-bonus context
    (which *other* nearby ships grant an assist) is read from the real,
    current `game_state` throughout, not each other member's own
    projected position -- getting that exactly right would need a full
    scratch-state copy, a bigger architectural change deliberately out
    of scope for this first pass; a force's own carriers rarely move far
    enough in one turn to flip bonus-radius membership anyway.

    `strategy` (see the `Strategy` enum) is threaded through only into the
    projected-position dry run below -- the "gained" half above is based on
    current, not projected, positions/reachability, so which strategy is
    being scored can't affect it; only the "at risk" half, which depends on
    where members actually end up, changes per candidate. This is what lets
    `_pick_strategy_for_force` score all four strategies for the same
    force and compare them directly.

    `model` (this force owner's own `EnemyModel`) feeds the at-risk half a
    second, fuzzier threat source alongside `visible_enemies` -- see the
    at-risk loop below for `AiConfig.probable_threat_engagement_enabled`'s
    exact meaning."""
    ai_config = game_state.config.ai
    stats = game_state.config.ship_stats.stats
    members = [game_state.ships[i] for i in force.member_ids if i in game_state.ships]
    if not members or not visible_enemies:
        return 0.0
    total_value = sum(_weighted_damage(m, ai_config, stats) for m in members)
    if total_value <= 0:
        return 0.0

    # Gained: enemy ships this force could secure-kill this turn (solo or
    # jointly among its own members only -- see tla.ai.tactics.
    # secures_kill/would_secure_kill, the same exact-combat-math
    # tla.ai.tactics.secure_kills_pass already uses, reused here rather
    # than re-derived).
    candidates: dict[int, list[Ship]] = {}
    enemy_by_position = {e.position: e for e in visible_enemies.values()}
    for m in members:
        for coord, _cost in scoring.reachable_attack_candidates(m, game_state, visible_enemies):
            target = enemy_by_position[coord]
            candidates.setdefault(target.id, []).append(m)
    gained = 0.0
    confirmed_kills: set[int] = set()
    for target_id, attackers in candidates.items():
        target = visible_enemies[target_id]
        secured = any(secures_kill(a, target, game_state) for a in attackers)
        if not secured and len(attackers) >= 2:
            secured = would_secure_kill(target, attackers, game_state) is not None
        if secured:
            gained += _weighted_damage(target, ai_config, stats)
            confirmed_kills.add(target_id)

    # Projected positions -- dry run, never executed.
    projected = {
        m.id: (
            _choose_destination(m, game_state, player, visible_enemies, forces, pace_by_force, strategy=strategy)
            or m.position
        )
        for m in members
    }

    # At risk: could a visible enemy reach a member's *projected*
    # position next turn and secure a kill against it there? Excludes any
    # enemy this same turn's plan already confirms sinking (see
    # confirmed_kills above) -- a ship that's dead this turn can't act
    # next turn regardless of anything else about its reach or matchup.
    # AiConfig.probable_threat_engagement_enabled additionally treats a
    # believed-but-never-sighted enemy concentration (model.expected_
    # strength_near, around this same projected position) the same way, a
    # second, fuzzier threat source alongside real sightings -- a force
    # advancing into fog toward a real production buildup we've never
    # actually seen shouldn't look any safer than one advancing on a
    # visible fleet of equal strength. believed_damage >= hypothetical.
    # current_hp mirrors tla.ai.tactics.secures_kill's own closed-form
    # "a guaranteed one-round kill needs no decide_battle check" logic as
    # closely as a fuzzy, position-only belief can (there's no real
    # attacker identity here for a retreat-mid-fight check to apply to).
    # kinds=None (all kinds), not DANGEROUS_TO_CARRIER_KINDS -- this is
    # general risk to any of our ships, not carrier-specific defense.
    # Never double-counts a visible responder's own threat: a currently-
    # visible ship is structurally absent from model's diffusing pools/
    # tracked fields at the same time (see EnemyModel's own module
    # docstring), so its threat is only ever counted once, via
    # secures_kill above.
    at_risk = 0.0
    responders = [e for e in visible_enemies.values() if e.id not in confirmed_kills]
    for m in members:
        hypothetical = replace(m, position=projected[m.id])
        visible_threat = any(
            hypothetical.position in enemy_reachable_next_turn(e, game_state)
            and secures_kill(e, hypothetical, game_state)
            for e in responders
        )
        probable_threat = False
        if not visible_threat and ai_config.probable_threat_engagement_enabled:
            _, believed_damage = model.expected_strength_near(hypothetical.position, ai_config.probable_threat_radius)
            probable_threat = believed_damage >= hypothetical.current_hp
        if visible_threat or probable_threat:
            at_risk += _weighted_damage(m, ai_config, stats)

    return (gained - at_risk) / total_value


def _pick_strategy_for_force(
    force: TaskForce,
    game_state: GameState,
    player: PlayerId,
    visible_enemies: dict[int, Ship],
    forces: list[TaskForce],
    pace_by_force: dict[int, int],
    model: EnemyModel,
) -> tuple[Strategy, bool]:
    """(the `Strategy` to use for `force` this turn, whether that choice
    means entering retreat) -- see `plan_movement`, which calls this once
    per non-retreating force each turn and feeds the second value into
    `tla.ai.task_force.update_task_force_stance` as this force's entry into
    `should_retreat_by_force`.

    Only actually evaluates the four candidates when the outcome looks
    genuinely close. Returns `(Strategy.ADVANCE, False)` -- today's ordinary
    behavior, no retreat -- with no members, no visible enemies, or when
    this force and the enemies within `AiConfig.task_force_threat_radius`
    of it clearly, one-sidedly outmatch each other either way (the same `group_power`/`outmatched`/`task_force_
    outnumbered_margin` race `tla.ai.task_force.is_outnumbered` itself uses,
    purely as a gate here, not a verdict). A clearly outmatched force is
    `is_outnumbered`'s own job (checked independently in `update_task_
    force_stance`, unaffected by this function); a clearly ahead force just
    proceeds with `ADVANCE`, since there's nothing to gain from spending
    four dry runs to prove what's already obvious.

    Otherwise, scores all four strategies via `_project_engagement_value`
    and picks the best. Ties favor, in order, `ADVANCE`, `HOLD`,
    `AGGRESSIVE`, `RETREAT` (`_STRATEGY_DRASTICNESS`) -- i.e. prefer the
    least drastic option that scores no worse than the alternatives, so a
    force doesn't retreat or commit to a trade when simply advancing (or
    holding) does exactly as well."""
    ai_config = game_state.config.ai
    members = [game_state.ships[i] for i in force.member_ids if i in game_state.ships]
    if not visible_enemies or not members:
        return Strategy.ADVANCE, False

    nearby_enemies = [
        e
        for e in visible_enemies.values()
        if any(distance(e.position, m.position) <= ai_config.task_force_threat_radius for m in members)
    ]
    ours = group_power(members, game_state)
    theirs = group_power(nearby_enemies, game_state)
    if outmatched(theirs, ours, ai_config.task_force_outnumbered_margin) or outmatched(
        ours, theirs, ai_config.task_force_outnumbered_margin
    ):
        return Strategy.ADVANCE, False

    scores = {
        s: _project_engagement_value(
            force, game_state, player, visible_enemies, forces, pace_by_force, model, strategy=s
        )
        for s in Strategy
    }
    best = max(Strategy, key=lambda s: (scores[s], -_STRATEGY_DRASTICNESS[s]))
    return best, best is Strategy.RETREAT


def _reevaluate_strategy_on_new_sightings(
    game_state: GameState,
    player: PlayerId,
    forces: list[TaskForce],
    pace_by_force: dict[int, int],
    model: EnemyModel,
    seen_enemy_ids: set[int],
    strategy_by_force: dict[int, Strategy],
    should_retreat_by_force: dict[int, bool],
) -> None:
    """Call after any real move that could have revealed something new (a
    later ship's own vision shifting) -- if any currently-visible enemy id
    isn't in `seen_enemy_ids` yet, re-runs `_pick_strategy_for_force`/
    `update_task_force_stance` for every not-already-retreating force
    against the full, current picture, mutating `strategy_by_force`/
    `should_retreat_by_force`/`force.retreating` in place. A no-op the
    instant nothing new turns up -- most calls should be this cheap, and
    `_pick_strategy_for_force`/`update_task_force_stance` are both bounded
    by one force's own membership times total visible enemies (no
    board-wide or all-forces scan), so repeated calls are not a
    performance concern.

    Found necessary by a real replay review (game61, turn 6): the AI's
    task force saw only 1 of 3 nearby enemy battleships when its strategy
    was first picked for the turn, trivially resolved to ADVANCE, then
    watched the other 2 get revealed mid-turn by its own ships' advancing
    vision with nothing left to reconsider that decision until the turn
    after -- by which point the force was already committed against what
    had become a 3-battleship concentration.

    Only ever passes the not-yet-retreating subset of `forces` into
    `update_task_force_stance` -- that function's own "already retreating"
    branch unconditionally increments `retreat_turns` and re-checks resume
    conditions every time it's called, with no notion of "this is a
    re-check within the same turn, not a new turn." Re-entering it for an
    already-retreating force would double-count `retreat_turns` and could
    resume it early; there's nothing to re-decide for it anyway -- a new
    sighting can't make an already-fleeing force flee "more." Deliberately
    does not re-run `update_task_force_goals` -- goal reassignment/stall
    handling is a separate concern (which port to pursue, when to give up
    a stuck goal) from risk to the fleet, which is what this exists for."""
    visible_enemies = enemy_ships_visible_to(game_state, player)
    new_ids = set(visible_enemies) - seen_enemy_ids
    if not new_ids:
        return
    seen_enemy_ids |= new_ids
    active_forces = [f for f in forces if not f.retreating]
    for f in active_forces:
        strategy, should_retreat = _pick_strategy_for_force(
            f, game_state, player, visible_enemies, forces, pace_by_force, model
        )
        strategy_by_force[f.id] = strategy
        should_retreat_by_force[f.id] = should_retreat
    update_task_force_stance(game_state, player, active_forces, visible_enemies, should_retreat_by_force)


def _choose_generic_destination(
    ship: Ship, game_state: GameState, player: PlayerId, visible_enemies: dict[int, Ship]
) -> AxialCoord | None:
    attack_hex = _favorable_attack(ship, game_state, visible_enemies, [ship])
    if attack_hex is not None:
        return attack_hex

    if visible_enemies:
        enemy_hexes = frozenset(e.position for e in visible_enemies.values())
        nearest_enemy_ship = scoring.nearest_enemy(ship, visible_enemies, game_state)
        threatened_port = _most_threatened_port(game_state, player, visible_enemies)
        if (
            threatened_port is not None
            and nearest_enemy_ship is not None
            and distance(ship.position, threatened_port) <= distance(ship.position, nearest_enemy_ship.position)
        ):
            # threatened_port is a friendly port -- guard it from an
            # adjacent hex rather than sitting on it, which would block
            # its own production for as long as the "threat" persists.
            return _step_toward(ship, game_state, threatened_port, avoid=enemy_hexes | {threatened_port})
        if nearest_enemy_ship is not None:
            return _step_toward(ship, game_state, nearest_enemy_ship.position, avoid=enemy_hexes)

    port = scoring.nearest_uncontrolled_port(game_state, player, ship.position)
    if port is not None:
        return _step_toward(ship, game_state, port)
    return None


def _carrier_cautious_max_steps(ship: Ship, game_state: GameState, max_steps: int | None) -> int:
    """`max_steps` tightened by `AiConfig.carrier_advance_reserve`: a
    carrier advancing toward a goal holds back that many of its own
    movement points rather than spending its full budget (or whatever
    `max_steps` the force's shared pace already allows), so it's never
    more than a couple hexes past wherever it last had a clear scan --
    see `_choose_carrier_destination`. Floored at 0, never negative, for a
    reserve that meets or exceeds the carrier's own remaining movement."""
    reserved = max(0, ship.movement_remaining - game_state.config.ai.carrier_advance_reserve)
    return min(max_steps, reserved) if max_steps is not None else reserved


def _carrier_cautious_step_toward(
    ship: Ship,
    game_state: GameState,
    target: AxialCoord,
    avoid: frozenset[AxialCoord],
    cautious_steps: int,
) -> AxialCoord | None:
    """`_step_toward(ship, game_state, target, avoid=avoid, max_steps=
    cautious_steps)`, except it holds `ship` at its current position
    instead of exceeding `cautious_steps` when nothing reachable actually
    satisfies both `avoid` and the cap together. `_step_toward`'s own
    `max_steps` fallback -- there so an ordinary force member's pace cap
    never stands the ship if the capped set comes up empty -- would
    otherwise silently revert to the ship's *full* movement the instant
    that happens, defeating the entire point of a cautious carrier's
    reserve right when it matters most: a real game showed the AI's own
    carriers and an escort clustered tightly enough that none of one
    carrier's true one-hex neighbors were free to stop at (friendly-
    occupied, and passing through needs 2 spare movement, not the 1 the
    reserve had left), so it took the next legal stop instead -- two
    hexes out, deep in a threat's reach it would otherwise have stayed
    clear of. Holding in place is always safe here: this is only ever
    called once the immediate-threat check in `_choose_carrier_
    destination` has already confirmed nothing dangerous can reach `ship`
    at its *current* hex."""
    reachable = movement.reachable_hexes(ship, game_state)
    if not any(h not in avoid and c <= cautious_steps for h, c in reachable.items()):
        return ship.position
    return _step_toward(ship, game_state, target, avoid=avoid, max_steps=cautious_steps)


def _carrier_dangerous_threat_hexes(
    ship: Ship, game_state: GameState, visible_enemies: dict[int, Ship]
) -> dict[int, set[AxialCoord]]:
    """Every `DANGEROUS_TO_CARRIER_KINDS` enemy's own next-turn reachable
    hex set (`enemy_reachable_next_turn`), keyed by ship id -- the shared
    primitive behind both `_carrier_reaching_threat` (is `ship`'s
    *current* hex in any of these) and `_choose_carrier_destination`'s
    `danger_hexes` (the union of all of them, used to steer a cautious
    advance around every one, not just whichever's nearest) and the
    carrier-scouting hop loop's own per-hop danger check."""
    dangerous_enemies = {i: e for i, e in visible_enemies.items() if e.kind in DANGEROUS_TO_CARRIER_KINDS}
    return {i: enemy_reachable_next_turn(e, game_state) for i, e in dangerous_enemies.items()}


def _carrier_reaching_threat(ship: Ship, game_state: GameState, visible_enemies: dict[int, Ship]) -> Ship | None:
    """The nearest `DANGEROUS_TO_CARRIER_KINDS` enemy that could reach
    `ship`'s *current* hex on its own next turn, or `None` -- extracted
    from `_choose_carrier_destination`'s own immediate-threat check so
    the carrier-scouting pass (see `_carrier_scouting_eligible`/
    `_carrier_scouting_advance`) can reuse the identical logic per hop
    instead of a second copy."""
    threat_hexes = _carrier_dangerous_threat_hexes(ship, game_state, visible_enemies)
    reaching = {i: visible_enemies[i] for i, hexes in threat_hexes.items() if ship.position in hexes}
    return scoring.nearest_enemy(ship, reaching, game_state) if reaching else None


def _carrier_formation_destination(
    ship: Ship,
    force: TaskForce,
    game_state: GameState,
    player: PlayerId,
    visible_enemies: dict[int, Ship],
    goal: TaskForceGoal,
    resolved: frozenset[int],
    max_steps: int | None,
    strategy: Strategy,
    cautious_steps: int,
    enemy_hexes: frozenset[AxialCoord],
    own_ports: frozenset[AxialCoord],
) -> AxialCoord | None:
    """Among this carrier's reachable-this-turn hexes, picks whichever
    maximizes how many of its own force's `AC_BONUS_ELIGIBLE_KINDS`
    escorts (battleship/cruiser/destroyer) end up within `CombatConfig.
    ac_bonus_radius` of it -- more supported ships means more of them
    fight at full strength next contact, not just "at least one" (see
    `_carrier_bonus_cohesion_destination`'s own narrower zero-coverage-
    only check, which this leaves entirely unchanged). Escort positions
    used for this are each escort's own honest *projected* destination
    this turn (a dry run of its ordinary `choose_task_force_destination`,
    exactly as `_project_engagement_value` already does elsewhere) --
    never its current position, and never a fictional re-dry-run for an
    escort that already moved for real this turn via an earlier pass
    (port defense, carrier defense, `secure_kills_pass` -- see
    `resolved`, which is checked first for exactly this reason).

    Among hexes tied on coverage, prefers one no `DANGEROUS_TO_CARRIER_
    KINDS` enemy could reach next turn -- evaluated against those same
    projected escort positions (`with_repositioned_ships`), so a screen
    that will only exist once escorts finish their own move this turn is
    correctly credited -- falling back to the least-exposed coverage-tied
    hex rather than refusing to move, matching `_carrier_cautious_step_
    toward`'s/`_make_way_for_scout`'s own "never strand the ship"
    philosophy elsewhere in this module. Coverage is deliberately
    *primary* and screening only a secondary tie-break, not a hard
    safety pre-filter: an open-ocean approach with no escort in place yet
    has nothing reachable that's screened at all, and a hard pre-filter
    would then throw away coverage entirely right when it matters most.

    Remaining ties break toward whichever hex is closest (real sea route)
    to `goal.target` for `AGGRESSIVE`/`ADVANCE`, or closest to `ship`'s
    own current position for `HOLD` -- though `HOLD` structurally never
    has more than one candidate here at all: it already zeroes `max_steps`
    before either destination function is ever called (see `_choose_
    destination`), which floors `cautious_steps` at 0, which collapses
    `candidates` below to just `ship.position`. Kept anyway for clarity
    should that relationship ever change, not because it's reachable
    today. `Strategy.RETREAT` never reaches this function at all -- its
    own shadow branch and the goal's own `RETREAT` kind both return
    earlier in `_choose_carrier_destination`.

    Returns `None` only when the force has no living `AC_BONUS_ELIGIBLE_
    KINDS`-minus-carrier member -- nothing to optimize formation against;
    the caller falls through to the unchanged `_carrier_cautious_step_
    toward` path in that case."""
    escorts = [
        game_state.ships[i]
        for i in force.member_ids
        if i in game_state.ships
        and game_state.ships[i].kind in AC_BONUS_ELIGIBLE_KINDS
        and game_state.ships[i].kind != ShipKind.CARRIER
    ]
    if not escorts:
        return None

    projected: dict[int, AxialCoord] = {}
    for escort in escorts:
        if escort.id in resolved:
            projected[escort.id] = escort.position  # already moved for real this turn
            continue
        dest = choose_task_force_destination(
            escort, goal, game_state, visible_enemies, force, max_steps=max_steps, strategy=strategy
        )
        projected[escort.id] = dest if dest is not None else escort.position

    escort_scratch = with_repositioned_ships(game_state, projected)
    threat_hexes = _carrier_dangerous_threat_hexes(ship, escort_scratch, visible_enemies)
    exposed_hexes = {h for hexes in threat_hexes.values() for h in hexes}

    raw_reachable = movement.reachable_hexes(ship, game_state)
    candidates = {
        h: c
        for h, c in raw_reachable.items()
        if c <= cautious_steps and h not in enemy_hexes and h not in own_ports
    }
    candidates[ship.position] = 0  # staying put is always a legal, safe-cost candidate

    radius = game_state.config.combat.ac_bonus_radius

    def coverage(h: AxialCoord) -> int:
        return sum(1 for pos in projected.values() if distance(h, pos) <= radius)

    field = None if strategy == Strategy.HOLD else sea_distance_field(game_state, goal.target)

    def key(h: AxialCoord) -> tuple:
        progress = sea_route_distance(field, h) if field is not None else distance(h, ship.position)
        return (-coverage(h), h in exposed_hexes, progress, candidates[h], h)

    return min(candidates, key=key)


def _choose_carrier_destination(
    ship: Ship,
    game_state: GameState,
    player: PlayerId,
    visible_enemies: dict[int, Ship],
    goal: TaskForceGoal | None = None,
    max_steps: int | None = None,
    force: TaskForce | None = None,
    strategy: Strategy = Strategy.ADVANCE,
    resolved: frozenset[int] = frozenset(),
) -> AxialCoord | None:
    """A carrier never *deliberately* self-initiates an attack, regardless
    of `goal` -- it's worth actively protecting, since it's what grants
    the air-cover bonus (`battle.carrier_bonus_for`) to nearby friendly
    warships in the first place. The threat-retreat check below always
    comes first and is unaffected by `goal` -- safety overrides
    everything, goal included (and it already moves the carrier *toward*
    its own force, not away, so it's never itself a cohesion risk the way
    advancing unchecked would be) -- and, since fleeing toward the
    nearest own warship is otherwise blind to anything else occupying a
    candidate hex, excludes every visible enemy's own hex from
    consideration too: a real game showed a carrier fleeing a battleship
    picking a "close to my own fleet" retreat hex that happened to be
    sitting right on top of an entirely different enemy ship, converting
    the retreat into an accidental attack on something it was never
    fleeing from or trying to fight in the first place. Modeled directly
    on a human player's own reported tactic:
    advance cautiously a couple hexes at a time (`AiConfig.carrier_
    advance_reserve`), scanning after each move for a battleship, cruiser,
    or submarine (`DANGEROUS_TO_CARRIER_KINDS` -- a destroyer or patrol
    boat alone doesn't count) that could reach this hex on its own next
    turn (`enemy_reachable_next_turn` -- a real reachability check, not a
    flat radius), and falling back the instant one turns up rather than
    committing the full move every turn.

    If not threatened: with a task-force `goal`, the carrier advances
    toward it (still escorted/protected by its own force-mates, who share
    the same goal), paced by `max_steps` (see `_compute_force_pace`) same
    as any other force member -- this is what "the carrier advances with
    the force" means concretely: not just heading the same direction, but
    not outrunning it either -- *and* further capped by `carrier_advance_
    reserve` (see `_carrier_cautious_max_steps`), so a turn with nothing
    visible nearby still doesn't spend the carrier's full budget: staying
    a couple hexes short of the group's own pace is what keeps a fall-back
    route open the instant next turn's scan finds something. If `force` is
    given and `AiConfig.task_force_max_separation` is set, `_cohesion_
    destination` can override this same as any other member -- the carrier
    is usually a force's anchor, but nothing stops it individually pulling
    ahead of a straggler otherwise; that correction is deliberately exempt
    from the reserve cap, since closing an existing cohesion gap takes
    priority over staying cautious. Failing that, the (capped) advance
    itself also avoids any hex a dangerous-kind enemy could reach next
    turn -- and, unlike an ordinary force member's own pace cap, holds the
    carrier in place rather than exceeding either the cap or a danger hex
    when nothing reachable satisfies both at once (see `_carrier_cautious_
    step_toward`) -- proactively steering wide of a fight instead of only
    reacting once already in range of the immediate-threat check above.
    The advance (and the goal-less closing-distance-to-own-fleet fallback
    at the very end) also never stops the carrier on one of its own
    already-controlled ports incidentally -- see the `own_ports` exclusion
    below -- since occupying a port (even your own) zeroes its production
    for the turn. An earlier version
    instead made the carrier structurally trail behind its own vanguard,
    unconditionally -- reverted after self-play showed it stalling several
    games indefinitely: the carrier's freedom to keep walking toward the
    goal *unmolested* while its own combat ships were still slugging it out
    nearby was, in this naive AI, often the only thing actually finishing a
    capture; forbidding it from ever leading took that away. Without a
    goal (an unassigned carrier), it closes distance to its nearest own
    warship instead, unpaced and unreserved, same as before task forces
    existed, so it's already in bonus range before a fight starts.

    `strategy == Strategy.RETREAT` (see `choose_task_force_destination`'s
    own docstring) disengages toward `rally_point(force, game_state)` even
    when `goal.kind` isn't already `RETREAT` -- the same hypothetical-
    scoring case, applied here since a carrier never self-initiates an
    attack regardless of strategy (`AGGRESSIVE`'s tie-tolerant relaxation
    has nothing to apply to) and `HOLD` needs no special case (expressed
    entirely by the caller passing `max_steps=0`, already honored by
    `_carrier_cautious_max_steps`/`_cohesion_destination` above). The
    immediate-threat safety check always comes first regardless."""
    own_warships = [s for s in game_state.ships_for(player) if s.kind != ShipKind.CARRIER]
    nearest_threat = _carrier_reaching_threat(ship, game_state, visible_enemies)
    enemy_hexes = frozenset(e.position for e in visible_enemies.values())

    if nearest_threat is not None:
        raw_reachable = movement.reachable_hexes(ship, game_state)
        if not raw_reachable:
            return None
        # Excludes every visible enemy's own hex, not just nearest_threat's
        # -- a carrier isn't itself a "dangerous" kind, so nothing else
        # here stops it from being picked as the "closest to my own fleet"
        # retreat spot if it happens to score well on that alone, sending
        # the carrier fleeing straight into an entirely different enemy
        # ship instead of away from the one it's actually retreating from.
        # Falls back to the unfiltered set rather than stranding the
        # carrier if literally every reachable hex is enemy-occupied.
        reachable = {h: c for h, c in raw_reachable.items() if h not in enemy_hexes} or raw_reachable

        def key(h: AxialCoord) -> tuple[int, int, AxialCoord]:
            d_friend = min((distance(h, w.position) for w in own_warships), default=_NO_ENEMY_SENTINEL)
            d_enemy = distance(h, nearest_threat.position)
            return (d_friend, -d_enemy, h)

        return min(reachable, key=key)

    # A port occupied by any ship -- friendly included -- earns nothing
    # that turn (see tla.production.run_production), so a carrier passing
    # *through* one of its own controlled ports incidentally (not
    # RETREAT's own single rally-port exclusion just below, which is
    # about a deliberate fall-back destination) shouldn't stop there
    # either: a real game showed a carrier parking on its own port for a
    # full turn cycle mid-advance toward a goal that had nothing to do
    # with that port, silently zeroing its production the whole time.
    own_ports = frozenset(game_state.board.controlled_ports_for(player))

    if force is not None and strategy == Strategy.RETREAT and (goal is None or goal.kind != GoalKind.RETREAT):
        retreat_target = rally_point(force, game_state)
        if retreat_target is not None:
            return _step_toward(ship, game_state, retreat_target, avoid=enemy_hexes | {retreat_target})

    if goal is not None:
        # See choose_task_force_destination: a RETREAT goal targets a
        # friendly port to fall back near, not onto -- excluded so the
        # carrier doesn't block its own production sitting there -- and
        # is never paced, same reasoning (get home without dawdling).
        if goal.kind == GoalKind.RETREAT:
            return _step_toward(ship, game_state, goal.target, avoid=enemy_hexes | {goal.target})
        if force is not None:
            cohesion_destination = _cohesion_destination(ship, force, game_state, goal.target, enemy_hexes)
            if cohesion_destination is not None:
                return cohesion_destination
        cautious_steps = _carrier_cautious_max_steps(ship, game_state, max_steps)
        if force is not None and game_state.config.ai.carrier_formation_optimization_enabled:
            formation_destination = _carrier_formation_destination(
                ship,
                force,
                game_state,
                player,
                visible_enemies,
                goal,
                resolved,
                max_steps,
                strategy,
                cautious_steps,
                enemy_hexes,
                own_ports,
            )
            if formation_destination is not None:
                return formation_destination
        threat_hexes = _carrier_dangerous_threat_hexes(ship, game_state, visible_enemies)
        danger_hexes = enemy_hexes | own_ports | {h for hexes in threat_hexes.values() for h in hexes}
        return _carrier_cautious_step_toward(ship, game_state, goal.target, danger_hexes, cautious_steps)

    if own_warships:
        nearest_warship = min(own_warships, key=lambda w: distance(ship.position, w.position))
        if distance(ship.position, nearest_warship.position) > game_state.config.combat.ac_bonus_radius:
            return _step_toward(ship, game_state, nearest_warship.position, avoid=enemy_hexes | own_ports)
    return None


def _carrier_scouting_eligible(
    ship: Ship,
    game_state: GameState,
    force: TaskForce | None,
    strategy: Strategy,
    visible_enemies: dict[int, Ship],
) -> bool:
    """Whether `ship` (already confirmed a carrier by the caller) would,
    absent `AiConfig.carrier_scouting_enabled`, land in exactly `_choose_
    carrier_destination`'s flat-capped cautious-advance branch this turn
    -- the one branch `_carrier_scouting_advance` replaces. Mirrors that
    function's actual branch order precisely: the immediate-threat flee
    branch, the `Strategy.RETREAT`/`force.retreating` shadow, a `RETREAT`
    or missing goal, and a `_cohesion_destination` correction all take
    priority there and must do so here too -- a carrier failing any of
    these checks is left unclaimed, falling through to `_choose_carrier_
    destination` itself, which recomputes the same branches fresh and
    handles it exactly as before this existed. Only a cheap, already-
    fast re-check (a handful of enemies, one force's members) -- no
    decision logic is duplicated, since the actual destination in every
    one of those cases is still computed by the unchanged function."""
    if force is None or force.goal is None or force.goal.kind == GoalKind.RETREAT:
        return False
    if force.retreating or strategy == Strategy.RETREAT:
        return False
    if _carrier_reaching_threat(ship, game_state, visible_enemies) is not None:
        return False
    enemy_hexes = frozenset(e.position for e in visible_enemies.values())
    if _cohesion_destination(ship, force, game_state, force.goal.target, enemy_hexes) is not None:
        return False
    return True


def _record_scout_sightings(model: EnemyModel, game_state: GameState, visible_enemies: dict[int, Ship]) -> None:
    """Folds every currently-visible enemy into `model` via `observe_
    ship_state` -- the same public, externally-callable mutator `_reconcile_
    defender_belief` already uses outside `EnemyModel`'s own `begin_turn`/
    `end_turn` lifecycle (safe/idempotent to call repeatedly per turn, see
    that method's own docstring). Called once per scouting hop, *before*
    any retreat that hop might trigger, so a sighting made only
    transiently -- visible from a hex the carrier ends up abandoning --
    is still permanently recorded: `end_turn`'s own later `_apply_
    sightings` pass only touches ships still visible *then*, so it can
    never clobber or duplicate what a mid-turn scouting hop already
    observed and the carrier has since retreated away from."""
    for enemy_id, enemy in visible_enemies.items():
        model.observe_ship_state(
            enemy_id, enemy.kind, enemy.position, enemy.current_hp, game_state, surfaced=enemy.surfaced
        )


def _scouting_path_blocker(ship: Ship, game_state: GameState, target: AxialCoord) -> Ship | None:
    """The friendly ship (if any) sitting on the single neighbor hex that
    most reduces real sea-route distance from `ship` toward `target` --
    the direct next step of the shortest route, currently unusable as a
    stop (a friendly-occupied hex is never a legal stop, only a
    pass-through -- `movement._classify_step`). `None` if that ideal hex
    is open (nothing to clear), off-board/non-sea, held by an enemy (a
    different situation already handled by `danger_hexes`/`avoid`
    elsewhere), or held by a ship that's already spent any movement this
    turn (nudging it now would disturb an already-finalized decision --
    checked via `movement_remaining == max_movement`, exact equality
    since nothing partial should have touched it yet)."""
    field = sea_distance_field(game_state, target)
    own_distance = sea_route_distance(field, ship.position)
    candidates = [n for n in neighbors(ship.position) if game_state.board.is_occupiable(n)]
    if not candidates:
        return None
    best = min(candidates, key=lambda h: (sea_route_distance(field, h), h))
    if sea_route_distance(field, best) >= own_distance:
        return None
    occupant = game_state.ship_at(best)
    if occupant is None or occupant.owner != ship.owner:
        return None
    stats = game_state.config.ship_stats.stats[occupant.kind]
    if occupant.movement_remaining != occupant.max_movement(stats):
        return None
    return occupant


def _make_way_for_scout(
    blocker: Ship,
    game_state: GameState,
    visible_enemies: dict[int, Ship],
    decide_battle,
) -> None:
    """Moves `blocker` exactly one hex off its current position (the
    scouting carrier's intended next stop) to whichever adjacent,
    currently-reachable hex isn't itself newly dangerous -- reachability
    via `enemy_reachable_next_turn` for *every* currently-visible enemy,
    not just `DANGEROUS_TO_CARRIER_KINDS`, since `blocker` isn't
    necessarily a carrier. `blocker`'s own hex and any friendly-occupied
    hex (the carrier's own included) are already excluded by `movement.
    reachable_hexes` itself -- a legal single-hex stop is always a
    genuinely different, open hex, no separate exclusion needed here. A
    no-op (declines to move at all) if no safe hex exists -- this never
    sacrifices a teammate's safety just to clear a path; the carrier's
    own step afterward simply routes around `blocker` instead, exactly
    as `_step_toward` would unaided. Like any ordinary move, this can
    still trigger a real battle if the chosen hex secretly hides a
    submerged enemy submarine invisible to `visible_enemies` (see
    `_execute`'s own docstring) -- the caller re-fetches vision fresh
    immediately after regardless of what happened here, so that outcome
    needs no special handling in this function itself.

    `blocker` is deliberately *not* claimed/resolved here -- it still
    takes its own ordinary turn later in the ordinary per-ship loop,
    computed fresh from wherever it ends up with whatever movement
    remains, same as any other ship; nothing about that later decision
    needs to know it was nudged first. A submarine blocker gets its
    once-per-turn surfaced/submerged toggle considered *before* the
    nudge (`_maybe_toggle_submarine`, the same call every other
    claim-and-act pass makes right before its own `_execute`) --
    toggling only ever being legal before a submarine has spent any
    movement this turn (`movement.toggle_submarine_state`), a nudge is
    itself real movement and would otherwise permanently forfeit that
    submarine's toggle for the whole turn without ever offering it the
    choice."""
    if blocker.kind == ShipKind.SUBMARINE:
        _maybe_toggle_submarine(blocker, game_state, visible_enemies)
    candidates = {h: c for h, c in movement.reachable_hexes(blocker, game_state).items() if c == 1}
    if not candidates:
        return
    threatened = {h for e in visible_enemies.values() for h in enemy_reachable_next_turn(e, game_state)}
    safe = {h: c for h, c in candidates.items() if h not in threatened}
    if not safe:
        return
    destination = min(safe, key=lambda h: h)
    _execute(blocker, destination, game_state, decide_battle)


def _carrier_scouting_advance(
    ship: Ship,
    force: TaskForce,
    game_state: GameState,
    player: PlayerId,
    model: EnemyModel,
    pace_by_force: dict[int, int],
    decide_battle,
) -> None:
    """Advances `ship` (a carrier already confirmed eligible by `_carrier_
    scouting_eligible`) toward `force.goal.target` one real hex at a
    time, verifying safety with freshly recomputed vision after each hop
    instead of committing a flat-capped move blind (see `AiConfig.
    carrier_scouting_enabled`'s own doc-comment for the doctrine this
    replaces, and the failure mode it independently also avoids). Falls
    back exactly one hop -- a real, costed `_execute`, not a free undo;
    `tla.movement.move_ship` has no special-cased undo anywhere in this
    codebase -- the instant a hop exposes the carrier to a newly-
    reachable `DANGEROUS_TO_CARRIER_KINDS` enemy (`_carrier_reaching_
    threat`). Before each hop, a single friendly ship directly occupying
    the ideal next hex (if any, if it hasn't moved yet this turn, and if
    a safe hex exists for it to step to) makes way first -- see
    `_scouting_path_blocker`/`_make_way_for_scout`. Every enemy visible
    at any point is folded into `model` before any retreat -- see
    `_record_scout_sightings` -- so a sighting made only transiently
    mid-scout is never lost even though the carrier itself won't be
    there at `end_turn`.

    `AiConfig.carrier_scout_retreat_reserve` (default 1) is subtracted
    from the hop ceiling once, up front: since the loop can't know in
    advance whether the *next* hop will turn out dangerous, and a retreat
    costs a real movement point same as any other move, "always fully
    spend the paced budget when safe" and "always keep enough to retreat
    from whatever hop was just taken" can't both hold unconditionally --
    this keeps a small, fixed, unconditional escape hatch instead,
    provably never strandable, at the cost of never spending the ship's
    literal last movement point."""
    goal = force.goal
    ceiling = min(pace_by_force.get(force.id, ship.movement_remaining), ship.movement_remaining)
    reserve = game_state.config.ai.carrier_scout_retreat_reserve
    hop_budget = max(0, ceiling - reserve)

    while hop_budget > 0:
        visible_enemies = enemy_ships_visible_to(game_state, player)
        _record_scout_sightings(model, game_state, visible_enemies)

        blocker = _scouting_path_blocker(ship, game_state, goal.target)
        if blocker is not None:
            _make_way_for_scout(blocker, game_state, visible_enemies, decide_battle)
            # Whether or not a safe nudge hex existed: proceed either way
            # -- if it moved, the ideal hex is now free; if not, the
            # ordinary step below routes around it exactly as today. A
            # real move (blocker's own or, in principle, this ship's)
            # can shift whose vision covers what, so re-fetch fresh
            # rather than reuse the pre-nudge snapshot for danger_hexes
            # below.
            visible_enemies = enemy_ships_visible_to(game_state, player)
            _record_scout_sightings(model, game_state, visible_enemies)

        own_ports = frozenset(game_state.board.controlled_ports_for(player))
        enemy_hexes = frozenset(e.position for e in visible_enemies.values())
        threat_hexes = _carrier_dangerous_threat_hexes(ship, game_state, visible_enemies)
        danger_hexes = enemy_hexes | own_ports | {h for hexes in threat_hexes.values() for h in hexes}

        next_hex = _carrier_cautious_step_toward(ship, game_state, goal.target, danger_hexes, cautious_steps=1)
        if next_hex == ship.position:
            return  # nothing safe reachable this hop -- hold, done for this turn

        previous_hex = ship.position
        _execute(ship, next_hex, game_state, decide_battle)
        hop_budget -= 1
        if ship.id not in game_state.ships:
            # next_hex looked clear (nothing in visible_enemies there),
            # but the *true* game state -- see _execute's own docstring
            # -- hid a submerged enemy submarine on it, exactly as it
            # would for a human sailing in blind; the resulting battle
            # sank the carrier. Nothing left to retreat.
            return

        visible_enemies = enemy_ships_visible_to(game_state, player)
        _record_scout_sightings(model, game_state, visible_enemies)  # preserve info before any retreat
        if _carrier_reaching_threat(ship, game_state, visible_enemies) is not None:
            _execute(ship, previous_hex, game_state, decide_battle)  # retreat exactly one hop
            return


def _port_defense_assignments(
    game_state: GameState, player: PlayerId, visible_enemies: dict[int, Ship], ai_config: AiConfig
) -> list[tuple[Ship, PortDefenseDirective]]:
    """Flattens `compute_port_defense_directives`' per-port directives
    (pure threat-detection/group-strength comparison, no movement/attack
    knowledge -- lives in tla.ai.task_force) into one (ship, directive)
    pair per responding ship, in a fixed order. Deliberately doesn't also
    compute each ship's actual destination here -- see
    `_port_defense_destination`, called fresh right before each ship
    moves, since an earlier responder's own move this turn can change
    what's reachable or occupied for a later one (computing every
    destination upfront from one shared snapshot let two responders
    converge on the same hex and the second one's move then read as a
    friendly-fire "attack" on the first)."""
    pairs: list[tuple[Ship, PortDefenseDirective]] = []
    for directive in compute_port_defense_directives(game_state, player, visible_enemies, ai_config):
        if directive.block is not None:
            pairs.append((directive.block, directive))
        else:
            pairs.extend((ship, directive) for ship in directive.counterattack)
    return pairs


def _port_defense_destination(
    ship: Ship, directive: PortDefenseDirective, game_state: GameState, visible_enemies: dict[int, Ship]
) -> AxialCoord | None:
    """This one ship's destination for `directive`, computed against the
    *current* game state (see `_port_defense_assignments`). A blocker
    heads for `block_hex` via `_step_toward` -- not guaranteed reachable
    in one turn, so this gets it as close as its actual movement budget
    allows, converging over subsequent turns if it can't arrive in one
    hop. A counterattacker takes a genuinely favorable reachable attack
    if one's available (`_favorable_attack`, the same safety gate task
    forces already use so one bad individual matchup doesn't get picked
    even though the group overall is favored), otherwise closes the
    distance on whichever threat is nearest -- filtered to still-living
    ones, in case an earlier responder already sank one this turn. The
    final close-the-distance fallback excludes every visible enemy hex
    (`avoid=enemy_hexes`, the same exclusion every other "reposition,
    don't attack" call to `_step_toward` in this file already uses) --
    without it, `nearest_threat.position` is itself an enemy-occupied hex
    and is almost always the closest reachable point toward that exact
    target, so "just get closer" would silently turn into an unconditional
    attack against a matchup this function just finished rejecting (a
    real bug, found via replay review: a losing counterattacker fell
    through here and walked straight onto the threat anyway)."""
    enemy_hexes = frozenset(e.position for e in visible_enemies.values())
    if directive.block is not None:
        return _step_toward(ship, game_state, directive.block_hex)
    attack_hex = _favorable_attack(ship, game_state, visible_enemies, directive.counterattack)
    if attack_hex is not None:
        return attack_hex
    living_threats = [t for t in directive.threats if t.id in game_state.ships] or directive.threats
    nearest_threat = min(living_threats, key=lambda t: (distance(ship.position, t.position), t.id))
    return _step_toward(ship, game_state, nearest_threat.position, avoid=enemy_hexes)


def _carrier_defense_assignments(
    game_state: GameState, player: PlayerId, visible_enemies: dict[int, Ship], ai_config: AiConfig, claimed: frozenset[int]
) -> list[tuple[Ship, CarrierDefenseDirective]]:
    """Same flattening `_port_defense_assignments` does for `compute_
    port_defense_directives`, here for `compute_carrier_defense_
    directives` -- `claimed` is threaded through so a ship already
    responding to a threatened port this turn isn't double-booked
    defending a carrier too; see that function's own docstring for why
    port defense gets priority over carrier defense when both want the
    same ship."""
    pairs: list[tuple[Ship, CarrierDefenseDirective]] = []
    for directive in compute_carrier_defense_directives(game_state, player, visible_enemies, ai_config, claimed):
        if directive.block is not None:
            pairs.append((directive.block, directive))
        else:
            pairs.extend((ship, directive) for ship in directive.counterattack)
    return pairs


def _carrier_defense_destination(
    ship: Ship, directive: CarrierDefenseDirective, game_state: GameState, visible_enemies: dict[int, Ship]
) -> AxialCoord | None:
    """This one ship's destination for `directive` -- same shape as
    `_port_defense_destination` (a blocker heads for `block_hex`, filtered
    to still-living threats otherwise), except a counterattacker here
    first tries the same strictly-favorable `_favorable_attack` any
    responder would (in case something even better than the threat itself
    is available), and only then falls back to a *tie-tolerant* attack
    specifically against `directive.threats` (`matchup_score >= 0`, not
    the ordinary `> 0`) -- `compute_carrier_defense_directives` already
    decided the group trade against this specific threat is worth it even
    at a mutual kill, so an individual ship taking that exact fight
    shouldn't be blocked by the same bar that (correctly) stops it from
    picking pointless ties against something unrelated. Same fix as
    `_port_defense_destination`'s identical fallback: the final close-
    the-distance step excludes every visible enemy hex, so a matchup
    this function just rejected (both the strict and tie-tolerant checks)
    can't still be forced through as a side effect of "get closer"."""
    enemy_hexes = frozenset(e.position for e in visible_enemies.values())
    if directive.block is not None:
        return _step_toward(ship, game_state, directive.block_hex)
    attack_hex = _favorable_attack(ship, game_state, visible_enemies, directive.counterattack)
    if attack_hex is None:
        living_threats = {t.id: t for t in directive.threats if t.id in game_state.ships}
        if living_threats:
            candidate = scoring.best_reachable_attack(ship, game_state, living_threats)
            if candidate is not None:
                defender = _ship_at(living_threats, candidate)
                if defender is not None and scoring.matchup_score(ship, defender, game_state) >= 0:
                    attack_hex = candidate
    if attack_hex is not None:
        return attack_hex
    living_threats = [t for t in directive.threats if t.id in game_state.ships] or directive.threats
    nearest_threat = min(living_threats, key=lambda t: (distance(ship.position, t.position), t.id))
    return _step_toward(ship, game_state, nearest_threat.position, avoid=enemy_hexes)


def _toggle_preserves_destination(ship: Ship, destination: AxialCoord, game_state: GameState) -> bool:
    """Whether `ship` (a submarine, already confirmed reachable at
    `destination` under its *current* surfaced/submerged budget) would
    still be able to reach `destination` after toggling -- surfaced and
    submerged movement budgets differ, so a toggle can retroactively
    invalidate a destination a caller already committed to before ever
    considering stealth. `_maybe_toggle_submarine` has its own, narrower
    version of this same protection for `_favorable_attack`'s specific
    notion of a good attack; this generalizes it to an arbitrary already-
    chosen destination (see `tla.ai.move_scoring`'s own scored path,
    which picks a destination first and only reaches this check
    afterward, unlike every older caller here which decides attack-vs-
    stealth as one combined step). A dry run only -- flips `surfaced`/
    `movement_remaining` to what a real toggle would set them to, checks
    reachability, then restores both; never actually toggles."""
    stats = game_state.config.ship_stats.stats[ship.kind]
    original_surfaced, original_remaining = ship.surfaced, ship.movement_remaining
    ship.surfaced = not ship.surfaced
    ship.movement_remaining = ship.max_movement(stats)
    try:
        return destination in movement.reachable_hexes(ship, game_state)
    finally:
        ship.surfaced, ship.movement_remaining = original_surfaced, original_remaining


def _maybe_toggle_submarine(ship: Ship, game_state: GameState, visible_enemies: dict[int, Ship]) -> None:
    """Naive submarine stealth, symmetric in both directions: submerged
    whenever a visible enemy is around to hide from, surfaced (full
    speed) otherwise -- `should_be_submerged = bool(visible_enemies)`
    below is the whole doctrine, checked fresh every turn so a sub
    resurfaces again the instant nothing is visible, rather than staying
    submerged indefinitely once it first dives. That's the fix for a real,
    twice-confirmed-in-replay bug: the previous version only ever
    considered surfaced->submerged and, having no visible enemy yet to
    react to, would dive on turn 1 by simple default (no favorable attack
    in range yet) and then *never reconsider*, crawling at its submerged
    speed for the rest of the game even through long stretches with
    nothing anywhere near it.

    Skipped entirely if a genuinely favorable attack is already reachable
    at `ship`'s *current* range (whichever state that happens to be) --
    landing it takes priority over adjusting stealth state, and toggling
    either way here could cost the range or the concealment that attack
    depends on. This still doesn't attempt the reverse speculative check
    (would surfacing/submerging *first* reveal a currently-unreachable
    favorable attack) -- a deliberately naive tier, same as before.

    A submarine gets exactly one toggle per turn and only before spending
    any movement (`tla.movement.toggle_submarine_state`), so this is the
    only chance to adjust either direction this turn. Every existing
    caller already only invokes this once per ship, immediately before
    that same ship's own single move this turn -- but a ship the carrier-
    scouting pass's `_make_way_for_scout` nudges out of the way is
    deliberately *not* marked resolved (see that function's own
    docstring), so it can still reach the main per-ship loop's own call
    to this function afterward -- by which point it may have already
    spent its nudge's movement, *or*, if `_make_way_for_scout` toggled it
    but then found no safe hex to actually nudge to (a no-op move),
    already spent its one toggle with `movement_remaining` left
    untouched (`toggle_submarine_state` always refreshes it to the new
    state's full budget). Rather than trust every caller to keep
    enforcing "exactly once, always before any movement" by construction,
    this checks both of `toggle_submarine_state`'s own preconditions
    directly and silently no-ops if either no longer holds -- toggling is
    simply no longer legal for this ship this turn, the same way it
    never was for a ship an *earlier* pass had already moved once and
    marked resolved (which the `resolved` set already keeps out of here
    entirely)."""
    if ship.toggled_this_turn:
        return  # already toggled once this turn (e.g. during a scouting-pass nudge) -- one toggle only, ever
    stats = game_state.config.ship_stats.stats[ship.kind]
    if ship.movement_remaining != ship.max_movement(stats):
        return  # already spent movement this turn -- toggling is no longer legal regardless of doctrine
    attack_hex = scoring.best_reachable_attack(ship, game_state, visible_enemies) if visible_enemies else None
    if attack_hex is not None:
        defender = _ship_at(visible_enemies, attack_hex)
        if scoring.matchup_score(ship, defender, game_state) >= 0:
            return  # a good attack is already reachable at the current range -- don't disturb it either way
    should_be_submerged = bool(visible_enemies)
    currently_submerged = not ship.surfaced
    if should_be_submerged != currently_submerged:
        movement.toggle_submarine_state(ship, stats)


def _execute(ship: Ship, destination: AxialCoord, game_state: GameState, decision_fn) -> None:
    """Move `ship` to `destination`, or attack it if occupied -- checked
    against the *true* game state, not just what this player can see, so a
    destination chosen as "probably safe" that turns out to hide a
    submerged enemy submarine still triggers a battle exactly as it would
    for a human sailing in blind.

    A no-op if `destination` is already `ship`'s own current position --
    every other caller only ever passes a destination from `reachable_
    hexes` (which never includes a ship's own current hex, see that
    function's own docstring), so this case never arose before `tla.ai.
    move_scoring.plan_force_movement_scored`'s own planning phase, which
    can legitimately decide a ship's best move is not moving at all.
    Without this guard, `ship_at(destination)` finds the ship *itself*,
    misreading "stay put" as "attack me"."""
    if destination == ship.position:
        return
    if game_state.ship_at(destination) is not None:
        path = shortest_path(ship, destination, game_state)
        defender = movement.begin_engagement(ship, path, game_state)
        run_battle(ship, defender, game_state, decision_fn=decision_fn)
    else:
        movement.move_ship(ship, destination, game_state)


