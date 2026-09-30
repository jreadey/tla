"""What a player's AI actually knows about the opponent's fleet, kept
current turn by turn: exact for kind/count/alive-or-sunk (deterministic
public information -- see below), a diffusing position-belief field for
anything currently out of sight.

Fog-of-war invariant (shared with the rest of `tla.ai` -- see
`tla.ai.vision`'s own module docstring): this module never reads
`game_state.ships` for anything belonging to the opponent, not even a bare
existence check. Its only two windows into concrete enemy ship data are
`tla.ai.vision.enemy_ships_visible_to` (direct sightings) and
`game_state.battle_log` entries this player's own ships generated as
attacker this half-turn, read before `tla.turn_manager.end_movement_phase`
clears it. A ship whose id we've never legitimately learned this way is
tracked anonymously in its kind's `KindPool`, not given a fabricated id.

Kind, count, and alive-or-sunk are *not* actually uncertain, despite the
above: `Config.fleet.counts` is one shared starting-fleet spec both
players are built from, `ProductionConfig.build_order` is a fixed shared
sequence, and both `Board.controlled_ports_for` and
`GameState.players[x].port_production` are plain fields with no
fog-of-war filtering at all -- the same information a rules-literate human
opponent could work out by hand from what's already on the map. Only a
ship's *position* (and, as a consequence, its HP while we haven't
personally hit it) is genuinely hidden, which is what the position-belief
field (see `tla.ai.hexfield`) is for.

`game_state.battle_log` has a half-turn lifecycle (cleared before our next
`plan_movement` call -- see `tla.turn_manager.end_movement_phase`), so in
isolation an `EnemyModel` could only ever observe combat where it was the
attacker this half-turn -- an enemy ship that attacked one of our own ships
and died in the mutual exchange would go undetected forever, since our own
model never runs while that battle_log still exists. `observe_ship_state`/
`observe_ship_sunk` close this: `NaivePolicy` owns both players' models, so
right after either player's own `plan_movement` call -- while their
battle_log is still populated -- it hands every entry they generated to the
*other* player's model too (see `NaivePolicy._reconcile_defender_belief`).
Being attacked reveals the attacker the same way spotting it would; a
defender obviously knows who and where just hit them. Between that and each
model's own `_observe_own_attacks`, every battle_log entry updates the
correct model on both sides of it, so no enemy ship's fate (sighted, hit,
or sunk) actually goes unrecorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tla.ai.belief_ordinal import turn_ordinal
from tla.ai.hexfield import HexField, HexFieldGeometry
from tla.ai.vision import enemy_ships_visible_to
from tla.board import Board
from tla.fow import visible_hexes_for
from tla.game_state import GameState, PortProduction, TurnPhase
from tla.hexgrid import AxialCoord, distance, hexes_in_range
from tla.mapgen import largest_sea_component
from tla.ship import Ship, ShipKind, ShipStats
from tla.tile import PlayerId

if TYPE_CHECKING:
    # Only for the type hint below -- never imported at runtime, so a game
    # with no belief_store given never needs h5py installed. See
    # tla.ai.belief_store's own module docstring.
    from tla.ai.belief_store import BeliefStore


def opponent_of(player: PlayerId, game_state: GameState) -> PlayerId:
    """The other player -- used to construct an `EnemyModel` for `player`
    without hardcoding which of `PLAYER_A`/`PLAYER_B` that is."""
    return next(p for p in game_state.players if p != player)


def _initial_candidate_hexes(board: Board, ports: list[AxialCoord]) -> set[AxialCoord]:
    """Every sea hex `tla.fleet_setup._pick_start_hex` could have placed a
    starting ship on, for the whole starting fleet at once -- the same
    growing-radius-over-ports-and-main-sea-component search, minus the RNG
    draw (this wants the whole candidate set, uniformly weighted, not one
    sample) and minus that function's own `occupied` exclusion
    (irreproducible without the game's seed; removes at most a handful of
    hexes from what's normally a much larger set, a negligible fidelity
    loss)."""
    main_sea = largest_sea_component(board)
    max_radius = board.width + board.height
    radius = 1
    while radius <= max_radius:
        candidates: set[AxialCoord] = set()
        for port in ports:
            candidates.update(hexes_in_range(port, radius))
        valid = candidates & main_sea
        if valid:
            return valid
        radius += 1
    return set()


def _diffusion_steps_for(kind: ShipKind, stats: dict[ShipKind, ShipStats]) -> int:
    """Movement points to diffuse this kind's belief by for one turn --
    always the surfaced/higher budget, including for submarines: relying
    on the lower submerged budget would need reliable continuity on a
    transient tactical toggle we don't have, and overestimating spread is
    the safe direction for anything defensive consuming this model."""
    return stats[kind].movement


def _detectable_if_present(kind: ShipKind, last_known_surfaced: bool | None) -> bool:
    """Whether a ship of `kind` sitting in a hex we can currently see
    would actually have been sighted there -- see `tla.fow.is_hidden`.
    False only for a submarine not confidently known to be surfaced
    (never individually resolved yet, or last seen submerged): stealth
    beats ordinary vision even within it, so its last known hex stays a
    live possibility despite being visible to us right now. True for
    every other kind, and for a submarine last seen surfaced."""
    if kind != ShipKind.SUBMARINE:
        return True
    return last_known_surfaced is True


@dataclass
class TrackedShip:
    """An enemy ship whose real id we've personally resolved (a sighting,
    or a battle we generated ourselves) at least once. Position/HP are
    "as of last observed", not necessarily current -- see module
    docstring. `field` is None exactly while currently in vision (nothing
    to diffuse: we already know exactly where it is); otherwise a
    `HexField` seeded from `last_seen_position` and diffusing forward."""

    ship_id: int
    kind: ShipKind
    last_seen_turn: int
    last_seen_position: AxialCoord
    last_known_hp: int
    last_known_surfaced: bool | None  # only meaningful for ShipKind.SUBMARINE
    field: HexField | None = None


@dataclass
class KindPool:
    """Belief for every enemy ship of `kind` never yet individually
    resolved -- the unsighted remainder of the starting fleet, plus
    deterministically-detected-but-unsighted production, plus any
    `TrackedShip` folded back after going stale. `count` is exact (see
    module docstring); `field`'s total mass is kept equal to it."""

    kind: ShipKind
    count: int
    field: HexField


class EnemyModel:
    """One player's belief about their opponent's fleet, persisted across
    turns on the owning `NaivePolicy` instance (see
    `tla.ai.policy.NaivePolicy._enemy_models`) exactly like task forces
    are -- never part of `GameState`, never visible to the UI or a human
    player."""

    def __init__(
        self,
        game_state: GameState,
        player: PlayerId,
        opponent: PlayerId,
        belief_store: "BeliefStore | None" = None,
    ) -> None:
        self.player = player
        self.opponent = opponent
        self._ship_stats = game_state.config.ship_stats.stats
        self._geometry = HexFieldGeometry.from_board(game_state.board)
        self._tracked: dict[int, TrackedShip] = {}
        self._pools: dict[ShipKind, KindPool] = {}
        self._port_next_index_seen: dict[AxialCoord, int] = {}
        self._controlled_ports_seen: set[AxialCoord] = set()
        # Optional -- see tla.ai.belief_store.BeliefStore. Duck-typed
        # (only ever called via .append(ship_id, values)), so this module
        # never has to import it or h5py itself.
        self._belief_store = belief_store
        self._seed_initial_fleet(game_state)

    def _pool_for(self, kind: ShipKind) -> KindPool:
        if kind not in self._pools:
            self._pools[kind] = KindPool(kind=kind, count=0, field=HexField(self._geometry))
        return self._pools[kind]

    def _seed_initial_fleet(self, game_state: GameState) -> None:
        candidates = _initial_candidate_hexes(game_state.board, game_state.board.ports_for(self.opponent))
        for kind, count in game_state.config.fleet.counts.items():
            if count <= 0:
                continue
            pool = self._pool_for(kind)
            pool.count += count
            for coord in candidates:
                pool.field.add_point_mass(coord, 1.0)
            pool.field.renormalize_to(pool.count)
        # Baseline the production-diffing state to what's already true at
        # construction time, for every port the opponent already controls
        # -- not just ones with an existing PortProduction entry (a port's
        # entry is only created lazily by run_production on its first
        # production tick, via progress.setdefault; a port controlled
        # since game start may have no entry at all yet at construction,
        # since EnemyModel is always built before any production has ever
        # run -- see tla.ai.policy.NaivePolicy._enemy_model_for). 0 is the
        # correct baseline for those: nothing has been built there yet.
        # Without this, that port's first-ever spawn would be silently
        # missed the first time _observe_production sees a real entry for
        # it, since there'd be no recorded baseline to diff against.
        port_production = game_state.players[self.opponent].port_production
        self._controlled_ports_seen = set(game_state.board.controlled_ports_for(self.opponent))
        for port in self._controlled_ports_seen:
            state = port_production.get(port)
            self._port_next_index_seen[port] = state.next_index if state is not None else 0

    def begin_turn(self, game_state: GameState) -> None:
        """Call once at the very start of this player's `plan_movement`,
        before anything else -- advances belief by one turn, then folds in
        whatever's newly known. `game_state.battle_log` is guaranteed
        empty here (just cleared by the prior turn transition), so this
        never sees combat -- only `end_turn` does.

        `_ensure_fields_for_unseen` runs *before* `_diffuse_all`, using
        the same visibility snapshot both use -- not after, the way
        `end_turn` orders it. A ship the opponent just moved out of our
        vision (between our last call and now) needs its fresh,
        undiffused field to exist *before* this turn's one diffusion
        pass, or it would miss that pass entirely (no field yet when
        `_diffuse_all` runs) and `field_as_of` would keep reporting its
        last sighting as still 100% certain for a full extra turn after
        it was actually lost from view -- a real bug caught by a replay
        review, not a hypothetical."""
        visible = enemy_ships_visible_to(game_state, self.player)
        self._ensure_fields_for_unseen(visible)
        self._diffuse_all(game_state)
        self._observe_production(game_state)
        self._apply_sightings(game_state, visible)
        self._fold_back_stale(game_state)

    def end_turn(self, game_state: GameState) -> None:
        """Call once as the very last action of this player's
        `plan_movement`, after everything else -- `game_state.battle_log`
        at this point holds exactly this half-turn's own battles (not yet
        cleared; that happens in the *next* `end_movement_phase` call), so
        this is the one place combat updates belief. No diffusion here --
        that only happens once per turn, in `begin_turn`."""
        self._observe_own_attacks(game_state)
        visible = enemy_ships_visible_to(game_state, self.player)
        self._apply_sightings(game_state, visible)
        self._ensure_fields_for_unseen(visible)

    def _ordinal(self, game_state: GameState) -> int:
        """`turn_ordinal` for right now -- `game_state.phase` at the
        moment `begin_turn`/`end_turn` run is always `self.player`'s own
        phase (this model's owner is exactly who `plan_movement` is being
        called for), so this always reflects when a belief_store write
        actually happens, not just which turn_number it shares with the
        other half of the turn."""
        return turn_ordinal(game_state.turn_number, game_state.phase == TurnPhase.MOVE_B)

    def _diffuse_all(self, game_state: GameState) -> None:
        """Advance every currently-diffusing field (tracked ships out of
        sight, and every kind's pool) by one turn's worth of movement,
        then -- once, after all of this turn's steps, not after each one
        individually -- exclude any hex within `self.player`'s own
        *current* vision (`tla.fow.visible_hexes_for`) from a field whose
        ships would actually have been detected there: an undetected ship
        genuinely cannot be sitting somewhere we're already looking, not
        merely less likely to be (see `HexField.exclude_and_renormalize`;
        a real case a replay review caught -- a tracked battleship still
        showing nonzero belief on a hex a friendly carrier had well
        within its own vision radius). The one exception is a submarine
        not confidently known to be surfaced: stealth defeats ordinary
        vision even within it (see `tla.fow.is_hidden`), so its last
        known hex stays a live possibility despite being visible to us
        right now.

        Excluding once at the end, not after every individual diffuse
        step, matters: cutting and renormalizing repeatedly, once per
        step, can cascade a field to *total* collapse (zero mass
        everywhere, forever, since diffusing an all-zero field stays
        all-zero) whenever a ship's whole reachable area for one turn
        happens to sit inside our own vision radius even briefly mid-turn
        -- each cut re-concentrates the survivors right back at the
        vision boundary, where the next step immediately re-diffuses some
        of them straight back in. Letting the full N-step diffusion run
        uninterrupted first, the way ordinary (vision-unaware) diffusion
        already does, gives the genuinely-reachable-but-unseen area a
        fair chance to be reached before any cutting happens -- a second
        real case a replay review caught, distinct from the first.

        `AiConfig.enemy_model_directional_diffusion` (the default) biases
        every kind pool's diffusion this turn toward `self.player`'s own
        currently most-threatened controlled port (see
        `_most_threatened_own_port`), computed once here -- not once per
        `HexField.diffuse_step` call, and not once per pool -- and reused
        across every pool and every one of its (up to 6, for the fastest
        kind) diffusion steps this turn. A `TrackedShip.field` never
        receives this bias -- see that loop below, unchanged."""
        own_visible_indices = self._own_visible_indices(game_state)
        bias_grid = None
        if game_state.config.ai.enemy_model_directional_diffusion:
            target_port = self._most_threatened_own_port(game_state)
            if target_port is not None:
                bias_grid = self._geometry.distance_grid_to(target_port)
        for tracked in self._tracked.values():
            if tracked.field is None:
                continue
            for _ in range(_diffusion_steps_for(tracked.kind, self._ship_stats)):
                tracked.field.diffuse_step()
                if self._belief_store is not None:
                    # One layer per recompute, not per turn -- a ship
                    # with N movement gets up to N layers this turn.
                    # KindPool fields aren't recorded: they have no
                    # single ship id to name a dataset after.
                    self._belief_store.append(tracked.ship_id, self._ordinal(game_state), tracked.field.values)
            if own_visible_indices and _detectable_if_present(tracked.kind, tracked.last_known_surfaced):
                tracked.field.exclude_and_renormalize(own_visible_indices)
                if self._belief_store is not None:
                    # A further, corrected layer for the same turn --
                    # field_as_of always returns the *last* layer for a
                    # turn, so this is what a reader actually sees;
                    # the raw pre-exclusion steps above stay in the file
                    # too, for anyone wanting the uncorrected progression.
                    self._belief_store.append(tracked.ship_id, self._ordinal(game_state), tracked.field.values)
        for pool in self._pools.values():
            for _ in range(_diffusion_steps_for(pool.kind, self._ship_stats)):
                pool.field.diffuse_step(bias_distance_grid=bias_grid)
            if own_visible_indices and _detectable_if_present(pool.kind, None):
                pool.field.exclude_and_renormalize(own_visible_indices)

    def _most_threatened_own_port(self, game_state: GameState) -> AxialCoord | None:
        """`self.player`'s own controlled ports, ranked by this model's own
        currently-believed total mass nearby (`mass_near`, all kinds --
        this is about where any unsighted ship should be assumed to be
        heading, not `tla.ai.global_strategy.DANGEROUS_TO_CARRIER_KINDS`-
        specific carrier-defense triggering). `None` if `self.player`
        controls no ports -- the caller's fallback is plain isotropic
        diffusion that turn, same as before this existed.

        Deliberately self-contained rather than reusing `tla.ai.
        global_strategy.rank_port_threats`, which does the same "rank our
        own ports by believed nearby threat" query already: that module
        imports `EnemyModel` (and `tla.ai.task_force`, which it also
        imports, imports `EnemyModel` too), so this module must never
        import back from either -- see this module's own docstring's
        fog-of-war/layering notes. Reuses `AiConfig.port_defense_trigger_
        radius` for consistency with `rank_port_threats`'s own radius
        choice, even though this is a separate computation."""
        ports = game_state.board.controlled_ports_for(self.player)
        if not ports:
            return None
        radius = game_state.config.ai.port_defense_trigger_radius
        return max(ports, key=lambda p: self.mass_near(p, radius))

    def _own_visible_indices(self, game_state: GameState) -> list[tuple[int, int]]:
        indices = []
        for coord in visible_hexes_for(game_state, self.player):
            row, col = self._geometry.to_index(coord)
            if self._geometry.in_bounds(row, col):
                indices.append((row, col))
        return indices

    def _observe_production(self, game_state: GameState) -> None:
        build_order = game_state.config.production.build_order
        current_ports = set(game_state.board.controlled_ports_for(self.opponent))
        port_production = game_state.players[self.opponent].port_production

        for port in current_ports - self._controlled_ports_seen:
            # Just captured (or the very first time we've observed it) --
            # production there just reset to the start, so there's no
            # prior index to diff against; take whatever it reports now as
            # the new baseline rather than reading a jump from 0 as spawns.
            self._port_next_index_seen[port] = port_production.get(port, PortProduction()).next_index

        for port in current_ports:
            state = port_production.get(port)
            if state is None:
                continue
            # The two loops above (constructor, and "newly controlled"
            # just above) mean every currently-controlled port always has
            # a baseline recorded by this point -- the 0 default is just a
            # safe fallback, not expected to ever actually apply.
            seen = self._port_next_index_seen.get(port, 0)
            for i in range(seen, state.next_index):
                kind = build_order[i % len(build_order)]
                pool = self._pool_for(kind)
                pool.count += 1
                pool.field.add_point_mass(port, 1.0)
            self._port_next_index_seen[port] = state.next_index

        for port in list(self._port_next_index_seen):
            if port not in current_ports:
                del self._port_next_index_seen[port]
        self._controlled_ports_seen = current_ports

    def _resolve(
        self, ship_id: int, kind: ShipKind, position: AxialCoord, hp: int, turn: int, surfaced: bool | None = None
    ) -> None:
        tracked = self._tracked.get(ship_id)
        if tracked is None:
            pool = self._pool_for(kind)
            if pool.count > 0:
                pool.count -= 1
                pool.field.renormalize_to(pool.count)
            self._tracked[ship_id] = TrackedShip(
                ship_id=ship_id,
                kind=kind,
                last_seen_turn=turn,
                last_seen_position=position,
                last_known_hp=hp,
                last_known_surfaced=surfaced,
                field=None,
            )
            return
        tracked.kind = kind
        tracked.last_seen_turn = turn
        tracked.last_seen_position = position
        tracked.last_known_hp = hp
        if surfaced is not None:
            tracked.last_known_surfaced = surfaced
        tracked.field = None

    def _apply_sightings(self, game_state: GameState, visible: dict[int, Ship]) -> None:
        for ship_id, ship in visible.items():
            self.observe_ship_state(
                ship_id, ship.kind, ship.position, ship.current_hp, game_state, surfaced=ship.surfaced
            )

    def observe_ship_state(
        self,
        ship_id: int,
        kind: ShipKind,
        position: AxialCoord,
        hp: int,
        game_state: GameState,
        surfaced: bool | None = None,
    ) -> None:
        """Record a direct, exact observation of one enemy ship -- either
        an ordinary sighting (`_apply_sightings`, above) or the position/
        kind/hp an enemy ship reveals about itself by attacking one of
        `self.player`'s own ships (see `NaivePolicy.plan_movement`'s
        cross-model reconciliation, the other caller of this from outside
        this class: being attacked reveals the attacker, the same way
        directly spotting it would -- a defender obviously knows who and
        where just hit them, even if they'd never have looked there on
        their own). Public because that reconciliation updates a
        *different* player's `EnemyModel` than the one whose `plan_
        movement` is currently running."""
        self._resolve(ship_id, kind, position, hp, game_state.turn_number, surfaced=surfaced)
        if self._belief_store is not None:
            self._record_sighting_layer(ship_id, position, self._ordinal(game_state))

    def observe_ship_sunk(self, ship_id: int, kind: ShipKind) -> None:
        """Record that an enemy ship is confirmed dead -- either one of
        `self.player`'s own kills (`_observe_own_attacks`, below) or one
        that died attacking `self.player`'s own ship (see
        `observe_ship_state`'s docstring for why that's legitimate
        knowledge; the same cross-model reconciliation calls this too)."""
        if ship_id in self._tracked:
            del self._tracked[ship_id]
            return
        pool = self._pool_for(kind)
        if pool.count > 0:
            pool.count -= 1
            pool.field.renormalize_to(pool.count)

    def _record_sighting_layer(self, ship_id: int, position: AxialCoord, ordinal: int) -> None:
        """A direct sighting collapses belief to certainty -- `_resolve`
        already reflects that in memory (`field = None`), but that alone
        leaves no trace in `self._belief_store`: only `_diffuse_all` ever
        writes a layer, and only while a ship is *not* currently visible.
        Without this, a reader of the stored file (e.g. the replay
        viewer) could only ever see the diffuse, pre-sighting cloud from
        the moment before a ship was spotted -- never the moment it
        actually became exactly known. Record that certainty as its own
        layer (point mass 1.0 at `position`) so `BeliefReader.field_as_of`
        can return it from this moment onward, until the next time this
        ship goes back out of vision and starts diffusing again."""
        field = HexField(self._geometry)
        field.set_point_mass(position)
        self._belief_store.append(ship_id, ordinal, field.values)

    def _ensure_fields_for_unseen(self, visible: dict[int, Ship]) -> None:
        for ship_id, tracked in self._tracked.items():
            if tracked.field is None and ship_id not in visible:
                tracked.field = HexField(self._geometry)
                tracked.field.set_point_mass(tracked.last_seen_position)
                if tracked.kind == ShipKind.SUBMARINE:
                    # A submarine last seen surfaced is *not* still
                    # confidently surfaced the instant it goes unseen --
                    # toggling is free and untracked while hidden, so it
                    # may have dived right where we lost it. Leaving
                    # last_known_surfaced at its stale True here would make
                    # _detectable_if_present treat it as "would be seen if
                    # present," so _diffuse_all's own-vision exclusion
                    # wipes its belief out of our own vision radius --
                    # exactly backwards, since that's the most likely place
                    # a freshly-submerged sub actually is. Caught via a
                    # real replay review: a sub sighted surfaced near a
                    # controlled port, then submerged, showed zero believed
                    # mass at its last known hex despite one turn passing.
                    tracked.last_known_surfaced = None

    def _observe_own_attacks(self, game_state: GameState) -> None:
        for entry in game_state.battle_log:
            if entry.attacker_owner != self.player or entry.defender_owner != self.opponent:
                continue
            if entry.defender_sunk:
                self.observe_ship_sunk(entry.defender_id, entry.defender_kind)
                continue
            self.observe_ship_state(
                entry.defender_id, entry.defender_kind, entry.battle_hex, entry.defender_hp_after, game_state
            )

    def _fold_back_stale(self, game_state: GameState) -> None:
        threshold = game_state.config.ai.enemy_model_stale_turns
        stale_ids = [
            sid
            for sid, tracked in self._tracked.items()
            if tracked.field is not None and game_state.turn_number - tracked.last_seen_turn > threshold
        ]
        for sid in stale_ids:
            tracked = self._tracked.pop(sid)
            pool = self._pool_for(tracked.kind)
            pool.count += 1
            assert tracked.field is not None
            pool.field.add_field(tracked.field)

    # -- query surface for later layers / tests --------------------------

    def tracked_ships(self) -> dict[int, TrackedShip]:
        """Read-only view -- never mutate the returned dict or its
        entries."""
        return self._tracked

    def kind_pools(self) -> dict[ShipKind, KindPool]:
        """Read-only view -- never mutate the returned dict or its
        entries."""
        return self._pools

    def alive_count(self, kind: ShipKind | None = None) -> int:
        total = sum(1 for t in self._tracked.values() if kind is None or t.kind == kind)
        total += sum(p.count for p in self._pools.values() if kind is None or p.kind == kind)
        return total

    def most_likely_hex(self, ship_id: int) -> AxialCoord | None:
        tracked = self._tracked.get(ship_id)
        if tracked is None:
            return None
        if tracked.field is None:
            return tracked.last_seen_position
        return tracked.field.most_likely_hex()

    def mass_near(self, coord: AxialCoord, radius: int, kinds: frozenset[ShipKind] | None = None) -> float:
        total = 0.0
        for tracked in self._tracked.values():
            if kinds is not None and tracked.kind not in kinds:
                continue
            if tracked.field is not None:
                total += tracked.field.mass_near(coord, radius)
            elif distance(tracked.last_seen_position, coord) <= radius:
                total += 1.0
        for pool in self._pools.values():
            if kinds is not None and pool.kind not in kinds:
                continue
            total += pool.field.mass_near(coord, radius)
        return total

    def expected_strength_near(
        self, coord: AxialCoord, radius: int, kinds: frozenset[ShipKind] | None = None
    ) -> tuple[int, int]:
        """(believed hp, believed damage-per-round) within `radius` sea
        hexes of `coord` -- the same `(hp, damage)` shape `group_power`/
        `expected_strength` use, so it's directly comparable via
        `outmatched`. Mirrors `mass_near`'s exact traversal (same tracked/
        pool fields, same `kinds` filter), but weights each contribution
        by its believed probability mass near `coord` instead of just
        counting it: a tracked ship contributes its `last_known_hp`/
        kind's `damage` scaled by `field.mass_near(coord, radius)` (or in
        full, same as `mass_near`, if resolved-but-fieldless and within
        `radius`); a kind pool contributes its full per-ship stats scaled
        by its own expected count near `coord`. Rounds to the nearest int
        to match `group_power`'s own return type."""
        total_hp = 0.0
        total_damage = 0.0
        for tracked in self._tracked.values():
            if kinds is not None and tracked.kind not in kinds:
                continue
            stats = self._ship_stats[tracked.kind]
            if tracked.field is not None:
                fraction = tracked.field.mass_near(coord, radius)
            elif distance(tracked.last_seen_position, coord) <= radius:
                fraction = 1.0
            else:
                fraction = 0.0
            total_hp += fraction * tracked.last_known_hp
            total_damage += fraction * stats.damage
        for pool in self._pools.values():
            if kinds is not None and pool.kind not in kinds:
                continue
            stats = self._ship_stats[pool.kind]
            fraction = pool.field.mass_near(coord, radius)
            total_hp += fraction * stats.hp
            total_damage += fraction * stats.damage
        return round(total_hp), round(total_damage)

    def expected_strength(self, kinds: frozenset[ShipKind] | None = None) -> tuple[int, int]:
        """(total expected HP, total expected damage-per-round) across
        every enemy ship this model believes is alive -- last-observed HP
        for a resolved ship, full stats HP for anything still only in a
        pool (a pooled ship's true HP is unknowable; full HP is the
        least-wrong default). Same upper-bound caveat as `alive_count` --
        see the module docstring."""
        total_hp = 0
        total_damage = 0
        for tracked in self._tracked.values():
            if kinds is not None and tracked.kind not in kinds:
                continue
            total_hp += tracked.last_known_hp
            total_damage += self._ship_stats[tracked.kind].damage
        for pool in self._pools.values():
            if kinds is not None and pool.kind not in kinds:
                continue
            stats = self._ship_stats[pool.kind]
            total_hp += pool.count * stats.hp
            total_damage += pool.count * stats.damage
        return total_hp, total_damage
