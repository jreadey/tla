"""Post-game replay logging: one JSON Lines (.jsonl) file per game, written
turn-by-turn by tla.rendering.game_view as a game is played. One-way
serialization only -- these records are for human review (see
replay_viewer.py at the repo root), never deserialized back into a live
GameState. No Arcade dependency, so ReplayWriter is equally usable from a
headless script (e.g. self-play debugging).

`write_half_turn`/`write_final` optionally also record AI-internal task
force state (goal, retreat status -- see tla.ai.task_force) for whichever
side(s) a caller passes in, so a replay can explain *why* an AI-controlled
side's ships moved where they did, not just where they moved. Purely a
diagnostic add-on -- omit `task_forces` entirely for a human-only game or
a caller that doesn't care.

Every record also includes `battle_log`: every combat round resolved
during the half-turn just written (see `GameState.battle_log`), so a
replay can be followed battle-by-battle -- attacker/defender identity,
exact carrier-assist contribution, round-by-round damage -- instead of
reconstructed after the fact from ship position/HP snapshots. Read
directly off `game_state`, unlike `task_forces` -- no separate param, and
never empty-by-omission the way `task_forces` can be.

`write_half_turn`/`write_final` also optionally record each AI-controlled
side's global-layer posture (see `tla.ai.global_strategy` and
`tla.ai.policy.NaivePolicy.posture_snapshot_for`) -- the posture value
itself plus the exact `(own, believed-enemy)` `(hp, damage)` totals it was
computed from. This exists because reconstructing that belief after the
fact from the rest of the replay log is lossy (the log doesn't capture
each port's internal production-queue state, so a reconstructed
`EnemyModel` can badly undercount a growing enemy fleet) -- logging it
directly avoids that reconstruction entirely. Same opt-in shape as
`task_forces`: omit for a human-only game or a caller that doesn't care.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from tla.ai.task_force import TaskForce
from tla.config import Config
from tla.game_state import BattleLogEntry, GameState, MoveLogEntry, TurnPhase, TurnStats
from tla.ship import Ship
from tla.tile import PlayerId, Tile


def _ship_dict(ship: Ship) -> dict:
    return {
        "id": ship.id,
        "kind": ship.kind.value,
        "owner": ship.owner,
        "position": [ship.position.q, ship.position.r],
        "hp": ship.current_hp,
        "surfaced": ship.surfaced,
        "movement_remaining": ship.movement_remaining,
    }


def _tile_dict(tile: Tile) -> dict:
    return {
        "coord": [tile.coord.q, tile.coord.r],
        "terrain": tile.terrain.value,
        "is_port": tile.is_port,
        "port_owner": tile.port_owner,
        "port_controller": tile.port_controller,
    }


def _port_dict(tile: Tile) -> dict:
    return {
        "coord": [tile.coord.q, tile.coord.r],
        "port_owner": tile.port_owner,
        "port_controller": tile.port_controller,
        "display_owner": tile.port_display_owner,
    }


def _turn_stats_dict(turn_stats: dict[PlayerId, TurnStats]) -> dict:
    return {
        str(player): {
            "hp_dealt": stats.hp_dealt,
            "hp_taken": stats.hp_taken,
            "ships_lost": [kind.value for kind in stats.ships_lost],
        }
        for player, stats in turn_stats.items()
    }


def _battle_log_entry_dict(entry: BattleLogEntry) -> dict:
    return {
        "attacker_id": entry.attacker_id,
        "attacker_kind": entry.attacker_kind.value,
        "attacker_owner": entry.attacker_owner,
        "defender_id": entry.defender_id,
        "defender_kind": entry.defender_kind.value,
        "defender_owner": entry.defender_owner,
        "battle_hex": [entry.battle_hex.q, entry.battle_hex.r],
        "damage_to_defender": entry.damage_to_defender,
        "damage_to_attacker": entry.damage_to_attacker,
        "attacker_carrier_bonus": entry.attacker_carrier_bonus,
        "defender_carrier_bonus": entry.defender_carrier_bonus,
        "defender_hp_after": entry.defender_hp_after,
        "attacker_hp_after": entry.attacker_hp_after,
        "defender_sunk": entry.defender_sunk,
        "attacker_sunk": entry.attacker_sunk,
    }


def _move_log_entry_dict(entry: MoveLogEntry) -> dict:
    return {
        "ship_id": entry.ship_id,
        "kind": entry.kind.value,
        "owner": entry.owner,
        "path": [[c.q, c.r] for c in entry.path],
    }


def _config_dict(config: Config) -> dict:
    """`dataclasses.asdict` makes every field of Config JSON-safe except the
    three spots that use ShipKind as a dict key or list value -- asdict
    leaves enum instances as-is (they're not dataclasses/list/dict), which
    json.dumps can't handle as dict keys and shouldn't have to for list
    values either."""
    data = dataclasses.asdict(config)
    data["ship_stats"]["stats"] = {
        kind.value: stats for kind, stats in data["ship_stats"]["stats"].items()
    }
    data["fleet"]["counts"] = {
        kind.value: count for kind, count in data["fleet"]["counts"].items()
    }
    data["production"]["build_order"] = [kind.value for kind in data["production"]["build_order"]]
    return data


def _posture_dict(posture: dict[PlayerId, dict]) -> dict:
    """`posture` is `{player: NaivePolicy.posture_snapshot_for(player)}`,
    pre-filtered by the caller to drop `None` entries (players that
    haven't had `plan_movement` called for them yet, or aren't
    AI-controlled) -- this just stringifies the keys the same way
    `_turn_stats_dict` does."""
    return {str(player): snapshot for player, snapshot in posture.items()}


def _ports(game_state: GameState) -> list[dict]:
    return [_port_dict(tile) for tile in game_state.board.tiles.values() if tile.is_port]


def _task_force_dict(force: TaskForce) -> dict:
    """AI-internal task-force state -- never part of `GameState` itself
    (see `tla.ai.task_force`'s own module docstring), included here purely
    as an opt-in diagnostic so a replay can explain *why* an AI-controlled
    side's ships moved where they did, not just where they moved."""
    goal = (
        {"kind": force.goal.kind.value, "target": [force.goal.target.q, force.goal.target.r]}
        if force.goal is not None
        else None
    )
    return {
        "id": force.id,
        "owner": force.owner,
        "member_ids": sorted(force.member_ids),
        "goal": goal,
        "turns_since_progress": force.turns_since_progress,
        "best_progress_distance": force.best_progress_distance,
        "retreating": force.retreating,
        "retreat_turns": force.retreat_turns,
        "retreat_threat_power": list(force.retreat_threat_power) if force.retreat_threat_power else None,
    }


class ReplayWriter:
    """Writes one JSON object per line to `path`: an "initial" record, then
    one "half_turn" record per player movement phase, then a single "final"
    record once the game ends. Each write flushes immediately -- turn
    frequency is low, and flushing means a crash mid-game doesn't lose the
    log."""

    def __init__(self, path: str | Path) -> None:
        self._file = open(path, "w")
        self.finalized = False

    def _write(self, record: dict) -> None:
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def write_initial(
        self, game_state: GameState, *, seed: int | None = None, belief_path: str | None = None
    ) -> None:
        """`belief_path` (as given on the command line, not resolved) is
        just carried along for `replay_gui.py` to auto-locate the paired
        HDF5 belief file (see `tla.ai.belief_store`) without needing it
        passed again explicitly -- None if `--belief` wasn't used."""
        board = game_state.board
        self._write(
            {
                "type": "initial",
                "seed": seed,
                "belief_path": belief_path,
                "config": _config_dict(game_state.config),
                "board": {
                    "width": board.width,
                    "height": board.height,
                    "hex_pixel_size": board.hex_pixel_size,
                    "tiles": [_tile_dict(tile) for tile in board.tiles.values()],
                },
                "ships": [_ship_dict(ship) for ship in game_state.ships.values()],
            }
        )

    def write_half_turn(
        self,
        game_state: GameState,
        *,
        phase: TurnPhase,
        player: PlayerId,
        task_forces: list[TaskForce] | None = None,
        posture: dict[PlayerId, dict] | None = None,
    ) -> None:
        self._write(
            {
                "type": "half_turn",
                "turn_number": game_state.turn_number,
                "player": player,
                "phase": phase.value,
                "ships": [_ship_dict(ship) for ship in game_state.ships.values()],
                "ports": _ports(game_state),
                "turn_stats": _turn_stats_dict(game_state.turn_stats),
                "battle_log": [_battle_log_entry_dict(e) for e in game_state.battle_log],
                "move_log": [_move_log_entry_dict(e) for e in game_state.move_log],
                "task_forces": [_task_force_dict(f) for f in (task_forces or [])],
                "posture": _posture_dict(posture or {}),
            }
        )

    def write_final(
        self,
        game_state: GameState,
        *,
        task_forces: list[TaskForce] | None = None,
        posture: dict[PlayerId, dict] | None = None,
    ) -> None:
        """No-op if already finalized -- safe to call speculatively (e.g.
        every frame once the winner is set) without writing duplicate
        records or writing to a closed file."""
        if self.finalized:
            return
        self._write(
            {
                "type": "final",
                "turn_number": game_state.turn_number,
                "winner": game_state.winner,
                "ships": [_ship_dict(ship) for ship in game_state.ships.values()],
                "ports": _ports(game_state),
                "turn_stats": _turn_stats_dict(game_state.turn_stats),
                "battle_log": [_battle_log_entry_dict(e) for e in game_state.battle_log],
                "move_log": [_move_log_entry_dict(e) for e in game_state.move_log],
                "task_forces": [_task_force_dict(f) for f in (task_forces or [])],
                "posture": _posture_dict(posture or {}),
            }
        )
        self.finalized = True
        self._file.close()
