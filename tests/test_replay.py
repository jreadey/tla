import json

from tla.ai.task_force import GoalKind, TaskForce, TaskForceGoal
from tla.board import Board
from tla.config import Config
from tla.game_state import BattleLogEntry, GameState, MoveLogEntry, TurnPhase, TurnStats
from tla.hexgrid import AxialCoord
from tla.replay import (
    ReplayWriter,
    _battle_log_entry_dict,
    _config_dict,
    _move_log_entry_dict,
    _port_dict,
    _ship_dict,
    _task_force_dict,
    _tile_dict,
    _turn_stats_dict,
)
from tla.ship import Ship, ShipKind
from tla.tile import PLAYER_A, PLAYER_B, Tile, TerrainType


def _ship(owner, ship_id, kind=ShipKind.DESTROYER, position=AxialCoord(1, 2)) -> Ship:
    return Ship(
        id=ship_id,
        kind=kind,
        owner=owner,
        position=position,
        current_hp=5,
        surfaced=False,
        movement_remaining=2,
    )


def _game_state() -> GameState:
    board = Board(width=5, height=5)
    port = AxialCoord(0, 0)
    board.tiles[port] = Tile(
        coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A, port_controller=PLAYER_B
    )
    board.tiles[AxialCoord(1, 0)] = Tile(coord=AxialCoord(1, 0), terrain=TerrainType.SEA)
    ships = [_ship(PLAYER_A, 1), _ship(PLAYER_B, 2, kind=ShipKind.CARRIER)]
    return GameState(config=Config(), board=board, ships={s.id: s for s in ships})


def test_ship_dict_encodes_position_and_fields():
    ship = _ship(PLAYER_A, 1, position=AxialCoord(3, -2))

    data = _ship_dict(ship)

    assert data == {
        "id": 1,
        "kind": "destroyer",
        "owner": PLAYER_A,
        "position": [3, -2],
        "hp": 5,
        "surfaced": False,
        "movement_remaining": 2,
    }


def test_tile_dict_encodes_port_fields():
    tile = Tile(
        coord=AxialCoord(2, 1),
        terrain=TerrainType.LAND,
        is_port=True,
        port_owner=PLAYER_A,
        port_controller=PLAYER_B,
    )

    data = _tile_dict(tile)

    assert data == {
        "coord": [2, 1],
        "terrain": "land",
        "is_port": True,
        "port_owner": PLAYER_A,
        "port_controller": PLAYER_B,
    }


def test_port_dict_uses_display_owner():
    tile = Tile(coord=AxialCoord(0, 0), terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)

    data = _port_dict(tile)

    assert data["display_owner"] == PLAYER_A  # falls back to port_owner, never occupied

    tile.port_controller = PLAYER_B
    data = _port_dict(tile)

    assert data["display_owner"] == PLAYER_B


def test_turn_stats_dict_converts_keys_and_ship_kinds():
    stats = {
        PLAYER_A: TurnStats(hp_dealt=3, hp_taken=1, ships_lost=[ShipKind.PATROL_BOAT]),
        PLAYER_B: TurnStats(),
    }

    data = _turn_stats_dict(stats)

    assert data == {
        "1": {"hp_dealt": 3, "hp_taken": 1, "ships_lost": ["patrol_boat"]},
        "2": {"hp_dealt": 0, "hp_taken": 0, "ships_lost": []},
    }


def test_config_dict_is_json_safe_and_preserves_values():
    data = _config_dict(Config())

    json.dumps(data)  # must not raise -- would fail if any ShipKind survived as a dict key/value
    assert data["ship_stats"]["stats"]["battleship"]["hp"] == 12
    assert data["fleet"]["counts"]["carrier"] == 2
    assert "battleship" in data["production"]["build_order"]
    assert data["player_kinds"] == {PLAYER_A: "human", PLAYER_B: "human"}


def test_task_force_dict_encodes_a_goal():
    force = TaskForce(
        id=3,
        owner=PLAYER_B,
        member_ids={8, 26, 2},
        goal=TaskForceGoal(kind=GoalKind.BLOCKADE, target=AxialCoord(4, 1)),
        turns_since_progress=5,
        best_progress_distance=7,
        retreating=True,
        retreat_turns=2,
        retreat_threat_power=(12, 4),
    )

    data = _task_force_dict(force)

    assert data == {
        "id": 3,
        "owner": PLAYER_B,
        "member_ids": [2, 8, 26],
        "goal": {"kind": "blockade", "target": [4, 1]},
        "turns_since_progress": 5,
        "best_progress_distance": 7,
        "retreating": True,
        "retreat_turns": 2,
        "retreat_threat_power": [12, 4],
    }


def test_task_force_dict_encodes_no_goal_as_none():
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={5})

    data = _task_force_dict(force)

    assert data["goal"] is None


def test_write_initial_produces_one_parseable_record(tmp_path):
    gs = _game_state()
    path = tmp_path / "replay.jsonl"
    writer = ReplayWriter(path)

    writer.write_initial(gs, seed=42)

    lines = path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["type"] == "initial"
    assert record["seed"] == 42
    assert len(record["ships"]) == 2
    assert len(record["board"]["tiles"]) == 2
    json.dumps(record)  # sanity: config round-tripped cleanly too


def test_write_half_turn_captures_current_state(tmp_path):
    gs = _game_state()
    path = tmp_path / "replay.jsonl"
    writer = ReplayWriter(path)
    writer.write_initial(gs)

    gs.ships[1].current_hp = 3
    writer.write_half_turn(gs, phase=TurnPhase.MOVE_A, player=PLAYER_A)

    record = json.loads(path.read_text().splitlines()[1])
    assert record["type"] == "half_turn"
    assert record["turn_number"] == 1
    assert record["player"] == PLAYER_A
    assert record["phase"] == "move_a"
    assert record["ports"] == [
        {"coord": [0, 0], "port_owner": PLAYER_A, "port_controller": PLAYER_B, "display_owner": PLAYER_B}
    ]
    ship_1 = next(s for s in record["ships"] if s["id"] == 1)
    assert ship_1["hp"] == 3
    assert record["task_forces"] == []  # omitted -- human-only or caller doesn't care


def test_write_half_turn_includes_task_forces_when_given(tmp_path):
    gs = _game_state()
    path = tmp_path / "replay.jsonl"
    writer = ReplayWriter(path)
    writer.write_initial(gs)

    force = TaskForce(id=1, owner=PLAYER_B, member_ids={2}, goal=TaskForceGoal(kind=GoalKind.CAPTURE_PORT, target=AxialCoord(0, 0)))
    writer.write_half_turn(gs, phase=TurnPhase.MOVE_B, player=PLAYER_B, task_forces=[force])

    record = json.loads(path.read_text().splitlines()[1])
    assert len(record["task_forces"]) == 1
    assert record["task_forces"][0]["goal"] == {"kind": "capture_port", "target": [0, 0]}


def test_write_final_is_idempotent_and_closes_file(tmp_path):
    gs = _game_state()
    path = tmp_path / "replay.jsonl"
    writer = ReplayWriter(path)
    writer.write_initial(gs)

    gs.winner = PLAYER_A
    writer.write_final(gs)
    assert writer.finalized is True

    writer.write_final(gs)  # no-op: must not raise (file already closed) or duplicate a record

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    final_record = json.loads(lines[1])
    assert final_record["type"] == "final"
    assert final_record["winner"] == PLAYER_A
    assert final_record["task_forces"] == []


def test_write_final_includes_task_forces_when_given(tmp_path):
    gs = _game_state()
    path = tmp_path / "replay.jsonl"
    writer = ReplayWriter(path)
    writer.write_initial(gs)
    gs.winner = PLAYER_A

    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1}, retreating=True, retreat_turns=3)
    writer.write_final(gs, task_forces=[force])

    final_record = json.loads(path.read_text().splitlines()[1])
    assert final_record["task_forces"][0]["retreat_turns"] == 3


def test_battle_log_entry_dict_encodes_all_fields():
    entry = BattleLogEntry(
        attacker_id=1,
        attacker_kind=ShipKind.DESTROYER,
        attacker_owner=PLAYER_A,
        defender_id=2,
        defender_kind=ShipKind.CRUISER,
        defender_owner=PLAYER_B,
        battle_hex=AxialCoord(3, -1),
        damage_to_defender=5,
        damage_to_attacker=2,
        attacker_carrier_bonus=1,
        defender_carrier_bonus=0,
        defender_hp_after=3,
        attacker_hp_after=4,
        defender_sunk=False,
        attacker_sunk=False,
    )

    data = _battle_log_entry_dict(entry)

    assert data == {
        "attacker_id": 1,
        "attacker_kind": "destroyer",
        "attacker_owner": PLAYER_A,
        "defender_id": 2,
        "defender_kind": "cruiser",
        "defender_owner": PLAYER_B,
        "battle_hex": [3, -1],
        "damage_to_defender": 5,
        "damage_to_attacker": 2,
        "attacker_carrier_bonus": 1,
        "defender_carrier_bonus": 0,
        "defender_hp_after": 3,
        "attacker_hp_after": 4,
        "defender_sunk": False,
        "attacker_sunk": False,
    }


def test_write_half_turn_includes_battle_log(tmp_path):
    gs = _game_state()
    path = tmp_path / "replay.jsonl"
    writer = ReplayWriter(path)
    writer.write_initial(gs)

    gs.battle_log.append(
        BattleLogEntry(
            attacker_id=1,
            attacker_kind=ShipKind.DESTROYER,
            attacker_owner=PLAYER_A,
            defender_id=2,
            defender_kind=ShipKind.CARRIER,
            defender_owner=PLAYER_B,
            battle_hex=AxialCoord(1, 0),
            damage_to_defender=4,
            damage_to_attacker=1,
            attacker_carrier_bonus=0,
            defender_carrier_bonus=0,
            defender_hp_after=3,
            attacker_hp_after=4,
            defender_sunk=False,
            attacker_sunk=False,
        )
    )
    writer.write_half_turn(gs, phase=TurnPhase.MOVE_A, player=PLAYER_A)

    record = json.loads(path.read_text().splitlines()[1])
    assert len(record["battle_log"]) == 1
    assert record["battle_log"][0]["attacker_id"] == 1


def test_write_half_turn_battle_log_empty_when_nothing_fought(tmp_path):
    gs = _game_state()
    path = tmp_path / "replay.jsonl"
    writer = ReplayWriter(path)
    writer.write_initial(gs)

    writer.write_half_turn(gs, phase=TurnPhase.MOVE_A, player=PLAYER_A)

    record = json.loads(path.read_text().splitlines()[1])
    assert record["battle_log"] == []


def test_write_final_includes_battle_log(tmp_path):
    gs = _game_state()
    path = tmp_path / "replay.jsonl"
    writer = ReplayWriter(path)
    writer.write_initial(gs)
    gs.winner = PLAYER_A
    gs.battle_log.append(
        BattleLogEntry(
            attacker_id=1,
            attacker_kind=ShipKind.DESTROYER,
            attacker_owner=PLAYER_A,
            defender_id=2,
            defender_kind=ShipKind.CARRIER,
            defender_owner=PLAYER_B,
            battle_hex=AxialCoord(1, 0),
            damage_to_defender=4,
            damage_to_attacker=1,
            attacker_carrier_bonus=0,
            defender_carrier_bonus=0,
            defender_hp_after=0,
            attacker_hp_after=4,
            defender_sunk=True,
            attacker_sunk=False,
        )
    )

    writer.write_final(gs)

    final_record = json.loads(path.read_text().splitlines()[1])
    assert len(final_record["battle_log"]) == 1
    assert final_record["battle_log"][0]["defender_sunk"] is True


def test_move_log_entry_dict_encodes_all_fields():
    entry = MoveLogEntry(
        ship_id=1, kind=ShipKind.DESTROYER, owner=PLAYER_A, path=[AxialCoord(0, 0), AxialCoord(1, 0)]
    )

    data = _move_log_entry_dict(entry)

    assert data == {
        "ship_id": 1,
        "kind": "destroyer",
        "owner": PLAYER_A,
        "path": [[0, 0], [1, 0]],
    }


def test_write_half_turn_includes_move_log(tmp_path):
    gs = _game_state()
    path = tmp_path / "replay.jsonl"
    writer = ReplayWriter(path)
    writer.write_initial(gs)

    gs.move_log.append(
        MoveLogEntry(ship_id=1, kind=ShipKind.DESTROYER, owner=PLAYER_A, path=[AxialCoord(1, 2), AxialCoord(1, 0)])
    )
    writer.write_half_turn(gs, phase=TurnPhase.MOVE_A, player=PLAYER_A)

    record = json.loads(path.read_text().splitlines()[1])
    assert len(record["move_log"]) == 1
    assert record["move_log"][0]["ship_id"] == 1
    assert record["move_log"][0]["path"] == [[1, 2], [1, 0]]


def test_write_half_turn_move_log_empty_when_nothing_moved(tmp_path):
    gs = _game_state()
    path = tmp_path / "replay.jsonl"
    writer = ReplayWriter(path)
    writer.write_initial(gs)

    writer.write_half_turn(gs, phase=TurnPhase.MOVE_A, player=PLAYER_A)

    record = json.loads(path.read_text().splitlines()[1])
    assert record["move_log"] == []


def test_write_final_includes_move_log(tmp_path):
    gs = _game_state()
    path = tmp_path / "replay.jsonl"
    writer = ReplayWriter(path)
    writer.write_initial(gs)
    gs.winner = PLAYER_A
    gs.move_log.append(
        MoveLogEntry(ship_id=1, kind=ShipKind.DESTROYER, owner=PLAYER_A, path=[AxialCoord(1, 2), AxialCoord(1, 0)])
    )

    writer.write_final(gs)

    final_record = json.loads(path.read_text().splitlines()[1])
    assert len(final_record["move_log"]) == 1
    assert final_record["move_log"][0]["ship_id"] == 1
