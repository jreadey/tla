from replay_viewer import _battle_lines, _task_force_lines, summarize_replay


def _initial_record():
    return {
        "type": "initial",
        "seed": 7,
        "board": {
            "width": 5,
            "height": 5,
            "tiles": [
                {
                    "coord": [0, 0],
                    "terrain": "land",
                    "is_port": True,
                    "port_owner": 1,
                    "port_controller": None,
                },
                {"coord": [1, 0], "terrain": "sea", "is_port": False, "port_owner": None, "port_controller": None},
            ],
        },
        "ships": [
            {"id": 1, "kind": "destroyer", "owner": 1, "position": [1, 0], "hp": 6, "surfaced": True, "movement_remaining": 3},
            {"id": 2, "kind": "submarine", "owner": 2, "position": [3, 0], "hp": 4, "surfaced": True, "movement_remaining": 3},
        ],
    }


def _half_turn(ships, ports, *, turn_number=1, player=1, phase="move_a", turn_stats=None):
    return {
        "type": "half_turn",
        "turn_number": turn_number,
        "player": player,
        "phase": phase,
        "ships": ships,
        "ports": ports,
        "turn_stats": turn_stats or {"1": {"hp_dealt": 0, "hp_taken": 0, "ships_lost": []}, "2": {"hp_dealt": 0, "hp_taken": 0, "ships_lost": []}},
    }


def test_summarize_reports_ship_movement_and_damage():
    initial = _initial_record()
    moved = list(initial["ships"])
    moved[0] = {**moved[0], "position": [2, 0], "hp": 4}
    record = _half_turn(moved, [{"coord": [0, 0], "port_owner": 1, "port_controller": None, "display_owner": 1}])

    lines = summarize_replay([initial, record])

    assert any("Ship 1" in line and "moved (1, 0) -> (2, 0)" in line and "hp 6 -> 4" in line for line in lines)


def test_summarize_reports_sunk_ship():
    initial = _initial_record()
    remaining = [initial["ships"][0]]  # ship 2 disappears -- sunk
    record = _half_turn(remaining, [{"coord": [0, 0], "port_owner": 1, "port_controller": None, "display_owner": 1}])

    lines = summarize_replay([initial, record])

    assert any("Ship 2" in line and "sunk" in line for line in lines)


def test_summarize_reports_new_ship_appearing():
    initial = _initial_record()
    grown = list(initial["ships"]) + [
        {"id": 3, "kind": "patrol_boat", "owner": 1, "position": [0, 0], "hp": 2, "surfaced": True, "movement_remaining": 6}
    ]
    record = _half_turn(grown, [{"coord": [0, 0], "port_owner": 1, "port_controller": None, "display_owner": 1}])

    lines = summarize_replay([initial, record])

    assert any("Ship 3" in line and "appeared" in line for line in lines)


def test_summarize_reports_port_capture():
    initial = _initial_record()
    record = _half_turn(
        initial["ships"], [{"coord": [0, 0], "port_owner": 1, "port_controller": 2, "display_owner": 2}]
    )

    lines = summarize_replay([initial, record])

    assert any("Port (0, 0) captured: P1 -> P2" in line for line in lines)


def test_summarize_reports_winner_on_final_record():
    initial = _initial_record()
    final = {
        "type": "final",
        "turn_number": 5,
        "winner": 1,
        "ships": initial["ships"],
        "ports": [{"coord": [0, 0], "port_owner": 1, "port_controller": None, "display_owner": 1}],
        "turn_stats": {"1": {"hp_dealt": 0, "hp_taken": 0, "ships_lost": []}, "2": {"hp_dealt": 0, "hp_taken": 0, "ships_lost": []}},
    }

    lines = summarize_replay([initial, final])

    assert any("GAME OVER" in line for line in lines)
    assert any("Winner: Player 1" in line for line in lines)


def test_task_force_lines_reports_a_goal():
    record = {
        "task_forces": [
            {
                "id": 3,
                "owner": 2,
                "member_ids": [8, 26],
                "goal": {"kind": "blockade", "target": [4, 1]},
                "turns_since_progress": 0,
                "best_progress_distance": None,
                "retreating": False,
                "retreat_turns": 0,
                "retreat_threat_power": None,
            }
        ]
    }

    lines = _task_force_lines(record)

    assert len(lines) == 1
    assert "TF3" in lines[0]
    assert "P2" in lines[0]
    assert "[8, 26]" in lines[0]
    assert "goal=blockade@(4, 1)" in lines[0]


def test_task_force_lines_reports_no_goal_and_flags():
    record = {
        "task_forces": [
            {
                "id": 1,
                "owner": 1,
                "member_ids": [5],
                "goal": None,
                "turns_since_progress": 6,
                "best_progress_distance": 3,
                "retreating": True,
                "retreat_turns": 2,
                "retreat_threat_power": [12, 4],
            }
        ]
    }

    lines = _task_force_lines(record)

    assert "goal=none" in lines[0]
    assert "retreating, 2 turns so far" in lines[0]
    assert "need power > (12, 4)" in lines[0]
    assert "no progress for 6 turns" in lines[0]


def test_task_force_lines_empty_when_absent():
    assert _task_force_lines({}) == []
    assert _task_force_lines({"task_forces": []}) == []


def test_summarize_includes_task_force_lines():
    initial = _initial_record()
    record = _half_turn(initial["ships"], [{"coord": [0, 0], "port_owner": 1, "port_controller": None, "display_owner": 1}])
    record["task_forces"] = [
        {
            "id": 1,
            "owner": 2,
            "member_ids": [2],
            "goal": {"kind": "capture_port", "target": [0, 0]},
            "turns_since_progress": 0,
            "best_progress_distance": None,
            "retreating": False,
            "retreat_turns": 0,
            "retreat_threat_power": None,
        }
    ]

    lines = summarize_replay([initial, record])

    assert any("TF1" in line and "goal=capture_port@(0, 0)" in line for line in lines)


def test_summarize_reports_nonzero_turn_stats():
    initial = _initial_record()
    record = _half_turn(
        initial["ships"],
        [{"coord": [0, 0], "port_owner": 1, "port_controller": None, "display_owner": 1}],
        turn_stats={
            "1": {"hp_dealt": 4, "hp_taken": 0, "ships_lost": []},
            "2": {"hp_dealt": 0, "hp_taken": 4, "ships_lost": ["submarine"]},
        },
    )

    lines = summarize_replay([initial, record])

    assert any("Player 2 stats" in line and "lost=['submarine']" in line for line in lines)


def _round(
    *,
    attacker_id=1,
    attacker_kind="destroyer",
    attacker_owner=1,
    defender_id=2,
    defender_kind="cruiser",
    defender_owner=2,
    battle_hex=(3, 0),
    damage_to_defender=4,
    damage_to_attacker=2,
    attacker_carrier_bonus=0,
    defender_carrier_bonus=0,
    defender_hp_after=4,
    attacker_hp_after=4,
    defender_sunk=False,
    attacker_sunk=False,
):
    return {
        "attacker_id": attacker_id,
        "attacker_kind": attacker_kind,
        "attacker_owner": attacker_owner,
        "defender_id": defender_id,
        "defender_kind": defender_kind,
        "defender_owner": defender_owner,
        "battle_hex": list(battle_hex),
        "damage_to_defender": damage_to_defender,
        "damage_to_attacker": damage_to_attacker,
        "attacker_carrier_bonus": attacker_carrier_bonus,
        "defender_carrier_bonus": defender_carrier_bonus,
        "defender_hp_after": defender_hp_after,
        "attacker_hp_after": attacker_hp_after,
        "defender_sunk": defender_sunk,
        "attacker_sunk": attacker_sunk,
    }


def test_battle_lines_empty_when_absent():
    assert _battle_lines({}) == []
    assert _battle_lines({"battle_log": []}) == []


def test_battle_lines_reports_a_single_round_battle():
    record = {"battle_log": [_round(defender_hp_after=0, defender_sunk=True)]}

    lines = _battle_lines(record)

    assert any("P1 destroyer1 vs P2 cruiser2" in line and "(3, 0)" in line for line in lines)
    assert any("round 1: dealt 4/2" in line for line in lines)
    assert any("defender sunk" in line for line in lines)


def test_battle_lines_groups_consecutive_rounds_between_the_same_pair():
    record = {
        "battle_log": [
            _round(damage_to_defender=4, defender_hp_after=4),
            _round(damage_to_defender=4, defender_hp_after=0, defender_sunk=True),
        ]
    }

    lines = _battle_lines(record)

    headers = [line for line in lines if "Battle:" in line]
    assert len(headers) == 1  # one battle, not two
    assert any("round 1:" in line for line in lines)
    assert any("round 2:" in line for line in lines)


def test_battle_lines_separates_different_attacker_defender_pairs():
    record = {
        "battle_log": [
            _round(attacker_id=1, defender_id=2),
            _round(attacker_id=3, defender_id=4, attacker_kind="battleship", defender_kind="submarine"),
        ]
    }

    lines = _battle_lines(record)

    headers = [line for line in lines if "Battle:" in line]
    assert len(headers) == 2


def test_battle_lines_annotates_carrier_assist():
    record = {"battle_log": [_round(attacker_carrier_bonus=1, defender_carrier_bonus=0)]}

    lines = _battle_lines(record)

    assert any("attacker+1" in line for line in lines)
    assert not any("defender+" in line for line in lines)


def test_battle_lines_reports_all_four_outcomes():
    both_sunk = _battle_lines({"battle_log": [_round(defender_sunk=True, attacker_sunk=True)]})
    attacker_sunk = _battle_lines({"battle_log": [_round(attacker_sunk=True)]})
    defender_sunk = _battle_lines({"battle_log": [_round(defender_sunk=True)]})
    retreated = _battle_lines({"battle_log": [_round()]})

    assert any("both sunk" in line for line in both_sunk)
    assert any("attacker sunk" in line for line in attacker_sunk) and not any(
        "both sunk" in line for line in attacker_sunk
    )
    assert any("defender sunk" in line for line in defender_sunk) and not any(
        "both sunk" in line for line in defender_sunk
    )
    assert any("attacker retreated" in line for line in retreated)


def test_summarize_includes_battle_lines():
    initial = _initial_record()
    record = _half_turn(initial["ships"], [{"coord": [0, 0], "port_owner": 1, "port_controller": None, "display_owner": 1}])
    record["battle_log"] = [_round(defender_sunk=True)]

    lines = summarize_replay([initial, record])

    assert any("Battle:" in line for line in lines)
