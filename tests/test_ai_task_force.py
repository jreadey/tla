from collections import Counter

from tla.ai.policy import NaivePolicy
from tla.ai.task_force import (
    GoalKind,
    TaskForce,
    TaskForceGoal,
    _block_hex_toward_port,
    _choose_blocker,
    _gather,
    _pullback_waypoint,
    assign_goal,
    compute_port_defense_directives,
    find_chokepoint,
    force_recapture_goal,
    form_task_forces,
    is_attritting,
    is_open,
    is_outmatched_by_target_support,
    is_outnumbered,
    rally_point,
    record_task_force_progress,
    recruit_into_open_forces,
    repair_task_forces,
    sea_distance_field,
    update_task_force_goals,
    update_task_force_stance,
)
from tla.board import Board
from tla.config import AiConfig, Config, FowConfig
from tla.game_state import GameState
from tla.hexgrid import AxialCoord, distance, hexes_in_range, neighbors
from tla.ship import Ship, ShipKind
from tla.tile import PLAYER_A, PLAYER_B, Tile, TerrainType


def _sea_board(radius: int = 8) -> Board:
    board = Board(width=radius * 2 + 1, height=radius * 2 + 1)
    for coord in hexes_in_range(AxialCoord(0, 0), radius):
        board.tiles[coord] = Tile(coord=coord, terrain=TerrainType.SEA)
    return board


def _ship(coord: AxialCoord, kind: ShipKind, owner, ship_id: int, hp: int | None = None) -> Ship:
    stats = Config().ship_stats.stats[kind]
    return Ship(
        id=ship_id,
        kind=kind,
        owner=owner,
        position=coord,
        current_hp=hp if hp is not None else stats.hp,
        movement_remaining=stats.movement,
    )


def _game_state(board: Board, ships: list[Ship], config: Config | None = None) -> GameState:
    return GameState(config=config or Config(), board=board, ships={s.id: s for s in ships})


def _alloc_id():
    counter = iter(range(1, 1000))
    return lambda: next(counter)


def test_repair_task_forces_drops_sunk_members():
    board = _sea_board()
    survivor = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    gs = _game_state(board, [survivor])  # ship 2 (formerly a member) is gone -- sunk
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})

    repair_task_forces(gs, PLAYER_A, [force])

    assert force.member_ids == {1}


def test_repair_task_forces_removes_a_force_left_with_no_members():
    board = _sea_board()
    gs = _game_state(board, [])  # both former members sunk
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})
    forces = [force]

    repair_task_forces(gs, PLAYER_A, forces)

    assert forces == []


def test_repair_task_forces_ignores_ships_belonging_to_the_other_player():
    board = _sea_board()
    own = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    enemy = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_B, 2)
    gs = _game_state(board, [own, enemy])
    # A force can't legitimately contain an enemy ship id, but repair should
    # still only ever validate membership against the owner's own fleet.
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})

    repair_task_forces(gs, PLAYER_A, [force])

    assert force.member_ids == {1}


def test_repair_task_forces_leaves_a_fully_intact_force_untouched():
    board = _sea_board()
    a = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    b = _ship(AxialCoord(1, 0), ShipKind.CARRIER, PLAYER_A, 2)
    gs = _game_state(board, [a, b])
    goal = TaskForceGoal(kind=GoalKind.CAPTURE_PORT, target=AxialCoord(5, 5))
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2}, goal=goal, turns_since_progress=3)
    forces = [force]

    repair_task_forces(gs, PLAYER_A, forces)

    assert forces == [force]
    assert force.goal is goal
    assert force.turns_since_progress == 3


def test_repair_task_forces_increments_consecutive_loss_turns_when_a_member_is_gone():
    board = _sea_board()
    survivor = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    gs = _game_state(board, [survivor])  # ship 2 is gone -- sunk
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2}, consecutive_loss_turns=1)

    repair_task_forces(gs, PLAYER_A, [force])

    assert force.consecutive_loss_turns == 2


def test_repair_task_forces_resets_consecutive_loss_turns_when_nothing_was_lost():
    board = _sea_board()
    a = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    b = _ship(AxialCoord(1, 0), ShipKind.CARRIER, PLAYER_A, 2)
    gs = _game_state(board, [a, b])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2}, consecutive_loss_turns=2)

    repair_task_forces(gs, PLAYER_A, [force])

    assert force.consecutive_loss_turns == 0


def test_form_task_forces_anchors_on_a_carrier_and_gathers_nearby_ships():
    board = _sea_board()
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    nearby_destroyer = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_A, 2)
    nearby_sub = _ship(AxialCoord(0, 1), ShipKind.SUBMARINE, PLAYER_A, 3)
    far_patrol_boat = _ship(AxialCoord(20, 20), ShipKind.PATROL_BOAT, PLAYER_A, 4)
    gs = _game_state(board, [carrier, nearby_destroyer, nearby_sub, far_patrol_boat])
    forces: list = []

    form_task_forces(gs, PLAYER_A, forces, gs.config.ai, _alloc_id())

    # The far patrol boat has no nearby partner, so it stays unassigned
    # (below task_force_min_size -- see the isolated-leftover test) rather
    # than forming a solo "force".
    assert len(forces) == 1
    carrier_force = forces[0]
    assert carrier_force.member_ids == {1, 2, 3}
    assert carrier_force.goal is None  # formation alone doesn't assign goals


def test_form_task_forces_respects_max_size():
    board = _sea_board()
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    escorts = [_ship(AxialCoord(0, i + 1), ShipKind.DESTROYER, PLAYER_A, i + 2) for i in range(5)]
    gs = _game_state(board, [carrier, *escorts], config=Config(ai=AiConfig(task_force_max_size=3)))
    forces: list = []

    form_task_forces(gs, PLAYER_A, forces, gs.config.ai, _alloc_id())

    carrier_force = next(f for f in forces if 1 in f.member_ids)
    assert len(carrier_force.member_ids) == 3  # carrier + 2 escorts, not all 5


def test_form_task_forces_excludes_a_second_carrier_from_being_gathered():
    board = _sea_board()
    carrier1 = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    carrier2 = _ship(AxialCoord(1, 0), ShipKind.CARRIER, PLAYER_A, 2)
    gs = _game_state(board, [carrier1, carrier2])
    forces: list = []

    form_task_forces(gs, PLAYER_A, forces, gs.config.ai, _alloc_id())

    assert len(forces) == 2  # each carrier anchors its own force
    for force in forces:
        assert len(force.member_ids) == 1  # no escorts available, and never each other


def test_form_task_forces_leaves_an_isolated_leftover_ship_unassigned():
    board = _sea_board()
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    escort = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_A, 2)
    lone_ship = _ship(AxialCoord(20, 20), ShipKind.PATROL_BOAT, PLAYER_A, 3)  # alone, no fallback partner
    gs = _game_state(board, [carrier, escort, lone_ship])
    forces: list = []

    form_task_forces(gs, PLAYER_A, forces, gs.config.ai, _alloc_id())

    assigned = {i for f in forces for i in f.member_ids}
    assert 3 not in assigned


def test_form_task_forces_does_not_reassign_ships_already_in_a_force():
    board = _sea_board()
    a = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    b = _ship(AxialCoord(1, 0), ShipKind.CARRIER, PLAYER_A, 2)
    gs = _game_state(board, [a, b])
    existing = TaskForce(id=99, owner=PLAYER_A, member_ids={1})
    forces = [existing]

    form_task_forces(gs, PLAYER_A, forces, gs.config.ai, _alloc_id())

    assert existing.member_ids == {1}  # untouched
    # ship 1 must not also show up in a second, newly-formed force
    other_forces = [f for f in forces if f is not existing]
    assert all(1 not in f.member_ids for f in other_forces)


def _strait_board() -> Board:
    """Two open sea basins (q in [-10,-1] and q in [1,10]) separated by a
    land wall at q=0, with a single-hex gap at r=0 -- the only connection
    between them, and the obvious "chokepoint" a blockade should target."""
    board = Board(width=41, height=21)
    for q in list(range(-10, 0)) + list(range(1, 11)):
        for r in range(-5, 6):
            board.tiles[AxialCoord(q, r)] = Tile(coord=AxialCoord(q, r), terrain=TerrainType.SEA)
    for r in range(-5, 6):
        terrain = TerrainType.SEA if r == 0 else TerrainType.LAND
        board.tiles[AxialCoord(0, r)] = Tile(coord=AxialCoord(0, r), terrain=terrain)
    return board


def _add_port(board: Board, coord: AxialCoord, owner) -> None:
    board.tiles[coord] = Tile(coord=coord, terrain=TerrainType.LAND, is_port=True, port_owner=owner)


def test_find_chokepoint_locates_the_narrow_strait():
    board = _strait_board()
    _add_port(board, AxialCoord(-8, 0), PLAYER_A)
    _add_port(board, AxialCoord(8, 0), PLAYER_B)
    gs = _game_state(board, [])

    chokepoint = find_chokepoint(gs)

    # Not necessarily exactly the gap hex (0, 0) itself -- hex-grid
    # geometry means a hex directly beside a 1-wide gap can be tied with
    # it for narrowness (two of *its* six neighbors can also land on the
    # wall), and the tie-break (closest to the found path's midpoint) can
    # legitimately favor either. What matters is it's tight against the
    # strait, not out in the open ocean on either side.
    assert distance(chokepoint, AxialCoord(0, 0)) <= 2
    wall_neighbors = sum(1 for n in neighbors(chokepoint) if n not in {c for c, t in board.tiles.items() if t.terrain == TerrainType.SEA})
    assert wall_neighbors >= 2


def test_find_chokepoint_returns_none_when_the_two_territories_are_disconnected():
    board = _strait_board()
    board.tiles[AxialCoord(0, 0)] = Tile(coord=AxialCoord(0, 0), terrain=TerrainType.LAND)  # seal the gap
    _add_port(board, AxialCoord(-8, 0), PLAYER_A)
    _add_port(board, AxialCoord(8, 0), PLAYER_B)
    gs = _game_state(board, [])

    assert find_chokepoint(gs) is None


def test_find_chokepoint_does_not_crash_on_a_fully_open_board():
    board = _sea_board(radius=10)
    _add_port(board, AxialCoord(-8, 0), PLAYER_A)
    _add_port(board, AxialCoord(8, 0), PLAYER_B)
    gs = _game_state(board, [])

    chokepoint = find_chokepoint(gs)

    assert chokepoint is not None
    assert chokepoint in gs.board.tiles


def test_find_chokepoint_returns_none_without_ports_for_both_sides():
    board = _sea_board(radius=5)
    gs = _game_state(board, [])  # no ports at all

    assert find_chokepoint(gs) is None


def test_assign_goal_targets_the_nearest_uncontrolled_port():
    board = _sea_board()
    near_port, far_port = AxialCoord(1, 0), AxialCoord(6, 0)
    board.tiles[near_port] = Tile(coord=near_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    board.tiles[far_port] = Tile(coord=far_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    gs = _game_state(board, [ship])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})

    assign_goal(force, gs, [force])

    assert force.goal == TaskForceGoal(kind=GoalKind.CAPTURE_PORT, target=near_port)
    assert force.turns_since_progress == 0


def test_assign_goal_avoids_a_port_already_claimed_by_another_force():
    board = _sea_board()
    near_port, far_port = AxialCoord(1, 0), AxialCoord(6, 0)
    board.tiles[near_port] = Tile(coord=near_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    board.tiles[far_port] = Tile(coord=far_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    other_ship = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 2)
    gs = _game_state(board, [ship, other_ship])
    claimant = TaskForce(id=1, owner=PLAYER_A, member_ids={2}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, near_port))
    force = TaskForce(id=2, owner=PLAYER_A, member_ids={1})

    assign_goal(force, gs, [claimant, force])

    assert force.goal == TaskForceGoal(kind=GoalKind.CAPTURE_PORT, target=far_port)


def test_update_task_force_goals_completes_and_reassigns_the_same_turn():
    board = _sea_board()
    captured_port = AxialCoord(1, 0)
    other_port = AxialCoord(6, 0)
    board.tiles[captured_port] = Tile(
        coord=captured_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B, port_controller=PLAYER_A
    )
    board.tiles[other_port] = Tile(coord=other_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    # min_size=1 -- a single-member non-carrier force is otherwise
    # dissolved as an under-strength "casualty" (see the dedicated
    # dissolution test), which isn't what this test is about.
    gs = _game_state(board, [ship], config=Config(ai=AiConfig(task_force_min_size=1)))
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, captured_port))

    update_task_force_goals(gs, PLAYER_A, [force])

    assert force.goal == TaskForceGoal(kind=GoalKind.CAPTURE_PORT, target=other_port)


def test_update_task_force_goals_dissolves_a_force_below_min_size():
    board = _sea_board()
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    gs = _game_state(board, [ship], config=Config(ai=AiConfig(task_force_min_size=2)))
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})  # only 1 member -- casualties took the rest

    update_task_force_goals(gs, PLAYER_A, [force])

    assert force.member_ids == set()


def test_stall_detection_reassigns_a_force_camping_next_to_a_fight_it_cannot_win():
    # Reproduces the reported bug's exact shape: a force's ship parked one
    # hex from an enemy-held port, beside a defender it correctly declines
    # to fight, with nothing else nearby to distract it. Without stall
    # detection this repeats forever; with it, the goal is abandoned after
    # AiConfig.task_force_stall_turns turns of zero progress.
    board = _sea_board(radius=15)
    target_port = AxialCoord(5, 0)
    board.tiles[target_port] = Tile(coord=target_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    other_port = AxialCoord(-5, 0)
    board.tiles[other_port] = Tile(coord=other_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)

    stall_turns = 3
    config = Config(
        fow=FowConfig(enabled=False),
        ai=AiConfig(
            task_force_stall_turns=stall_turns,
            # min_size=1 -- a single-member non-carrier force is otherwise
            # dissolved as an under-strength "casualty" before stall
            # detection even gets a chance to run.
            task_force_min_size=1,
            # threat_radius=0 -- deliberately disables the separate
            # outnumbered/retreat mechanism for this test (a lone ship
            # declining a fight via matchup_score is, by the same race
            # logic, almost always also "outnumbered" by is_outnumbered's
            # cruder group-power check, which would otherwise retreat
            # long before stall detection ever gets a chance to run). See
            # test_stall_detection_and_retreat_dont_interfere for the
            # combined-mechanism case, and test_update_task_force_stance_*
            # for retreat in isolation.
            task_force_threat_radius=0,
        ),
    )
    attacker = _ship(AxialCoord(4, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)  # weak, adjacent to the port
    attacker.movement_remaining = 0  # already at its closest safe approach hex
    defender = _ship(target_port, ShipKind.BATTLESHIP, PLAYER_B, 2, hp=100)  # far too strong to attack
    gs = _game_state(board, [attacker, defender], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, target_port))

    policy = NaivePolicy()
    policy._task_forces[PLAYER_A] = [force]
    policy._next_force_id = 2

    original_goal = TaskForceGoal(kind=GoalKind.CAPTURE_PORT, target=target_port)
    stalled_after = None
    for i in range(1, stall_turns + 5):  # generous upper bound past stall_turns
        list(policy.plan_movement(gs, PLAYER_A))
        attacker.movement_remaining = 0  # stays put -- no real progress possible against this defender
        if force.goal != original_goal:
            stalled_after = i
            break

    assert stalled_after is not None, "force never gave up on an unwinnable, unmoving goal"
    assert stalled_after > stall_turns, "gave up before actually stalling out (turns_since_progress not honored)"


def test_carrier_advances_toward_its_forces_goal_when_unthreatened():
    board = _sea_board(radius=11)
    target_port = AxialCoord(10, 0)
    board.tiles[target_port] = Tile(coord=target_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    escort = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_A, 2)
    gs = _game_state(board, [carrier, escort], config=Config(fow=FowConfig(enabled=False)))
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, target_port))

    policy = NaivePolicy()
    policy._task_forces[PLAYER_A] = [force]
    list(policy.plan_movement(gs, PLAYER_A))

    assert distance(gs.ships[1].position, target_port) < distance(AxialCoord(0, 0), target_port)


def test_is_outnumbered_true_when_force_is_clearly_weaker():
    board = _sea_board()
    weak = _ship(AxialCoord(0, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)  # hp 2, dmg 1
    strong = _ship(AxialCoord(1, 0), ShipKind.BATTLESHIP, PLAYER_B, 2, hp=100)  # dmg 4
    gs = _game_state(board, [weak, strong])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})
    ai_config = AiConfig(task_force_threat_radius=4, task_force_outnumbered_margin=0)

    assert is_outnumbered(force, gs, {2: strong}, ai_config) is True


def test_is_outnumbered_false_when_force_is_clearly_stronger():
    board = _sea_board()
    strong = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1, hp=100)
    weak = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    gs = _game_state(board, [strong, weak])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})
    ai_config = AiConfig(task_force_threat_radius=4, task_force_outnumbered_margin=0)

    assert is_outnumbered(force, gs, {2: weak}, ai_config) is False


def test_is_outnumbered_false_when_no_enemy_within_threat_radius():
    board = _sea_board()
    weak = _ship(AxialCoord(0, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)
    strong = _ship(AxialCoord(10, 0), ShipKind.BATTLESHIP, PLAYER_B, 2, hp=100)
    gs = _game_state(board, [weak, strong])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})
    ai_config = AiConfig(task_force_threat_radius=2, task_force_outnumbered_margin=0)

    assert is_outnumbered(force, gs, {2: strong}, ai_config) is False


def test_is_outnumbered_respects_the_margin():
    # A marginal disadvantage (loses the race by exactly 1 round) should
    # only trip with a small enough margin -- confirms the config knob
    # actually gates sensitivity rather than is_outnumbered ignoring it.
    board = _sea_board()
    destroyer = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)  # hp 6, dmg 2
    cruiser = _ship(AxialCoord(1, 0), ShipKind.CRUISER, PLAYER_B, 2)  # hp 8, dmg 3
    gs = _game_state(board, [destroyer, cruiser])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})

    strict = AiConfig(task_force_threat_radius=4, task_force_outnumbered_margin=0)
    lenient = AiConfig(task_force_threat_radius=4, task_force_outnumbered_margin=4)
    assert is_outnumbered(force, gs, {2: cruiser}, strict) is True
    assert is_outnumbered(force, gs, {2: cruiser}, lenient) is False


def test_is_attritting_true_when_losing_streak_meets_threshold_and_stalled():
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1}, consecutive_loss_turns=3, turns_since_progress=1)
    ai_config = AiConfig(task_force_attrition_turns=3)

    assert is_attritting(force, ai_config) is True


def test_is_attritting_false_below_the_loss_streak_threshold():
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1}, consecutive_loss_turns=2, turns_since_progress=1)
    ai_config = AiConfig(task_force_attrition_turns=3)

    assert is_attritting(force, ai_config) is False


def test_is_attritting_false_when_still_making_progress():
    # Taking losses while still closing in on the goal is a costly advance,
    # not a losing war of attrition -- don't retreat from that alone.
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1}, consecutive_loss_turns=5, turns_since_progress=0)
    ai_config = AiConfig(task_force_attrition_turns=3)

    assert is_attritting(force, ai_config) is False


def test_is_outmatched_by_target_support_true_when_target_has_backup():
    board = _sea_board()
    attacker = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)  # an easy-looking 1v1...
    target = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    backup = _ship(AxialCoord(2, 0), ShipKind.BATTLESHIP, PLAYER_B, 3, hp=100)  # ...but not alone
    gs = _game_state(board, [attacker, target, backup])
    ai_config = AiConfig(task_force_threat_radius=4, task_force_outnumbered_margin=0)

    assert is_outmatched_by_target_support(
        [attacker], target, gs, {2: target, 3: backup}, ai_config
    ) is True


def test_is_outmatched_by_target_support_false_when_target_is_alone():
    board = _sea_board()
    attacker = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    target = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    gs = _game_state(board, [attacker, target])
    ai_config = AiConfig(task_force_threat_radius=4, task_force_outnumbered_margin=0)

    assert is_outmatched_by_target_support([attacker], target, gs, {2: target}, ai_config) is False


def test_is_outmatched_by_target_support_false_when_backup_is_out_of_range():
    board = _sea_board()
    attacker = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    target = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    backup = _ship(AxialCoord(10, 0), ShipKind.BATTLESHIP, PLAYER_B, 3, hp=100)
    gs = _game_state(board, [attacker, target, backup])
    ai_config = AiConfig(task_force_threat_radius=2, task_force_outnumbered_margin=0)

    assert is_outmatched_by_target_support(
        [attacker], target, gs, {2: target, 3: backup}, ai_config
    ) is False


def test_is_outmatched_by_target_support_false_for_empty_attacker_group():
    board = _sea_board()
    target = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    gs = _game_state(board, [target])

    assert is_outmatched_by_target_support([], target, gs, {2: target}, AiConfig()) is False


def test_gather_prefers_kind_diversity_over_pure_distance():
    # No existing_kinds seed -- diversity still emerges organically: the
    # first pick is whichever candidate is nearest (all kinds tied at
    # zero so far), but the second pick prefers the still-unrepresented
    # cruiser over a second, closer destroyer.
    center = AxialCoord(0, 0)
    destroyer_close = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    destroyer_far = _ship(AxialCoord(2, 0), ShipKind.DESTROYER, PLAYER_A, 2)
    cruiser_farthest = _ship(AxialCoord(3, 0), ShipKind.CRUISER, PLAYER_A, 3)
    pool = [destroyer_close, destroyer_far, cruiser_farthest]

    picked = _gather(center, frozenset(), pool, gather_radius=5, capacity=2, exclude_carriers=False)

    assert picked == {1, 3}


def test_gather_seeded_existing_kinds_biases_the_very_first_pick():
    # With the force already all-destroyer (seeded via existing_kinds),
    # even the *first* pick should prefer the cruiser over a closer
    # destroyer.
    center = AxialCoord(0, 0)
    destroyer_closest = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    cruiser_farther = _ship(AxialCoord(2, 0), ShipKind.CRUISER, PLAYER_A, 2)
    pool = [destroyer_closest, cruiser_farther]

    picked = _gather(
        center, frozenset(), pool, gather_radius=5, capacity=1, exclude_carriers=False,
        existing_kinds=Counter({ShipKind.DESTROYER: 3}),
    )

    assert picked == {2}


def test_gather_still_respects_radius_and_capacity_with_diversity_enabled():
    center = AxialCoord(0, 0)
    in_range = _ship(AxialCoord(1, 0), ShipKind.CRUISER, PLAYER_A, 1)
    out_of_range = _ship(AxialCoord(10, 0), ShipKind.SUBMARINE, PLAYER_A, 2)
    pool = [in_range, out_of_range]

    picked = _gather(center, frozenset(), pool, gather_radius=3, capacity=5, exclude_carriers=False)

    assert picked == {1}


def test_rally_point_returns_the_nearest_controlled_port():
    board = _sea_board()
    near_port, far_port = AxialCoord(2, 0), AxialCoord(-6, 0)
    board.tiles[near_port] = Tile(coord=near_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    board.tiles[far_port] = Tile(coord=far_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    gs = _game_state(board, [ship])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})

    assert rally_point(force, gs) == near_port


def test_rally_point_returns_none_without_any_controlled_ports():
    board = _sea_board()
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    gs = _game_state(board, [ship])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})

    assert rally_point(force, gs) is None


def test_update_task_force_stance_triggers_retreat_when_newly_outnumbered():
    board = _sea_board()
    weak = _ship(AxialCoord(0, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)
    strong = _ship(AxialCoord(1, 0), ShipKind.BATTLESHIP, PLAYER_B, 2, hp=100)
    config = Config(ai=AiConfig(task_force_threat_radius=4, task_force_outnumbered_margin=0))
    gs = _game_state(board, [weak, strong], config=config)
    goal = TaskForceGoal(GoalKind.CAPTURE_PORT, AxialCoord(5, 0))
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1}, goal=goal)

    update_task_force_stance(gs, PLAYER_A, [force], {2: strong})

    assert force.retreating is True
    assert force.retreat_turns == 0
    assert force.retreat_threat_power == (100, 4)  # strong's own (hp, damage), snapshotted
    assert force.goal == goal  # untouched -- only the stance changed


def test_update_task_force_stance_keeps_retreating_while_still_weaker_than_the_snapshot():
    board = _sea_board()
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)  # (hp=6, damage=2)
    config = Config(ai=AiConfig(task_force_outnumbered_margin=0))
    gs = _game_state(board, [ship], config=config)
    goal = TaskForceGoal(GoalKind.CAPTURE_PORT, AxialCoord(5, 0))
    force = TaskForce(
        id=1,
        owner=PLAYER_A,
        member_ids={1},
        goal=goal,
        retreating=True,
        retreat_turns=0,
        retreat_threat_power=(1000, 1000),  # far stronger than this lone destroyer will ever be alone
    )

    update_task_force_stance(gs, PLAYER_A, [force], {})
    assert force.retreating is True
    assert force.retreat_turns == 1  # counts up, not a countdown
    assert force.goal == goal

    update_task_force_stance(gs, PLAYER_A, [force], {})
    assert force.retreating is True
    assert force.retreat_turns == 2


def test_update_task_force_stance_resumes_once_stronger_than_the_snapshot():
    board = _sea_board()
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)  # (hp=6, damage=2)
    gs = _game_state(board, [ship])  # default margin=4
    goal = TaskForceGoal(GoalKind.CAPTURE_PORT, AxialCoord(5, 0))
    force = TaskForce(
        id=1,
        owner=PLAYER_A,
        member_ids={1},
        goal=goal,
        retreating=True,
        retreat_turns=3,
        retreat_threat_power=(1, 1),  # trivially weak -- easily beaten by the lone destroyer now
        turns_since_progress=5,
        best_progress_distance=3,
    )

    update_task_force_stance(gs, PLAYER_A, [force], {})

    assert force.retreating is False
    assert force.retreat_threat_power is None
    assert force.retreat_turns == 0
    assert force.turns_since_progress == 0
    assert force.best_progress_distance is None
    assert force.goal == goal  # resumes the SAME goal


def test_update_task_force_stance_gives_up_after_max_retreat_turns():
    # The safety valve: if the force never actually catches up (e.g. the
    # opponent is reinforcing just as fast), it must eventually give up on
    # the goal rather than retreat forever.
    board = _sea_board()
    original_target = AxialCoord(5, 0)
    other_port = AxialCoord(6, 0)
    board.tiles[original_target] = Tile(
        coord=original_target, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B
    )
    board.tiles[other_port] = Tile(coord=other_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    config = Config(ai=AiConfig(task_force_max_retreat_turns=2))
    gs = _game_state(board, [ship], config=config)
    force = TaskForce(
        id=1,
        owner=PLAYER_A,
        member_ids={1},
        goal=TaskForceGoal(GoalKind.CAPTURE_PORT, original_target),
        retreating=True,
        retreat_turns=1,
        retreat_threat_power=(1000, 1000),  # never actually beatable
    )

    update_task_force_stance(gs, PLAYER_A, [force], {})  # retreat_turns 1 -> 2, hits the cap

    assert force.retreating is False
    assert force.retreat_threat_power is None
    assert force.retreat_turns == 0
    assert force.goal is not None
    assert force.goal.target == other_port  # original_target excluded


def test_update_task_force_stance_triggers_retreat_from_attrition_alone():
    # No enemy anywhere nearby -- is_outnumbered would never trip -- but a
    # long losing streak with no progress is itself a retreat trigger (see
    # is_attritting): the enemy that ground this force down piecemeal
    # doesn't have to be visible/nearby *right now* for the damage to be
    # real. With nothing visible to snapshot, the threat is (0, 0) -- see
    # _threat_snapshot -- which the force will trivially outmatch as soon
    # as it has any damage output at all.
    board = _sea_board()
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    config = Config(ai=AiConfig(task_force_attrition_turns=3))
    gs = _game_state(board, [ship], config=config)
    goal = TaskForceGoal(GoalKind.CAPTURE_PORT, AxialCoord(5, 0))
    force = TaskForce(
        id=1, owner=PLAYER_A, member_ids={1}, goal=goal, consecutive_loss_turns=3, turns_since_progress=2
    )

    update_task_force_stance(gs, PLAYER_A, [force], {})

    assert force.retreating is True
    assert force.retreat_turns == 0
    assert force.retreat_threat_power == (0, 0)
    assert force.goal == goal  # untouched -- only the stance changed


def test_retreat_preserves_goal_across_the_pullback_and_resumes_once_reinforced():
    # Integration test via NaivePolicy.plan_movement: a force pursuing
    # CAPTURE_PORT runs into an overwhelming defender, retreats instead of
    # stalling out (run well past task_force_stall_turns while the threat
    # persists -- the goal must never change), then resumes toward the
    # SAME original port once new production actually makes it strong
    # enough to beat what it retreated from -- not merely because the old
    # threat happens to no longer be visible (that's a deliberate change
    # from the old fixed-cooldown design: home isn't safety, growing past
    # the threat is).
    board = _sea_board(radius=20)
    target_port = AxialCoord(10, 0)
    board.tiles[target_port] = Tile(coord=target_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    home_port = AxialCoord(-10, 0)
    board.tiles[home_port] = Tile(coord=home_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)

    config = Config(
        fow=FowConfig(enabled=False),
        ai=AiConfig(
            task_force_threat_radius=6,
            task_force_outnumbered_margin=0,
            task_force_stall_turns=3,
            task_force_min_size=1,
            task_force_max_retreat_turns=100,  # isolate from the dedicated safety-valve test above
            retreat_cancel_port_distance=None,  # isolate from the dedicated distance-cancel tests
        ),
    )
    attacker = _ship(AxialCoord(9, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)
    defender = _ship(target_port, ShipKind.BATTLESHIP, PLAYER_B, 2, hp=100)
    gs = _game_state(board, [attacker, defender], config=config)
    original_goal = TaskForceGoal(GoalKind.CAPTURE_PORT, target_port)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1}, goal=original_goal)

    policy = NaivePolicy()
    policy._task_forces[PLAYER_A] = [force]
    policy._next_force_id = 2
    patrol_boat_movement = Config().ship_stats.stats[ShipKind.PATROL_BOAT].movement

    for _ in range(6):  # well past task_force_stall_turns
        attacker.movement_remaining = patrol_boat_movement  # simulate a fresh turn's movement budget
        list(policy.plan_movement(gs, PLAYER_A))
        assert force.goal == original_goal, "retreat must suppress stall/completion, not just delay it"
        assert force.retreating is True

    # A strong new production ship shows up near the retreating force and
    # gets absorbed via ordinary recruit_into_open_forces (untouched by
    # retreat state) -- this, not the enemy disappearing, is what's meant
    # to end the retreat.
    reinforcement = _ship(attacker.position, ShipKind.BATTLESHIP, PLAYER_A, 3, hp=100)
    gs.ships[3] = reinforcement

    attacker.movement_remaining = patrol_boat_movement
    list(policy.plan_movement(gs, PLAYER_A))

    assert 3 in force.member_ids, "reinforcement must actually have been recruited for this to prove anything"
    assert force.retreating is False
    assert force.goal == original_goal  # resumes the SAME goal, not a new one


def test_carrier_retreats_with_its_force_when_outnumbered():
    board = _sea_board()
    target_port = AxialCoord(10, 0)
    home_port = AxialCoord(-5, 0)
    board.tiles[target_port] = Tile(coord=target_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    board.tiles[home_port] = Tile(coord=home_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    escort = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_A, 2)
    # Distance 5 from the carrier -- outside the default carrier_threat_radius
    # (3, its own *individual* threat check), but within task_force_threat_
    # radius (6) used for the *force*-level outnumbered check, so a
    # confirmed rally-point move here can only be explained by the force
    # mechanism, not the carrier's pre-existing personal one.
    strong_enemy = _ship(AxialCoord(5, 0), ShipKind.BATTLESHIP, PLAYER_B, 3, hp=100)
    config = Config(
        fow=FowConfig(enabled=False), ai=AiConfig(task_force_threat_radius=6, task_force_outnumbered_margin=0)
    )
    gs = _game_state(board, [carrier, escort, strong_enemy], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, target_port))

    policy = NaivePolicy()
    policy._task_forces[PLAYER_A] = [force]
    list(policy.plan_movement(gs, PLAYER_A))

    assert force.retreating is True  # confirms outnumbered triggered
    assert distance(gs.ships[1].position, home_port) < distance(AxialCoord(0, 0), home_port)


def test_is_open_below_target_min_size():
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})
    assert is_open(force, AiConfig(task_force_target_min_size=3)) is True


def test_is_open_at_or_above_target_min_size():
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2, 3})
    assert is_open(force, AiConfig(task_force_target_min_size=3)) is False


def test_recruit_into_open_forces_absorbs_a_nearby_unassigned_ship():
    board = _sea_board()
    member = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    stray = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_A, 2)
    config = Config(ai=AiConfig(task_force_target_min_size=3, task_force_target_max_size=6, task_force_recruit_radius=5))
    gs = _game_state(board, [member, stray], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})

    recruit_into_open_forces(gs, PLAYER_A, [force], gs.config.ai)

    assert force.member_ids == {1, 2}


def test_recruit_into_open_forces_does_nothing_when_already_at_target_min():
    board = _sea_board()
    member1 = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    member2 = _ship(AxialCoord(0, 1), ShipKind.DESTROYER, PLAYER_A, 2)
    stray = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_A, 3)
    config = Config(ai=AiConfig(task_force_target_min_size=2, task_force_recruit_radius=5))
    gs = _game_state(board, [member1, member2, stray], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})  # already at target_min_size

    recruit_into_open_forces(gs, PLAYER_A, [force], gs.config.ai)

    assert force.member_ids == {1, 2}  # stray left alone


def test_recruit_into_open_forces_respects_target_max_capacity():
    board = _sea_board()
    member = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    strays = [_ship(AxialCoord(0, i + 1), ShipKind.DESTROYER, PLAYER_A, i + 2) for i in range(5)]
    config = Config(
        ai=AiConfig(task_force_target_min_size=10, task_force_target_max_size=3, task_force_recruit_radius=10)
    )
    gs = _game_state(board, [member, *strays], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})

    recruit_into_open_forces(gs, PLAYER_A, [force], gs.config.ai)

    assert len(force.member_ids) == 3  # capped at target_max_size, not all 5 strays


def test_recruit_into_open_forces_ignores_strays_outside_recruit_radius():
    board = _sea_board(radius=15)
    member = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    far_stray = _ship(AxialCoord(10, 0), ShipKind.DESTROYER, PLAYER_A, 2)
    config = Config(ai=AiConfig(task_force_target_min_size=5, task_force_recruit_radius=3))
    gs = _game_state(board, [member, far_stray], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})

    recruit_into_open_forces(gs, PLAYER_A, [force], gs.config.ai)

    assert force.member_ids == {1}


def test_recruit_into_open_forces_can_recruit_a_second_carrier():
    # Unlike initial formation (which keeps each carrier anchoring its own
    # force to avoid double-booking one mid-pass), ongoing recruitment has
    # no such restriction -- this is how multiple carriers actually end up
    # consolidated under one force over the course of a game.
    board = _sea_board()
    carrier_member = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    stray_carrier = _ship(AxialCoord(1, 0), ShipKind.CARRIER, PLAYER_A, 2)
    config = Config(ai=AiConfig(task_force_target_min_size=5, task_force_recruit_radius=5))
    gs = _game_state(board, [carrier_member, stray_carrier], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})

    recruit_into_open_forces(gs, PLAYER_A, [force], gs.config.ai)

    assert force.member_ids == {1, 2}  # the stray carrier is recruited


def test_recruit_into_open_forces_can_recruit_a_carrier_into_a_carrier_less_force():
    board = _sea_board()
    member = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    stray_carrier = _ship(AxialCoord(1, 0), ShipKind.CARRIER, PLAYER_A, 2)
    config = Config(ai=AiConfig(task_force_target_min_size=5, task_force_recruit_radius=5))
    gs = _game_state(board, [member, stray_carrier], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})

    recruit_into_open_forces(gs, PLAYER_A, [force], gs.config.ai)

    assert force.member_ids == {1, 2}


def test_recruit_into_open_forces_never_poaches_from_another_force():
    board = _sea_board()
    member = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    already_committed = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_A, 2)
    config = Config(ai=AiConfig(task_force_target_min_size=5, task_force_recruit_radius=5))
    gs = _game_state(board, [member, already_committed], config=config)
    open_force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})
    other_force = TaskForce(id=2, owner=PLAYER_A, member_ids={2})  # ship 2 already belongs here

    recruit_into_open_forces(gs, PLAYER_A, [open_force, other_force], gs.config.ai)

    assert open_force.member_ids == {1}
    assert other_force.member_ids == {2}


def test_a_newly_produced_ship_is_absorbed_by_an_existing_open_force():
    board = _sea_board()
    port = AxialCoord(0, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    member = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    config = Config(
        fow=FowConfig(enabled=False),
        ai=AiConfig(task_force_target_min_size=3, task_force_recruit_radius=5),
    )
    gs = _game_state(board, [member], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})

    policy = NaivePolicy()
    policy._task_forces[PLAYER_A] = [force]
    policy._next_force_id = 2

    # A new ship spawns at the port -- as production.run_production would --
    # not yet a member of anything.
    new_ship = _ship(port, ShipKind.DESTROYER, PLAYER_A, 2)
    gs.ships[2] = new_ship

    list(policy.plan_movement(gs, PLAYER_A))

    assert force.member_ids == {1, 2}
    assert len(policy._task_forces[PLAYER_A]) == 1  # no separate new force formed for it


def test_force_recapture_goal_overrides_an_active_goal():
    board = _sea_board()
    lost_port = AxialCoord(5, 0)
    board.tiles[lost_port] = Tile(coord=lost_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    gs = _game_state(board, [ship])
    original_goal = TaskForceGoal(GoalKind.CAPTURE_PORT, AxialCoord(-5, 0))
    force = TaskForce(
        id=1, owner=PLAYER_A, member_ids={1}, goal=original_goal, turns_since_progress=6, best_progress_distance=3
    )

    force_recapture_goal(force, gs, {lost_port})

    assert force.goal == TaskForceGoal(GoalKind.CAPTURE_PORT, lost_port)
    assert force.turns_since_progress == 0
    assert force.best_progress_distance is None


def test_force_recapture_goal_targets_the_nearest_of_multiple_lost_ports():
    board = _sea_board()
    near_port = AxialCoord(2, 0)
    far_port = AxialCoord(-8, 0)
    for p in (near_port, far_port):
        board.tiles[p] = Tile(coord=p, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    gs = _game_state(board, [ship])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})

    force_recapture_goal(force, gs, {near_port, far_port})

    assert force.goal.target == near_port


def test_force_recapture_goal_does_not_touch_retreat_state():
    board = _sea_board()
    lost_port = AxialCoord(5, 0)
    board.tiles[lost_port] = Tile(coord=lost_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    gs = _game_state(board, [ship])
    force = TaskForce(
        id=1,
        owner=PLAYER_A,
        member_ids={1},
        goal=TaskForceGoal(GoalKind.CAPTURE_PORT, AxialCoord(-5, 0)),
        retreating=True,
        retreat_turns=2,
        retreat_threat_power=(20, 8),
    )

    force_recapture_goal(force, gs, {lost_port})

    assert force.goal.target == lost_port  # the real goal changes...
    assert force.retreating is True  # ...but a fleeing force isn't forced to charge back in
    assert force.retreat_turns == 2
    assert force.retreat_threat_power == (20, 8)


def test_force_recapture_goal_is_a_noop_with_no_lost_ports_or_no_members():
    board = _sea_board()
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    gs = _game_state(board, [ship])
    goal = TaskForceGoal(GoalKind.CAPTURE_PORT, AxialCoord(5, 0))
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1}, goal=goal)

    force_recapture_goal(force, gs, set())
    assert force.goal == goal

    empty_force = TaskForce(id=2, owner=PLAYER_A, member_ids=set(), goal=goal)
    force_recapture_goal(empty_force, gs, {AxialCoord(5, 0)})
    assert empty_force.goal == goal


def test_update_task_force_stance_resumes_when_close_enough_to_a_port_even_if_still_weaker():
    board = _sea_board(radius=20)
    home_port = AxialCoord(-8, 0)
    board.tiles[home_port] = Tile(coord=home_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    ship = _ship(AxialCoord(-6, 0), ShipKind.DESTROYER, PLAYER_A, 1)  # 2 hexes from home_port
    config = Config(ai=AiConfig(retreat_cancel_port_distance=6, task_force_outnumbered_margin=4))
    gs = _game_state(board, [ship], config=config)
    goal = TaskForceGoal(GoalKind.CAPTURE_PORT, AxialCoord(5, 0))
    force = TaskForce(
        id=1,
        owner=PLAYER_A,
        member_ids={1},
        goal=goal,
        retreating=True,
        retreat_turns=1,
        retreat_threat_power=(1000, 1000),  # never beatable by strength alone
    )

    update_task_force_stance(gs, PLAYER_A, [force], {})

    assert force.retreating is False  # close enough to home to stand and fight anyway
    assert force.retreat_threat_power is None
    assert force.retreat_turns == 0
    assert force.goal == goal


def test_update_task_force_stance_keeps_retreating_when_too_far_from_a_port():
    board = _sea_board(radius=20)
    home_port = AxialCoord(-8, 0)
    board.tiles[home_port] = Tile(coord=home_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)  # 8 hexes from home_port
    config = Config(ai=AiConfig(retreat_cancel_port_distance=6, task_force_outnumbered_margin=4))
    gs = _game_state(board, [ship], config=config)
    force = TaskForce(
        id=1,
        owner=PLAYER_A,
        member_ids={1},
        goal=TaskForceGoal(GoalKind.CAPTURE_PORT, AxialCoord(5, 0)),
        retreating=True,
        retreat_turns=1,
        retreat_threat_power=(1000, 1000),
    )

    update_task_force_stance(gs, PLAYER_A, [force], {})

    assert force.retreating is True  # still 8 hexes out -- past the 6-hex cutoff
    assert force.retreat_turns == 2


def test_update_task_force_stance_distance_cancel_disabled_when_none():
    board = _sea_board(radius=20)
    home_port = AxialCoord(-8, 0)
    board.tiles[home_port] = Tile(coord=home_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    ship = _ship(AxialCoord(-7, 0), ShipKind.DESTROYER, PLAYER_A, 1)  # 1 hex from home_port
    config = Config(ai=AiConfig(retreat_cancel_port_distance=None))
    gs = _game_state(board, [ship], config=config)
    force = TaskForce(
        id=1,
        owner=PLAYER_A,
        member_ids={1},
        goal=TaskForceGoal(GoalKind.CAPTURE_PORT, AxialCoord(5, 0)),
        retreating=True,
        retreat_turns=1,
        retreat_threat_power=(1000, 1000),
    )

    update_task_force_stance(gs, PLAYER_A, [force], {})

    assert force.retreating is True  # distance cancel is off -- only strength can end this


def test_distance_resume_resets_progress_when_flag_enabled():
    board = _sea_board(radius=20)
    home_port = AxialCoord(-8, 0)
    board.tiles[home_port] = Tile(coord=home_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    ship = _ship(AxialCoord(-6, 0), ShipKind.DESTROYER, PLAYER_A, 1)  # 2 hexes from home_port
    config = Config(
        ai=AiConfig(
            retreat_cancel_port_distance=6,
            task_force_outnumbered_margin=4,
            reset_progress_on_distance_resume=True,
        )
    )
    gs = _game_state(board, [ship], config=config)
    force = TaskForce(
        id=1,
        owner=PLAYER_A,
        member_ids={1},
        goal=TaskForceGoal(GoalKind.CAPTURE_PORT, AxialCoord(5, 0)),
        retreating=True,
        retreat_turns=1,
        retreat_threat_power=(1000, 1000),  # never beatable by strength
        turns_since_progress=5,
        best_progress_distance=3,
    )

    update_task_force_stance(gs, PLAYER_A, [force], {})

    assert force.retreating is False  # resumed via distance, not strength
    assert force.turns_since_progress == 0
    assert force.best_progress_distance is None


def test_distance_resume_preserves_progress_when_flag_disabled():
    board = _sea_board(radius=20)
    home_port = AxialCoord(-8, 0)
    board.tiles[home_port] = Tile(coord=home_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    ship = _ship(AxialCoord(-6, 0), ShipKind.DESTROYER, PLAYER_A, 1)  # 2 hexes from home_port
    config = Config(
        ai=AiConfig(
            retreat_cancel_port_distance=6,
            task_force_outnumbered_margin=4,
            reset_progress_on_distance_resume=False,
        )
    )
    gs = _game_state(board, [ship], config=config)
    force = TaskForce(
        id=1,
        owner=PLAYER_A,
        member_ids={1},
        goal=TaskForceGoal(GoalKind.CAPTURE_PORT, AxialCoord(5, 0)),
        retreating=True,
        retreat_turns=1,
        retreat_threat_power=(1000, 1000),
        turns_since_progress=5,
        best_progress_distance=3,
    )

    update_task_force_stance(gs, PLAYER_A, [force], {})

    assert force.retreating is False  # still resumed via distance
    assert force.turns_since_progress == 5  # ...but progress tracking survives
    assert force.best_progress_distance == 3


def test_strength_resume_always_resets_progress_even_when_distance_flag_disabled():
    board = _sea_board(radius=20)
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)  # (hp=6, damage=2)
    config = Config(
        ai=AiConfig(
            retreat_cancel_port_distance=None,  # isolate from distance entirely
            reset_progress_on_distance_resume=False,
        )
    )
    gs = _game_state(board, [ship], config=config)
    force = TaskForce(
        id=1,
        owner=PLAYER_A,
        member_ids={1},
        goal=TaskForceGoal(GoalKind.CAPTURE_PORT, AxialCoord(5, 0)),
        retreating=True,
        retreat_turns=3,
        retreat_threat_power=(1, 1),  # trivially weak -- easily beaten by strength
        turns_since_progress=5,
        best_progress_distance=3,
    )

    update_task_force_stance(gs, PLAYER_A, [force], {})

    assert force.retreating is False  # resumed via strength
    assert force.turns_since_progress == 0  # strength-based resume always resets
    assert force.best_progress_distance is None


def test_pullback_waypoint_caps_short_of_the_port():
    board = _sea_board(radius=20)
    port = AxialCoord(-10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    gs = _game_state(board, [ship])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})

    waypoint = _pullback_waypoint(force, gs, 3)

    assert waypoint is not None
    assert waypoint != port
    assert distance(AxialCoord(0, 0), waypoint) == 3


def test_pullback_waypoint_falls_through_to_the_port_when_already_closer():
    board = _sea_board(radius=20)
    port = AxialCoord(-2, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    gs = _game_state(board, [ship])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})

    waypoint = _pullback_waypoint(force, gs, 10)  # cap far larger than the real distance

    assert waypoint == port


def test_update_task_force_stance_snapshots_and_clears_retreat_waypoint():
    board = _sea_board(radius=20)
    port = AxialCoord(-10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    weak = _ship(AxialCoord(0, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)
    strong = _ship(AxialCoord(1, 0), ShipKind.BATTLESHIP, PLAYER_B, 2, hp=100)
    config = Config(
        ai=AiConfig(
            task_force_threat_radius=4,
            task_force_outnumbered_margin=0,
            retreat_pullback_hexes=3,
            retreat_cancel_port_distance=None,
        )
    )
    gs = _game_state(board, [weak, strong], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, AxialCoord(5, 0)))

    update_task_force_stance(gs, PLAYER_A, [force], {2: strong})  # triggers retreat

    assert force.retreating is True
    assert force.retreat_waypoint is not None
    assert force.retreat_waypoint != port  # capped short, not the port itself

    # Now make it trivially strong enough to resume via strength.
    force.retreat_threat_power = (1, 1)
    update_task_force_stance(gs, PLAYER_A, [force], {})

    assert force.retreating is False
    assert force.retreat_waypoint is None  # cleared on resume


# -- port defense (see compute_port_defense_directives) --------------------


def _port_board(port: AxialCoord, radius: int = 10) -> Board:
    board = _sea_board(radius=radius)
    board.tiles[port] = Tile(
        coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A, port_controller=PLAYER_A
    )
    return board


def test_compute_port_defense_directives_counterattacks_when_not_outmatched():
    port = AxialCoord(0, 0)
    board = _port_board(port)
    defender = _ship(AxialCoord(2, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    threat = _ship(AxialCoord(4, 0), ShipKind.DESTROYER, PLAYER_B, 2)  # 4 hexes -- exactly the trigger radius
    gs = _game_state(board, [defender, threat])

    directives = compute_port_defense_directives(gs, PLAYER_A, {2: threat}, AiConfig())

    assert len(directives) == 1
    directive = directives[0]
    assert directive.port == port
    assert directive.threats == [threat]
    assert directive.counterattack == [defender]
    assert directive.block is None


def test_compute_port_defense_directives_blocks_when_outmatched():
    port = AxialCoord(0, 0)
    board = _port_board(port)
    weak = _ship(AxialCoord(2, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)
    sub = _ship(AxialCoord(2, 1), ShipKind.SUBMARINE, PLAYER_A, 2)
    strong_threat = _ship(AxialCoord(4, 0), ShipKind.BATTLESHIP, PLAYER_B, 3)
    gs = _game_state(board, [weak, sub, strong_threat])

    directives = compute_port_defense_directives(gs, PLAYER_A, {3: strong_threat}, AiConfig())

    assert len(directives) == 1
    directive = directives[0]
    assert directive.counterattack == []
    assert directive.block is sub  # submarine preferred as the delaying picket
    assert directive.block_hex is not None


def test_compute_port_defense_directives_ignores_a_threat_past_the_trigger_radius():
    port = AxialCoord(0, 0)
    board = _port_board(port)
    defender = _ship(AxialCoord(1, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    far_threat = _ship(AxialCoord(9, 0), ShipKind.DESTROYER, PLAYER_B, 2)  # past the default radius of 4

    directives = compute_port_defense_directives(
        _game_state(board, [defender, far_threat]), PLAYER_A, {2: far_threat}, AiConfig()
    )

    assert directives == []


def test_compute_port_defense_directives_ignores_a_responder_too_far_to_help():
    port = AxialCoord(0, 0)
    board = _port_board(port, radius=15)
    far_defender = _ship(AxialCoord(12, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)  # past the default radius of 8
    threat = _ship(AxialCoord(4, 0), ShipKind.DESTROYER, PLAYER_B, 2)

    directives = compute_port_defense_directives(
        _game_state(board, [far_defender, threat]), PLAYER_A, {2: threat}, AiConfig()
    )

    assert directives == []


def test_compute_port_defense_directives_does_not_double_claim_a_responder():
    port_a = AxialCoord(0, 0)
    port_b = AxialCoord(3, 0)
    board = _sea_board(radius=15)
    for port in (port_a, port_b):
        board.tiles[port] = Tile(
            coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A, port_controller=PLAYER_A
        )
    responder = _ship(AxialCoord(1, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)  # in range of both ports
    threat_a = _ship(AxialCoord(2, 0), ShipKind.DESTROYER, PLAYER_B, 2)
    threat_b = _ship(AxialCoord(4, 0), ShipKind.DESTROYER, PLAYER_B, 3)
    gs = _game_state(board, [responder, threat_a, threat_b])

    directives = compute_port_defense_directives(gs, PLAYER_A, {2: threat_a, 3: threat_b}, AiConfig())

    claimed_ids = [s.id for d in directives for s in (d.counterattack or ([d.block] if d.block else []))]
    assert claimed_ids.count(1) == 1


def test_choose_blocker_prefers_a_submarine_over_a_patrol_boat_over_anything_else():
    sub = _ship(AxialCoord(0, 0), ShipKind.SUBMARINE, PLAYER_A, 1)
    patrol = _ship(AxialCoord(0, 0), ShipKind.PATROL_BOAT, PLAYER_A, 2)
    battleship = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 3)

    assert _choose_blocker([patrol, battleship, sub]) is sub
    assert _choose_blocker([patrol, battleship]) is patrol
    assert _choose_blocker([battleship]) is battleship


def test_block_hex_toward_port_is_the_threats_own_next_step():
    port = AxialCoord(0, 0)
    board = _port_board(port)
    threat = _ship(AxialCoord(3, 0), ShipKind.DESTROYER, PLAYER_B, 1)
    gs = _game_state(board, [threat])
    field = sea_distance_field(gs, port)

    block_hex = _block_hex_toward_port([threat], port, field, gs)

    # A genuine next step on the threat's own shortest route in -- adjacent
    # to where it's actually standing. Not asserting it's strictly closer
    # to `port` by straight-line distance: the port itself is land, so the
    # path's real endpoint is whichever of its sea-adjacent neighbors
    # _nearest_in's tie-break picks (see find_chokepoint's own use of the
    # same helper), which isn't always the neighbor closest to the threat's
    # approach direction.
    assert block_hex is not None
    assert block_hex != threat.position
    assert distance(block_hex, threat.position) == 1
