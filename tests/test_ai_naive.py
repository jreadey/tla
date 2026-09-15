from unittest.mock import patch

from tla.ai.policy import (
    NaivePolicy,
    _choose_destination,
    _compute_force_pace,
    _favorable_attack,
    _scout_prepass,
    _step_toward,
    choose_task_force_destination,
)
from tla.ai.task_force import GoalKind, TaskForce, TaskForceGoal
from tla.board import Board
from tla.config import AiConfig, Config, FowConfig
from tla.game_state import GameState
from tla.hexgrid import AxialCoord, distance, hexes_in_range
from tla.ship import Ship, ShipKind
from tla.tile import PLAYER_A, PLAYER_B, Tile, TerrainType


def _sea_board(radius: int = 8) -> Board:
    board = Board(width=radius * 2 + 1, height=radius * 2 + 1)
    for coord in hexes_in_range(AxialCoord(0, 0), radius):
        board.tiles[coord] = Tile(coord=coord, terrain=TerrainType.SEA)
    return board


def _ship(
    coord: AxialCoord, kind: ShipKind, owner, ship_id: int, hp: int | None = None, surfaced: bool = True
) -> Ship:
    stats = Config().ship_stats.stats[kind]
    return Ship(
        id=ship_id,
        kind=kind,
        owner=owner,
        position=coord,
        current_hp=hp if hp is not None else stats.hp,
        surfaced=surfaced,
        movement_remaining=stats.movement,
    )


def _game_state(board: Board, ships: list[Ship], config: Config | None = None) -> GameState:
    return GameState(config=config or Config(fow=FowConfig(enabled=False)), board=board, ships={s.id: s for s in ships})


def _run(policy: NaivePolicy, gs: GameState, player) -> None:
    list(policy.plan_movement(gs, player))


def test_ai_moves_toward_a_visible_enemy_it_cannot_yet_reach():
    board = _sea_board(radius=10)
    mover = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)  # movement 3
    enemy = _ship(AxialCoord(8, 0), ShipKind.DESTROYER, PLAYER_B, 2)  # well out of reach this turn
    gs = _game_state(board, [mover, enemy])

    _run(NaivePolicy(), gs, PLAYER_A)

    assert 1 in gs.ships
    assert distance(gs.ships[1].position, enemy.position) < 8


def test_ai_never_issues_an_illegal_move():
    # A broad smoke test: run several ships of every kind through a shared
    # board with both sides present and assert nothing raises -- every
    # movement/battle/production call the AI makes goes through the same
    # legality checks a human's input does, so a clean run here is direct
    # evidence nothing illegal happened.
    board = _sea_board()
    ships = []
    next_id = 1
    for i, kind in enumerate(ShipKind):
        ships.append(_ship(AxialCoord(i, 0), kind, PLAYER_A, next_id))
        next_id += 1
        ships.append(_ship(AxialCoord(i, 3), kind, PLAYER_B, next_id))
        next_id += 1
    gs = _game_state(board, ships)

    _run(NaivePolicy(), gs, PLAYER_A)
    _run(NaivePolicy(), gs, PLAYER_B)


def test_ai_declines_a_clearly_losing_engagement():
    board = _sea_board()
    patrol_boat = _ship(AxialCoord(0, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)
    battleship = _ship(AxialCoord(1, 0), ShipKind.BATTLESHIP, PLAYER_B, 2, hp=100)
    gs = _game_state(board, [patrol_boat, battleship])

    _run(NaivePolicy(), gs, PLAYER_A)

    # The patrol boat should still be alive (didn't attack and get wiped
    # out) and the battleship untouched (never engaged).
    assert 1 in gs.ships
    assert gs.ships[2].current_hp == 100


def test_ai_attacks_a_clearly_favorable_target():
    board = _sea_board()
    battleship = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    patrol_boat = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    gs = _game_state(board, [battleship, patrol_boat])

    _run(NaivePolicy(), gs, PLAYER_A)

    assert 2 not in gs.ships  # the patrol boat was sunk


def test_ai_retreats_mid_battle_once_badly_damaged():
    # Two destroyers, attacker at full HP vs an already-damaged defender --
    # a clearly favorable race (matchup_score > 0), so the attack actually
    # starts. A near-90% damaged-withdraw threshold then pulls the attacker
    # out after the very first round purely because of its own HP loss,
    # independent of the race still looking favorable.
    board = _sea_board()
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(damaged_withdraw_fraction=0.9))
    attacker = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    defender = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_B, 2, hp=4)
    gs = _game_state(board, [attacker, defender], config=config)

    _run(NaivePolicy(), gs, PLAYER_A)

    assert 1 in gs.ships
    assert gs.ships[1].position == AxialCoord(0, 0)  # retreated to its approach hex


def test_ai_respects_fog_of_war_and_ignores_a_ship_it_cannot_see():
    board = _sea_board()
    config = Config(fow=FowConfig(enabled=True, ship_visibility_radius=1))
    mover = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    far_enemy = _ship(AxialCoord(6, 0), ShipKind.DESTROYER, PLAYER_B, 2)
    port = AxialCoord(-6, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    gs = _game_state(board, [mover, far_enemy], config=config)

    _run(NaivePolicy(), gs, PLAYER_A)

    # With nothing visible, the AI should head toward the uncontrolled
    # port (there is none of its own already controlled here) rather than
    # magically beelining for the enemy it can't see.
    assert distance(gs.ships[1].position, far_enemy.position) >= distance(mover.position, far_enemy.position)


def test_ai_never_reveals_a_hidden_submerged_submarine_by_avoiding_it():
    # A submerged enemy sub outside vision is invisible -- the AI has no
    # way to know to route around it, so a ship that ends up moving onto
    # it should trigger a real battle exactly like a human's accidental
    # contact would, not silently dodge it.
    board = _sea_board()
    config = Config(fow=FowConfig(enabled=True, ship_visibility_radius=1))
    mover = _ship(AxialCoord(0, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)
    mover.movement_remaining = 1
    sub = _ship(AxialCoord(1, 0), ShipKind.SUBMARINE, PLAYER_B, 2, surfaced=False)
    gs = _game_state(board, [mover, sub], config=config)

    _run(NaivePolicy(), gs, PLAYER_A)

    # No visible enemy and no uncontrolled port anywhere -> the naive
    # policy has nothing to do and holds position; this just confirms
    # nothing raised despite a totally hidden adjacent threat.
    assert 1 in gs.ships


def test_carrier_never_initiates_an_attack():
    board = _sea_board()
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    weak_enemy = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    gs = _game_state(board, [carrier, weak_enemy])

    _run(NaivePolicy(), gs, PLAYER_A)

    assert 2 in gs.ships  # never attacked, even though it's a favorable matchup on paper
    assert gs.ships[2].current_hp == weak_enemy.current_hp


def test_carrier_falls_back_toward_its_escort_when_threatened():
    board = _sea_board()
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(carrier_threat_radius=5))
    carrier_start, escort_start, enemy_start = AxialCoord(0, 0), AxialCoord(-3, 0), AxialCoord(2, 0)
    carrier = _ship(carrier_start, ShipKind.CARRIER, PLAYER_A, 1)
    escort = _ship(escort_start, ShipKind.DESTROYER, PLAYER_A, 2)
    enemy = _ship(enemy_start, ShipKind.DESTROYER, PLAYER_B, 3)
    gs = _game_state(board, [carrier, escort, enemy], config=config)

    _run(NaivePolicy(), gs, PLAYER_A)

    # Ships are mutated in place, so compare against the coordinates
    # captured before the run, not the (now-moved) fixture objects.
    carrier_after = gs.ships[1]
    assert distance(carrier_after.position, escort_start) < distance(carrier_start, escort_start)
    assert distance(carrier_after.position, enemy_start) > distance(carrier_start, enemy_start)


def test_capital_ship_proceeds_unescorted_when_no_scout_is_available():
    # A previous version held the capital ship in place regardless of
    # whether an escort was actually available, which could camp it
    # indefinitely if one never was -- often right on its own home port,
    # silently blocking that port's production (see
    # tla.ai.policy._scout_prepass). Scouting is now purely opportunistic:
    # with no one free to sweep ahead, the capital ship just proceeds with
    # its own plan instead of freezing.
    board = _sea_board()
    battleship = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    enemy = _ship(AxialCoord(6, 0), ShipKind.SUBMARINE, PLAYER_B, 2, surfaced=False)
    # No patrol boat/destroyer escort anywhere in the fleet.
    gs = _game_state(board, [battleship, enemy])

    _run(NaivePolicy(), gs, PLAYER_A)

    assert gs.ships[1].position != AxialCoord(0, 0)  # proceeded, not held


def test_capital_ship_sends_an_escort_ahead_before_advancing():
    board = _sea_board()
    battleship = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    scout = _ship(AxialCoord(0, 1), ShipKind.DESTROYER, PLAYER_A, 2)
    enemy = _ship(AxialCoord(4, 0), ShipKind.SUBMARINE, PLAYER_B, 3, surfaced=False)
    gs = _game_state(board, [battleship, scout, enemy])

    _run(NaivePolicy(), gs, PLAYER_A)

    # The battleship holds; the scout is the one that moved this turn.
    assert gs.ships[1].position == AxialCoord(0, 0)
    assert gs.ships[2].position != AxialCoord(0, 1)


def test_scout_prepass_survives_losing_the_last_visible_contact_mid_pass():
    # Regression: fog-of-war vision is recomputed fresh from each ship's
    # current position (see tla.fow), so a scout's own move earlier in the
    # prepass can shrink the player's vision enough to lose the only
    # visible enemy entirely -- before this was guarded, the *next*
    # capital ship's scout-distance check crashed with
    # "min() iterable argument is empty" instead of gracefully falling
    # back to normal (non-scouted) movement. Reproduced deterministically
    # by mocking enemy_ships_visible_to to go from one visible enemy to
    # none, rather than relying on fragile real-geometry vision loss.
    board = _sea_board(radius=25)
    port = AxialCoord(0, 20)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    capital1 = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    escort1 = _ship(AxialCoord(0, -3), ShipKind.DESTROYER, PLAYER_A, 2)
    capital2 = _ship(AxialCoord(5, 0), ShipKind.BATTLESHIP, PLAYER_A, 3)
    fake_enemy = _ship(AxialCoord(0, -6), ShipKind.DESTROYER, PLAYER_B, 4)
    gs = _game_state(board, [capital1, escort1, capital2, fake_enemy], config=Config(fow=FowConfig(enabled=False)))

    calls = [{4: fake_enemy}, {}]

    def fake_visible(game_state, player):
        return calls.pop(0) if calls else {}

    with patch("tla.ai.policy.enemy_ships_visible_to", side_effect=fake_visible):
        list(NaivePolicy().plan_movement(gs, PLAYER_A))  # must not raise


def test_task_force_declines_a_favorable_target_backed_by_a_bigger_force():
    board = _sea_board()
    destroyer = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    bait = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)  # an easy 1v1 alone...
    backup = _ship(AxialCoord(2, 0), ShipKind.BATTLESHIP, PLAYER_B, 3, hp=100)  # ...but not alone
    port = AxialCoord(6, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    config = Config(
        fow=FowConfig(enabled=False), ai=AiConfig(task_force_threat_radius=4, task_force_outnumbered_margin=0)
    )
    gs = _game_state(board, [destroyer, bait, backup], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})
    goal = TaskForceGoal(kind=GoalKind.CAPTURE_PORT, target=port)

    destination = choose_task_force_destination(destroyer, goal, gs, {2: bait, 3: backup}, force)

    assert destination != bait.position


def test_task_force_still_attacks_a_favorable_target_with_no_backup():
    # Same shape as above minus the backup ship -- confirms the new check
    # only blocks attacks that are actually outmatched, not attacks in
    # general.
    board = _sea_board()
    destroyer = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    bait = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    port = AxialCoord(6, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    config = Config(
        fow=FowConfig(enabled=False), ai=AiConfig(task_force_threat_radius=4, task_force_outnumbered_margin=0)
    )
    gs = _game_state(board, [destroyer, bait], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})
    goal = TaskForceGoal(kind=GoalKind.CAPTURE_PORT, target=port)

    destination = choose_task_force_destination(destroyer, goal, gs, {2: bait}, force)

    assert destination == bait.position


def test_retreating_task_force_never_attacks_even_a_clearly_favorable_target():
    # "Retreat and wait for reinforcements" means actually disengaging --
    # a RETREAT goal must never fight anything on the way home, even an
    # easy, unsupported target it happens to pass.
    board = _sea_board()
    destroyer = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    bait = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    home_port = AxialCoord(-3, 0)
    board.tiles[home_port] = Tile(coord=home_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    gs = _game_state(board, [destroyer, bait], config=Config(fow=FowConfig(enabled=False)))
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})
    retreat_goal = TaskForceGoal(kind=GoalKind.RETREAT, target=home_port)

    destination = choose_task_force_destination(destroyer, retreat_goal, gs, {2: bait}, force)

    assert destination != bait.position


def test_unassigned_ship_declines_a_favorable_target_backed_by_a_bigger_force():
    board = _sea_board()
    destroyer = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    bait = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    backup = _ship(AxialCoord(2, 0), ShipKind.BATTLESHIP, PLAYER_B, 3, hp=100)
    config = Config(
        fow=FowConfig(enabled=False), ai=AiConfig(task_force_threat_radius=4, task_force_outnumbered_margin=0)
    )
    gs = _game_state(board, [destroyer, bait, backup], config=config)

    attack_hex = _favorable_attack(destroyer, gs, {2: bait, 3: backup}, [destroyer])

    assert attack_hex is None


def test_step_toward_respects_max_steps():
    board = _sea_board(radius=10)
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    ship.movement_remaining = 5
    gs = _game_state(board, [ship])
    target = AxialCoord(8, 0)

    unrestricted = _step_toward(ship, gs, target)
    paced = _step_toward(ship, gs, target, max_steps=2)

    assert distance(AxialCoord(0, 0), unrestricted) == 5
    assert distance(AxialCoord(0, 0), paced) == 2


def test_compute_force_pace_takes_the_slowest_non_submarine_member():
    board = _sea_board()
    fast = _ship(AxialCoord(0, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)  # movement 6
    slow = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_A, 2)  # movement 3
    sub = _ship(AxialCoord(2, 0), ShipKind.SUBMARINE, PLAYER_A, 3)
    sub.movement_remaining = 1  # e.g. submerged -- slower than either surface ship
    gs = _game_state(board, [fast, slow, sub])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2, 3})

    pace = _compute_force_pace(gs, [force])

    assert pace[1] == 3  # the destroyer's speed -- the submarine doesn't drag it down further


def test_compute_force_pace_omits_a_force_of_submarines_only():
    board = _sea_board()
    sub = _ship(AxialCoord(0, 0), ShipKind.SUBMARINE, PLAYER_A, 1)
    gs = _game_state(board, [sub])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})

    pace = _compute_force_pace(gs, [force])

    assert 1 not in pace


def test_task_force_advance_is_paced_to_the_slowest_member():
    board = _sea_board(radius=10)
    fast = _ship(AxialCoord(0, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)  # movement 6
    slow = _ship(AxialCoord(0, 1), ShipKind.DESTROYER, PLAYER_A, 2)  # movement 3
    port = AxialCoord(8, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    gs = _game_state(board, [fast, slow])
    force = TaskForce(
        id=1, owner=PLAYER_A, member_ids={1, 2}, goal=TaskForceGoal(kind=GoalKind.CAPTURE_PORT, target=port)
    )
    pace = _compute_force_pace(gs, [force])
    assert pace[1] == 3  # the destroyer's speed

    destination = _choose_destination(fast, gs, PLAYER_A, {}, [force], pace)

    # The patrol boat could reach 6 hexes toward the port unpaced -- capped
    # to the destroyer's pace instead, so the group doesn't fragment.
    assert distance(AxialCoord(0, 0), destination) == 3


def test_carrier_advance_is_also_paced_to_the_slowest_member():
    board = _sea_board(radius=10)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)  # movement 4
    slow = _ship(AxialCoord(0, 1), ShipKind.DESTROYER, PLAYER_A, 2)  # movement 3
    port = AxialCoord(8, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    gs = _game_state(board, [carrier, slow])
    force = TaskForce(
        id=1, owner=PLAYER_A, member_ids={1, 2}, goal=TaskForceGoal(kind=GoalKind.CAPTURE_PORT, target=port)
    )
    pace = _compute_force_pace(gs, [force])

    destination = _choose_destination(carrier, gs, PLAYER_A, {}, [force], pace)

    assert distance(AxialCoord(0, 0), destination) == 3  # not the carrier's own faster 4


def test_retreating_task_force_is_not_paced():
    board = _sea_board(radius=10)
    fast = _ship(AxialCoord(0, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)  # movement 6
    slow = _ship(AxialCoord(0, 1), ShipKind.DESTROYER, PLAYER_A, 2)  # movement 3
    home_port = AxialCoord(-8, 0)
    board.tiles[home_port] = Tile(coord=home_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    gs = _game_state(board, [fast, slow])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})
    retreat_goal = TaskForceGoal(kind=GoalKind.RETREAT, target=home_port)
    pace = {1: 3}  # as if _compute_force_pace had capped this force to 3

    destination = choose_task_force_destination(fast, retreat_goal, gs, {}, force, max_steps=pace[1])

    # choose_task_force_destination never applies max_steps on a RETREAT
    # goal in the first place -- get home without dawdling.
    assert distance(AxialCoord(0, 0), destination) > 3




def test_scout_prepass_increments_hold_count_while_under_the_cap():
    board = _sea_board()
    battleship = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    scout = _ship(AxialCoord(0, 1), ShipKind.DESTROYER, PLAYER_A, 2)
    enemy = _ship(AxialCoord(4, 0), ShipKind.SUBMARINE, PLAYER_B, 3, surfaced=False)
    gs = _game_state(board, [battleship, scout, enemy])
    hold_counts: dict[int, int] = {}
    resolved: set[int] = set()

    list(_scout_prepass(gs, PLAYER_A, resolved, NaivePolicy().decide_battle, [], {}, hold_counts))

    assert 1 in resolved  # held this turn, as before
    assert hold_counts == {1: 1}


def test_scout_prepass_stops_holding_once_the_cap_is_reached():
    # Distinct from test_capital_ship_proceeds_unescorted_when_no_scout_is_
    # available: here an escort genuinely *is* available every turn, so
    # holding is individually justified each time -- but nothing about that
    # guarantees the capital ship ever actually gets to move. Without a cap
    # this could repeat indefinitely (confirmed against game11's replay: a
    # carrier sat motionless for 5 turns straight this way).
    board = _sea_board()
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(max_consecutive_scout_holds=2))
    battleship = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    scout = _ship(AxialCoord(0, 1), ShipKind.DESTROYER, PLAYER_A, 2)
    enemy = _ship(AxialCoord(4, 0), ShipKind.SUBMARINE, PLAYER_B, 3, surfaced=False)
    gs = _game_state(board, [battleship, scout, enemy], config=config)
    hold_counts = {1: 2}  # already held 2 turns running -- at the configured cap
    resolved: set[int] = set()

    list(_scout_prepass(gs, PLAYER_A, resolved, NaivePolicy().decide_battle, [], {}, hold_counts))

    assert 1 not in resolved  # proceeds with its own plan instead of holding a 3rd time
    assert 1 not in hold_counts  # streak cleared, not left dangling at the cap


def test_plan_movement_redirects_every_force_the_turn_a_port_is_lost():
    board = _sea_board(radius=12)
    lost_port = AxialCoord(5, 0)
    board.tiles[lost_port] = Tile(
        coord=lost_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A, port_controller=PLAYER_A
    )
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    gs = _game_state(board, [ship], config=Config(fow=FowConfig(enabled=False), ai=AiConfig(task_force_min_size=1)))
    original_goal = TaskForceGoal(GoalKind.CAPTURE_PORT, AxialCoord(-8, 0))
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1}, goal=original_goal)

    policy = NaivePolicy()
    policy._task_forces[PLAYER_A] = [force]
    policy._next_force_id = 2

    _run(policy, gs, PLAYER_A)  # first call -- just establishes the "controlled" snapshot
    assert force.goal == original_goal, "no port lost yet -- goal must be untouched"

    # An enemy ship now occupies the port -- a capture, per Tile.port_display_owner.
    board.tiles[lost_port] = Tile(
        coord=lost_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A, port_controller=PLAYER_B
    )
    ship.movement_remaining = Config().ship_stats.stats[ShipKind.DESTROYER].movement

    _run(policy, gs, PLAYER_A)

    assert force.goal.target == lost_port


def test_plan_movement_port_loss_redirect_leaves_a_retreating_forces_stance_alone():
    board = _sea_board(radius=12)
    lost_port = AxialCoord(5, 0)
    board.tiles[lost_port] = Tile(
        coord=lost_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A, port_controller=PLAYER_A
    )
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    gs = _game_state(board, [ship], config=Config(fow=FowConfig(enabled=False), ai=AiConfig(task_force_min_size=1)))
    force = TaskForce(
        id=1,
        owner=PLAYER_A,
        member_ids={1},
        goal=TaskForceGoal(GoalKind.CAPTURE_PORT, AxialCoord(-8, 0)),
        retreating=True,
        retreat_turns=1,
        retreat_threat_power=(20, 8),
    )

    policy = NaivePolicy()
    policy._task_forces[PLAYER_A] = [force]
    policy._next_force_id = 2
    policy._controlled_ports_seen[PLAYER_A] = {lost_port}  # as if already established last turn

    board.tiles[lost_port] = Tile(
        coord=lost_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A, port_controller=PLAYER_B
    )

    _run(policy, gs, PLAYER_A)

    assert force.goal.target == lost_port  # the real goal was redirected...
    # ...but the redirect itself must not force the force out of retreat --
    # with no enemy anywhere on this board, a lone destroyer (hp=6, dmg=2)
    # still doesn't outmatch the snapshotted (20, 8) threat, so
    # update_task_force_stance keeps it retreating on its own merits,
    # exactly as it would have without any port having been lost.
    assert force.retreating is True
    assert force.retreat_turns == 2  # ticked up by one more turn, not reset
