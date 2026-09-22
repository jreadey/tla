from tla import movement
from tla.ai.policy import (
    NaivePolicy,
    _carrier_bonus_cohesion_destination,
    _carrier_cautious_max_steps,
    _choose_carrier_destination,
    _choose_destination,
    _cohesion_destination,
    _compute_force_pace,
    _favorable_attack,
    _maybe_toggle_submarine,
    _rearguard_target,
    _step_toward,
    choose_task_force_destination,
)
from tla.ai.task_force import GoalKind, TaskForce, TaskForceGoal, compute_port_defense_directives
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


# -- submarine surface/submerge doctrine (see _maybe_toggle_submarine) ------


def test_maybe_toggle_submarine_submerges_with_a_visible_enemy_and_no_attack():
    board = _sea_board()
    sub = _ship(AxialCoord(0, 0), ShipKind.SUBMARINE, PLAYER_A, 1)
    enemy = _ship(AxialCoord(6, 0), ShipKind.BATTLESHIP, PLAYER_B, 2)  # visible, but out of reach
    gs = _game_state(board, [sub, enemy])

    _maybe_toggle_submarine(sub, gs, {2: enemy})

    assert sub.surfaced is False


def test_maybe_toggle_submarine_stays_surfaced_with_no_visible_enemy():
    board = _sea_board()
    sub = _ship(AxialCoord(0, 0), ShipKind.SUBMARINE, PLAYER_A, 1)
    gs = _game_state(board, [sub])

    _maybe_toggle_submarine(sub, gs, {})

    assert sub.surfaced is True


def test_maybe_toggle_submarine_resurfaces_once_nothing_is_visible():
    # The fix for a real, twice-confirmed-in-replay bug: a submerged sub
    # with nothing around to hide from used to just stay submerged
    # forever, crawling at its submerged speed for the rest of the game.
    board = _sea_board()
    sub = _ship(AxialCoord(0, 0), ShipKind.SUBMARINE, PLAYER_A, 1, surfaced=False)
    stats = Config().ship_stats.stats[ShipKind.SUBMARINE]
    sub.movement_remaining = stats.movement_submerged  # matches start_movement_phase's own budget
    gs = _game_state(board, [sub])

    _maybe_toggle_submarine(sub, gs, {})

    assert sub.surfaced is True
    assert sub.movement_remaining == stats.movement  # refreshed to the full surfaced budget


def test_maybe_toggle_submarine_stays_submerged_with_a_visible_enemy():
    board = _sea_board()
    sub = _ship(AxialCoord(0, 0), ShipKind.SUBMARINE, PLAYER_A, 1, surfaced=False)
    sub.movement_remaining = Config().ship_stats.stats[ShipKind.SUBMARINE].movement_submerged
    enemy = _ship(AxialCoord(6, 0), ShipKind.BATTLESHIP, PLAYER_B, 2)
    gs = _game_state(board, [sub, enemy])

    _maybe_toggle_submarine(sub, gs, {2: enemy})

    assert sub.surfaced is False


def test_maybe_toggle_submarine_never_disturbs_an_already_reachable_favorable_attack():
    board = _sea_board()
    sub = _ship(AxialCoord(0, 0), ShipKind.SUBMARINE, PLAYER_A, 1)
    weak_enemy = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    gs = _game_state(board, [sub, weak_enemy])

    _maybe_toggle_submarine(sub, gs, {2: weak_enemy})

    # Stays surfaced -- submerging now would risk losing the range/attack
    # this turn's move is about to take.
    assert sub.surfaced is True


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
    # A dangerous kind (_DANGEROUS_TO_CARRIER_KINDS) -- a destroyer alone
    # wouldn't trigger this, see test_carrier_ignores_a_destroyer_nearby.
    enemy = _ship(enemy_start, ShipKind.CRUISER, PLAYER_B, 3)
    gs = _game_state(board, [carrier, escort, enemy], config=config)

    _run(NaivePolicy(), gs, PLAYER_A)

    # Ships are mutated in place, so compare against the coordinates
    # captured before the run, not the (now-moved) fixture objects.
    carrier_after = gs.ships[1]
    assert distance(carrier_after.position, escort_start) < distance(carrier_start, escort_start)
    assert distance(carrier_after.position, enemy_start) > distance(carrier_start, enemy_start)


def test_carrier_ignores_a_destroyer_nearby():
    board = _sea_board()
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(carrier_threat_radius=5))
    carrier_start = AxialCoord(0, 0)
    carrier = _ship(carrier_start, ShipKind.CARRIER, PLAYER_A, 1)
    # Within carrier_threat_radius, but not a kind the carrier's own
    # threat-avoidance worries about -- see _DANGEROUS_TO_CARRIER_KINDS.
    enemy = _ship(AxialCoord(2, 0), ShipKind.DESTROYER, PLAYER_B, 2)
    gs = _game_state(board, [carrier, enemy], config=config)

    _run(NaivePolicy(), gs, PLAYER_A)

    # No force/goal and no other own warship to close distance to, so an
    # unmolested carrier stays put -- if the destroyer had tripped the
    # threat check, it would have fled instead.
    assert gs.ships[1].position == carrier_start


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
    # Isolates pacing from AiConfig.carrier_advance_reserve (see the
    # dedicated reserve tests below) -- zeroed here so this test still
    # proves the pace cap alone, not the two caps combined.
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(carrier_advance_reserve=0))
    gs = _game_state(board, [carrier, slow], config=config)
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




def test_rearguard_target_is_the_rearmost_capital_ships_position():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    escort = _ship(AxialCoord(3, 0), ShipKind.DESTROYER, PLAYER_A, 1)  # ahead of both capital ships
    lead = _ship(AxialCoord(4, 0), ShipKind.BATTLESHIP, PLAYER_A, 2)
    rear = _ship(AxialCoord(1, 0), ShipKind.CARRIER, PLAYER_A, 3)  # made the least progress
    gs = _game_state(board, [escort, lead, rear])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2, 3})

    target = _rearguard_target(escort, force, gs, port)

    assert target == rear.position


def test_rearguard_target_is_none_without_a_capital_ship_in_the_force():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    escort = _ship(AxialCoord(3, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    other_escort = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_A, 2)
    gs = _game_state(board, [escort, other_escort])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})

    assert _rearguard_target(escort, force, gs, port) is None


def test_rearguard_target_is_none_once_the_escort_is_already_behind():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    escort = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)  # already the rearmost of the group
    lead = _ship(AxialCoord(4, 0), ShipKind.BATTLESHIP, PLAYER_A, 2)
    gs = _game_state(board, [escort, lead])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})

    assert _rearguard_target(escort, force, gs, port) is None


def test_task_force_destroyer_trails_a_lone_battleship_instead_of_the_goal():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    destroyer = _ship(AxialCoord(3, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    battleship = _ship(AxialCoord(1, 0), ShipKind.BATTLESHIP, PLAYER_A, 2)  # behind the destroyer
    gs = _game_state(board, [destroyer, battleship])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, port))

    destination = choose_task_force_destination(destroyer, force.goal, gs, {}, force, max_steps=None)

    # Heads back toward the battleship, not onward toward the port.
    assert destination is not None
    assert distance(destination, battleship.position) < distance(destroyer.position, battleship.position)


# -- task_force_max_separation (see _cohesion_destination) -------------------


def test_cohesion_destination_is_none_when_not_configured():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    ship = _ship(AxialCoord(9, 0), ShipKind.CRUISER, PLAYER_A, 1)
    teammate = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 2)
    gs = _game_state(board, [ship, teammate], config=Config(fow=FowConfig(enabled=False)))  # default AiConfig
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})

    assert _cohesion_destination(ship, force, gs, port, frozenset()) is None


def test_cohesion_destination_is_none_when_already_within_range():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    ship = _ship(AxialCoord(7, 0), ShipKind.CRUISER, PLAYER_A, 1)
    teammate = _ship(AxialCoord(4, 0), ShipKind.DESTROYER, PLAYER_A, 2)  # only 3 hexes away
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(task_force_max_separation=4))
    gs = _game_state(board, [ship, teammate], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})

    assert _cohesion_destination(ship, force, gs, port, frozenset()) is None


def test_cohesion_destination_closes_the_gap_when_reachable_in_one_move():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    ship = _ship(AxialCoord(6, 0), ShipKind.CRUISER, PLAYER_A, 1)  # movement 4
    teammate = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 2)  # 6 hexes away -- over the cap
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(task_force_max_separation=4))
    gs = _game_state(board, [ship, teammate], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})

    destination = _cohesion_destination(ship, force, gs, port, frozenset())

    # A one-move-away compliant hex exists (e.g. (2,0)), so the result must
    # actually satisfy the cap, not just move in the right direction --
    # this is exactly what an earlier, pre-move-only version of this check
    # got wrong (it let a ship right at the boundary overshoot the cap).
    assert destination is not None
    assert distance(destination, teammate.position) <= 4


def test_cohesion_destination_minimizes_separation_when_the_cap_is_unreachable_in_one_move():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    ship = _ship(AxialCoord(10, 0), ShipKind.CRUISER, PLAYER_A, 1)  # movement 4
    teammate = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 2)  # 10 hexes -- can't close to <=4 in one hop
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(task_force_max_separation=4))
    gs = _game_state(board, [ship, teammate], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})

    destination = _cohesion_destination(ship, force, gs, port, frozenset())

    # Can't fully comply in one move -- best effort: close the gap as much
    # as this turn's movement allows instead of ignoring the cap entirely.
    assert destination is not None
    assert distance(destination, teammate.position) < distance(ship.position, teammate.position)


def test_cohesion_destination_never_applies_to_a_submarine():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    sub = _ship(AxialCoord(9, 0), ShipKind.SUBMARINE, PLAYER_A, 1)
    teammate = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 2)
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(task_force_max_separation=4))
    gs = _game_state(board, [sub, teammate], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})

    assert _cohesion_destination(sub, force, gs, port, frozenset()) is None


def test_cohesion_destination_excludes_enemy_occupied_hexes():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    ship = _ship(AxialCoord(6, 0), ShipKind.CRUISER, PLAYER_A, 1)
    teammate = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 2)
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(task_force_max_separation=4))
    gs = _game_state(board, [ship, teammate], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})
    enemy_hexes = frozenset(movement.reachable_hexes(ship, gs))  # pretend every reachable hex is enemy-held

    assert _cohesion_destination(ship, force, gs, port, enemy_hexes) is None


def test_task_force_destination_prefers_cohesion_over_the_goal():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    ship = _ship(AxialCoord(6, 0), ShipKind.CRUISER, PLAYER_A, 1)
    teammate = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 2)
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(task_force_max_separation=4))
    gs = _game_state(board, [ship, teammate], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, port))

    destination = choose_task_force_destination(ship, force.goal, gs, {}, force, max_steps=None)

    assert destination is not None
    assert distance(destination, teammate.position) <= 4


def test_carrier_destination_prefers_cohesion_over_the_goal_when_not_threatened():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    carrier = _ship(AxialCoord(6, 0), ShipKind.CARRIER, PLAYER_A, 1)
    teammate = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 2)
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(task_force_max_separation=4))
    gs = _game_state(board, [carrier, teammate], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, port))

    destination = _choose_carrier_destination(carrier, gs, PLAYER_A, {}, goal=force.goal, force=force)

    assert destination is not None
    assert distance(destination, teammate.position) <= 4


def test_carrier_advance_steers_around_a_visible_enemys_threat_radius():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)  # movement 4
    # Sits on the straight-line path to the port, but far enough away that
    # the carrier's own immediate-threat check (carrier_threat_radius)
    # never fires -- this is about the *destination* choice, not that
    # reactive safety net. A dangerous kind (_DANGEROUS_TO_CARRIER_KINDS),
    # since a destroyer wouldn't be avoided at all.
    enemy = _ship(AxialCoord(5, 0), ShipKind.CRUISER, PLAYER_B, 2)
    # carrier_advance_reserve zeroed to isolate danger-zone avoidance from
    # the separate cautious-advance cap (see the dedicated reserve tests).
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(carrier_threat_radius=2, carrier_advance_reserve=0))
    gs = _game_state(board, [carrier, enemy], config=config)
    goal = TaskForceGoal(GoalKind.CAPTURE_PORT, port)

    destination = _choose_carrier_destination(carrier, gs, PLAYER_A, {2: enemy}, goal=goal)

    # A plain step straight toward the port would land on (4, 0), only 1
    # hex from the enemy -- well inside carrier_threat_radius.
    assert destination is not None
    assert distance(destination, enemy.position) > config.ai.carrier_threat_radius


# -- cautious carrier advance (see _carrier_cautious_max_steps) --------------


def test_carrier_cautious_max_steps_reserves_the_configured_points():
    board = _sea_board(radius=10)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)  # movement 4
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(carrier_advance_reserve=2))
    gs = _game_state(board, [carrier], config=config)

    assert _carrier_cautious_max_steps(carrier, gs, max_steps=None) == 2


def test_carrier_cautious_max_steps_respects_a_tighter_external_cap():
    board = _sea_board(radius=10)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)  # movement 4
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(carrier_advance_reserve=2))
    gs = _game_state(board, [carrier], config=config)

    # The force's own shared pace (1) is tighter than what the reserve
    # alone would allow (2) -- the smaller of the two wins.
    assert _carrier_cautious_max_steps(carrier, gs, max_steps=1) == 1


def test_carrier_cautious_max_steps_floors_at_zero():
    board = _sea_board(radius=10)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)  # movement 4
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(carrier_advance_reserve=10))  # exceeds movement 4
    gs = _game_state(board, [carrier], config=config)

    assert _carrier_cautious_max_steps(carrier, gs, max_steps=None) == 0


def test_carrier_advance_toward_a_goal_holds_back_the_reserve():
    board = _sea_board(radius=10)
    port = AxialCoord(8, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)  # movement 4
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(carrier_advance_reserve=2))
    gs = _game_state(board, [carrier], config=config)
    goal = TaskForceGoal(GoalKind.CAPTURE_PORT, port)

    destination = _choose_carrier_destination(carrier, gs, PLAYER_A, {}, goal=goal)

    # Unpaced (no force, no slower teammate) and nothing visible, so
    # without the reserve this would reach the carrier's own full 4.
    assert destination is not None
    assert distance(AxialCoord(0, 0), destination) == 2


# -- carrier-bonus cohesion (see _carrier_bonus_cohesion_destination) --------


def test_carrier_bonus_cohesion_is_none_for_a_carrier_itself():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    gs = _game_state(board, [carrier])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})

    assert _carrier_bonus_cohesion_destination(carrier, force, gs, port, frozenset()) is None


def test_carrier_bonus_cohesion_is_none_without_a_living_carrier_in_the_force():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    cruiser = _ship(AxialCoord(9, 0), ShipKind.CRUISER, PLAYER_A, 1)
    gs = _game_state(board, [cruiser])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})

    assert _carrier_bonus_cohesion_destination(cruiser, force, gs, port, frozenset()) is None


def test_carrier_bonus_cohesion_is_none_when_already_within_radius():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    cruiser = _ship(AxialCoord(9, 0), ShipKind.CRUISER, PLAYER_A, 1)
    carrier = _ship(AxialCoord(8, 0), ShipKind.CARRIER, PLAYER_A, 2)  # 1 hex away -- within default radius 2
    gs = _game_state(board, [cruiser, carrier])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})

    assert _carrier_bonus_cohesion_destination(cruiser, force, gs, port, frozenset()) is None


def test_carrier_bonus_cohesion_closes_the_gap_when_reachable_in_one_move():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    cruiser = _ship(AxialCoord(4, 0), ShipKind.CRUISER, PLAYER_A, 1)  # movement 4
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 2)  # 4 hexes -- over the default radius 2
    gs = _game_state(board, [cruiser, carrier])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})

    destination = _carrier_bonus_cohesion_destination(cruiser, force, gs, port, frozenset())

    assert destination is not None
    assert distance(destination, carrier.position) <= 2


def test_carrier_bonus_cohesion_minimizes_separation_when_the_cap_is_unreachable_in_one_move():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    cruiser = _ship(AxialCoord(10, 0), ShipKind.CRUISER, PLAYER_A, 1)  # movement 4
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 2)  # 10 hexes -- can't close to <=2 in one hop
    gs = _game_state(board, [cruiser, carrier])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})

    destination = _carrier_bonus_cohesion_destination(cruiser, force, gs, port, frozenset())

    assert destination is not None
    assert distance(destination, carrier.position) < distance(cruiser.position, carrier.position)


def test_carrier_bonus_cohesion_excludes_enemy_occupied_hexes():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    cruiser = _ship(AxialCoord(4, 0), ShipKind.CRUISER, PLAYER_A, 1)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 2)
    gs = _game_state(board, [cruiser, carrier])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})
    enemy_hexes = frozenset(movement.reachable_hexes(cruiser, gs))  # pretend every reachable hex is enemy-held

    assert _carrier_bonus_cohesion_destination(cruiser, force, gs, port, enemy_hexes) is None


def test_task_force_destination_prefers_carrier_bonus_cohesion_over_the_goal():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    cruiser = _ship(AxialCoord(6, 0), ShipKind.CRUISER, PLAYER_A, 1)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 2)
    gs = _game_state(board, [cruiser, carrier], config=Config(fow=FowConfig(enabled=False)))  # default AiConfig
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, port))

    destination = choose_task_force_destination(cruiser, force.goal, gs, {}, force, max_steps=None)

    # No task_force_max_separation configured, so the general cohesion
    # check is disabled -- this pull toward the carrier can only be
    # _carrier_bonus_cohesion_destination.
    assert destination is not None
    assert distance(destination, carrier.position) <= 2


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


# -- port defense (see tla.ai.task_force.compute_port_defense_directives) --


def _port_board(port: AxialCoord, radius: int = 10) -> Board:
    board = _sea_board(radius=radius)
    board.tiles[port] = Tile(
        coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A, port_controller=PLAYER_A
    )
    return board


def test_plan_movement_counterattacks_a_threat_near_a_controlled_port():
    port = AxialCoord(0, 0)
    board = _port_board(port)
    defender = _ship(AxialCoord(1, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    weak_threat = _ship(AxialCoord(3, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)  # within reach and clearly losing
    gs = _game_state(board, [defender, weak_threat], config=Config(fow=FowConfig(enabled=False)))

    _run(NaivePolicy(), gs, PLAYER_A)

    # A reachable, clearly favorable target within trigger range -- the
    # defender should have engaged it directly rather than pursuing
    # whatever _choose_generic_destination would otherwise have picked.
    assert len(gs.battle_log) >= 1
    assert gs.battle_log[0].attacker_id == 1
    assert gs.battle_log[0].defender_id == 2


def test_plan_movement_port_defense_does_not_trigger_with_no_controlled_port():
    board = _sea_board(radius=10)  # no port on the board at all
    defender = _ship(AxialCoord(10, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    distant_enemy = _ship(AxialCoord(-10, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    gs = _game_state(board, [defender, distant_enemy], config=Config(fow=FowConfig(enabled=False)))

    directives = compute_port_defense_directives(gs, PLAYER_A, {2: distant_enemy}, gs.config.ai)

    assert directives == []  # nothing to defend -- controlled_ports_for(PLAYER_A) is empty


def test_plan_movement_blocks_with_a_submarine_when_outmatched():
    port = AxialCoord(0, 0)
    board = _port_board(port)
    weak = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)
    sub = _ship(AxialCoord(1, 1), ShipKind.SUBMARINE, PLAYER_A, 2)
    strong_threat = _ship(AxialCoord(4, 0), ShipKind.BATTLESHIP, PLAYER_B, 3)
    gs = _game_state(board, [weak, sub, strong_threat], config=Config(fow=FowConfig(enabled=False)))

    _run(NaivePolicy(), gs, PLAYER_A)

    # Outmatched -- the submarine (preferred picket) should have moved
    # toward blocking the threat's route in and submerged for stealth
    # (see _maybe_toggle_submarine, run for it in the port-defense pass
    # same as the main loop would for any other submarine); it, not the
    # patrol boat, is the one thrown at the losing fight.
    assert gs.ships[2].position != AxialCoord(1, 1)
    assert gs.ships[2].surfaced is False
    assert 1 in gs.ships and gs.ships[1].current_hp == 2  # patrol boat untouched by combat
