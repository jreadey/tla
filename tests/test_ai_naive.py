from dataclasses import replace

from tla import movement
from tla.ai import scoring
from tla.ai.enemy_model import EnemyModel
from tla.ai.policy import (
    NaivePolicy,
    Strategy,
    _carrier_bonus_cohesion_destination,
    _carrier_cautious_max_steps,
    _carrier_dangerous_threat_hexes,
    _carrier_formation_destination,
    _carrier_reaching_threat,
    _carrier_defense_destination,
    _carrier_screen_destination,
    _carrier_scouting_advance,
    _carrier_scouting_eligible,
    _choose_carrier_destination,
    _choose_destination,
    _cohesion_destination,
    _compute_force_pace,
    _favorable_attack,
    _make_way_for_scout,
    _maybe_toggle_submarine,
    _pick_strategy_for_force,
    _port_defense_destination,
    _project_engagement_value,
    _rearguard_target,
    _reevaluate_strategy_on_new_sightings,
    _scouting_path_blocker,
    _step_toward,
    _weighted_damage,
    choose_task_force_destination,
)
from tla.ai.global_strategy import Posture
from tla.ai.task_force import (
    CarrierDefenseDirective,
    GoalKind,
    PortDefenseDirective,
    TaskForce,
    TaskForceGoal,
    compute_carrier_defense_directives,
    compute_port_defense_directives,
    enemy_reachable_next_turn,
)
from tla.board import Board
from tla.config import AiConfig, Config, FleetConfig, FowConfig
from tla.game_state import GameState
from tla.hexgrid import AxialCoord, distance, hexes_in_range, neighbors
from tla.ship import Ship, ShipKind
from tla.tile import PLAYER_A, PLAYER_B, Tile, TerrainType
from tla.turn_manager import start_movement_phase


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


def _empty_enemy_model(gs: GameState) -> EnemyModel:
    """An `EnemyModel` seeded with no starting-fleet belief at all -- for
    `_project_engagement_value`/`_pick_strategy_for_force` tests that
    predate probable-threat scoring and shouldn't pick up spurious
    `KindPool` mass from `gs.config`'s (usually default, nonzero)
    `FleetConfig` -- a naively-constructed `EnemyModel(gs, ...)` would
    otherwise seed real believed mass across the board and could silently
    change one of these tests' assertions now that
    `AiConfig.probable_threat_engagement_enabled` defaults `True`."""
    empty_gs = replace(gs, config=replace(gs.config, fleet=FleetConfig(counts={})))
    return EnemyModel(empty_gs, PLAYER_A, PLAYER_B)


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


def test_ai_presses_a_winning_fight_to_the_kill_despite_ending_very_low():
    # Battleship (dmg4, max hp12) starting at hp7 vs a full-hp cruiser
    # (dmg3, hp8) -- a clearly favorable race (matchup_score == +1: the
    # attacker outlasts the defender by exactly one round), but round 1
    # alone (4 dmg dealt, 3 taken) already drops the attacker to 4/12 --
    # under the old, removed damaged_withdraw_fraction (0.34) floor this
    # would have retreated right there with the defender still alive.
    # decide_battle no longer has that floor: it presses on and secures
    # the kill, ending at 1 HP.
    board = _sea_board()
    attacker = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1, hp=7)
    defender = _ship(AxialCoord(1, 0), ShipKind.CRUISER, PLAYER_B, 2)
    gs = _game_state(board, [attacker, defender], config=Config(fow=FowConfig(enabled=False)))

    _run(NaivePolicy(), gs, PLAYER_A)

    assert 2 not in gs.ships  # the cruiser was sunk
    assert 1 in gs.ships and gs.ships[1].current_hp == 1  # attacker survived, badly hurt


def test_decide_battle_stays_in_a_winning_fight_despite_low_hp():
    attacker = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1, hp=2)  # 17% of max -- very low
    defender = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2, hp=2)
    board = _sea_board()
    gs = _game_state(board, [attacker, defender], config=Config(fow=FowConfig(enabled=False)))
    assert scoring.matchup_score(attacker, defender, gs) > 0  # confirm this really is a clean win

    assert NaivePolicy().decide_battle(attacker, defender, gs) == "stay"


def test_decide_battle_retreats_once_the_race_turns_unfavorable():
    attacker = _ship(AxialCoord(0, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)
    defender = _ship(AxialCoord(1, 0), ShipKind.BATTLESHIP, PLAYER_B, 2)
    board = _sea_board()
    gs = _game_state(board, [attacker, defender], config=Config(fow=FowConfig(enabled=False)))

    assert NaivePolicy().decide_battle(attacker, defender, gs) == "retreat"


def test_decide_battle_retreats_on_a_non_worthwhile_tie():
    # Destroyer (cost 4) vs patrol boat (cost 1) at HP levels that make it
    # an exact race tie -- trading a cost-4 destroyer for a cost-1 patrol
    # boat isn't worth it even at even odds.
    attacker = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1, hp=2)
    defender = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2, hp=4)
    board = _sea_board()
    gs = _game_state(board, [attacker, defender], config=Config(fow=FowConfig(enabled=False)))
    assert scoring.matchup_score(attacker, defender, gs) == 0  # confirm this really is a tie

    assert NaivePolicy().decide_battle(attacker, defender, gs) == "retreat"


def test_decide_battle_stays_on_a_worthwhile_tie():
    # Cruiser (cost 7) vs battleship (cost 10) at HP levels that make it an
    # exact tie -- worth taking even at 1-for-1.
    attacker = _ship(AxialCoord(0, 0), ShipKind.CRUISER, PLAYER_A, 1, hp=4)
    defender = _ship(AxialCoord(1, 0), ShipKind.BATTLESHIP, PLAYER_B, 2, hp=3)
    board = _sea_board()
    gs = _game_state(board, [attacker, defender], config=Config(fow=FowConfig(enabled=False)))
    assert scoring.matchup_score(attacker, defender, gs) == 0  # confirm this really is a tie

    assert NaivePolicy().decide_battle(attacker, defender, gs) == "stay"


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
    config = Config(fow=FowConfig(enabled=False))
    carrier_start, escort_start, enemy_start = AxialCoord(0, 0), AxialCoord(-3, 0), AxialCoord(2, 0)
    carrier = _ship(carrier_start, ShipKind.CARRIER, PLAYER_A, 1)
    escort = _ship(escort_start, ShipKind.DESTROYER, PLAYER_A, 2)
    # A dangerous kind (_DANGEROUS_TO_CARRIER_KINDS) within its own
    # movement (4) of the carrier's hex -- a destroyer alone wouldn't
    # trigger this, see test_carrier_ignores_a_destroyer_nearby.
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
    config = Config(fow=FowConfig(enabled=False))
    carrier_start = AxialCoord(0, 0)
    carrier = _ship(carrier_start, ShipKind.CARRIER, PLAYER_A, 1)
    # Well within the destroyer's own reach, but not a kind the carrier's
    # own threat-avoidance worries about -- see _DANGEROUS_TO_CARRIER_KINDS.
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


def test_task_force_ignores_a_distant_force_mate_when_judging_backup():
    # Real bug (game64 turn 5): attacker_group used to be the ship's whole
    # task force by membership alone, regardless of distance -- so a
    # teammate racing to catch up from far away (see task_force_max_
    # separation's own straggler doctrine) counted its full strength as
    # "available support" for a fight breaking out right now elsewhere.
    # Same shape as the test above (an easy-looking 1v1 with real backup),
    # but the force now has a second, distant member strong enough to flip
    # group_power's own verdict from outmatched to not-outmatched if
    # wrongly included -- confirmed by construction: destroyer alone is
    # (6, 2) vs backup+bait's (102, 5) -- outmatched; destroyer plus this
    # teammate is (106, 6) -- not outmatched. The attack must still be
    # declined: the teammate is 10 hexes away, well past task_force_
    # threat_radius (4), so it isn't real backup for this fight.
    board = _sea_board()
    destroyer = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    bait = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    backup = _ship(AxialCoord(2, 0), ShipKind.BATTLESHIP, PLAYER_B, 3, hp=100)
    distant_teammate = _ship(AxialCoord(10, 0), ShipKind.BATTLESHIP, PLAYER_A, 4, hp=100)
    port = AxialCoord(6, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    config = Config(
        fow=FowConfig(enabled=False), ai=AiConfig(task_force_threat_radius=4, task_force_outnumbered_margin=0)
    )
    gs = _game_state(board, [destroyer, bait, backup, distant_teammate], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 4})
    goal = TaskForceGoal(kind=GoalKind.CAPTURE_PORT, target=port)

    destination = choose_task_force_destination(destroyer, goal, gs, {2: bait, 3: backup}, force)

    assert destination != bait.position


def test_task_force_counts_a_force_mate_once_it_is_actually_close_enough():
    # Same setup as above, but the teammate is now within task_force_
    # threat_radius (4) of the fight -- real, timely backup, not a
    # straggler. It should be counted, tipping the fight back to "safe."
    board = _sea_board()
    destroyer = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    bait = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    backup = _ship(AxialCoord(2, 0), ShipKind.BATTLESHIP, PLAYER_B, 3, hp=100)
    close_teammate = _ship(AxialCoord(1, 3), ShipKind.BATTLESHIP, PLAYER_A, 4, hp=100)  # 3 hexes from bait
    port = AxialCoord(6, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    config = Config(
        fow=FowConfig(enabled=False), ai=AiConfig(task_force_threat_radius=4, task_force_outnumbered_margin=0)
    )
    gs = _game_state(board, [destroyer, bait, backup, close_teammate], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 4})
    goal = TaskForceGoal(kind=GoalKind.CAPTURE_PORT, target=port)

    destination = choose_task_force_destination(destroyer, goal, gs, {2: bait, 3: backup}, force)

    assert destination == bait.position


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


def test_compute_force_pace_a_submerged_submarine_drags_the_whole_force_down():
    # The user's own tested doctrine: once a submarine submerges (a
    # crawl), the whole force should deliberately slow to stay with it,
    # not leave it behind.
    board = _sea_board()
    fast = _ship(AxialCoord(0, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)  # movement 6
    slow = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_A, 2)  # movement 3
    sub = _ship(AxialCoord(2, 0), ShipKind.SUBMARINE, PLAYER_A, 3)
    sub.movement_remaining = 1  # submerged
    gs = _game_state(board, [fast, slow, sub])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2, 3})

    pace = _compute_force_pace(gs, [force])

    assert pace[1] == 1  # the submerged submarine's crawl now sets the pace


def test_compute_force_pace_a_surfaced_submarine_rarely_bottlenecks_the_group():
    board = _sea_board()
    slow = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_A, 2)  # movement 3
    sub = _ship(AxialCoord(2, 0), ShipKind.SUBMARINE, PLAYER_A, 3)  # surfaced, movement 3
    gs = _game_state(board, [slow, sub])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={2, 3})

    pace = _compute_force_pace(gs, [force])

    assert pace[1] == 3  # surfaced speeds are comparable -- no artificial slowdown


def test_compute_force_pace_ignores_submarines_when_cohesion_disabled():
    board = _sea_board()
    slow = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_A, 2)  # movement 3
    sub = _ship(AxialCoord(2, 0), ShipKind.SUBMARINE, PLAYER_A, 3)
    sub.movement_remaining = 1  # submerged
    config = Config(ai=AiConfig(submarine_task_force_cohesion=False))
    gs = _game_state(board, [slow, sub], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={2, 3})

    pace = _compute_force_pace(gs, [force])

    assert pace[1] == 3  # escape hatch -- the submarine no longer sets the pace


def test_compute_force_pace_gives_an_all_submarine_force_its_own_pace():
    board = _sea_board()
    sub = _ship(AxialCoord(0, 0), ShipKind.SUBMARINE, PLAYER_A, 1)
    gs = _game_state(board, [sub])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})

    pace = _compute_force_pace(gs, [force])

    assert pace[1] == sub.movement_remaining  # no longer specially omitted


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


def test_submarines_own_advance_is_now_capped_by_pace_too():
    board = _sea_board(radius=10)
    sub = _ship(AxialCoord(0, 0), ShipKind.SUBMARINE, PLAYER_A, 1)  # surfaced, movement 3
    slow = _ship(AxialCoord(0, 1), ShipKind.DESTROYER, PLAYER_A, 2)
    slow.movement_remaining = 1  # a slow companion sets the pace
    port = AxialCoord(8, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    gs = _game_state(board, [sub, slow])
    force = TaskForce(
        id=1, owner=PLAYER_A, member_ids={1, 2}, goal=TaskForceGoal(kind=GoalKind.CAPTURE_PORT, target=port)
    )
    pace = _compute_force_pace(gs, [force])
    assert pace[1] == 1

    destination = _choose_destination(sub, gs, PLAYER_A, {}, [force], pace)

    assert distance(AxialCoord(0, 0), destination) == 1  # capped, not the sub's own full 3


def test_submarines_own_advance_ignores_pace_when_cohesion_disabled():
    board = _sea_board(radius=10)
    sub = _ship(AxialCoord(0, 0), ShipKind.SUBMARINE, PLAYER_A, 1)  # surfaced, movement 3
    slow = _ship(AxialCoord(0, 1), ShipKind.DESTROYER, PLAYER_A, 2)
    slow.movement_remaining = 1
    port = AxialCoord(8, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    config = Config(ai=AiConfig(submarine_task_force_cohesion=False))
    gs = _game_state(board, [sub, slow], config=config)
    force = TaskForce(
        id=1, owner=PLAYER_A, member_ids={1, 2}, goal=TaskForceGoal(kind=GoalKind.CAPTURE_PORT, target=port)
    )
    pace = _compute_force_pace(gs, [force])
    assert pace[1] == 1  # still set by the non-submarine companion

    destination = _choose_destination(sub, gs, PLAYER_A, {}, [force], pace)

    assert distance(AxialCoord(0, 0), destination) == 3  # escape hatch -- the sub itself ignores it


def test_plan_movement_whole_force_crawls_with_an_embedded_submerged_submarine():
    # End-to-end proof of the user's own tested doctrine: a submerged
    # submarine embedded in its task force drags the whole force down to
    # its crawl (1 hex/turn), rather than the group leaving it behind.
    board = _sea_board(radius=15)
    destroyer = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    battleship = _ship(AxialCoord(0, 1), ShipKind.BATTLESHIP, PLAYER_A, 2)
    sub = _ship(AxialCoord(1, 0), ShipKind.SUBMARINE, PLAYER_A, 3, surfaced=False)
    sub.movement_remaining = Config().ship_stats.stats[ShipKind.SUBMARINE].movement_submerged
    # Visible (fow disabled) but far out of reach this turn -- keeps this
    # test about pacing, not triggering an actual attack.
    enemy = _ship(AxialCoord(25, 0), ShipKind.BATTLESHIP, PLAYER_B, 4)
    port = AxialCoord(12, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    gs = _game_state(board, [destroyer, battleship, sub, enemy])
    force = TaskForce(
        id=1, owner=PLAYER_A, member_ids={1, 2, 3}, goal=TaskForceGoal(kind=GoalKind.CAPTURE_PORT, target=port)
    )
    policy = NaivePolicy()
    policy._task_forces[PLAYER_A] = [force]
    policy._next_force_id = 2

    _run(policy, gs, PLAYER_A)

    assert gs.ships[3].surfaced is False  # still submerged -- the enemy is still visible
    assert distance(AxialCoord(0, 0), gs.ships[1].position) <= 1
    assert distance(AxialCoord(0, 1), gs.ships[2].position) <= 1


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


def test_cohesion_destination_now_pulls_a_straggling_submarine_back():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    sub = _ship(AxialCoord(6, 0), ShipKind.SUBMARINE, PLAYER_A, 1)  # movement 3
    teammate = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 2)  # 6 hexes away -- over the cap
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(task_force_max_separation=4))
    gs = _game_state(board, [sub, teammate], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})

    destination = _cohesion_destination(sub, force, gs, port, frozenset())

    assert destination is not None
    assert distance(destination, teammate.position) <= 4


def test_cohesion_destination_a_submarine_now_counts_as_an_anchor_too():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    destroyer = _ship(AxialCoord(6, 0), ShipKind.DESTROYER, PLAYER_A, 1)  # movement 3
    sub = _ship(AxialCoord(0, 0), ShipKind.SUBMARINE, PLAYER_A, 2)  # 6 hexes away -- over the cap
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(task_force_max_separation=4))
    gs = _game_state(board, [destroyer, sub], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})

    destination = _cohesion_destination(destroyer, force, gs, port, frozenset())

    assert destination is not None
    assert distance(destination, sub.position) <= 4


def test_cohesion_destination_submarine_exemption_restored_when_disabled():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    sub = _ship(AxialCoord(9, 0), ShipKind.SUBMARINE, PLAYER_A, 1)
    teammate = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 2)
    config = Config(
        fow=FowConfig(enabled=False),
        ai=AiConfig(task_force_max_separation=4, submarine_task_force_cohesion=False),
    )
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


def test_cohesion_destination_a_fresh_straggler_does_not_anchor_the_established_group_back():
    # The user's own reported doctrine, and a real replay-observed problem
    # (game63 turn 3): a freshly produced ship recruited into an already-
    # advanced force used to count as a full anchor from the moment it
    # joined, symmetrically pulling the *whole* advanced formation back
    # toward it -- even though it had only just spawned at a home port far
    # behind. Two established members already within range of each other
    # (a real "task force"); a third, brand-new member sits far away, at
    # roughly the force's own home port. The established pair should not
    # be pulled toward it.
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    vanguard = _ship(AxialCoord(6, 0), ShipKind.CRUISER, PLAYER_A, 1)
    escort = _ship(AxialCoord(4, 0), ShipKind.DESTROYER, PLAYER_A, 2)  # 2 hexes from vanguard -- established pair
    fresh_recruit = _ship(AxialCoord(-10, 0), ShipKind.BATTLESHIP, PLAYER_A, 3)  # just spawned, far behind
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(task_force_max_separation=4))
    gs = _game_state(board, [vanguard, escort, fresh_recruit], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2, 3})

    # Neither established member should be pulled toward the straggler --
    # None (no correction needed) since each is already within range of
    # its one established peer.
    assert _cohesion_destination(vanguard, force, gs, port, frozenset()) is None
    assert _cohesion_destination(escort, force, gs, port, frozenset()) is None


def test_cohesion_destination_a_fresh_straggler_still_steams_to_catch_up():
    # Same scenario as above, from the straggler's own side: it should
    # still be pulled toward the established pair (matching the user's
    # "steams at max speed to join the TF" half of the doctrine) -- only
    # the reverse direction (established anchoring back onto it) is what
    # changed.
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    vanguard = _ship(AxialCoord(6, 0), ShipKind.CRUISER, PLAYER_A, 1)
    escort = _ship(AxialCoord(4, 0), ShipKind.DESTROYER, PLAYER_A, 2)
    fresh_recruit = _ship(AxialCoord(-10, 0), ShipKind.BATTLESHIP, PLAYER_A, 3)
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(task_force_max_separation=4))
    gs = _game_state(board, [vanguard, escort, fresh_recruit], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2, 3})

    destination = _cohesion_destination(fresh_recruit, force, gs, port, frozenset())

    assert destination is not None
    assert distance(destination, fresh_recruit.position) > 0  # actually moved toward the pair


def test_cohesion_destination_a_recruit_becomes_a_full_anchor_once_it_catches_up():
    # Once the straggler has actually closed to within task_force_max_
    # separation of the established group, it's no longer a special case
    # -- it becomes a normal anchor like anyone else, same as before this
    # fix for a force where everyone's already close together.
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    vanguard = _ship(AxialCoord(6, 0), ShipKind.CRUISER, PLAYER_A, 1)
    escort = _ship(AxialCoord(4, 0), ShipKind.DESTROYER, PLAYER_A, 2)
    caught_up_recruit = _ship(AxialCoord(1, 0), ShipKind.BATTLESHIP, PLAYER_A, 3)  # within 4 of escort now
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(task_force_max_separation=4))
    gs = _game_state(board, [vanguard, escort, caught_up_recruit], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2, 3})

    # vanguard (6,0) is 5 hexes from caught_up_recruit (1,0) -- now over
    # the cap against its newly-established peer, so it should correct.
    destination = _cohesion_destination(vanguard, force, gs, port, frozenset())

    assert destination is not None
    assert distance(destination, caught_up_recruit.position) <= 4


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


def test_carrier_advance_steers_around_a_visible_enemys_reach():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)  # movement 4
    # Distance 5 from the carrier -- outside the cruiser's own movement (4),
    # so the carrier's immediate-threat check never fires at its *current*
    # hex; this is about the *destination* choice, not that reactive safety
    # net. A dangerous kind (_DANGEROUS_TO_CARRIER_KINDS), since a destroyer
    # wouldn't be avoided at all.
    enemy = _ship(AxialCoord(5, 0), ShipKind.CRUISER, PLAYER_B, 2)
    # carrier_advance_reserve zeroed to isolate danger-zone avoidance from
    # the separate cautious-advance cap (see the dedicated reserve tests).
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(carrier_advance_reserve=0))
    gs = _game_state(board, [carrier, enemy], config=config)
    goal = TaskForceGoal(GoalKind.CAPTURE_PORT, port)

    destination = _choose_carrier_destination(carrier, gs, PLAYER_A, {2: enemy}, goal=goal)

    # A plain step straight toward the port would land on (4, 0), well
    # within the cruiser's own reach from (5, 0).
    assert destination is not None
    assert destination not in enemy_reachable_next_turn(enemy, gs)


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


# -- carrier formation optimization (see _carrier_formation_destination) -----
# FowConfig(enabled=True) throughout, matching carrier scouting below --
# escort reachability/screening genuinely depends on real vision.


def test_carrier_formation_prefers_more_covered_hex_over_fewer():
    board = _sea_board(radius=20)
    port = AxialCoord(20, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    # Both already "resolved" (real, fixed positions) -- a hex near (1, 0)
    # is within ac_bonus_radius (2) of both; hexes elsewhere reach at most one.
    bb1 = _ship(AxialCoord(2, 1), ShipKind.BATTLESHIP, PLAYER_A, 2)
    bb2 = _ship(AxialCoord(2, -1), ShipKind.BATTLESHIP, PLAYER_A, 3)
    gs = _game_state(board, [carrier, bb1, bb2], config=Config(fow=FowConfig(enabled=True)))
    goal = TaskForceGoal(GoalKind.CAPTURE_PORT, port)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2, 3}, goal=goal)
    resolved = frozenset({2, 3})
    cautious_steps = _carrier_cautious_max_steps(carrier, gs, max_steps=4)

    dest = _carrier_formation_destination(
        carrier, force, gs, PLAYER_A, {}, goal, resolved, 4, Strategy.ADVANCE, cautious_steps, frozenset(), frozenset()
    )

    radius = gs.config.combat.ac_bonus_radius
    assert dest is not None
    assert distance(dest, bb1.position) <= radius
    assert distance(dest, bb2.position) <= radius


def test_carrier_formation_uses_escorts_honestly_projected_positions():
    board = _sea_board(radius=20)
    port = AxialCoord(20, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    # Not resolved -- its own ordinary goal-directed dry run moves it from
    # its current position (west of the carrier) toward the goal (east),
    # landing on the *opposite* side of the carrier from where it started.
    bb1 = _ship(AxialCoord(-3, 0), ShipKind.BATTLESHIP, PLAYER_A, 2)
    gs = _game_state(board, [carrier, bb1], config=Config(fow=FowConfig(enabled=True)))
    goal = TaskForceGoal(GoalKind.CAPTURE_PORT, port)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2}, goal=goal)
    cautious_steps = _carrier_cautious_max_steps(carrier, gs, max_steps=4)

    projected = choose_task_force_destination(bb1, goal, gs, {}, force, max_steps=4, strategy=Strategy.ADVANCE)
    dest = _carrier_formation_destination(
        carrier, force, gs, PLAYER_A, {}, goal, frozenset(), 4, Strategy.ADVANCE, cautious_steps, frozenset(), frozenset()
    )

    radius = gs.config.combat.ac_bonus_radius
    assert dest is not None
    assert distance(dest, projected) <= radius  # covers where the escort WILL be
    assert distance(dest, bb1.position) > radius  # not where it currently sits


def test_carrier_formation_already_resolved_escort_uses_its_real_position():
    board = _sea_board(radius=20)
    port = AxialCoord(20, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    bb1 = _ship(AxialCoord(-3, 0), ShipKind.BATTLESHIP, PLAYER_A, 2)
    gs = _game_state(board, [carrier, bb1], config=Config(fow=FowConfig(enabled=True)))
    goal = TaskForceGoal(GoalKind.CAPTURE_PORT, port)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2}, goal=goal)
    cautious_steps = _carrier_cautious_max_steps(carrier, gs, max_steps=4)

    # Marked resolved -- already moved for real this turn (e.g. via an
    # earlier pass). Its own ordinary dry run would send it east, but that
    # move will never actually happen -- coverage must be scored against
    # its real, current (unresolved-dry-run-ignoring) position instead.
    dest = _carrier_formation_destination(
        carrier, force, gs, PLAYER_A, {}, goal, frozenset({2}), 4, Strategy.ADVANCE, cautious_steps, frozenset(), frozenset()
    )

    radius = gs.config.combat.ac_bonus_radius
    assert dest is not None
    assert distance(dest, bb1.position) <= radius


def test_carrier_formation_prefers_screened_hex_over_equally_covered_exposed_hex():
    board = _sea_board(radius=20)
    port = AxialCoord(20, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    bb1 = _ship(AxialCoord(0, 1), ShipKind.BATTLESHIP, PLAYER_A, 2)
    # A cruiser (DANGEROUS_TO_CARRIER_KINDS) that can reach (1, -1) and
    # (0, -1) -- both coverage-tied candidates of bb1 -- but not (-1, 0),
    # an equally coverage-tied candidate on the far side.
    enemy = _ship(AxialCoord(4, -4), ShipKind.CRUISER, PLAYER_B, 3)
    gs = _game_state(board, [carrier, bb1, enemy], config=Config(fow=FowConfig(enabled=True)))
    goal = TaskForceGoal(GoalKind.CAPTURE_PORT, port)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2}, goal=goal)
    cautious_steps = _carrier_cautious_max_steps(carrier, gs, max_steps=4)

    dest = _carrier_formation_destination(
        carrier, force, gs, PLAYER_A, {3: enemy}, goal, frozenset({2}), 4, Strategy.ADVANCE, cautious_steps,
        frozenset(), frozenset(),
    )

    assert dest == AxialCoord(-1, 0)


def test_carrier_formation_falls_back_gracefully_when_nothing_is_screened():
    board = _sea_board(radius=20)
    port = AxialCoord(20, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    bb1 = _ship(AxialCoord(0, 1), ShipKind.BATTLESHIP, PLAYER_A, 2)
    # Close enough to reach every coverage-tied candidate around bb1 --
    # nothing this turn is actually screened.
    enemy = _ship(AxialCoord(-2, -2), ShipKind.CRUISER, PLAYER_B, 3)
    gs = _game_state(board, [carrier, bb1, enemy], config=Config(fow=FowConfig(enabled=True)))
    goal = TaskForceGoal(GoalKind.CAPTURE_PORT, port)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2}, goal=goal)
    cautious_steps = _carrier_cautious_max_steps(carrier, gs, max_steps=4)

    dest = _carrier_formation_destination(
        carrier, force, gs, PLAYER_A, {3: enemy}, goal, frozenset({2}), 4, Strategy.ADVANCE, cautious_steps,
        frozenset(), frozenset(),
    )

    # Never strands the ship -- still returns a real, coverage-maximizing
    # candidate rather than None or a coverage-losing retreat in place.
    radius = gs.config.combat.ac_bonus_radius
    assert dest is not None
    assert distance(dest, bb1.position) <= radius


def test_carrier_formation_strategy_tiebreak_advance_vs_hold():
    board = _sea_board(radius=20)
    port = AxialCoord(20, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    bb1 = _ship(AxialCoord(0, 1), ShipKind.BATTLESHIP, PLAYER_A, 2)
    gs = _game_state(board, [carrier, bb1], config=Config(fow=FowConfig(enabled=True)))
    goal = TaskForceGoal(GoalKind.CAPTURE_PORT, port)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2}, goal=goal)
    resolved = frozenset({2})

    advance_steps = _carrier_cautious_max_steps(carrier, gs, max_steps=4)
    advance_dest = _carrier_formation_destination(
        carrier, force, gs, PLAYER_A, {}, goal, resolved, 4, Strategy.ADVANCE, advance_steps, frozenset(), frozenset()
    )
    # ADVANCE breaks the coverage tie toward the goal (east) over the ship's
    # own starting position.
    assert advance_dest is not None
    assert advance_dest != carrier.position

    hold_steps = _carrier_cautious_max_steps(carrier, gs, max_steps=0)
    assert hold_steps == 0
    hold_dest = _carrier_formation_destination(
        carrier, force, gs, PLAYER_A, {}, goal, resolved, 0, Strategy.HOLD, hold_steps, frozenset(), frozenset()
    )
    # HOLD's max_steps=0 floors cautious_steps at 0, which structurally
    # collapses the candidate set to just the ship's own current position.
    assert hold_dest == carrier.position


def test_carrier_formation_returns_none_without_a_living_ac_bonus_eligible_escort():
    board = _sea_board(radius=20)
    port = AxialCoord(20, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    sub = _ship(AxialCoord(0, 1), ShipKind.SUBMARINE, PLAYER_A, 2)  # not AC_BONUS_ELIGIBLE_KINDS
    gs = _game_state(board, [carrier, sub], config=Config(fow=FowConfig(enabled=True)))
    goal = TaskForceGoal(GoalKind.CAPTURE_PORT, port)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2}, goal=goal)
    cautious_steps = _carrier_cautious_max_steps(carrier, gs, max_steps=4)

    dest = _carrier_formation_destination(
        carrier, force, gs, PLAYER_A, {}, goal, frozenset(), 4, Strategy.ADVANCE, cautious_steps, frozenset(), frozenset()
    )

    assert dest is None


def test_carrier_formation_disabled_reproduces_the_flat_reserve_pipeline():
    # Regression pin: with the flag off (the default), _choose_carrier_
    # destination's own output is completely unaffected by anything above
    # -- a force whose formation optimization would clearly prefer a
    # different hex still gets the plain cautious advance.
    board = _sea_board(radius=20)
    port = AxialCoord(20, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    bb1 = _ship(AxialCoord(2, 1), ShipKind.BATTLESHIP, PLAYER_A, 2)
    bb2 = _ship(AxialCoord(2, -1), ShipKind.BATTLESHIP, PLAYER_A, 3)
    config = Config(
        fow=FowConfig(enabled=True), ai=AiConfig(carrier_advance_reserve=2, carrier_formation_optimization_enabled=False)
    )
    gs = _game_state(board, [carrier, bb1, bb2], config=config)
    goal = TaskForceGoal(GoalKind.CAPTURE_PORT, port)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2, 3}, goal=goal)

    destination = _choose_carrier_destination(carrier, gs, PLAYER_A, {}, goal=goal, force=force)

    # Plain cautious advance toward the goal (east), same as _choose_
    # carrier_destination's own existing reserve test -- not the
    # formation-optimized pick near (1, 0) the tests above show it would
    # otherwise prefer.
    assert destination is not None
    assert distance(AxialCoord(0, 0), destination) == 2


def test_carrier_formation_ignored_by_a_scouting_eligible_carrier():
    # A carrier eligible for hex-by-hex scouting is claimed by that
    # earlier pass and never reaches _choose_carrier_destination's plain-
    # advance branch at all -- confirm the eligibility check itself
    # doesn't even look at carrier_formation_optimization_enabled, so
    # turning this flag on can't change whether a carrier scouts.
    board = _sea_board(radius=20)
    port = AxialCoord(20, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    bb1 = _ship(AxialCoord(2, 1), ShipKind.BATTLESHIP, PLAYER_A, 2)
    config = Config(
        fow=FowConfig(enabled=True),
        ai=AiConfig(carrier_scouting_enabled=True, carrier_formation_optimization_enabled=True),
    )
    gs = _game_state(board, [carrier, bb1], config=config)
    goal = TaskForceGoal(GoalKind.CAPTURE_PORT, port)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2}, goal=goal)

    assert _carrier_scouting_eligible(carrier, gs, force, Strategy.ADVANCE, {}) is True


# -- carrier scouting advance (see _carrier_scouting_advance) ----------------
# FowConfig(enabled=True) throughout -- vision genuinely moving with the
# ship as it hops is the entire point, unlike most tests above.


def test_carrier_scouting_advances_pace_minus_retreat_reserve_when_clear():
    board = _sea_board(radius=20)
    port = AxialCoord(20, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)  # movement 4
    gs = _game_state(board, [carrier], config=Config(fow=FowConfig(enabled=True)))
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, port))
    pace = _compute_force_pace(gs, [force])  # 4, unpaced (sole member)
    model = _empty_enemy_model(gs)

    _carrier_scouting_advance(carrier, force, gs, PLAYER_A, model, pace, lambda a, d, g: "stay")

    # Default carrier_scout_retreat_reserve=1 -- 4 - 1 = 3, not the flat
    # cautious-advance path's 1 (carrier_advance_reserve=3 on movement 4).
    assert distance(AxialCoord(0, 0), carrier.position) == 3


def test_carrier_scouting_retreats_exactly_one_hop_and_preserves_the_sighting():
    board = _sea_board(radius=20)
    port = AxialCoord(20, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    # Outside FowConfig.port_and_carrier_visibility_radius (4) from hop 1
    # (1, 0) -- distance 5 -- but inside it, and within its own movement
    # (4), from hop 2 (2, 0) -- distance 4: invisible and unreachable
    # until the carrier is actually there to see it.
    enemy = _ship(AxialCoord(2, 4), ShipKind.CRUISER, PLAYER_B, 2)
    gs = _game_state(board, [carrier, enemy], config=Config(fow=FowConfig(enabled=True)))
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, port))
    pace = _compute_force_pace(gs, [force])
    model = _empty_enemy_model(gs)

    _carrier_scouting_advance(carrier, force, gs, PLAYER_A, model, pace, lambda a, d, g: "stay")

    assert carrier.position == AxialCoord(1, 0)  # hop 1, not hop 2, not the start
    # The core guarantee: the sighting at hop 2 survives even though the
    # carrier itself retreated away from it.
    tracked = model.tracked_ships()
    assert 2 in tracked
    assert tracked[2].last_seen_position == AxialCoord(2, 4)


def test_carrier_scouting_does_not_strand_itself_at_the_movement_edge():
    board = _sea_board(radius=20)
    port = AxialCoord(20, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    # Exactly carrier_scout_retreat_reserve (1) + 1 movement left -- just
    # enough for one hop and (if needed) one retreat, no more.
    carrier = replace(_ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1), movement_remaining=2)
    # Invisible/unreachable from the start (0, 0) (distance 5, outside
    # radius 4), but visible and reachable from hop 1 (1, 0) (distance 4).
    enemy = _ship(AxialCoord(1, 4), ShipKind.CRUISER, PLAYER_B, 2)
    gs = _game_state(board, [carrier, enemy], config=Config(fow=FowConfig(enabled=True)))
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, port))
    pace = _compute_force_pace(gs, [force])
    model = _empty_enemy_model(gs)

    _carrier_scouting_advance(carrier, force, gs, PLAYER_A, model, pace, lambda a, d, g: "stay")

    assert carrier.position == AxialCoord(0, 0)  # successfully retreated all the way back
    assert carrier.movement_remaining == 0  # spent exactly what it had, never raised/got stuck


def test_carrier_scouting_eligible_false_when_cohesion_correction_needed():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    carrier = _ship(AxialCoord(6, 0), ShipKind.CARRIER, PLAYER_A, 1)
    straggler = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 2)
    config = Config(fow=FowConfig(enabled=True), ai=AiConfig(task_force_max_separation=4))
    gs = _game_state(board, [carrier, straggler], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, port))

    assert _carrier_scouting_eligible(carrier, gs, force, Strategy.ADVANCE, {}) is False
    # Left unclaimed, the ordinary pipeline still recalls it for cohesion.
    destination = _choose_carrier_destination(carrier, gs, PLAYER_A, {}, goal=force.goal, force=force)
    assert distance(destination, straggler.position) <= 4


def test_carrier_scouting_eligible_false_when_already_threatened():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    enemy = _ship(AxialCoord(2, 0), ShipKind.BATTLESHIP, PLAYER_B, 2)  # within its own movement (4) of carrier
    gs = _game_state(board, [carrier, enemy], config=Config(fow=FowConfig(enabled=True)))
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, port))

    assert _carrier_scouting_eligible(carrier, gs, force, Strategy.ADVANCE, {2: enemy}) is False


def test_carrier_scouting_eligible_false_without_a_real_goal():
    board = _sea_board(radius=12)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    gs = _game_state(board, [carrier], config=Config(fow=FowConfig(enabled=True)))

    assert _carrier_scouting_eligible(carrier, gs, None, Strategy.ADVANCE, {}) is False
    force_no_goal = TaskForce(id=1, owner=PLAYER_A, member_ids={1})
    assert _carrier_scouting_eligible(carrier, gs, force_no_goal, Strategy.ADVANCE, {}) is False


def test_carrier_reaching_threat_extraction_matches_prior_inline_behavior():
    board = _sea_board(radius=12)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    reaching = _ship(AxialCoord(2, 0), ShipKind.BATTLESHIP, PLAYER_B, 2)  # reaches (movement 4)
    too_far = _ship(AxialCoord(10, 0), ShipKind.BATTLESHIP, PLAYER_B, 3)  # doesn't reach
    harmless_kind = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 4)  # not DANGEROUS_TO_CARRIER_KINDS
    gs = _game_state(board, [carrier, reaching, too_far, harmless_kind], config=Config(fow=FowConfig(enabled=False)))
    visible = {2: reaching, 3: too_far, 4: harmless_kind}

    assert _carrier_reaching_threat(carrier, gs, visible) is reaching
    threat_hexes = _carrier_dangerous_threat_hexes(carrier, gs, visible)
    assert set(threat_hexes.keys()) == {2, 3}  # dangerous kinds only, reachable or not


def test_scouting_path_blocker_steps_aside_and_still_takes_its_own_turn():
    board = _sea_board(radius=20)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    blocker = _ship(AxialCoord(1, 0), ShipKind.BATTLESHIP, PLAYER_A, 2)  # movement 4, directly in the way
    gs = _game_state(board, [carrier, blocker], config=Config(fow=FowConfig(enabled=True)))
    target = AxialCoord(20, 0)

    assert _scouting_path_blocker(carrier, gs, target) is blocker

    _make_way_for_scout(blocker, gs, {}, lambda a, d, g: "stay")

    assert blocker.position != AxialCoord(1, 0)  # moved off the ideal hex
    assert blocker.movement_remaining == 3  # spent exactly 1 of its own 4 -- still has a real turn left
    assert _scouting_path_blocker(carrier, gs, target) is None  # the ideal hex is free now


def test_make_way_for_scout_declines_when_no_safe_hex_exists():
    board = _sea_board(radius=12)
    blocker = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    # One cruiser 2 hexes out in each of the 6 directions -- each reaches
    # (movement 4, only 1 hex needed) the one neighbor hex nearest it,
    # without needing to path *through* blocker's own occupied hex (which
    # would block it -- an enemy-occupied hex is a legal terminal, not a
    # pass-through, see movement.reachable_hexes). Together every one of
    # blocker's 6 neighbors is covered -- nowhere safe to step.
    enemies = [
        _ship(AxialCoord(d.q * 2, d.r * 2), ShipKind.CRUISER, PLAYER_B, 2 + i)
        for i, d in enumerate(neighbors(AxialCoord(0, 0)))
    ]
    gs = _game_state(board, [blocker, *enemies], config=Config(fow=FowConfig(enabled=True)))
    visible = {e.id: e for e in enemies}

    _make_way_for_scout(blocker, gs, visible, lambda a, d, g: "stay")

    assert blocker.position == AxialCoord(0, 0)  # declined -- never sacrifices safety to clear a path
    assert blocker.movement_remaining == Config().ship_stats.stats[ShipKind.BATTLESHIP].movement  # untouched


def test_scouting_path_blocker_ignores_a_ship_that_already_moved_this_turn():
    board = _sea_board(radius=20)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    already_moved = replace(
        _ship(AxialCoord(1, 0), ShipKind.BATTLESHIP, PLAYER_A, 2), movement_remaining=3
    )  # spent 1 already
    gs = _game_state(board, [carrier, already_moved], config=Config(fow=FowConfig(enabled=True)))

    assert _scouting_path_blocker(carrier, gs, AxialCoord(20, 0)) is None


def test_carrier_scouting_disabled_reproduces_the_flat_reserve_pipeline():
    # Regression pin: _choose_carrier_destination's own behavior (the
    # pipeline every carrier still goes through unless plan_movement's
    # separate scouting pass claims it first) is completely unmodified by
    # any of this, regardless of AiConfig.carrier_scouting_enabled --
    # this function itself never reads that flag, only the new pass does
    # -- see test_carrier_advance_toward_a_goal_holds_back_the_reserve
    # above for the direct 1-hex-of-4 pin this mirrors at the
    # plan_movement level. Explicitly disabled here regardless of the
    # dataclass default (False -- see that field's own docstring: safe
    # only paired with a real task_force_max_separation, which this
    # single-carrier scenario has no force to set) to isolate this from
    # that separate concern entirely.
    board = _sea_board(radius=10)
    port = AxialCoord(8, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    config = Config(
        fow=FowConfig(enabled=False), ai=AiConfig(carrier_advance_reserve=2, carrier_scouting_enabled=False)
    )
    gs = _game_state(board, [carrier], config=config)
    goal = TaskForceGoal(GoalKind.CAPTURE_PORT, port)

    destination = _choose_carrier_destination(carrier, gs, PLAYER_A, {}, goal=goal)

    assert destination is not None
    assert distance(AxialCoord(0, 0), destination) == 2


# -- carrier screening (see _carrier_screen_destination) ---------------------
# FowConfig(enabled=False) throughout so visible_enemies can be handed in
# directly rather than needing real vision.


def test_carrier_screen_is_none_when_disabled():
    board = _sea_board(radius=12)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    own_bb = _ship(AxialCoord(0, 1), ShipKind.BATTLESHIP, PLAYER_A, 2)
    threat = _ship(AxialCoord(3, 0), ShipKind.BATTLESHIP, PLAYER_B, 3)
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(carrier_screen_enabled=False))
    gs = _game_state(board, [carrier, own_bb, threat], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})

    assert _carrier_screen_destination(own_bb, force, gs, frozenset(), {3: threat}) is None


def test_carrier_screen_is_none_without_a_living_carrier():
    board = _sea_board(radius=12)
    own_bb = _ship(AxialCoord(0, 1), ShipKind.BATTLESHIP, PLAYER_A, 2)
    threat = _ship(AxialCoord(3, 0), ShipKind.BATTLESHIP, PLAYER_B, 3)
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(carrier_screen_enabled=True))
    gs = _game_state(board, [own_bb, threat], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={2})

    assert _carrier_screen_destination(own_bb, force, gs, frozenset(), {3: threat}) is None


def test_carrier_screen_is_none_for_a_kind_that_does_not_screen():
    # A patrol boat is neither a heavy nor a light screener -- always None,
    # regardless of what's nearby.
    board = _sea_board(radius=12)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    patrol = _ship(AxialCoord(0, 1), ShipKind.PATROL_BOAT, PLAYER_A, 2)
    threat = _ship(AxialCoord(3, 0), ShipKind.BATTLESHIP, PLAYER_B, 3)
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(carrier_screen_enabled=True))
    gs = _game_state(board, [carrier, patrol, threat], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})

    assert _carrier_screen_destination(patrol, force, gs, frozenset(), {3: threat}) is None


def test_carrier_screen_is_none_for_a_destroyer_against_a_heavy_threat():
    # Kind-matched only -- a destroyer never screens a battleship/cruiser
    # threat, even though it's an eligible *light* screener in general.
    board = _sea_board(radius=12)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    own_dd = _ship(AxialCoord(0, 1), ShipKind.DESTROYER, PLAYER_A, 2)
    threat = _ship(AxialCoord(3, 0), ShipKind.BATTLESHIP, PLAYER_B, 3)
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(carrier_screen_enabled=True))
    gs = _game_state(board, [carrier, own_dd, threat], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})

    assert _carrier_screen_destination(own_dd, force, gs, frozenset(), {3: threat}) is None


def test_carrier_screen_is_none_when_the_threat_is_outside_radius():
    board = _sea_board(radius=12)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    own_bb = _ship(AxialCoord(0, 1), ShipKind.BATTLESHIP, PLAYER_A, 2)
    far_threat = _ship(AxialCoord(6, 0), ShipKind.BATTLESHIP, PLAYER_B, 3)  # 6 hexes -- past the default radius 4
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(carrier_screen_enabled=True))
    gs = _game_state(board, [carrier, own_bb, far_threat], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})

    assert _carrier_screen_destination(own_bb, force, gs, frozenset(), {3: far_threat}) is None


def test_carrier_screen_redirects_a_battleship_toward_a_heavy_threats_block_hex():
    board = _sea_board(radius=12)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    own_bb = _ship(AxialCoord(0, 1), ShipKind.BATTLESHIP, PLAYER_A, 2)
    threat = _ship(AxialCoord(3, 0), ShipKind.BATTLESHIP, PLAYER_B, 3)  # 3 hexes from carrier -- within radius 4
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(carrier_screen_enabled=True))
    gs = _game_state(board, [carrier, own_bb, threat], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})

    destination = _carrier_screen_destination(own_bb, force, gs, frozenset(), {3: threat})

    # Ends up genuinely between the carrier and the threat -- closer to
    # the threat than the carrier is.
    assert destination is not None
    assert distance(destination, threat.position) < distance(carrier.position, threat.position)


def test_carrier_screen_stays_in_ac_bonus_range_when_the_route_offers_it():
    # game65 turn 2, reproduced: a threat right at the heavy-screen radius
    # boundary (4 hexes). The old plain next-step block put the screen
    # three hexes from its own carrier -- outside ac_bonus_radius (2) --
    # even though later points on the exact same route stayed within it.
    board = _sea_board(radius=12)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    own_ca = _ship(AxialCoord(0, 1), ShipKind.CRUISER, PLAYER_A, 2)
    threat = _ship(AxialCoord(4, 0), ShipKind.BATTLESHIP, PLAYER_B, 3)  # 4 hexes -- right at radius 4
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(carrier_screen_enabled=True))
    gs = _game_state(board, [carrier, own_ca, threat], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})

    destination = _carrier_screen_destination(own_ca, force, gs, frozenset(), {3: threat})

    assert destination is not None
    assert distance(destination, carrier.position) <= gs.config.combat.ac_bonus_radius


def test_carrier_screen_is_none_when_already_screened_by_a_different_ship():
    board = _sea_board(radius=12)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    existing_screen = _ship(AxialCoord(2, 0), ShipKind.BATTLESHIP, PLAYER_A, 2)  # already in front
    candidate = _ship(AxialCoord(-3, 0), ShipKind.CRUISER, PLAYER_A, 4)  # a second eligible ship, elsewhere
    threat = _ship(AxialCoord(3, 0), ShipKind.BATTLESHIP, PLAYER_B, 3)
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(carrier_screen_enabled=True))
    gs = _game_state(board, [carrier, existing_screen, candidate, threat], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2, 4})

    assert _carrier_screen_destination(candidate, force, gs, frozenset(), {3: threat}) is None


def test_carrier_screen_redirects_a_destroyer_toward_a_light_threats_block_hex():
    board = _sea_board(radius=12)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    own_dd = _ship(AxialCoord(0, 1), ShipKind.DESTROYER, PLAYER_A, 2)
    threat = _ship(AxialCoord(2, 0), ShipKind.DESTROYER, PLAYER_B, 3)  # 2 hexes -- within default radius 3
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(carrier_screen_enabled=True))
    gs = _game_state(board, [carrier, own_dd, threat], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})

    destination = _carrier_screen_destination(own_dd, force, gs, frozenset(), {3: threat})

    assert destination is not None
    assert distance(destination, threat.position) < distance(carrier.position, threat.position)


def test_carrier_screen_ignores_a_submerged_enemy_submarine():
    board = _sea_board(radius=12)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    own_dd = _ship(AxialCoord(0, 1), ShipKind.DESTROYER, PLAYER_A, 2)
    submerged_threat = _ship(AxialCoord(2, 0), ShipKind.SUBMARINE, PLAYER_B, 3, surfaced=False)
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(carrier_screen_enabled=True))
    gs = _game_state(board, [carrier, own_dd, submerged_threat], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})

    assert _carrier_screen_destination(own_dd, force, gs, frozenset(), {3: submerged_threat}) is None


def test_task_force_destination_screen_takes_priority_over_the_goal():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    own_bb = _ship(AxialCoord(0, 1), ShipKind.BATTLESHIP, PLAYER_A, 2)
    threat = _ship(AxialCoord(3, 0), ShipKind.BATTLESHIP, PLAYER_B, 3)
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(carrier_screen_enabled=True))
    gs = _game_state(board, [carrier, own_bb, threat], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, port))

    destination = choose_task_force_destination(own_bb, force.goal, gs, {3: threat}, force)

    assert destination is not None
    assert distance(destination, threat.position) < distance(carrier.position, threat.position)


def test_task_force_destination_max_separation_still_wins_over_screening():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    # 6 hexes from its nearest force-mate (the carrier) -- over max_separation (4).
    own_bb = _ship(AxialCoord(6, 0), ShipKind.BATTLESHIP, PLAYER_A, 2)
    threat = _ship(AxialCoord(3, 0), ShipKind.BATTLESHIP, PLAYER_B, 3)
    config = Config(
        fow=FowConfig(enabled=False),
        ai=AiConfig(carrier_screen_enabled=True, task_force_max_separation=4),
    )
    gs = _game_state(board, [carrier, own_bb, threat], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, port))

    destination = choose_task_force_destination(own_bb, force.goal, gs, {3: threat}, force)

    # Pulled back toward the carrier to close the separation gap, not
    # toward the threat's block hex.
    assert destination is not None
    assert distance(destination, carrier.position) <= distance(own_bb.position, carrier.position)


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


# -- carrier defense (see tla.ai.task_force.compute_carrier_defense_directives) --


def test_plan_movement_carrier_defense_intercepts_a_tied_matchup():
    # The exact scenario a real game (game44) showed the naive AI getting
    # wrong: a dangerous enemy converging on an undefended carrier, with a
    # friendly ship able to intercept it this turn -- but only at an exact
    # matchup tie, which ordinary combat (_favorable_attack, `> 0` only)
    # refuses to initiate. Carrier defense should take the fight anyway.
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    threat = _ship(AxialCoord(3, 0), ShipKind.CRUISER, PLAYER_B, 2)  # reachable to the carrier next turn
    # Same kind/stats as the threat -- an exact matchup_score == 0 tie --
    # and far enough from the carrier that only carrier defense, not any
    # other doctrine, explains it engaging.
    responder = _ship(AxialCoord(7, 0), ShipKind.CRUISER, PLAYER_A, 3)
    gs = _game_state(_sea_board(radius=10), [carrier, threat, responder], config=Config(fow=FowConfig(enabled=False)))

    _run(NaivePolicy(), gs, PLAYER_A)

    assert len(gs.battle_log) >= 1
    assert gs.battle_log[0].attacker_id == 3
    assert gs.battle_log[0].defender_id == 2


def test_plan_movement_carrier_defense_does_not_trigger_when_carrier_is_safe():
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    responder = _ship(AxialCoord(7, 0), ShipKind.CRUISER, PLAYER_A, 3)
    # Same tied matchup as above, but too far from the carrier (cruiser
    # movement 4) to threaten it next turn -- no directive should fire.
    distant_threat = _ship(AxialCoord(8, 0), ShipKind.CRUISER, PLAYER_B, 2)
    gs = _game_state(
        _sea_board(radius=12), [carrier, responder, distant_threat], config=Config(fow=FowConfig(enabled=False))
    )

    directives = compute_carrier_defense_directives(gs, PLAYER_A, {2: distant_threat}, gs.config.ai)

    assert directives == []


def test_port_defense_fallback_does_not_force_an_unfavorable_attack():
    # Real bug (found via game63 replay review): a counterattack directive
    # is a *group*-power decision (compute_port_defense_directives), so a
    # hopelessly outmatched individual responder -- like this patrol boat
    # against a battleship -- can still be listed in `counterattack`.
    # _favorable_attack correctly refuses that 1v1 (matchup_score deeply
    # negative), but the old final fallback (`_step_toward(ship, ...,
    # nearest_threat.position)`, no `avoid`) then walked the ship straight
    # onto the threat's own hex anyway, converting "close the distance"
    # into the very attack just rejected.
    port = AxialCoord(0, 0)
    board = _port_board(port)
    threat = _ship(AxialCoord(2, 0), ShipKind.BATTLESHIP, PLAYER_B, 2)
    weak_responder = _ship(AxialCoord(0, 1), ShipKind.PATROL_BOAT, PLAYER_A, 1)
    gs = _game_state(board, [weak_responder, threat], config=Config(fow=FowConfig(enabled=False)))
    directive = PortDefenseDirective(port=port, threats=[threat], counterattack=[weak_responder])

    destination = _port_defense_destination(weak_responder, directive, gs, {2: threat})

    assert destination is not None
    assert destination != threat.position
    assert gs.ship_at(destination) is None  # never silently resolves into an attack


def test_carrier_defense_fallback_does_not_force_an_unfavorable_attack():
    # Same fix, same bug, for the carrier-defense counterpart.
    carrier = _ship(AxialCoord(-3, 0), ShipKind.CARRIER, PLAYER_A, 1)
    threat = _ship(AxialCoord(2, 0), ShipKind.BATTLESHIP, PLAYER_B, 2)
    weak_responder = _ship(AxialCoord(0, 1), ShipKind.PATROL_BOAT, PLAYER_A, 3)
    gs = _game_state(_sea_board(radius=10), [carrier, weak_responder, threat], config=Config(fow=FowConfig(enabled=False)))
    directive = CarrierDefenseDirective(carrier=carrier, threats=[threat], counterattack=[weak_responder])

    destination = _carrier_defense_destination(weak_responder, directive, gs, {2: threat})

    assert destination is not None
    assert destination != threat.position
    assert gs.ship_at(destination) is None


def test_plan_movement_port_defense_does_not_sacrifice_an_outmatched_counterattacker():
    # End-to-end: a strong enough group (battleship + patrol boat) is not
    # outmatched overall, so both get a counterattack directive -- but the
    # patrol boat's own 1v1 against the battleship threat is hopeless. It
    # should approach without engaging, not walk into a battle it can't
    # win as a side effect of "get closer."
    port = AxialCoord(0, 0)
    board = _port_board(port)
    strong = _ship(AxialCoord(1, 0), ShipKind.CRUISER, PLAYER_A, 1)
    weak = _ship(AxialCoord(0, 1), ShipKind.PATROL_BOAT, PLAYER_A, 2)
    threat = _ship(AxialCoord(3, 0), ShipKind.BATTLESHIP, PLAYER_B, 3)
    gs = _game_state(board, [strong, weak, threat], config=Config(fow=FowConfig(enabled=False)))

    _run(NaivePolicy(), gs, PLAYER_A)

    assert 2 in gs.ships and gs.ships[2].current_hp == 2  # patrol boat untouched by combat
    assert not any(entry.attacker_id == 2 or entry.defender_id == 2 for entry in gs.battle_log)


# -- AiConfig.defense_stall_turns (see NaivePolicy._defense_stall_turns) -----
# fleet=FleetConfig(counts={}) throughout: an empty starting-fleet belief
# keeps EnemyModel.expected_strength() at (0, 0) until something is
# actually sighted, which keeps posture NEUTRAL (margin 0) so these
# scenarios' counterattack-vs-block group decision is exactly what the
# plain (0, 0) inputs -- not a posture-shifted margin -- would predict.


def test_defense_stall_turns_counts_up_then_excludes_the_ship():
    # A counterattack-role ship whose own matchup is hopeless (see
    # test_plan_movement_port_defense_does_not_sacrifice_an_outmatched_
    # counterattacker) now keeps trying, and failing, every turn -- until
    # AiConfig.defense_stall_turns is reached, after which it's skipped
    # entirely (excluded from `resolved`, left to ordinary movement)
    # rather than the count climbing forever.
    port = AxialCoord(0, 0)
    board = _port_board(port)
    strong = _ship(AxialCoord(1, 0), ShipKind.CRUISER, PLAYER_A, 1)
    weak = _ship(AxialCoord(0, 1), ShipKind.PATROL_BOAT, PLAYER_A, 2)
    threat = _ship(AxialCoord(3, 0), ShipKind.BATTLESHIP, PLAYER_B, 3)
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(defense_stall_turns=2), fleet=FleetConfig(counts={}))
    gs = _game_state(board, [strong, weak, threat], config=config)
    policy = NaivePolicy()

    for _ in range(2):
        start_movement_phase(gs, PLAYER_A)
        list(policy.plan_movement(gs, PLAYER_A))
    assert policy._defense_stall_turns[PLAYER_A] == {1: 2, 2: 2}

    for _ in range(2):
        start_movement_phase(gs, PLAYER_A)
        list(policy.plan_movement(gs, PLAYER_A))
        # Frozen at the limit, not climbing past it -- confirms exclusion,
        # not just a slower-growing count.
        assert policy._defense_stall_turns[PLAYER_A] == {1: 2, 2: 2}

    # Never once resolved into the old, unconditional-attack bug either.
    assert not gs.battle_log


def test_defense_stall_turns_resets_the_moment_the_ship_actually_attacks():
    port = AxialCoord(0, 0)
    board = _port_board(port)
    defender = _ship(AxialCoord(1, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    weak_threat = _ship(AxialCoord(3, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)  # clearly favorable
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(defense_stall_turns=2), fleet=FleetConfig(counts={}))
    gs = _game_state(board, [defender, weak_threat], config=config)
    policy = NaivePolicy()

    start_movement_phase(gs, PLAYER_A)
    list(policy.plan_movement(gs, PLAYER_A))

    assert len(gs.battle_log) >= 1
    # Landed a real attack -- never even entered the tracker.
    assert 1 not in policy._defense_stall_turns.get(PLAYER_A, {})


def test_defense_stall_turns_drops_a_ship_no_longer_a_candidate():
    # A ship no longer considered for defense duty at all (its threat is
    # gone) has its count dropped outright, not just held -- so it never
    # unfairly starts pre-excluded against a later, unrelated threat.
    port = AxialCoord(0, 0)
    board = _port_board(port)
    strong = _ship(AxialCoord(1, 0), ShipKind.CRUISER, PLAYER_A, 1)
    weak = _ship(AxialCoord(0, 1), ShipKind.PATROL_BOAT, PLAYER_A, 2)
    threat = _ship(AxialCoord(3, 0), ShipKind.BATTLESHIP, PLAYER_B, 3)
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(defense_stall_turns=2), fleet=FleetConfig(counts={}))
    gs = _game_state(board, [strong, weak, threat], config=config)
    policy = NaivePolicy()

    for _ in range(2):
        start_movement_phase(gs, PLAYER_A)
        list(policy.plan_movement(gs, PLAYER_A))
    assert policy._defense_stall_turns[PLAYER_A] == {1: 2, 2: 2}

    del gs.ships[3]  # threat gone (sunk/left) -- no longer a candidate at all
    start_movement_phase(gs, PLAYER_A)
    list(policy.plan_movement(gs, PLAYER_A))

    assert policy._defense_stall_turns[PLAYER_A] == {}


def test_defense_stall_turns_never_tracks_a_block_role_ship():
    # A block-role responder (a deliberate, expected-loss sacrifice -- see
    # test_plan_movement_blocks_with_a_submarine_when_outmatched) is never
    # entered into the tracker at all, turn after turn -- it isn't
    # "repeatedly declining a fight," it's doing exactly what it was
    # assigned to do.
    port = AxialCoord(0, 0)
    board = _port_board(port)
    weak = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)
    sub = _ship(AxialCoord(1, 1), ShipKind.SUBMARINE, PLAYER_A, 2)
    strong_threat = _ship(AxialCoord(4, 0), ShipKind.BATTLESHIP, PLAYER_B, 3)
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(defense_stall_turns=2))
    gs = _game_state(board, [weak, sub, strong_threat], config=config)
    policy = NaivePolicy()

    for _ in range(4):
        start_movement_phase(gs, PLAYER_A)
        list(policy.plan_movement(gs, PLAYER_A))
        assert policy._defense_stall_turns.get(PLAYER_A, {}) == {}


# -- cross-model belief reconciliation (see NaivePolicy._reconcile_defender_belief) --
# Regression tests for a real gap a replay review caught: EnemyModel could
# only ever observe combat where it was the attacker (game_state.battle_log
# has a half-turn lifecycle, gone by the time the defending side's own model
# next runs) -- so an enemy ship that attacked one of our ships was never
# recorded as sighted, even though being attacked obviously reveals the
# attacker. NaivePolicy owns both players' models, so it can hand each
# half-turn's battle_log to the *other* player's model too, before the game
# loop clears it.


def test_being_attacked_reveals_the_attacker_to_the_defenders_own_model():
    from tla.game_state import BattleLogEntry

    board = _sea_board(radius=10)
    gs = _game_state(board, [], config=Config(fow=FowConfig(enabled=True)))
    gs.battle_log.append(
        BattleLogEntry(
            attacker_id=7, attacker_kind=ShipKind.SUBMARINE, attacker_owner=PLAYER_A,
            defender_id=2, defender_kind=ShipKind.BATTLESHIP, defender_owner=PLAYER_B,
            battle_hex=AxialCoord(3, 0), damage_to_defender=1, damage_to_attacker=1,
            attacker_carrier_bonus=0, defender_carrier_bonus=0,
            defender_hp_after=11, attacker_hp_after=3, defender_sunk=False, attacker_sunk=False,
        )
    )
    policy = NaivePolicy()

    # Player B's own plan_movement/begin_turn has never been called at all
    # -- yet its EnemyModel should already know about the attacking
    # submarine, purely from having been attacked by it.
    policy._reconcile_defender_belief(gs, PLAYER_A)

    b_model = policy.enemy_model_for(PLAYER_B)
    assert b_model is not None
    assert 7 in b_model.tracked_ships()
    tracked = b_model.tracked_ships()[7]
    assert tracked.kind == ShipKind.SUBMARINE
    assert tracked.last_seen_position == AxialCoord(3, 0)
    assert tracked.last_known_hp == 3


def test_an_attacker_that_dies_in_the_exchange_is_recorded_sunk_by_the_defenders_model():
    from tla.game_state import BattleLogEntry

    board = _sea_board(radius=10)
    config = Config(fow=FowConfig(enabled=True), fleet=FleetConfig(counts={ShipKind.SUBMARINE: 1}))
    gs = _game_state(board, [], config=config)
    policy = NaivePolicy()
    # Seed B's model so the submarine starts out merely "somewhere in the
    # pool", to also confirm the pool count drops correctly on this path.
    b_model = policy._enemy_model_for(gs, PLAYER_B)
    assert b_model.alive_count(ShipKind.SUBMARINE) == 1

    gs.battle_log.append(
        BattleLogEntry(
            attacker_id=7, attacker_kind=ShipKind.SUBMARINE, attacker_owner=PLAYER_A,
            defender_id=2, defender_kind=ShipKind.BATTLESHIP, defender_owner=PLAYER_B,
            battle_hex=AxialCoord(3, 0), damage_to_defender=1, damage_to_attacker=4,
            attacker_carrier_bonus=0, defender_carrier_bonus=0,
            defender_hp_after=11, attacker_hp_after=0, defender_sunk=False, attacker_sunk=True,
        )
    )

    policy._reconcile_defender_belief(gs, PLAYER_A)

    assert 7 not in b_model.tracked_ships()
    assert b_model.alive_count(ShipKind.SUBMARINE) == 0


# -- global layer: posture shifts engagement margins (see tla.ai.global_strategy) --


def test_plan_movement_shifts_engagement_margins_by_posture_without_compounding():
    board = _sea_board(radius=10)
    # Heavily outmatch the enemy's whole believed fleet (one patrol boat,
    # never sighted) with three of our own battleships -- should read as
    # clearly AGGRESSIVE.
    own_ships = [_ship(AxialCoord(i, 0), ShipKind.BATTLESHIP, PLAYER_A, i + 1) for i in range(3)]
    config = Config(fow=FowConfig(enabled=True), fleet=FleetConfig(counts={ShipKind.PATROL_BOAT: 1}))
    gs = _game_state(board, own_ships, config=config)
    policy = NaivePolicy()
    base_margin = gs.config.ai.task_force_outnumbered_margin
    shift = gs.config.ai.posture_margin_shift

    _run(policy, gs, PLAYER_A)

    assert policy.posture_for(PLAYER_A) == Posture.AGGRESSIVE
    assert gs.config.ai.task_force_outnumbered_margin == base_margin + shift
    assert gs.config.ai.port_defense_margin == shift
    assert gs.config.ai.carrier_defense_margin == shift

    # A second call for the same player, same imbalance -- the shift must
    # not compound onto itself turn after turn.
    _run(policy, gs, PLAYER_A)

    assert policy.posture_for(PLAYER_A) == Posture.AGGRESSIVE
    assert gs.config.ai.task_force_outnumbered_margin == base_margin + shift


def test_posture_snapshot_for_reports_posture_and_its_own_believed_enemy_inputs():
    board = _sea_board(radius=10)
    own_ships = [_ship(AxialCoord(i, 0), ShipKind.BATTLESHIP, PLAYER_A, i + 1) for i in range(3)]
    config = Config(fow=FowConfig(enabled=True), fleet=FleetConfig(counts={ShipKind.PATROL_BOAT: 1}))
    gs = _game_state(board, own_ships, config=config)
    policy = NaivePolicy()

    assert policy.posture_snapshot_for(PLAYER_A) is None  # nothing planned for yet

    _run(policy, gs, PLAYER_A)

    stats = Config().ship_stats.stats
    battleship_hp, battleship_dmg = stats[ShipKind.BATTLESHIP].hp, stats[ShipKind.BATTLESHIP].damage
    patrol_hp, patrol_dmg = stats[ShipKind.PATROL_BOAT].hp, stats[ShipKind.PATROL_BOAT].damage
    assert policy.posture_snapshot_for(PLAYER_A) == {
        "posture": "aggressive",
        "own_hp": battleship_hp * 3,
        "own_damage": battleship_dmg * 3,
        "believed_enemy_hp": patrol_hp,
        "believed_enemy_damage": patrol_dmg,
    }


def test_plan_movement_assigns_a_defend_port_goal_under_sustained_defensive_pressure():
    port = AxialCoord(0, 0)
    board = _port_board(port, radius=12)
    ship = _ship(AxialCoord(2, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)  # weak -- posture should read DEFENSIVE
    gs = _game_state(
        board,
        [ship],
        config=Config(
            fow=FowConfig(enabled=False),
            fleet=FleetConfig(counts={ShipKind.BATTLESHIP: 3, ShipKind.SUBMARINE: 1}),
            ai=AiConfig(task_force_min_size=1),
        ),
    )
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})
    policy = NaivePolicy()
    policy._task_forces[PLAYER_A] = [force]
    policy._next_force_id = 2
    # A believed submerged submarine right by the port -- deliberately a
    # submarine (not a battleship): the port's own vision radius (default
    # 4) equals port_defense_trigger_radius, so a non-submarine belief
    # this close would be immediately falsified/excluded by our own real
    # vision (correctly -- see EnemyModel's vision-exclusion diffusion).
    # A submerged sub stays undetected by ordinary vision (tla.fow.
    # is_hidden), so its belief survives -- exactly the case this feature
    # exists for: a threat compute_port_defense_directives' visible-enemy
    # check could never see coming either.
    model = policy._enemy_model_for(gs, PLAYER_A)
    model._resolve(99, ShipKind.SUBMARINE, AxialCoord(1, 0), 4, turn=1, surfaced=False)
    model._ensure_fields_for_unseen({})

    _run(policy, gs, PLAYER_A)

    assert policy.posture_for(PLAYER_A) == Posture.DEFENSIVE
    assert force.goal == TaskForceGoal(kind=GoalKind.DEFEND_PORT, target=port)


# -- tactical layer: kill-prioritization (see tla.ai.tactics) ---------------


def test_plan_movement_prioritizes_the_higher_value_target_via_the_tactical_layer():
    # A lone battleship can reach and secure either a weak patrol boat or
    # a more valuable cruiser this turn -- it only gets to attack once, so
    # the tactical layer (tla.ai.tactics.secure_kills_pass, run *before*
    # the ordinary per-ship loop) should commit it to the higher future-
    # damage-removed target (the cruiser) rather than whichever the
    # ordinary loop's own per-ship pick happens to prefer. See tests/
    # test_ai_tactics.py for the same behavior tested directly against
    # secure_kills_pass; this is the same thing confirmed end-to-end.
    board = _sea_board(radius=10)
    attacker = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    weak = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    juicier = _ship(AxialCoord(0, 1), ShipKind.CRUISER, PLAYER_B, 3)
    gs = _game_state(board, [attacker, weak, juicier])

    _run(NaivePolicy(), gs, PLAYER_A)

    assert 2 in gs.ships  # the patrol boat was left alone
    assert 3 not in gs.ships  # the cruiser was sunk instead


# -- "don't advance into a losing battle" (see _project_engagement_value) ---


def test_weighted_damage_is_just_the_stat_for_a_non_carrier():
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    ai_config = AiConfig(carrier_assist_value=2.5)
    stats = Config().ship_stats.stats

    assert _weighted_damage(ship, ai_config, stats) == stats[ShipKind.DESTROYER].damage


def test_weighted_damage_adds_the_carrier_assist_value():
    ship = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    ai_config = AiConfig(carrier_assist_value=2.5)
    stats = Config().ship_stats.stats

    assert _weighted_damage(ship, ai_config, stats) == stats[ShipKind.CARRIER].damage + 2.5


def test_project_engagement_value_is_zero_with_no_visible_enemies():
    board = _sea_board()
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    gs = _game_state(board, [ship])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})

    assert _project_engagement_value(force, gs, PLAYER_A, {}, [force], {}, _empty_enemy_model(gs)) == 0.0


def test_project_engagement_value_is_positive_for_a_clean_kill_with_no_risk():
    board = _sea_board()
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    target = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)  # one-shot kill, and the only visible enemy
    gs = _game_state(board, [ship, target])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})
    pace = _compute_force_pace(gs, [force])

    value = _project_engagement_value(force, gs, PLAYER_A, {2: target}, [force], pace, _empty_enemy_model(gs))

    assert value > 0


def test_project_engagement_value_is_negative_for_a_hopeless_fight():
    board = _sea_board()
    ship = _ship(AxialCoord(0, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)
    enemy = _ship(AxialCoord(1, 0), ShipKind.BATTLESHIP, PLAYER_B, 2)  # can't be hurt, would sink us for nothing
    gs = _game_state(board, [ship, enemy])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})
    pace = _compute_force_pace(gs, [force])

    value = _project_engagement_value(force, gs, PLAYER_A, {2: enemy}, [force], pace, _empty_enemy_model(gs))

    assert value < 0


def test_project_engagement_value_a_confirmed_kill_is_excluded_from_the_risk_side():
    # A one-shot-kill target shouldn't also count as a threat -- it's dead
    # this turn and can't respond next turn regardless of its own stats.
    board = _sea_board()
    ship = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    target = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)  # one-shot kill for the battleship
    gs = _game_state(board, [ship, target])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})
    pace = _compute_force_pace(gs, [force])

    value = _project_engagement_value(force, gs, PLAYER_A, {2: target}, [force], pace, _empty_enemy_model(gs))

    stats = Config().ship_stats.stats
    # gained = patrol boat's damage stat; at_risk = 0 (its only threat is
    # the target we're confirmed to sink) -- not negative from double
    # counting the same ship as both a kill and a live threat.
    assert value == stats[ShipKind.PATROL_BOAT].damage / stats[ShipKind.BATTLESHIP].damage


def test_project_engagement_value_weighs_an_at_risk_carrier_more_heavily():
    # Two otherwise-identical scenarios -- only the at-risk ship's kind
    # differs -- isolate the carrier weighting's actual effect on the
    # aggregate ratio, not just tla.ai.policy._weighted_damage in
    # isolation (already covered above).
    board = _sea_board(radius=15)
    stats = Config().ship_stats.stats
    config = Config(ai=AiConfig(carrier_assist_value=2.5))

    def scenario_value(at_risk_kind: ShipKind, safe_kind: ShipKind) -> float:
        at_risk_ship = Ship(
            id=1, kind=at_risk_kind, owner=PLAYER_A, position=AxialCoord(1, 0),
            current_hp=stats[at_risk_kind].hp, movement_remaining=0,
        )
        safe_ship = Ship(
            id=3, kind=safe_kind, owner=PLAYER_A, position=AxialCoord(-8, 0),
            current_hp=stats[safe_kind].hp, movement_remaining=0,
        )
        enemy = Ship(
            id=2, kind=ShipKind.BATTLESHIP, owner=PLAYER_B, position=AxialCoord(0, 0),
            current_hp=stats[ShipKind.BATTLESHIP].hp, movement_remaining=stats[ShipKind.BATTLESHIP].movement,
        )
        gs = GameState(config=config, board=board, ships={1: at_risk_ship, 3: safe_ship, 2: enemy})
        force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 3})
        pace = _compute_force_pace(gs, [force])
        return _project_engagement_value(force, gs, PLAYER_A, {2: enemy}, [force], pace, _empty_enemy_model(gs))

    carrier_at_risk = scenario_value(ShipKind.CARRIER, ShipKind.DESTROYER)
    destroyer_at_risk = scenario_value(ShipKind.DESTROYER, ShipKind.DESTROYER)

    assert carrier_at_risk < destroyer_at_risk < 0


# -- Multi-candidate tactical strategy: AGGRESSIVE/ADVANCE/HOLD/RETREAT ------


def test_favorable_attack_tie_tolerant_accepts_a_mutual_kill_trade():
    board = _sea_board()
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    mirror = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_B, 2)  # identical stats -> an exact tie
    gs = _game_state(board, [ship, mirror])

    # Default (strict > 0) gate declines a tie -- it's a mutual kill, not a
    # clean win, and an equal-cost mirror matchup isn't a worthwhile one
    # either (see worth_a_tie -- needs the target to cost strictly more).
    assert _favorable_attack(ship, gs, {2: mirror}, [ship]) is None
    # tie_tolerant=True (Strategy.AGGRESSIVE) accepts it regardless.
    assert _favorable_attack(ship, gs, {2: mirror}, [ship], tie_tolerant=True) == mirror.position


def test_favorable_attack_declines_a_non_worthwhile_tie_even_without_aggressive():
    # Destroyer (cost 4) vs a patrol boat (cost 1) at HP levels that tie
    # the race -- not accepted even under the default (non-tie_tolerant)
    # gate, since trading down in value isn't worth it.
    board = _sea_board()
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1, hp=2)
    target = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2, hp=4)
    gs = _game_state(board, [ship, target])
    assert scoring.matchup_score(ship, target, gs) == 0  # confirm this really is a tie

    assert _favorable_attack(ship, gs, {2: target}, [ship]) is None


def test_favorable_attack_accepts_a_worthwhile_tie_without_aggressive():
    # Cruiser (cost 7) vs a battleship (cost 10) at HP levels that tie the
    # race -- accepted under the plain default gate (no tie_tolerant, no
    # Strategy.AGGRESSIVE needed): trading up in value is worth it even at
    # even odds. Real replay-found gap (game65): cruisers declining
    # exactly this kind of trade against enemy carriers.
    board = _sea_board()
    ship = _ship(AxialCoord(0, 0), ShipKind.CRUISER, PLAYER_A, 1, hp=4)
    target = _ship(AxialCoord(1, 0), ShipKind.BATTLESHIP, PLAYER_B, 2, hp=3)
    gs = _game_state(board, [ship, target])
    assert scoring.matchup_score(ship, target, gs) == 0  # confirm this really is a tie

    assert _favorable_attack(ship, gs, {2: target}, [ship]) == target.position


def test_strategy_hold_holds_a_ship_with_no_cohesion_issue_in_place():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    ship = _ship(AxialCoord(0, 0), ShipKind.CRUISER, PLAYER_A, 1)
    gs = _game_state(board, [ship])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, port))
    pace = _compute_force_pace(gs, [force])

    advance = _choose_destination(ship, gs, PLAYER_A, {}, [force], pace, strategy=Strategy.ADVANCE)
    hold = _choose_destination(ship, gs, PLAYER_A, {}, [force], pace, strategy=Strategy.HOLD)

    assert advance != ship.position  # ADVANCE (default) actually moves toward the goal
    assert hold == ship.position  # HOLD holds exactly in place


def test_strategy_hold_still_lets_a_straggler_close_the_cohesion_gap():
    board = _sea_board(radius=12)
    port = AxialCoord(10, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    straggler = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    anchor = _ship(AxialCoord(6, 0), ShipKind.CRUISER, PLAYER_A, 2)
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(task_force_max_separation=4))
    gs = _game_state(board, [straggler, anchor], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, port))
    pace = _compute_force_pace(gs, [force])

    # max_steps=0, exactly what Strategy.HOLD forces via _choose_destination
    # -- _cohesion_destination ignores it entirely, by design (see its own
    # docstring), so the straggler still gets pulled toward the group.
    destination = _choose_destination(straggler, gs, PLAYER_A, {}, [force], pace, strategy=Strategy.HOLD)

    assert destination is not None
    assert destination != straggler.position
    assert distance(destination, anchor.position) < distance(straggler.position, anchor.position)


def test_strategy_retreat_shadows_the_goal_even_when_the_force_is_not_retreating():
    # The hypothetical-scoring case: force.retreating is still False (never
    # set), but strategy=Strategy.RETREAT alone must disengage toward home
    # instead of the real goal -- what lets _pick_strategy_for_force ask
    # "what would retreating look like this turn" before ever committing.
    board = _sea_board(radius=12)
    home_port = AxialCoord(-6, 0)
    board.tiles[home_port] = Tile(coord=home_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    enemy_port = AxialCoord(6, 0)
    board.tiles[enemy_port] = Tile(coord=enemy_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    destroyer = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    bait = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)  # an easy, favorable target
    gs = _game_state(board, [destroyer, bait], config=Config(fow=FowConfig(enabled=False)))
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, enemy_port))
    assert force.retreating is False

    advance_destination = choose_task_force_destination(destroyer, force.goal, gs, {2: bait}, force)
    assert advance_destination == bait.position  # ADVANCE (default) takes the favorable fight

    retreat_destination = choose_task_force_destination(
        destroyer, force.goal, gs, {2: bait}, force, strategy=Strategy.RETREAT
    )
    assert retreat_destination != bait.position
    assert distance(retreat_destination, home_port) < distance(destroyer.position, home_port)


def test_pick_strategy_for_force_defaults_to_advance_with_no_visible_enemies():
    board = _sea_board()
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    gs = _game_state(board, [ship])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, AxialCoord(5, 0)))
    pace = _compute_force_pace(gs, [force])

    result = _pick_strategy_for_force(force, gs, PLAYER_A, {}, [force], pace, _empty_enemy_model(gs))

    assert result == (Strategy.ADVANCE, False)


def test_pick_strategy_for_force_skips_evaluation_when_clearly_ahead():
    board = _sea_board()
    strong = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1, hp=12)
    weak_enemy = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    gs = _game_state(board, [strong, weak_enemy], config=Config(fow=FowConfig(enabled=False)))
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, AxialCoord(5, 0)))
    pace = _compute_force_pace(gs, [force])

    result = _pick_strategy_for_force(force, gs, PLAYER_A, {2: weak_enemy}, [force], pace, _empty_enemy_model(gs))

    assert result == (Strategy.ADVANCE, False)


def test_pick_strategy_for_force_skips_evaluation_when_clearly_outmatched():
    board = _sea_board()
    weak = _ship(AxialCoord(0, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)
    strong_enemy = _ship(AxialCoord(1, 0), ShipKind.BATTLESHIP, PLAYER_B, 2, hp=12)
    gs = _game_state(board, [weak, strong_enemy], config=Config(fow=FowConfig(enabled=False)))
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, AxialCoord(5, 0)))
    pace = _compute_force_pace(gs, [force])

    result = _pick_strategy_for_force(force, gs, PLAYER_A, {2: strong_enemy}, [force], pace, _empty_enemy_model(gs))

    assert result == (Strategy.ADVANCE, False)


def test_pick_strategy_for_force_prefers_advance_in_a_genuine_parity_standoff():
    # Mirrors the user's own worked example: forces are roughly matched
    # (an exact tie in group_power here) and nothing is actually reachable
    # this turn either way -- every candidate scores the same (0.0), so the
    # least drastic one (ADVANCE) should win the tie, not RETREAT.
    board = _sea_board(radius=15)
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    enemy = _ship(AxialCoord(10, 0), ShipKind.DESTROYER, PLAYER_B, 2)  # same kind -- exact parity, out of reach
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(task_force_threat_radius=12))
    gs = _game_state(board, [ship, enemy], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, AxialCoord(5, 0)))
    pace = _compute_force_pace(gs, [force])

    strategy, should_retreat = _pick_strategy_for_force(
        force, gs, PLAYER_A, {2: enemy}, [force], pace, _empty_enemy_model(gs)
    )

    assert strategy in (Strategy.ADVANCE, Strategy.HOLD)
    assert should_retreat is False


def test_pick_strategy_for_force_retreats_when_only_retreating_avoids_a_clear_loss():
    # Group power is an exact tie (so the cheap gate doesn't short-circuit),
    # but one member (a heavily damaged patrol boat) sits exposed to a
    # one-shot kill it can't escape unless the whole force actually
    # retreats -- ADVANCE/HOLD/AGGRESSIVE all leave it in place (nothing
    # about their pipeline moves a ship away from an unfavorable fight, and
    # this force has no capital ship to trigger the escort-rearguard
    # pull), while RETREAT's unpaced dash home is fast enough to clear the
    # enemy's reach.
    board = _sea_board(radius=35)
    home_port = AxialCoord(30, 0)
    board.tiles[home_port] = Tile(coord=home_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    cruiser = _ship(AxialCoord(25, 0), ShipKind.CRUISER, PLAYER_A, 1)
    patrol = _ship(AxialCoord(2, 0), ShipKind.PATROL_BOAT, PLAYER_A, 2, hp=1)  # heavily damaged, one-shot fodder
    enemy = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_B, 3, hp=12)
    gs = _game_state(board, [cruiser, patrol, enemy], config=Config(fow=FowConfig(enabled=False)))
    goal = TaskForceGoal(GoalKind.CAPTURE_PORT, target=patrol.position)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2}, goal=goal)
    pace = _compute_force_pace(gs, [force])

    strategy, should_retreat = _pick_strategy_for_force(
        force, gs, PLAYER_A, {3: enemy}, [force], pace, _empty_enemy_model(gs)
    )

    assert strategy == Strategy.RETREAT
    assert should_retreat is True


# -- probable-threat risk (AiConfig.probable_threat_engagement_enabled) -----


def test_project_engagement_value_at_risk_increases_for_a_probable_unseen_enemy():
    # A visible enemy is present (so the outer not-visible_enemies gate
    # doesn't short-circuit before the new logic ever runs) but far too
    # weak/distant to threaten us -- only injected belief mass drives the
    # difference here. strategy=Strategy.HOLD makes the hypothetical
    # position deterministic (exactly ship.position, via _step_toward's
    # max_steps=0 special case), so the belief point mass placed there is
    # a guaranteed match, not a coordinate-arithmetic guess.
    board = _sea_board(radius=20)
    ship = _ship(AxialCoord(0, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)
    weak_visible = _ship(AxialCoord(15, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)  # too far to reach or threaten us
    gs = _game_state(board, [ship, weak_visible])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, ship.position))
    pace = _compute_force_pace(gs, [force])

    model = _empty_enemy_model(gs)
    pool = model._pool_for(ShipKind.BATTLESHIP)
    pool.count = 1
    pool.field.set_point_mass(ship.position, 1.0)  # a believed one-shot-kill threat, right where we'll be

    enabled = _project_engagement_value(
        force, gs, PLAYER_A, {2: weak_visible}, [force], pace, model, strategy=Strategy.HOLD
    )
    disabled_gs = replace(gs, config=replace(gs.config, ai=AiConfig(probable_threat_engagement_enabled=False)))
    disabled = _project_engagement_value(
        force, disabled_gs, PLAYER_A, {2: weak_visible}, [force], pace, model, strategy=Strategy.HOLD
    )

    assert enabled < disabled


def test_project_engagement_value_ignores_probable_threat_below_one_round_kill_threshold():
    board = _sea_board(radius=20)
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)  # hp 6
    weak_visible = _ship(AxialCoord(15, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    gs = _game_state(board, [ship, weak_visible])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, ship.position))
    pace = _compute_force_pace(gs, [force])

    model = _empty_enemy_model(gs)
    pool = model._pool_for(ShipKind.PATROL_BOAT)  # damage 1 -- can't plausibly one-shot a 6-hp destroyer
    pool.count = 1
    pool.field.set_point_mass(ship.position, 1.0)

    enabled = _project_engagement_value(
        force, gs, PLAYER_A, {2: weak_visible}, [force], pace, model, strategy=Strategy.HOLD
    )
    disabled_gs = replace(gs, config=replace(gs.config, ai=AiConfig(probable_threat_engagement_enabled=False)))
    disabled = _project_engagement_value(
        force, disabled_gs, PLAYER_A, {2: weak_visible}, [force], pace, model, strategy=Strategy.HOLD
    )

    assert enabled == disabled


def test_project_engagement_value_no_double_counting_from_a_visible_enemys_own_mass():
    # A real, visible threat already drives at_risk on its own; the model
    # otherwise carries no belief at all (nothing else to double-count).
    # Enabling the flag must not change the score.
    board = _sea_board(radius=20)
    ship = _ship(AxialCoord(0, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)
    strong_visible = _ship(AxialCoord(1, 0), ShipKind.BATTLESHIP, PLAYER_B, 2)
    gs = _game_state(board, [ship, strong_visible])
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, ship.position))
    pace = _compute_force_pace(gs, [force])
    model = _empty_enemy_model(gs)

    enabled = _project_engagement_value(
        force, gs, PLAYER_A, {2: strong_visible}, [force], pace, model, strategy=Strategy.HOLD
    )
    disabled_gs = replace(gs, config=replace(gs.config, ai=AiConfig(probable_threat_engagement_enabled=False)))
    disabled = _project_engagement_value(
        force, disabled_gs, PLAYER_A, {2: strong_visible}, [force], pace, model, strategy=Strategy.HOLD
    )

    assert enabled == disabled


def test_probable_threat_radius_bounds_which_believed_mass_counts():
    board = _sea_board(radius=20)
    ship = _ship(AxialCoord(0, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)
    weak_visible = _ship(AxialCoord(15, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(probable_threat_radius=3))
    gs = _game_state(board, [ship, weak_visible], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, ship.position))
    pace = _compute_force_pace(gs, [force])

    outside = _empty_enemy_model(gs)
    outside._pool_for(ShipKind.BATTLESHIP).count = 1
    outside._pool_for(ShipKind.BATTLESHIP).field.set_point_mass(AxialCoord(4, 0), 1.0)  # radius 3 + 1
    value_outside = _project_engagement_value(
        force, gs, PLAYER_A, {2: weak_visible}, [force], pace, outside, strategy=Strategy.HOLD
    )

    inside = _empty_enemy_model(gs)
    inside._pool_for(ShipKind.BATTLESHIP).count = 1
    inside._pool_for(ShipKind.BATTLESHIP).field.set_point_mass(AxialCoord(3, 0), 1.0)  # exactly radius 3
    value_inside = _project_engagement_value(
        force, gs, PLAYER_A, {2: weak_visible}, [force], pace, inside, strategy=Strategy.HOLD
    )

    assert value_outside == 0.0
    assert value_inside < 0.0


def test_pick_strategy_for_force_flips_to_retreat_with_a_strong_probable_threat():
    # Mirrors test_pick_strategy_for_force_retreats_when_only_retreating_
    # avoids_a_clear_loss's shape, but the kill threat to the exposed
    # patrol boat comes entirely from injected belief, not a visible
    # enemy -- the visible enemies here (a mirrored cruiser + patrol boat,
    # near the *safe* cruiser only) exist purely to give the cheap
    # group_power gate an exact tie to compare, so the full 4-way
    # evaluation actually runs; neither can reach our patrol boat's
    # position regardless of strategy. probable_threat_radius is lowered
    # so RETREAT's one-turn, unpaced dash is far enough to clear it (a
    # radius comparable to a ship's own single-turn movement needs more
    # separation than one hop provides -- see probable_threat_radius's own
    # docstring for why 6, the default, is already a deliberately generous
    # upper bound).
    board = _sea_board(radius=35)
    home_port = AxialCoord(30, 0)
    board.tiles[home_port] = Tile(coord=home_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    safe_cruiser = _ship(AxialCoord(25, 0), ShipKind.CRUISER, PLAYER_A, 1)
    patrol = _ship(AxialCoord(2, 0), ShipKind.PATROL_BOAT, PLAYER_A, 2)
    enemy_cruiser = _ship(AxialCoord(25, 4), ShipKind.CRUISER, PLAYER_B, 3)  # mirrors safe_cruiser -- exact tie
    enemy_patrol = _ship(AxialCoord(21, 0), ShipKind.PATROL_BOAT, PLAYER_B, 4)  # mirrors patrol -- exact tie
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(probable_threat_radius=3))
    gs = _game_state(board, [safe_cruiser, patrol, enemy_cruiser, enemy_patrol], config=config)
    goal = TaskForceGoal(GoalKind.CAPTURE_PORT, target=patrol.position)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2}, goal=goal)
    pace = _compute_force_pace(gs, [force])
    visible = {3: enemy_cruiser, 4: enemy_patrol}

    model = _empty_enemy_model(gs)
    pool = model._pool_for(ShipKind.BATTLESHIP)
    pool.count = 1
    pool.field.set_point_mass(patrol.position, 1.0)

    strategy, should_retreat = _pick_strategy_for_force(force, gs, PLAYER_A, visible, [force], pace, model)

    assert strategy == Strategy.RETREAT
    assert should_retreat is True

    # Without the injected belief, the same scenario has nothing exposing
    # patrol at all -- ADVANCE (or another non-retreat pick) should win.
    strategy_no_belief, _ = _pick_strategy_for_force(
        force, gs, PLAYER_A, visible, [force], pace, _empty_enemy_model(gs)
    )
    assert strategy_no_belief != Strategy.RETREAT


def test_probable_threat_engagement_disabled_is_a_pure_no_op():
    # Regression pin: with the flag off, injecting belief mass that would
    # otherwise clearly trigger a probable threat changes nothing.
    board = _sea_board(radius=20)
    ship = _ship(AxialCoord(0, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)
    weak_visible = _ship(AxialCoord(15, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(probable_threat_engagement_enabled=False))
    gs = _game_state(board, [ship, weak_visible], config=config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, ship.position))
    pace = _compute_force_pace(gs, [force])

    model = _empty_enemy_model(gs)
    pool = model._pool_for(ShipKind.BATTLESHIP)
    pool.count = 1
    pool.field.set_point_mass(ship.position, 1.0)

    value = _project_engagement_value(
        force, gs, PLAYER_A, {2: weak_visible}, [force], pace, model, strategy=Strategy.HOLD
    )

    assert value == 0.0  # gained (0) - at_risk (0, flag off) over total_value


# -- re-evaluate strategy on new sightings mid-turn --------------------------
# (see _reevaluate_strategy_on_new_sightings, AiConfig.reevaluate_strategy_
# on_new_sighting)


def _clear_loss_scenario():
    """The exact scenario from test_pick_strategy_for_force_retreats_when_
    only_retreating_avoids_a_clear_loss: group power an exact tie (so the
    cheap gate doesn't short-circuit), but the patrol boat sits exposed to
    a one-shot kill only a real retreat can escape. Reused here as the
    "newly revealed enemy that should flip the verdict" half of a mid-turn
    re-evaluation."""
    board = _sea_board(radius=35)
    home_port = AxialCoord(30, 0)
    board.tiles[home_port] = Tile(coord=home_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    cruiser = _ship(AxialCoord(25, 0), ShipKind.CRUISER, PLAYER_A, 1)
    patrol = _ship(AxialCoord(2, 0), ShipKind.PATROL_BOAT, PLAYER_A, 2, hp=1)
    enemy = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_B, 3, hp=12)
    gs = _game_state(board, [cruiser, patrol, enemy], config=Config(fow=FowConfig(enabled=False)))
    goal = TaskForceGoal(GoalKind.CAPTURE_PORT, target=patrol.position)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2}, goal=goal)
    return gs, force, enemy, home_port, patrol


def test_reevaluate_strategy_on_new_sightings_flips_advance_to_retreat():
    gs, force, enemy, home_port, patrol = _clear_loss_scenario()
    pace = _compute_force_pace(gs, [force])
    forces = [force]
    # As of the original (now-stale) pick, nothing was visible yet -- the
    # force defaulted to ADVANCE, same as game61's own turn 6.
    strategy_by_force = {force.id: Strategy.ADVANCE}
    should_retreat_by_force = {force.id: False}
    seen_enemy_ids: set[int] = set()

    _reevaluate_strategy_on_new_sightings(
        gs, PLAYER_A, forces, pace, _empty_enemy_model(gs), seen_enemy_ids, strategy_by_force, should_retreat_by_force
    )

    assert seen_enemy_ids == {enemy.id}
    assert strategy_by_force[force.id] == Strategy.RETREAT
    assert should_retreat_by_force[force.id] is True
    assert force.retreating is True

    # The core guarantee: a ship whose own per-ship decision hasn't run
    # yet this turn now sees the updated force.retreating live, with no
    # other code changes -- it takes the retreat path, not the stale
    # ADVANCE one.
    destination = _choose_destination(patrol, gs, PLAYER_A, {3: enemy}, forces, pace)
    assert destination != enemy.position
    assert distance(destination, home_port) < distance(patrol.position, home_port)


def test_reevaluate_strategy_on_new_sightings_is_a_noop_without_new_ids():
    gs, force, enemy, home_port, patrol = _clear_loss_scenario()
    pace = _compute_force_pace(gs, [force])
    forces = [force]
    strategy_by_force = {force.id: Strategy.ADVANCE}
    should_retreat_by_force = {force.id: False}
    seen_enemy_ids = {enemy.id}  # already accounted for -- nothing new this call

    _reevaluate_strategy_on_new_sightings(
        gs, PLAYER_A, forces, pace, _empty_enemy_model(gs), seen_enemy_ids, strategy_by_force, should_retreat_by_force
    )

    assert strategy_by_force[force.id] == Strategy.ADVANCE  # untouched
    assert should_retreat_by_force[force.id] is False
    assert force.retreating is False


def test_reevaluate_strategy_on_new_sightings_leaves_an_already_retreating_force_alone():
    gs, force, enemy, home_port, patrol = _clear_loss_scenario()
    pace = _compute_force_pace(gs, [force])
    forces = [force]
    # Simulate: this force already retreated once this same turn (e.g. via
    # the original, top-of-turn strategy pick).
    force.retreating = True
    force.retreat_turns = 0
    strategy_by_force = {force.id: Strategy.RETREAT}
    should_retreat_by_force = {force.id: True}
    seen_enemy_ids: set[int] = set()  # the enemy is "new" to this tracker

    _reevaluate_strategy_on_new_sightings(
        gs, PLAYER_A, forces, pace, _empty_enemy_model(gs), seen_enemy_ids, strategy_by_force, should_retreat_by_force
    )

    # Still excluded from re-evaluation -- update_task_force_stance's own
    # "already retreating" branch was never re-entered for it, so
    # retreat_turns wasn't double-incremented and its dict entries are
    # untouched (nothing to re-decide for an already-fleeing force).
    assert force.retreat_turns == 0
    assert force.id not in strategy_by_force or strategy_by_force[force.id] == Strategy.RETREAT
    # seen_enemy_ids itself still updates -- the sighting is real, even if
    # this particular force had nothing left to react to.
    assert seen_enemy_ids == {enemy.id}


def test_reevaluate_strategy_on_new_sighting_disabled_never_runs():
    # Regression pin at the plan_movement level: with the flag off (the
    # default), a force never reconsiders mid-turn even when a clear-loss
    # enemy becomes visible only after the original pick -- the exact
    # stale-snapshot behavior this feature replaces when enabled.
    board = _sea_board(radius=35)
    home_port = AxialCoord(30, 0)
    board.tiles[home_port] = Tile(coord=home_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(reevaluate_strategy_on_new_sighting=False))
    assert config.ai.reevaluate_strategy_on_new_sighting is False
    cruiser = _ship(AxialCoord(25, 0), ShipKind.CRUISER, PLAYER_A, 1)
    patrol = _ship(AxialCoord(2, 0), ShipKind.PATROL_BOAT, PLAYER_A, 2, hp=1)
    enemy = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_B, 3, hp=12)
    gs = _game_state(board, [cruiser, patrol, enemy], config=config)
    goal = TaskForceGoal(GoalKind.CAPTURE_PORT, target=patrol.position)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2}, goal=goal)
    pace = _compute_force_pace(gs, [force])

    # Nothing calls _reevaluate_strategy_on_new_sightings at all when the
    # flag is off -- force.retreating simply never changes on its own.
    assert force.retreating is False
    destination = _choose_destination(patrol, gs, PLAYER_A, {3: enemy}, [force], pace)
    # Stale ADVANCE behavior: the patrol boat is still advancing toward its
    # goal (own position -- so at most a one-hex drift, per the ordinary
    # non-RETREAT pipeline; see choose_task_force_destination), never the
    # RETREAT path's real flight toward home_port.
    assert destination is not None
    assert distance(destination, home_port) >= distance(patrol.position, home_port)
