from tla.ai.enemy_model import EnemyModel
from tla.ai.move_scoring import plan_force_movement_scored, score_move
from tla.ai.tactics import action_value
from tla.ai.task_force import GoalKind, TaskForce, TaskForceGoal, sea_distance_field
from tla.board import Board
from tla.config import AiConfig, Config, FleetConfig, FowConfig
from tla.game_state import GameState
from tla.hexgrid import AxialCoord, distance, hexes_in_range
from tla.ship import Ship, ShipKind
from tla.tile import PLAYER_A, PLAYER_B, Tile, TerrainType


def _sea_board(radius: int = 10) -> Board:
    board = Board(width=radius * 2 + 1, height=radius * 2 + 1)
    for coord in hexes_in_range(AxialCoord(0, 0), radius):
        board.tiles[coord] = Tile(coord=coord, terrain=TerrainType.SEA)
    return board


def _ship(
    coord: AxialCoord, kind: ShipKind, owner, ship_id: int, hp: int | None = None, surfaced: bool = True
) -> Ship:
    stats = Config().ship_stats.stats[kind]
    return Ship(
        id=ship_id, kind=kind, owner=owner, position=coord,
        current_hp=hp if hp is not None else stats.hp, surfaced=surfaced, movement_remaining=stats.movement,
    )


def _config(ai: AiConfig | None = None) -> Config:
    """FowConfig disabled (every test scenario's own visible_enemies dict
    is explicit, not FOW-derived) and FleetConfig zeroed -- a naively
    default FleetConfig would otherwise seed EnemyModel with real believed
    mass for the opponent's whole unplaced starting fleet, contaminating
    the exposure/vision terms with phantom "unseen ship" pool mass that
    has nothing to do with what a given scenario actually placed on the
    board. Same reasoning test_ai_naive.py's own _empty_enemy_model
    exists for."""
    return Config(fow=FowConfig(enabled=False), fleet=FleetConfig(counts={}), ai=ai or AiConfig())


def _game_state(board: Board, ships: list[Ship], config: Config | None = None) -> GameState:
    return GameState(config=config or _config(), board=board, ships={s.id: s for s in ships})


def _zeroed_ai_config(**overrides) -> AiConfig:
    """Every move_score_* weight at 0 except whatever `overrides` sets --
    isolates a single scoring term for a test, same reasoning
    `test_ai_scoring.py` isolates matchup_score's own sub-terms."""
    fields = {
        "move_score_goal_weight": 0.0,
        "move_score_attack_weight": 0.0,
        "move_score_screen_weight": 0.0,
        "move_score_cohesion_weight": 0.0,
        "move_score_exposure_weight": 0.0,
        "move_score_vision_weight": 0.0,
        "move_score_rearguard_weight": 0.0,
    }
    fields.update(overrides)
    return AiConfig(**fields)


# -- score_move: attack term ------------------------------------------------


def test_score_move_attack_term_equals_action_value_at_an_enemy_hex():
    board = _sea_board()
    config = _config(ai=_zeroed_ai_config(move_score_attack_weight=1.0))
    attacker = _ship(AxialCoord(0, 0), ShipKind.CRUISER, PLAYER_A, 1)
    target = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_B, 2, hp=2)
    gs = _game_state(board, [attacker, target], config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)

    score = score_move(attacker, target.position, gs, force, {2: target}, model, None, gs.config.ai)

    assert score == action_value(attacker, target, gs, gs.config.ai)


def test_score_move_attack_term_is_zero_at_an_empty_hex():
    board = _sea_board()
    config = _config(ai=_zeroed_ai_config(move_score_attack_weight=1.0))
    ship = _ship(AxialCoord(0, 0), ShipKind.CRUISER, PLAYER_A, 1)
    gs = _game_state(board, [ship], config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)

    assert score_move(ship, AxialCoord(1, 0), gs, force, {}, model, None, gs.config.ai) == 0.0


# -- score_move: the CA-vs-screening-vs-a-securable-kill case ---------------


def test_score_move_prefers_a_securable_kill_over_screening_the_carrier():
    # game65/game66-shaped: a cruiser could either stand between its own
    # carrier and a visible battleship threat, or attack a weak destroyer
    # it can solo-secure -- the exact scenario that motivated this whole
    # redesign (screening used to win unconditionally by being checked
    # first in the old procedural chain, regardless of what else was on
    # the table).
    board = _sea_board()
    config = _config(ai=AiConfig())
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    cruiser = _ship(AxialCoord(2, 0), ShipKind.CRUISER, PLAYER_A, 2)
    threat = _ship(AxialCoord(4, 0), ShipKind.BATTLESHIP, PLAYER_B, 3)
    weak_target = _ship(AxialCoord(2, -2), ShipKind.DESTROYER, PLAYER_B, 4, hp=2)
    gs = _game_state(board, [carrier, cruiser, threat, weak_target], config)
    force = TaskForce(
        id=1, owner=PLAYER_A, member_ids={1, 2}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, AxialCoord(8, 0))
    )
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)
    visible = {3: threat, 4: weak_target}
    goal_field = sea_distance_field(gs, force.goal.target)

    screen_hex = AxialCoord(3, 0)  # genuinely between the carrier and the threat
    attack_hex = weak_target.position

    screen_score = score_move(cruiser, screen_hex, gs, force, visible, model, goal_field, gs.config.ai)
    attack_score = score_move(cruiser, attack_hex, gs, force, visible, model, goal_field, gs.config.ai)

    assert attack_score > screen_score


# -- score_move: exposure -----------------------------------------------


def test_score_move_exposure_weighs_a_carrier_more_than_a_destroyer():
    board = _sea_board()
    config = _config(ai=_zeroed_ai_config(move_score_exposure_weight=1.0))
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    destroyer = _ship(AxialCoord(0, 1), ShipKind.DESTROYER, PLAYER_A, 2)
    enemy_bb = _ship(AxialCoord(5, 0), ShipKind.BATTLESHIP, PLAYER_B, 3)
    gs = _game_state(board, [carrier, destroyer, enemy_bb], config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)
    model.observe_ship_state(3, ShipKind.BATTLESHIP, enemy_bb.position, enemy_bb.current_hp, gs)
    risky_hex = AxialCoord(4, 0)

    carrier_score = score_move(carrier, risky_hex, gs, force, {3: enemy_bb}, model, None, gs.config.ai)
    destroyer_score = score_move(destroyer, risky_hex, gs, force, {3: enemy_bb}, model, None, gs.config.ai)

    assert carrier_score < destroyer_score  # same risk, but the carrier's own loss would cost more


def test_score_move_exposure_does_not_double_count_the_target_being_attacked():
    # A real bug caught via a full plan_movement integration run: scoring
    # an attack hex used to fold the target's own believed damage into
    # the exposure penalty, as if it would still be standing there
    # hitting back next turn -- when landing the attack is exactly what
    # resolves it. Confirmed via the actual numbers: a cruiser (cost 7)
    # attacking a destroyer (damage 2) it can solo-secure should NOT pay
    # a -14 (2 * 7) exposure penalty on top of the attack's own value.
    board = _sea_board()
    config = _config(ai=_zeroed_ai_config(move_score_exposure_weight=1.0))
    cruiser = _ship(AxialCoord(0, 0), ShipKind.CRUISER, PLAYER_A, 1)
    target = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_B, 2, hp=2)
    gs = _game_state(board, [cruiser, target], config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)
    model.observe_ship_state(2, ShipKind.DESTROYER, target.position, target.current_hp, gs)

    score = score_move(cruiser, target.position, gs, force, {2: target}, model, None, gs.config.ai)

    assert score == 0.0  # the target's own damage stat is exactly what got subtracted out


def test_score_move_exposure_still_counts_a_bystander_near_an_attack_hex():
    # The fix above only excludes the specific ship being attacked --
    # some other, still-alive visible threat near the same hex should
    # still count against it.
    board = _sea_board()
    config = _config(ai=_zeroed_ai_config(move_score_exposure_weight=1.0))
    cruiser = _ship(AxialCoord(0, 0), ShipKind.CRUISER, PLAYER_A, 1)
    target = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_B, 2, hp=2)
    bystander = _ship(AxialCoord(1, 1), ShipKind.BATTLESHIP, PLAYER_B, 3)
    gs = _game_state(board, [cruiser, target, bystander], config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)
    model.observe_ship_state(2, ShipKind.DESTROYER, target.position, target.current_hp, gs)
    model.observe_ship_state(3, ShipKind.BATTLESHIP, bystander.position, bystander.current_hp, gs)

    score = score_move(cruiser, target.position, gs, force, {2: target, 3: bystander}, model, None, gs.config.ai)

    assert score < 0.0  # the bystander battleship's own damage still counts


def test_score_move_exposure_is_zero_when_the_ship_dies_in_a_worthwhile_tie():
    # Real example: a submerged submarine (hp 3) attacking a full-hp
    # battleship is an exact tie (matchup_score == 0) that's worth taking
    # (scoring.worth_a_tie -- a battleship is worth far more than the
    # submarine). This engine's combat is deterministic, so a tie always
    # means mutual destruction -- the submarine is dead after this exact
    # move, so a second battleship sitting right next to the first has no
    # future submarine left to threaten. Exposure must be exactly 0, not
    # just reduced.
    board = _sea_board()
    config = _config(ai=_zeroed_ai_config(move_score_exposure_weight=1.0))
    sub = _ship(AxialCoord(0, 0), ShipKind.SUBMARINE, PLAYER_A, 1, hp=3, surfaced=False)
    target = _ship(AxialCoord(1, 0), ShipKind.BATTLESHIP, PLAYER_B, 2)
    bystander = _ship(AxialCoord(2, 0), ShipKind.BATTLESHIP, PLAYER_B, 3)
    gs = _game_state(board, [sub, target, bystander], config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)
    model.observe_ship_state(2, ShipKind.BATTLESHIP, target.position, target.current_hp, gs)
    model.observe_ship_state(3, ShipKind.BATTLESHIP, bystander.position, bystander.current_hp, gs)

    score = score_move(sub, target.position, gs, force, {2: target, 3: bystander}, model, None, gs.config.ai)

    assert score == 0.0


def test_score_move_exposure_nets_against_the_value_secured_by_a_clean_win():
    # Same shape, but the submarine has full HP: a clean win
    # (matchup_score > 0), so it survives at low HP -- a second battleship
    # nearby is a real future risk this time, but it should be discounted
    # by the value this exact attack already banked (tactics.action_value
    # for sinking the first battleship), not charged in full on top of it.
    board = _sea_board()
    config = _config(ai=_zeroed_ai_config(move_score_exposure_weight=1.0))
    sub = _ship(AxialCoord(0, 0), ShipKind.SUBMARINE, PLAYER_A, 1, surfaced=False)
    target = _ship(AxialCoord(1, 0), ShipKind.BATTLESHIP, PLAYER_B, 2)
    bystander = _ship(AxialCoord(2, 0), ShipKind.BATTLESHIP, PLAYER_B, 3)
    gs = _game_state(board, [sub, target, bystander], config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)
    model.observe_ship_state(2, ShipKind.BATTLESHIP, target.position, target.current_hp, gs)
    model.observe_ship_state(3, ShipKind.BATTLESHIP, bystander.position, bystander.current_hp, gs)
    secured = action_value(sub, target, gs, gs.config.ai)

    score = score_move(sub, target.position, gs, force, {2: target, 3: bystander}, model, None, gs.config.ai)

    # Raw exposure (bystander's damage(4) * submarine's cost(4)) is -16;
    # netted against the value just secured, it should land at
    # -(16 - secured), strictly better than the un-netted -16.
    assert score == -(16.0 - secured)
    assert score > -16.0


# -- score_move: vision -------------------------------------------------


def test_score_move_vision_term_only_applies_to_a_carrier():
    board = _sea_board()
    config = _config(ai=_zeroed_ai_config(move_score_vision_weight=1.0))
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    destroyer = _ship(AxialCoord(0, 1), ShipKind.DESTROYER, PLAYER_A, 2)
    gs = _game_state(board, [carrier, destroyer], config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 2})
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)

    assert score_move(carrier, AxialCoord(1, 0), gs, force, {}, model, None, gs.config.ai) > 0.0
    assert score_move(destroyer, AxialCoord(1, 1), gs, force, {}, model, None, gs.config.ai) == 0.0


def test_score_move_vision_term_is_suppressed_near_believed_submarine_mass():
    board = _sea_board()
    config = _config(ai=_zeroed_ai_config(move_score_vision_weight=1.0, move_score_sub_threat_threshold=0.15))
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    enemy_sub = _ship(AxialCoord(3, 0), ShipKind.SUBMARINE, PLAYER_B, 2)
    gs = _game_state(board, [carrier, enemy_sub], config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1})
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)
    # A direct sighting collapses belief to certainty at the sub's own
    # hex -- mass_near a nearby destination should clear the threshold.
    model.observe_ship_state(2, ShipKind.SUBMARINE, enemy_sub.position, enemy_sub.current_hp, gs, surfaced=True)

    safe_score = score_move(carrier, AxialCoord(-2, 0), gs, force, {}, model, None, gs.config.ai)
    risky_score = score_move(carrier, enemy_sub.position, gs, force, {2: enemy_sub}, model, None, gs.config.ai)

    assert safe_score > 0.0
    assert risky_score == 0.0


# -- plan_force_movement_scored ------------------------------------------


def test_plan_force_movement_scored_moves_every_living_member_exactly_once():
    board = _sea_board()
    config = _config(ai=AiConfig())
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    cruiser = _ship(AxialCoord(1, 0), ShipKind.CRUISER, PLAYER_A, 2)
    gs = _game_state(board, [carrier, cruiser], config)
    force = TaskForce(
        id=1, owner=PLAYER_A, member_ids={1, 2}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, AxialCoord(8, 0))
    )
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)
    moved: list[int] = []

    def apply_move(ship: Ship, destination: AxialCoord) -> None:
        moved.append(ship.id)
        ship.position = destination
        ship.movement_remaining = 0

    list(plan_force_movement_scored(gs, PLAYER_A, force, model, gs.config.ai, lambda: {}, apply_move))

    assert sorted(moved) == [1, 2]


def test_plan_force_movement_scored_ignores_a_dead_member_id():
    # A force whose member_ids still references a sunk ship (not yet
    # pruned elsewhere) shouldn't crash or loop -- just skipped.
    board = _sea_board()
    config = _config(ai=AiConfig())
    cruiser = _ship(AxialCoord(0, 0), ShipKind.CRUISER, PLAYER_A, 1)
    gs = _game_state(board, [cruiser], config)
    force = TaskForce(id=1, owner=PLAYER_A, member_ids={1, 99}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, AxialCoord(8, 0)))
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)
    moved: list[int] = []

    def apply_move(ship: Ship, destination: AxialCoord) -> None:
        moved.append(ship.id)
        ship.position = destination
        ship.movement_remaining = 0

    list(plan_force_movement_scored(gs, PLAYER_A, force, model, gs.config.ai, lambda: {}, apply_move))

    assert moved == [1]


def test_plan_force_movement_scored_keeps_a_fast_cheap_ship_with_the_pack():
    # The real bug this phase's whole redesign fixes (game68, seed 431795,
    # turn 2): a patrol boat -- the fastest kind in the game (movement 6,
    # vs. 4 for a battleship) -- used to win the very first iteration of
    # the old single-phase algorithm and run several hexes ahead of the
    # rest of a slower, mixed-speed force, alone, straight into the enemy.
    # `task_force_max_separation` set explicitly (bare AiConfig() defaults
    # it to None, making cohesion a no-op) to match the real config
    # (configs/dev.json) the bug was actually found under.
    board = _sea_board(radius=15)
    config = _config(ai=AiConfig(task_force_max_separation=4))
    battleship = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    patrol_boat = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_A, 2)
    gs = _game_state(board, [battleship, patrol_boat], config)
    force = TaskForce(
        id=1, owner=PLAYER_A, member_ids={1, 2}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, AxialCoord(14, 0))
    )
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)

    def apply_move(ship: Ship, destination: AxialCoord) -> None:
        ship.position = destination
        ship.movement_remaining = 0

    list(plan_force_movement_scored(gs, PLAYER_A, force, model, gs.config.ai, lambda: {}, apply_move))

    assert distance(gs.ships[1].position, gs.ships[2].position) <= gs.config.ai.task_force_max_separation


def _corridor_board(length: int = 10) -> Board:
    """A sea lane exactly one hex wide -- the only way forward or back is
    straight along it, no diagonal detours available at all."""
    board = Board(width=length + 2, height=3)
    for q in range(-1, length + 1):
        board.tiles[AxialCoord(q, 0)] = Tile(coord=AxialCoord(q, 0), terrain=TerrainType.SEA)
    return board


def test_plan_force_movement_scored_reorders_around_a_blocking_escort():
    # A capital ship's own best next step is occupied by a slower,
    # not-yet-moved escort -- rather than stalling there for the rest of
    # the turn, the escort's own turn is resolved first (out of the
    # normal round-robin order), which should clear the way. A one-hex-
    # wide corridor (rather than open sea) rules out the battleship simply
    # detouring around the destroyer diagonally without ever needing it to
    # move -- the only way past is genuinely through where it's sitting.
    board = _corridor_board()
    config = _config(ai=AiConfig())
    battleship = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    destroyer = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_A, 2)  # directly in BB's path to the goal
    gs = _game_state(board, [battleship, destroyer], config)
    force = TaskForce(
        id=1, owner=PLAYER_A, member_ids={1, 2}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, AxialCoord(9, 0))
    )
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)

    def apply_move(ship: Ship, destination: AxialCoord) -> None:
        ship.position = destination
        ship.movement_remaining = 0

    list(plan_force_movement_scored(gs, PLAYER_A, force, model, gs.config.ai, lambda: {}, apply_move))

    # The battleship must have actually advanced past destroyer's own
    # starting hex (1, 0) -- if reordering hadn't kicked in, it would have
    # stalled at (0, 0) for the whole turn, since the corridor leaves it
    # no other way forward at all.
    assert gs.ships[1].position.q > 1


def test_plan_force_movement_scored_breaks_a_reordering_cycle_without_hanging():
    # Two ships each want the other's current hex (a manufactured mutual
    # block) -- must not deadlock or infinitely recurse; falls back to
    # leaving both wherever their own best score already is.
    board = _sea_board(radius=6)
    config = _config(ai=AiConfig())
    ship_a = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    ship_b = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_A, 2)
    gs = _game_state(board, [ship_a, ship_b], config)
    force = TaskForce(
        id=1, owner=PLAYER_A, member_ids={1, 2}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, AxialCoord(8, 0))
    )
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)
    moved: list[int] = []

    def apply_move(ship: Ship, destination: AxialCoord) -> None:
        moved.append(ship.id)
        ship.position = destination
        ship.movement_remaining = 0

    # No assertion on the exact outcome -- just that this terminates at
    # all (a real risk with naive recursive reordering) and still resolves
    # every member exactly once.
    list(plan_force_movement_scored(gs, PLAYER_A, force, model, gs.config.ai, lambda: {}, apply_move))

    assert sorted(moved) == [1, 2]


def test_plan_force_movement_scored_replans_on_a_new_sighting_mid_execution():
    # Confirms the mechanism the user's "re-run the evaluation after any
    # CV step" point depends on: a sighting revealed by one ship's real
    # move causes a fresh Phase-1 plan for the ships that haven't moved
    # yet this call, rather than blindly continuing the stale one built
    # before anything executed. `visible_enemies_fn` starts empty and
    # "reveals" an enemy the moment the first ship's real move executes
    # (a stateful thunk stands in for a real newly-revealed sighting).
    board = _sea_board()
    config = _config(ai=AiConfig())
    ship_a = _ship(AxialCoord(0, 0), ShipKind.CRUISER, PLAYER_A, 1)
    ship_b = _ship(AxialCoord(1, 0), ShipKind.CRUISER, PLAYER_A, 2)
    enemy = _ship(AxialCoord(5, 0), ShipKind.DESTROYER, PLAYER_B, 3)
    gs = _game_state(board, [ship_a, ship_b], config)
    force = TaskForce(
        id=1, owner=PLAYER_A, member_ids={1, 2}, goal=TaskForceGoal(GoalKind.CAPTURE_PORT, AxialCoord(8, 0))
    )
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)
    revealed = False
    plan_calls = 0

    def visible_enemies_fn():
        return {3: enemy} if revealed else {}

    def apply_move(ship: Ship, destination: AxialCoord) -> None:
        nonlocal revealed
        ship.position = destination
        ship.movement_remaining = 0
        revealed = True  # this move "reveals" the enemy from here on

    import tla.ai.move_scoring as move_scoring_module

    original_plan_moves = move_scoring_module._plan_moves

    def counting_plan_moves(*args, **kwargs):
        nonlocal plan_calls
        plan_calls += 1
        return original_plan_moves(*args, **kwargs)

    move_scoring_module._plan_moves = counting_plan_moves
    try:
        list(plan_force_movement_scored(gs, PLAYER_A, force, model, gs.config.ai, visible_enemies_fn, apply_move))
    finally:
        move_scoring_module._plan_moves = original_plan_moves

    assert plan_calls >= 2  # the reveal after ship_a's move forced a fresh plan for ship_b
