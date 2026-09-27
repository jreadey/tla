from dataclasses import replace

from tla import movement
from tla.ai.policy import NaivePolicy
from tla.ai.scoring import matchup_score
from tla.ai.tactics import action_value, secure_kills_pass, secures_kill, would_secure_kill
from tla.battle import run_battle
from tla.board import Board
from tla.config import AiConfig, Config, ShipStatsConfig
from tla.game_state import GameState
from tla.hexgrid import AxialCoord, hexes_in_range
from tla.movement import shortest_path
from tla.ship import Ship, ShipKind
from tla.tile import PLAYER_A, PLAYER_B, Tile, TerrainType


def _sea_board(radius: int = 6) -> Board:
    board = Board(width=radius * 2 + 1, height=radius * 2 + 1)
    for coord in hexes_in_range(AxialCoord(0, 0), radius):
        board.tiles[coord] = Tile(coord=coord, terrain=TerrainType.SEA)
    return board


def _ship(coord: AxialCoord, kind: ShipKind, owner, ship_id: int, hp: int | None = None) -> Ship:
    stats = Config().ship_stats.stats[kind]
    return Ship(
        id=ship_id, kind=kind, owner=owner, position=coord,
        current_hp=hp if hp is not None else stats.hp, movement_remaining=stats.movement,
    )


def _game_state(board: Board, ships: list[Ship], config: Config | None = None) -> GameState:
    return GameState(config=config or Config(), board=board, ships={s.id: s for s in ships})


def _drain(gen):
    """Exhaust a generator and return its `return` value -- `list(gen)`
    only collects the yielded values, not the `StopIteration.value`."""
    try:
        while True:
            next(gen)
    except StopIteration as exc:
        return exc.value


# -- secures_kill -------------------------------------------------------


def test_secures_kill_ignores_own_hp_for_a_one_round_kill():
    # battleship (dmg 4) one-shots a patrol boat (hp 2): rounds_to_kill_
    # them == 1, so decide_battle is never even consulted -- the floor
    # doesn't matter, no matter how damaged the attacker already is (as
    # long as it survives the patrol boat's own round-1 return fire, dmg
    # 1 -- hp=2 clears that with 1 to spare, well under the 34% floor).
    board = _sea_board()
    attacker = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1, hp=2)  # 2/12 hp -- way below the floor
    target = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    gs = _game_state(board, [attacker, target])

    assert secures_kill(attacker, target, gs) is True


def test_secures_kill_respects_the_hp_floor_on_a_multi_round_fight():
    # battleship (dmg 4, hp 12) vs. cruiser (hp 8, dmg 3): rounds_to_kill_
    # them == 2, so decide_battle IS consulted once, after round 1.
    board = _sea_board()
    target = _ship(AxialCoord(1, 0), ShipKind.CRUISER, PLAYER_B, 2)

    full_health = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1, hp=12)
    gs = _game_state(board, [full_health, target])
    assert secures_kill(full_health, target, gs) is True  # 12-3=9 >= floor (4.08)

    already_damaged = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1, hp=5)
    gs2 = _game_state(board, [already_damaged, _ship(AxialCoord(1, 0), ShipKind.CRUISER, PLAYER_B, 2)])
    assert secures_kill(already_damaged, gs2.ships[2], gs2) is False  # 5-3=2 < floor (4.08)


def test_secures_kill_is_false_on_a_tie_even_though_the_race_is_even():
    board = _sea_board()
    a = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    b = _ship(AxialCoord(1, 0), ShipKind.BATTLESHIP, PLAYER_B, 2)
    gs = _game_state(board, [a, b])

    assert secures_kill(a, b, gs) is False  # a tie is a mutual kill, not a secured one


def test_secures_kill_is_false_when_the_attacker_cannot_hurt_the_target():
    # A submerged submarine can only be hurt by asw -- zero it out on the
    # attacker so it genuinely can't touch the target at all.
    stats = dict(Config().ship_stats.stats)
    stats[ShipKind.PATROL_BOAT] = replace(stats[ShipKind.PATROL_BOAT], asw=0)
    config = Config(ship_stats=ShipStatsConfig(stats=stats))
    board = _sea_board()
    attacker = _ship(AxialCoord(0, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)
    submerged_sub = _ship(AxialCoord(1, 0), ShipKind.SUBMARINE, PLAYER_B, 2)
    submerged_sub.surfaced = False
    gs = _game_state(board, [attacker, submerged_sub], config=config)

    assert secures_kill(attacker, submerged_sub, gs) is False


# -- would_secure_kill ----------------------------------------------------


def test_would_secure_kill_chains_two_attackers_neither_can_solo():
    board = _sea_board()
    # battleship, already damaged to 5/12 hp -- can't solo-secure the
    # destroyer target (floor cuts it short after round 1), but still
    # lands one round of chip damage (4) before retreating.
    ship1 = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1, hp=5)
    # destroyer vs. the ORIGINAL full-hp destroyer target is an exact tie
    # (rejected solo) -- but against the chipped-down remainder it's a
    # clean one-round kill.
    ship2 = _ship(AxialCoord(2, 0), ShipKind.DESTROYER, PLAYER_A, 2)
    target = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_B, 3)
    gs = _game_state(board, [ship1, ship2, target])

    assert secures_kill(ship1, target, gs) is False
    assert secures_kill(ship2, target, gs) is False  # a tie against the full-hp target

    prefix = would_secure_kill(target, [ship1, ship2], gs)

    assert prefix == [ship1, ship2]


def test_would_secure_kill_returns_none_when_the_group_still_cannot_finish_it():
    board = _sea_board()
    ship1 = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1, hp=5)
    target = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_B, 3)
    gs = _game_state(board, [ship1, target])

    # Alone, ship1 only chips 4 off a 6-hp target (2 left) -- with no
    # second attacker in the list, that's not a kill.
    assert would_secure_kill(target, [ship1], gs) is None


# -- action_value -----------------------------------------------------------


def test_action_value_adds_a_kill_bonus_weighted_by_the_targets_damage_stat():
    board = _sea_board()
    attacker = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)  # full hp
    target = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)  # one-shot kill
    gs = _game_state(board, [attacker, target])
    ai_config = AiConfig(tactics_kill_weight=1.0)

    base = matchup_score(attacker, target, gs)
    assert secures_kill(attacker, target, gs) is True

    value = action_value(attacker, target, gs, ai_config)

    patrol_boat_damage = gs.config.ship_stats.stats[ShipKind.PATROL_BOAT].damage
    assert value == base + 1.0 * patrol_boat_damage


def test_action_value_matches_matchup_score_when_tactics_disabled():
    board = _sea_board()
    attacker = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    target = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    gs = _game_state(board, [attacker, target])
    ai_config = AiConfig(tactics_enabled=False)

    assert action_value(attacker, target, gs, ai_config) == matchup_score(attacker, target, gs)


# -- secure_kills_pass ------------------------------------------------------


def _naive_apply_attack(game_state: GameState, decision_fn):
    def apply(ship: Ship, attack_hex: AxialCoord) -> None:
        path = shortest_path(ship, attack_hex, game_state)
        defender = movement.begin_engagement(ship, path, game_state)
        run_battle(ship, defender, game_state, decision_fn=decision_fn)

    return apply


def test_secure_kills_pass_commits_a_joint_kill_neither_ship_can_solo():
    board = _sea_board()
    ship1 = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1, hp=5)
    ship2 = _ship(AxialCoord(2, 0), ShipKind.DESTROYER, PLAYER_A, 2)
    target = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_B, 3)
    gs = _game_state(board, [ship1, ship2, target])
    policy = NaivePolicy()
    apply_attack = _naive_apply_attack(gs, policy.decide_battle)

    claimed = _drain(secure_kills_pass(gs, PLAYER_A, {3: target}, gs.config.ai, apply_attack))

    assert claimed == {1, 2}
    assert 3 not in gs.ships  # the target is sunk


def test_secure_kills_pass_is_a_no_op_when_tactics_disabled():
    board = _sea_board()
    ship1 = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1, hp=5)
    ship2 = _ship(AxialCoord(2, 0), ShipKind.DESTROYER, PLAYER_A, 2)
    target = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_B, 3)
    gs = _game_state(board, [ship1, ship2, target], config=Config(ai=AiConfig(tactics_enabled=False)))
    policy = NaivePolicy()
    apply_attack = _naive_apply_attack(gs, policy.decide_battle)

    claimed = _drain(secure_kills_pass(gs, PLAYER_A, {3: target}, gs.config.ai, apply_attack))

    assert claimed == frozenset()
    assert 3 in gs.ships  # untouched


def test_secure_kills_pass_leaves_a_solo_securable_target_to_the_ordinary_loop():
    board = _sea_board()
    # A one-shot kill any single ship already handles -- not this pass's job.
    lone_attacker = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    weak_target = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    gs = _game_state(board, [lone_attacker, weak_target])
    policy = NaivePolicy()
    apply_attack = _naive_apply_attack(gs, policy.decide_battle)

    claimed = _drain(secure_kills_pass(gs, PLAYER_A, {2: weak_target}, gs.config.ai, apply_attack))

    assert claimed == frozenset()
    assert 2 in gs.ships  # left alone for the ordinary per-ship loop to finish off
