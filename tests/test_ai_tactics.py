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
    # them == 1, so decide_battle is never even consulted -- no matter how
    # damaged the attacker already is (as long as it survives the patrol
    # boat's own round-1 return fire, dmg 1 -- hp=2 clears that with 1 to
    # spare).
    board = _sea_board()
    attacker = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1, hp=2)  # 2/12 hp -- very low
    target = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    gs = _game_state(board, [attacker, target])

    assert secures_kill(attacker, target, gs) is True


def test_secures_kill_always_completes_a_strictly_favorable_multi_round_fight():
    # battleship (dmg 4, hp 12 max) vs. cruiser (hp 8, dmg 3): rounds_to_
    # kill_them == 2, so decide_battle IS consulted once, after round 1 --
    # but with the old HP-floor retreat removed, a strictly-favorable race
    # (matchup_score > 0) always secures the kill regardless of how low
    # that leaves the attacker. hp=7 here ends up well below where the
    # old 0.34 floor used to cut this exact fight short (4/12 after round
    # 1, under the old 4.08 floor) -- now it still secures.
    board = _sea_board()
    target = _ship(AxialCoord(1, 0), ShipKind.CRUISER, PLAYER_B, 2)
    attacker = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1, hp=7)
    gs = _game_state(board, [attacker, target])
    assert matchup_score(attacker, target, gs) > 0  # confirm it's a clean win, not a tie

    assert secures_kill(attacker, target, gs) is True


def test_secures_kill_is_false_on_a_non_worthwhile_tie():
    board = _sea_board()
    a = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    b = _ship(AxialCoord(1, 0), ShipKind.BATTLESHIP, PLAYER_B, 2)
    gs = _game_state(board, [a, b])
    assert matchup_score(a, b, gs) == 0  # an identical mirror matchup -- an exact tie

    assert secures_kill(a, b, gs) is False  # equal cost -- not worth trading evenly


def test_secures_kill_is_true_on_a_worthwhile_tie():
    # Cruiser (cost 7) vs battleship (cost 10) at HP levels that tie the
    # race -- worth taking even at 1-for-1 (a real replay-found gap,
    # game65: cruisers declining exactly this kind of trade against enemy
    # carriers).
    board = _sea_board()
    attacker = _ship(AxialCoord(0, 0), ShipKind.CRUISER, PLAYER_A, 1, hp=4)
    target = _ship(AxialCoord(1, 0), ShipKind.BATTLESHIP, PLAYER_B, 2, hp=3)
    gs = _game_state(board, [attacker, target])
    assert matchup_score(attacker, target, gs) == 0  # confirm this really is a tie

    assert secures_kill(attacker, target, gs) is True


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


def test_would_secure_kill_finds_the_candidate_that_secures_it():
    # Neither ship needs to be "the best matchup" to be picked -- would_
    # secure_kill just needs to find one that actually secures the kill
    # (see secures_kill): the patrol boat (cost 1) is a clear losing race
    # against a battleship, but the cruiser (cost 7) ties it at HP levels
    # that make the trade worthwhile (target cost 10 > 7).
    board = _sea_board()
    target = _ship(AxialCoord(1, 0), ShipKind.BATTLESHIP, PLAYER_B, 3, hp=3)
    patrol = _ship(AxialCoord(2, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)
    cruiser = _ship(AxialCoord(0, 0), ShipKind.CRUISER, PLAYER_A, 2, hp=4)
    gs = _game_state(board, [patrol, cruiser, target])

    assert secures_kill(patrol, target, gs) is False
    assert secures_kill(cruiser, target, gs) is True

    assert would_secure_kill(target, [patrol, cruiser], gs) == [cruiser]


def test_would_secure_kill_returns_none_when_nobody_can_secure_it():
    board = _sea_board()
    patrol = _ship(AxialCoord(0, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)
    target = _ship(AxialCoord(1, 0), ShipKind.BATTLESHIP, PLAYER_B, 3)
    gs = _game_state(board, [patrol, target])

    assert secures_kill(patrol, target, gs) is False
    assert would_secure_kill(target, [patrol], gs) is None


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


def test_secure_kills_pass_commits_a_securable_kill_ahead_of_the_ordinary_loop():
    # A one-shot kill any single ship can secure alone -- still claimed
    # *here*, ahead of the ordinary per-ship loop, rather than left for it
    # to find independently: with would_secure_kill now a single-
    # candidate search (see its own docstring for why genuine multi-ship
    # chaining no longer applies), this pass's remaining value is
    # claiming/prioritizing securable kills before ships start picking
    # their own targets independently, not just coordinating cases no
    # single ship could handle.
    board = _sea_board()
    attacker = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    target = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    gs = _game_state(board, [attacker, target])
    policy = NaivePolicy()
    apply_attack = _naive_apply_attack(gs, policy.decide_battle)

    claimed = _drain(secure_kills_pass(gs, PLAYER_A, {2: target}, gs.config.ai, apply_attack))

    assert claimed == {1}
    assert 2 not in gs.ships  # the target is sunk


def test_secure_kills_pass_is_a_no_op_when_tactics_disabled():
    board = _sea_board()
    attacker = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    target = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    gs = _game_state(board, [attacker, target], config=Config(ai=AiConfig(tactics_enabled=False)))
    policy = NaivePolicy()
    apply_attack = _naive_apply_attack(gs, policy.decide_battle)

    claimed = _drain(secure_kills_pass(gs, PLAYER_A, {2: target}, gs.config.ai, apply_attack))

    assert claimed == frozenset()
    assert 2 in gs.ships  # untouched


def test_secure_kills_pass_prioritizes_the_higher_value_target_first():
    # One battleship can reach and secure either a patrol boat (cost 1,
    # damage 1) or a cruiser (cost 7, damage 3) -- it only gets to attack
    # once this turn, so the pass should commit it to the higher future-
    # damage-removed target (the cruiser), leaving the patrol boat alone,
    # rather than whichever happens to be considered first.
    board = _sea_board()
    attacker = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    weak = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    juicier = _ship(AxialCoord(0, 1), ShipKind.CRUISER, PLAYER_B, 3)
    gs = _game_state(board, [attacker, weak, juicier])
    policy = NaivePolicy()
    apply_attack = _naive_apply_attack(gs, policy.decide_battle)

    claimed = _drain(secure_kills_pass(gs, PLAYER_A, {2: weak, 3: juicier}, gs.config.ai, apply_attack))

    assert claimed == {1}
    assert 2 in gs.ships  # the patrol boat was left alone
    assert 3 not in gs.ships  # the cruiser was sunk instead
