import math
from dataclasses import replace

from tla.ai.scoring import best_reachable_attack, matchup_score, nearest_enemy, nearest_uncontrolled_port
from tla.board import Board
from tla.config import Config, ShipStatsConfig
from tla.game_state import GameState
from tla.hexgrid import AxialCoord, hexes_in_range
from tla.ship import Ship, ShipKind
from tla.tile import PLAYER_A, PLAYER_B, Tile, TerrainType


def _sea_board(radius: int = 5) -> Board:
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
    return GameState(config=config or Config(), board=board, ships={s.id: s for s in ships})


def _with_asw_override(kind: ShipKind, asw: int) -> Config:
    stats = dict(Config().ship_stats.stats)
    stats[kind] = replace(stats[kind], asw=asw)
    return Config(ship_stats=ShipStatsConfig(stats=stats))


def test_matchup_score_favors_the_faster_killer():
    board = _sea_board()
    # Battleship (dmg 4, hp 12) vs. patrol boat (dmg 1, hp 2): the
    # battleship wins the race easily.
    battleship = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    patrol_boat = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    gs = _game_state(board, [battleship, patrol_boat])

    assert matchup_score(battleship, patrol_boat, gs) > 0
    assert matchup_score(patrol_boat, battleship, gs) < 0


def test_matchup_score_zero_is_a_tie():
    board = _sea_board()
    a = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    b = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_B, 2)
    gs = _game_state(board, [a, b])

    assert matchup_score(a, b, gs) == 0


def test_matchup_score_uses_asw_against_a_submerged_submarine():
    board = _sea_board()
    # Padded to 10 hp so the asw (2) vs. damage (3) difference actually
    # changes the round count needed to kill it (ceil(10/2)=5 vs.
    # ceil(10/3)=4) -- at the sub's real 4 hp, both round up to the same
    # 2 rounds and the distinction wouldn't show up in the score at all.
    cruiser = _ship(AxialCoord(0, 0), ShipKind.CRUISER, PLAYER_A, 1)  # asw 2, damage 3
    sub = _ship(AxialCoord(1, 0), ShipKind.SUBMARINE, PLAYER_B, 2, hp=10, surfaced=False)
    gs = _game_state(board, [cruiser, sub])

    with_asw = matchup_score(cruiser, sub, gs)
    surfaced_sub = _ship(AxialCoord(1, 0), ShipKind.SUBMARINE, PLAYER_B, 2, hp=10, surfaced=True)
    gs_surfaced = _game_state(board, [cruiser, surfaced_sub])
    with_damage = matchup_score(cruiser, surfaced_sub, gs_surfaced)

    # Cruiser's asw (2) is lower than its damage (3), so it takes longer
    # to kill the submerged sub than it would the surfaced one.
    assert with_asw < with_damage


def test_matchup_score_is_negative_infinity_when_attacker_cannot_damage_defender():
    board = _sea_board()
    config = _with_asw_override(ShipKind.BATTLESHIP, asw=0)
    battleship = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    sub = _ship(AxialCoord(1, 0), ShipKind.SUBMARINE, PLAYER_B, 2, surfaced=False)
    gs = _game_state(board, [battleship, sub], config=config)

    assert matchup_score(battleship, sub, gs) == -math.inf


def test_matchup_score_is_positive_infinity_when_defender_cannot_damage_attacker():
    board = _sea_board()
    # A submerged sub attacking a cruiser whose asw is forced to 0: the sub
    # deals its normal damage (the *defender* here isn't a submerged sub,
    # so the attacker's own submerged status doesn't gate its output), but
    # the cruiser's counter-damage against a submerged target uses its asw
    # stat, which is 0 -- it truly can't scratch the sub back at all.
    config = _with_asw_override(ShipKind.CRUISER, asw=0)
    sub = _ship(AxialCoord(0, 0), ShipKind.SUBMARINE, PLAYER_A, 1, surfaced=False)
    cruiser = _ship(AxialCoord(1, 0), ShipKind.CRUISER, PLAYER_B, 2)
    gs = _game_state(board, [sub, cruiser], config=config)

    assert matchup_score(sub, cruiser, gs) == math.inf


def test_matchup_score_includes_the_carrier_bonus():
    board = _sea_board()
    attacker = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    defender = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_B, 2, hp=100)
    gs_no_carrier = _game_state(board, [attacker, defender])
    score_no_carrier = matchup_score(attacker, defender, gs_no_carrier)

    carrier = _ship(AxialCoord(0, 1), ShipKind.CARRIER, PLAYER_A, 3)
    gs_with_carrier = _game_state(
        board,
        [
            _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1),
            _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_B, 2, hp=100),
            carrier,
        ],
    )
    score_with_carrier = matchup_score(
        gs_with_carrier.ships[1], gs_with_carrier.ships[2], gs_with_carrier
    )

    assert score_with_carrier > score_no_carrier


def test_matchup_score_carrier_bonus_does_not_apply_against_a_submerged_submarine():
    board = _sea_board()
    battleship = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    sub = _ship(AxialCoord(1, 0), ShipKind.SUBMARINE, PLAYER_B, 2, surfaced=False)
    gs_no_carrier = _game_state(board, [battleship, sub])
    score_no_carrier = matchup_score(battleship, sub, gs_no_carrier)

    carrier = _ship(AxialCoord(0, 1), ShipKind.CARRIER, PLAYER_A, 3)
    gs_with_carrier = _game_state(
        board,
        [
            _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1),
            _ship(AxialCoord(1, 0), ShipKind.SUBMARINE, PLAYER_B, 2, surfaced=False),
            carrier,
        ],
    )
    score_with_carrier = matchup_score(gs_with_carrier.ships[1], gs_with_carrier.ships[2], gs_with_carrier)

    assert score_with_carrier == score_no_carrier


def test_nearest_enemy_picks_the_closest_and_breaks_ties_by_id():
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    far = _ship(AxialCoord(5, 0), ShipKind.DESTROYER, PLAYER_B, 2)
    near = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_B, 3)
    enemies = {2: far, 3: near}

    assert nearest_enemy(ship, enemies, _game_state(_sea_board(), [])) is near


def test_nearest_enemy_returns_none_when_nothing_visible():
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    assert nearest_enemy(ship, {}, _game_state(_sea_board(), [])) is None


def test_nearest_uncontrolled_port_excludes_already_controlled_ports():
    board = _sea_board()
    near_port = AxialCoord(1, 0)
    far_port = AxialCoord(4, 0)
    board.tiles[near_port] = Tile(
        coord=near_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A
    )
    board.tiles[far_port] = Tile(
        coord=far_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B
    )
    gs = _game_state(board, [])

    assert nearest_uncontrolled_port(gs, PLAYER_A, AxialCoord(0, 0)) == far_port


def test_nearest_uncontrolled_port_returns_none_if_all_ports_controlled():
    board = _sea_board()
    port = AxialCoord(1, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    gs = _game_state(board, [])

    assert nearest_uncontrolled_port(gs, PLAYER_A, AxialCoord(0, 0)) is None


def test_best_reachable_attack_finds_an_enemy_within_reach():
    board = _sea_board()
    attacker = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    enemy = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_B, 2)
    gs = _game_state(board, [attacker, enemy])

    assert best_reachable_attack(attacker, gs, {2: enemy}) == AxialCoord(1, 0)


def test_best_reachable_attack_returns_none_when_out_of_range():
    board = _sea_board()
    attacker = _ship(AxialCoord(0, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)
    attacker.movement_remaining = 1
    enemy = _ship(AxialCoord(5, 0), ShipKind.DESTROYER, PLAYER_B, 2)
    gs = _game_state(board, [attacker, enemy])

    assert best_reachable_attack(attacker, gs, {2: enemy}) is None


def test_best_reachable_attack_picks_the_more_favorable_of_two_targets():
    board = _sea_board()
    attacker = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    weak = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    strong = _ship(AxialCoord(-1, 0), ShipKind.BATTLESHIP, PLAYER_B, 3, hp=100)
    gs = _game_state(board, [attacker, weak, strong])

    assert best_reachable_attack(attacker, gs, {2: weak, 3: strong}) == AxialCoord(1, 0)
