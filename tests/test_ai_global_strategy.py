from tla.ai.enemy_model import EnemyModel
from tla.ai.global_strategy import (
    Posture,
    compute_posture,
    own_strength,
    posture_adjusted_ai_config,
    rank_port_threats,
)
from tla.board import Board
from tla.config import AiConfig, Config, FleetConfig
from tla.game_state import GameState
from tla.hexgrid import AxialCoord, hexes_in_range
from tla.ship import Ship, ShipKind
from tla.tile import PLAYER_A, PLAYER_B, Tile, TerrainType


def _sea_board(radius: int = 10) -> Board:
    board = Board(width=radius * 2 + 1, height=radius * 2 + 1)
    for coord in hexes_in_range(AxialCoord(0, 0), radius):
        board.tiles[coord] = Tile(coord=coord, terrain=TerrainType.SEA)
    return board


def _ship(coord: AxialCoord, kind: ShipKind, owner, ship_id: int) -> Ship:
    stats = Config().ship_stats.stats[kind]
    return Ship(id=ship_id, kind=kind, owner=owner, position=coord, current_hp=stats.hp)


def _game_state(board: Board, ships: list[Ship], config: Config) -> GameState:
    return GameState(config=config, board=board, ships={s.id: s for s in ships})


# -- own_strength -------------------------------------------------------


def test_own_strength_sums_current_hp_and_damage_stat():
    board = _sea_board()
    ships = [
        _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1),
        _ship(AxialCoord(1, 0), ShipKind.CRUISER, PLAYER_A, 2),
    ]
    gs = _game_state(board, ships, config=Config())

    hp, damage = own_strength(gs, PLAYER_A)

    assert hp == 12 + 8  # battleship + cruiser stats.hp
    assert damage == 4 + 3  # battleship + cruiser stats.damage


def test_own_strength_ignores_the_other_players_ships():
    board = _sea_board()
    ships = [
        _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1),
        _ship(AxialCoord(1, 0), ShipKind.BATTLESHIP, PLAYER_B, 2),
    ]
    gs = _game_state(board, ships, config=Config())

    assert own_strength(gs, PLAYER_A) == (12, 4)


# -- compute_posture ------------------------------------------------------


def test_compute_posture_is_aggressive_when_we_clearly_outmatch_the_enemy():
    board = _sea_board()
    # Our own fleet: three battleships. The opponent's whole believed
    # fleet (never sighted, so purely from FleetConfig.counts) is one
    # patrol boat.
    own_ships = [_ship(AxialCoord(i, 0), ShipKind.BATTLESHIP, PLAYER_A, i + 1) for i in range(3)]
    config = Config(fleet=FleetConfig(counts={ShipKind.PATROL_BOAT: 1}))
    gs = _game_state(board, own_ships, config=config)
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)

    assert compute_posture(gs, PLAYER_A, model, AiConfig(posture_margin=2)) == Posture.AGGRESSIVE


def test_compute_posture_is_defensive_when_the_enemy_clearly_outmatches_us():
    board = _sea_board()
    own_ships = [_ship(AxialCoord(0, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)]
    config = Config(fleet=FleetConfig(counts={ShipKind.BATTLESHIP: 3}))
    gs = _game_state(board, own_ships, config=config)
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)

    assert compute_posture(gs, PLAYER_A, model, AiConfig(posture_margin=2)) == Posture.DEFENSIVE


def test_compute_posture_is_neutral_for_roughly_even_forces():
    board = _sea_board()
    own_ships = [_ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)]
    config = Config(fleet=FleetConfig(counts={ShipKind.BATTLESHIP: 1}))
    gs = _game_state(board, own_ships, config=config)
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)

    assert compute_posture(gs, PLAYER_A, model, AiConfig(posture_margin=2)) == Posture.NEUTRAL


# -- posture_adjusted_ai_config -------------------------------------------


def test_posture_adjusted_ai_config_raises_margins_when_aggressive():
    base = AiConfig(task_force_outnumbered_margin=0, port_defense_margin=0, carrier_defense_margin=0)

    adjusted = posture_adjusted_ai_config(base, Posture.AGGRESSIVE)

    assert adjusted.task_force_outnumbered_margin == base.posture_margin_shift
    assert adjusted.port_defense_margin == base.posture_margin_shift
    assert adjusted.carrier_defense_margin == base.posture_margin_shift


def test_posture_adjusted_ai_config_lowers_margins_when_defensive():
    base = AiConfig(task_force_outnumbered_margin=0, port_defense_margin=0, carrier_defense_margin=0)

    adjusted = posture_adjusted_ai_config(base, Posture.DEFENSIVE)

    assert adjusted.task_force_outnumbered_margin == -base.posture_margin_shift
    assert adjusted.port_defense_margin == -base.posture_margin_shift
    assert adjusted.carrier_defense_margin == -base.posture_margin_shift


def test_posture_adjusted_ai_config_is_a_no_op_when_neutral():
    base = AiConfig()

    assert posture_adjusted_ai_config(base, Posture.NEUTRAL) is base


# -- rank_port_threats ------------------------------------------------------


def test_rank_port_threats_orders_most_threatened_first():
    port_near = AxialCoord(0, 0)
    port_far = AxialCoord(8, 0)
    board = _sea_board(radius=12)
    board.tiles[port_near] = Tile(
        coord=port_near, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A, port_controller=PLAYER_A
    )
    board.tiles[port_far] = Tile(
        coord=port_far, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A, port_controller=PLAYER_A
    )
    config = Config(fleet=FleetConfig(counts={ShipKind.BATTLESHIP: 1}))
    gs = _game_state(board, [], config=config)
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)

    # Individually resolve the tracked battleship right next to port_near,
    # then let it go unseen -- its belief field concentrates there, far
    # from port_far.
    model._resolve(7, ShipKind.BATTLESHIP, AxialCoord(1, 0), 12, turn=1)
    model._ensure_fields_for_unseen({})

    ranked = rank_port_threats(gs, PLAYER_A, model, radius=4)

    assert [port for port, _ in ranked] == [port_near, port_far]
    assert ranked[0][1] > ranked[1][1]
