from tla.ai.pathing import shortest_path
from tla.board import Board
from tla.config import Config
from tla.game_state import GameState
from tla.hexgrid import AxialCoord, hexes_in_range
from tla.ship import Ship, ShipKind
from tla.tile import PLAYER_A, PLAYER_B, Tile, TerrainType


def _sea_board(radius: int = 5) -> Board:
    board = Board(width=radius * 2 + 1, height=radius * 2 + 1)
    for coord in hexes_in_range(AxialCoord(0, 0), radius):
        board.tiles[coord] = Tile(coord=coord, terrain=TerrainType.SEA)
    return board


def _ship(coord: AxialCoord, kind: ShipKind, owner, ship_id: int) -> Ship:
    stats = Config().ship_stats.stats[kind]
    return Ship(
        id=ship_id, kind=kind, owner=owner, position=coord, current_hp=stats.hp, movement_remaining=stats.movement
    )


def test_shortest_path_starts_and_ends_correctly():
    board = _sea_board()
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    gs = GameState(config=Config(), board=board, ships={1: ship})

    path = shortest_path(ship, AxialCoord(2, 0), gs)

    assert path[0] == AxialCoord(0, 0)
    assert path[-1] == AxialCoord(2, 0)
    assert len(path) - 1 <= ship.movement_remaining


def test_shortest_path_to_own_position_is_a_single_hex():
    board = _sea_board()
    ship = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    gs = GameState(config=Config(), board=board, ships={1: ship})

    assert shortest_path(ship, AxialCoord(0, 0), gs) == [AxialCoord(0, 0)]


def test_shortest_path_ends_on_an_enemy_hex_for_an_attack():
    board = _sea_board()
    attacker = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    defender = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_B, 2)
    gs = GameState(config=Config(), board=board, ships={1: attacker, 2: defender})

    path = shortest_path(attacker, AxialCoord(1, 0), gs)

    assert path == [AxialCoord(0, 0), AxialCoord(1, 0)]


def test_shortest_path_raises_when_unreachable():
    board = _sea_board()
    ship = _ship(AxialCoord(0, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)
    ship.movement_remaining = 1
    gs = GameState(config=Config(), board=board, ships={1: ship})

    try:
        shortest_path(ship, AxialCoord(5, 0), gs)
        assert False, "expected a ValueError"
    except ValueError:
        pass
