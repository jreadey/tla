from tla.board import Board
from tla.config import Config
from tla.game_state import GameState
from tla.hexgrid import AxialCoord
from tla.ship import Ship, ShipKind
from tla.tile import PLAYER_A, PLAYER_B, Tile, TerrainType
from tla.win_condition import check_elimination, check_port_control


def _ship(owner, ship_id, coord=AxialCoord(0, 0)) -> Ship:
    return Ship(id=ship_id, kind=ShipKind.DESTROYER, owner=owner, position=coord, current_hp=6)


def test_check_elimination_returns_none_when_both_sides_have_ships():
    board = Board(width=5, height=5)
    gs = GameState(config=Config(), board=board, ships={1: _ship(PLAYER_A, 1), 2: _ship(PLAYER_B, 2)})
    assert check_elimination(gs) is None


def test_check_elimination_declares_the_other_player_the_winner():
    board = Board(width=5, height=5)
    gs = GameState(config=Config(), board=board, ships={1: _ship(PLAYER_A, 1)})
    assert check_elimination(gs) == PLAYER_A  # B has no ships left


def _board_with_ports(ports: dict[AxialCoord, int], controllers: dict[AxialCoord, int] | None = None) -> Board:
    board = Board(width=5, height=5)
    controllers = controllers or {}
    for coord, owner in ports.items():
        board.tiles[coord] = Tile(
            coord=coord,
            terrain=TerrainType.LAND,
            is_port=True,
            port_owner=owner,
            port_controller=controllers.get(coord),
        )
    return board


def test_check_port_control_returns_none_with_no_ports():
    board = Board(width=5, height=5)
    gs = GameState(config=Config(), board=board, ships={})
    assert check_port_control(gs) is None


def test_check_port_control_returns_none_when_split_between_both_sides():
    port_a, port_b = AxialCoord(0, 0), AxialCoord(2, 0)
    board = _board_with_ports({port_a: PLAYER_A, port_b: PLAYER_B})
    gs = GameState(config=Config(), board=board, ships={})
    assert check_port_control(gs) is None


def test_check_port_control_declares_a_winner_when_they_hold_every_port():
    port_a, port_b = AxialCoord(0, 0), AxialCoord(2, 0)
    # port_b is owned by B but has been captured and now displays as A's.
    board = _board_with_ports({port_a: PLAYER_A, port_b: PLAYER_B}, controllers={port_b: PLAYER_A})
    gs = GameState(config=Config(), board=board, ships={})
    assert check_port_control(gs) == PLAYER_A


def test_check_port_control_ignores_live_occupancy_only_the_display_owner_matters():
    # A ship physically sitting on a port doesn't matter -- only the
    # (sticky) displayed controller does.
    port_a, port_b = AxialCoord(0, 0), AxialCoord(2, 0)
    board = _board_with_ports({port_a: PLAYER_A, port_b: PLAYER_B})
    besieger = _ship(PLAYER_A, 1, coord=port_b)  # occupies B's port, but hasn't flipped it
    gs = GameState(config=Config(), board=board, ships={1: besieger})
    assert check_port_control(gs) is None


def test_check_port_control_with_a_single_port_on_the_map():
    port = AxialCoord(0, 0)
    board = _board_with_ports({port: PLAYER_B})
    gs = GameState(config=Config(), board=board, ships={})
    assert check_port_control(gs) == PLAYER_B
