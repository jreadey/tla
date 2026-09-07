from tla.board import Board
from tla.config import Config
from tla.game_state import GameState
from tla.hexgrid import AxialCoord
from tla.ship import Ship, ShipKind
from tla.tile import PLAYER_A, PLAYER_B, Tile, TerrainType
from tla.win_condition import check_elimination, check_port_siege


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


def _board_with_ports(ports: dict[AxialCoord, int]) -> Board:
    board = Board(width=5, height=5)
    for coord, owner in ports.items():
        board.tiles[coord] = Tile(coord=coord, terrain=TerrainType.LAND, is_port=True, port_owner=owner)
    return board


def test_check_port_siege_increments_streak_when_fully_besieged():
    port = AxialCoord(0, 0)
    board = _board_with_ports({port: PLAYER_A})
    besieger = _ship(PLAYER_B, 1, coord=port)
    gs = GameState(config=Config(), board=board, ships={1: besieger})

    winner = check_port_siege(gs)

    assert winner is None  # only one turn-end of siege so far
    assert gs.players[PLAYER_A].siege_streak == 1


def test_check_port_siege_declares_winner_after_two_consecutive_turn_ends():
    port = AxialCoord(0, 0)
    board = _board_with_ports({port: PLAYER_A})
    besieger = _ship(PLAYER_B, 1, coord=port)
    gs = GameState(config=Config(), board=board, ships={1: besieger})
    gs.players[PLAYER_A].siege_streak = 1  # already besieged once

    winner = check_port_siege(gs)

    assert winner == PLAYER_B
    assert gs.players[PLAYER_A].siege_streak == 2


def test_check_port_siege_resets_streak_when_a_port_is_retaken():
    port_1, port_2 = AxialCoord(0, 0), AxialCoord(2, 0)
    board = _board_with_ports({port_1: PLAYER_A, port_2: PLAYER_A})
    besieger = _ship(PLAYER_B, 1, coord=port_1)
    # port_2 has no enemy ship on it -- siege isn't total this turn.
    gs = GameState(config=Config(), board=board, ships={1: besieger})
    gs.players[PLAYER_A].siege_streak = 1

    winner = check_port_siege(gs)

    assert winner is None
    assert gs.players[PLAYER_A].siege_streak == 0


def test_check_port_siege_requires_all_ports_simultaneously_occupied():
    port_1, port_2 = AxialCoord(0, 0), AxialCoord(2, 0)
    board = _board_with_ports({port_1: PLAYER_A, port_2: PLAYER_A})
    besieger = _ship(PLAYER_B, 1, coord=port_1)
    friendly = _ship(PLAYER_A, 2, coord=port_2)  # own ship still holds the other port
    gs = GameState(config=Config(), board=board, ships={1: besieger, 2: friendly})

    winner = check_port_siege(gs)

    assert winner is None
    assert gs.players[PLAYER_A].siege_streak == 0


def test_check_port_siege_checks_both_players_independently():
    port_a, port_b = AxialCoord(0, 0), AxialCoord(2, 0)
    board = _board_with_ports({port_a: PLAYER_A, port_b: PLAYER_B})
    a_besieger = _ship(PLAYER_B, 1, coord=port_a)
    gs = GameState(config=Config(), board=board, ships={1: a_besieger})
    gs.players[PLAYER_A].siege_streak = 1

    winner = check_port_siege(gs)

    assert winner == PLAYER_B
    assert gs.players[PLAYER_B].siege_streak == 0  # B's own port was never besieged
