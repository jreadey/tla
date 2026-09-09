from tla.board import Board
from tla.config import Config
from tla.game_state import GameState
from tla.hexgrid import AxialCoord
from tla.ship import Ship, ShipKind
from tla.tile import PLAYER_A, PLAYER_B, Tile, TerrainType
from tla.win_condition import advance_port_control_claim, check_elimination, check_port_control


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


def _full_control_game_state(controller) -> GameState:
    port_a, port_b = AxialCoord(0, 0), AxialCoord(2, 0)
    board = _board_with_ports({port_a: PLAYER_A, port_b: PLAYER_B}, controllers={port_b: controller, port_a: controller})
    return GameState(config=Config(), board=board, ships={})


def test_advance_port_control_claim_does_not_win_the_first_time_control_is_achieved():
    gs = _full_control_game_state(PLAYER_A)

    advance_port_control_claim(gs)

    assert gs.winner is None
    assert gs.port_control_claimant == PLAYER_A


def test_advance_port_control_claim_wins_once_the_same_player_holds_it_at_the_next_boundary():
    gs = _full_control_game_state(PLAYER_A)

    advance_port_control_claim(gs)  # turn boundary 1: starts the clock
    assert gs.winner is None
    advance_port_control_claim(gs)  # turn boundary 2: still A -- wins

    assert gs.winner == PLAYER_A


def test_advance_port_control_claim_resets_the_clock_if_control_is_lost_in_between():
    port_a, port_b = AxialCoord(0, 0), AxialCoord(2, 0)
    board = _board_with_ports({port_a: PLAYER_A, port_b: PLAYER_B}, controllers={port_b: PLAYER_A})
    gs = GameState(config=Config(), board=board, ships={})

    advance_port_control_claim(gs)  # boundary 1: A controls everything
    assert gs.port_control_claimant == PLAYER_A

    # B retakes their own port before boundary 2 -- control is now split.
    board.tiles[port_b].port_controller = PLAYER_B
    advance_port_control_claim(gs)  # boundary 2: contested, not a win

    assert gs.winner is None
    assert gs.port_control_claimant is None

    # A recaptures and holds it again -- needs two fresh consecutive
    # boundaries, same as starting over.
    board.tiles[port_b].port_controller = PLAYER_A
    advance_port_control_claim(gs)  # boundary 3: starts a new clock
    assert gs.winner is None
    advance_port_control_claim(gs)  # boundary 4: confirmed

    assert gs.winner == PLAYER_A


def test_advance_port_control_claim_never_overrides_an_existing_winner():
    gs = _full_control_game_state(PLAYER_A)
    gs.winner = PLAYER_B  # e.g. already decided by elimination

    advance_port_control_claim(gs)
    advance_port_control_claim(gs)

    assert gs.winner == PLAYER_B
