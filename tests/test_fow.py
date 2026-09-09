from tla.board import Board
from tla.config import Config, FowConfig
from tla.fow import is_hidden, visible_hexes_for
from tla.game_state import GameState
from tla.hexgrid import AxialCoord, hexes_in_range
from tla.ship import Ship, ShipKind
from tla.tile import PLAYER_A, PLAYER_B, Tile, TerrainType


def _sea_board(radius: int = 8) -> Board:
    board = Board(width=radius * 2 + 1, height=radius * 2 + 1)
    for coord in hexes_in_range(AxialCoord(0, 0), radius):
        board.tiles[coord] = Tile(coord=coord, terrain=TerrainType.SEA)
    return board


def _ship(coord, owner=PLAYER_A, kind=ShipKind.DESTROYER, ship_id=1) -> Ship:
    return Ship(id=ship_id, kind=kind, owner=owner, position=coord, current_hp=6)


def _game_state(board: Board, ships: list[Ship], fow: FowConfig | None = None) -> GameState:
    config = Config(fow=fow or FowConfig())
    return GameState(config=config, board=board, ships={s.id: s for s in ships})


def test_visible_hexes_include_a_ships_own_hex_and_its_neighbors():
    origin = AxialCoord(0, 0)
    ship = _ship(origin, ship_id=1)
    gs = _game_state(_sea_board(), [ship])

    visible = visible_hexes_for(gs, PLAYER_A)

    assert origin in visible
    assert set(hexes_in_range(origin, 1)) <= visible


def test_visible_hexes_exclude_hexes_beyond_the_ship_radius():
    origin = AxialCoord(0, 0)
    ship = _ship(origin, ship_id=1)
    gs = _game_state(_sea_board(), [ship])

    visible = visible_hexes_for(gs, PLAYER_A)

    assert AxialCoord(2, 0) not in visible  # distance 2, beyond radius 1


def test_visible_hexes_extend_further_around_a_controlled_port():
    port = AxialCoord(0, 0)
    board = _sea_board()
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    gs = _game_state(board, [])

    visible = visible_hexes_for(gs, PLAYER_A)

    assert set(hexes_in_range(port, 4)) <= visible
    assert AxialCoord(5, 0) not in visible  # distance 5, beyond the port's radius 4


def test_visible_hexes_extend_further_around_a_friendly_carrier():
    origin = AxialCoord(0, 0)
    carrier = _ship(origin, kind=ShipKind.CARRIER, ship_id=1)
    gs = _game_state(_sea_board(), [carrier])

    visible = visible_hexes_for(gs, PLAYER_A)

    assert set(hexes_in_range(origin, 4)) <= visible


def test_a_captured_enemy_port_extends_the_capturing_players_vision():
    from tla.production import handle_port_capture

    port = AxialCoord(0, 0)
    board = _sea_board()
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    capturer = _ship(port, owner=PLAYER_A, ship_id=1)
    gs = _game_state(board, [capturer])
    handle_port_capture(gs, port)  # flips control to A

    visible = visible_hexes_for(gs, PLAYER_A)

    assert set(hexes_in_range(port, 4)) <= visible


def test_visibility_radii_are_configurable():
    origin = AxialCoord(0, 0)
    ship = _ship(origin, ship_id=1)
    fow = FowConfig(ship_visibility_radius=3, port_and_carrier_visibility_radius=6)
    gs = _game_state(_sea_board(), [ship], fow=fow)

    visible = visible_hexes_for(gs, PLAYER_A)

    assert AxialCoord(3, 0) in visible
    assert AxialCoord(4, 0) not in visible


def test_is_hidden_is_always_false_for_the_viewers_own_ships():
    ship = _ship(AxialCoord(5, 5), owner=PLAYER_A)
    assert is_hidden(PLAYER_A, ship, visible_hexes=set()) is False


def test_is_hidden_is_true_for_an_enemy_ship_outside_vision():
    ship = _ship(AxialCoord(5, 5), owner=PLAYER_B)
    assert is_hidden(PLAYER_A, ship, visible_hexes=set()) is True


def test_is_hidden_is_false_for_a_normal_enemy_ship_inside_vision():
    coord = AxialCoord(1, 1)
    ship = _ship(coord, owner=PLAYER_B, kind=ShipKind.DESTROYER)
    assert is_hidden(PLAYER_A, ship, visible_hexes={coord}) is False


def test_is_hidden_is_true_for_a_submerged_enemy_submarine_even_inside_vision():
    coord = AxialCoord(1, 1)
    sub = Ship(id=1, kind=ShipKind.SUBMARINE, owner=PLAYER_B, position=coord, current_hp=4, surfaced=False)
    assert is_hidden(PLAYER_A, sub, visible_hexes={coord}) is True


def test_is_hidden_is_false_for_a_surfaced_enemy_submarine_inside_vision():
    coord = AxialCoord(1, 1)
    sub = Ship(id=1, kind=ShipKind.SUBMARINE, owner=PLAYER_B, position=coord, current_hp=4, surfaced=True)
    assert is_hidden(PLAYER_A, sub, visible_hexes={coord}) is False
