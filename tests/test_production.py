from tla.board import Board
from tla.config import Config, ProductionConfig
from tla.game_state import GameState
from tla.hexgrid import AxialCoord
from tla.production import handle_port_capture, order, run_production
from tla.ship import Ship, ShipKind
from tla.tile import PLAYER_A, PLAYER_B, Tile, TerrainType


def _board_with_ports(*coords: AxialCoord, owner=PLAYER_A) -> Board:
    board = Board(width=5, height=5)
    for coord in coords:
        board.tiles[coord] = Tile(coord=coord, terrain=TerrainType.LAND, is_port=True, port_owner=owner)
    return board


def _game_state(board: Board, points_per_turn: int = 20) -> GameState:
    config = Config(production=ProductionConfig(points_per_turn=points_per_turn))
    return GameState(config=config, board=board, ships={})


def test_order_appends_to_that_ports_own_queue_without_any_cost_check():
    port = AxialCoord(0, 0)
    gs = _game_state(_board_with_ports(port))
    order(gs, PLAYER_A, port, ShipKind.BATTLESHIP)
    order(gs, PLAYER_A, port, ShipKind.DESTROYER)
    assert gs.players[PLAYER_A].port_production[port].orders == [
        ShipKind.BATTLESHIP,
        ShipKind.DESTROYER,
    ]


def test_run_production_does_nothing_with_no_orders_anywhere():
    gs = _game_state(_board_with_ports(AxialCoord(0, 0)))
    run_production(gs, PLAYER_A)
    assert gs.ships == {}


def test_run_production_spawns_once_a_ports_banked_points_cover_the_cost():
    port = AxialCoord(0, 0)
    gs = _game_state(_board_with_ports(port), points_per_turn=20)
    order(gs, PLAYER_A, port, ShipKind.PATROL_BOAT)  # cost 1, one port -> spawns turn one

    run_production(gs, PLAYER_A)

    spawned = list(gs.ships.values())
    assert len(spawned) == 1
    assert spawned[0].position == port
    assert spawned[0].kind == ShipKind.PATROL_BOAT
    assert gs.players[PLAYER_A].port_production[port].orders == []


def test_run_production_carries_leftover_points_to_the_next_order_in_that_ports_queue():
    port = AxialCoord(0, 0)
    gs = _game_state(_board_with_ports(port), points_per_turn=6)
    order(gs, PLAYER_A, port, ShipKind.DESTROYER)  # cost 4
    order(gs, PLAYER_A, port, ShipKind.DESTROYER)  # cost 4

    run_production(gs, PLAYER_A)  # 6 pts: first destroyer spawns, 2 left banked for the second

    assert len(gs.ships) == 1
    progress = gs.players[PLAYER_A].port_production[port]
    assert progress.orders == [ShipKind.DESTROYER]
    assert progress.points == 2


def test_run_production_splits_budget_evenly_across_ports_with_something_queued():
    port1, port2 = AxialCoord(0, 0), AxialCoord(2, 0)
    gs = _game_state(_board_with_ports(port1, port2), points_per_turn=20)
    order(gs, PLAYER_A, port1, ShipKind.CRUISER)  # cost 7
    order(gs, PLAYER_A, port2, ShipKind.CRUISER)  # cost 7

    run_production(gs, PLAYER_A)  # 10 pts/port -> both cruisers complete this turn

    assert len(gs.ships) == 2
    positions = {s.position for s in gs.ships.values()}
    assert positions == {port1, port2}


def test_run_production_ignores_a_port_with_an_empty_queue_when_splitting_budget():
    port1, port2 = AxialCoord(0, 0), AxialCoord(2, 0)
    gs = _game_state(_board_with_ports(port1, port2), points_per_turn=20)
    order(gs, PLAYER_A, port1, ShipKind.CRUISER)  # cost 7; port2 has nothing queued

    run_production(gs, PLAYER_A)  # port1 alone gets the full 20, not 10

    spawned = [s for s in gs.ships.values() if s.owner == PLAYER_A]
    assert len(spawned) == 1
    assert spawned[0].position == port1
    assert gs.players[PLAYER_A].port_production[port1].points == 13


def test_run_production_skips_occupied_ports_and_redistributes_their_share():
    port1, port2 = AxialCoord(0, 0), AxialCoord(2, 0)
    gs = _game_state(_board_with_ports(port1, port2), points_per_turn=20)
    blocker = Ship(id=1, kind=ShipKind.DESTROYER, owner=PLAYER_B, position=port1, current_hp=6)
    gs.ships[1] = blocker
    order(gs, PLAYER_A, port1, ShipKind.CRUISER)
    order(gs, PLAYER_A, port2, ShipKind.CRUISER)  # cost 7, only port2 is free -> gets the full 20

    run_production(gs, PLAYER_A)

    non_blocker_ships = [s for s in gs.ships.values() if s.owner == PLAYER_A]
    assert len(non_blocker_ships) == 1
    assert non_blocker_ships[0].position == port2
    # port1 was occupied all turn -- its queue is untouched, no points banked.
    assert gs.players[PLAYER_A].port_production[port1].orders == [ShipKind.CRUISER]
    assert gs.players[PLAYER_A].port_production[port1].points == 0


def test_run_production_assigns_ever_increasing_ids():
    port = AxialCoord(0, 0)
    gs = _game_state(_board_with_ports(port), points_per_turn=20)
    gs.next_ship_id = 42
    order(gs, PLAYER_A, port, ShipKind.PATROL_BOAT)

    run_production(gs, PLAYER_A)

    assert 42 in gs.ships
    assert gs.next_ship_id == 43


def test_handle_port_capture_wipes_queue_points_and_current_order():
    port = AxialCoord(0, 0)
    gs = _game_state(_board_with_ports(port))
    order(gs, PLAYER_A, port, ShipKind.DESTROYER)
    order(gs, PLAYER_A, port, ShipKind.CRUISER)
    gs.players[PLAYER_A].port_production[port].points = 3
    enemy = Ship(id=1, kind=ShipKind.DESTROYER, owner=PLAYER_B, position=port, current_hp=6)
    gs.ships[1] = enemy

    handle_port_capture(gs, port)

    assert port not in gs.players[PLAYER_A].port_production


def test_handle_port_capture_does_nothing_for_a_friendly_occupant():
    port = AxialCoord(0, 0)
    gs = _game_state(_board_with_ports(port))
    order(gs, PLAYER_A, port, ShipKind.DESTROYER)
    friendly = Ship(id=1, kind=ShipKind.DESTROYER, owner=PLAYER_A, position=port, current_hp=6)
    gs.ships[1] = friendly

    handle_port_capture(gs, port)

    assert gs.players[PLAYER_A].port_production[port].orders == [ShipKind.DESTROYER]


def test_handle_port_capture_does_nothing_for_a_non_port_hex():
    gs = _game_state(Board(width=5, height=5))
    coord = AxialCoord(1, 1)
    handle_port_capture(gs, coord)  # should not raise


def test_a_captured_port_resumes_from_empty_once_recaptured():
    port = AxialCoord(0, 0)
    gs = _game_state(_board_with_ports(port), points_per_turn=20)
    order(gs, PLAYER_A, port, ShipKind.PATROL_BOAT)
    enemy = Ship(id=1, kind=ShipKind.DESTROYER, owner=PLAYER_B, position=port, current_hp=6)
    gs.ships[1] = enemy
    handle_port_capture(gs, port)
    del gs.ships[1]  # enemy leaves

    run_production(gs, PLAYER_A)  # nothing queued anymore -> no spawn

    assert gs.ships == {}
