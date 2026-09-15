from tla.board import Board
from tla.config import Config, ProductionConfig
from tla.game_state import GameState, PortProduction
from tla.hexgrid import AxialCoord
from tla.production import handle_port_capture, run_production
from tla.ship import Ship, ShipKind
from tla.tile import PLAYER_A, PLAYER_B, Tile, TerrainType


def _board_with_ports(*coords: AxialCoord, owner=PLAYER_A) -> Board:
    board = Board(width=5, height=5)
    for coord in coords:
        board.tiles[coord] = Tile(coord=coord, terrain=TerrainType.LAND, is_port=True, port_owner=owner)
    return board


def _game_state(
    board: Board, points_per_turn: int = 20, build_order: list[ShipKind] | None = None
) -> GameState:
    production = ProductionConfig(points_per_turn=points_per_turn)
    if build_order is not None:
        production = ProductionConfig(points_per_turn=points_per_turn, build_order=build_order)
    return GameState(config=Config(production=production), board=board, ships={})


def test_run_production_does_nothing_with_an_empty_build_order():
    gs = _game_state(_board_with_ports(AxialCoord(0, 0)), build_order=[])
    run_production(gs, PLAYER_A)
    assert gs.ships == {}


def test_run_production_spawns_the_first_kind_in_build_order_once_points_cover_its_cost():
    port = AxialCoord(0, 0)
    gs = _game_state(
        _board_with_ports(port), points_per_turn=20, build_order=[ShipKind.PATROL_BOAT]  # cost 1
    )

    run_production(gs, PLAYER_A)

    spawned = list(gs.ships.values())
    assert len(spawned) == 1
    assert spawned[0].position == port
    assert spawned[0].kind == ShipKind.PATROL_BOAT


def test_run_production_advances_through_the_build_order_in_sequence():
    # A port can only ever have one spawn in progress at a time -- the hex
    # becomes occupied the instant a ship is built there -- so each
    # produced ship is removed here to simulate it moving out to sea
    # before the port's next build.
    port = AxialCoord(0, 0)
    gs = _game_state(
        _board_with_ports(port),
        points_per_turn=20,  # covers even the most expensive kind (10) every turn
        build_order=[ShipKind.PATROL_BOAT, ShipKind.DESTROYER, ShipKind.BATTLESHIP],
    )

    kinds_built = []
    for _ in range(3):
        run_production(gs, PLAYER_A)
        (new_id,) = gs.ships
        kinds_built.append(gs.ships[new_id].kind)
        del gs.ships[new_id]

    assert kinds_built == [ShipKind.PATROL_BOAT, ShipKind.DESTROYER, ShipKind.BATTLESHIP]


def test_run_production_wraps_back_to_the_start_of_the_build_order():
    port = AxialCoord(0, 0)
    gs = _game_state(
        _board_with_ports(port), points_per_turn=20, build_order=[ShipKind.PATROL_BOAT, ShipKind.DESTROYER]
    )

    kinds_built = []
    for _ in range(3):  # patrol_boat, destroyer, patrol_boat again
        run_production(gs, PLAYER_A)
        (new_id,) = gs.ships
        kinds_built.append(gs.ships[new_id].kind)
        del gs.ships[new_id]

    assert kinds_built == [ShipKind.PATROL_BOAT, ShipKind.DESTROYER, ShipKind.PATROL_BOAT]


def test_run_production_carries_leftover_points_to_the_next_build():
    port = AxialCoord(0, 0)
    gs = _game_state(
        _board_with_ports(port), points_per_turn=6, build_order=[ShipKind.DESTROYER, ShipKind.DESTROYER]
    )  # cost 4 each

    run_production(gs, PLAYER_A)  # 6 pts: first destroyer spawns, 2 left banked for the second

    assert len(gs.ships) == 1
    progress = gs.players[PLAYER_A].port_production[port]
    assert progress.next_index == 1
    assert progress.points == 2


def test_run_production_gives_each_controlled_port_the_full_points_per_turn_independently():
    # points_per_turn=10 with a cost-7 build at each of two ports: if this
    # were still a shared budget split two ways (5 each), neither would
    # complete. Each port earning the full 10 independently means both do.
    port1, port2 = AxialCoord(0, 0), AxialCoord(2, 0)
    gs = _game_state(
        _board_with_ports(port1, port2), points_per_turn=10, build_order=[ShipKind.CRUISER]
    )

    run_production(gs, PLAYER_A)

    assert len(gs.ships) == 2
    positions = {s.position for s in gs.ships.values()}
    assert positions == {port1, port2}


def test_run_production_banks_points_on_an_unfinished_port():
    port = AxialCoord(0, 0)
    gs = _game_state(_board_with_ports(port), points_per_turn=5, build_order=[ShipKind.BATTLESHIP])  # cost 10

    run_production(gs, PLAYER_A)

    assert gs.ships == {}
    assert gs.players[PLAYER_A].port_production[port].points == 5

    run_production(gs, PLAYER_A)  # 10 banked now -- completes

    assert len(gs.ships) == 1
    assert gs.players[PLAYER_A].port_production[port].points == 0


def test_run_production_gives_nothing_to_an_occupied_port():
    port1, port2 = AxialCoord(0, 0), AxialCoord(2, 0)
    gs = _game_state(
        _board_with_ports(port1, port2), points_per_turn=20, build_order=[ShipKind.CRUISER]
    )
    blocker = Ship(id=1, kind=ShipKind.DESTROYER, owner=PLAYER_B, position=port1, current_hp=6)
    gs.ships[1] = blocker

    run_production(gs, PLAYER_A)

    non_blocker_ships = [s for s in gs.ships.values() if s.owner == PLAYER_A]
    assert len(non_blocker_ships) == 1
    assert non_blocker_ships[0].position == port2
    # port1 was occupied all turn -- nothing progressed there.
    assert port1 not in gs.players[PLAYER_A].port_production


def test_run_production_assigns_ever_increasing_ids():
    port = AxialCoord(0, 0)
    gs = _game_state(_board_with_ports(port), points_per_turn=20, build_order=[ShipKind.PATROL_BOAT])
    gs.next_ship_id = 42

    run_production(gs, PLAYER_A)

    assert 42 in gs.ships
    assert gs.next_ship_id == 43


def test_handle_port_capture_wipes_progress_and_banked_points():
    port = AxialCoord(0, 0)
    gs = _game_state(_board_with_ports(port))
    gs.players[PLAYER_A].port_production[port] = PortProduction(next_index=2, points=3)
    enemy = Ship(id=1, kind=ShipKind.DESTROYER, owner=PLAYER_B, position=port, current_hp=6)
    gs.ships[1] = enemy

    handle_port_capture(gs, port)

    assert port not in gs.players[PLAYER_A].port_production


def test_handle_port_capture_does_nothing_for_a_friendly_occupant():
    port = AxialCoord(0, 0)
    gs = _game_state(_board_with_ports(port))
    gs.players[PLAYER_A].port_production[port] = PortProduction(next_index=2)
    friendly = Ship(id=1, kind=ShipKind.DESTROYER, owner=PLAYER_A, position=port, current_hp=6)
    gs.ships[1] = friendly

    handle_port_capture(gs, port)

    assert gs.players[PLAYER_A].port_production[port].next_index == 2


def test_handle_port_capture_does_nothing_for_a_non_port_hex():
    gs = _game_state(Board(width=5, height=5))
    coord = AxialCoord(1, 1)
    handle_port_capture(gs, coord)  # should not raise


def test_handle_port_capture_flips_the_displayed_controller_on_enemy_occupation():
    port = AxialCoord(0, 0)
    gs = _game_state(_board_with_ports(port))
    assert gs.board.tiles[port].port_display_owner == PLAYER_A
    enemy = Ship(id=1, kind=ShipKind.DESTROYER, owner=PLAYER_B, position=port, current_hp=6)
    gs.ships[1] = enemy

    handle_port_capture(gs, port)

    assert gs.board.tiles[port].port_display_owner == PLAYER_B


def test_port_controller_stays_flipped_after_the_capturing_ship_leaves():
    port = AxialCoord(0, 0)
    gs = _game_state(_board_with_ports(port))
    enemy = Ship(id=1, kind=ShipKind.DESTROYER, owner=PLAYER_B, position=port, current_hp=6)
    gs.ships[1] = enemy
    handle_port_capture(gs, port)
    assert gs.board.tiles[port].port_display_owner == PLAYER_B

    del gs.ships[1]  # the enemy ship sails away; port is empty again

    assert gs.board.tiles[port].port_display_owner == PLAYER_B  # still reads captured


def test_port_controller_flips_back_when_the_owner_recaptures():
    port = AxialCoord(0, 0)
    gs = _game_state(_board_with_ports(port))
    enemy = Ship(id=1, kind=ShipKind.DESTROYER, owner=PLAYER_B, position=port, current_hp=6)
    gs.ships[1] = enemy
    handle_port_capture(gs, port)
    del gs.ships[1]

    friendly = Ship(id=2, kind=ShipKind.DESTROYER, owner=PLAYER_A, position=port, current_hp=6)
    gs.ships[2] = friendly
    handle_port_capture(gs, port)

    assert gs.board.tiles[port].port_display_owner == PLAYER_A


def test_a_captured_port_restarts_its_build_order_for_the_new_controller():
    port = AxialCoord(0, 0)
    gs = _game_state(
        _board_with_ports(port), points_per_turn=20, build_order=[ShipKind.PATROL_BOAT, ShipKind.DESTROYER]
    )
    run_production(gs, PLAYER_A)  # A builds a patrol_boat, next_index -> 1
    assert gs.players[PLAYER_A].port_production[port].next_index == 1

    first_ship_id = next(iter(gs.ships))
    del gs.ships[first_ship_id]  # it sails off / is sunk elsewhere
    enemy = Ship(id=99, kind=ShipKind.DESTROYER, owner=PLAYER_B, position=port, current_hp=6)
    gs.ships[99] = enemy
    handle_port_capture(gs, port)

    assert port not in gs.players[PLAYER_A].port_production
    del gs.ships[99]  # B's ship moves off so the port is free to build on
    run_production(gs, PLAYER_B)  # B's first build at its newly-captured port

    (new_id,) = gs.ships
    assert gs.ships[new_id].kind == ShipKind.PATROL_BOAT  # restarted from the top, not index 1


def test_controlled_ports_for_includes_a_captured_enemy_port():
    port = AxialCoord(0, 0)
    board = _board_with_ports(port)  # owned by PLAYER_A
    gs = _game_state(board)

    assert board.controlled_ports_for(PLAYER_A) == [port]
    assert board.controlled_ports_for(PLAYER_B) == []

    enemy = Ship(id=1, kind=ShipKind.DESTROYER, owner=PLAYER_B, position=port, current_hp=6)
    gs.ships[1] = enemy
    handle_port_capture(gs, port)

    assert board.controlled_ports_for(PLAYER_A) == []
    assert board.controlled_ports_for(PLAYER_B) == [port]


def test_capturing_side_can_produce_from_a_flipped_port():
    port = AxialCoord(0, 0)
    gs = _game_state(_board_with_ports(port), points_per_turn=20, build_order=[ShipKind.PATROL_BOAT])
    enemy = Ship(id=1, kind=ShipKind.DESTROYER, owner=PLAYER_B, position=port, current_hp=6)
    gs.ships[1] = enemy
    handle_port_capture(gs, port)  # flips control to B, wipes A's progress there
    del gs.ships[1]  # B's ship moves off so the port is free to build on

    run_production(gs, PLAYER_B)  # B builds from A's old port

    assert len(gs.ships) == 1
    spawned = next(iter(gs.ships.values()))
    assert spawned.owner == PLAYER_B
    assert spawned.position == port
    # The original owner no longer controls it, so their own production
    # run does nothing here even though nothing has changed on their side.
    run_production(gs, PLAYER_A)
    assert len(gs.ships) == 1  # unchanged -- A doesn't control this port anymore


def test_handle_port_capture_wipes_the_previous_capturers_progress_on_recapture():
    port = AxialCoord(0, 0)
    gs = _game_state(_board_with_ports(port), points_per_turn=20, build_order=[ShipKind.PATROL_BOAT])
    enemy = Ship(id=1, kind=ShipKind.DESTROYER, owner=PLAYER_B, position=port, current_hp=6)
    gs.ships[1] = enemy
    handle_port_capture(gs, port)  # B captures
    del gs.ships[1]
    run_production(gs, PLAYER_B)  # B builds something at their new port, next_index -> 1

    spawned_id = next(iter(gs.ships))
    del gs.ships[spawned_id]  # it moves off elsewhere
    friendly = Ship(id=2, kind=ShipKind.DESTROYER, owner=PLAYER_A, position=port, current_hp=6)
    gs.ships[2] = friendly
    handle_port_capture(gs, port)  # A retakes it

    assert port not in gs.players[PLAYER_B].port_production
