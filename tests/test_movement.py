import pytest

from tla.board import Board
from tla.config import Config
from tla.game_state import GameState
from tla.hexgrid import AxialCoord, hexes_in_range
from tla.movement import (
    begin_engagement,
    move_ship,
    move_ship_along_path,
    reachable_hexes,
    toggle_submarine_state,
    validate_path,
)
from tla.ship import Ship, ShipKind
from tla.tile import PLAYER_A, PLAYER_B, Tile, TerrainType


def _sea_board(radius: int = 5) -> Board:
    board = Board(width=radius * 2 + 1, height=radius * 2 + 1)
    for coord in hexes_in_range(AxialCoord(0, 0), radius):
        board.tiles[coord] = Tile(coord=coord, terrain=TerrainType.SEA)
    return board


def _make_ship(
    coord: AxialCoord,
    movement_remaining: int,
    owner=PLAYER_A,
    kind: ShipKind = ShipKind.DESTROYER,
    ship_id: int = 1,
    surfaced: bool = True,
) -> Ship:
    return Ship(
        id=ship_id,
        kind=kind,
        owner=owner,
        position=coord,
        current_hp=6,
        surfaced=surfaced,
        movement_remaining=movement_remaining,
    )


def _game_state(board: Board, ships: list[Ship]) -> GameState:
    return GameState(config=Config(), board=board, ships={s.id: s for s in ships})


def test_reachable_hexes_respects_movement_budget():
    board = _sea_board()
    ship = _make_ship(AxialCoord(0, 0), movement_remaining=2)
    gs = _game_state(board, [ship])

    reachable = reachable_hexes(ship, gs)

    assert all(cost <= 2 for cost in reachable.values())
    assert AxialCoord(0, 0) not in reachable  # own hex excluded
    # Every hex exactly 2 steps away in a fully open sea should be reachable.
    two_step_ring = hexes_in_range(AxialCoord(0, 0), 2) - hexes_in_range(AxialCoord(0, 0), 1)
    assert two_step_ring <= reachable.keys()


def test_reachable_hexes_zero_budget_is_empty():
    board = _sea_board()
    ship = _make_ship(AxialCoord(0, 0), movement_remaining=0)
    gs = _game_state(board, [ship])
    assert reachable_hexes(ship, gs) == {}


def test_land_hex_is_impassable():
    board = _sea_board()
    board.tiles[AxialCoord(1, 0)] = Tile(coord=AxialCoord(1, 0), terrain=TerrainType.LAND)
    ship = _make_ship(AxialCoord(0, 0), movement_remaining=3)
    gs = _game_state(board, [ship])

    reachable = reachable_hexes(ship, gs)
    assert AxialCoord(1, 0) not in reachable


def test_friendly_occupied_hex_is_fully_impassable():
    board = _sea_board()
    blocker = _make_ship(AxialCoord(1, 0), movement_remaining=0, owner=PLAYER_A, ship_id=2)
    # Budget of 2 is exactly enough for the direct path through (1, 0) but
    # not enough for any detour around it (hex grids have alternate routes,
    # so a blocked hex isn't a full blockade -- just too far to detour
    # around within this budget).
    mover = _make_ship(AxialCoord(0, 0), movement_remaining=2, ship_id=1)
    gs = _game_state(board, [mover, blocker])

    reachable = reachable_hexes(mover, gs)
    assert AxialCoord(1, 0) not in reachable
    assert AxialCoord(2, 0) not in reachable


def test_enemy_occupied_hex_is_reachable_only_as_a_terminal():
    board = _sea_board()
    enemy = _make_ship(AxialCoord(1, 0), movement_remaining=0, owner=PLAYER_B, ship_id=2)
    mover = _make_ship(AxialCoord(0, 0), movement_remaining=2, ship_id=1)
    gs = _game_state(board, [mover, enemy])

    reachable = reachable_hexes(mover, gs)
    # Reachable (triggers a battle on arrival)...
    assert AxialCoord(1, 0) in reachable
    assert reachable[AxialCoord(1, 0)] == 1
    # ...but not passable through to reach further hexes in a straight line
    # (a detour around it is a separate matter, not what this checks).
    assert AxialCoord(2, 0) not in reachable


def test_leaving_port_must_step_onto_sea_not_another_port():
    board = _sea_board()
    port_coord = AxialCoord(0, 0)
    board.tiles[port_coord] = Tile(
        coord=port_coord, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A
    )
    other_port = AxialCoord(1, 0)
    board.tiles[other_port] = Tile(
        coord=other_port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A
    )
    # Budget of 1 isolates the direct first-step restriction: the other port
    # is adjacent (would be reachable in 1 step if ports counted), but a
    # longer detour through open sea to reach it is a separate, allowed
    # case not being tested here.
    ship = _make_ship(port_coord, movement_remaining=1)
    gs = _game_state(board, [ship])

    reachable = reachable_hexes(ship, gs)
    assert other_port not in reachable
    sea_neighbor = AxialCoord(0, 1)
    assert sea_neighbor in reachable


def test_submarine_uses_submerged_budget_field_directly():
    # reachable_hexes only consults movement_remaining (already resolved by
    # the caller/turn manager), so this just confirms a lower budget limits
    # range regardless of ship kind.
    board = _sea_board()
    ship = _make_ship(AxialCoord(0, 0), movement_remaining=1, kind=ShipKind.SUBMARINE, surfaced=False)
    gs = _game_state(board, [ship])
    reachable = reachable_hexes(ship, gs)
    assert all(cost <= 1 for cost in reachable.values())


def test_move_ship_updates_position_and_deducts_cost():
    board = _sea_board()
    ship = _make_ship(AxialCoord(0, 0), movement_remaining=3)
    gs = _game_state(board, [ship])

    destination = AxialCoord(2, 0)
    result = move_ship(ship, destination, gs)

    assert ship.position == destination
    assert result.cost == 2
    assert ship.movement_remaining == 1


def test_move_ship_rejects_unreachable_destination():
    board = _sea_board()
    ship = _make_ship(AxialCoord(0, 0), movement_remaining=1)
    gs = _game_state(board, [ship])

    with pytest.raises(ValueError):
        move_ship(ship, AxialCoord(3, 0), gs)


def test_move_ship_rejects_enemy_occupied_destination():
    board = _sea_board()
    ship = _make_ship(AxialCoord(0, 0), movement_remaining=2)
    enemy = _make_ship(AxialCoord(1, 0), movement_remaining=0, owner=PLAYER_B, ship_id=2)
    gs = _game_state(board, [ship, enemy])

    with pytest.raises(ValueError):
        move_ship(ship, AxialCoord(1, 0), gs)


def test_move_ship_along_path_rejects_enemy_occupied_final_hex():
    board = _sea_board()
    ship = _make_ship(AxialCoord(0, 0), movement_remaining=2)
    enemy = _make_ship(AxialCoord(1, 0), movement_remaining=0, owner=PLAYER_B, ship_id=2)
    gs = _game_state(board, [ship, enemy])

    with pytest.raises(ValueError):
        move_ship_along_path(ship, [AxialCoord(0, 0), AxialCoord(1, 0)], gs)


def test_begin_engagement_applies_approach_and_zeroes_movement():
    board = _sea_board()
    ship = _make_ship(AxialCoord(0, 0), movement_remaining=3)
    enemy = _make_ship(AxialCoord(2, -1), movement_remaining=0, owner=PLAYER_B, ship_id=2)
    gs = _game_state(board, [ship, enemy])

    path = [AxialCoord(0, 0), AxialCoord(1, -1), AxialCoord(2, -1)]
    defender = begin_engagement(ship, path, gs)

    assert defender is enemy
    assert ship.position == AxialCoord(1, -1)  # stopped at the approach hex
    assert ship.movement_remaining == 0  # movement ends here regardless of outcome


def test_begin_engagement_with_adjacent_enemy_does_not_move_the_attacker():
    board = _sea_board()
    ship = _make_ship(AxialCoord(0, 0), movement_remaining=3)
    enemy = _make_ship(AxialCoord(1, 0), movement_remaining=0, owner=PLAYER_B, ship_id=2)
    gs = _game_state(board, [ship, enemy])

    begin_engagement(ship, [AxialCoord(0, 0), AxialCoord(1, 0)], gs)

    assert ship.position == AxialCoord(0, 0)
    assert ship.movement_remaining == 0


def test_begin_engagement_rejects_a_path_not_ending_on_an_enemy():
    board = _sea_board()
    ship = _make_ship(AxialCoord(0, 0), movement_remaining=2)
    gs = _game_state(board, [ship])

    with pytest.raises(ValueError):
        begin_engagement(ship, [AxialCoord(0, 0), AxialCoord(1, 0)], gs)


def test_move_ship_along_path_takes_the_exact_drawn_route():
    board = _sea_board()
    ship = _make_ship(AxialCoord(0, 0), movement_remaining=3)
    gs = _game_state(board, [ship])

    # A deliberately roundabout 3-step path to a hex that's only 1 step away
    # in a straight line -- the whole point is that the drawn route, not
    # the shortest one, is what gets taken and charged for.
    path = [AxialCoord(0, 0), AxialCoord(1, -1), AxialCoord(1, 0), AxialCoord(0, 1)]
    result = move_ship_along_path(ship, path, gs)

    assert ship.position == AxialCoord(0, 1)
    assert result.cost == 3
    assert ship.movement_remaining == 0


def test_validate_path_rejects_path_not_starting_at_ship():
    board = _sea_board()
    ship = _make_ship(AxialCoord(0, 0), movement_remaining=3)
    gs = _game_state(board, [ship])
    with pytest.raises(ValueError):
        validate_path(ship, [AxialCoord(1, 0), AxialCoord(2, 0)], gs)


def test_validate_path_rejects_non_adjacent_step():
    board = _sea_board()
    ship = _make_ship(AxialCoord(0, 0), movement_remaining=3)
    gs = _game_state(board, [ship])
    # (0,0) and (2,0) are not neighbors -- a legal path can't skip a hex.
    with pytest.raises(ValueError):
        validate_path(ship, [AxialCoord(0, 0), AxialCoord(2, 0)], gs)


def test_validate_path_rejects_route_through_blocked_hex():
    board = _sea_board()
    blocker = _make_ship(AxialCoord(1, 0), movement_remaining=0, owner=PLAYER_B, ship_id=2)
    mover = _make_ship(AxialCoord(0, 0), movement_remaining=3, ship_id=1)
    gs = _game_state(board, [mover, blocker])
    with pytest.raises(ValueError):
        validate_path(mover, [AxialCoord(0, 0), AxialCoord(1, 0), AxialCoord(2, 0)], gs)


def test_validate_path_rejects_exceeding_budget():
    board = _sea_board()
    ship = _make_ship(AxialCoord(0, 0), movement_remaining=1)
    gs = _game_state(board, [ship])
    with pytest.raises(ValueError):
        validate_path(ship, [AxialCoord(0, 0), AxialCoord(1, 0), AxialCoord(2, 0)], gs)


def test_toggle_submarine_state_allows_two_then_blocks_third():
    stats = Config().ship_stats.stats[ShipKind.SUBMARINE]
    ship = _make_ship(AxialCoord(0, 0), movement_remaining=stats.movement, kind=ShipKind.SUBMARINE)
    assert ship.surfaced is True

    # Pre-move toggle: submerges and refreshes the budget to the (lower)
    # submerged movement, since nothing has been spent yet.
    toggle_submarine_state(ship, stats)
    assert ship.surfaced is False
    assert ship.toggled_pre_move is True
    assert ship.movement_remaining == stats.movement_submerged

    # Post-move toggle: resurfaces but does not touch movement_remaining.
    ship.movement_remaining = 0
    toggle_submarine_state(ship, stats)
    assert ship.surfaced is True
    assert ship.toggled_post_move is True
    assert ship.movement_remaining == 0

    with pytest.raises(ValueError):
        toggle_submarine_state(ship, stats)


def test_toggle_submarine_state_rejects_non_submarine():
    stats = Config().ship_stats.stats[ShipKind.DESTROYER]
    ship = _make_ship(AxialCoord(0, 0), movement_remaining=3, kind=ShipKind.DESTROYER)
    with pytest.raises(ValueError):
        toggle_submarine_state(ship, stats)
