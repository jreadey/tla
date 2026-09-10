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


def test_friendly_occupied_hex_is_never_a_valid_stop():
    board = _sea_board()
    blocker = _make_ship(AxialCoord(1, 0), movement_remaining=0, owner=PLAYER_A, ship_id=2)
    # Plenty of budget -- even so, the blocker's own hex can never be a
    # legal stop (at most one ship per hex at the end of a turn).
    mover = _make_ship(AxialCoord(0, 0), movement_remaining=5, ship_id=1)
    gs = _game_state(board, [mover, blocker])

    reachable = reachable_hexes(mover, gs)
    assert AxialCoord(1, 0) not in reachable


def test_friendly_occupied_hex_is_passable_with_enough_movement_left():
    board = _sea_board()
    blocker = _make_ship(AxialCoord(1, 0), movement_remaining=0, owner=PLAYER_A, ship_id=2)
    # Exactly 2 remaining when reaching the blocker's hex -- just enough to
    # pass through (1 to enter, 1 more to clear it) and stop just beyond.
    mover = _make_ship(AxialCoord(0, 0), movement_remaining=2, ship_id=1)
    gs = _game_state(board, [mover, blocker])

    reachable = reachable_hexes(mover, gs)
    assert AxialCoord(1, 0) not in reachable  # still never a valid stop
    assert reachable[AxialCoord(2, 0)] == 2  # but reachable by passing through it


def test_friendly_occupied_hex_blocks_passage_with_only_one_movement_left():
    board = _sea_board()
    blocker = _make_ship(AxialCoord(1, 0), movement_remaining=0, owner=PLAYER_A, ship_id=2)
    # Only 1 remaining when reaching the blocker's hex (after using 1 of 2
    # on a detour step first) -- not enough to pass through, so nothing
    # past it via this route is reachable either.
    detour_ship = _make_ship(AxialCoord(0, -1), movement_remaining=2, ship_id=1)
    gs = _game_state(board, [detour_ship, blocker])

    reachable = reachable_hexes(detour_ship, gs)
    assert AxialCoord(1, 0) not in reachable
    assert AxialCoord(2, 0) not in reachable


def test_validate_path_allows_passing_through_a_friendly_ship_with_movement_to_spare():
    board = _sea_board()
    blocker = _make_ship(AxialCoord(1, 0), movement_remaining=0, owner=PLAYER_A, ship_id=2)
    mover = _make_ship(AxialCoord(0, 0), movement_remaining=2, ship_id=1)
    gs = _game_state(board, [mover, blocker])

    validate_path(mover, [AxialCoord(0, 0), AxialCoord(1, 0), AxialCoord(2, 0)], gs)  # no raise


def test_validate_path_rejects_stopping_on_a_friendly_occupied_hex():
    board = _sea_board()
    blocker = _make_ship(AxialCoord(1, 0), movement_remaining=0, owner=PLAYER_A, ship_id=2)
    mover = _make_ship(AxialCoord(0, 0), movement_remaining=5, ship_id=1)
    gs = _game_state(board, [mover, blocker])

    with pytest.raises(ValueError):
        validate_path(mover, [AxialCoord(0, 0), AxialCoord(1, 0)], gs)


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


def test_reachable_hexes_treat_as_open_lets_preview_pass_an_enemy_hex():
    board = _sea_board()
    enemy = _make_ship(AxialCoord(1, 0), movement_remaining=0, owner=PLAYER_B, ship_id=2)
    mover = _make_ship(AxialCoord(0, 0), movement_remaining=2, ship_id=1)
    gs = _game_state(board, [mover, enemy])

    reachable = reachable_hexes(mover, gs, treat_as_open=frozenset({AxialCoord(1, 0)}))

    # The overridden hex is now a normal stop, not a battle-only terminal...
    assert reachable[AxialCoord(1, 0)] == 1
    # ...and, being "open", can be passed through to reach hexes beyond it,
    # unlike the real "enemy" classification (see the terminal-only test).
    assert reachable[AxialCoord(2, 0)] == 2


def test_reachable_hexes_treat_as_open_does_not_affect_other_hexes():
    board = _sea_board()
    enemy_a = _make_ship(AxialCoord(1, 0), movement_remaining=0, owner=PLAYER_B, ship_id=2)
    enemy_b = _make_ship(AxialCoord(-1, 0), movement_remaining=0, owner=PLAYER_B, ship_id=3)
    mover = _make_ship(AxialCoord(0, 0), movement_remaining=2, ship_id=1)
    gs = _game_state(board, [mover, enemy_a, enemy_b])

    reachable = reachable_hexes(mover, gs, treat_as_open=frozenset({AxialCoord(1, 0)}))

    assert reachable[AxialCoord(1, 0)] == 1  # overridden -> open
    assert AxialCoord(-1, 0) in reachable  # untouched enemy hex still reachable...
    assert reachable[AxialCoord(-1, 0)] == 1
    assert AxialCoord(-2, 0) not in reachable  # ...but still not passable through


def test_validate_path_treat_as_open_allows_a_route_through_an_enemy_hex():
    board = _sea_board()
    enemy = _make_ship(AxialCoord(1, 0), movement_remaining=0, owner=PLAYER_B, ship_id=2)
    mover = _make_ship(AxialCoord(0, 0), movement_remaining=2, ship_id=1)
    gs = _game_state(board, [mover, enemy])

    # No raise: with the override, this reads as an ordinary open-water path.
    validate_path(
        mover,
        [AxialCoord(0, 0), AxialCoord(1, 0), AxialCoord(2, 0)],
        gs,
        treat_as_open=frozenset({AxialCoord(1, 0)}),
    )


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


def test_move_ship_captures_an_undefended_enemy_port():
    # Regression: move_ship is the AI's own mover (see its docstring), and
    # unlike move_ship_along_path it used to never call
    # handle_port_capture -- an AI ship could sail onto an empty enemy
    # port and simply sit there without ever taking it.
    board = _sea_board()
    port = AxialCoord(1, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    ship = _make_ship(AxialCoord(0, 0), movement_remaining=2, owner=PLAYER_A)
    gs = _game_state(board, [ship])

    move_ship(ship, port, gs)

    assert board.tiles[port].port_display_owner == PLAYER_A


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


def test_begin_engagement_applies_approach_and_charges_one_point_for_the_attack():
    board = _sea_board()
    ship = _make_ship(AxialCoord(0, 0), movement_remaining=3)
    enemy = _make_ship(AxialCoord(2, -1), movement_remaining=0, owner=PLAYER_B, ship_id=2)
    gs = _game_state(board, [ship, enemy])

    path = [AxialCoord(0, 0), AxialCoord(1, -1), AxialCoord(2, -1)]
    defender = begin_engagement(ship, path, gs)

    assert defender is enemy
    assert ship.position == AxialCoord(1, -1)  # stopped at the approach hex
    # 1 point for the approach step, 1 more for the attack itself -- same
    # per-step cost as a normal move, not a flat zero-out.
    assert ship.movement_remaining == 1


def test_begin_engagement_with_adjacent_enemy_costs_exactly_one_point():
    board = _sea_board()
    ship = _make_ship(AxialCoord(0, 0), movement_remaining=3)
    enemy = _make_ship(AxialCoord(1, 0), movement_remaining=0, owner=PLAYER_B, ship_id=2)
    gs = _game_state(board, [ship, enemy])

    begin_engagement(ship, [AxialCoord(0, 0), AxialCoord(1, 0)], gs)

    assert ship.position == AxialCoord(0, 0)
    assert ship.movement_remaining == 2  # left over movement can still be used this turn


def test_begin_engagement_stops_short_when_the_only_approach_hex_is_a_friendly_ship():
    # Regression: if the hex right before the target is a friendly ship
    # merely passed through en route (the "passthrough" rule), the
    # attacker can't literally rest there alongside it -- doing so used to
    # either raise a spurious "can't stop on a friendly ship" error or
    # (in an earlier, wrong fix) silently leave two ships sharing a hex.
    # Correct behavior: the attacker stops one hex earlier (its own
    # starting hex, here) and the whole remaining stretch -- crossing the
    # friendly hex and then the target -- is charged as the attack, at the
    # same per-hex rate as an ordinary move (never a flat 1 regardless of
    # distance).
    board = _sea_board()
    ship = _make_ship(AxialCoord(0, 0), movement_remaining=3)
    friendly = _make_ship(AxialCoord(1, 0), movement_remaining=0, owner=PLAYER_A, ship_id=2)
    enemy = _make_ship(AxialCoord(2, 0), movement_remaining=0, owner=PLAYER_B, ship_id=3)
    gs = _game_state(board, [ship, friendly, enemy])

    path = [AxialCoord(0, 0), AxialCoord(1, 0), AxialCoord(2, 0)]
    defender = begin_engagement(ship, path, gs)

    assert defender is enemy
    assert ship.position == AxialCoord(0, 0)  # never advances onto the friendly ship's hex
    assert friendly.position == AxialCoord(1, 0)  # no overlap -- the friendly ship is untouched
    assert ship.movement_remaining == 1  # 3 - 2 (crossing the friendly hex, then the target)


def test_begin_engagement_stops_short_with_an_open_hex_further_back_too():
    # A longer approach with the blocking friendly ship immediately before
    # the target: the attacker still gets as close as it safely can (one
    # hex further back, not all the way to its own start), and only the
    # final blocked stretch is charged at the "attack" rate.
    board = _sea_board()
    ship = _make_ship(AxialCoord(0, 0), movement_remaining=4)
    friendly = _make_ship(AxialCoord(2, 0), movement_remaining=0, owner=PLAYER_A, ship_id=2)
    enemy = _make_ship(AxialCoord(3, 0), movement_remaining=0, owner=PLAYER_B, ship_id=3)
    gs = _game_state(board, [ship, friendly, enemy])

    path = [AxialCoord(0, 0), AxialCoord(1, 0), AxialCoord(2, 0), AxialCoord(3, 0)]
    defender = begin_engagement(ship, path, gs)

    assert defender is enemy
    assert ship.position == AxialCoord(1, 0)  # as close as it can safely get
    assert ship.movement_remaining == 1  # 4 - 1 (approach to (1,0)) - 2 (crossing (2,0) then the target)


def test_ship_can_keep_moving_after_an_engagement_with_movement_left():
    board = _sea_board()
    ship = _make_ship(AxialCoord(0, 0), movement_remaining=3)
    enemy = _make_ship(AxialCoord(1, 0), movement_remaining=0, owner=PLAYER_B, ship_id=2)
    gs = _game_state(board, [ship, enemy])

    begin_engagement(ship, [AxialCoord(0, 0), AxialCoord(1, 0)], gs)
    assert ship.movement_remaining == 2  # retreat is free; 2 of the original 3 remain

    # A retreat leaves the ship back at its own hex, free to draw a fresh
    # move with whatever's left.
    move_ship_along_path(ship, [AxialCoord(0, 0), AxialCoord(-1, 0)], gs)
    assert ship.position == AxialCoord(-1, 0)
    assert ship.movement_remaining == 1


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
