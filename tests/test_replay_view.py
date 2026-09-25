import arcade

from tla.hexgrid import AxialCoord, axial_to_pixel
from tla.rendering.replay_view import (
    PLAY_STEP_SECONDS,
    STEP_ANIM_MIN_SECONDS,
    STEP_ANIM_SECONDS_PER_HEX,
    ReplayView,
    _apply_port_state,
    _board_from_initial,
    _build_ship_histories,
    _compute_tally,
    _interpolate_path,
    _path_color_for,
    _record_label,
    _step_animation_duration,
    _step_animation_queue,
)
from tla.ship import ShipKind
from tla.tile import PLAYER_A, PLAYER_B, TerrainType


def _tile(coord, terrain="sea", is_port=False, port_owner=None, port_controller=None):
    return {
        "coord": list(coord),
        "terrain": terrain,
        "is_port": is_port,
        "port_owner": port_owner,
        "port_controller": port_controller,
    }


def _ship(id, kind, owner, position, hp=6, surfaced=True, movement_remaining=3):
    return {
        "id": id,
        "kind": kind,
        "owner": owner,
        "position": list(position),
        "hp": hp,
        "surfaced": surfaced,
        "movement_remaining": movement_remaining,
    }


def _initial_record():
    return {
        "type": "initial",
        "seed": 42,
        "board": {
            "width": 3,
            "height": 3,
            "hex_pixel_size": 18.0,
            "tiles": [
                _tile((0, 0)),
                _tile((1, 0), terrain="land", is_port=True, port_owner=PLAYER_A),
                _tile((2, 0), terrain="land", is_port=True, port_owner=PLAYER_B),
            ],
        },
        "ships": [
            _ship(1, "destroyer", PLAYER_A, (0, 0)),
            _ship(2, "patrol_boat", PLAYER_B, (2, 0), hp=2),
        ],
    }


def _move(ship_id, kind, owner, path):
    return {"ship_id": ship_id, "kind": kind, "owner": owner, "path": [list(c) for c in path]}


def _half_turn(ships, ports, *, turn_number=1, player=PLAYER_A, phase="move_a", move_log=None):
    return {
        "type": "half_turn",
        "turn_number": turn_number,
        "player": player,
        "phase": phase,
        "ships": ships,
        "ports": ports,
        "move_log": move_log or [],
    }


def _final_record(ships, ports, *, turn_number=2, winner=PLAYER_A):
    return {"type": "final", "turn_number": turn_number, "winner": winner, "ships": ships, "ports": ports}


def _make_view(records, cursor=0) -> ReplayView:
    """A ReplayView with just enough state wired up to exercise cursor/
    playback logic -- avoids __init__, which needs a live Arcade window
    (Camera2D/Text) that doesn't exist in a headless test run."""
    view = ReplayView.__new__(ReplayView)
    view.records = records
    view.board = _board_from_initial(records[0])
    view.ship_histories = _build_ship_histories(records)
    view.hex_size = 18.0
    view.cursor = cursor
    view._play_direction = 0
    view._play_timer = 0.0
    view._held_pan_keys = set()
    view._step_animation = None
    view.belief_reader = None  # no --belief file -- _set_cursor's scroll-clamp step is a no-op
    return view


def test_board_from_initial_reconstructs_terrain_and_ports():
    board = _board_from_initial(_initial_record())

    assert board.width == 3
    assert board.height == 3
    tile = board.get_tile(AxialCoord(1, 0))
    assert tile.terrain == TerrainType.LAND
    assert tile.is_port
    assert tile.port_owner == PLAYER_A


def test_apply_port_state_updates_controller_from_a_record():
    board = _board_from_initial(_initial_record())
    record = _half_turn(
        [], [{"coord": [1, 0], "port_owner": PLAYER_A, "port_controller": PLAYER_B, "display_owner": PLAYER_B}]
    )

    _apply_port_state(board, record)

    assert board.get_tile(AxialCoord(1, 0)).port_controller == PLAYER_B


def test_apply_port_state_can_revert_when_stepping_backward():
    board = _board_from_initial(_initial_record())
    captured = _half_turn(
        [], [{"coord": [1, 0], "port_owner": PLAYER_A, "port_controller": PLAYER_B, "display_owner": PLAYER_B}]
    )
    _apply_port_state(board, captured)
    assert board.get_tile(AxialCoord(1, 0)).port_controller == PLAYER_B

    # Stepping back to the initial record's own port state (controller None).
    _apply_port_state(board, {"ports": [{"coord": [1, 0], "port_owner": PLAYER_A, "port_controller": None}]})

    assert board.get_tile(AxialCoord(1, 0)).port_controller is None


def test_build_ship_histories_tracks_position_over_time():
    initial = _initial_record()
    moved = _half_turn([_ship(1, "destroyer", PLAYER_A, (1, 1)), _ship(2, "patrol_boat", PLAYER_B, (2, 0))], [])
    records = [initial, moved]

    histories = _build_ship_histories(records)

    assert histories[1].kind == ShipKind.DESTROYER
    assert histories[1].owner == PLAYER_A
    assert histories[1].positions == {0: AxialCoord(0, 0), 1: AxialCoord(1, 1)}
    assert histories[1].first_index == 0
    assert histories[1].last_index == 1


def test_build_ship_histories_detects_a_sunk_ship_by_its_absence():
    initial = _initial_record()
    # Ship 2 (the patrol boat) is gone from this record -- sunk.
    after_battle = _half_turn([_ship(1, "destroyer", PLAYER_A, (0, 0))], [])
    records = [initial, after_battle]

    histories = _build_ship_histories(records)

    assert histories[2].last_index == 0  # only ever appeared in the initial record
    assert 1 not in histories[2].positions


def test_record_label_variants():
    initial = _initial_record()
    half = _half_turn([], [])
    final_won = _final_record([], [], winner=PLAYER_A)
    final_undecided = _final_record([], [], winner=None)

    assert "Initial state" in _record_label(initial)
    assert "Player A's move (move_a)" in _record_label(half)
    assert "Winner: Player 1" in _record_label(final_won)
    assert "No winner recorded" in _record_label(final_undecided)


def test_path_color_for_stays_close_to_the_owner_color_but_varies_by_ship_id():
    from tla.rendering.hex_render import PLAYER_COLORS

    base = PLAYER_COLORS[PLAYER_A]
    color_a = _path_color_for(1, PLAYER_A)
    color_b = _path_color_for(2, PLAYER_A)

    assert color_a != color_b  # distinguishable per ship id
    # Lightened, not a totally different hue -- every channel moves toward white.
    for base_c, lightened_c in zip(base, color_a):
        assert lightened_c >= base_c


def test_ship_status_at_reflects_alive_then_sunk():
    initial = _initial_record()
    after_battle = _half_turn([_ship(1, "destroyer", PLAYER_A, (0, 0))], [])
    records = [initial, after_battle]
    view = ReplayView.__new__(ReplayView)
    view.records = records
    view.ship_histories = _build_ship_histories(records)
    view.hex_size = 18.0

    destroyer = view.ship_histories[1]
    patrol_boat = view.ship_histories[2]

    assert view._ship_status_at(destroyer, 1) == (AxialCoord(0, 0), False)
    assert view._ship_status_at(patrol_boat, 0) == (AxialCoord(2, 0), False)
    assert view._ship_status_at(patrol_boat, 1) == (AxialCoord(2, 0), True)  # sunk -- last known position


def test_inventory_rows_excludes_a_ship_once_it_sinks():
    initial = _initial_record()
    after_battle = _half_turn([_ship(1, "destroyer", PLAYER_A, (0, 0))], [])  # ship 2 gone -- sunk
    records = [initial, after_battle]
    view = ReplayView.__new__(ReplayView)
    view.records = records
    view.ship_histories = _build_ship_histories(records)

    view.cursor = 0
    assert view._inventory_rows() == [1, 2]

    view.cursor = 1
    assert view._inventory_rows() == [1]  # sunk ship 2 dropped off the list


def test_inventory_rows_excludes_a_ship_not_yet_produced():
    initial = _initial_record()
    produced = _half_turn(
        [_ship(1, "destroyer", PLAYER_A, (0, 0)), _ship(2, "patrol_boat", PLAYER_B, (2, 0)), _ship(3, "cruiser", PLAYER_A, (1, 1))],
        [],
    )
    records = [initial, produced]
    view = ReplayView.__new__(ReplayView)
    view.records = records
    view.ship_histories = _build_ship_histories(records)

    view.cursor = 0
    assert view._inventory_rows() == [1, 2]  # ship 3 doesn't exist yet

    view.cursor = 1
    assert view._inventory_rows() == [1, 3, 2]  # sorted by owner then id -- both PLAYER_A ships come first


class _IdentityCamera:
    """A stand-in for arcade.Camera2D -- unproject is the identity, so
    screen coordinates passed to _update_hover can be picked directly as
    a target hex's own pixel center (see axial_to_pixel) without needing a
    real camera/GL context in a headless test."""

    def unproject(self, screen_pos):
        return screen_pos


def _hover_view(records, cursor) -> ReplayView:
    view = ReplayView.__new__(ReplayView)
    view.records = records
    view.board = _board_from_initial(records[0])
    view.ship_histories = _build_ship_histories(records)
    view.hex_size = 18.0
    view.cursor = cursor
    view.camera = _IdentityCamera()
    return view


def test_update_hover_prefers_an_active_ship_over_a_sunk_one_sharing_a_hex():
    initial = _initial_record()  # ship 1 at (0,0), ship 2 at (2,0)
    # Ship 1 sinks in place at (0,0); ship 2 survives and moves onto that
    # same now-vacant hex.
    after = _half_turn([_ship(2, "patrol_boat", PLAYER_B, (0, 0), hp=2)], [])
    records = [initial, after]
    view = _hover_view(records, cursor=1)

    x, y = axial_to_pixel(AxialCoord(0, 0), view.hex_size)
    view._update_hover(x, y)

    assert view._hovered_ship_id == 2  # the active occupant, not the sunk destroyer


def test_update_hover_falls_back_to_a_sunk_ship_when_nothing_active_is_there():
    initial = _initial_record()
    after = _half_turn([_ship(2, "patrol_boat", PLAYER_B, (2, 0), hp=2)], [])  # ship 1 gone -- sunk at (0,0)
    records = [initial, after]
    view = _hover_view(records, cursor=1)

    x, y = axial_to_pixel(AxialCoord(0, 0), view.hex_size)
    view._update_hover(x, y)

    assert view._hovered_ship_id == 1  # still inspectable where it died


def test_set_cursor_clamps_inventory_scroll_when_the_row_count_shrinks():
    from tla.rendering.replay_view import INVENTORY_MAX_VISIBLE_ROWS

    initial = _initial_record()
    initial["ships"] = [_ship(i, "patrol_boat", PLAYER_A, (i, 0)) for i in range(1, INVENTORY_MAX_VISIBLE_ROWS + 3)]
    # Only ships 1 and 2 survive into this record -- the rest sunk.
    after = _half_turn(
        [_ship(1, "patrol_boat", PLAYER_A, (1, 0)), _ship(2, "patrol_boat", PLAYER_A, (2, 0))], []
    )
    records = [initial, after]
    view = _make_view(records, cursor=0)
    view.belief_reader = object()  # any non-None stand-in -- _set_cursor only checks it's not None
    view._inventory_scroll = 10  # deep into the original, longer list

    view._set_cursor(1)

    assert view._inventory_scroll == 0  # only 2 rows left now -- nothing to scroll to


def test_path_points_only_includes_positions_up_to_the_cursor():
    initial = _initial_record()
    moved = _half_turn([_ship(1, "destroyer", PLAYER_A, (1, 1)), _ship(2, "patrol_boat", PLAYER_B, (2, 0))], [])
    moved_again = _half_turn(
        [_ship(1, "destroyer", PLAYER_A, (1, 2)), _ship(2, "patrol_boat", PLAYER_B, (2, 0))], []
    )
    records = [initial, moved, moved_again]
    view = ReplayView.__new__(ReplayView)
    view.records = records
    view.ship_histories = _build_ship_histories(records)
    view.hex_size = 18.0

    destroyer = view.ship_histories[1]

    assert len(view._path_points(destroyer, 0)) == 1
    assert len(view._path_points(destroyer, 1)) == 2
    assert len(view._path_points(destroyer, 2)) == 3


def test_build_ship_histories_records_move_segments_from_move_log():
    initial = _initial_record()
    moved = _half_turn(
        [_ship(1, "destroyer", PLAYER_A, (2, -1)), _ship(2, "patrol_boat", PLAYER_B, (2, 0))],
        [],
        move_log=[_move(1, "destroyer", PLAYER_A, [(0, 0), (1, -1), (2, -1)])],
    )
    histories = _build_ship_histories([initial, moved])

    assert histories[1].move_segments == {1: [[AxialCoord(0, 0), AxialCoord(1, -1), AxialCoord(2, -1)]]}
    assert histories[2].move_segments == {}  # didn't move -- no move_log entry for it


def test_build_ship_histories_move_segments_merges_multiple_entries_for_the_same_ship():
    initial = _initial_record()
    moved = _half_turn(
        [_ship(1, "destroyer", PLAYER_A, (2, 0)), _ship(2, "patrol_boat", PLAYER_B, (2, 0))],
        [],
        move_log=[
            _move(1, "destroyer", PLAYER_A, [(0, 0), (1, 0)]),
            _move(1, "destroyer", PLAYER_A, [(1, 0), (2, 0)]),  # e.g. approach then a capture step
        ],
    )
    histories = _build_ship_histories([initial, moved])

    # Both entries kept as separate segments (in order) for the record --
    # _step_animation_queue is what merges them for playback purposes.
    assert histories[1].move_segments[1] == [
        [AxialCoord(0, 0), AxialCoord(1, 0)],
        [AxialCoord(1, 0), AxialCoord(2, 0)],
    ]


def test_path_points_uses_the_fine_grained_route_when_move_log_is_present():
    initial = _initial_record()
    moved = _half_turn(
        [_ship(1, "destroyer", PLAYER_A, (2, -1)), _ship(2, "patrol_boat", PLAYER_B, (2, 0))],
        [],
        move_log=[_move(1, "destroyer", PLAYER_A, [(0, 0), (1, -1), (2, -1)])],
    )
    view = ReplayView.__new__(ReplayView)
    view.records = [initial, moved]
    view.ship_histories = _build_ship_histories(view.records)
    view.hex_size = 18.0

    points = view._path_points(view.ship_histories[1], 1)

    # 3 hexes traveled (not just the 2-point straight jump the old,
    # snapshot-only behavior would have given).
    assert len(points) == 3


def test_path_points_falls_back_to_a_straight_hop_without_move_log():
    # An older replay file (written before move_log existed) has no
    # move_log for this record at all -- the path should still render,
    # just as the old coarse straight-line jump between snapshots.
    initial = _initial_record()
    moved = _half_turn([_ship(1, "destroyer", PLAYER_A, (2, -1)), _ship(2, "patrol_boat", PLAYER_B, (2, 0))], [])
    del moved["move_log"]
    view = ReplayView.__new__(ReplayView)
    view.records = [initial, moved]
    view.ship_histories = _build_ship_histories(view.records)
    view.hex_size = 18.0

    points = view._path_points(view.ship_histories[1], 1)

    assert len(points) == 2


def test_step_animation_queue_builds_one_entry_per_ship_in_move_order():
    record = _half_turn(
        [],
        [],
        move_log=[
            _move(2, "patrol_boat", PLAYER_B, [(2, 0), (2, 1)]),
            _move(1, "destroyer", PLAYER_A, [(0, 0), (1, 0)]),
        ],
    )
    queue = _step_animation_queue(record)

    assert [sid for sid, _ in queue] == [2, 1]
    assert queue[0][1] == [AxialCoord(2, 0), AxialCoord(2, 1)]


def test_step_animation_queue_merges_adjacent_entries_for_the_same_ship():
    record = _half_turn(
        [],
        [],
        move_log=[
            _move(1, "destroyer", PLAYER_A, [(0, 0), (1, 0)]),
            _move(1, "destroyer", PLAYER_A, [(1, 0), (2, 0)]),
        ],
    )
    queue = _step_animation_queue(record)

    assert len(queue) == 1
    assert queue[0] == (1, [AxialCoord(0, 0), AxialCoord(1, 0), AxialCoord(2, 0)])


def test_step_animation_queue_is_empty_without_move_log():
    assert _step_animation_queue(_half_turn([], [])) == []


def test_step_animation_duration_scales_with_hex_count_but_has_a_floor():
    assert _step_animation_duration([AxialCoord(0, 0)]) == STEP_ANIM_MIN_SECONDS
    long_path = [AxialCoord(i, 0) for i in range(5)]
    assert _step_animation_duration(long_path) == 4 * STEP_ANIM_SECONDS_PER_HEX


def test_interpolate_path_hits_every_hex_center_exactly_at_its_fraction():
    path = [AxialCoord(0, 0), AxialCoord(1, 0), AxialCoord(2, 0)]
    start = _interpolate_path(path, 0.0, 18.0)
    midpoint = _interpolate_path(path, 0.5, 18.0)
    end = _interpolate_path(path, 1.0, 18.0)

    assert start == axial_to_pixel(AxialCoord(0, 0), 18.0)
    assert midpoint == axial_to_pixel(AxialCoord(1, 0), 18.0)
    assert end == axial_to_pixel(AxialCoord(2, 0), 18.0)


def test_interpolate_path_single_hex_is_stationary():
    assert _interpolate_path([AxialCoord(3, -1)], 0.7, 18.0) == _interpolate_path(
        [AxialCoord(3, -1)], 0.0, 18.0
    )


def _three_record_replay():
    initial = _initial_record()
    r1 = _half_turn([_ship(1, "destroyer", PLAYER_A, (1, 0)), _ship(2, "patrol_boat", PLAYER_B, (2, 0))], [])
    r2 = _half_turn([_ship(1, "destroyer", PLAYER_A, (1, 1)), _ship(2, "patrol_boat", PLAYER_B, (2, 0))], [])
    return [initial, r1, r2]


def test_set_play_direction_updates_state_and_resets_timer():
    view = _make_view(_three_record_replay())
    view._play_timer = 1.23

    view._set_play_direction(1)

    assert view._play_direction == 1
    assert view._play_timer == 0.0


def test_advance_playback_steps_forward_after_the_pacing_interval():
    view = _make_view(_three_record_replay(), cursor=0)
    view._set_play_direction(1)

    view._advance_playback(PLAY_STEP_SECONDS / 2)
    assert view.cursor == 0  # not yet -- under the interval

    view._advance_playback(PLAY_STEP_SECONDS / 2 + 0.001)
    assert view.cursor == 1
    assert view._play_direction == 1  # keeps playing


def test_advance_playback_steps_backward_when_playing_reverse():
    view = _make_view(_three_record_replay(), cursor=2)
    view._set_play_direction(-1)

    view._advance_playback(PLAY_STEP_SECONDS + 0.001)

    assert view.cursor == 1
    assert view._play_direction == -1


def test_advance_playback_stops_itself_at_the_end():
    records = _three_record_replay()
    view = _make_view(records, cursor=len(records) - 1)
    view._set_play_direction(1)

    view._advance_playback(PLAY_STEP_SECONDS + 0.001)

    assert view.cursor == len(records) - 1  # didn't run off the end
    assert view._play_direction == 0  # stopped itself


def test_advance_playback_stops_itself_at_the_start():
    view = _make_view(_three_record_replay(), cursor=0)
    view._set_play_direction(-1)

    view._advance_playback(PLAY_STEP_SECONDS + 0.001)

    assert view.cursor == 0
    assert view._play_direction == 0


def test_advance_playback_does_nothing_while_stopped():
    view = _make_view(_three_record_replay(), cursor=0)

    view._advance_playback(10.0)  # way more than one interval

    assert view.cursor == 0


def test_space_steps_forward_one_half_turn_while_paused():
    view = _make_view(_three_record_replay(), cursor=0)

    view.on_key_press(arcade.key.SPACE, 0)

    assert view.cursor == 1
    assert view._play_direction == 0  # stepping doesn't start playback


def test_space_pauses_instead_of_stepping_while_playing():
    view = _make_view(_three_record_replay(), cursor=0)
    view._set_play_direction(1)

    view.on_key_press(arcade.key.SPACE, 0)

    assert view._play_direction == 0
    assert view.cursor == 0  # paused in place, did not also step


def test_l_toggles_forward_play():
    view = _make_view(_three_record_replay(), cursor=0)

    view.on_key_press(arcade.key.L, 0)
    assert view._play_direction == 1

    view.on_key_press(arcade.key.L, 0)  # pressing again pauses
    assert view._play_direction == 0


def test_j_and_l_play_reverse_and_forward():
    view = _make_view(_three_record_replay(), cursor=1)

    view.on_key_press(arcade.key.J, 0)
    assert view._play_direction == -1

    view.on_key_press(arcade.key.L, 0)
    assert view._play_direction == 1

    view.on_key_press(arcade.key.K, 0)
    assert view._play_direction == 0


def test_manual_stepping_stops_active_playback():
    view = _make_view(_three_record_replay(), cursor=1)
    view._set_play_direction(1)

    view.on_key_press(arcade.key.LEFT, 0)

    assert view._play_direction == 0
    assert view.cursor == 0


def test_home_and_end_stop_active_playback():
    records = _three_record_replay()
    view = _make_view(records, cursor=1)
    view._set_play_direction(-1)

    view.on_key_press(arcade.key.END, 0)

    assert view._play_direction == 0
    assert view.cursor == len(records) - 1


def _two_ship_move_replay():
    """3 records: an initial state, a half-turn where two ships each move
    (ship 1 first, then ship 2 -- see its move_log order), and a further
    half-turn where neither moves. Used to exercise step-animation
    sequencing (_set_cursor's animate flag, _advance_step_animation,
    _ship_render_status)."""
    initial = {
        "type": "initial",
        "seed": 1,
        "board": {
            "width": 5,
            "height": 1,
            "hex_pixel_size": 18.0,
            "tiles": [_tile((q, 0)) for q in range(5)],
        },
        "ships": [_ship(1, "destroyer", PLAYER_A, (0, 0)), _ship(2, "patrol_boat", PLAYER_A, (4, 0))],
    }
    r1 = _half_turn(
        [_ship(1, "destroyer", PLAYER_A, (2, 0)), _ship(2, "patrol_boat", PLAYER_A, (3, 0))],
        [],
        move_log=[
            _move(1, "destroyer", PLAYER_A, [(0, 0), (1, 0), (2, 0)]),
            _move(2, "patrol_boat", PLAYER_A, [(4, 0), (3, 0)]),
        ],
    )
    r2 = _half_turn([_ship(1, "destroyer", PLAYER_A, (2, 0)), _ship(2, "patrol_boat", PLAYER_A, (3, 0))], [])
    return [initial, r1, r2]


def test_set_cursor_animate_starts_a_queue_for_a_genuine_forward_step():
    view = _make_view(_two_ship_move_replay(), cursor=0)

    view._set_cursor(1, animate=True)

    assert view.cursor == 1
    assert view._step_animation is not None
    assert [sid for sid, _ in view._step_animation.queue] == [1, 2]  # move order, not id order


def test_set_cursor_animate_ignored_for_a_jump_of_more_than_one_record():
    view = _make_view(_two_ship_move_replay(), cursor=0)

    view._set_cursor(2, animate=True)  # skips record 1 entirely

    assert view.cursor == 2
    assert view._step_animation is None


def test_set_cursor_without_animate_cancels_an_in_progress_animation():
    view = _make_view(_two_ship_move_replay(), cursor=0)
    view._set_cursor(1, animate=True)
    assert view._step_animation is not None

    view._set_cursor(0)  # step back, no animate

    assert view._step_animation is None


def test_set_cursor_animate_on_a_record_with_no_moves_leaves_no_animation():
    view = _make_view(_two_ship_move_replay(), cursor=1)

    view._set_cursor(2, animate=True)  # r2's move_log is empty

    assert view._step_animation is None


def test_advance_step_animation_progresses_then_moves_to_the_next_ship():
    view = _make_view(_two_ship_move_replay(), cursor=0)
    view._set_cursor(1, animate=True)
    anim = view._step_animation
    duration = _step_animation_duration(anim.queue[0][1])  # ship 1's 2-hex route

    view._advance_step_animation(duration / 2)
    assert view._step_animation is anim
    assert view._step_animation.index == 0

    view._advance_step_animation(duration / 2)  # finishes ship 1's segment
    assert view._step_animation.index == 1
    assert view._step_animation.elapsed == 0.0


def test_advance_step_animation_clears_itself_once_the_whole_queue_finishes():
    view = _make_view(_two_ship_move_replay(), cursor=0)
    view._set_cursor(1, animate=True)
    total = sum(_step_animation_duration(path) for _, path in view._step_animation.queue)

    view._advance_step_animation(total + 0.01)

    assert view._step_animation is None


def test_advance_step_animation_does_nothing_when_no_animation_is_active():
    view = _make_view(_two_ship_move_replay(), cursor=1)

    view._advance_step_animation(1.0)  # no crash, no-op

    assert view._step_animation is None


def test_ship_render_status_shows_a_pending_ship_at_its_pre_step_position():
    view = _make_view(_two_ship_move_replay(), cursor=0)
    view._set_cursor(1, animate=True)  # queue: [1, 2] -- ship 2 hasn't gone yet

    center, sunk, _ = view._ship_render_status(view.ship_histories[2])

    assert center == axial_to_pixel(AxialCoord(4, 0), 18.0)  # its position before this step, not (3, 0)
    assert sunk is False


def test_ship_render_status_shows_the_active_ship_interpolated_along_its_route():
    view = _make_view(_two_ship_move_replay(), cursor=0)
    view._set_cursor(1, animate=True)  # queue starts on ship 1: (0,0) -> (1,0) -> (2,0)
    duration = _step_animation_duration(view._step_animation.queue[0][1])

    view._advance_step_animation(duration / 2)  # halfway through its 2-hex route
    center, _, _ = view._ship_render_status(view.ship_histories[1])

    assert center == axial_to_pixel(AxialCoord(1, 0), 18.0)


def test_ship_render_status_shows_a_completed_ship_at_its_real_final_position():
    view = _make_view(_two_ship_move_replay(), cursor=0)
    view._set_cursor(1, animate=True)
    duration_ship1 = _step_animation_duration(view._step_animation.queue[0][1])

    view._advance_step_animation(duration_ship1)  # ship 1's segment finishes, ship 2 now active
    center, sunk, _ = view._ship_render_status(view.ship_histories[1])

    assert center == axial_to_pixel(AxialCoord(2, 0), 18.0)
    assert sunk is False


def test_ship_render_status_matches_normal_status_when_nothing_is_animating():
    view = _make_view(_two_ship_move_replay(), cursor=1)  # no active animation

    center, sunk, _ = view._ship_render_status(view.ship_histories[1])

    assert center == axial_to_pixel(AxialCoord(2, 0), 18.0)
    assert sunk is False


def test_right_key_steps_forward_with_animation():
    view = _make_view(_two_ship_move_replay(), cursor=0)

    view.on_key_press(arcade.key.RIGHT, 0)

    assert view.cursor == 1
    assert view._step_animation is not None


def test_left_key_steps_backward_without_animation():
    view = _make_view(_two_ship_move_replay(), cursor=1)

    view.on_key_press(arcade.key.LEFT, 0)

    assert view.cursor == 0
    assert view._step_animation is None


class _FakeWindow:
    """Just enough of arcade.Window for _button_rects, which only reads
    .width -- avoids needing a live Arcade window for these tests."""

    def __init__(self, width=800):
        self.width = width


def test_button_rects_are_centered_and_in_order():
    view = _make_view(_three_record_replay())
    view.window = _FakeWindow(width=800)
    view._mouse_screen_pos = (0.0, 0.0)

    rects = view._button_rects()

    assert [r[0] for r in rects] == ["begin", "step_back", "play", "pause", "step_forward", "end"]
    total_width = 6 * 40.0 + 5 * 5.0
    assert rects[0][1] == (800 - total_width) / 2  # centered horizontally
    for (_, left, _, w, _), (_, next_left, *_) in zip(rects, rects[1:]):
        assert next_left == left + w + 5.0  # laid out left to right with the gap


def test_button_at_hits_the_right_button_and_misses_outside():
    view = _make_view(_three_record_replay())
    view.window = _FakeWindow(width=800)

    _, left, bottom, w, h = view._button_rects()[2]  # "play"
    assert view._button_at(left + w / 2, bottom + h / 2) == "play"
    assert view._button_at(-100, -100) is None


def test_handle_button_click_begin_and_end():
    records = _three_record_replay()
    view = _make_view(records, cursor=1)
    view.window = _FakeWindow(width=800)

    begin_rect = view._button_rects()[0]
    view._handle_button_click(begin_rect[1] + 1, begin_rect[2] + 1)
    assert view.cursor == 0

    end_rect = view._button_rects()[5]
    view._handle_button_click(end_rect[1] + 1, end_rect[2] + 1)
    assert view.cursor == len(records) - 1


def test_handle_button_click_step_back_and_step_forward():
    records = _three_record_replay()
    view = _make_view(records, cursor=1)
    view.window = _FakeWindow(width=800)

    step_back_rect = view._button_rects()[1]
    view._handle_button_click(step_back_rect[1] + 1, step_back_rect[2] + 1)
    assert view.cursor == 0

    step_forward_rect = view._button_rects()[4]
    view._handle_button_click(step_forward_rect[1] + 1, step_forward_rect[2] + 1)
    assert view.cursor == 1


def test_handle_button_click_play_and_pause():
    view = _make_view(_three_record_replay(), cursor=0)
    view.window = _FakeWindow(width=800)

    play_rect = view._button_rects()[2]
    view._handle_button_click(play_rect[1] + 1, play_rect[2] + 1)
    assert view._play_direction == 1

    pause_rect = view._button_rects()[3]
    view._handle_button_click(pause_rect[1] + 1, pause_rect[2] + 1)
    assert view._play_direction == 0


def test_handle_button_click_pause_does_nothing_while_already_stopped():
    view = _make_view(_three_record_replay(), cursor=1)
    view.window = _FakeWindow(width=800)
    assert view._play_direction == 0  # pause is disabled in this state

    pause_rect = view._button_rects()[3]
    view._handle_button_click(pause_rect[1] + 1, pause_rect[2] + 1)

    assert view._play_direction == 0
    assert view.cursor == 1


def test_handle_button_click_outside_any_button_does_nothing():
    view = _make_view(_three_record_replay(), cursor=1)
    view.window = _FakeWindow(width=800)

    view._handle_button_click(-500, -500)

    assert view.cursor == 1
    assert view._play_direction == 0


def _round(
    *,
    attacker_owner=PLAYER_A,
    defender_owner=PLAYER_B,
    damage_to_defender=4,
    damage_to_attacker=2,
    attacker_sunk=False,
    defender_sunk=False,
):
    return {
        "attacker_id": 1,
        "attacker_kind": "destroyer",
        "attacker_owner": attacker_owner,
        "defender_id": 2,
        "defender_kind": "cruiser",
        "defender_owner": defender_owner,
        "battle_hex": [0, 0],
        "damage_to_defender": damage_to_defender,
        "damage_to_attacker": damage_to_attacker,
        "attacker_carrier_bonus": 0,
        "defender_carrier_bonus": 0,
        "defender_hp_after": 4,
        "attacker_hp_after": 4,
        "defender_sunk": defender_sunk,
        "attacker_sunk": attacker_sunk,
    }


def test_compute_tally_sums_hp_dealt_and_taken_for_both_sides():
    records = [{"battle_log": [_round(damage_to_defender=5, damage_to_attacker=3)]}]

    tally = _compute_tally(records, 0)

    assert tally.hp_dealt[PLAYER_A] == 5
    assert tally.hp_taken[PLAYER_A] == 3
    assert tally.hp_dealt[PLAYER_B] == 3
    assert tally.hp_taken[PLAYER_B] == 5
    assert tally.ships_sunk == {PLAYER_A: 0, PLAYER_B: 0}
    assert tally.ships_lost == {PLAYER_A: 0, PLAYER_B: 0}


def test_compute_tally_credits_a_sink_to_both_sides_correctly():
    records = [{"battle_log": [_round(defender_sunk=True)]}]

    tally = _compute_tally(records, 0)

    assert tally.ships_sunk[PLAYER_A] == 1  # attacker sank the defender's ship
    assert tally.ships_lost[PLAYER_B] == 1  # defender lost their own ship
    assert tally.ships_sunk[PLAYER_B] == 0
    assert tally.ships_lost[PLAYER_A] == 0


def test_compute_tally_handles_a_mutual_kill():
    records = [{"battle_log": [_round(attacker_sunk=True, defender_sunk=True)]}]

    tally = _compute_tally(records, 0)

    assert tally.ships_lost[PLAYER_A] == 1
    assert tally.ships_sunk[PLAYER_B] == 1
    assert tally.ships_lost[PLAYER_B] == 1
    assert tally.ships_sunk[PLAYER_A] == 1


def test_compute_tally_accumulates_across_records_up_to_the_cursor():
    records = [
        {"battle_log": [_round(damage_to_defender=5, damage_to_attacker=1)]},
        {"battle_log": [_round(damage_to_defender=3, damage_to_attacker=2, defender_sunk=True)]},
        {"battle_log": [_round(damage_to_defender=100, damage_to_attacker=100)]},  # beyond the cursor
    ]

    tally = _compute_tally(records, 1)

    assert tally.hp_dealt[PLAYER_A] == 8  # 5 + 3, not the third record's 100
    assert tally.ships_sunk[PLAYER_A] == 1


def test_compute_tally_treats_a_missing_battle_log_as_no_combat():
    records = [{"turn_number": 1}]  # a replay written before battle_log existed

    tally = _compute_tally(records, 0)

    assert tally.hp_dealt == {PLAYER_A: 0, PLAYER_B: 0}


def test_set_cursor_updates_the_tally():
    records = _three_record_replay()
    records[1]["battle_log"] = [_round(damage_to_defender=6)]
    view = _make_view(records, cursor=0)

    view._set_cursor(1)
    assert view.tally.hp_dealt[PLAYER_A] == 6

    view._set_cursor(0)
    assert view.tally.hp_dealt[PLAYER_A] == 0  # stepping back drops the not-yet-happened battle
