"""Graphical, read-only replay viewer.

Reconstructs the board and every ship's full position history from a
parsed .jsonl replay (see tla.replay) and draws them -- including a path
line per ship -- reusing the same rendering building blocks the live game
uses (tla.rendering.hex_render/ship_glyphs). Never mutates anything: no
legal-move checking or game rules are involved here, just playback of an
already-finished game.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import arcade

from tla.board import Board
from tla.hexgrid import AxialCoord, axial_to_pixel, pixel_to_axial
from tla.rendering.hex_render import PLAYER_COLORS, _lighten, board_pixel_bounds, draw_board
from tla.rendering.ship_glyphs import draw_ship_glyph
from tla.ship import ShipKind
from tla.tile import PLAYER_A, PLAYER_B, PlayerId, Tile, TerrainType

PAN_SPEED = 600.0
MIN_ZOOM = 0.2
MAX_ZOOM = 4.0
# LEFT/RIGHT are reserved for turn-stepping (see on_key_press) -- panning
# uses WASD and the vertical arrows only.
_PAN_KEYS = {
    arcade.key.A: (-1, 0),
    arcade.key.D: (1, 0),
    arcade.key.UP: (0, 1),
    arcade.key.W: (0, 1),
    arcade.key.DOWN: (0, -1),
    arcade.key.S: (0, -1),
}

SUNK_MARKER_COLOR = (20, 20, 20)
PATH_LINE_WIDTH = 2

# Seconds spent on each record while auto-playing (see _set_play_direction/
# on_update) -- matches AiConfig.turn_pacing_seconds's default, the same
# "watch it happen at a comfortable pace" feel the live game's AI turn uses.
PLAY_STEP_SECONDS = 0.35

# Stepping forward one half-turn (the step_forward button, or Right) plays
# a short animation instead of an instant jump: each ship that moved that
# half-turn glides along its actual route, one ship at a time, in the exact
# order it was moved -- see _StepAnimation/ReplayView._start_step_animation.
STEP_ANIM_SECONDS_PER_HEX = 0.15
STEP_ANIM_MIN_SECONDS = 0.15  # floor per ship, so even a 1-hex move is visible

TOOLTIP_BG_COLOR = (25, 25, 25, 230)
TOOLTIP_TEXT_COLOR = arcade.color.WHITE
TOOLTIP_PADDING = 10
TOOLTIP_LINE_HEIGHT = 18
TOOLTIP_WIDTH = 170
TOOLTIP_OFFSET = 16

# Transport control buttons drawn bottom-center -- see
# _draw_transport_controls/_button_rects. Six buttons, left to right: go to
# the beginning, step back one half-turn, play forward, pause (only
# meaningful -- and only drawn enabled -- while actually playing), step
# forward one half-turn, go to the end. Each icon is a small row of bar/
# triangle primitives (see _draw_icon_row/ICON_ELEMENTS) composed to match
# the user's own exact spec for each button.
BUTTON_WIDTH = 40.0
BUTTON_HEIGHT = 28.0
BUTTON_GAP = 5.0
BUTTON_MARGIN_BOTTOM = 10.0
BUTTON_BG_COLOR = (45, 45, 45, 230)
BUTTON_BG_HOVER_COLOR = (80, 80, 80, 230)
BUTTON_BG_DISABLED_COLOR = (32, 32, 32, 230)
BUTTON_ICON_COLOR = arcade.color.WHITE
BUTTON_ICON_DISABLED_COLOR = (90, 90, 90)
BUTTON_NAMES = ("begin", "step_back", "play", "pause", "step_forward", "end")
# Each element is "bar" or a triangle direction (1 = right, -1 = left),
# laid out left to right by _draw_icon_row.
ICON_ELEMENTS: dict[str, list] = {
    "begin": ["bar", "bar", -1],
    "step_back": ["bar", -1],
    "play": [1],
    "pause": ["bar", "bar"],
    "step_forward": [1, "bar"],
    "end": [1, "bar", "bar"],
}

# Running combat tally, top-right -- see _compute_tally/_draw_tally.
TALLY_BG_COLOR = (25, 25, 25, 200)
TALLY_WIDTH = 320.0
TALLY_LINE_HEIGHT = 18.0
TALLY_PADDING = 8.0
TALLY_MARGIN = 10.0


@dataclass
class _ShipHistory:
    id: int
    kind: ShipKind
    owner: PlayerId
    # Record index (position in the parsed records list, not turn_number --
    # simpler to reason about since it's a single, densely-increasing
    # sequence regardless of half-turn/final record boundaries) -> that
    # ship's position/hp as of that record. A ship is present at every
    # index in [min(positions), max(positions)] (ships never reappear once
    # gone), so "alive at cursor" is just a range check -- see
    # ReplayView._ship_status_at.
    positions: dict[int, AxialCoord] = field(default_factory=dict)
    hp: dict[int, int] = field(default_factory=dict)
    surfaced: dict[int, bool] = field(default_factory=dict)
    # Record index -> the hex-by-hex route(s) this ship actually traveled
    # during that half-turn (see tla.game_state.MoveLogEntry) -- usually 0
    # or 1 segments, occasionally 2 (an engagement's approach, then a
    # separate final capture step). Absent for a record written before
    # move_log existed, or a half-turn this ship simply didn't move --
    # _path_points/ReplayView._start_step_animation both degrade gracefully
    # when this is empty (see their own docstrings).
    move_segments: dict[int, list[list[AxialCoord]]] = field(default_factory=dict)

    @property
    def first_index(self) -> int:
        return min(self.positions)

    @property
    def last_index(self) -> int:
        return max(self.positions)


@dataclass
class _StepAnimation:
    """In-progress ship-by-ship animation of one forward step (see
    ReplayView._start_step_animation/_advance_step_animation). `queue` is
    one entry per moving ship, `(ship_id, path)`, in the exact order those
    ships were moved that half-turn; `index` is which entry is currently
    animating, `elapsed` how far into it (seconds)."""

    queue: list[tuple[int, list[AxialCoord]]]
    index: int = 0
    elapsed: float = 0.0

    @property
    def current(self) -> tuple[int, list[AxialCoord]]:
        return self.queue[self.index]

    @property
    def pending_ship_ids(self) -> set[int]:
        """Ships still waiting their turn -- not yet started this step's
        animation, so they must keep rendering at their pre-step position
        (see _step_animation_start_position), not the destination cursor
        already reflects."""
        return {sid for sid, _ in self.queue[self.index + 1 :]}


def _step_animation_queue(record: dict) -> list[tuple[int, list[AxialCoord]]]:
    """Builds the animation order for one half-turn record's `move_log`:
    one queue entry per ship, in move order, with consecutive entries for
    the same ship (an engagement's approach immediately followed by its
    capture step -- see MoveLogEntry) merged into a single continuous
    route rather than animated as two separate turns for that ship."""
    queue: list[tuple[int, list[AxialCoord]]] = []
    for move in record.get("move_log", []):
        sid = move["ship_id"]
        path = [AxialCoord(*c) for c in move["path"]]
        if queue and queue[-1][0] == sid:
            prev_path = queue[-1][1]
            extra = path[1:] if prev_path and path and prev_path[-1] == path[0] else path
            queue[-1] = (sid, prev_path + extra)
        else:
            queue.append((sid, path))
    return queue


def _step_animation_duration(path: list[AxialCoord]) -> float:
    return max(STEP_ANIM_MIN_SECONDS, (len(path) - 1) * STEP_ANIM_SECONDS_PER_HEX)


def _interpolate_path(path: list[AxialCoord], progress: float, hex_size: float) -> tuple[float, float]:
    """The pixel position `progress` (0-1) of the way along `path` -- hex
    centers on a regular hex grid are always equidistant from their
    neighbors, so interpolating piecewise-linearly in pixel space between
    consecutive hexes, evenly by hex count, gives constant-speed movement
    with no extra easing needed."""
    if len(path) < 2:
        return axial_to_pixel(path[0], hex_size)
    segments = len(path) - 1
    t = max(0.0, min(1.0, progress)) * segments
    seg_index = min(int(t), segments - 1)
    seg_t = t - seg_index
    x0, y0 = axial_to_pixel(path[seg_index], hex_size)
    x1, y1 = axial_to_pixel(path[seg_index + 1], hex_size)
    return (x0 + (x1 - x0) * seg_t, y0 + (y1 - y0) * seg_t)


def _path_color_for(ship_id: int, owner: PlayerId) -> tuple[int, int, int]:
    """The owner's base color, nudged lighter by a small amount that
    depends on the ship's own id -- so several same-owner ships whose
    paths cross stay visually distinguishable without needing a full
    legend, while still reading unmistakably as "that player's ship"."""
    amount = (ship_id % 7) / 18.0  # small, deterministic spread: 0.0-0.33
    return _lighten(PLAYER_COLORS[owner], amount)


def _tile_from_dict(d: dict) -> Tile:
    coord = AxialCoord(*d["coord"])
    return Tile(
        coord=coord,
        terrain=TerrainType(d["terrain"]),
        is_port=d["is_port"],
        port_owner=d.get("port_owner"),
        port_controller=d.get("port_controller"),
    )


def _board_from_initial(record: dict) -> Board:
    board_data = record["board"]
    board = Board(
        width=board_data["width"],
        height=board_data["height"],
        hex_pixel_size=board_data.get("hex_pixel_size", 18.0),
    )
    for tile_dict in board_data["tiles"]:
        tile = _tile_from_dict(tile_dict)
        board.tiles[tile.coord] = tile
    return board


def _apply_port_state(board: Board, record: dict) -> None:
    """Update every port tile's port_owner/port_controller to match
    `record`'s own `ports` field -- called on every cursor change (not
    just forward steps), so stepping backward through the replay shows
    port control exactly as it was at that point, not a stale forward
    patch."""
    for port in record.get("ports", []):
        coord = AxialCoord(*port["coord"])
        tile = board.tiles.get(coord)
        if tile is None:
            continue
        tile.port_owner = port.get("port_owner")
        tile.port_controller = port.get("port_controller")


def _build_ship_histories(records: list[dict]) -> dict[int, _ShipHistory]:
    histories: dict[int, _ShipHistory] = {}
    for index, record in enumerate(records):
        for ship_dict in record.get("ships", []):
            sid = ship_dict["id"]
            history = histories.get(sid)
            if history is None:
                history = _ShipHistory(id=sid, kind=ShipKind(ship_dict["kind"]), owner=ship_dict["owner"])
                histories[sid] = history
            history.positions[index] = AxialCoord(*ship_dict["position"])
            history.hp[index] = ship_dict["hp"]
            history.surfaced[index] = ship_dict["surfaced"]
        for move in record.get("move_log", []):
            sid = move["ship_id"]
            history = histories.get(sid)
            if history is None:
                history = _ShipHistory(id=sid, kind=ShipKind(move["kind"]), owner=move["owner"])
                histories[sid] = history
            path = [AxialCoord(*c) for c in move["path"]]
            history.move_segments.setdefault(index, []).append(path)
    return histories


@dataclass
class _Tally:
    """Running combat totals, both players -- see _compute_tally."""

    hp_dealt: dict[PlayerId, int] = field(default_factory=lambda: {PLAYER_A: 0, PLAYER_B: 0})
    hp_taken: dict[PlayerId, int] = field(default_factory=lambda: {PLAYER_A: 0, PLAYER_B: 0})
    ships_sunk: dict[PlayerId, int] = field(default_factory=lambda: {PLAYER_A: 0, PLAYER_B: 0})
    ships_lost: dict[PlayerId, int] = field(default_factory=lambda: {PLAYER_A: 0, PLAYER_B: 0})


def _compute_tally(records: list[dict], cursor: int) -> _Tally:
    """Sums every round in `records[0..cursor]`'s `battle_log` (see
    GameState.battle_log/tla.replay) into running per-player totals --
    recomputed whenever the cursor changes (see ReplayView._set_cursor),
    not cached across the whole game, so stepping backward always reflects
    exactly what had happened by that point, not the final tally. Missing
    `battle_log` (a replay written before that field existed) is treated
    as no combat recorded for that record, not an error."""
    tally = _Tally()
    for record in records[: cursor + 1]:
        for entry in record.get("battle_log", []):
            a_owner, d_owner = entry["attacker_owner"], entry["defender_owner"]
            tally.hp_dealt[a_owner] += entry["damage_to_defender"]
            tally.hp_taken[a_owner] += entry["damage_to_attacker"]
            tally.hp_dealt[d_owner] += entry["damage_to_attacker"]
            tally.hp_taken[d_owner] += entry["damage_to_defender"]
            if entry["attacker_sunk"]:
                tally.ships_lost[a_owner] += 1
                tally.ships_sunk[d_owner] += 1
            if entry["defender_sunk"]:
                tally.ships_lost[d_owner] += 1
                tally.ships_sunk[a_owner] += 1
    return tally


def _record_label(record: dict) -> str:
    record_type = record.get("type")
    if record_type == "initial":
        return f"Initial state (seed={record.get('seed')})"
    if record_type == "final":
        winner = record.get("winner")
        outcome = f"Winner: Player {winner}" if winner is not None else "No winner recorded"
        return f"Turn {record['turn_number']} -- GAME OVER -- {outcome}"
    player_label = "Player A" if record["player"] == PLAYER_A else "Player B"
    return f"Turn {record['turn_number']} -- {player_label}'s move ({record['phase']})"


class ReplayView(arcade.View):
    def __init__(self, records: list[dict], hex_size: float | None = None) -> None:
        super().__init__()
        if not records or records[0].get("type") != "initial":
            raise ValueError("replay records must start with an 'initial' record")
        self.records = records
        self.board = _board_from_initial(records[0])
        self.hex_size = hex_size if hex_size is not None else self.board.hex_pixel_size
        self.ship_histories = _build_ship_histories(records)

        min_x, min_y, max_x, max_y = board_pixel_bounds(self.board, self.hex_size)
        self._board_pixel_bounds = (min_x, min_y, max_x, max_y)
        self.camera = arcade.Camera2D(position=((min_x + max_x) / 2, (min_y + max_y) / 2))
        self.ui_camera = arcade.Camera2D()

        # In-progress ship-by-ship "step forward" animation, or None --
        # see _StepAnimation/_start_step_animation/_advance_step_animation.
        # Set before the first _set_cursor call since that method clears it.
        self._step_animation: _StepAnimation | None = None

        # Index into `records` -- defaults to the *last* record, so the
        # first thing shown is the whole game's paths end-to-end; Left/
        # Right step it one record at a time (see on_key_press). Also sets
        # self.tally (see _set_cursor/_compute_tally).
        self._set_cursor(len(records) - 1)

        # VCR-style auto-play: 0 = stopped, 1 = playing forward, -1 =
        # playing reverse -- see _set_play_direction/on_update. Any manual
        # step/jump (on_key_press's Left/Right/Home/End) stops it.
        self._play_direction = 0
        self._play_timer = 0.0

        self._held_pan_keys: set[int] = set()
        self._dragging = False
        self._mouse_screen_pos = (0.0, 0.0)
        self._hovered_ship_id: int | None = None
        # The hex currently under the cursor, if it's a real board tile and
        # no ship is hovered there (see _update_hover/on_draw) -- lets a
        # reader pin down exactly which hex an odd move/battle happened at
        # without having to count hexes by eye. None off the board, or
        # whenever a ship's own tooltip already covers that hex.
        self._hovered_empty_hex: AxialCoord | None = None

        self._hud_text = arcade.Text("", 10, 0, arcade.color.WHITE, 13)
        self._tooltip_texts = [arcade.Text("", 0, 0, TOOLTIP_TEXT_COLOR, 12) for _ in range(4)]
        self._hex_coord_text = arcade.Text("", 0, 0, TOOLTIP_TEXT_COLOR, 12)

    # -- cursor / ship status -------------------------------------------------

    def _set_cursor(self, index: int, *, animate: bool = False) -> None:
        """Ports/tally/HUD always update immediately to the new record's
        actual state, regardless of `animate` -- only the *ship glyphs'*
        rendered positions lag behind while a step animation plays (see
        on_draw), catching up ship by ship. `animate=True` only ever
        starts an animation for a genuine single-step-forward (`index ==
        self.cursor + 1`); any other cursor change (a jump, a step back,
        or even a forward jump of more than one record) cancels whatever
        animation was in progress instead, snapping straight to the new
        state -- there's no sensible "order ships moved" to play across a
        jump spanning more than one half-turn."""
        new_index = max(0, min(len(self.records) - 1, index))
        if animate and new_index == self.cursor + 1:
            self._start_step_animation(new_index)
        else:
            self._step_animation = None
        self.cursor = new_index
        _apply_port_state(self.board, self.records[self.cursor])
        self.tally = _compute_tally(self.records, self.cursor)

    def _start_step_animation(self, new_index: int) -> None:
        queue = _step_animation_queue(self.records[new_index])
        self._step_animation = _StepAnimation(queue=queue) if queue else None

    def _advance_step_animation(self, delta_time: float) -> None:
        anim = self._step_animation
        if anim is None:
            return
        anim.elapsed += delta_time
        while anim is not None:
            duration = _step_animation_duration(anim.current[1])
            if anim.elapsed < duration:
                return
            anim.elapsed -= duration
            anim.index += 1
            if anim.index >= len(anim.queue):
                self._step_animation = None
                return
            anim = self._step_animation

    def _ship_status_at(self, history: _ShipHistory, cursor: int) -> tuple[AxialCoord, bool] | None:
        """(position, is_sunk) for `history` as of `cursor`, or None if the
        ship hasn't been produced yet at this point in the replay."""
        if cursor < history.first_index:
            return None
        if cursor <= history.last_index:
            return history.positions[cursor], False
        return history.positions[history.last_index], True

    def _ship_render_status(self, history: _ShipHistory) -> tuple[tuple[float, float], bool, bool] | None:
        """(pixel center, is_sunk, is_submerged) to actually draw this
        frame -- like _ship_status_at, but folds in an in-progress step
        animation (see _StepAnimation): a ship not yet up in the queue
        renders at its pre-step position (the start of its own logged
        route, so it doesn't prematurely show this step's destination), the
        one currently animating renders interpolated along its route, and
        every other ship (already animated, or not part of this step's
        moves at all) renders at its normal, already-updated position.

        HP/sunk/submerged status is always read as of the (already-
        updated) cursor, even for a ship still pending or mid-animation --
        only *position* is staged across the step, not those. In
        particular a defender with no move_log entry of its own (it never
        moves) shows as sunk from the very start of the step's animation,
        not at the moment its attacker's animation actually reaches it --
        a deliberate simplification, not a bug: this animation shows real
        movement routes ship by ship, not a full re-staging of combat
        timing within the half-turn."""
        anim = self._step_animation
        submerged = history.kind == ShipKind.SUBMARINE and not history.surfaced.get(self.cursor, True)
        if anim is not None and history.id in anim.pending_ship_ids:
            path = next(p for sid, p in anim.queue if sid == history.id)
            return axial_to_pixel(path[0], self.hex_size), False, submerged
        if anim is not None and history.id == anim.current[0]:
            active_path = anim.current[1]
            duration = _step_animation_duration(active_path)
            progress = anim.elapsed / duration if duration > 0 else 1.0
            return _interpolate_path(active_path, progress, self.hex_size), False, submerged
        status = self._ship_status_at(history, self.cursor)
        if status is None:
            return None
        position, sunk = status
        return axial_to_pixel(position, self.hex_size), sunk, submerged

    def _path_points(self, history: _ShipHistory, cursor: int) -> list[tuple[float, float]]:
        """The full hex-by-hex route `history` actually traveled, up to
        `cursor` -- reads `history.move_segments` (see MoveLogEntry) for
        each record's real path instead of just connecting each record's
        start/end snapshot with a straight line. A record with no logged
        segment for this ship (either it didn't move that half-turn, or
        the replay predates move_log) falls back to a single straight hop
        to that record's snapshot position, exactly the old behavior --
        so an older replay file still renders a path, just a coarser one."""
        start = history.first_index
        if cursor < start or start not in history.positions:
            return []
        coords: list[AxialCoord] = [history.positions[start]]
        for index in range(start + 1, min(cursor, history.last_index) + 1):
            segments = history.move_segments.get(index)
            if segments:
                for segment in segments:
                    for coord in segment:
                        if coords[-1] != coord:
                            coords.append(coord)
            else:
                end = history.positions.get(index)
                if end is not None and coords[-1] != end:
                    coords.append(end)
        return [axial_to_pixel(c, self.hex_size) for c in coords]

    # -- drawing ---------------------------------------------------------------

    def on_draw(self) -> None:
        self.clear()
        self.camera.use()
        draw_board(self.board, self.hex_size)

        for history in self.ship_histories.values():
            points = self._path_points(history, self.cursor)
            if len(points) >= 2:
                arcade.draw_line_strip(points, _path_color_for(history.id, history.owner), PATH_LINE_WIDTH)

        for history in self.ship_histories.values():
            render = self._ship_render_status(history)
            if render is None:
                continue
            center, sunk, submerged = render
            if sunk:
                self._draw_sunk_marker(center)
            else:
                draw_ship_glyph(
                    center, self.hex_size, history.kind, PLAYER_COLORS[history.owner], submerged=submerged
                )

        self.ui_camera.use()
        self._hud_text.text = _record_label(self.records[self.cursor])
        self._hud_text.y = self.window.height - 22
        self._hud_text.draw()
        self._draw_transport_controls()
        self._draw_tally()

        hovered = self.ship_histories.get(self._hovered_ship_id) if self._hovered_ship_id is not None else None
        if hovered is not None:
            status = self._ship_status_at(hovered, self.cursor)
            if status is not None:
                self._draw_hover_tooltip(hovered, status)
        elif self._hovered_empty_hex is not None:
            self._draw_hex_coord_tooltip(self._hovered_empty_hex)

    def _draw_tally(self) -> None:
        """Running HP dealt/taken and ships sunk/lost, both players, as of
        the current cursor -- see _compute_tally."""
        lines = []
        for owner in (PLAYER_A, PLAYER_B):
            label = "Player A" if owner == PLAYER_A else "Player B"
            lines.append(
                f"{label}: HP dealt {self.tally.hp_dealt[owner]} / taken {self.tally.hp_taken[owner]}"
                f"    ships sunk {self.tally.ships_sunk[owner]} / lost {self.tally.ships_lost[owner]}"
            )
        height = TALLY_PADDING * 2 + TALLY_LINE_HEIGHT * len(lines)
        top = self.window.height - TALLY_MARGIN
        right = self.window.width - TALLY_MARGIN
        left = right - TALLY_WIDTH

        arcade.draw_lbwh_rectangle_filled(left, top - height, TALLY_WIDTH, height, TALLY_BG_COLOR)
        for i, (owner, line) in enumerate(zip((PLAYER_A, PLAYER_B), lines)):
            text_obj = arcade.Text(
                line, left + TALLY_PADDING, top - TALLY_PADDING - (i + 1) * TALLY_LINE_HEIGHT + 4, PLAYER_COLORS[owner], 11
            )
            text_obj.draw()

    # -- transport controls -------------------------------------------------

    def _button_rects(self) -> list[tuple[str, float, float, float, float]]:
        total_width = len(BUTTON_NAMES) * BUTTON_WIDTH + (len(BUTTON_NAMES) - 1) * BUTTON_GAP
        left = (self.window.width - total_width) / 2
        bottom = BUTTON_MARGIN_BOTTOM
        rects = []
        for name in BUTTON_NAMES:
            rects.append((name, left, bottom, BUTTON_WIDTH, BUTTON_HEIGHT))
            left += BUTTON_WIDTH + BUTTON_GAP
        return rects

    def _button_at(self, x: float, y: float) -> str | None:
        for name, left, bottom, w, h in self._button_rects():
            if left <= x <= left + w and bottom <= y <= bottom + h:
                return name
        return None

    def _draw_transport_controls(self) -> None:
        """Six buttons, bottom-center (see BUTTON_NAMES/ICON_ELEMENTS for
        exactly what each looks like), plus the "n/m" step counter (see
        the user's own request) right after them. Pause is only ever
        enabled while actually playing -- drawn dimmed and unresponsive to
        hover/click otherwise, since pausing something that isn't moving
        has nothing to do. A separate, always-available control -- reverse
        play -- has no button (not asked for) but stays reachable via the
        J key (K also pauses, L also plays forward)."""
        mouse_x, mouse_y = self._mouse_screen_pos
        rects = self._button_rects()
        for name, left, bottom, w, h in rects:
            disabled = name == "pause" and self._play_direction == 0
            hovered = not disabled and left <= mouse_x <= left + w and bottom <= mouse_y <= bottom + h
            if disabled:
                bg = BUTTON_BG_DISABLED_COLOR
            elif hovered:
                bg = BUTTON_BG_HOVER_COLOR
            else:
                bg = BUTTON_BG_COLOR
            arcade.draw_lbwh_rectangle_filled(left, bottom, w, h, bg)
            self._draw_button_icon(name, left, bottom, w, h, disabled=disabled)

        _, last_left, last_bottom, last_w, last_h = rects[-1]
        counter = arcade.Text(
            f"{self.cursor + 1}/{len(self.records)}",
            last_left + last_w + 12,
            last_bottom + last_h / 2,
            (190, 190, 190),
            13,
            anchor_y="center",
        )
        counter.draw()

    def _draw_triangle(self, cx: float, cy: float, height: float, direction: int, color) -> None:
        """A filled triangle centered at (cx, cy) -- direction=1 points
        right, -1 points left. `height` controls both its vertical extent
        and (scaled down) how far it reaches horizontally, independent of
        how many other icon elements it's sharing a button with -- see
        _draw_icon_row."""
        width = height * 0.9
        apex_x = cx + direction * width * 0.6
        base_x = cx - direction * width * 0.4
        arcade.draw_triangle_filled(apex_x, cy, base_x, cy - height / 2, base_x, cy + height / 2, color)

    def _draw_bar(self, cx: float, cy: float, height: float, color) -> None:
        width = max(2.0, height * 0.18)
        arcade.draw_lbwh_rectangle_filled(cx - width / 2, cy - height / 2, width, height, color)

    def _draw_icon_row(self, cx: float, cy: float, height: float, elements: list, color) -> None:
        """Lays `elements` (see ICON_ELEMENTS) out left to right as a
        group centered at (cx, cy), each a fixed `height` regardless of
        how many elements share the row -- so a single-element icon (play)
        and a three-element one (begin/end) read as the same visual weight,
        just packed differently, rather than the busier icon's pieces
        shrinking to fit."""
        step = 7.0
        start_x = cx - step * (len(elements) - 1) / 2
        for i, element in enumerate(elements):
            x = start_x + i * step
            if element == "bar":
                self._draw_bar(x, cy, height, color)
            else:
                self._draw_triangle(x, cy, height, element, color)

    def _draw_button_icon(self, name: str, left: float, bottom: float, w: float, h: float, disabled: bool = False) -> None:
        cx, cy = left + w / 2, bottom + h / 2
        color = BUTTON_ICON_DISABLED_COLOR if disabled else BUTTON_ICON_COLOR
        self._draw_icon_row(cx, cy, h * 0.5, ICON_ELEMENTS[name], color)

    def _draw_sunk_marker(self, center: tuple[float, float]) -> None:
        cx, cy = center
        r = self.hex_size * 0.35
        arcade.draw_line(cx - r, cy - r, cx + r, cy + r, SUNK_MARKER_COLOR, 3)
        arcade.draw_line(cx - r, cy + r, cx + r, cy - r, SUNK_MARKER_COLOR, 3)

    def _draw_hover_tooltip(self, history: _ShipHistory, status: tuple[AxialCoord, bool]) -> None:
        _, sunk = status
        lines = [
            history.kind.value.replace("_", " ").title(),
            f"Player {'A' if history.owner == PLAYER_A else 'B'}, id {history.id}",
            f"HP: {history.hp[self.cursor if self.cursor in history.hp else history.last_index]}",
            "Sunk" if sunk else "Active",
        ]
        height = TOOLTIP_PADDING * 2 + TOOLTIP_LINE_HEIGHT * len(lines)
        mouse_x, mouse_y = self._mouse_screen_pos
        left = mouse_x + TOOLTIP_OFFSET
        if left + TOOLTIP_WIDTH > self.window.width:
            left = mouse_x - TOOLTIP_OFFSET - TOOLTIP_WIDTH
        top = mouse_y + TOOLTIP_OFFSET + height
        if top > self.window.height:
            top = mouse_y - TOOLTIP_OFFSET

        arcade.draw_lbwh_rectangle_filled(left, top - height, TOOLTIP_WIDTH, height, TOOLTIP_BG_COLOR)
        arcade.draw_lbwh_rectangle_filled(left, top - 4, TOOLTIP_WIDTH, 4, PLAYER_COLORS[history.owner])
        for i, line in enumerate(lines):
            text_obj = self._tooltip_texts[i]
            text_obj.text = line
            text_obj.x = left + TOOLTIP_PADDING
            text_obj.y = top - TOOLTIP_PADDING - (i + 1) * TOOLTIP_LINE_HEIGHT + 4
            text_obj.draw()

    def _draw_hex_coord_tooltip(self, hex_coord: AxialCoord) -> None:
        """A small "(q, r)" label next to the cursor for whichever empty
        board hex it's over -- see _update_hover. Same corner/flip-to-fit
        placement as _draw_hover_tooltip, just one line and no player-color
        accent bar (there's no owner to accent)."""
        text = f"({hex_coord.q}, {hex_coord.r})"
        width = TOOLTIP_PADDING * 2 + len(text) * 7 + 6
        height = TOOLTIP_PADDING * 2 + TOOLTIP_LINE_HEIGHT
        mouse_x, mouse_y = self._mouse_screen_pos
        left = mouse_x + TOOLTIP_OFFSET
        if left + width > self.window.width:
            left = mouse_x - TOOLTIP_OFFSET - width
        top = mouse_y + TOOLTIP_OFFSET + height
        if top > self.window.height:
            top = mouse_y - TOOLTIP_OFFSET

        arcade.draw_lbwh_rectangle_filled(left, top - height, width, height, TOOLTIP_BG_COLOR)
        self._hex_coord_text.text = text
        self._hex_coord_text.x = left + TOOLTIP_PADDING
        self._hex_coord_text.y = top - TOOLTIP_PADDING - TOOLTIP_LINE_HEIGHT + 4
        self._hex_coord_text.draw()

    # -- input -------------------------------------------------------------

    def on_resize(self, width: int, height: int) -> None:
        self.camera.match_window(position=False)
        self.ui_camera.match_window(position=True)
        self._grow_zoom_to_fill_window(width, height)

    def _grow_zoom_to_fill_window(self, width: int, height: int) -> None:
        """If the window has grown past the map's natural size at the
        current zoom, zoom in just enough that the map keeps filling the
        frame -- same behavior as the live game (see
        tla.rendering.game_view.GameView._grow_zoom_to_fill_window).
        Never zooms *out* on a resize: a deliberate manual zoom the viewer
        already made isn't undone by an incidental window resize."""
        min_x, min_y, max_x, max_y = self._board_pixel_bounds
        natural_width = max_x - min_x
        natural_height = max_y - min_y
        if natural_width <= 0 or natural_height <= 0:
            return
        cover_zoom = max(width / natural_width, height / natural_height)
        self.camera.zoom = max(self.camera.zoom, min(cover_zoom, MAX_ZOOM))

    def _set_play_direction(self, direction: int) -> None:
        """0 = stop, 1 = play forward, -1 = play reverse -- see on_update,
        which actually advances the cursor one step every PLAY_STEP_SECONDS
        while a direction is set. Resets the timer so switching direction
        (or resuming after a manual step) doesn't inherit a stale partial
        interval."""
        self._play_direction = direction
        self._play_timer = 0.0

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol == arcade.key.LEFT:
            self._set_play_direction(0)
            self._set_cursor(self.cursor - 1)
            return
        if symbol == arcade.key.RIGHT:
            self._set_play_direction(0)
            self._set_cursor(self.cursor + 1, animate=True)
            return
        if symbol == arcade.key.HOME:
            self._set_play_direction(0)
            self._set_cursor(0)
            return
        if symbol == arcade.key.END:
            self._set_play_direction(0)
            self._set_cursor(len(self.records) - 1)
            return
        if symbol == arcade.key.SPACE:
            # While paused: step forward one half-turn, same as Right --
            # while playing: pause, so Space always does *something* useful
            # rather than being a dead key mid-animation.
            if self._play_direction == 0:
                self._set_cursor(self.cursor + 1, animate=True)
            else:
                self._set_play_direction(0)
            return
        if symbol == arcade.key.L:
            self._set_play_direction(0 if self._play_direction == 1 else 1)
            return
        if symbol == arcade.key.J:
            self._set_play_direction(0 if self._play_direction == -1 else -1)
            return
        if symbol == arcade.key.K:
            self._set_play_direction(0)
            return
        if symbol in _PAN_KEYS:
            self._held_pan_keys.add(symbol)

    def on_key_release(self, symbol: int, modifiers: int) -> None:
        self._held_pan_keys.discard(symbol)

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        if button == arcade.MOUSE_BUTTON_LEFT:
            self._handle_button_click(x, y)
        if button == arcade.MOUSE_BUTTON_RIGHT:
            self._dragging = True

    def _handle_button_click(self, x: float, y: float) -> None:
        action = self._button_at(x, y)
        if action == "begin":
            self._set_play_direction(0)
            self._set_cursor(0)
        elif action == "step_back":
            self._set_play_direction(0)
            self._set_cursor(self.cursor - 1)
        elif action == "play":
            self._set_play_direction(1)
        elif action == "pause":
            if self._play_direction != 0:  # disabled otherwise -- see _draw_transport_controls
                self._set_play_direction(0)
        elif action == "step_forward":
            self._set_play_direction(0)
            self._set_cursor(self.cursor + 1, animate=True)
        elif action == "end":
            self._set_play_direction(0)
            self._set_cursor(len(self.records) - 1)

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> None:
        if button == arcade.MOUSE_BUTTON_RIGHT:
            self._dragging = False

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> None:
        if self._dragging:
            cx, cy = self.camera.position
            zoom = self.camera.zoom
            self.camera.position = (cx - dx / zoom, cy - dy / zoom)
        self._update_hover(x, y)

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
        self._update_hover(x, y)

    def _update_hover(self, screen_x: float, screen_y: float) -> None:
        self._mouse_screen_pos = (screen_x, screen_y)
        world = self.camera.unproject((screen_x, screen_y))
        hex_coord = pixel_to_axial(world[0], world[1], self.hex_size)
        found: int | None = None
        for history in self.ship_histories.values():
            status = self._ship_status_at(history, self.cursor)
            if status is not None and status[0] == hex_coord:
                found = history.id
                break
        self._hovered_ship_id = found
        self._hovered_empty_hex = hex_coord if found is None and hex_coord in self.board.tiles else None

    def on_mouse_scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        factor = 1.1 if scroll_y > 0 else (1 / 1.1 if scroll_y < 0 else 1.0)
        new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.camera.zoom * factor))
        if new_zoom == self.camera.zoom:
            return
        world_before = self.camera.unproject((x, y))
        self.camera.zoom = new_zoom
        world_after = self.camera.unproject((x, y))
        cx, cy = self.camera.position
        self.camera.position = (cx + world_before[0] - world_after[0], cy + world_before[1] - world_after[1])

    def on_update(self, delta_time: float) -> None:
        self._advance_step_animation(delta_time)
        self._advance_playback(delta_time)
        if not self._held_pan_keys:
            return
        move_x = sum(dx for key, (dx, _) in _PAN_KEYS.items() if key in self._held_pan_keys)
        move_y = sum(dy for key, (_, dy) in _PAN_KEYS.items() if key in self._held_pan_keys)
        if move_x == 0 and move_y == 0:
            return
        cx, cy = self.camera.position
        speed = PAN_SPEED / self.camera.zoom
        self.camera.position = (cx + move_x * speed * delta_time, cy + move_y * speed * delta_time)

    def _advance_playback(self, delta_time: float) -> None:
        """Step the cursor one record every PLAY_STEP_SECONDS while auto-
        play is active (see _set_play_direction). Stops itself at either
        end of the replay rather than trying to wrap around."""
        if self._play_direction == 0:
            return
        self._play_timer += delta_time
        if self._play_timer < PLAY_STEP_SECONDS:
            return
        self._play_timer -= PLAY_STEP_SECONDS
        next_cursor = self.cursor + self._play_direction
        if next_cursor < 0 or next_cursor >= len(self.records):
            self._play_direction = 0
            return
        self._set_cursor(next_cursor)
