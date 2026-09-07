"""Phase 2: board + starting fleets rendered. Camera pans; hex size is fixed."""

from __future__ import annotations

from dataclasses import dataclass, field

import arcade

from tla.battle import RoundResult, apply_battle_outcome, resolve_round
from tla.elevation import marching_squares_segments
from tla.game_state import GameState
from tla.hexgrid import AxialCoord, axial_to_pixel, pixel_to_axial
from tla.movement import (
    begin_engagement,
    move_ship_along_path,
    reachable_hexes,
    toggle_submarine_state,
    validate_path,
)
from tla.production import order
from tla.rendering.hex_render import (
    PATH_HIGHLIGHT_COLOR,
    PATH_LINE_COLOR,
    PLAYER_COLORS,
    RANGE_PREVIEW_COLOR,
    board_pixel_bounds,
    draw_board,
    draw_contour,
    draw_hex_highlight,
    draw_ships,
)
from tla.rendering.ship_glyphs import draw_ship_glyph
from tla.ship import Ship, ShipKind
from tla.tile import PLAYER_A
from tla.turn_manager import TurnManager

# The six ship kinds shown, in this fixed order, as clickable glyph buttons
# in a port's production panel.
_PORT_PANEL_KINDS = [
    ShipKind.BATTLESHIP,
    ShipKind.CARRIER,
    ShipKind.CRUISER,
    ShipKind.DESTROYER,
    ShipKind.SUBMARINE,
    ShipKind.PATROL_BOAT,
]
PORT_PANEL_BG_COLOR = (25, 25, 25, 235)
PORT_PANEL_ICON_BOX = 84.0
PORT_PANEL_ICON_HEX_SIZE = 32.0
PORT_PANEL_MARGIN = 14.0
PORT_PANEL_ICON_ROW_HEIGHT = 92.0
PORT_PANEL_HEADER_HEIGHT = 44.0
# Wide enough for the longest title ("Choose a ship to order" at this font
# size measures ~182px) plus margins, so the title never overflows the
# panel when there's only 1-2 icon slots (an empty or near-empty queue).
PORT_PANEL_MIN_WIDTH = 230.0


@dataclass
class ActiveBattle:
    """A battle awaiting the human attacker's stay/retreat decision after
    the round that just happened -- driven by key presses (Enter/Escape)
    rather than a synchronous decision_fn, since the player needs to see
    each round's outcome before choosing."""

    attacker: Ship
    defender: Ship
    rounds: list[RoundResult] = field(default_factory=list)

# Screen pixels/second for keyboard panning (divided by zoom so it always
# feels like the same on-screen speed, not the same world-space speed).
PAN_SPEED = 600.0

_PAN_KEYS = {
    arcade.key.LEFT: (-1, 0),
    arcade.key.A: (-1, 0),
    arcade.key.RIGHT: (1, 0),
    arcade.key.D: (1, 0),
    arcade.key.UP: (0, 1),
    arcade.key.W: (0, 1),
    arcade.key.DOWN: (0, -1),
    arcade.key.S: (0, -1),
}

# zoom > 1 magnifies (zoomed in); zoom < 1 shrinks (zoomed out).
ZOOM_STEP = 1.1
MIN_ZOOM = 0.4
MAX_ZOOM = 3.0

TOOLTIP_BG_COLOR = (25, 25, 25, 230)
TOOLTIP_TEXT_COLOR = arcade.color.WHITE
TOOLTIP_PADDING = 10
TOOLTIP_LINE_HEIGHT = 18
TOOLTIP_WIDTH = 170
TOOLTIP_OFFSET = 16
# Title, Movement, HP, Damage, and one of (ASW | Surfaced/Submerged).
TOOLTIP_MAX_LINES = 5

SUNK_BG_COLOR = (40, 10, 10, 235)
SUNK_BORDER_COLOR = (220, 60, 60)
SUNK_TEXT_COLOR = arcade.color.WHITE

GAME_OVER_BG_COLOR = (10, 10, 10, 245)
GAME_OVER_TEXT_COLOR = arcade.color.WHITE


class GameView(arcade.View):
    def __init__(self, game_state: GameState, hex_size: float | None = None) -> None:
        super().__init__()
        self.game_state = game_state
        board = game_state.board
        self.hex_size = hex_size if hex_size is not None else board.hex_pixel_size

        self.contour_segments = (
            marching_squares_segments(board.elevation) if board.elevation else []
        )

        min_x, min_y, max_x, max_y = board_pixel_bounds(board, self.hex_size)
        self.camera = arcade.Camera2D(position=((min_x + max_x) / 2, (min_y + max_y) / 2))
        # Screen-space camera for the hover tooltip -- fixed 1:1 with window
        # pixels regardless of the world camera's pan/zoom.
        self.ui_camera = arcade.Camera2D()

        self.turn_manager = TurnManager(game_state)
        # An in-progress drag: the ship being moved, the exact route drawn
        # so far (starting with its current hex), and a background "how far
        # could I go" hint computed once at drag-start.
        self.drag_ship: Ship | None = None
        self.drag_path: list[AxialCoord] = []
        self.range_preview: dict[AxialCoord, int] = {}
        self.active_battle: ActiveBattle | None = None
        # Set when a battle just concluded with a sink, dismissed by any
        # key press or click.
        self.sunk_message: str | None = None
        # The empty friendly port currently showing its production panel,
        # if any -- opened by clicking it, closed by clicking elsewhere,
        # Escape, or ending movement.
        self.selected_port: AxialCoord | None = None
        # Within an open port panel: False shows the queue + an "Order"
        # button, True shows the six ship-kind glyphs to pick from.
        self._port_panel_picking = False

        self._held_pan_keys: set[int] = set()
        self._dragging = False
        self._mouse_screen_pos = (0.0, 0.0)
        self._hovered_ship: Ship | None = None

        # arcade.Text objects are reused and repositioned every frame rather
        # than calling arcade.draw_text() fresh each time, which rebuilds a
        # full text layout from scratch and is too slow to do every frame.
        self._hud_texts = [arcade.Text("", 10, 0, arcade.color.WHITE, 13) for _ in range(2)]
        self._tooltip_texts = [
            arcade.Text("", 0, 0, TOOLTIP_TEXT_COLOR, 12) for _ in range(TOOLTIP_MAX_LINES)
        ]
        self._battle_texts = [arcade.Text("", 0, 0, arcade.color.WHITE, 14) for _ in range(3)]
        self._sunk_text = arcade.Text("", 0, 0, SUNK_TEXT_COLOR, 16, anchor_x="center")
        self._game_over_text = arcade.Text(
            "", 0, 0, GAME_OVER_TEXT_COLOR, 36, anchor_x="center", bold=True
        )
        self._port_panel_title_text = arcade.Text("", 0, 0, arcade.color.WHITE, 13)
        self._port_panel_cost_texts = [
            arcade.Text("", 0, 0, arcade.color.WHITE, 11, anchor_x="center") for _ in range(6)
        ]
        self._port_panel_order_text = arcade.Text(
            "+", 0, 0, arcade.color.WHITE, 20, anchor_x="center", anchor_y="center"
        )
        self._port_panel_hover_text = arcade.Text(
            "", 0, 0, arcade.color.WHITE, 12, anchor_x="center"
        )

    def on_show_view(self) -> None:
        self.window.background_color = arcade.color.BLACK

    def on_resize(self, width: int, height: int) -> None:
        # Keep the camera's current position -- only the viewport/projection
        # need to match the new window size, so panning isn't reset.
        self.camera.match_window(position=False)
        self.ui_camera.match_window(position=True)

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if self.game_state.winner is not None:
            return
        if self.sunk_message is not None:
            self.sunk_message = None
            return
        if self.active_battle is not None:
            if symbol == arcade.key.ENTER:
                self._resolve_battle_round()
            elif symbol == arcade.key.ESCAPE:
                self._conclude_battle(retreated=True)
            return

        if symbol in _PAN_KEYS:
            self._held_pan_keys.add(symbol)
        elif symbol == arcade.key.F11:
            self.window.set_fullscreen(not self.window.fullscreen)
        elif symbol == arcade.key.ESCAPE:
            if self.window.fullscreen:
                self.window.set_fullscreen(False)
            elif self.drag_ship is not None:
                self._abort_drag()
            elif self._port_panel_picking:
                self._port_panel_picking = False
            elif self.selected_port is not None:
                self._close_port_panel()
        elif symbol in (arcade.key.PLUS, arcade.key.EQUAL, arcade.key.NUM_ADD):
            self._zoom_toward_screen_point(self.window.width / 2, self.window.height / 2, ZOOM_STEP)
        elif symbol in (arcade.key.MINUS, arcade.key.NUM_SUBTRACT):
            self._zoom_toward_screen_point(self.window.width / 2, self.window.height / 2, 1 / ZOOM_STEP)
        elif symbol == arcade.key.KEY_0:
            self.camera.zoom = 1.0
        elif symbol == arcade.key.ENTER:
            self._abort_drag()
            self._close_port_panel()
            self.turn_manager.end_movement_phase()
        elif symbol == arcade.key.T:
            self._toggle_hovered_submarine()

    def on_key_release(self, symbol: int, modifiers: int) -> None:
        self._held_pan_keys.discard(symbol)

    def _toggle_hovered_submarine(self) -> None:
        """T toggles surfaced/submerged for whichever of your own submarines
        is under the cursor -- independent of the drag-to-move gesture,
        since there's no natural pause in a press-drag-release move to
        squeeze a key press into otherwise."""
        ship = self._hovered_ship
        if ship is None or ship.kind != ShipKind.SUBMARINE or ship.owner != self.game_state.current_player:
            return
        stats = self.game_state.config.ship_stats.stats[ship.kind]
        try:
            toggle_submarine_state(ship, stats)
        except ValueError:
            return
        if self.drag_ship is ship:
            self._abort_drag()

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        if self.game_state.winner is not None:
            return
        if self.sunk_message is not None:
            self.sunk_message = None
            return
        if button == arcade.MOUSE_BUTTON_RIGHT:
            self._dragging = True
            return
        if button != arcade.MOUSE_BUTTON_LEFT or self.active_battle is not None:
            return
        if self.selected_port is not None:
            if self._port_panel_picking:
                clicked_kind = self._picker_glyph_at(x, y)
                if clicked_kind is not None:
                    order(self.game_state, self.game_state.current_player, self.selected_port, clicked_kind)
                    self._port_panel_picking = False
                    # Return immediately rather than falling into the bounds
                    # check below: adding to the queue changes its length,
                    # which resizes/re-centers the panel, so re-checking
                    # against the now-different geometry could wrongly
                    # decide this same click landed outside it.
                    return
            elif self._order_button_at(x, y):
                self._port_panel_picking = True
                return
            # Any click within the panel's (still-current, since neither
            # branch above matched) bounds is consumed here even though it
            # missed every button -- it must never fall through to
            # _start_drag, which would reinterpret it as a click on
            # whatever map hex happens to be underneath the panel.
            if self._point_in_port_panel(x, y):
                return
        self._start_drag(x, y)

    def _open_port_panel(self, coord: AxialCoord) -> None:
        self.selected_port = coord
        self._port_panel_picking = False

    def _close_port_panel(self) -> None:
        self.selected_port = None
        self._port_panel_picking = False

    def _start_drag(self, screen_x: float, screen_y: float) -> None:
        gs = self.game_state
        world = self.camera.unproject((screen_x, screen_y))
        hex_coord = pixel_to_axial(world[0], world[1], self.hex_size)

        ship = gs.ship_at(hex_coord)
        if ship is not None and ship.owner == gs.current_player and ship.movement_remaining > 0:
            self._close_port_panel()
            self.drag_ship = ship
            self.drag_path = [hex_coord]
            self.range_preview = reachable_hexes(ship, gs)
            return

        tile = gs.board.get_tile(hex_coord)
        if ship is None and tile is not None and tile.is_port and tile.port_display_owner == gs.current_player:
            if self.selected_port == hex_coord:
                self._close_port_panel()
            else:
                self._open_port_panel(hex_coord)
            return

        self._close_port_panel()

    def _extend_drag(self, screen_x: float, screen_y: float) -> None:
        world = self.camera.unproject((screen_x, screen_y))
        hex_coord = pixel_to_axial(world[0], world[1], self.hex_size)

        if hex_coord == self.drag_path[-1]:
            return
        if hex_coord in self.drag_path:
            # Dragging back over an already-drawn hex undoes the path back
            # to that point, rather than requiring a precise one-step undo.
            index = self.drag_path.index(hex_coord)
            self.drag_path = self.drag_path[: index + 1]
            return
        trial_path = self.drag_path + [hex_coord]
        try:
            # The newest hex is only provisionally the end of the drag --
            # the player may keep going -- so a friendly-occupied hex is
            # tolerated here even though it could never actually be the
            # final stop; _commit_drag's move call enforces that for real.
            validate_path(self.drag_ship, trial_path, self.game_state, allow_passthrough_final=True)
        except ValueError:
            return
        self.drag_path = trial_path

    def _commit_drag(self) -> None:
        ship = self.drag_ship
        path = self.drag_path
        self._abort_drag()
        if ship is None or len(path) < 2:
            return

        if self.game_state.ship_at(path[-1]) is not None:
            try:
                defender = begin_engagement(ship, path, self.game_state)
            except ValueError:
                return
            self._start_battle(ship, defender)
        else:
            try:
                # Captures a port and ends the game on the spot if that
                # completes total port control -- see move_ship_along_path.
                move_ship_along_path(ship, path, self.game_state)
            except ValueError:
                return

    def _abort_drag(self) -> None:
        self.drag_ship = None
        self.drag_path = []
        self.range_preview = {}

    def _start_battle(self, attacker: Ship, defender: Ship) -> None:
        self.active_battle = ActiveBattle(attacker=attacker, defender=defender)
        self._resolve_battle_round()

    def _resolve_battle_round(self) -> None:
        battle = self.active_battle
        round_result = resolve_round(battle.attacker, battle.defender, self.game_state)
        battle.rounds.append(round_result)
        if round_result.attacker_sunk or round_result.defender_sunk:
            self._conclude_battle(retreated=False)

    def _conclude_battle(self, retreated: bool) -> None:
        battle = self.active_battle
        attacker, defender = battle.attacker, battle.defender
        self.sunk_message = self._sunk_message(attacker, defender)
        if defender.is_sunk and self._hovered_ship is defender:
            self._hovered_ship = None
        if attacker.is_sunk and self._hovered_ship is attacker:
            self._hovered_ship = None
        # Removes sunk ship(s), repositions a surviving attacker (which may
        # capture a port), and refreshes game_state.winner -- see
        # tla.battle.apply_battle_outcome. A pure retreat is a no-op here.
        apply_battle_outcome(self.game_state, attacker, defender)
        self.active_battle = None

    def _sunk_message(self, attacker: Ship, defender: Ship) -> str | None:
        def label(ship: Ship) -> str:
            owner_label = "Player A" if ship.owner == PLAYER_A else "Player B"
            kind_label = ship.kind.value.replace("_", " ").title()
            return f"{owner_label} {kind_label}"

        if attacker.is_sunk and defender.is_sunk:
            return f"{label(attacker)} and {label(defender)} both sunk!"
        if attacker.is_sunk:
            return f"{label(attacker)} sunk!"
        if defender.is_sunk:
            return f"{label(defender)} sunk!"
        return None

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> None:
        if button == arcade.MOUSE_BUTTON_LEFT:
            self._commit_drag()
        if button == arcade.MOUSE_BUTTON_RIGHT:
            self._dragging = False

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> None:
        if self._dragging:
            cx, cy = self.camera.position
            zoom = self.camera.zoom
            self.camera.position = (cx - dx / zoom, cy - dy / zoom)
        if self.drag_ship is not None:
            self._extend_drag(x, y)
        self._update_hover(x, y)

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
        self._update_hover(x, y)

    def _update_hover(self, screen_x: float, screen_y: float) -> None:
        self._mouse_screen_pos = (screen_x, screen_y)
        world = self.camera.unproject((screen_x, screen_y))
        hex_coord = pixel_to_axial(world[0], world[1], self.hex_size)
        self._hovered_ship = self.game_state.ship_at(hex_coord)

    def on_mouse_scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        if scroll_y > 0:
            self._zoom_toward_screen_point(x, y, ZOOM_STEP)
        elif scroll_y < 0:
            self._zoom_toward_screen_point(x, y, 1 / ZOOM_STEP)

    def _zoom_toward_screen_point(self, screen_x: float, screen_y: float, factor: float) -> None:
        """Change zoom by `factor`, keeping the world point under
        (screen_x, screen_y) fixed on screen -- so scrolling over a spot on
        the map zooms toward that spot rather than the window's center."""
        new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.camera.zoom * factor))
        if new_zoom == self.camera.zoom:
            return
        world_before = self.camera.unproject((screen_x, screen_y))
        self.camera.zoom = new_zoom
        world_after = self.camera.unproject((screen_x, screen_y))
        cx, cy = self.camera.position
        self.camera.position = (
            cx + (world_before[0] - world_after[0]),
            cy + (world_before[1] - world_after[1]),
        )

    def on_update(self, delta_time: float) -> None:
        if not self._held_pan_keys:
            return
        move_x = sum(dx for key, (dx, _) in _PAN_KEYS.items() if key in self._held_pan_keys)
        move_y = sum(dy for key, (_, dy) in _PAN_KEYS.items() if key in self._held_pan_keys)
        if move_x == 0 and move_y == 0:
            return
        cx, cy = self.camera.position
        speed = PAN_SPEED / self.camera.zoom
        self.camera.position = (
            cx + move_x * speed * delta_time,
            cy + move_y * speed * delta_time,
        )

    def on_draw(self) -> None:
        self.clear()
        self.camera.use()
        draw_board(self.game_state.board, self.hex_size)
        for coord in self.range_preview:
            draw_hex_highlight(coord, self.hex_size, RANGE_PREVIEW_COLOR)
        for coord in self.drag_path:
            draw_hex_highlight(coord, self.hex_size, PATH_HIGHLIGHT_COLOR)
        self._draw_drag_path_line()
        draw_contour(self.contour_segments)
        draw_ships(
            self.game_state.ships.values(), self.hex_size, current_player=self.game_state.current_player
        )

        self.ui_camera.use()
        if self.game_state.winner is not None:
            self._draw_game_over_overlay()
            return
        self._draw_hud()
        if self.selected_port is not None:
            self._draw_port_panel()
        if self.sunk_message is not None:
            self._draw_sunk_overlay()
        elif self.active_battle is not None:
            self._draw_battle_banner(self.active_battle)
        elif self._hovered_ship is not None:
            self._draw_hover_tooltip(self._hovered_ship)

    def _draw_drag_path_line(self) -> None:
        if len(self.drag_path) < 2:
            return
        points = [axial_to_pixel(coord, self.hex_size) for coord in self.drag_path]
        arcade.draw_line_strip(points, PATH_LINE_COLOR, 3)

    def _draw_hud(self) -> None:
        gs = self.game_state
        player_label = "Player A" if gs.current_player == PLAYER_A else "Player B"
        lines = [
            f"Turn {gs.turn_number} -- {player_label}'s move    "
            "[Drag a ship] Move    [Esc] Cancel move    "
            "[Enter] End Movement    [T] Toggle hovered submarine",
            "[Click an empty friendly port] Manage its production queue",
        ]
        for i, line in enumerate(lines):
            text_obj = self._hud_texts[i]
            text_obj.text = line
            text_obj.y = self.window.height - 22 - i * 20
            text_obj.draw()

    def _selected_port_queue(self) -> list[ShipKind]:
        if self.selected_port is None:
            return []
        gs = self.game_state
        progress = gs.players[gs.current_player].port_production.get(self.selected_port)
        return progress.orders if progress else []

    def _port_panel_slot_count(self) -> int:
        return len(_PORT_PANEL_KINDS) if self._port_panel_picking else len(self._selected_port_queue()) + 1

    def _port_panel_geometry(self) -> tuple[float, float, float, float]:
        """(left, bottom, width, height) of the port panel in screen space.
        Width is at least PORT_PANEL_MIN_WIDTH so the title text (measured
        for the longer of the two possible titles) never overflows the
        panel when there are only 1-2 icon slots -- see
        `_port_panel_slot_centers` for how the icon row stays centered
        within whatever width this ends up being."""
        content_width = PORT_PANEL_MARGIN * 2 + PORT_PANEL_ICON_BOX * self._port_panel_slot_count()
        width = max(content_width, PORT_PANEL_MIN_WIDTH)
        height = PORT_PANEL_MARGIN + PORT_PANEL_HEADER_HEIGHT + PORT_PANEL_ICON_ROW_HEIGHT
        left = (self.window.width - width) / 2
        return left, PORT_PANEL_MARGIN, width, height

    def _port_panel_slot_centers(self) -> list[tuple[float, float]]:
        left, bottom, width, _height = self._port_panel_geometry()
        slot_count = self._port_panel_slot_count()
        icons_left = left + (width - PORT_PANEL_ICON_BOX * slot_count) / 2
        icon_cy = bottom + PORT_PANEL_ICON_ROW_HEIGHT / 2
        return [
            (icons_left + PORT_PANEL_ICON_BOX * (i + 0.5), icon_cy)
            for i in range(slot_count)
        ]

    def _point_in_port_panel(self, screen_x: float, screen_y: float) -> bool:
        if self.selected_port is None:
            return False
        left, bottom, width, height = self._port_panel_geometry()
        return left <= screen_x <= left + width and bottom <= screen_y <= bottom + height

    def _picker_glyph_at(self, screen_x: float, screen_y: float) -> ShipKind | None:
        half = PORT_PANEL_ICON_BOX / 2
        for kind, (cx, cy) in zip(_PORT_PANEL_KINDS, self._port_panel_slot_centers()):
            if abs(screen_x - cx) <= half and abs(screen_y - cy) <= half:
                return kind
        return None

    def _order_button_at(self, screen_x: float, screen_y: float) -> bool:
        half = PORT_PANEL_ICON_BOX / 2
        cx, cy = self._port_panel_slot_centers()[-1]  # the button is always the last slot
        return abs(screen_x - cx) <= half and abs(screen_y - cy) <= half

    def _draw_port_panel(self) -> None:
        gs = self.game_state
        player = gs.current_player
        progress = gs.players[player].port_production.get(self.selected_port)
        queue = progress.orders if progress else []
        banked_points = progress.points if progress else 0
        stats = gs.config.ship_stats.stats

        left, bottom, width, height = self._port_panel_geometry()
        top = bottom + height
        arcade.draw_lbwh_rectangle_filled(left, bottom, width, height, PORT_PANEL_BG_COLOR)
        arcade.draw_lbwh_rectangle_filled(left, top - 4, width, 4, PLAYER_COLORS[player])

        self._port_panel_title_text.text = (
            "Choose a ship to order" if self._port_panel_picking else "Port Production"
        )
        self._port_panel_title_text.x = left + PORT_PANEL_MARGIN
        self._port_panel_title_text.y = top - 20
        self._port_panel_title_text.draw()

        half = PORT_PANEL_ICON_BOX / 2
        slot_centers = self._port_panel_slot_centers()

        mouse_x, mouse_y = self._mouse_screen_pos

        if self._port_panel_picking:
            hovered_kind: ShipKind | None = None
            hovered_cx = 0.0
            for i, (kind, (cx, cy)) in enumerate(zip(_PORT_PANEL_KINDS, slot_centers)):
                arcade.draw_lbwh_rectangle_outline(cx - half, cy - half, PORT_PANEL_ICON_BOX, PORT_PANEL_ICON_BOX, (100, 100, 100), 1)
                draw_ship_glyph((cx, cy + 9), PORT_PANEL_ICON_HEX_SIZE, kind, PLAYER_COLORS[player])
                cost_text = self._port_panel_cost_texts[i]
                cost_text.text = str(stats[kind].cost)
                cost_text.x = cx
                cost_text.y = cy - half + 5
                cost_text.draw()
                if abs(mouse_x - cx) <= half and abs(mouse_y - cy) <= half:
                    hovered_kind, hovered_cx = kind, cx

            if hovered_kind is not None:
                self._port_panel_hover_text.text = hovered_kind.value.replace("_", " ").title()
                self._port_panel_hover_text.x = hovered_cx
                self._port_panel_hover_text.y = top + 6
                self._port_panel_hover_text.draw()
            return

        hovered_index: int | None = None
        for i, (cx, cy) in enumerate(slot_centers):
            arcade.draw_lbwh_rectangle_outline(cx - half, cy - half, PORT_PANEL_ICON_BOX, PORT_PANEL_ICON_BOX, (100, 100, 100), 1)
            if i == len(queue):  # the trailing "Order" slot
                self._port_panel_order_text.x = cx
                self._port_panel_order_text.y = cy
                self._port_panel_order_text.draw()
                continue
            draw_ship_glyph((cx, cy + 9), PORT_PANEL_ICON_HEX_SIZE, queue[i], PLAYER_COLORS[player])
            if abs(mouse_x - cx) <= half and abs(mouse_y - cy) <= half:
                hovered_index = i

        if hovered_index is not None:
            kind = queue[hovered_index]
            cx, _cy = slot_centers[hovered_index]
            points = banked_points if hovered_index == 0 else 0
            self._port_panel_hover_text.text = (
                f"{kind.value.replace('_', ' ').title()}  {points}/{stats[kind].cost}"
            )
            self._port_panel_hover_text.x = cx
            self._port_panel_hover_text.y = top + 6
            self._port_panel_hover_text.draw()

    def _draw_game_over_overlay(self) -> None:
        winner_label = "Player A" if self.game_state.winner == PLAYER_A else "Player B"
        arcade.draw_lbwh_rectangle_filled(
            0, 0, self.window.width, self.window.height, GAME_OVER_BG_COLOR
        )
        self._game_over_text.text = f"{winner_label} wins!"
        self._game_over_text.x = self.window.width / 2
        self._game_over_text.y = self.window.height / 2
        self._game_over_text.draw()

    def _draw_battle_banner(self, battle: ActiveBattle) -> None:
        attacker, defender = battle.attacker, battle.defender
        last_round = battle.rounds[-1]
        stats = self.game_state.config.ship_stats.stats
        a_label = "Player A" if attacker.owner == PLAYER_A else "Player B"
        d_label = "Player A" if defender.owner == PLAYER_A else "Player B"

        def kind_name(ship: Ship) -> str:
            return ship.kind.value.replace("_", " ").title()

        lines = [
            f"BATTLE -- {a_label} {kind_name(attacker)} ({attacker.current_hp}/{stats[attacker.kind].hp} HP)"
            f"  vs  {d_label} {kind_name(defender)} ({defender.current_hp}/{stats[defender.kind].hp} HP)",
            f"Round {len(battle.rounds)}: dealt {last_round.damage_to_defender}, "
            f"took {last_round.damage_to_attacker} damage",
            "[Enter] Stay and Fight        [Esc] Retreat",
        ]

        width = 560
        line_height = 24
        height = 16 + line_height * len(lines)
        left = (self.window.width - width) / 2
        top = self.window.height - 40

        arcade.draw_lbwh_rectangle_filled(left, top - height, width, height, (20, 20, 20, 235))
        arcade.draw_lbwh_rectangle_filled(left, top - 4, width, 4, (200, 60, 60))

        for i, line in enumerate(lines):
            text_obj = self._battle_texts[i]
            text_obj.text = line
            text_obj.x = left + 12
            text_obj.y = top - 12 - (i + 1) * line_height + 6
            text_obj.draw()

    def _draw_sunk_overlay(self) -> None:
        display_text = f"{self.sunk_message}   (press any key to continue)"
        width = min(self.window.width - 40, max(360, len(display_text) * 9 + 40))
        height = 60
        left = (self.window.width - width) / 2
        top = self.window.height - 40

        arcade.draw_lbwh_rectangle_filled(left, top - height, width, height, SUNK_BG_COLOR)
        arcade.draw_lbwh_rectangle_filled(left, top - 4, width, 4, SUNK_BORDER_COLOR)

        self._sunk_text.text = display_text
        self._sunk_text.x = self.window.width / 2
        self._sunk_text.y = top - height / 2 - 6
        self._sunk_text.draw()

    def _draw_hover_tooltip(self, ship: Ship) -> None:
        stats = self.game_state.config.ship_stats.stats[ship.kind]
        max_movement = ship.max_movement(stats)

        lines = [
            ship.kind.value.replace("_", " ").title(),
            f"Movement: {ship.movement_remaining}/{max_movement}",
            f"HP: {ship.current_hp}/{stats.hp}",
            f"Damage: {stats.damage}",
        ]
        if ship.kind == ShipKind.SUBMARINE:
            lines.append("Surfaced" if ship.surfaced else "Submerged")
        else:
            lines.append(f"ASW: {stats.asw}")

        height = TOOLTIP_PADDING * 2 + TOOLTIP_LINE_HEIGHT * len(lines)
        mouse_x, mouse_y = self._mouse_screen_pos

        left = mouse_x + TOOLTIP_OFFSET
        if left + TOOLTIP_WIDTH > self.window.width:
            left = mouse_x - TOOLTIP_OFFSET - TOOLTIP_WIDTH
        top = mouse_y + TOOLTIP_OFFSET + height
        if top > self.window.height:
            top = mouse_y - TOOLTIP_OFFSET

        arcade.draw_lbwh_rectangle_filled(left, top - height, TOOLTIP_WIDTH, height, TOOLTIP_BG_COLOR)
        arcade.draw_lbwh_rectangle_filled(left, top - 4, TOOLTIP_WIDTH, 4, PLAYER_COLORS[ship.owner])

        for i, line in enumerate(lines):
            text_obj = self._tooltip_texts[i]
            text_obj.text = line
            text_obj.x = left + TOOLTIP_PADDING
            text_obj.y = top - TOOLTIP_PADDING - (i + 1) * TOOLTIP_LINE_HEIGHT + 4
            text_obj.draw()
