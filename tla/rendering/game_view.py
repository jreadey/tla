"""Phase 2: board + starting fleets rendered. Camera pans; hex size is fixed."""

from __future__ import annotations

from dataclasses import dataclass, field

import arcade

from tla.battle import RoundResult, resolve_round
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
from tla.ship import Ship, ShipKind
from tla.tile import PLAYER_A
from tla.turn_manager import TurnManager


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

        self._held_pan_keys: set[int] = set()
        self._dragging = False
        self._mouse_screen_pos = (0.0, 0.0)
        self._hovered_ship: Ship | None = None

        # arcade.Text objects are reused and repositioned every frame rather
        # than calling arcade.draw_text() fresh each time, which rebuilds a
        # full text layout from scratch and is too slow to do every frame.
        self._hud_text = arcade.Text("", 10, 0, arcade.color.WHITE, 13)
        self._tooltip_texts = [
            arcade.Text("", 0, 0, TOOLTIP_TEXT_COLOR, 12) for _ in range(TOOLTIP_MAX_LINES)
        ]
        self._battle_texts = [arcade.Text("", 0, 0, arcade.color.WHITE, 14) for _ in range(3)]
        self._sunk_text = arcade.Text("", 0, 0, SUNK_TEXT_COLOR, 16, anchor_x="center")

    def on_show_view(self) -> None:
        self.window.background_color = arcade.color.BLACK

    def on_resize(self, width: int, height: int) -> None:
        # Keep the camera's current position -- only the viewport/projection
        # need to match the new window size, so panning isn't reset.
        self.camera.match_window(position=False)
        self.ui_camera.match_window(position=True)

    def on_key_press(self, symbol: int, modifiers: int) -> None:
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
        elif symbol in (arcade.key.PLUS, arcade.key.EQUAL, arcade.key.NUM_ADD):
            self._zoom_toward_screen_point(self.window.width / 2, self.window.height / 2, ZOOM_STEP)
        elif symbol in (arcade.key.MINUS, arcade.key.NUM_SUBTRACT):
            self._zoom_toward_screen_point(self.window.width / 2, self.window.height / 2, 1 / ZOOM_STEP)
        elif symbol == arcade.key.KEY_0:
            self.camera.zoom = 1.0
        elif symbol == arcade.key.ENTER:
            self._abort_drag()
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
        if self.sunk_message is not None:
            self.sunk_message = None
            return
        if button == arcade.MOUSE_BUTTON_RIGHT:
            self._dragging = True
        elif button == arcade.MOUSE_BUTTON_LEFT and self.active_battle is None:
            self._start_drag(x, y)

    def _start_drag(self, screen_x: float, screen_y: float) -> None:
        world = self.camera.unproject((screen_x, screen_y))
        hex_coord = pixel_to_axial(world[0], world[1], self.hex_size)
        gs = self.game_state

        ship = gs.ship_at(hex_coord)
        if ship is not None and ship.owner == gs.current_player and ship.movement_remaining > 0:
            self.drag_ship = ship
            self.drag_path = [hex_coord]
            self.range_preview = reachable_hexes(ship, gs)

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
            validate_path(self.drag_ship, trial_path, self.game_state)
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
                move_ship_along_path(ship, path, self.game_state)
            except ValueError:
                pass

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
        if defender.is_sunk:
            del self.game_state.ships[defender.id]
            if self._hovered_ship is defender:
                self._hovered_ship = None
            if not attacker.is_sunk:
                attacker.position = defender.position
        if attacker.is_sunk:
            del self.game_state.ships[attacker.id]
            if self._hovered_ship is attacker:
                self._hovered_ship = None
        # A retreat (both survive) needs no position change -- the attacker
        # is already sitting at the approach hex from begin_engagement.
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
        self._draw_hud()
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
        player_label = "Player A" if self.game_state.current_player == PLAYER_A else "Player B"
        self._hud_text.text = (
            f"Turn {self.game_state.turn_number} -- {player_label}'s move    "
            "[Drag a ship] Move    [Esc] Cancel move    "
            "[Enter] End Movement    [T] Toggle hovered submarine"
        )
        self._hud_text.y = self.window.height - 22
        self._hud_text.draw()

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
