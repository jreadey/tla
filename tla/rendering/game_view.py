"""Phase 2: board + starting fleets rendered. Camera pans; hex size is fixed."""

from __future__ import annotations

import arcade

from tla.elevation import marching_squares_segments
from tla.game_state import GameState
from tla.rendering.hex_render import board_pixel_bounds, draw_board, draw_contour, draw_ships

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

        self._held_pan_keys: set[int] = set()
        self._dragging = False

    def on_show_view(self) -> None:
        self.window.background_color = arcade.color.BLACK

    def on_resize(self, width: int, height: int) -> None:
        # Keep the camera's current position -- only the viewport/projection
        # need to match the new window size, so panning isn't reset.
        self.camera.match_window(position=False)

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol in _PAN_KEYS:
            self._held_pan_keys.add(symbol)
        elif symbol == arcade.key.F11:
            self.window.set_fullscreen(not self.window.fullscreen)
        elif symbol == arcade.key.ESCAPE and self.window.fullscreen:
            self.window.set_fullscreen(False)
        elif symbol in (arcade.key.PLUS, arcade.key.EQUAL, arcade.key.NUM_ADD):
            self._zoom_toward_screen_point(self.window.width / 2, self.window.height / 2, ZOOM_STEP)
        elif symbol in (arcade.key.MINUS, arcade.key.NUM_SUBTRACT):
            self._zoom_toward_screen_point(self.window.width / 2, self.window.height / 2, 1 / ZOOM_STEP)
        elif symbol == arcade.key.KEY_0:
            self.camera.zoom = 1.0

    def on_key_release(self, symbol: int, modifiers: int) -> None:
        self._held_pan_keys.discard(symbol)

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        if button == arcade.MOUSE_BUTTON_RIGHT:
            self._dragging = True

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> None:
        if button == arcade.MOUSE_BUTTON_RIGHT:
            self._dragging = False

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> None:
        if self._dragging:
            cx, cy = self.camera.position
            zoom = self.camera.zoom
            self.camera.position = (cx - dx / zoom, cy - dy / zoom)

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
        draw_contour(self.contour_segments)
        draw_ships(self.game_state.ships.values(), self.hex_size)
