"""Arcade window entrypoint."""

from __future__ import annotations

import math
import sys

import arcade

from tla.game_state import GameState
from tla.rendering.game_view import GameView
from tla.rendering.hex_render import board_pixel_bounds

# Leave room for OS chrome (menu bar, dock, window title bar) around the
# window rather than filling the whole physical screen.
_SCREEN_FIT_MARGIN_X = 0.9
_SCREEN_FIT_MARGIN_Y = 0.85

# Bounds on the auto-computed default map size, so a very small display
# still gets a sensible fleet/port layout and a very large one doesn't
# balloon the map past what "large.json" is meant to represent.
_MIN_MAP_WIDTH = 36
_MIN_MAP_HEIGHT = 18
_MAX_MAP_WIDTH = 70
_MAX_MAP_HEIGHT = 35


def compute_default_map_size(hex_pixel_size: float) -> tuple[int, int]:
    """Pick a map width/height (in hex cells) that fills most of the current
    screen at hex_pixel_size, so the whole map fits without panning
    regardless of the user's actual screen resolution.

    Inverts the flat-top axial_to_pixel spacing: consecutive columns are
    hex_pixel_size*1.5 apart, consecutive rows hex_pixel_size*sqrt(3) apart.
    """
    screen_width, screen_height = arcade.get_display_size()
    target_width = screen_width * _SCREEN_FIT_MARGIN_X
    target_height = screen_height * _SCREEN_FIT_MARGIN_Y

    width = int(target_width / (hex_pixel_size * 1.5))
    height = int(target_height / (hex_pixel_size * math.sqrt(3)))

    width = max(_MIN_MAP_WIDTH, min(_MAX_MAP_WIDTH, width))
    height = max(_MIN_MAP_HEIGHT, min(_MAX_MAP_HEIGHT, height))
    return width, height


def _disable_native_macos_fullscreen(window: arcade.Window) -> None:
    """Stop the titlebar's green button from offering native (Spaces)
    fullscreen.

    Pyglet's Cocoa backend has no handling for the windowWillEnterFullScreen
    / windowDidEnterFullScreen notifications that transition triggers, so it
    hangs the app in a state with no menu bar and no way back except a force
    quit. The green button falls back to an ordinary maximize instead. Our
    own fullscreen (F11) goes through arcade's set_fullscreen(), which uses
    pyglet's own borderless-window implementation and isn't affected.
    """
    if sys.platform != "darwin":
        return
    nswindow = getattr(window, "_nswindow", None)
    if nswindow is None:
        return
    ns_window_collection_behavior_managed = 1 << 2
    ns_window_collection_behavior_fullscreen_none = 1 << 9
    nswindow.setCollectionBehavior_(
        ns_window_collection_behavior_managed | ns_window_collection_behavior_fullscreen_none
    )


def run(game_state: GameState) -> None:
    board = game_state.board
    natural_min_x, natural_min_y, natural_max_x, natural_max_y = board_pixel_bounds(
        board, board.hex_pixel_size
    )
    natural_width = natural_max_x - natural_min_x
    natural_height = natural_max_y - natural_min_y

    screen_width, screen_height = arcade.get_display_size()

    # The window (viewport) is capped to comfortably fit the screen, but the
    # map itself is drawn at its configured native hex size regardless --
    # panning (arrow keys/WASD, or right-click drag) reveals the rest rather
    # than shrinking hexes to force the whole map to fit at once.
    width = int(min(natural_width, screen_width * _SCREEN_FIT_MARGIN_X))
    height = int(min(natural_height, screen_height * _SCREEN_FIT_MARGIN_Y))

    # Deliberately not passing center_window=True: arcade's own
    # center_window() mixes logical-point and physical-pixel sizes, which on
    # a Retina display shoves the window (and its titlebar traffic-light
    # buttons) off-screen. Pyglet's own default placement centers correctly.
    window = arcade.Window(width, height, "tla - Navy Strategy", resizable=True)
    _disable_native_macos_fullscreen(window)
    window.show_view(GameView(game_state))
    arcade.run()
