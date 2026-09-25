"""Arcade window entrypoint for the graphical replay viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tla.rendering.app import _disable_native_macos_fullscreen
from tla.rendering.hex_render import board_pixel_bounds
from tla.rendering.replay_view import ReplayView, _board_from_initial

import arcade

if TYPE_CHECKING:
    from tla.ai.belief_store import BeliefReader

# Same screen-fit margins tla.rendering.app.run uses, for a consistent
# window-sizing feel between the live game and this viewer.
_SCREEN_FIT_MARGIN_X = 0.9
_SCREEN_FIT_MARGIN_Y = 0.85


def run(records: list[dict], belief_reader: "BeliefReader | None" = None) -> None:
    board = _board_from_initial(records[0])
    natural_min_x, natural_min_y, natural_max_x, natural_max_y = board_pixel_bounds(
        board, board.hex_pixel_size
    )
    natural_width = natural_max_x - natural_min_x
    natural_height = natural_max_y - natural_min_y

    screen_width, screen_height = arcade.get_display_size()
    width = int(min(natural_width, screen_width * _SCREEN_FIT_MARGIN_X))
    height = int(min(natural_height, screen_height * _SCREEN_FIT_MARGIN_Y))

    window = arcade.Window(width, height, "tla - Replay Viewer", resizable=True)
    _disable_native_macos_fullscreen(window)
    window.show_view(ReplayView(records, belief_reader=belief_reader))
    arcade.run()
