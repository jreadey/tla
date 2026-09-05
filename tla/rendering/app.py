"""Arcade window entrypoint."""

from __future__ import annotations

import arcade

from tla.board import Board
from tla.rendering.game_view import HEX_SIZE, GameView
from tla.rendering.hex_render import board_pixel_bounds


def run(board: Board) -> None:
    min_x, min_y, max_x, max_y = board_pixel_bounds(board, HEX_SIZE)
    width = int(max_x - min_x)
    height = int(max_y - min_y)
    window = arcade.Window(width, height, "tla - Navy Strategy")
    window.show_view(GameView(board))
    arcade.run()
