"""Phase 1: static board rendering only, no interaction yet."""

from __future__ import annotations

import arcade

from tla.board import Board
from tla.rendering.hex_render import board_pixel_bounds, draw_board

HEX_SIZE = 18.0


class GameView(arcade.View):
    def __init__(self, board: Board) -> None:
        super().__init__()
        self.board = board
        min_x, min_y, max_x, max_y = board_pixel_bounds(board, HEX_SIZE)
        self.origin_x = -min_x
        self.origin_y = -min_y

    def on_show_view(self) -> None:
        self.window.background_color = arcade.color.BLACK

    def on_draw(self) -> None:
        self.clear()
        draw_board(self.board, HEX_SIZE, self.origin_x, self.origin_y)
