"""Phase 1: static board rendering only, no interaction yet."""

from __future__ import annotations

import arcade

from tla.board import Board
from tla.elevation import marching_squares_segments
from tla.rendering.hex_render import board_pixel_bounds, draw_board, draw_contour


class GameView(arcade.View):
    def __init__(self, board: Board) -> None:
        super().__init__()
        self.board = board
        self.hex_size = board.hex_pixel_size
        min_x, min_y, max_x, max_y = board_pixel_bounds(board, self.hex_size)
        self.origin_x = -min_x
        self.origin_y = -min_y
        self.contour_segments = (
            marching_squares_segments(board.elevation) if board.elevation else []
        )

    def on_show_view(self) -> None:
        self.window.background_color = arcade.color.BLACK

    def on_draw(self) -> None:
        self.clear()
        draw_board(self.board, self.hex_size, self.origin_x, self.origin_y)
        draw_contour(self.contour_segments, self.origin_x, self.origin_y)
