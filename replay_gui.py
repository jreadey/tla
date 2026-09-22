"""Graphical, read-only replay viewer -- opens a window showing the map
and every ship's actual hex-by-hex route across the whole game (see
tla/rendering/replay_view.py). Usage:

    python replay_gui.py path/to/replay.jsonl

Controls: on-screen transport buttons (go to start / step back / play /
pause / step forward / go to end) at the bottom of the window, or the
keyboard -- Left/Right steps one recorded half-turn at a time, Space steps
forward while paused (or pauses while playing), J plays in reverse (no
button for this -- L also plays forward, K also pauses), Home/End jump to
the start/end. Right-click-drag pans, mouse wheel zooms, and hovering a
ship shows a tooltip.

Stepping forward (the step-forward button, or Right/Space) plays a short
animation: each ship that moved that half-turn glides along its actual
route, one ship at a time, in the order it was actually moved -- older
replay files written before this existed still work, just with a plain
straight-line jump for any half-turn with no recorded route.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tla.rendering.replay_app import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Graphical viewer for a tla replay .jsonl file")
    parser.add_argument("path", type=str, help="Path to a .jsonl replay file written by --replay")
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.path).read_text().splitlines() if line.strip()]
    run(records)


if __name__ == "__main__":
    main()
