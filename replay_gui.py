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

Passing --belief adds a ship inventory panel: click a ship to overlay its
recorded position-belief field (see tla/ai/belief_store.py, written by
NaivePolicy(enemy_belief_path=...)) as a heatmap, alongside a marker at its
real position that turn -- the replay log always has ground truth, unlike
the AI's own fog-of-war-limited view, so this is a believed-vs-actual
comparison. --belief is normally optional: if main.py was run with both
--replay and --belief together, the belief file's path is saved in the
replay log itself, and this viewer auto-locates it from there -- pass
--belief explicitly only to override that, or for an older replay log
written before this existed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tla.rendering.replay_app import run


def _resolve_belief_path(replay_path: str, explicit_belief_path: str | None, records: list[dict]) -> str | None:
    """`explicit_belief_path` (the --belief flag) wins if given. Otherwise,
    fall back to whatever path was saved in the replay log's own initial
    record (see tla.replay.ReplayWriter.write_initial) -- None if that's
    empty, absent (an older replay log), or `--belief` was never used
    when the replay was recorded. A saved *relative* path is resolved
    against the replay file's own directory, not the current working
    directory -- the two files are meant to travel together, so this
    works regardless of where this viewer happens to be run from. A saved
    *absolute* path is returned unchanged (pathlib's `/` already does the
    right thing when the right-hand side is absolute)."""
    if explicit_belief_path is not None:
        return explicit_belief_path
    saved = records[0].get("belief_path") if records else None
    if saved is None:
        return None
    return str(Path(replay_path).parent / saved)


def main() -> None:
    parser = argparse.ArgumentParser(description="Graphical viewer for a tla replay .jsonl file")
    parser.add_argument("path", type=str, help="Path to a .jsonl replay file written by --replay")
    parser.add_argument(
        "--belief",
        type=str,
        default=None,
        help="Path to an HDF5 belief file written by NaivePolicy(enemy_belief_path=...) -- "
        "adds a clickable ship inventory with a position-belief heatmap overlay. Usually unnecessary: "
        "auto-located from the replay log if main.py was run with both --replay and --belief together. "
        "Pass this to override that, or for a replay log that has none saved.",
    )
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.path).read_text().splitlines() if line.strip()]

    belief_path = _resolve_belief_path(args.path, args.belief, records)
    if belief_path is not None and args.belief is None:
        print(f"Using belief file saved in the replay log: {belief_path}")

    belief_reader = None
    if belief_path is not None:
        from tla.ai.belief_store import BeliefReader  # local: only needs h5py if actually used

        belief_reader = BeliefReader(belief_path)

    run(records, belief_reader=belief_reader)


if __name__ == "__main__":
    main()
