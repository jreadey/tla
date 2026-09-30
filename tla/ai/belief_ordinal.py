"""`turn_ordinal` alone, split out of `tla.ai.belief_store` so it can be
imported by both the writer (`tla.ai.enemy_model`, which must stay
importable with no `h5py` installed) and the reader
(`tla.rendering.replay_view`) without either pulling in `h5py` just for
this one piece of arithmetic -- `tla.ai.belief_store` itself imports `h5py`
unconditionally at module level, so importing anything from it at all
would defeat the "no h5py needed unless belief persistence is actually
used" guarantee both of those modules otherwise keep.
"""

from __future__ import annotations


def turn_ordinal(turn_number: int, move_b: bool) -> int:
    """A single, chronologically-sortable integer for a specific
    half-turn -- plain `turn_number` alone can't distinguish "before this
    turn's move_a" from "after this turn's move_b" (both share the same
    `turn_number`), which matters because each side's own `EnemyModel`
    only ever updates belief during *its own* half of a turn (see
    `EnemyModel.begin_turn`/`end_turn`, which pass
    `game_state.phase == TurnPhase.MOVE_B` for `move_b`). A reader
    computes the exact same ordinal from a replay record's own
    `turn_number`/`phase` fields (see
    `tla.rendering.replay_view._record_ordinal`) so "what did we believe
    as of this specific record" lines up with when the belief that's
    stored was actually written, not just which turn it happened to
    share. `move_b=True` sorts after `move_b=False` for the same
    `turn_number`, matching move_a always preceding move_b within a
    turn."""
    return 2 * turn_number + (1 if move_b else 0)
