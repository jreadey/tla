"""Optional HDF5 persistence of `tla.ai.enemy_model.EnemyModel` position-
belief fields, for offline inspection/animation after a game.

Entirely opt-in and isolated: `tla.ai.enemy_model` never imports this module
or `h5py` itself -- it only calls `.append(ship_id, ordinal, values)` on
whatever object it's handed (see `EnemyModel`'s `belief_store` parameter),
so a game run with no persistence requested never needs `h5py` installed at
all. Requires the optional "hdf5" extra -- `pip install -e .[hdf5]`.

`BeliefStore` (write) and `BeliefReader` (read) are the two halves of one
file format: one HDF5 group per tracked ship id (named by that id's string
form), each holding two resizable datasets --

- `field`: `(layers, rows, cols)` float64, one `(rows, cols)` layer per
  actual belief recompute (each `tla.ai.hexfield.HexField.diffuse_step`
  call `EnemyModel` drives) -- not per turn; a ship whose movement stat is
  N gets up to N layers in a single turn.
- `ordinal`: `(layers,)` int64, `turn_ordinal(...)` as of each field layer,
  monotonically non-decreasing -- lets a reader find "the most recent
  belief as of this exact point in the game" without needing to know a
  ship's movement stat or count diffuse_step calls itself.

Only individually-tracked ships are recorded (their id is the natural group
name); a `KindPool`'s shared, anonymous field has no single id to name a
group after and is never persisted here.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from tla.ai.belief_ordinal import turn_ordinal  # noqa: F401 -- re-exported, see its own module docstring


class BeliefStore:
    """Each write flushes immediately -- same crash-safety tradeoff as
    `tla.replay.ReplayWriter`."""

    def __init__(self, path: str | Path) -> None:
        self._file = h5py.File(path, "w")
        self.finalized = False

    def append(self, ship_id: int, ordinal: int, values: np.ndarray) -> None:
        group = self._file.require_group(str(ship_id))
        field = group.get("field")
        if field is None:
            rows, cols = values.shape
            field = group.create_dataset(
                "field", shape=(0, rows, cols), maxshape=(None, rows, cols), chunks=(1, rows, cols), dtype="float64"
            )
            ordinal_ds = group.create_dataset("ordinal", shape=(0,), maxshape=(None,), chunks=(64,), dtype="int64")
        else:
            ordinal_ds = group["ordinal"]
        field.resize(field.shape[0] + 1, axis=0)
        field[-1] = values
        ordinal_ds.resize(ordinal_ds.shape[0] + 1, axis=0)
        ordinal_ds[-1] = ordinal
        self._file.flush()

    def close(self) -> None:
        """No-op if already closed -- safe to call speculatively, same
        convention as `tla.replay.ReplayWriter.write_final`."""
        if self.finalized:
            return
        self.finalized = True
        self._file.close()


class BeliefReader:
    """Read-only counterpart to `BeliefStore` -- opens an existing belief
    file and answers "what did we believe about ship X's position as of
    this point in the game" queries. `h5py` slices lazily, so this never
    loads the whole file into memory up front."""

    def __init__(self, path: str | Path) -> None:
        self._file = h5py.File(path, "r")

    def tracked_ship_ids(self) -> set[int]:
        return {int(name) for name in self._file.keys()}

    def field_as_of(self, ship_id: int, ordinal: int) -> np.ndarray | None:
        """The most recently recorded belief field for `ship_id` at or
        before `ordinal` (see `turn_ordinal`) -- None if `ship_id` was
        never tracked at all, or nothing had been recorded for it yet by
        that point in the game."""
        group = self._file.get(str(ship_id))
        if group is None:
            return None
        ordinals = group["ordinal"][:]
        index = int(np.searchsorted(ordinals, ordinal, side="right")) - 1
        if index < 0:
            return None
        return group["field"][index]

    def close(self) -> None:
        self._file.close()
