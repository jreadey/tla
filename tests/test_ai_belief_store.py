import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from tla.ai.belief_store import BeliefReader, BeliefStore
from tla.ai.enemy_model import EnemyModel
from tla.ai.policy import NaivePolicy
from tla.board import Board
from tla.config import Config, FleetConfig, FowConfig
from tla.game_state import GameState, TurnPhase, new_game
from tla.hexgrid import AxialCoord, hexes_in_range
from tla.rendering.replay_view import _record_ordinal
from tla.ship import ShipKind
from tla.tile import PLAYER_A, PLAYER_B, Tile, TerrainType
from tla.turn_manager import TurnManager


def _sea_board(radius: int = 10) -> Board:
    board = Board(width=radius * 2 + 1, height=radius * 2 + 1)
    for coord in hexes_in_range(AxialCoord(0, 0), radius):
        board.tiles[coord] = Tile(coord=coord, terrain=TerrainType.SEA)
    return board


def test_belief_store_appends_a_resizable_layer_per_call_with_its_ordinal(tmp_path):
    path = tmp_path / "belief.h5"
    store = BeliefStore(path)

    a = np.zeros((3, 4))
    a[0, 0] = 1.0
    store.append(42, 1, a)
    b = np.zeros((3, 4))
    b[1, 2] = 1.0
    store.append(42, 3, b)
    store.close()

    with h5py.File(path, "r") as f:
        assert list(f.keys()) == ["42"]
        group = f["42"]
        assert group["field"].shape == (2, 3, 4)
        assert group["field"][0, 0, 0] == 1.0
        assert group["field"][1, 1, 2] == 1.0
        assert list(group["ordinal"]) == [1, 3]


def test_belief_store_close_is_idempotent(tmp_path):
    store = BeliefStore(tmp_path / "belief.h5")
    store.close()
    store.close()  # must not raise


def test_enemy_model_records_one_layer_per_diffuse_step_not_per_turn(tmp_path):
    path = tmp_path / "belief.h5"
    store = BeliefStore(path)

    board = _sea_board()
    config = Config(fow=FowConfig(enabled=True), fleet=FleetConfig(counts={ShipKind.PATROL_BOAT: 1}))
    gs = GameState(config=config, board=board, ships={})
    model = EnemyModel(gs, PLAYER_A, PLAYER_B, belief_store=store)

    # Resolve one ship individually, then let it go unseen so it gets a
    # diffusing field -- patrol_boat movement is 6, so one begin_turn call
    # (one turn's worth of diffusion) should append exactly 6 layers, all
    # stamped with that same ordinal (turn_number=2, default phase MOVE_A
    # -> turn_ordinal(2, move_b=False) == 4).
    model._resolve(7, ShipKind.PATROL_BOAT, AxialCoord(0, 0), 2, turn=1)
    model._ensure_fields_for_unseen({})

    gs.turn_number = 2
    model.begin_turn(gs)
    store.close()

    with h5py.File(path, "r") as f:
        assert list(f.keys()) == ["7"]
        assert f["7"]["field"].shape[0] == 6
        assert list(f["7"]["ordinal"]) == [4] * 6


def test_a_sighting_records_a_point_mass_layer_after_any_diffuse_layers(tmp_path):
    path = tmp_path / "belief.h5"
    store = BeliefStore(path)

    board = _sea_board()
    config = Config(fow=FowConfig(enabled=False), fleet=FleetConfig(counts={ShipKind.DESTROYER: 1}))
    from tla.ship import Ship

    stats = Config().ship_stats.stats[ShipKind.DESTROYER]
    enemy_ship = Ship(id=50, kind=ShipKind.DESTROYER, owner=PLAYER_B, position=AxialCoord(3, 0), current_hp=stats.hp)
    gs = GameState(config=config, board=board, ships={50: enemy_ship})
    model = EnemyModel(gs, PLAYER_A, PLAYER_B, belief_store=store)

    # First tracked while unseen -- give it a diffusing field manually --
    # then a sighting happens (fow disabled, so the ship is always
    # "visible"): begin_turn should diffuse it first (writing pre-sighting
    # layers), then immediately collapse+record a point-mass sighting
    # layer for the same turn, appended *after* those.
    model._resolve(50, ShipKind.DESTROYER, AxialCoord(9, 0), stats.hp, turn=1)
    model._ensure_fields_for_unseen({})

    gs.turn_number = 2
    model.begin_turn(gs)
    store.close()

    with h5py.File(path, "r") as f:
        ordinals = list(f["50"]["ordinal"][:])
        assert ordinals[-1] == 4  # turn_ordinal(2, move_b=False), default phase MOVE_A
        last_layer = f["50"]["field"][-1]
        assert last_layer.sum() == 1.0
        peak_row, peak_col = divmod(int(last_layer.argmax()), last_layer.shape[1])

    reader = BeliefReader(path)
    # field_as_of must return this crisp, just-sighted layer -- not the
    # stale diffuse cloud recorded moments earlier in the same call.
    field = reader.field_as_of(50, 4)
    assert field is not None
    assert (field == last_layer).all()
    reader.close()


def test_a_ship_that_goes_unseen_starts_diffusing_the_same_turn_not_a_turn_late(tmp_path):
    """Regression test for a real bug found via replay review: a ship's
    field used to only get created by _ensure_fields_for_unseen *after*
    that turn's _diffuse_all pass already ran, so it missed its own first
    diffusion step -- field_as_of would keep reporting the last sighting
    as still 100% certain for a full extra turn after the ship actually
    left view."""
    path = tmp_path / "belief.h5"
    store = BeliefStore(path)

    board = _sea_board()
    config = Config(fow=FowConfig(enabled=False), fleet=FleetConfig(counts={ShipKind.BATTLESHIP: 1}))
    from tla.ship import Ship

    stats = Config().ship_stats.stats[ShipKind.BATTLESHIP]
    enemy_ship = Ship(id=1, kind=ShipKind.BATTLESHIP, owner=PLAYER_B, position=AxialCoord(3, 0), current_hp=stats.hp)
    gs = GameState(config=config, board=board, ships={1: enemy_ship})
    model = EnemyModel(gs, PLAYER_A, PLAYER_B, belief_store=store)

    gs.turn_number = 1
    model.begin_turn(gs)  # fow disabled -- always "visible" -- sighted, field stays None
    model.end_turn(gs)
    assert model.tracked_ships()[1].field is None

    # The ship "leaves view" -- removed from game_state.ships entirely,
    # equivalent to moving out of range from this model's perspective (it
    # never reads game_state.ships directly, only via
    # enemy_ships_visible_to -- see the module docstring).
    del gs.ships[1]
    gs.turn_number = 2
    model.begin_turn(gs)
    store.close()

    with h5py.File(path, "r") as f:
        ordinals = list(f["1"]["ordinal"][:])
        # turn_ordinal(2, move_b=False) == 4, default phase MOVE_A
        assert 4 in ordinals, "should have started diffusing (and recording) the same turn it went unseen"
        assert float(f["1"]["field"][-1].max()) < 1.0, "should no longer be a 100%-certain point mass"

    reader = BeliefReader(path)
    field = reader.field_as_of(1, 4)
    assert field.max() < 1.0
    reader.close()


def test_field_as_of_does_not_see_this_turns_sighting_before_it_happens(tmp_path):
    """Regression test for a real bug found via replay review: a viewer
    looking at "turn 6, before player B's own move" was seeing player
    B's turn-6 sighting anyway. Both halves of a turn share the same
    turn_number, and the old code stamped/queried belief data using bare
    turn_number alone -- with nothing to tell "before this turn's move_a"
    from "after this turn's move_b" apart. A sighting made during move_b
    must not be visible yet from a move_a record of the very same turn."""
    path = tmp_path / "belief.h5"
    store = BeliefStore(path)

    board = _sea_board()
    config = Config(fow=FowConfig(enabled=False), fleet=FleetConfig(counts={ShipKind.BATTLESHIP: 1}))
    from tla.ship import Ship

    stats = Config().ship_stats.stats[ShipKind.BATTLESHIP]
    enemy_ship = Ship(id=1, kind=ShipKind.BATTLESHIP, owner=PLAYER_A, position=AxialCoord(9, 1), current_hp=stats.hp)
    gs = GameState(config=config, board=board, ships={1: enemy_ship}, turn_number=6, phase=TurnPhase.MOVE_B)
    model = EnemyModel(gs, PLAYER_B, PLAYER_A, belief_store=store)
    model.begin_turn(gs)  # fow disabled -- ship 1 always "visible" -- records a sighting for this turn's move_b
    store.close()

    reader = BeliefReader(path)
    move_a_record = {"turn_number": 6, "phase": "move_a"}
    move_b_record = {"turn_number": 6, "phase": "move_b"}

    assert reader.field_as_of(1, _record_ordinal(move_a_record)) is None
    field = reader.field_as_of(1, _record_ordinal(move_b_record))
    assert field is not None
    assert field.max() == 1.0
    reader.close()


def test_kind_pool_fields_are_not_persisted(tmp_path):
    path = tmp_path / "belief.h5"
    store = BeliefStore(path)

    board = _sea_board()
    config = Config(fow=FowConfig(enabled=True), fleet=FleetConfig(counts={ShipKind.SUBMARINE: 2}))
    gs = GameState(config=config, board=board, ships={})
    model = EnemyModel(gs, PLAYER_A, PLAYER_B, belief_store=store)

    gs.turn_number = 2
    model.begin_turn(gs)  # diffuses the SUBMARINE pool field -- never individually tracked
    store.close()

    with h5py.File(path, "r") as f:
        assert list(f.keys()) == []  # nothing recorded -- pools have no single ship id


def test_naive_policy_wires_a_shared_belief_store_end_to_end(tmp_path):
    path = tmp_path / "belief.h5"
    config = Config.load("configs/dev.json")
    gs = new_game(config, seed=1)
    tm = TurnManager(gs)
    policy = NaivePolicy(enemy_belief_path=path)

    # Enough half-turns for at least one sighting to happen and produce a
    # per-ship diffusing field -- 6 wasn't reliably enough on this map/fow
    # config (initial fleets start out of vision range of each other).
    for _ in range(20):
        player = 1 if gs.phase == TurnPhase.MOVE_A else 2
        list(policy.plan_movement(gs, player))
        tm.end_movement_phase()
    policy.close_enemy_belief_store()
    policy.close_enemy_belief_store()  # idempotent

    with h5py.File(path, "r") as f:
        assert len(f.keys()) > 0
        for name in f.keys():
            int(name)  # every group name is a valid ship id
            assert f[name]["field"].shape[0] == f[name]["ordinal"].shape[0]


# -- BeliefReader -------------------------------------------------------


def test_belief_reader_field_as_of_finds_the_most_recent_layer_at_or_before_turn(tmp_path):
    path = tmp_path / "belief.h5"
    store = BeliefStore(path)
    for turn, value in ((1, 1.0), (1, 2.0), (3, 3.0), (5, 4.0)):
        layer = np.zeros((2, 2))
        layer[0, 0] = value
        store.append(9, turn, layer)
    store.close()

    reader = BeliefReader(path)
    assert reader.field_as_of(9, 0) is None  # nothing recorded yet by turn 0
    assert reader.field_as_of(9, 1)[0, 0] == 2.0  # last of the two turn-1 layers
    assert reader.field_as_of(9, 2)[0, 0] == 2.0  # still the turn-1 layer, nothing newer yet
    assert reader.field_as_of(9, 3)[0, 0] == 3.0
    assert reader.field_as_of(9, 100)[0, 0] == 4.0  # far future -- most recent layer overall
    reader.close()


def test_belief_reader_returns_none_for_an_untracked_ship(tmp_path):
    path = tmp_path / "belief.h5"
    store = BeliefStore(path)
    store.append(1, 1, np.zeros((2, 2)))
    store.close()

    reader = BeliefReader(path)
    assert 1 in reader.tracked_ship_ids()
    assert reader.field_as_of(999, 5) is None
    reader.close()
