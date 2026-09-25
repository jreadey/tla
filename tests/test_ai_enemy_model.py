import random

from tla.ai.enemy_model import EnemyModel, _initial_candidate_hexes
from tla.ai.hexfield import HexField, HexFieldGeometry
from tla.board import Board
from tla.config import AiConfig, Config, FleetConfig, FowConfig, ProductionConfig
from tla.fleet_setup import _pick_start_hex
from tla.game_state import BattleLogEntry, GameState, PortProduction
from tla.hexgrid import AxialCoord, hexes_in_range
from tla.mapgen import largest_sea_component
from tla.ship import Ship, ShipKind
from tla.tile import PLAYER_A, PLAYER_B, Tile, TerrainType


def _sea_board(radius: int = 10) -> Board:
    board = Board(width=radius * 2 + 1, height=radius * 2 + 1)
    for coord in hexes_in_range(AxialCoord(0, 0), radius):
        board.tiles[coord] = Tile(coord=coord, terrain=TerrainType.SEA)
    return board


def _ship(coord: AxialCoord, kind: ShipKind, owner, ship_id: int, hp: int | None = None) -> Ship:
    stats = Config().ship_stats.stats[kind]
    return Ship(
        id=ship_id, kind=kind, owner=owner, position=coord,
        current_hp=hp if hp is not None else stats.hp, movement_remaining=stats.movement,
    )


def _game_state(board: Board, ships: list[Ship], config: Config) -> GameState:
    return GameState(config=config, board=board, ships={s.id: s for s in ships})


# -- roster: persistence, sightings, our own attacks ------------------------


def test_tracked_ship_persists_across_unseen_turns():
    board = _sea_board()
    config = Config(fow=FowConfig(enabled=True), fleet=FleetConfig(counts={ShipKind.SUBMARINE: 1}))
    gs = _game_state(board, [], config=config)
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)

    model._resolve(99, ShipKind.SUBMARINE, AxialCoord(1, 0), 4, turn=1)
    model._ensure_fields_for_unseen({})
    assert model.alive_count(ShipKind.SUBMARINE) == 1

    for t in range(2, 6):
        gs.turn_number = t
        model.begin_turn(gs)
        model.end_turn(gs)

    assert 99 in model.tracked_ships()
    assert model.tracked_ships()[99].kind == ShipKind.SUBMARINE
    assert model.alive_count(ShipKind.SUBMARINE) == 1


def test_sighting_collapses_belief_to_the_observed_hex():
    board = _sea_board()
    config = Config(fow=FowConfig(enabled=False), fleet=FleetConfig(counts={ShipKind.DESTROYER: 1}))
    own_ship = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    enemy_ship = _ship(AxialCoord(3, 0), ShipKind.DESTROYER, PLAYER_B, 50)
    gs = _game_state(board, [own_ship, enemy_ship], config=config)
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)

    model.begin_turn(gs)

    assert 50 in model.tracked_ships()
    assert model.most_likely_hex(50) == AxialCoord(3, 0)
    assert model.kind_pools()[ShipKind.DESTROYER].count == 0


def _battle_entry(defender_id, defender_kind, hp_after, sunk, hex_=AxialCoord(5, 0)) -> BattleLogEntry:
    return BattleLogEntry(
        attacker_id=1, attacker_kind=ShipKind.BATTLESHIP, attacker_owner=PLAYER_A,
        defender_id=defender_id, defender_kind=defender_kind, defender_owner=PLAYER_B,
        battle_hex=hex_, damage_to_defender=4, damage_to_attacker=3,
        attacker_carrier_bonus=0, defender_carrier_bonus=0,
        defender_hp_after=hp_after, attacker_hp_after=9, defender_sunk=sunk, attacker_sunk=False,
    )


def test_end_turn_updates_hp_and_promotes_from_our_own_attack():
    board = _sea_board()
    config = Config(fow=FowConfig(enabled=True), fleet=FleetConfig(counts={ShipKind.CRUISER: 1}))
    gs = _game_state(board, [], config=config)
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)
    assert model.alive_count(ShipKind.CRUISER) == 1
    assert 42 not in model.tracked_ships()

    gs.battle_log.append(_battle_entry(42, ShipKind.CRUISER, hp_after=4, sunk=False))
    model.end_turn(gs)

    assert 42 in model.tracked_ships()
    tracked = model.tracked_ships()[42]
    assert tracked.last_known_hp == 4
    assert tracked.last_seen_position == AxialCoord(5, 0)
    assert model.alive_count(ShipKind.CRUISER) == 1  # promoted, not double-counted
    assert model.kind_pools()[ShipKind.CRUISER].count == 0


def test_end_turn_removes_a_ship_we_sink_whether_or_not_it_was_already_tracked():
    board = _sea_board()
    config = Config(fow=FowConfig(enabled=True), fleet=FleetConfig(counts={ShipKind.CRUISER: 2}))
    gs = _game_state(board, [], config=config)
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)

    # Ship 7 already individually tracked; ship 8 still anonymous in the pool.
    model._resolve(7, ShipKind.CRUISER, AxialCoord(2, 0), 8, turn=1)
    assert model.alive_count(ShipKind.CRUISER) == 2

    gs.battle_log.append(_battle_entry(7, ShipKind.CRUISER, hp_after=0, sunk=True, hex_=AxialCoord(2, 0)))
    gs.battle_log.append(_battle_entry(8, ShipKind.CRUISER, hp_after=0, sunk=True, hex_=AxialCoord(6, 0)))
    model.end_turn(gs)

    assert 7 not in model.tracked_ships()
    assert model.alive_count(ShipKind.CRUISER) == 0


# -- diffusion excludes our own current vision -------------------------------
# Regression tests for a real bug found via replay review: a hex within the
# tracking player's own current vision (e.g. near one of their carriers) kept
# a nonzero belief reading for a ship that would certainly have been sighted
# there if it really were present.


def test_diffusion_excludes_a_hex_within_our_own_current_vision():
    board = _sea_board(radius=15)
    config = Config(fow=FowConfig(enabled=True), fleet=FleetConfig(counts={ShipKind.BATTLESHIP: 1}))
    own_carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 100)
    gs = _game_state(board, [own_carrier], config=config)
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)

    # Distance 6 from the carrier -- a battleship's own movement (4) can
    # reach as close as hex-distance 2 (well within the carrier's own
    # port_and_carrier_visibility_radius=4) but also as far as distance
    # 10 (well outside it), so this turn's diffusion genuinely straddles
    # the vision boundary -- a partial exclusion, not the ship's whole
    # spread (see the *_all_reachable_positions_within_vision variant
    # below for why that all-excluded case is actually correct too).
    model._resolve(7, ShipKind.BATTLESHIP, AxialCoord(6, 0), 12, turn=1)
    model._ensure_fields_for_unseen({})

    gs.turn_number = 2
    model._diffuse_all(gs)

    field = model.tracked_ships()[7].field
    row, col = model._geometry.to_index(AxialCoord(2, 0))  # within the carrier's vision
    assert field.values[row, col] == 0.0
    assert abs(field.total_mass() - 1.0) < 1e-9


def test_diffusion_excluding_only_once_per_turn_avoids_total_collapse():
    """Regression test for a real bug found via replay review: excluding
    after *every individual* diffuse step (rather than once, after the
    full turn's worth of steps) could cascade a field to total collapse
    -- zero mass everywhere, forever, since diffusing an all-zero field
    stays all-zero -- whenever a ship's reachable area for one turn
    happened to sit entirely inside our own vision radius even briefly
    mid-turn: each cut re-concentrated the survivors right at the vision
    boundary, where the very next step immediately re-diffused some of
    them straight back in. A battleship (movement 4) starting only 2 hexes
    from a carrier with a 4-hex vision radius is exactly such a case --
    its single-step reach (3) can't yet escape the radius, so
    exclude-every-step finds nothing to survive on step 1 and stays at
    zero forever after. Letting the full 4-step diffusion run
    uninterrupted first (reaching as far as hex-distance 6, well outside
    the radius) before excluding once at the end fixes this -- real,
    positive belief mass should survive."""
    board = _sea_board(radius=15)
    config = Config(fow=FowConfig(enabled=True), fleet=FleetConfig(counts={ShipKind.BATTLESHIP: 1}))
    own_carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 100)
    gs = _game_state(board, [own_carrier], config=config)
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)

    model._resolve(7, ShipKind.BATTLESHIP, AxialCoord(2, 0), 12, turn=1)
    model._ensure_fields_for_unseen({})

    gs.turn_number = 2
    model._diffuse_all(gs)

    field = model.tracked_ships()[7].field
    assert abs(field.total_mass() - 1.0) < 1e-9
    row, col = model._geometry.to_index(AxialCoord(2, 0))  # still within the carrier's vision
    assert field.values[row, col] == 0.0  # excluded, as always


def test_diffusion_does_not_exclude_our_vision_for_a_submarine_last_seen_submerged():
    board = _sea_board()
    config = Config(fow=FowConfig(enabled=True), fleet=FleetConfig(counts={ShipKind.SUBMARINE: 1}))
    own_carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 100)
    gs = _game_state(board, [own_carrier], config=config)
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)

    model._resolve(7, ShipKind.SUBMARINE, AxialCoord(2, 0), 4, turn=1, surfaced=False)
    model._ensure_fields_for_unseen({})

    gs.turn_number = 2
    model._diffuse_all(gs)

    field = model.tracked_ships()[7].field
    row, col = model._geometry.to_index(AxialCoord(2, 0))
    assert field.values[row, col] > 0.0  # stealth beats our vision here -- not excluded


def test_ensure_fields_for_unseen_forgets_a_submarines_confirmed_surfaced_state():
    """Regression test for a real bug found via replay review: a submarine
    actually sighted surfaced stayed "confidently surfaced" in our model
    forever after, even once it went unseen -- so _detectable_if_present
    kept saying "we'd see it if it were here," and the very next diffusion
    pass wiped its believed mass out of our own vision radius entirely,
    when the truth is the opposite: a submarine we lose sight of might
    have just dived right where we last saw it."""
    board = _sea_board()
    config = Config(fow=FowConfig(enabled=True), fleet=FleetConfig(counts={ShipKind.SUBMARINE: 1}))
    gs = _game_state(board, [], config=config)
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)

    model._resolve(7, ShipKind.SUBMARINE, AxialCoord(2, 0), 4, turn=1, surfaced=True)
    assert model._tracked[7].last_known_surfaced is True  # a real sighting, confidently surfaced

    model._ensure_fields_for_unseen({})  # goes unseen the very next turn, no re-sighting

    assert model._tracked[7].last_known_surfaced is None


def test_diffusion_does_not_exclude_our_vision_for_a_submarine_that_went_unseen_after_being_surfaced():
    board = _sea_board()
    config = Config(fow=FowConfig(enabled=True), fleet=FleetConfig(counts={ShipKind.SUBMARINE: 1}))
    own_carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 100)
    gs = _game_state(board, [own_carrier], config=config)
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)

    model._resolve(7, ShipKind.SUBMARINE, AxialCoord(2, 0), 4, turn=1, surfaced=True)
    model._ensure_fields_for_unseen({})  # unseen the next turn -- may well have dived right here

    gs.turn_number = 2
    model._diffuse_all(gs)

    field = model.tracked_ships()[7].field
    row, col = model._geometry.to_index(AxialCoord(2, 0))
    assert field.values[row, col] > 0.0  # not excluded, same as a submarine last seen submerged


def test_diffusion_excludes_our_vision_for_a_kind_pool_too():
    board = _sea_board(radius=15)
    config = Config(fow=FowConfig(enabled=True), fleet=FleetConfig(counts={ShipKind.DESTROYER: 0}))
    own_carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 100)
    gs = _game_state(board, [own_carrier], config=config)
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)

    # Distance 6 -- destroyer movement 3 reaches as close as distance 3
    # (within the carrier's vision radius 4) and as far as distance 9
    # (outside it), a partial overlap -- see the battleship test above
    # for why starting too close instead would correctly zero everything.
    pool = model._pool_for(ShipKind.DESTROYER)
    pool.count = 1
    pool.field.set_point_mass(AxialCoord(6, 0))

    gs.turn_number = 2
    model._diffuse_all(gs)

    row, col = model._geometry.to_index(AxialCoord(2, 0))
    assert pool.field.values[row, col] == 0.0
    assert abs(pool.field.total_mass() - 1.0) < 1e-9


# -- deterministic production tracking ---------------------------------------


def _port_board(port: AxialCoord) -> Board:
    board = _sea_board()
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B, port_controller=PLAYER_B)
    return board


def test_production_diffing_detects_a_new_ship_and_ignores_a_recapture_reset():
    port = AxialCoord(0, 0)
    board = _port_board(port)
    config = Config(
        fow=FowConfig(enabled=True),
        fleet=FleetConfig(counts={ShipKind.PATROL_BOAT: 0}),
        production=ProductionConfig(build_order=[ShipKind.PATROL_BOAT, ShipKind.DESTROYER]),
    )
    gs = _game_state(board, [], config=config)
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)
    assert model.alive_count() == 0

    gs.players[PLAYER_B].port_production[port] = PortProduction(next_index=1, points=0)
    gs.turn_number = 2
    model.begin_turn(gs)
    assert model.alive_count(ShipKind.PATROL_BOAT) == 1
    assert model.kind_pools()[ShipKind.PATROL_BOAT].field.mass_near(port, 0) > 0

    # Opponent loses the port (control flips away) -- must not be read as a
    # loss of ships, just no longer a production source to watch.
    board.tiles[port].port_controller = PLAYER_A
    del gs.players[PLAYER_B].port_production[port]
    gs.turn_number = 3
    model.begin_turn(gs)
    assert model.alive_count(ShipKind.PATROL_BOAT) == 1

    # Recaptured back -- production resets to next_index=0. That reset must
    # not be misread as a spawn (or worse, as negative spawns).
    board.tiles[port].port_controller = PLAYER_B
    gs.players[PLAYER_B].port_production[port] = PortProduction(next_index=0, points=0)
    gs.turn_number = 4
    model.begin_turn(gs)
    assert model.alive_count(ShipKind.PATROL_BOAT) == 1

    # A genuine second spawn afterward is still counted correctly.
    gs.players[PLAYER_B].port_production[port].next_index = 1
    gs.turn_number = 5
    model.begin_turn(gs)
    assert model.alive_count(ShipKind.PATROL_BOAT) == 2


# -- stale fold-back ----------------------------------------------------------


def test_stale_tracked_ship_folds_back_into_its_pool():
    board = _sea_board()
    config = Config(
        fow=FowConfig(enabled=True),
        fleet=FleetConfig(counts={ShipKind.CRUISER: 1}),
        ai=AiConfig(enemy_model_stale_turns=3),
    )
    gs = _game_state(board, [], config=config)
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)

    model._resolve(7, ShipKind.CRUISER, AxialCoord(2, 0), 8, turn=1)
    model._ensure_fields_for_unseen({})
    assert model.kind_pools()[ShipKind.CRUISER].count == 0

    for t in range(2, 7):
        gs.turn_number = t
        model.begin_turn(gs)
        model.end_turn(gs)

    assert 7 not in model.tracked_ships()
    assert model.kind_pools()[ShipKind.CRUISER].count == 1


# -- initial belief matches tla.fleet_setup's own placement logic -----------


def test_initial_candidate_hexes_matches_fleet_setups_own_placement_logic():
    board = _sea_board()
    port = AxialCoord(0, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    main_sea = largest_sea_component(board)
    ports = [port]

    candidates = _initial_candidate_hexes(board, ports)

    rng = random.Random(12345)
    for _ in range(500):
        picked = _pick_start_hex(board, ports, set(), main_sea, rng)
        assert picked in candidates


# -- HexField / HexFieldGeometry diffusion math ------------------------------


def _land_wall_board(radius: int = 10, wall_q: int = 3) -> Board:
    board = _sea_board(radius)
    land_cells = set()
    for r in range(-5, 6):
        coord = AxialCoord(wall_q, r)
        if coord in board.tiles:
            board.tiles[coord] = Tile(coord=coord, terrain=TerrainType.LAND)
            land_cells.add(coord)
    return board, land_cells


def test_diffusion_conserves_mass_and_never_crosses_land():
    board, land_cells = _land_wall_board()
    geo = HexFieldGeometry.from_board(board)
    field = HexField(geo)
    field.set_point_mass(AxialCoord(0, 0))

    for _ in range(15):
        field.diffuse_step()
        assert abs(field.total_mass() - 1.0) < 1e-9

    occupied = field.as_dict()
    assert not (set(occupied) & land_cells)


def test_diffusion_spread_scales_with_movement_stat():
    board = _sea_board()
    geo = HexFieldGeometry.from_board(board)

    fast = HexField(geo)  # e.g. a patrol boat, movement 6
    fast.set_point_mass(AxialCoord(0, 0))
    for _ in range(6):
        fast.diffuse_step()

    slow = HexField(geo)  # e.g. a battleship, movement 4
    slow.set_point_mass(AxialCoord(0, 0))
    for _ in range(4):
        slow.diffuse_step()

    far_hex = AxialCoord(6, 0)
    assert fast.mass_near(far_hex, 0) > 0
    assert slow.mass_near(far_hex, 0) == 0
