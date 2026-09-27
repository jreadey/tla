import random

from tla.ai.enemy_model import EnemyModel, _initial_candidate_hexes
from tla.ai.hexfield import HexField, HexFieldGeometry, _UNREACHABLE_DISTANCE
from tla.board import Board
from tla.config import AiConfig, Config, FleetConfig, FowConfig, ProductionConfig
from tla.fleet_setup import _pick_start_hex
from tla.game_state import BattleLogEntry, GameState, PortProduction
from tla.hexgrid import AxialCoord, distance, hexes_in_range
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


def test_distance_grid_to_matches_open_sea_hex_distance():
    board = _sea_board()
    geo = HexFieldGeometry.from_board(board)
    target = AxialCoord(6, 0)

    grid = geo.distance_grid_to(target)

    for coord in [AxialCoord(0, 0), AxialCoord(-3, 2), AxialCoord(6, -4)]:
        row, col = geo.to_index(coord)
        assert grid[row, col] == distance(coord, target)


def test_distance_grid_to_routes_around_land_wall():
    board, _land_cells = _land_wall_board()
    geo = HexFieldGeometry.from_board(board)
    target = AxialCoord(6, 0)
    origin = AxialCoord(0, 0)

    grid = geo.distance_grid_to(target)

    row, col = geo.to_index(origin)
    # A solid wall sits directly between the two -- the real sea route is
    # strictly longer than the straight-line distance the wall blocks.
    assert grid[row, col] > distance(origin, target)


def test_distance_grid_to_unreachable_sentinel_for_disconnected_pocket():
    board = _sea_board()
    pocket = AxialCoord(-9, 0)
    # Wall the pocket off completely from the rest of the sea.
    for n in [AxialCoord(-8, 0), AxialCoord(-8, -1), AxialCoord(-9, 1), AxialCoord(-9, -1), AxialCoord(-10, 0), AxialCoord(-10, 1)]:
        if n in board.tiles:
            board.tiles[n] = Tile(coord=n, terrain=TerrainType.LAND)
    geo = HexFieldGeometry.from_board(board)

    grid = geo.distance_grid_to(AxialCoord(6, 0))

    row, col = geo.to_index(pocket)
    assert geo.sea_mask[row, col]  # still a legal sea hex, just cut off
    assert grid[row, col] == _UNREACHABLE_DISTANCE


def test_diffuse_step_with_bias_leans_toward_target():
    board = _sea_board()
    geo = HexFieldGeometry.from_board(board)
    target = AxialCoord(6, 0)
    grid = geo.distance_grid_to(target)

    biased = HexField(geo)
    biased.set_point_mass(AxialCoord(0, 0))
    unbiased = HexField(geo)
    unbiased.set_point_mass(AxialCoord(0, 0))

    for _ in range(6):
        biased.diffuse_step(bias_distance_grid=grid)
        unbiased.diffuse_step()

    assert biased.mass_near(target, 0) > unbiased.mass_near(target, 0)


def test_diffuse_step_with_bias_conserves_mass_around_land_wall():
    board, land_cells = _land_wall_board()
    geo = HexFieldGeometry.from_board(board)
    grid = geo.distance_grid_to(AxialCoord(6, 0))
    field = HexField(geo)
    field.set_point_mass(AxialCoord(0, 0))

    for _ in range(15):
        field.diffuse_step(bias_distance_grid=grid)
        assert abs(field.total_mass() - 1.0) < 1e-9

    occupied = field.as_dict()
    assert not (set(occupied) & land_cells)


def test_diffuse_step_with_bias_falls_back_to_isotropic_when_unreachable():
    board = _sea_board()
    geo = HexFieldGeometry.from_board(board)
    # An all-unreachable grid (target out of bounds) means no cell ever has
    # a "progress" neighbor -- every step must fall back to today's plain
    # isotropic split, matching an unbiased run exactly.
    unreachable_grid = geo.distance_grid_to(AxialCoord(9999, 9999))

    biased = HexField(geo)
    biased.set_point_mass(AxialCoord(0, 0))
    unbiased = HexField(geo)
    unbiased.set_point_mass(AxialCoord(0, 0))

    for _ in range(4):
        biased.diffuse_step(bias_distance_grid=unreachable_grid)
        unbiased.diffuse_step()

    assert (biased.values == unbiased.values).all()


# -- expected_strength_near --------------------------------------------------


def test_expected_strength_near_includes_a_resolved_ship_within_radius():
    board = _sea_board()
    config = Config(fleet=FleetConfig(counts={}))
    gs = _game_state(board, [], config=config)
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)

    model._resolve(7, ShipKind.BATTLESHIP, AxialCoord(1, 0), 9, turn=1)
    model._ensure_fields_for_unseen({})  # currently unseen -- point mass at (1, 0)

    stats = Config().ship_stats.stats[ShipKind.BATTLESHIP]
    hp, damage = model.expected_strength_near(AxialCoord(0, 0), radius=4)

    assert hp == 9  # last-known hp, not full stats hp
    assert damage == stats.damage


def test_expected_strength_near_scales_a_kind_pool_by_its_expected_count():
    board = _sea_board()
    config = Config(fleet=FleetConfig(counts={ShipKind.CRUISER: 4}))
    gs = _game_state(board, [], config=config)
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)

    pool = model._pool_for(ShipKind.CRUISER)
    pool.count = 4
    pool.field.set_point_mass(AxialCoord(0, 0))
    pool.field.renormalize_to(4)  # 4 expected ships, all at (0, 0)

    stats = Config().ship_stats.stats[ShipKind.CRUISER]
    hp, damage = model.expected_strength_near(AxialCoord(0, 0), radius=0)

    assert hp == 4 * stats.hp
    assert damage == 4 * stats.damage


def test_expected_strength_near_respects_the_kinds_filter():
    board = _sea_board()
    config = Config(fleet=FleetConfig(counts={}))
    gs = _game_state(board, [], config=config)
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)

    model._resolve(7, ShipKind.PATROL_BOAT, AxialCoord(0, 0), 2, turn=1)
    model._ensure_fields_for_unseen({})

    hp, damage = model.expected_strength_near(
        AxialCoord(0, 0), radius=4, kinds=frozenset({ShipKind.BATTLESHIP, ShipKind.CRUISER})
    )

    assert (hp, damage) == (0, 0)  # patrol boat filtered out


def test_expected_strength_near_is_zero_far_from_everything():
    board = _sea_board()
    config = Config(fleet=FleetConfig(counts={}))
    gs = _game_state(board, [], config=config)
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)

    model._resolve(7, ShipKind.BATTLESHIP, AxialCoord(9, 0), 12, turn=1)
    model._ensure_fields_for_unseen({})

    assert model.expected_strength_near(AxialCoord(0, 0), radius=1) == (0, 0)


# -- directional diffusion toward a threatened own port ----------------------


def _two_port_board() -> Board:
    board = _sea_board(radius=30)
    board.tiles[_PORT_NEAR] = Tile(
        coord=_PORT_NEAR, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A, port_controller=PLAYER_A
    )
    board.tiles[_PORT_FAR] = Tile(
        coord=_PORT_FAR, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A, port_controller=PLAYER_A
    )
    return board


_PORT_NEAR = AxialCoord(-20, 0)
_PORT_FAR = AxialCoord(20, 0)


def _model_with_threat_and_test_pool(ai_config: AiConfig) -> tuple[EnemyModel, GameState]:
    """A model belonging to PLAYER_A (reasoning about PLAYER_B) with two of
    PLAYER_A's own controlled ports -- _PORT_NEAR already showing believed
    PLAYER_B mass nearby (seeded directly, standing in for e.g. a real
    sighting or a diffused-in production ship), _PORT_FAR with none -- plus
    a fresh, never-sighted PATROL_BOAT pool seeded at the origin, exactly
    equidistant (20 hexes) from both ports, so any lean after diffusion is
    attributable only to the threat-mass bias, not starting proximity."""
    board = _two_port_board()
    config = Config(fow=FowConfig(enabled=True), fleet=FleetConfig(counts={}), ai=ai_config)
    gs = GameState(config=config, board=board, ships={})
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)

    threat_pool = model._pool_for(ShipKind.CRUISER)
    threat_pool.count = 3
    threat_pool.field.set_point_mass(AxialCoord(-19, 0), 3.0)

    test_pool = model._pool_for(ShipKind.PATROL_BOAT)
    test_pool.count = 1
    test_pool.field.set_point_mass(AxialCoord(0, 0), 1.0)

    return model, gs


def test_kind_pool_diffusion_leans_toward_most_threatened_own_port():
    model, gs = _model_with_threat_and_test_pool(AiConfig(enemy_model_directional_diffusion=True))
    assert model._most_threatened_own_port(gs) == _PORT_NEAR

    gs.turn_number = 2
    model._diffuse_all(gs)

    test_pool = model.kind_pools()[ShipKind.PATROL_BOAT]
    assert abs(test_pool.field.total_mass() - 1.0) < 1e-9
    # A straight, unobstructed line toward _PORT_NEAR has exactly one
    # "progress" direction at every step, so with no stay share retained
    # (see HexField.diffuse_step's bias_distance_grid docstring) the whole
    # mass lands on one exact hex after 6 (patrol boat's movement) steps --
    # a strong, precise signal that the bias is real, not just a vague lean.
    assert test_pool.field.mass_near(AxialCoord(-6, 0), 0) == 1.0
    assert test_pool.field.mass_near(_PORT_FAR, 20) == 0.0  # nothing drifted the other way


def test_directional_diffusion_disabled_matches_prior_isotropic_behavior():
    model, gs = _model_with_threat_and_test_pool(AiConfig(enemy_model_directional_diffusion=False))

    gs.turn_number = 2
    model._diffuse_all(gs)

    test_pool = model.kind_pools()[ShipKind.PATROL_BOAT]
    # A parallel, hand-run isotropic HexField (today's exact prior
    # behavior) started from the same point mass, same number of steps.
    reference = HexField(model._geometry)
    reference.set_point_mass(AxialCoord(0, 0))
    for _ in range(6):
        reference.diffuse_step()

    assert (test_pool.field.values == reference.values).all()


def test_directional_diffusion_only_affects_pools_not_tracked_ships():
    model, gs = _model_with_threat_and_test_pool(AiConfig(enemy_model_directional_diffusion=True))
    model._resolve(99, ShipKind.PATROL_BOAT, AxialCoord(0, 0), 2, turn=1)
    model._ensure_fields_for_unseen({})  # now out of sight -- gets a diffusing field

    gs.turn_number = 2
    model._diffuse_all(gs)

    tracked = model.tracked_ships()[99]
    # The pool (never individually sighted) concentrates on the biased
    # target exactly as in the test above; the tracked ship -- resolved,
    # so its intent is deliberately not assumed -- stays spread across
    # several hexes the plain isotropic walk would reach instead of
    # collapsing onto one.
    assert len(tracked.field.as_dict()) > 1
    assert tracked.field.mass_near(AxialCoord(-6, 0), 0) < tracked.field.total_mass()


def test_most_threatened_own_port_none_with_no_controlled_ports():
    board = _sea_board()
    config = Config(fow=FowConfig(enabled=True), fleet=FleetConfig(counts={}))
    gs = GameState(config=config, board=board, ships={})
    model = EnemyModel(gs, PLAYER_A, PLAYER_B)

    assert model._most_threatened_own_port(gs) is None
