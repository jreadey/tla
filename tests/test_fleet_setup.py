from tla.config import Config, FleetConfig, MapConfig, PortConfig
from tla.fleet_setup import place_initial_fleets
from tla.mapgen import generate_map, largest_sea_component
from tla.ship import ShipKind
from tla.tile import PLAYER_A, PLAYER_B


def _small_config() -> Config:
    return Config(
        map=MapConfig(width=16, height=10, noise_scale=5.0),
        ports=PortConfig(ports_per_player=2, min_port_spacing=2),
        fleet=FleetConfig(counts={ShipKind.DESTROYER: 3, ShipKind.SUBMARINE: 2}),
    )


def test_fleet_counts_match_config_per_player():
    config = _small_config()
    board = generate_map(config.map, config.ports, seed=3)
    ships = place_initial_fleets(board, config, seed=3)

    for player in (PLAYER_A, PLAYER_B):
        owned = [s for s in ships.values() if s.owner == player]
        assert sum(1 for s in owned if s.kind == ShipKind.DESTROYER) == 3
        assert sum(1 for s in owned if s.kind == ShipKind.SUBMARINE) == 2


def test_ships_start_on_occupiable_hexes():
    config = _small_config()
    board = generate_map(config.map, config.ports, seed=5)
    ships = place_initial_fleets(board, config, seed=5)

    for ship in ships.values():
        assert board.is_occupiable(ship.position)


def test_no_two_ships_of_the_same_owner_share_a_hex():
    config = _small_config()
    board = generate_map(config.map, config.ports, seed=11)
    ships = place_initial_fleets(board, config, seed=11)

    for player in (PLAYER_A, PLAYER_B):
        positions = [s.position for s in ships.values() if s.owner == player]
        assert len(positions) == len(set(positions))


def test_current_hp_matches_configured_stat():
    config = _small_config()
    board = generate_map(config.map, config.ports, seed=8)
    ships = place_initial_fleets(board, config, seed=8)

    for ship in ships.values():
        assert ship.current_hp == config.ship_stats.stats[ship.kind].hp


def test_starting_ships_never_occupy_a_port():
    # Ports must start free so production has somewhere to place a ship
    # from turn one, rather than being blocked by the starting fleet.
    config = _small_config()
    board = generate_map(config.map, config.ports, seed=5)
    ships = place_initial_fleets(board, config, seed=5)

    port_coords = set(board.ports_for(PLAYER_A)) | set(board.ports_for(PLAYER_B))
    ship_positions = {s.position for s in ships.values()}
    assert port_coords.isdisjoint(ship_positions)


def test_starting_ships_are_never_placed_in_an_isolated_pond():
    # Regression: seed 323368 with the default config stranded a destroyer
    # in a small sea pocket within hex-distance of a port but with no
    # actual path (through occupiable hexes) to the rest of the map --
    # hex-distance search doesn't account for land blocking the route.
    config = Config()
    seed = 323368
    board = generate_map(config.map, config.ports, seed=seed)
    ships = place_initial_fleets(board, config, seed)
    main_sea = largest_sea_component(board)

    for ship in ships.values():
        assert ship.position in main_sea


def test_placement_is_deterministic_given_seed():
    config = _small_config()
    board = generate_map(config.map, config.ports, seed=42)
    ships1 = place_initial_fleets(board, config, seed=42)
    ships2 = place_initial_fleets(board, config, seed=42)

    positions1 = sorted((s.owner, s.kind.value, s.position) for s in ships1.values())
    positions2 = sorted((s.owner, s.kind.value, s.position) for s in ships2.values())
    assert positions1 == positions2
