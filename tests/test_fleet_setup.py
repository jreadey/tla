from tla.config import Config, FleetConfig, MapConfig, PortConfig
from tla.fleet_setup import place_initial_fleets
from tla.mapgen import generate_map
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


def test_placement_is_deterministic_given_seed():
    config = _small_config()
    board = generate_map(config.map, config.ports, seed=42)
    ships1 = place_initial_fleets(board, config, seed=42)
    ships2 = place_initial_fleets(board, config, seed=42)

    positions1 = sorted((s.owner, s.kind.value, s.position) for s in ships1.values())
    positions2 = sorted((s.owner, s.kind.value, s.position) for s in ships2.values())
    assert positions1 == positions2
