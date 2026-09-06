from tla.config import Config, MapConfig, PortConfig
from tla.hexgrid import axial_to_offset, distance, neighbors
from tla.mapgen import _largest_sea_component, generate_map
from tla.tile import PLAYER_A, PLAYER_B, TerrainType


def _small_configs(seed_map_seed=None):
    map_config = MapConfig(width=16, height=10, noise_scale=5.0, seed=seed_map_seed)
    port_config = PortConfig(ports_per_player=2, min_port_spacing=2)
    return map_config, port_config


def test_generated_board_has_correct_dimensions():
    map_config, port_config = _small_configs()
    board = generate_map(map_config, port_config, seed=1)
    assert len(board.tiles) == map_config.width * map_config.height
    for coord in board.tiles:
        col, row = axial_to_offset(coord)
        assert 0 <= col < map_config.width
        assert 0 <= row < map_config.height


def test_generation_is_deterministic_given_seed():
    map_config, port_config = _small_configs()
    board1 = generate_map(map_config, port_config, seed=99)
    board2 = generate_map(map_config, port_config, seed=99)
    assert {c: t.terrain for c, t in board1.tiles.items()} == {
        c: t.terrain for c, t in board2.tiles.items()
    }
    assert board1.ports_for(PLAYER_A) == board2.ports_for(PLAYER_A)
    assert board1.ports_for(PLAYER_B) == board2.ports_for(PLAYER_B)


def test_different_seeds_can_produce_different_maps():
    map_config, port_config = _small_configs()
    board1 = generate_map(map_config, port_config, seed=1)
    board2 = generate_map(map_config, port_config, seed=2)
    terrains1 = {c: t.terrain for c, t in board1.tiles.items()}
    terrains2 = {c: t.terrain for c, t in board2.tiles.items()}
    assert terrains1 != terrains2


def test_ports_are_on_coastal_land_and_split_evenly():
    map_config, port_config = _small_configs()
    board = generate_map(map_config, port_config, seed=7)

    ports_a = board.ports_for(PLAYER_A)
    ports_b = board.ports_for(PLAYER_B)
    assert len(ports_a) == port_config.ports_per_player
    assert len(ports_b) == port_config.ports_per_player
    assert set(ports_a).isdisjoint(ports_b)

    for coord in ports_a + ports_b:
        tile = board.tiles[coord]
        assert tile.terrain == TerrainType.LAND
        assert tile.is_port
        assert any(
            board.tiles[n].terrain == TerrainType.SEA
            for n in neighbors(coord)
            if n in board.tiles
        )


def test_each_port_is_closer_to_friendly_ports_than_enemy_ports():
    # Use a bigger map so there's enough coastline for the clustering to have
    # real room to work with, across several seeds for robustness.
    map_config = MapConfig(width=40, height=24, noise_scale=8.0)
    port_config = PortConfig(ports_per_player=4, min_port_spacing=2)

    for seed in range(10):
        board = generate_map(map_config, port_config, seed=seed)
        ports_a = board.ports_for(PLAYER_A)
        ports_b = board.ports_for(PLAYER_B)

        for owner_ports, other_ports in ((ports_a, ports_b), (ports_b, ports_a)):
            for port in owner_ports:
                other_friendly = [p for p in owner_ports if p != port]
                avg_friendly = sum(distance(port, p) for p in other_friendly) / len(other_friendly)
                avg_enemy = sum(distance(port, p) for p in other_ports) / len(other_ports)
                assert avg_friendly < avg_enemy, f"seed={seed} port={port}"


def test_ports_border_the_main_sea_not_an_isolated_pond():
    # Regression: seed 533501 on the default config generated several tiny
    # landlocked ponds (disconnected from the main ocean), and a couple of
    # ports ended up bordering only those -- ships there could never reach
    # the open sea, and the enemy could never besiege them.
    config = Config()
    for seed in (533501, 1, 2, 3, 4, 5):
        board = generate_map(config.map, config.ports, seed=seed)
        main_sea = _largest_sea_component(board)
        for player in (PLAYER_A, PLAYER_B):
            for port in board.ports_for(player):
                assert any(n in main_sea for n in neighbors(port)), (
                    f"seed={seed} port={port} does not border the main sea"
                )


def test_port_tiles_are_occupiable_but_other_land_is_not():
    map_config, port_config = _small_configs()
    board = generate_map(map_config, port_config, seed=7)

    for coord, tile in board.tiles.items():
        if tile.terrain == TerrainType.LAND and not tile.is_port:
            assert not tile.occupiable
        if tile.is_port:
            assert tile.occupiable
