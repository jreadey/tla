from tla.config import MapConfig, PortConfig
from tla.hexgrid import axial_to_offset, neighbors
from tla.mapgen import generate_map
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


def test_port_tiles_are_occupiable_but_other_land_is_not():
    map_config, port_config = _small_configs()
    board = generate_map(map_config, port_config, seed=7)

    for coord, tile in board.tiles.items():
        if tile.terrain == TerrainType.LAND and not tile.is_port:
            assert not tile.occupiable
        if tile.is_port:
            assert tile.occupiable
