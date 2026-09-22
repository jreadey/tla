import dataclasses
from unittest.mock import patch

import pytest

from tla.board import Board
from tla.config import Config, MapConfig, PortConfig
from tla.hexgrid import AxialCoord, axial_to_offset, axial_to_pixel, distance, neighbors
from tla.mapgen import (
    _derive_seed,
    _map_is_playable,
    _ports_have_sea_room,
    filter_islet_contours,
    generate_map,
    has_fully_connected_sea,
    has_non_homotopic_sea_paths,
    has_sea_route_coverage,
    has_wide_enough_sea_passage,
    largest_sea_component,
)
from tla.tile import PLAYER_A, PLAYER_B, Tile, TerrainType


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
    # real room to work with, across several seeds for robustness. Tests
    # port-clustering fairness specifically, not map generation itself --
    # has_fully_connected_sea (requiring literally zero stray disconnected
    # ponds anywhere) can need many dozens of attempts by chance alone at
    # this map size, so max_generation_attempts is raised well past the
    # (smaller-map-tuned) default here rather than risk flakiness in an
    # unrelated test.
    map_config = MapConfig(width=40, height=24, noise_scale=8.0, max_generation_attempts=200)
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
    # Tests port/sea-room placement specifically, not map generation itself
    # -- see test_each_port_is_closer_to_friendly_ports_than_enemy_ports for
    # why max_generation_attempts is relaxed here rather than tuning the
    # default around the bare default Config's own (large, 4-port) map size.
    map_config = dataclasses.replace(config.map, max_generation_attempts=200)
    for seed in (533501, 1, 2, 3, 4, 5):
        board = generate_map(map_config, config.ports, seed=seed)
        main_sea = largest_sea_component(board)
        for player in (PLAYER_A, PLAYER_B):
            for port in board.ports_for(player):
                assert any(n in main_sea for n in neighbors(port)), (
                    f"seed={seed} port={port} does not border the main sea"
                )


_HEX_SIZE = 18.0


def _small_square_loop(center: tuple[float, float]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """A tiny closed 4-segment loop of points all well within one hex around
    `center`, matching what an isolated noise bump would draw."""
    cx, cy = center
    a, b, c, d = (cx - 0.5, cy - 0.5), (cx + 0.5, cy - 0.5), (cx + 0.5, cy + 0.5), (cx - 0.5, cy + 0.5)
    return [(a, b), (b, c), (c, d), (d, a)]


def _board_with(coord_terrains: dict[AxialCoord, TerrainType]) -> Board:
    board = Board(width=10, height=10, hex_pixel_size=_HEX_SIZE)
    for coord, terrain in coord_terrains.items():
        board.tiles[coord] = Tile(coord=coord, terrain=terrain)
    return board


def test_filter_islet_contours_drops_a_closed_loop_over_sea_only():
    sea_coord = AxialCoord(0, 0)
    board = _board_with({sea_coord: TerrainType.SEA})
    segments = _small_square_loop(axial_to_pixel(sea_coord, _HEX_SIZE))

    assert filter_islet_contours(segments, board) == []


def test_filter_islet_contours_keeps_a_closed_loop_touching_land():
    land_coord = AxialCoord(3, 3)
    board = _board_with({land_coord: TerrainType.LAND})
    segments = _small_square_loop(axial_to_pixel(land_coord, _HEX_SIZE))

    kept = filter_islet_contours(segments, board)

    assert sorted(kept) == sorted(segments)


def test_filter_islet_contours_always_keeps_an_open_path():
    # A path with an unshared (degree-1) endpoint represents a coastline
    # that reaches the raster's outer boundary and continues past the
    # generated bounds -- always real geography, regardless of the hexes
    # its points happen to fall in.
    sea_coord = AxialCoord(0, 0)
    board = _board_with({sea_coord: TerrainType.SEA})
    cx, cy = axial_to_pixel(sea_coord, _HEX_SIZE)
    p1, p2, p3 = (cx - 0.5, cy), (cx, cy), (cx + 0.5, cy)
    segments = [(p1, p2), (p2, p3)]

    assert filter_islet_contours(segments, board) == segments


def test_filter_islet_contours_treats_independent_loops_separately():
    sea_coord = AxialCoord(0, 0)
    land_coord = AxialCoord(3, 3)
    board = _board_with({sea_coord: TerrainType.SEA, land_coord: TerrainType.LAND})
    islet = _small_square_loop(axial_to_pixel(sea_coord, _HEX_SIZE))
    real_island = _small_square_loop(axial_to_pixel(land_coord, _HEX_SIZE))

    kept = filter_islet_contours(islet + real_island, board)

    assert sorted(kept) == sorted(real_island)


def test_port_tiles_are_occupiable_but_other_land_is_not():
    map_config, port_config = _small_configs()
    board = generate_map(map_config, port_config, seed=7)

    for coord, tile in board.tiles.items():
        if tile.terrain == TerrainType.LAND and not tile.is_port:
            assert not tile.occupiable
        if tile.is_port:
            assert tile.occupiable


# -- map playability rejection (see _map_is_playable) -----------------------


def _dumbbell_board(*, two_wide: bool) -> Board:
    """Two small sea "rooms" -- {(0,0),(0,1)} bordering Player A's port at
    (-1,1) (the one land hex adjacent to *both* of them), and {(2,0),(2,1)}
    bordering Player B's port at (3,0) (likewise adjacent to both) -- joined
    either by a single bridge hex (1,0) (a true one-hex chokepoint) or, if
    `two_wide`, also a second, parallel bridge hex (1,1) so the two rooms
    stay connected even with either bridge blocked."""
    coords = {
        AxialCoord(-1, 1): TerrainType.LAND,
        AxialCoord(3, 0): TerrainType.LAND,
        AxialCoord(0, 0): TerrainType.SEA,
        AxialCoord(0, 1): TerrainType.SEA,
        AxialCoord(1, 0): TerrainType.SEA,
        AxialCoord(2, 0): TerrainType.SEA,
        AxialCoord(2, 1): TerrainType.SEA,
    }
    if two_wide:
        coords[AxialCoord(1, 1)] = TerrainType.SEA
    board = _board_with(coords)
    board.tiles[AxialCoord(-1, 1)] = Tile(
        coord=AxialCoord(-1, 1), terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A
    )
    board.tiles[AxialCoord(3, 0)] = Tile(
        coord=AxialCoord(3, 0), terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B
    )
    return board


# -- fully connected sea (see has_fully_connected_sea) -----------------------


def test_has_fully_connected_sea_true_for_a_single_sea_component():
    assert has_fully_connected_sea(_dumbbell_board(two_wide=True)) is True


def test_has_fully_connected_sea_false_with_a_disconnected_pond():
    board = _dumbbell_board(two_wide=True)
    # Nowhere near the dumbbell's own sea -- a genuinely separate pond.
    board.tiles[AxialCoord(20, 20)] = Tile(coord=AxialCoord(20, 20), terrain=TerrainType.SEA)

    assert has_fully_connected_sea(board) is False


def test_has_wide_enough_sea_passage_rejects_a_single_hex_chokepoint():
    assert has_wide_enough_sea_passage(_dumbbell_board(two_wide=False)) is False


def test_has_wide_enough_sea_passage_accepts_a_two_hex_wide_corridor():
    assert has_wide_enough_sea_passage(_dumbbell_board(two_wide=True)) is True


def test_has_wide_enough_sea_passage_rejects_totally_disconnected_sides():
    board = _board_with(
        {
            AxialCoord(-1, 0): TerrainType.LAND,
            AxialCoord(0, 0): TerrainType.SEA,
            AxialCoord(5, 0): TerrainType.SEA,
            AxialCoord(6, 0): TerrainType.LAND,
        }
    )
    board.tiles[AxialCoord(-1, 0)] = Tile(
        coord=AxialCoord(-1, 0), terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A
    )
    board.tiles[AxialCoord(6, 0)] = Tile(
        coord=AxialCoord(6, 0), terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B
    )

    assert has_wide_enough_sea_passage(board) is False


# -- non-homotopic sea paths (see has_non_homotopic_sea_paths) --------------


def test_has_non_homotopic_sea_paths_true_for_a_true_island():
    # A land hex fully ringed by sea, with every one of its 6 neighbors
    # present on the board -- a genuine bounded hole to route around.
    island = AxialCoord(0, 0)
    board = _board_with({island: TerrainType.LAND, **{n: TerrainType.SEA for n in neighbors(island)}})

    assert has_non_homotopic_sea_paths(board) is True


def test_has_non_homotopic_sea_paths_false_with_no_land_at_all():
    board = _board_with({AxialCoord(0, 0): TerrainType.SEA, AxialCoord(1, 0): TerrainType.SEA})

    assert has_non_homotopic_sea_paths(board) is False


def test_has_non_homotopic_sea_paths_false_when_the_only_land_touches_the_board_edge():
    # Only 5 of the island's 6 neighbors are actually on the board -- the
    # missing 6th is exactly what a real generated map's edge looks like
    # (see _touches_board_edge), so this doesn't count as a bounded hole:
    # it reads as a peninsula reaching past the drawn map, not a closed
    # loop a route could wind around.
    island = AxialCoord(0, 0)
    board = _board_with(
        {island: TerrainType.LAND, **{n: TerrainType.SEA for n in neighbors(island)[:5]}}
    )

    assert has_non_homotopic_sea_paths(board) is False


def test_has_non_homotopic_sea_paths_false_when_the_island_is_not_in_the_main_sea():
    # The island here is ringed by its own tiny, disconnected pond, not
    # the larger sea a second, separate blob represents -- doesn't create
    # a hole a route between the two *ports'* sea actually has to route
    # around.
    island = AxialCoord(0, 0)
    coords = {island: TerrainType.LAND, **{n: TerrainType.SEA for n in neighbors(island)}}
    for i in range(10, 20):  # a much bigger, unrelated sea blob elsewhere
        coords[AxialCoord(i, 0)] = TerrainType.SEA

    assert has_non_homotopic_sea_paths(_board_with(coords)) is False


# -- sea route coverage (see has_sea_route_coverage) -------------------------


def _corridor_board(branch_length: int = 0) -> Board:
    """A straight 11-hex sea corridor from Player A's port at (-1,0) to
    Player B's port at (11,0), optionally with a perpendicular dead-end
    branch of `branch_length` extra sea hexes off its midpoint (5,0) --
    still part of the same connected sea, but progressively farther from
    the direct corridor route between the two ports."""
    coords: dict[AxialCoord, TerrainType] = {AxialCoord(q, 0): TerrainType.SEA for q in range(11)}
    for i in range(1, branch_length + 1):
        coords[AxialCoord(5, i)] = TerrainType.SEA
    board = _board_with(coords)
    board.tiles[AxialCoord(-1, 0)] = Tile(
        coord=AxialCoord(-1, 0), terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A
    )
    board.tiles[AxialCoord(11, 0)] = Tile(
        coord=AxialCoord(11, 0), terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B
    )
    return board


def test_has_sea_route_coverage_true_when_every_hex_is_on_the_route():
    board = _corridor_board()

    # _board_with (see _corridor_board) always builds a 10x10 Board
    # regardless of the coordinates actually used, so max_route_distance_
    # fraction here applies against that fixed 10, not the corridor's own
    # length -- 0.8 * 10 = 8.
    assert has_sea_route_coverage(board, MapConfig(max_route_distance_fraction=0.8)) is True


def test_has_sea_route_coverage_false_for_a_branch_beyond_max_route_distance():
    board = _corridor_board(branch_length=12)

    assert has_fully_connected_sea(board) is True  # the branch is still one connected sea
    # Same fixed-10 basis as above: 0.8 * 10 = 8 (too tight for the
    # branch's real length), 1.5 * 10 = 15 (comfortably covers it).
    assert has_sea_route_coverage(board, MapConfig(max_route_distance_fraction=0.8)) is False
    assert has_sea_route_coverage(board, MapConfig(max_route_distance_fraction=1.5)) is True


def test_has_sea_route_coverage_false_without_a_port_for_one_side():
    board = _corridor_board()
    board.tiles[AxialCoord(11, 0)] = Tile(coord=AxialCoord(11, 0), terrain=TerrainType.SEA)  # no Player B port

    assert has_sea_route_coverage(board, MapConfig()) is False


def test_ports_have_sea_room_rejects_a_port_with_only_one_sea_neighbor():
    board = _board_with({AxialCoord(0, 0): TerrainType.SEA})
    board.tiles[AxialCoord(-1, 0)] = Tile(
        coord=AxialCoord(-1, 0), terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A
    )

    assert _ports_have_sea_room(board, PortConfig(min_port_sea_neighbors=2)) is False


def test_ports_have_sea_room_accepts_a_port_with_enough_sea_neighbors():
    board = _board_with({AxialCoord(0, 0): TerrainType.SEA, AxialCoord(0, -1): TerrainType.SEA})
    board.tiles[AxialCoord(-1, 0)] = Tile(
        coord=AxialCoord(-1, 0), terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A
    )

    assert _ports_have_sea_room(board, PortConfig(min_port_sea_neighbors=2)) is True


def test_map_is_playable_requires_every_condition():
    # The two-wide dumbbell alone satisfies the sea-room/passage conditions
    # but not the island one -- add one, sharing hex (0,0) with the
    # existing sea so it's part of the same (and therefore still the
    # largest, and only) connected sea component rather than an ambiguous
    # separate one.
    playable = _dumbbell_board(two_wide=True)
    island = AxialCoord(0, -1)
    for coord in neighbors(island):
        if coord not in playable.tiles:
            playable.tiles[coord] = Tile(coord=coord, terrain=TerrainType.SEA)
    playable.tiles[island] = Tile(coord=island, terrain=TerrainType.LAND)

    chokepoint = _dumbbell_board(two_wide=False)
    lenient_map_config = MapConfig()
    lenient_config = PortConfig(min_port_sea_neighbors=2)

    assert has_non_homotopic_sea_paths(playable) is True  # confirms the setup above actually worked
    assert _map_is_playable(playable, lenient_map_config, lenient_config) is True
    # Fails on sea-passage and non-homotopic-paths both.
    assert _map_is_playable(chokepoint, lenient_map_config, lenient_config) is False


def test_derive_seed_is_deterministic():
    assert _derive_seed(42, 3) == _derive_seed(42, 3)


def test_derive_seed_does_not_collapse_nearby_base_seeds_to_the_same_sequence():
    # Regression: plain `base_seed + attempt` let two different starting
    # seeds converge on retrying through the exact same later seed (e.g.
    # seed=1's attempt=1 and seed=2's attempt=0 both landing on "2"),
    # which made generate_map(seed=1) and generate_map(seed=2) produce an
    # identical accepted map once rejection-retry existed.
    seed1_attempts = {_derive_seed(1, a) for a in range(5)}
    seed2_attempts = {_derive_seed(2, a) for a in range(5)}
    assert seed1_attempts.isdisjoint(seed2_attempts)


def test_generate_map_retries_and_eventually_raises_if_never_playable():
    map_config, port_config = _small_configs()
    map_config = dataclasses.replace(map_config, max_generation_attempts=3)

    with patch("tla.mapgen._map_is_playable", return_value=False) as mocked:
        with pytest.raises(RuntimeError):
            generate_map(map_config, port_config, seed=1)

    assert mocked.call_count == 3
