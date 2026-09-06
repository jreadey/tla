from tla.hexgrid import (
    AxialCoord,
    axial_to_offset,
    axial_to_pixel,
    distance,
    hexes_in_range,
    neighbors,
    offset_to_axial,
    pixel_to_axial,
)


def test_neighbors_count_and_distance():
    center = AxialCoord(3, -2)
    ns = neighbors(center)
    assert len(ns) == 6
    assert len(set(ns)) == 6
    for n in ns:
        assert distance(center, n) == 1


def test_distance_self_is_zero():
    a = AxialCoord(5, 5)
    assert distance(a, a) == 0


def test_distance_symmetric():
    a = AxialCoord(0, 0)
    b = AxialCoord(3, -1)
    assert distance(a, b) == distance(b, a)


def test_hexes_in_range_radius_zero_is_just_center():
    center = AxialCoord(1, 1)
    assert hexes_in_range(center, 0) == {center}


def test_hexes_in_range_matches_distance_definition():
    center = AxialCoord(0, 0)
    radius = 2
    in_range = hexes_in_range(center, radius)
    for coord in in_range:
        assert distance(center, coord) <= radius
    # Every neighbor-of-neighbor within radius must be included.
    for n in neighbors(center):
        for n2 in neighbors(n):
            if distance(center, n2) <= radius:
                assert n2 in in_range


def test_offset_axial_roundtrip():
    for col in range(-5, 6):
        for row in range(-5, 6):
            coord = offset_to_axial(col, row)
            assert axial_to_offset(coord) == (col, row)


def test_pixel_round_trip_recovers_exact_hex_center():
    hex_size = 18.0
    for q in range(-6, 7):
        for r in range(-6, 7):
            coord = AxialCoord(q, r)
            x, y = axial_to_pixel(coord, hex_size)
            assert pixel_to_axial(x, y, hex_size) == coord


def test_pixel_to_axial_nearby_point_still_resolves_to_same_hex():
    hex_size = 18.0
    coord = AxialCoord(3, -2)
    x, y = axial_to_pixel(coord, hex_size)
    # A small nudge (well inside the hex) should not cross into a neighbor.
    assert pixel_to_axial(x + 2.0, y + 1.0, hex_size) == coord
