"""Elevation-driven map generation.

Builds a fine elevation raster (see tla.elevation), then classifies each hex
by the fraction of its raster samples above sea level: more than
`land_area_threshold` -> LAND, otherwise -> SEA. There is no separate shore
terrain; ports are placed on coastal LAND hexes (see `_place_ports`). Pure
function of config + seed, no reliance on global RNG state, so results are
deterministic and easy to unit test.
"""

from __future__ import annotations

import random

from tla.board import Board
from tla.config import MapConfig, PortConfig
from tla.elevation import ElevationGrid, Point, Segment, build_elevation_grid
from tla.hexgrid import AxialCoord, axial_to_pixel, distance, neighbors, offset_to_axial, pixel_to_axial
from tla.tile import PLAYER_A, PLAYER_B, Tile, TerrainType


def generate_map(
    map_config: MapConfig, port_config: PortConfig, seed: int | None = None
) -> Board:
    """Generates a candidate map + port placement, rejecting and retrying
    (with a different derived seed each attempt -- `_derive_seed(base_seed,
    attempt)`, so the whole sequence stays reproducible from `base_seed`
    alone) any candidate that fails any of the playability conditions
    checked by `_map_is_playable`: every sea hex belongs to one single
    connected body of water, every port needs room to maneuver, the two
    sides' territories need at least one route between them that a single
    blockading ship can't fully choke off, there needs to be a real island
    somewhere in the sea so at least one such route has a genuinely
    distinct alternative (not just one shape every path is stuck inside),
    and no sea hex may sit too far from either of those two routes -- see
    `has_fully_connected_sea`/`_ports_have_sea_room`/`has_wide_enough_sea_
    passage`/`has_non_homotopic_sea_paths`/`has_sea_route_coverage`.
    Raises `RuntimeError` after `MapConfig.max_generation_attempts` straight
    rejections -- the same "give up and tell the caller to adjust config"
    contract `_place_ports` already uses for too few coastal tiles."""
    if seed is None:
        seed = map_config.seed if map_config.seed is not None else random.randrange(1_000_000)
    base_seed = seed
    for attempt in range(map_config.max_generation_attempts):
        board = _generate_map_once(map_config, port_config, _derive_seed(base_seed, attempt))
        if _map_is_playable(board, map_config, port_config):
            return board
    max_route_distance = map_config.max_route_distance_fraction * max(map_config.width, map_config.height)
    raise RuntimeError(
        f"No playable map found in {map_config.max_generation_attempts} attempts from seed "
        f"{base_seed} -- the sea needs to be a single connected body of water, every port needs "
        f">= {port_config.min_port_sea_neighbors} sea-hex neighbors, the two sides need at least "
        "one route a single blockading ship can't fully cut, the sea needs at least one real "
        f"island for routes to actually differ around, and no sea hex may sit more than "
        f"{max_route_distance:.0f} hexes ({map_config.max_route_distance_fraction} of the longer map "
        "dimension) from either resulting route. Adjust map size/shape config, "
        "min_port_sea_neighbors, max_route_distance_fraction, or max_generation_attempts."
    )


def _derive_seed(base_seed: int, attempt: int) -> int:
    """A well-scattered per-attempt seed derived from `base_seed` -- deter-
    ministic (same inputs always give the same output), but nearby base
    seeds (e.g. two callers using seed=1 and seed=2) don't retry through
    near-identical sequences the way plain `base_seed + attempt` would,
    which could otherwise make two *different* starting seeds converge on
    generating the exact same accepted map. Knuth's multiplicative-hash
    constant mixes the two inputs across a full 32-bit range."""
    return (base_seed * 2_654_435_761 + attempt * 40_503 + 1) & 0xFFFFFFFF


def _generate_map_once(map_config: MapConfig, port_config: PortConfig, seed: int) -> Board:
    noise_base = seed % 256
    hex_size = map_config.hex_pixel_size

    coords = [
        offset_to_axial(col, row)
        for col in range(map_config.width)
        for row in range(map_config.height)
    ]
    coord_set = set(coords)

    pixels = [axial_to_pixel(c, hex_size) for c in coords]
    pad = hex_size
    bounds = (
        min(x for x, _ in pixels) - pad,
        min(y for _, y in pixels) - pad,
        max(x for x, _ in pixels) + pad,
        max(y for _, y in pixels) + pad,
    )

    elevation = build_elevation_grid(
        bounds=bounds,
        hex_pixel_size=hex_size,
        supersample=map_config.elevation_supersample,
        noise_scale=map_config.noise_scale,
        octaves=map_config.octaves,
        sea_level=map_config.sea_level,
        noise_base=noise_base,
    )

    hex_samples: dict[AxialCoord, list[float]] = {}
    for row_idx in range(elevation.rows):
        y = elevation.origin_y + row_idx * elevation.cell_size
        for col_idx in range(elevation.cols):
            x = elevation.origin_x + col_idx * elevation.cell_size
            coord = pixel_to_axial(x, y, hex_size)
            if coord not in coord_set:
                continue
            hex_samples.setdefault(coord, []).append(elevation.values[row_idx][col_idx])

    board = Board(
        width=map_config.width,
        height=map_config.height,
        hex_pixel_size=hex_size,
        elevation=elevation,
    )
    for coord in coords:
        values = hex_samples.get(coord)
        if not values:
            # Rare: a hex too small to catch any raster sample point. Fall
            # back to the continuous value at its own center.
            x, y = axial_to_pixel(coord, hex_size)
            values = [elevation.sample(x, y)]
        board.tiles[coord] = Tile(
            coord=coord, terrain=_classify(values, map_config.land_area_threshold)
        )

    _place_ports(board, port_config, seed)
    return board


def _classify(values: list[float], land_area_threshold: float) -> TerrainType:
    land_fraction = sum(1 for v in values if v > 0) / len(values)
    return TerrainType.LAND if land_fraction > land_area_threshold else TerrainType.SEA


def filter_islet_contours(segments: list[Segment], board: Board) -> list[Segment]:
    """Drop closed coastline loops that don't correspond to any actual LAND
    hex -- pure visual clutter, not real geography.

    The elevation raster (see tla.elevation) is sampled far finer than a
    hex, and a hex only classifies as LAND once more than
    `MapConfig.land_area_threshold` of its own samples are above sea
    level (see `_classify`). An isolated noise bump can cross sea level
    locally -- drawing a tiny closed contour ring in open water -- without
    ever pushing any hex there over that threshold. Real coastline is
    unaffected: an open path (one that reaches the raster's outer
    boundary, meaning the landmass continues past the generated bounds)
    is always kept, and a genuine island's loop reliably touches at least
    one LAND hex somewhere along its length even if a threshold-adjacent
    edge hex or two along the same loop doesn't.
    """
    if not segments:
        return segments
    hex_size = board.hex_pixel_size

    adjacency: dict[Point, list[int]] = {}
    for i, (p1, p2) in enumerate(segments):
        adjacency.setdefault(p1, []).append(i)
        adjacency.setdefault(p2, []).append(i)

    visited: set[int] = set()
    kept: list[Segment] = []

    for start in range(len(segments)):
        if start in visited:
            continue
        component: set[int] = set()
        stack = [start]
        while stack:
            idx = stack.pop()
            if idx in component:
                continue
            component.add(idx)
            for point in segments[idx]:
                component_neighbors = adjacency[point]
                stack.extend(other for other in component_neighbors if other not in component)
        visited |= component

        points = [p for idx in component for p in segments[idx]]
        is_closed_loop = all(len(adjacency[p]) == 2 for p in points)
        touches_land = any(
            (tile := board.tiles.get(pixel_to_axial(x, y, hex_size))) is not None
            and tile.terrain == TerrainType.LAND
            for x, y in points
        )
        if not is_closed_loop or touches_land:
            kept.extend(segments[idx] for idx in component)

    return kept


def largest_sea_component(board: Board) -> set[AxialCoord]:
    """The biggest connected body of SEA tiles (flood fill over sea-sea
    adjacency). Small enclosed ponds end up as separate, smaller components,
    so ports can be required to border this one instead -- otherwise a port
    could open onto a landlocked puddle with no way for ships to reach the
    open ocean, or for the enemy to ever besiege it. Also used by
    tla.fleet_setup so starting ships never land in one of those ponds
    either (see _pick_start_hex)."""
    sea_tiles = {c for c, t in board.tiles.items() if t.terrain == TerrainType.SEA}
    seen: set[AxialCoord] = set()
    largest: set[AxialCoord] = set()
    for start in sea_tiles:
        if start in seen:
            continue
        component: set[AxialCoord] = set()
        stack = [start]
        while stack:
            coord = stack.pop()
            if coord in component:
                continue
            component.add(coord)
            for n in neighbors(coord):
                if n in sea_tiles and n not in component:
                    stack.append(n)
        seen |= component
        if len(component) > len(largest):
            largest = component
    return largest


def _place_ports(board: Board, port_config: PortConfig, seed: int) -> None:
    rng = random.Random(seed)
    main_sea = largest_sea_component(board)

    def is_coastal_land(coord: AxialCoord) -> bool:
        tile = board.tiles[coord]
        if tile.terrain != TerrainType.LAND:
            return False
        return any(n in main_sea for n in neighbors(coord))

    coastal_coords = [c for c in board.tiles if is_coastal_land(c)]
    needed_per_player = port_config.ports_per_player

    if len(coastal_coords) < needed_per_player * 2:
        raise RuntimeError(
            f"Only {len(coastal_coords)} coastal land tiles available, need "
            f"{needed_per_player * 2} for {needed_per_player} ports/player. Adjust "
            "map size or sea_level/land_area_threshold."
        )

    # Two seed hexes as far apart as possible, so each player's ports cluster
    # near their own seed and away from the other player's -- this is what
    # keeps a port's average distance to friendly ports below its average
    # distance to enemy ports.
    shuffled = coastal_coords[:]
    rng.shuffle(shuffled)
    seed_a = shuffled[0]
    seed_b = max(coastal_coords, key=lambda c: distance(c, seed_a))

    cluster1 = _cluster_near(seed_a, coastal_coords, needed_per_player, port_config.min_port_spacing)
    remaining = [c for c in coastal_coords if c not in cluster1]
    cluster2 = _cluster_near(seed_b, remaining, needed_per_player, port_config.min_port_spacing)

    # Player A is on the west (smaller q) side of the map, Player B on the
    # east -- a stable, predictable left/right layout rather than whichever
    # cluster happened to get the randomly-picked first seed.
    if sum(c.q for c in cluster1) <= sum(c.q for c in cluster2):
        ports_a, ports_b = cluster1, cluster2
    else:
        ports_a, ports_b = cluster2, cluster1

    for coord in ports_a:
        tile = board.tiles[coord]
        tile.is_port = True
        tile.port_owner = PLAYER_A
    for coord in ports_b:
        tile = board.tiles[coord]
        tile.is_port = True
        tile.port_owner = PLAYER_B


def _cluster_near(
    anchor: AxialCoord, pool: list[AxialCoord], count: int, spacing: int
) -> list[AxialCoord]:
    """Pick `count` hexes from `pool`, closest to `anchor` first, spread apart
    by at least `spacing` where the pool allows it."""
    by_distance = sorted(pool, key=lambda c: distance(c, anchor))
    chosen: list[AxialCoord] = []
    for coord in by_distance:
        if len(chosen) >= count:
            break
        if all(distance(coord, other) >= spacing for other in chosen):
            chosen.append(coord)
    if len(chosen) < count:
        remaining = [c for c in by_distance if c not in chosen]
        chosen.extend(remaining[: count - len(chosen)])
    return chosen


def _map_is_playable(board: Board, map_config: MapConfig, port_config: PortConfig) -> bool:
    """The playability conditions a candidate map/port-placement must all
    pass, user-specified directly: (1) the sea is a single connected body
    of water, not several disconnected ones, (2) every port has room to
    maneuver, not just a single blockadable doorway, (3) the two sides
    aren't separated only by chokepoints a single ship can plug, (4)
    there's a real choice of route between the sides, not just one region
    of open water with a single topological "shape" every route is stuck
    inside, and (5) no sea hex sits too far from wherever that choice of
    route actually goes -- see `has_fully_connected_sea`/`_ports_have_sea_
    room`/`has_wide_enough_sea_passage`/`has_non_homotopic_sea_paths`/
    `has_sea_route_coverage`. Ordered roughly cheapest-first so a
    thoroughly broken candidate is rejected before paying for the pricier
    checks later in the chain."""
    return (
        has_fully_connected_sea(board)
        and _ports_have_sea_room(board, port_config)
        and has_wide_enough_sea_passage(board)
        and has_non_homotopic_sea_paths(board)
        and has_sea_route_coverage(board, map_config)
    )


def has_fully_connected_sea(board: Board) -> bool:
    """Whether every SEA hex on `board` belongs to one single connected
    body of water -- no isolated pond anywhere on the map, not just away
    from a port's own doorstep (`_ports_have_sea_room` and every other
    check here already only care about `largest_sea_component`, so a small
    stray pond elsewhere would otherwise pass every other condition
    silently). An empty sea counts as trivially connected -- nothing to be
    disconnected from."""
    sea_tiles = {c for c, t in board.tiles.items() if t.terrain == TerrainType.SEA}
    return sea_tiles == largest_sea_component(board)


def _ports_have_sea_room(board: Board, port_config: PortConfig) -> bool:
    """Every port on `board` borders at least `PortConfig.min_port_sea_
    neighbors` hexes of the main sea component -- fewer would mean a
    single blockading ship could pin the port shut right at its own
    doorstep, with no other way in or out at all. Only counts neighbors
    in `largest_sea_component` (not a tiny, disconnected pond next door --
    that wouldn't actually give the port anywhere useful to go)."""
    main_sea = largest_sea_component(board)
    ports = board.ports_for(PLAYER_A) + board.ports_for(PLAYER_B)
    return all(
        sum(1 for n in neighbors(port) if n in main_sea) >= port_config.min_port_sea_neighbors
        for port in ports
    )


def has_wide_enough_sea_passage(board: Board) -> bool:
    """Whether Player A's ports stay sea-connected to Player B's ports no
    matter which single sea hex is removed -- i.e. there's always some
    surviving route between the two sides, so one blockading ship can
    never fully cut naval movement between them. Equivalent to asking
    whether the two sides are 2-vertex-connected in the sea-hex adjacency
    graph; checked directly (try removing each sea hex in turn and see if
    a source->sink route still exists for every one of them) rather than
    via a dedicated min-cut algorithm, since the small board sizes here
    make the brute-force version plenty fast and far simpler to get right.

    False (map should be rejected) if the two sides aren't even connected
    at all with nothing removed, which is the same failure by definition."""
    main_sea = largest_sea_component(board)
    ports_a = board.ports_for(PLAYER_A)
    ports_b = board.ports_for(PLAYER_B)
    sources = {n for p in ports_a for n in neighbors(p) if n in main_sea}
    sinks = {n for p in ports_b for n in neighbors(p) if n in main_sea}
    if not sources or not sinks:
        return False
    if not _sea_connected(main_sea, sources, sinks):
        return False
    return all(
        _sea_connected(main_sea - {blocked}, sources - {blocked}, sinks - {blocked})
        for blocked in main_sea
    )


def _sea_connected(sea: set[AxialCoord], sources: set[AxialCoord], sinks: set[AxialCoord]) -> bool:
    """Plain BFS reachability from any hex in `sources` to any hex in
    `sinks`, moving only through hexes in `sea`."""
    frontier = list(sources & sea)
    visited = set(frontier)
    while frontier:
        coord = frontier.pop()
        if coord in sinks:
            return True
        for n in neighbors(coord):
            if n in sea and n not in visited:
                visited.add(n)
                frontier.append(n)
    return bool(visited & sinks)


def has_non_homotopic_sea_paths(board: Board) -> bool:
    """Whether the main sea has at least one island in it -- a land
    component fully enclosed by sea, not touching the map's outer edge --
    so a route between any two ports has a genuinely different
    alternative: going around such an island one way can never be
    continuously slid into going around it the other way (they wind
    around the hole differently), the way any two routes through a
    hole-free ("simply connected") sea always can be. With at least one
    such hole, *every* pair of points in the same connected sea already
    has infinitely many pairwise non-homotopic routes between them, so
    this only needs checking once for the whole map, not per port pair --
    a landmass that instead touches the map's edge doesn't count: treat
    the finite generated map as a window onto an unbounded ocean, and a
    landmass reaching that edge is a peninsula/mainland extending past
    what's drawn, not a closed loop a route can wind around."""
    main_sea = largest_sea_component(board)
    for component in _land_components(board):
        if _touches_board_edge(component, board):
            continue
        if any(n in main_sea for coord in component for n in neighbors(coord)):
            return True
    return False


def _land_components(board: Board) -> list[set[AxialCoord]]:
    """Every maximal connected group of LAND hexes on `board` (flood fill
    over land-land adjacency -- the land-side mirror of
    `largest_sea_component`'s sea-side one)."""
    land_tiles = {c for c, t in board.tiles.items() if t.terrain == TerrainType.LAND}
    seen: set[AxialCoord] = set()
    components: list[set[AxialCoord]] = []
    for start in land_tiles:
        if start in seen:
            continue
        component: set[AxialCoord] = set()
        stack = [start]
        while stack:
            coord = stack.pop()
            if coord in component:
                continue
            component.add(coord)
            for n in neighbors(coord):
                if n in land_tiles and n not in component:
                    stack.append(n)
        seen |= component
        components.append(component)
    return components


def _touches_board_edge(component: set[AxialCoord], board: Board) -> bool:
    """Whether any hex in `component` has a neighbor coordinate that
    isn't part of `board` at all -- i.e. the component reaches the edge
    of the generated area, regardless of the board's exact offset-
    coordinate shape (simpler and more robust than comparing against
    `board.width`/`height` directly)."""
    return any(n not in board.tiles for coord in component for n in neighbors(coord))


def has_sea_route_coverage(board: Board, map_config: MapConfig) -> bool:
    """Whether every hex in `largest_sea_component(board)` is within
    `MapConfig.max_route_distance_fraction * max(board.width, board.
    height)` real sea-route hexes of at least one of the (up to two)
    shortest routes between the two sides' ports -- the single shortest
    route, plus a second one forced around whichever island actually
    gives the map its route diversity (see `has_non_homotopic_sea_paths`),
    found by re-running the same search with the first route's own
    interior blocked. Without this, a map can pass every other
    playability condition while still burying a large stretch of open
    ocean nowhere near anywhere the two sides could plausibly ever fight
    -- user's own diagnosis of a real generated map (see the project's
    own map-playability memory). A fraction of `board`'s own size rather
    than a flat hex count, so this scales automatically instead of
    needing its own per-map-size retuning -- see `MapConfig.max_route_
    distance_fraction`. Deliberately reads size off `board` itself, not
    `map_config` (in the real `generate_map` pipeline the two always
    agree, since `board` is built directly from `map_config`, but a
    hand-built `board` passed here standalone -- e.g. in tests -- might
    not otherwise match whatever unrelated `map_config` happens to be
    passed alongside it).

    False outright if either side has no port bordering the main sea at
    all (nothing to route between -- `_ports_have_sea_room`/`has_wide_
    enough_sea_passage` already guard the real pipeline against this, but
    this is independently callable and testable). Only the second route
    can fail to exist (no real island, or its far side is otherwise
    unreachable without the first route's own hexes) -- when that happens,
    coverage is judged by the first route alone rather than rejecting the
    map here too for a shortfall `has_non_homotopic_sea_paths` already
    owns catching."""
    main_sea = largest_sea_component(board)
    sources = {n for p in board.ports_for(PLAYER_A) for n in neighbors(p) if n in main_sea}
    sinks = {n for p in board.ports_for(PLAYER_B) for n in neighbors(p) if n in main_sea}
    if not sources or not sinks:
        return False
    route1 = _shortest_sea_route(main_sea, sources, sinks)
    if route1 is None:
        return False
    route2 = _shortest_sea_route(main_sea, sources, sinks, avoid=frozenset(route1) - sources - sinks)
    route_hexes = set(route1) | (set(route2) if route2 is not None else set())

    max_distance = map_config.max_route_distance_fraction * max(board.width, board.height)
    distance_from_route = _multi_source_sea_distance(main_sea, route_hexes)
    return all(distance_from_route[coord] <= max_distance for coord in main_sea)


def _shortest_sea_route(
    sea: set[AxialCoord],
    sources: set[AxialCoord],
    sinks: set[AxialCoord],
    avoid: frozenset[AxialCoord] = frozenset(),
) -> list[AxialCoord] | None:
    """Shortest hex path (BFS, unweighted) from any hex in `sources` to any
    hex in `sinks`, moving only through `sea` hexes not in `avoid` --
    `sources`/`sinks` are always usable regardless of `avoid`, so blocking
    a previously-found route's own interior (see `has_sea_route_coverage`,
    forcing a second, differently-routed search) can never also block the
    very ports that search still has to start and end at. None if no such
    path exists."""
    usable = (sea - avoid) | sources | sinks
    parents: dict[AxialCoord, AxialCoord] = {}
    frontier = list(sources & usable)
    visited = set(frontier)
    for start in frontier:
        if start in sinks:
            return [start]
    while frontier:
        next_frontier: list[AxialCoord] = []
        for coord in frontier:
            for n in neighbors(coord):
                if n not in usable or n in visited:
                    continue
                visited.add(n)
                parents[n] = coord
                if n in sinks:
                    path = [n]
                    while path[-1] in parents:
                        path.append(parents[path[-1]])
                    path.reverse()
                    return path
                next_frontier.append(n)
        frontier = next_frontier
    return None


def _multi_source_sea_distance(sea: set[AxialCoord], sources: set[AxialCoord]) -> dict[AxialCoord, int]:
    """BFS distance (hex steps, moving only through `sea`) from the
    nearest hex in `sources` to every hex in `sea` -- every hex in a
    connected `sea` is reachable from any non-empty `sources` subset of it,
    so (unlike `tla.ai.task_force.sea_distance_field`, which has to handle
    a source outside the sea or a target sea disconnected from it) this
    never needs a fallback for an unreachable hex."""
    dist: dict[AxialCoord, int] = {s: 0 for s in sources}
    frontier = list(dist)
    while frontier:
        next_frontier: list[AxialCoord] = []
        for coord in frontier:
            for n in neighbors(coord):
                if n in sea and n not in dist:
                    dist[n] = dist[coord] + 1
                    next_frontier.append(n)
        frontier = next_frontier
    return dist
