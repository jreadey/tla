from tla.battle import resolve_round, run_battle
from tla.board import Board
from tla.config import Config
from tla.game_state import GameState
from tla.hexgrid import AxialCoord, hexes_in_range
from tla.ship import Ship, ShipKind
from tla.tile import PLAYER_A, PLAYER_B, Tile, TerrainType


def _sea_board(radius: int = 5) -> Board:
    board = Board(width=radius * 2 + 1, height=radius * 2 + 1)
    for coord in hexes_in_range(AxialCoord(0, 0), radius):
        board.tiles[coord] = Tile(coord=coord, terrain=TerrainType.SEA)
    return board


def _ship(
    coord: AxialCoord,
    kind: ShipKind,
    owner,
    ship_id: int,
    hp: int | None = None,
    surfaced: bool = True,
) -> Ship:
    stats = Config().ship_stats.stats[kind]
    return Ship(
        id=ship_id,
        kind=kind,
        owner=owner,
        position=coord,
        current_hp=hp if hp is not None else stats.hp,
        surfaced=surfaced,
    )


def _game_state(board: Board, ships: list[Ship]) -> GameState:
    return GameState(config=Config(), board=board, ships={s.id: s for s in ships})


def test_resolve_round_applies_simultaneous_damage():
    board = _sea_board()
    attacker = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)  # damage 2
    defender = _ship(AxialCoord(1, 0), ShipKind.CRUISER, PLAYER_B, 2)  # damage 4
    gs = _game_state(board, [attacker, defender])

    result = resolve_round(attacker, defender, gs)

    assert result.damage_to_defender == 2
    assert result.damage_to_attacker == 4
    assert defender.current_hp == Config().ship_stats.stats[ShipKind.CRUISER].hp - 2
    assert attacker.current_hp == Config().ship_stats.stats[ShipKind.DESTROYER].hp - 4
    assert not result.attacker_sunk
    assert not result.defender_sunk


def test_resolve_round_floors_hp_at_zero_and_flags_sunk():
    board = _sea_board()
    attacker = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)  # damage 4
    defender = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2, hp=2)
    gs = _game_state(board, [attacker, defender])

    result = resolve_round(attacker, defender, gs)

    assert defender.current_hp == 0
    assert result.defender_sunk is True


def test_aws_zero_attacker_cannot_damage_a_submerged_submarine():
    # The explicit scenario from the design spec: a battleship (asw=0) is
    # completely unable to hurt a submerged submarine, while the submarine's
    # own damage still applies normally against the battleship.
    board = _sea_board()
    battleship = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    sub = _ship(AxialCoord(1, 0), ShipKind.SUBMARINE, PLAYER_B, 2, surfaced=False)
    gs = _game_state(board, [battleship, sub])

    result = resolve_round(battleship, sub, gs)

    assert result.damage_to_defender == 0  # sub takes no damage while submerged
    assert result.damage_to_attacker == Config().ship_stats.stats[ShipKind.SUBMARINE].damage
    assert sub.current_hp == Config().ship_stats.stats[ShipKind.SUBMARINE].hp


def test_battleships_are_wiped_out_by_submerged_submarines_over_repeated_rounds():
    board = _sea_board()
    battleship = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)  # hp 12
    sub = _ship(AxialCoord(1, 0), ShipKind.SUBMARINE, PLAYER_B, 2, surfaced=False)  # dmg 4
    gs = _game_state(board, [battleship, sub])

    result = run_battle(battleship, sub, gs, decision_fn=lambda *_: "stay")

    assert result.attacker_sunk is True  # the battleship
    assert result.defender_sunk is False  # the sub never took a scratch
    assert sub.current_hp == Config().ship_stats.stats[ShipKind.SUBMARINE].hp
    assert len(result.rounds) == 3  # 12 hp / 4 damage per round


def test_asw_capable_attacker_can_damage_a_submerged_submarine():
    board = _sea_board()
    cruiser = _ship(AxialCoord(0, 0), ShipKind.CRUISER, PLAYER_A, 1)  # asw 2
    sub = _ship(AxialCoord(1, 0), ShipKind.SUBMARINE, PLAYER_B, 2, surfaced=False)
    gs = _game_state(board, [cruiser, sub])

    result = resolve_round(cruiser, sub, gs)

    assert result.damage_to_defender == Config().ship_stats.stats[ShipKind.CRUISER].asw


def test_carrier_bonus_applies_per_side_independently():
    board = _sea_board()
    attacker = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    defender = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_B, 2)
    attacker_carrier = _ship(AxialCoord(0, 1), ShipKind.CARRIER, PLAYER_A, 3)
    gs = _game_state(board, [attacker, defender, attacker_carrier])

    result = resolve_round(attacker, defender, gs)

    base_damage = Config().ship_stats.stats[ShipKind.DESTROYER].damage
    assert result.damage_to_defender == base_damage + 1  # one friendly carrier in range
    assert result.damage_to_attacker == base_damage  # no carrier on the defender's side


def test_carrier_bonus_is_recomputed_each_round_not_cached():
    board = _sea_board()
    attacker = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1, hp=100)
    defender = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_B, 2, hp=100)
    carrier = _ship(AxialCoord(0, 1), ShipKind.CARRIER, PLAYER_A, 3)
    gs = _game_state(board, [attacker, defender, carrier])
    base_damage = Config().ship_stats.stats[ShipKind.DESTROYER].damage

    round1 = resolve_round(attacker, defender, gs)
    assert round1.damage_to_defender == base_damage + 1

    # The carrier retreats out of range before the next round.
    carrier.position = AxialCoord(5, 5)
    round2 = resolve_round(attacker, defender, gs)
    assert round2.damage_to_defender == base_damage


def test_run_battle_stops_on_retreat_leaving_both_ships_alive():
    board = _sea_board()
    attacker = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1, hp=100)
    defender = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_B, 2, hp=100)
    gs = _game_state(board, [attacker, defender])

    result = run_battle(attacker, defender, gs, decision_fn=lambda *_: "retreat")

    assert result.retreated is True
    assert len(result.rounds) == 1
    assert not result.attacker_sunk
    assert not result.defender_sunk


def test_run_battle_never_asks_before_the_first_round():
    board = _sea_board()
    attacker = _ship(AxialCoord(0, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)
    defender = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    gs = _game_state(board, [attacker, defender])

    calls = []

    def decision_fn(a, d, g):
        calls.append(1)
        return "retreat"

    run_battle(attacker, defender, gs, decision_fn=decision_fn)
    assert len(calls) == 1  # asked only after the (only) round that occurred
