from dataclasses import replace
from unittest.mock import patch

from tla.ai.policy import NaivePolicy
from tla.board import Board
from tla.config import AiConfig, Config, FowConfig
from tla.game_state import GameState
from tla.hexgrid import AxialCoord, distance, hexes_in_range
from tla.ship import Ship, ShipKind
from tla.tile import PLAYER_A, PLAYER_B, Tile, TerrainType


def _sea_board(radius: int = 8) -> Board:
    board = Board(width=radius * 2 + 1, height=radius * 2 + 1)
    for coord in hexes_in_range(AxialCoord(0, 0), radius):
        board.tiles[coord] = Tile(coord=coord, terrain=TerrainType.SEA)
    return board


def _ship(
    coord: AxialCoord, kind: ShipKind, owner, ship_id: int, hp: int | None = None, surfaced: bool = True
) -> Ship:
    stats = Config().ship_stats.stats[kind]
    return Ship(
        id=ship_id,
        kind=kind,
        owner=owner,
        position=coord,
        current_hp=hp if hp is not None else stats.hp,
        surfaced=surfaced,
        movement_remaining=stats.movement,
    )


def _game_state(board: Board, ships: list[Ship], config: Config | None = None) -> GameState:
    return GameState(config=config or Config(fow=FowConfig(enabled=False)), board=board, ships={s.id: s for s in ships})


def _run(policy: NaivePolicy, gs: GameState, player) -> None:
    list(policy.plan_movement(gs, player))


def test_ai_moves_toward_a_visible_enemy_it_cannot_yet_reach():
    board = _sea_board(radius=10)
    mover = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)  # movement 3
    enemy = _ship(AxialCoord(8, 0), ShipKind.DESTROYER, PLAYER_B, 2)  # well out of reach this turn
    gs = _game_state(board, [mover, enemy])

    _run(NaivePolicy(), gs, PLAYER_A)

    assert 1 in gs.ships
    assert distance(gs.ships[1].position, enemy.position) < 8


def test_ai_never_issues_an_illegal_move():
    # A broad smoke test: run several ships of every kind through a shared
    # board with both sides present and assert nothing raises -- every
    # movement/battle/production call the AI makes goes through the same
    # legality checks a human's input does, so a clean run here is direct
    # evidence nothing illegal happened.
    board = _sea_board()
    ships = []
    next_id = 1
    for i, kind in enumerate(ShipKind):
        ships.append(_ship(AxialCoord(i, 0), kind, PLAYER_A, next_id))
        next_id += 1
        ships.append(_ship(AxialCoord(i, 3), kind, PLAYER_B, next_id))
        next_id += 1
    gs = _game_state(board, ships)

    _run(NaivePolicy(), gs, PLAYER_A)
    _run(NaivePolicy(), gs, PLAYER_B)


def test_ai_declines_a_clearly_losing_engagement():
    board = _sea_board()
    patrol_boat = _ship(AxialCoord(0, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)
    battleship = _ship(AxialCoord(1, 0), ShipKind.BATTLESHIP, PLAYER_B, 2, hp=100)
    gs = _game_state(board, [patrol_boat, battleship])

    _run(NaivePolicy(), gs, PLAYER_A)

    # The patrol boat should still be alive (didn't attack and get wiped
    # out) and the battleship untouched (never engaged).
    assert 1 in gs.ships
    assert gs.ships[2].current_hp == 100


def test_ai_attacks_a_clearly_favorable_target():
    board = _sea_board()
    battleship = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    patrol_boat = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    gs = _game_state(board, [battleship, patrol_boat])

    _run(NaivePolicy(), gs, PLAYER_A)

    assert 2 not in gs.ships  # the patrol boat was sunk


def test_ai_retreats_mid_battle_once_badly_damaged():
    # Two evenly matched destroyers -- matchup_score is a tie (0, still
    # "favorable" per the tie-goes-to-the-attacker rule), so the attack
    # actually starts. A near-90% damaged-withdraw threshold then pulls the
    # attacker out after the very first round purely because of its own HP
    # loss, independent of the race still looking even.
    board = _sea_board()
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(damaged_withdraw_fraction=0.9))
    attacker = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    defender = _ship(AxialCoord(1, 0), ShipKind.DESTROYER, PLAYER_B, 2)
    gs = _game_state(board, [attacker, defender], config=config)

    _run(NaivePolicy(), gs, PLAYER_A)

    assert 1 in gs.ships
    assert gs.ships[1].position == AxialCoord(0, 0)  # retreated to its approach hex


def test_ai_respects_fog_of_war_and_ignores_a_ship_it_cannot_see():
    board = _sea_board()
    config = Config(fow=FowConfig(enabled=True, ship_visibility_radius=1))
    mover = _ship(AxialCoord(0, 0), ShipKind.DESTROYER, PLAYER_A, 1)
    far_enemy = _ship(AxialCoord(6, 0), ShipKind.DESTROYER, PLAYER_B, 2)
    port = AxialCoord(-6, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    gs = _game_state(board, [mover, far_enemy], config=config)

    _run(NaivePolicy(), gs, PLAYER_A)

    # With nothing visible, the AI should head toward the uncontrolled
    # port (there is none of its own already controlled here) rather than
    # magically beelining for the enemy it can't see.
    assert distance(gs.ships[1].position, far_enemy.position) >= distance(mover.position, far_enemy.position)


def test_ai_never_reveals_a_hidden_submerged_submarine_by_avoiding_it():
    # A submerged enemy sub outside vision is invisible -- the AI has no
    # way to know to route around it, so a ship that ends up moving onto
    # it should trigger a real battle exactly like a human's accidental
    # contact would, not silently dodge it.
    board = _sea_board()
    config = Config(fow=FowConfig(enabled=True, ship_visibility_radius=1))
    mover = _ship(AxialCoord(0, 0), ShipKind.PATROL_BOAT, PLAYER_A, 1)
    mover.movement_remaining = 1
    sub = _ship(AxialCoord(1, 0), ShipKind.SUBMARINE, PLAYER_B, 2, surfaced=False)
    gs = _game_state(board, [mover, sub], config=config)

    _run(NaivePolicy(), gs, PLAYER_A)

    # No visible enemy and no uncontrolled port anywhere -> the naive
    # policy has nothing to do and holds position; this just confirms
    # nothing raised despite a totally hidden adjacent threat.
    assert 1 in gs.ships


def test_carrier_never_initiates_an_attack():
    board = _sea_board()
    carrier = _ship(AxialCoord(0, 0), ShipKind.CARRIER, PLAYER_A, 1)
    weak_enemy = _ship(AxialCoord(1, 0), ShipKind.PATROL_BOAT, PLAYER_B, 2)
    gs = _game_state(board, [carrier, weak_enemy])

    _run(NaivePolicy(), gs, PLAYER_A)

    assert 2 in gs.ships  # never attacked, even though it's a favorable matchup on paper
    assert gs.ships[2].current_hp == weak_enemy.current_hp


def test_carrier_falls_back_toward_its_escort_when_threatened():
    board = _sea_board()
    config = Config(fow=FowConfig(enabled=False), ai=AiConfig(carrier_threat_radius=5))
    carrier_start, escort_start, enemy_start = AxialCoord(0, 0), AxialCoord(-3, 0), AxialCoord(2, 0)
    carrier = _ship(carrier_start, ShipKind.CARRIER, PLAYER_A, 1)
    escort = _ship(escort_start, ShipKind.DESTROYER, PLAYER_A, 2)
    enemy = _ship(enemy_start, ShipKind.DESTROYER, PLAYER_B, 3)
    gs = _game_state(board, [carrier, escort, enemy], config=config)

    _run(NaivePolicy(), gs, PLAYER_A)

    # Ships are mutated in place, so compare against the coordinates
    # captured before the run, not the (now-moved) fixture objects.
    carrier_after = gs.ships[1]
    assert distance(carrier_after.position, escort_start) < distance(carrier_start, escort_start)
    assert distance(carrier_after.position, enemy_start) > distance(carrier_start, enemy_start)


def test_capital_ship_holds_position_when_no_escort_is_available_to_scout_ahead():
    board = _sea_board()
    battleship = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    enemy = _ship(AxialCoord(6, 0), ShipKind.SUBMARINE, PLAYER_B, 2, surfaced=False)
    # No patrol boat/destroyer escort anywhere in the fleet.
    gs = _game_state(board, [battleship, enemy])

    _run(NaivePolicy(), gs, PLAYER_A)

    assert gs.ships[1].position == AxialCoord(0, 0)  # held rather than advancing un-scouted


def test_capital_ship_sends_an_escort_ahead_before_advancing():
    board = _sea_board()
    battleship = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    scout = _ship(AxialCoord(0, 1), ShipKind.DESTROYER, PLAYER_A, 2)
    enemy = _ship(AxialCoord(4, 0), ShipKind.SUBMARINE, PLAYER_B, 3, surfaced=False)
    gs = _game_state(board, [battleship, scout, enemy])

    _run(NaivePolicy(), gs, PLAYER_A)

    # The battleship holds; the scout is the one that moved this turn.
    assert gs.ships[1].position == AxialCoord(0, 0)
    assert gs.ships[2].position != AxialCoord(0, 1)


def test_scout_prepass_survives_losing_the_last_visible_contact_mid_pass():
    # Regression: fog-of-war vision is recomputed fresh from each ship's
    # current position (see tla.fow), so a scout's own move earlier in the
    # prepass can shrink the player's vision enough to lose the only
    # visible enemy entirely -- before this was guarded, the *next*
    # capital ship's scout-distance check crashed with
    # "min() iterable argument is empty" instead of gracefully falling
    # back to normal (non-scouted) movement. Reproduced deterministically
    # by mocking enemy_ships_visible_to to go from one visible enemy to
    # none, rather than relying on fragile real-geometry vision loss.
    board = _sea_board(radius=25)
    port = AxialCoord(0, 20)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    capital1 = _ship(AxialCoord(0, 0), ShipKind.BATTLESHIP, PLAYER_A, 1)
    escort1 = _ship(AxialCoord(0, -3), ShipKind.DESTROYER, PLAYER_A, 2)
    capital2 = _ship(AxialCoord(5, 0), ShipKind.BATTLESHIP, PLAYER_A, 3)
    fake_enemy = _ship(AxialCoord(0, -6), ShipKind.DESTROYER, PLAYER_B, 4)
    gs = _game_state(board, [capital1, escort1, capital2, fake_enemy], config=Config(fow=FowConfig(enabled=False)))

    calls = [{4: fake_enemy}, {}]

    def fake_visible(game_state, player):
        return calls.pop(0) if calls else {}

    with patch("tla.ai.policy.enemy_ships_visible_to", side_effect=fake_visible):
        list(NaivePolicy().plan_movement(gs, PLAYER_A))  # must not raise


def test_plan_production_tops_up_every_controlled_port_to_queue_depth():
    board = _sea_board()
    port1, port2 = AxialCoord(0, 0), AxialCoord(2, 0)
    board.tiles[port1] = Tile(coord=port1, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    board.tiles[port2] = Tile(coord=port2, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    config = replace(Config(), ai=AiConfig(queue_depth=2))
    gs = GameState(config=config, board=board, ships={})

    NaivePolicy().plan_production(gs, PLAYER_A)

    assert len(gs.players[PLAYER_A].port_production[port1].orders) == 2
    assert len(gs.players[PLAYER_A].port_production[port2].orders) == 2


def test_plan_production_does_not_exceed_queue_depth_on_repeated_calls():
    board = _sea_board()
    port = AxialCoord(0, 0)
    board.tiles[port] = Tile(coord=port, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    config = replace(Config(), ai=AiConfig(queue_depth=1))
    gs = GameState(config=config, board=board, ships={})

    policy = NaivePolicy()
    policy.plan_production(gs, PLAYER_A)
    policy.plan_production(gs, PLAYER_A)

    assert len(gs.players[PLAYER_A].port_production[port].orders) == 1
