from tla.board import Board
from tla.config import Config
from tla.game_state import GameState, TurnPhase
from tla.hexgrid import AxialCoord
from tla.production import order
from tla.ship import Ship, ShipKind
from tla.tile import PLAYER_A, PLAYER_B, Tile, TerrainType
from tla.turn_manager import TurnManager


def _ship(owner, ship_id, kind=ShipKind.DESTROYER, movement_remaining=0, surfaced=True) -> Ship:
    return Ship(
        id=ship_id,
        kind=kind,
        owner=owner,
        position=AxialCoord(ship_id, 0),
        current_hp=6,
        surfaced=surfaced,
        movement_remaining=movement_remaining,
        toggled_pre_move=True,
        toggled_post_move=True,
    )


def _game_state() -> GameState:
    board = Board(width=5, height=5)
    ships = [
        _ship(PLAYER_A, 1, movement_remaining=1),
        _ship(PLAYER_B, 2, movement_remaining=0),
    ]
    return GameState(config=Config(), board=board, ships={s.id: s for s in ships})


def test_end_movement_phase_advances_to_player_b():
    gs = _game_state()
    manager = TurnManager(gs)

    manager.end_movement_phase()

    assert gs.phase == TurnPhase.MOVE_B
    assert gs.current_player == PLAYER_B
    assert gs.turn_number == 1


def test_end_movement_phase_resets_the_new_players_ships():
    gs = _game_state()
    manager = TurnManager(gs)

    manager.end_movement_phase()  # -> Player B's phase

    ship_b = gs.ships[2]
    assert ship_b.movement_remaining == gs.config.ship_stats.stats[ship_b.kind].movement
    assert ship_b.toggled_pre_move is False
    assert ship_b.toggled_post_move is False
    # Player A's ships are untouched by B's phase starting.
    ship_a = gs.ships[1]
    assert ship_a.toggled_pre_move is True


def test_end_movement_phase_from_move_b_starts_a_new_turn():
    gs = _game_state()
    manager = TurnManager(gs)

    manager.end_movement_phase()  # A -> B
    manager.end_movement_phase()  # B -> production (automatic) -> new turn, move A

    assert gs.phase == TurnPhase.MOVE_A
    assert gs.current_player == PLAYER_A
    assert gs.turn_number == 2
    ship_a = gs.ships[1]
    assert ship_a.movement_remaining == gs.config.ship_stats.stats[ship_a.kind].movement
    assert ship_a.toggled_pre_move is False


def test_end_movement_phase_runs_production_for_both_players():
    board = Board(width=5, height=5)
    port_a, port_b = AxialCoord(0, 0), AxialCoord(4, 0)
    board.tiles[port_a] = Tile(coord=port_a, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    board.tiles[port_b] = Tile(coord=port_b, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B)
    gs = GameState(config=Config(), board=board, ships={})
    order(gs, PLAYER_A, port_a, ShipKind.PATROL_BOAT)
    order(gs, PLAYER_B, port_b, ShipKind.PATROL_BOAT)
    manager = TurnManager(gs)

    manager.end_movement_phase()  # A -> B
    manager.end_movement_phase()  # B -> production for both -> new turn

    assert any(s.owner == PLAYER_A and s.position == port_a for s in gs.ships.values())
    assert any(s.owner == PLAYER_B and s.position == port_b for s in gs.ships.values())


def test_end_movement_phase_advances_the_port_control_claim_at_each_turn_boundary():
    # A holds every port from the start -- shouldn't win on the very first
    # boundary, only once they've still held it at a second consecutive one.
    port_a, port_b = AxialCoord(0, 0), AxialCoord(4, 0)
    board = Board(width=5, height=5)
    board.tiles[port_a] = Tile(coord=port_a, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_A)
    board.tiles[port_b] = Tile(
        coord=port_b, terrain=TerrainType.LAND, is_port=True, port_owner=PLAYER_B, port_controller=PLAYER_A
    )
    gs = GameState(config=Config(), board=board, ships={})
    manager = TurnManager(gs)

    manager.end_movement_phase()  # A -> B, no boundary yet
    manager.end_movement_phase()  # B -> new turn: boundary 1

    assert gs.winner is None
    assert gs.port_control_claimant == PLAYER_A

    manager.end_movement_phase()  # A -> B
    manager.end_movement_phase()  # B -> new turn: boundary 2 -- confirmed

    assert gs.winner == PLAYER_A


def test_end_movement_phase_resets_turn_stats_only_once_a_new_turn_starts():
    gs = _game_state()
    manager = TurnManager(gs)
    gs.turn_stats[PLAYER_A].hp_dealt = 5
    gs.turn_stats[PLAYER_B].hp_taken = 5

    manager.end_movement_phase()  # A -> B: still mid-turn, stats untouched

    assert gs.turn_stats[PLAYER_A].hp_dealt == 5

    manager.end_movement_phase()  # B -> new turn: now reset

    assert gs.turn_stats[PLAYER_A].hp_dealt == 0
    assert gs.turn_stats[PLAYER_B].hp_taken == 0
