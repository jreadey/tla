from tla.board import Board
from tla.config import Config
from tla.game_state import GameState, TurnPhase
from tla.hexgrid import AxialCoord
from tla.ship import Ship, ShipKind
from tla.tile import PLAYER_A, PLAYER_B
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


def test_full_round_trip_starts_a_new_turn():
    gs = _game_state()
    manager = TurnManager(gs)

    manager.end_movement_phase()  # A -> B
    manager.end_movement_phase()  # B -> A, new turn

    assert gs.phase == TurnPhase.MOVE_A
    assert gs.current_player == PLAYER_A
    assert gs.turn_number == 2
    ship_a = gs.ships[1]
    assert ship_a.movement_remaining == gs.config.ship_stats.stats[ship_a.kind].movement
    assert ship_a.toggled_pre_move is False
