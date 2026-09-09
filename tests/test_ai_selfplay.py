"""A full NaivePolicy-vs-itself game, driven directly through TurnManager
on a small generated map -- the single test most likely to catch
interaction bugs (stalls, illegal-state edge cases) that narrow unit tests
would miss, since it exercises the whole movement/battle/production loop
together the way an actual game would."""

from __future__ import annotations

from tla.ai.policy import NaivePolicy
from tla.config import Config, FleetConfig, FowConfig, MapConfig, PortConfig
from tla.game_state import TurnPhase, new_game
from tla.ship import ShipKind
from tla.tile import PLAYER_A, PLAYER_B
from tla.turn_manager import TurnManager

_SMALL_CONFIG = Config(
    map=MapConfig(width=14, height=10, noise_scale=5.0),
    ports=PortConfig(ports_per_player=2, min_port_spacing=2),
    fleet=FleetConfig(
        counts={
            ShipKind.BATTLESHIP: 1,
            ShipKind.CARRIER: 1,
            ShipKind.CRUISER: 1,
            ShipKind.DESTROYER: 1,
            ShipKind.SUBMARINE: 1,
            ShipKind.PATROL_BOAT: 1,
        }
    ),
    fow=FowConfig(enabled=True),
)

_MAX_TURNS = 400


def test_naive_policy_vs_itself_reaches_a_winner_without_raising():
    for seed in (1, 2, 3):
        config = _SMALL_CONFIG
        game_state = new_game(config, seed=seed)
        turn_manager = TurnManager(game_state)
        policy = NaivePolicy()

        while game_state.winner is None and game_state.turn_number <= _MAX_TURNS:
            player = PLAYER_A if game_state.phase == TurnPhase.MOVE_A else PLAYER_B
            policy.plan_production(game_state, player)
            list(policy.plan_movement(game_state, player))
            turn_manager.end_movement_phase()

        assert game_state.winner is not None, f"seed {seed} never reached a winner within {_MAX_TURNS} turns"
        assert game_state.winner in (PLAYER_A, PLAYER_B)
