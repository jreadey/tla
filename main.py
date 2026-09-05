"""Entrypoint: generate a map from config and open the game window."""

from __future__ import annotations

import argparse

from tla.config import Config
from tla.mapgen import generate_map
from tla.rendering.app import run


def main() -> None:
    parser = argparse.ArgumentParser(description="tla - navy strategy game")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to a JSON config override file"
    )
    parser.add_argument("--seed", type=int, default=None, help="Map generation seed override")
    args = parser.parse_args()

    config = Config.load(args.config)
    board = generate_map(config.map, config.ports, seed=args.seed)
    run(board)


if __name__ == "__main__":
    main()
