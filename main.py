"""Entrypoint: generate a map from config and open the game window."""

from __future__ import annotations

import argparse
import random

from tla.config import Config
from tla.mapgen import generate_map
from tla.rendering.app import run


def main() -> None:
    parser = argparse.ArgumentParser(description="tla - navy strategy game")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to a JSON config override file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fixed map generation seed, to reproduce a specific map. Omit for a random map.",
    )
    args = parser.parse_args()

    config = Config.load(args.config)
    seed = args.seed if args.seed is not None else config.map.seed
    if seed is None:
        seed = random.randrange(1_000_000)
    print(f"Map seed: {seed} (pass --seed {seed} to reproduce this map)")

    board = generate_map(config.map, config.ports, seed=seed)
    run(board)


if __name__ == "__main__":
    main()
