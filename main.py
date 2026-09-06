"""Entrypoint: generate a map from config and open the game window."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

from tla.config import Config
from tla.game_state import new_game
from tla.rendering.app import compute_default_map_size, run


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

    # Auto-fit the map to the current screen so the whole thing is visible
    # without panning, unless the chosen config file explicitly pins a map
    # size itself (e.g. configs/large.json, which is meant to require
    # panning regardless of screen size).
    raw_map_config = json.loads(Path(args.config).read_text()).get("map", {}) if args.config else {}
    if "width" not in raw_map_config and "height" not in raw_map_config:
        width, height = compute_default_map_size(config.map.hex_pixel_size)
        updates = {"width": width, "height": height}
        if "noise_scale" not in raw_map_config:
            updates["noise_scale"] = width * 0.15
        config = replace(config, map=replace(config.map, **updates))

    seed = args.seed if args.seed is not None else config.map.seed
    if seed is None:
        seed = random.randrange(1_000_000)
    print(f"Map seed: {seed} (pass --seed {seed} to reproduce this map)")
    print(f"Map size: {config.map.width}x{config.map.height} hexes (fit to your screen)")

    game_state = new_game(config, seed)
    run(game_state)


if __name__ == "__main__":
    main()
