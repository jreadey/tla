"""Print a human-readable, turn-by-turn summary of a .jsonl replay file
written by `main.py --replay PATH` (see tla/replay.py). Usage:

    python replay_viewer.py path/to/replay.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _task_force_lines(record: dict) -> list[str]:
    """Human-readable summary of each side's AI task forces as of this
    record, if any were included when the replay was written (see
    tla.replay's optional `task_forces` param) -- empty for a human-only
    game, or a record written without them."""
    lines: list[str] = []
    for tf in sorted(record.get("task_forces", []), key=lambda f: f["id"]):
        goal = tf["goal"]
        goal_str = f"{goal['kind']}@{tuple(goal['target'])}" if goal else "none"
        flags = []
        if tf["retreating"]:
            threat = tf["retreat_threat_power"]
            threat_str = f" (need power > {tuple(threat)})" if threat else ""
            flags.append(f"retreating, {tf['retreat_turns']} turns so far{threat_str}")
        if tf["turns_since_progress"] > 0:
            flags.append(f"no progress for {tf['turns_since_progress']} turns")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        lines.append(
            f"  TF{tf['id']} (P{tf['owner']}, {len(tf['member_ids'])} ships {tf['member_ids']}): "
            f"goal={goal_str}{flag_str}"
        )
    return lines


def _battle_lines(record: dict) -> list[str]:
    """Human-readable summary of every combat round in this record's
    `battle_log` (see tla.replay/tla.game_state.BattleLogEntry) -- empty
    for a record with no `battle_log` (an older replay file) or none this
    half-turn. Groups consecutive rounds that share the same attacker/
    defender pair into one printed "battle" -- the raw log has no explicit
    battle boundary, but two ships breaking off and re-fighting later in
    the very same half-turn is rare enough that simple adjacency is a
    reliable grouping in practice."""
    lines: list[str] = []
    battles = record.get("battle_log", [])
    i = 0
    while i < len(battles):
        first = battles[i]
        group = [first]
        j = i + 1
        while (
            j < len(battles)
            and battles[j]["attacker_id"] == first["attacker_id"]
            and battles[j]["defender_id"] == first["defender_id"]
        ):
            group.append(battles[j])
            j += 1

        lines.append(
            f"  Battle: P{first['attacker_owner']} {first['attacker_kind']}{first['attacker_id']} vs "
            f"P{first['defender_owner']} {first['defender_kind']}{first['defender_id']} "
            f"@ {tuple(first['battle_hex'])}"
        )
        for round_num, r in enumerate(group, start=1):
            assists = []
            if r["attacker_carrier_bonus"]:
                assists.append(f"attacker+{r['attacker_carrier_bonus']}")
            if r["defender_carrier_bonus"]:
                assists.append(f"defender+{r['defender_carrier_bonus']}")
            assist_str = f" ({', '.join(assists)})" if assists else ""
            lines.append(
                f"    round {round_num}: dealt {r['damage_to_defender']}/{r['damage_to_attacker']}"
                f"{assist_str} -> hp {r['defender_hp_after']}/{r['attacker_hp_after']}"
            )
        last = group[-1]
        if last["defender_sunk"] and last["attacker_sunk"]:
            lines.append("    -> both sunk")
        elif last["defender_sunk"]:
            lines.append("    -> defender sunk")
        elif last["attacker_sunk"]:
            lines.append("    -> attacker sunk")
        else:
            lines.append("    -> attacker retreated")

        i = j
    return lines


def _ports_from_initial(record: dict) -> list[dict]:
    return [
        {
            "coord": tile["coord"],
            "display_owner": (
                tile["port_controller"]
                if tile["port_controller"] is not None
                else tile["port_owner"]
            ),
        }
        for tile in record["board"]["tiles"]
        if tile["is_port"]
    ]


def summarize_replay(records: list[dict]) -> list[str]:
    """Diffs each record's ship/port snapshot against the previous one and
    reports what changed in plain text: ships that moved and/or took
    damage, ships that sank or newly appeared (e.g. produced at a port),
    port captures, non-zero turn stats, and the final winner. Keeps its own
    running 'previous ships/ports' state purely for this diff -- that state
    lives only here, never in the writer."""
    lines: list[str] = []
    prev_ships: dict[int, dict] = {}
    prev_ports: dict[tuple, dict] = {}

    for record in records:
        record_type = record["type"]

        if record_type == "initial":
            board = record["board"]
            lines.append(f"== Initial state (seed={record.get('seed')}) ==")
            lines.append(
                f"  {len(record['ships'])} ships on a {board['width']}x{board['height']} map"
            )
            prev_ships = {ship["id"]: ship for ship in record["ships"]}
            prev_ports = {
                tuple(port["coord"]): port for port in _ports_from_initial(record)
            }
            continue

        if record_type == "half_turn":
            lines.append(
                f"== Turn {record['turn_number']} — Player {record['player']} "
                f"({record['phase']}) =="
            )
        else:
            lines.append(f"== GAME OVER (turn {record['turn_number']}) ==")

        cur_ships = {ship["id"]: ship for ship in record["ships"]}
        for ship_id in sorted(set(prev_ships) - set(cur_ships)):
            s = prev_ships[ship_id]
            lines.append(
                f"  Ship {ship_id} ({s['kind']}, P{s['owner']}) sunk at {tuple(s['position'])}"
            )
        for ship_id in sorted(set(cur_ships) - set(prev_ships)):
            s = cur_ships[ship_id]
            lines.append(
                f"  Ship {ship_id} ({s['kind']}, P{s['owner']}) appeared at {tuple(s['position'])}"
            )
        for ship_id in sorted(set(cur_ships) & set(prev_ships)):
            before, after = prev_ships[ship_id], cur_ships[ship_id]
            moved = tuple(before["position"]) != tuple(after["position"])
            hp_changed = before["hp"] != after["hp"]
            if moved or hp_changed:
                bits = []
                if moved:
                    bits.append(f"moved {tuple(before['position'])} -> {tuple(after['position'])}")
                if hp_changed:
                    bits.append(f"hp {before['hp']} -> {after['hp']}")
                lines.append(
                    f"  Ship {ship_id} ({after['kind']}, P{after['owner']}) " + ", ".join(bits)
                )

        cur_ports = {tuple(port["coord"]): port for port in record["ports"]}
        for coord in sorted(set(cur_ports) & set(prev_ports)):
            before, after = prev_ports[coord], cur_ports[coord]
            if before["display_owner"] != after["display_owner"]:
                lines.append(
                    f"  Port {coord} captured: P{before['display_owner']} -> P{after['display_owner']}"
                )

        lines.extend(_battle_lines(record))

        for player, stats in sorted(record.get("turn_stats", {}).items()):
            if stats["hp_dealt"] or stats["hp_taken"] or stats["ships_lost"]:
                lines.append(
                    f"  Player {player} stats: dealt={stats['hp_dealt']} "
                    f"taken={stats['hp_taken']} lost={stats['ships_lost']}"
                )

        lines.extend(_task_force_lines(record))

        if record_type == "final":
            winner = record.get("winner")
            lines.append(f"  Winner: Player {winner}" if winner is not None else "  No winner recorded")

        prev_ships = cur_ships
        prev_ports = cur_ports

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a human-readable summary of a tla replay .jsonl file"
    )
    parser.add_argument("path", type=str, help="Path to a .jsonl replay file written by --replay")
    args = parser.parse_args()

    records = [
        json.loads(line) for line in Path(args.path).read_text().splitlines() if line.strip()
    ]
    for line in summarize_replay(records):
        print(line)


if __name__ == "__main__":
    main()
