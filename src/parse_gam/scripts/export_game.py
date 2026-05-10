"""Export a parsed game.json to a human-readable match record.

Output format is a plain-text score sheet that most backgammon programs
can display and that can be manually converted to any import format needed
(e.g. GNU Backgammon SGF, FIBS match record, BGBlitz .bgm).

Usage
-----
    uv run export-game video10_test/game.json           # print to stdout
    uv run export-game video10_test/game.json game.txt  # write to file
"""

import argparse
import json
from pathlib import Path

from parse_gam.moves import moves_to_str, BAR, OFF


def __parse_args():
    parser = argparse.ArgumentParser(
        description="Export parsed game.json to a readable match record."
    )
    parser.add_argument("game_json", type=Path, help="Path to game.json")
    parser.add_argument(
        "output", type=Path, nargs="?", default=None,
        help="Output file (default: print to stdout)",
    )
    return parser.parse_args()


def _dice_str(dice: list[int]) -> str:
    return "-".join(str(d) for d in dice) if dice else "?"


def _moves_str(moves: list[list[int]]) -> str:
    if not moves:
        return "(no moves recorded)"
    return moves_to_str([tuple(m) for m in moves])  # type: ignore[arg-type]


def format_game(game: dict, game_num: int, match_length: int = 1) -> list[str]:
    lines: list[str] = []

    header = f"Game {game_num}"
    lines.append(header)
    lines.append("=" * len(header))
    lines.append("")

    turns = game.get("turns", [])
    if not turns:
        lines.append("  (no turns detected)")
    else:
        # Column headers
        lines.append(f"  {'#':>3}  {'Player':>6}  {'Dice':>5}  Moves")
        lines.append(f"  {'-'*3}  {'-'*6}  {'-'*5}  {'-'*30}")
        for i, turn in enumerate(turns):
            player = turn.get("player", "?")
            dice = _dice_str(turn.get("dice", []))
            moves = _moves_str(turn.get("moves", []))
            lines.append(f"  {i+1:>3}  P{player!s:<5}  {dice:>5}  {moves}")

    lines.append("")
    result = game.get("result", "unknown")
    winner = game.get("winner")
    points = game.get("points_won", 0)
    w_str = f"P{winner}" if winner else "unknown"
    lines.append(
        f"  Result: {result}  —  Winner: {w_str}  ({points} point{'s' if points != 1 else ''})"
    )

    return lines


def format_match(data: dict) -> str:
    games = data.get("games", [])
    total_turns = data.get("total_turns", 0)

    sections: list[str] = []
    sections.append(f"{len(games)}-game match  ({total_turns} turns total)")
    sections.append("")

    for i, game in enumerate(games):
        sections.extend(format_game(game, i + 1))
        sections.append("")

    return "\n".join(sections)


def main():
    args = __parse_args()

    with args.game_json.open() as f:
        data = json.load(f)

    text = format_match(data)

    if args.output:
        args.output.write_text(text)
        print(f"Written to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
