"""Parse a full backgammon game from a sequence of smoothed board states.

Detects:
  - Turn boundaries (dice appear -> board changes -> dice disappear)
  - Game boundaries (starting position / all checkers borne off)
  - Game result (single / gammon / backgammon / drop)
"""

from pathlib import Path
import json
import argparse
from collections import Counter

import pandas as pd

from parse_gam.models import (
    BoardState,
    GameRecord,
    Turn,
    POINT_COLUMNS,
    STARTING_POSITION,
)


def __parse_args():
    parser = argparse.ArgumentParser(
        description="Parse game structure from smoothed board states."
    )
    parser.add_argument("states", type=Path, help="Directory of smoothed state JSONs")
    parser.add_argument("output", type=Path, help="Output JSON file for game record")
    parser.add_argument(
        "--stable-window",
        type=int,
        default=3,
        help="Min consecutive identical frames to consider state stable (default: 3)",
    )
    return parser.parse_args()


def load_states(states_dir: Path) -> list[BoardState]:
    states = []
    for p in sorted(states_dir.iterdir()):
        if p.suffix != ".json":
            continue
        states.append(BoardState.load(p))
    return states


def _points_match(a: list[int], b: list[int]) -> bool:
    return a == b


def _board_changed(a: BoardState, b: BoardState) -> bool:
    return (
        a.points != b.points
        or a.bar_p1 != b.bar_p1
        or a.bar_p2 != b.bar_p2
    )


def _most_common_dice_half(states: list[BoardState]) -> int | None:
    halves = [s.dice_board_half for s in states if s.dice_board_half is not None]
    if not halves:
        return None
    counts = Counter(halves)
    return counts.most_common(1)[0][0]


def _collect_dice_values(states: list[BoardState]) -> list[int]:
    """Collect dice values seen during a turn segment.

    Returns the most frequently observed pair of values.
    """
    value_lists = [
        tuple(sorted(s.dice_values))
        for s in states
        if s.dice_values and len(s.dice_values) >= 2
    ]
    if not value_lists:
        # Fall back to single values
        all_vals = []
        for s in states:
            all_vals.extend(s.dice_values)
        return sorted(all_vals)[:2] if all_vals else []

    counts = Counter(value_lists)
    return list(counts.most_common(1)[0][0])


def detect_turns(states: list[BoardState]) -> list[Turn]:
    """Detect turns by tracking dice lifecycle.

    A turn is a segment where:
      1. Dice become visible (has_dice transitions False -> True)
      2. Board state may change
      3. Dice disappear (has_dice transitions True -> False)

    The pre-turn state is the last stable state before dice appeared.
    The post-turn state is the first stable state after dice disappear.
    """
    valid = [s for s in states if s.status.value == "VALID"]
    if len(valid) < 2:
        return []

    turns = []
    in_turn = False
    turn_dice_states = []
    pre_turn_state = valid[0]

    for i, state in enumerate(valid):
        if not in_turn and state.has_dice:
            # Dice just appeared -> turn starts
            in_turn = True
            turn_dice_states = [state]
            # pre_turn_state is the last state before dice appeared
            if i > 0:
                pre_turn_state = valid[i - 1]

        elif in_turn and state.has_dice:
            turn_dice_states.append(state)

        elif in_turn and not state.has_dice:
            # Dice disappeared -> turn ends
            in_turn = False
            post_turn_state = state

            if _board_changed(pre_turn_state, post_turn_state):
                dice_half = _most_common_dice_half(turn_dice_states)
                # Map dice_board_half to player: 0 (left board) -> P1, 1 (right) -> P2
                player = (dice_half + 1) if dice_half is not None else 0

                turn = Turn(
                    player=player,
                    dice=_collect_dice_values(turn_dice_states),
                    state_before=pre_turn_state.to_dict(),
                    state_after=post_turn_state.to_dict(),
                    frame_start=turn_dice_states[0].file_index or 0,
                    frame_end=post_turn_state.file_index or 0,
                )
                turns.append(turn)

            pre_turn_state = post_turn_state

    return turns


def detect_game_boundaries(states: list[BoardState]) -> list[tuple[int, int]]:
    """Detect game start/end frame indices.

    Game start: board matches starting position.
    Game end: one player has borne off all 15 checkers, or board resets.
    Returns list of (start_frame, end_frame) tuples.
    """
    valid = [s for s in states if s.status.value == "VALID"]
    if not valid:
        return []

    boundaries = []
    game_start = valid[0].file_index or 0

    for i, state in enumerate(valid):
        # Game end: all checkers borne off
        if state.off_p1 >= 15 or state.off_p2 >= 15:
            boundaries.append((game_start, state.file_index or 0))
            # Next game starts after this
            if i + 1 < len(valid):
                game_start = valid[i + 1].file_index or 0

        # Game start: board reset to starting position
        elif i > 0 and state.is_starting_position() and not valid[i - 1].is_starting_position():
            game_start = state.file_index or 0

    # If no end was detected, treat the whole sequence as one game
    if not boundaries:
        boundaries.append((game_start, valid[-1].file_index or 0))

    return boundaries


def classify_result(final_state: BoardState) -> tuple[str, int | None]:
    """Classify game result from the final board state.

    Returns (result_type, winner).
    """
    if final_state.off_p1 >= 15:
        winner = 1
        loser_off = final_state.off_p2
        loser_bar = final_state.bar_p2
        # Check if loser has checkers in winner's home board (points 19-24 for P1)
        loser_in_winner_home = any(
            final_state.points[i] > 0 for i in range(18, 24)
        )
    elif final_state.off_p2 >= 15:
        winner = 2
        loser_off = final_state.off_p1
        loser_bar = final_state.bar_p1
        # Check if loser has checkers in winner's home board (points 1-6 for P2)
        loser_in_winner_home = any(
            final_state.points[i] < 0 for i in range(0, 6)
        )
    else:
        return "unknown", None

    if loser_off == 0 and (loser_bar > 0 or loser_in_winner_home):
        return "backgammon", winner
    elif loser_off == 0:
        return "gammon", winner
    else:
        return "single", winner


def parse_game(states: list[BoardState]) -> list[GameRecord]:
    """Parse one or more games from a sequence of board states."""
    boundaries = detect_game_boundaries(states)
    turns = detect_turns(states)
    games = []

    for start_frame, end_frame in boundaries:
        game_turns = [
            t for t in turns
            if t.frame_start >= start_frame and t.frame_end <= end_frame
        ]

        # Find the last valid state in this game's range
        game_states = [
            s for s in states
            if s.status.value == "VALID"
            and (s.file_index or 0) >= start_frame
            and (s.file_index or 0) <= end_frame
        ]

        result = "unknown"
        winner = None
        cube_final = 1

        if game_states:
            final = game_states[-1]
            result, winner = classify_result(final)
            if final.cube_value is not None:
                cube_final = final.cube_value

        game = GameRecord(
            turns=game_turns,
            result=result,
            winner=winner,
            cube_final=cube_final,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        games.append(game)

    return games


def main():
    args = __parse_args()
    states = load_states(args.states)
    games = parse_game(states)

    output = {
        "games": [g.to_dict() for g in games],
        "total_games": len(games),
        "total_turns": sum(len(g.turns) for g in games),
    }

    with args.output.open("w") as f:
        json.dump(output, f, indent=2)

    for i, game in enumerate(games):
        print(
            f"Game {i+1}: {len(game.turns)} turns, "
            f"result={game.result}, winner=P{game.winner or '?'}, "
            f"frames {game.start_frame}-{game.end_frame}"
        )

    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
