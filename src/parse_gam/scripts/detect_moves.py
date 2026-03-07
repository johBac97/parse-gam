"""Detect discrete backgammon moves from a sequence of parsed board states.

Reads smoothed state JSON files, computes frame-to-frame diffs, and
coalesces nearby changes within a time window into individual moves.
"""

from pathlib import Path
import json
import argparse
import pandas as pd

from parse_gam.models import BoardState, POINT_COLUMNS

BAR_COLUMNS = ["bar_p1", "bar_p2"]
OFF_COLUMNS = ["off_p1", "off_p2"]
DIFF_COLUMNS = [*POINT_COLUMNS, *BAR_COLUMNS, *OFF_COLUMNS]


def __parse_args():
    parser = argparse.ArgumentParser(
        description="Detect moves from smoothed board states."
    )
    parser.add_argument("states", type=Path, help="Directory of smoothed state JSONs")
    parser.add_argument("output", type=Path, help="Output JSON file for detected moves")
    parser.add_argument(
        "--move-window",
        type=int,
        default=10,
        help="Max frame gap to coalesce changes into one move (default: 10)",
    )
    return parser.parse_args()


def load_states_df(states_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(states_dir.iterdir()):
        if not p.suffix == ".json":
            continue
        state = BoardState.load(p)
        d = state.to_dict()
        d["filename"] = p.name
        rows.append(d)
    return pd.DataFrame(rows)


def extract_moves(df: pd.DataFrame, move_window: int = 10) -> list[dict]:
    """Extract discrete moves from a DataFrame of board states.

    Computes frame-to-frame point diffs on VALID frames, then groups
    consecutive changes that occur within `move_window` frames into
    single logical moves.
    """
    valid = df[df.status == "VALID"].sort_values("file_index").reset_index(drop=True)

    if len(valid) < 2:
        return []

    # Ensure diff columns exist with defaults
    for col in DIFF_COLUMNS:
        if col not in valid.columns:
            valid[col] = 0

    frame_diff = valid[DIFF_COLUMNS].diff()
    frame_diff["file_index"] = valid["file_index"].values

    # Carry forward dice info for context
    if "dice_board_half" in valid.columns:
        frame_diff["dice_board_half"] = valid["dice_board_half"].values
    if "dice_values" in valid.columns:
        frame_diff["dice_values"] = valid["dice_values"].values

    # Keep only rows where something changed
    has_change = frame_diff[DIFF_COLUMNS].abs().sum(axis="columns") != 0
    change_rows = frame_diff[has_change].to_dict(orient="records")

    if not change_rows:
        return []

    moves = []
    current_window = [change_rows[0]]

    for row in change_rows[1:]:
        if row["file_index"] <= current_window[0]["file_index"] + move_window:
            current_window.append(row)
        else:
            moves.append(_finalize_move(current_window))
            current_window = [row]

    # Don't forget the last window
    moves.append(_finalize_move(current_window))

    return moves


def _finalize_move(window: list[dict]) -> dict:
    """Sum point/bar/off deltas within a move window into a single move."""
    move_df = pd.DataFrame(window)
    summed = move_df[DIFF_COLUMNS].sum()
    result = summed.to_dict()
    result["start_frame"] = int(move_df["file_index"].iloc[0])
    result["end_frame"] = int(move_df["file_index"].iloc[-1])

    # Include dice context if available
    if "dice_board_half" in move_df.columns:
        half = move_df["dice_board_half"].dropna()
        result["dice_board_half"] = int(half.mode().iloc[0]) if not half.empty else None
    if "dice_values" in move_df.columns:
        vals = move_df["dice_values"].dropna()
        if not vals.empty:
            # Take the most common dice values seen during this move
            result["dice_values"] = vals.iloc[0]

    return result


def main():
    args = __parse_args()

    df = load_states_df(args.states)
    moves = extract_moves(df, move_window=args.move_window)

    with args.output.open("w") as f:
        json.dump(moves, f, indent=2)

    print(f"Detected {len(moves)} moves, written to {args.output}")


if __name__ == "__main__":
    main()
