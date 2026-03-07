import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

from parse_gam.models import BoardState, POINT_COLUMNS


def __parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("states", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def load_states(path: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(path.iterdir()):
        if not p.suffix == ".json":
            continue
        state = BoardState.load(p)
        d = state.to_dict()
        d["filename"] = p.name
        rows.append(d)
    return pd.DataFrame(rows)


def smooth_states(df, imputation_method="linear", window_size=9):
    dice_cols = ["board_1_dice", "board_2_dice"]
    bar_cols = ["bar_p1", "bar_p2"]
    off_cols = ["off_p1", "off_p2"]
    # Columns that get rolling-mean smoothing (numeric, continuous-ish)
    smooth_cols = [*POINT_COLUMNS, *dice_cols, *bar_cols, *off_cols]
    # Columns that are passed through without smoothing (categorical / list)
    passthrough_cols = ["dice_values", "dice_board_half", "cube_value", "cube_position"]

    smoothed_df = df.copy()
    smoothed_df = smoothed_df.sort_values(["file_index"]).reset_index(drop=True)

    # Drop frames that are invalid
    smoothed_df = smoothed_df[smoothed_df.status == "VALID"]

    for col in smooth_cols:
        if col not in smoothed_df.columns:
            continue
        if imputation_method == "linear":
            smoothed_df[col] = smoothed_df[col].interpolate(
                method="linear", limit_direction="both"
            )
        elif imputation_method == "ffill":
            smoothed_df[col] = smoothed_df[col].fillna(method="ffill")
        elif imputation_method == "bfill":
            smoothed_df[col] = smoothed_df[col].fillna(method="bfill")
        else:
            raise ValueError("imputation_method must be 'linear', 'ffill', or 'bfill'")

    round_cols = [*POINT_COLUMNS, *bar_cols, *off_cols]
    for col in round_cols:
        if col not in smoothed_df.columns:
            continue
        smoothed_df[col] = (
            smoothed_df[col]
            .rolling(window=window_size, min_periods=1, center=True)
            .mean()
            .round()
            .astype("Int64")
        )

    # Dice counts: round but don't rolling-average (they're 0/1/2 counts)
    for col in dice_cols:
        if col not in smoothed_df.columns:
            continue
        smoothed_df[col] = (
            smoothed_df[col]
            .rolling(window=window_size, min_periods=1, center=True)
            .median()
            .round()
            .astype("Int64")
        )

    # dice_board_half: use mode within window (categorical)
    if "dice_board_half" in smoothed_df.columns:
        smoothed_df["dice_board_half"] = (
            smoothed_df["dice_board_half"]
            .rolling(window=window_size, min_periods=1, center=True)
            .apply(lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else np.nan, raw=False)
        )

    return smoothed_df


def save_states(df, path: Path):
    path.mkdir(exist_ok=True)

    for _, row in df.iterrows():
        data = row.to_dict()
        filename = data.pop("filename")

        with (path / filename).open("w") as f:
            json.dump(data, f)


def main():
    args = __parse_args()
    df = load_states(args.states)
    smoothed_df = smooth_states(df)
    save_states(smoothed_df, args.output)


if __name__ == "__main__":
    main()
