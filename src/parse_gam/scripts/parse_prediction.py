from pathlib import Path
import json

import geopandas as gpd
from shapely.geometry import Polygon
import pandas as pd
import argparse
from tqdm import tqdm

CLASS_MAPPING = {"BOARD": 0, "CHECKER_P1": 1, "CHECKER_P2": 2, "DIE": 3, "POINT": 4}


def __parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("predictions", type=Path)
    args.add_argument("output", type=Path)

    return args.parse_args()


def to_polygon(x):
    return Polygon(
        [
            (x.x_center - x.width / 2, x.y_center - x.height / 2),
            (x.x_center + x.width / 2, x.y_center - x.height / 2),
            (x.x_center + x.width / 2, x.y_center + x.height / 2),
            (x.x_center - x.width / 2, x.y_center + x.height / 2),
        ]
    )


def parse_yolo_predictions(predictions_path):
    df = pd.read_csv(
        predictions_path,
        names=["clas", "x_center", "y_center", "width", "height", "conf"],
        delimiter=" ",
    )

    df["geometry"] = df.apply(to_polygon, axis="columns")

    gdf = gpd.GeoDataFrame(df)

    return gdf


def project_onto_board(row):
    return {
        "clas": row["clas_pred"],
        "conf": row["conf_pred"],
        "board_index": row["board_index"],
        "x_center": (row.x_center_pred - row.x_center_board + row.width_board / 2)
        / row.width_board,
        "y_center": (row.y_center_pred - row.y_center_board + row.height_board / 2)
        / row.height_board,
        "width": row.width_pred / row.width_board,
        "height": row.height_pred / row.height_board,
    }


def iou(p1, p2):
    inter = p1.intersection(p2).area
    union = p1.union(p2).area
    return inter / union if union > 0 else 0


def deduplicate_gdf(gdf, iou_threshold: float = 0.8):
    """Deduplicate a geopandas frame based on IoU between geometries."""
    keep = []
    dropped = set()

    for i, geom_i in enumerate(gdf.geometry):
        if i in dropped:
            continue

        keep.append(i)

        for j in range(i + 1, len(gdf)):
            if j in dropped:
                continue
            geom_j = gdf.geometry[j]

            if iou(geom_i, geom_j) > iou_threshold:
                dropped.add(j)

    return gdf.loc[keep].reset_index(drop=True)


def parse_half_board_state(gdf):
    """
    Parse a half-board's state.

    Assumes x and y center columns are coordinates within [0,1] of the center of the checker position.
    Divides the board into 6 Backgammon 'Points' on the upper and lower sides. Assumes the points cover the length of [0,1]
    """
    state = {}

    CHECKER_POINT_X_TOLERANCE = 0.07

    for half in ["UPPER", "LOWER"]:
        for h_point_index in range(0, 6):
            h_pos = h_point_index / 6 + (1 / 6 / 2)
            if half == "LOWER":
                # Lower half
                d = gdf[
                    (gdf.y_center > 0.5)
                    & ((gdf.x_center - h_pos).abs() < CHECKER_POINT_X_TOLERANCE)
                ].reset_index(drop=True)
                point_index = 6 - h_point_index
            elif half == "UPPER":
                d = gdf[
                    (gdf.y_center < 0.5)
                    & ((gdf.x_center - h_pos).abs() < CHECKER_POINT_X_TOLERANCE)
                ].reset_index(drop=True)
                point_index = 7 + h_point_index
            else:
                raise RuntimeError()

            # Select which player by majority vote
            class_index_temp = (
                d.groupby("clas")
                .count()
                .reset_index()
                .nlargest(1, "board_index")["clas"]
            )

            if class_index_temp.shape[0] == 0:
                class_index = CLASS_MAPPING["CHECKER_P2"]
            else:
                class_index = class_index_temp.values[0]

            num_checkers = deduplicate_gdf(d).shape[0]
            val = (
                num_checkers
                if class_index == CLASS_MAPPING["CHECKER_P2"]
                else -num_checkers
            )

            state[f"Point_{point_index}"] = val

    return state


def parse_board_state(predictions: gpd.GeoDataFrame) -> dict | None:
    BOARD_CLASS = CLASS_MAPPING["BOARD"]
    CHECKER_P1_CLASS = CLASS_MAPPING["CHECKER_P1"]
    CHECKER_P2_CLASS = CLASS_MAPPING["CHECKER_P2"]

    # There are two 'boards' give the one to the left index 0 and the one to the right index 1
    boards = (
        predictions[predictions.clas == BOARD_CLASS]
        .sort_values(by="x_center")
        .reset_index(drop=True)
    )
    boards.index.names = ["board_index"]
    boards = boards.reset_index()

    boards = deduplicate_gdf(boards)

    if boards.shape[0] != 2:
        print("Not two board predictions")
        return None

    # Project each prediction on a [0,1] coordinate system within its respective board
    projected = boards.sjoin(
        predictions, how="inner", lsuffix="board", rsuffix="pred"
    ).apply(project_onto_board, axis="columns", result_type="expand")

    projected = projected[projected.clas.isin([CHECKER_P1_CLASS, CHECKER_P2_CLASS])]

    if projected.shape[0] == 0:
        print("No non-board predictions")
        return None
    # Drop all predictions expcept ofr the Checkers P1 and Checkers P2 classes

    projected["geometry"] = projected.apply(to_polygon, axis="columns")

    projected = gpd.GeoDataFrame(projected)

    # Now we have all Checker predictions with [0,1] x and y coordinates within their respective board along with board index
    # Count checker positions in each board half separately and then merge the board-half states
    state_board_1 = parse_half_board_state(projected[projected["board_index"] == 0])
    state_board_2 = parse_half_board_state(projected[projected["board_index"] == 1])

    full_state = {}
    # merge two states
    for x in range(1, 25):
        if x > 0 and x <= 6:
            full_state[f"Point_{x}"] = state_board_2[f"Point_{x}"]
        elif x > 6 and x <= 12:
            full_state[f"Point_{x}"] = state_board_1[f"Point_{x - 6}"]
        elif x > 12 and x <= 18:
            full_state[f"Point_{x}"] = state_board_1[f"Point_{x - 6}"]
        elif x > 18 and x <= 24:
            full_state[f"Point_{x}"] = state_board_2[f"Point_{x - 12}"]

    full_state["status"] = "VALID"

    return full_state, projected


def parse_single_prediction(prediction_path, output_path):
    predictions = parse_yolo_predictions(prediction_path)

    parse_output = parse_board_state(predictions)

    if parse_output is None:
        print("Unable to parse board state")
        board_state = {"status": "UNPARSEABLE"}
    else:
        board_state, projected_predictions = parse_output

    with output_path.open("w") as io:
        json.dump(board_state, io, indent=4)


def main():
    args = __parse_args()

    if args.predictions.is_dir():
        args.output.mkdir(exist_ok=True)

        for pred in tqdm(args.predictions.iterdir()):
            output_name = args.output / pred.with_suffix(".json").name
            parse_single_prediction(pred, output_name)
    else:
        parse_single_prediction(args.predictions, args.output)


if __name__ == "__main__":
    main()
