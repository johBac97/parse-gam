from pathlib import Path
import json
import logging
import multiprocessing
import argparse

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from parse_gam.utils import (
    deduplicate_gdf,
    to_polygon,
    project_onto_board,
    parse_yolo_predictions,
)
from parse_gam.models import (
    BoardState,
    FrameStatus,
    CLASS_MAPPING,
)

log = logging.getLogger(__name__)


def __parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("predictions", type=Path)
    args.add_argument("output", type=Path)
    args.add_argument("--num-proc", type=int, default=4)
    return args.parse_args()


def parse_half_board_state(gdf):
    """Parse a half-board's state.

    Assumes x and y center columns are coordinates within [0,1] of the
    center of the checker position. Divides the board into 6 Backgammon
    'Points' on the upper and lower sides.
    """
    state = {}

    CHECKER_POINT_X_TOLERANCE = 0.07
    CHECKER_P1_CLASS = CLASS_MAPPING["CHECKER_P1"]
    CHECKER_P2_CLASS = CLASS_MAPPING["CHECKER_P2"]

    for half in ["UPPER", "LOWER"]:
        for h_point_index in range(0, 6):
            h_pos = h_point_index / 6 + (1 / 6 / 2)
            if half == "LOWER":
                d = gdf[
                    (gdf.y_center > 0.5)
                    & ((gdf.x_center - h_pos).abs() < CHECKER_POINT_X_TOLERANCE)
                    & (gdf.clas.isin([CHECKER_P1_CLASS, CHECKER_P2_CLASS]))
                ].reset_index(drop=True)
                point_index = 6 - h_point_index
            else:
                d = gdf[
                    (gdf.y_center < 0.5)
                    & ((gdf.x_center - h_pos).abs() < CHECKER_POINT_X_TOLERANCE)
                    & (gdf.clas.isin([CHECKER_P1_CLASS, CHECKER_P2_CLASS]))
                ].reset_index(drop=True)
                point_index = 7 + h_point_index

            class_counts = d.groupby("clas").size().reset_index(name="count")

            top_classes = class_counts.nlargest(1, "count", keep="all")

            if top_classes.empty:
                selected_class_index = CLASS_MAPPING["CHECKER_P2"]
            elif len(top_classes) > 1 and top_classes["count"].nunique() == 1:
                mean_confidence = d.groupby("clas")["conf"].mean()
                selected_class_index = mean_confidence.idxmax()
            else:
                selected_class_index = int(top_classes["clas"].iloc[0])

            num_checkers = deduplicate_gdf(d).shape[0]
            val = (
                num_checkers
                if selected_class_index == CLASS_MAPPING["CHECKER_P2"]
                else -num_checkers
            )

            state[f"Point_{point_index}"] = val

    # Dice detection: count how many dice are visible on this board half.
    DIE_CLASS = CLASS_MAPPING["DIE"]
    die_detections = gdf[gdf.clas == DIE_CLASS]
    num_dice = deduplicate_gdf(die_detections, iou_threshold=0.4).shape[0]

    state["dice"] = num_dice

    return state


def _detect_bar_checkers(predictions, boards):
    """Find checkers in the bar region between the two board halves."""
    CHECKER_P1_CLASS = CLASS_MAPPING["CHECKER_P1"]
    CHECKER_P2_CLASS = CLASS_MAPPING["CHECKER_P2"]

    board_left = boards.iloc[0]
    board_right = boards.iloc[1]

    bar_x_left = board_left.x_center + board_left.width / 2
    bar_x_right = board_right.x_center - board_right.width / 2

    all_checkers = predictions[
        predictions.clas.isin([CHECKER_P1_CLASS, CHECKER_P2_CLASS])
    ]

    bar_mask = (all_checkers.x_center >= bar_x_left) & (
        all_checkers.x_center <= bar_x_right
    )
    bar_checkers = all_checkers[bar_mask]
    bar_checkers = deduplicate_gdf(bar_checkers)

    bar_p1 = int((bar_checkers.clas == CHECKER_P1_CLASS).sum())
    bar_p2 = int((bar_checkers.clas == CHECKER_P2_CLASS).sum())

    return bar_p1, bar_p2


def _detect_dice_board_half(predictions, boards):
    """Determine which board half contains the dice."""
    DIE_CLASS = CLASS_MAPPING["DIE"]
    dice = predictions[predictions.clas == DIE_CLASS]

    if dice.empty:
        return None

    board_left = boards.iloc[0]
    board_right = boards.iloc[1]
    bar_center = (
        board_left.x_center + board_left.width / 2
        + board_right.x_center - board_right.width / 2
    ) / 2

    dice_on_left = (dice.x_center < bar_center).sum()
    dice_on_right = (dice.x_center >= bar_center).sum()

    if dice_on_left > dice_on_right:
        return 0
    elif dice_on_right > 0:
        return 1
    return None


def _detect_cube(predictions, boards):
    """Detect doubling cube value and position.

    Returns (cube_value, cube_position) or (None, None) if no cube detected.
    Cube value classification requires a separate classifier model — this
    function only determines spatial position. The value is set to None
    until a cube classifier is integrated.
    """
    CUBE_CLASS = CLASS_MAPPING["DOUBLING_CUBE"]
    cubes = predictions[predictions.clas == CUBE_CLASS]

    if cubes.empty:
        return None, None

    cube = cubes.iloc[0]  # take highest-confidence if multiple

    board_left = boards.iloc[0]
    board_right = boards.iloc[1]
    left_edge = board_left.x_center - board_left.width / 2
    right_edge = board_right.x_center + board_right.width / 2
    bar_center = (
        board_left.x_center + board_left.width / 2
        + board_right.x_center - board_right.width / 2
    ) / 2

    if cube.x_center < left_edge:
        position = "p1"
    elif cube.x_center > right_edge:
        position = "p2"
    else:
        position = "center"

    # Value is None until a cube classifier is integrated
    return None, position


def parse_board_state(predictions: gpd.GeoDataFrame) -> BoardState:
    BOARD_CLASS = CLASS_MAPPING["BOARD"]
    CHECKER_P1_CLASS = CLASS_MAPPING["CHECKER_P1"]
    CHECKER_P2_CLASS = CLASS_MAPPING["CHECKER_P2"]
    HAND_CLASS = CLASS_MAPPING["HAND"]
    DIE_CLASS = CLASS_MAPPING["DIE"]

    boards = (
        predictions[predictions.clas == BOARD_CLASS]
        .sort_values(by="x_center")
        .reset_index(drop=True)
    )
    boards.index.names = ["board_index"]
    boards = boards.reset_index()
    boards = deduplicate_gdf(boards)

    if boards.shape[0] != 2:
        log.debug("Expected 2 boards, got %d", boards.shape[0])
        return BoardState.unparseable()

    # Bar detection
    bar_p1, bar_p2 = _detect_bar_checkers(predictions, boards)

    # Dice board half
    dice_board_half = _detect_dice_board_half(predictions, boards)

    # Doubling cube
    cube_value, cube_position = _detect_cube(predictions, boards)

    # Project each prediction onto [0,1] coordinate system within its respective board
    projected = boards.sjoin(
        predictions, how="inner", lsuffix="board", rsuffix="pred"
    ).apply(project_onto_board, axis="columns", result_type="expand")

    all_relevant = {CHECKER_P1_CLASS, CHECKER_P2_CLASS, HAND_CLASS, DIE_CLASS}
    projected = projected[projected.clas.isin(all_relevant)]

    if projected.shape[0] == 0:
        log.debug("No non-board predictions found")
        return BoardState.unparseable()

    projected["geometry"] = projected.apply(to_polygon, axis="columns")
    projected = gpd.GeoDataFrame(projected)

    if (projected.clas == HAND_CLASS).sum() > 0:
        log.debug("Hand detected, frame obscured")
        return BoardState.obscured()

    state_board_1 = parse_half_board_state(projected[projected["board_index"] == 0])
    state_board_2 = parse_half_board_state(projected[projected["board_index"] == 1])

    # Merge two half-board states into full 24-point board
    points = [0] * 24
    for x in range(1, 25):
        if 1 <= x <= 6:
            points[x - 1] = state_board_2[f"Point_{x}"]
        elif 7 <= x <= 12:
            points[x - 1] = state_board_1[f"Point_{x - 6}"]
        elif 13 <= x <= 18:
            points[x - 1] = state_board_1[f"Point_{x - 6}"]
        elif 19 <= x <= 24:
            points[x - 1] = state_board_2[f"Point_{x - 12}"]

    # Borne off: 15 checkers per player minus visible ones
    p1_on_points = sum(abs(p) for p in points if p < 0)
    p2_on_points = sum(abs(p) for p in points if p > 0)
    off_p1 = max(0, 15 - p1_on_points - bar_p1)
    off_p2 = max(0, 15 - p2_on_points - bar_p2)

    from parse_gam.models import CubePosition

    return BoardState(
        points=points,
        status=FrameStatus.VALID,
        bar_p1=bar_p1,
        bar_p2=bar_p2,
        off_p1=off_p1,
        off_p2=off_p2,
        board_1_dice=state_board_1["dice"],
        board_2_dice=state_board_2["dice"],
        dice_board_half=dice_board_half,
        cube_value=cube_value,
        cube_position=CubePosition(cube_position) if cube_position else None,
    )


def parse_single_prediction(prediction_path: Path, output_path: Path):
    predictions = parse_yolo_predictions(prediction_path)
    board_state = parse_board_state(predictions)

    # Extract file index from filename (e.g. "frame_0042.txt" -> 42)
    try:
        file_index = int(prediction_path.stem.split("_")[-1])
    except ValueError:
        log.warning("Could not extract file index from %s", prediction_path.name)
        file_index = None

    board_state.file_index = file_index
    board_state.save(output_path)


def _process_file(args):
    pred, output_dir = args
    output_name = output_dir / pred.with_suffix(".json").name
    parse_single_prediction(pred, output_name)
    return pred


def main():
    logging.basicConfig(level=logging.WARNING)
    args = __parse_args()

    if args.predictions.is_dir():
        args.output.mkdir(exist_ok=True)
        all_preds = list(args.predictions.iterdir())
        if args.num_proc > 1:
            with multiprocessing.Pool(processes=args.num_proc) as pool:
                list(
                    tqdm(
                        pool.imap_unordered(
                            _process_file,
                            [(pred, args.output) for pred in all_preds],
                            chunksize=max(1, len(all_preds) // (args.num_proc * 20)),
                        ),
                        total=len(all_preds),
                        desc="Parsing predictions",
                    )
                )
        else:
            for pred in tqdm(all_preds, desc="Parsing predictions"):
                _process_file((pred, args.output))
    else:
        parse_single_prediction(args.predictions, args.output)


if __name__ == "__main__":
    main()
