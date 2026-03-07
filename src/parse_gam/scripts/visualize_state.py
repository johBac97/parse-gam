import argparse
from PIL import Image
import io
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.patches as patches
from pathlib import Path

from parse_gam.models import BoardState, FrameStatus


def __parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("states", type=Path)
    args.add_argument("--frames", type=Path, default=None)
    args.add_argument("output", type=Path)
    return args.parse_args()


POINT_X = {
    **{f"Point_{i}": 3.5 - (i - 1) * 0.5 for i in range(1, 7)},
    **{f"Point_{i}": -1.0 - (i - 7) * 0.5 for i in range(7, 13)},
    **{f"Point_{i}": -3.5 + (i - 13) * 0.5 for i in range(13, 19)},
    **{f"Point_{i}": 1.0 + (i - 19) * 0.5 for i in range(19, 25)},
}

BAR_X = 0.0  # Bar is at center between the two board halves


def _draw_checker_stack(ax, x, y_start, count, color, is_top):
    """Draw a stack of checkers at a given point."""
    checker_radius = 0.2
    max_checkers = 6
    abs_count = abs(count)

    for j in range(min(abs_count, max_checkers)):
        y = y_start + (-1 if is_top else 1) * (
            j * 2 * checker_radius + checker_radius
        )
        circle = patches.Circle(
            (x, y), checker_radius, facecolor=color, edgecolor="black"
        )
        ax.add_patch(circle)

    if abs_count > max_checkers:
        y_end = 1.5 if is_top else -1.5
        y = y_end + (-1 if is_top else 1) * (
            max_checkers * 2 * checker_radius + checker_radius
        )
        ax.text(x, y, str(abs_count), ha="center", va="center", fontsize=8)


def visualize_state(state: BoardState, show=False):
    f, ax = plt.subplots(1, 1, figsize=(16, 8))

    left_board = patches.Rectangle(
        (-4.0, -2.5), 3.5, 5, linewidth=1, edgecolor="black", facecolor="none"
    )
    right_board = patches.Rectangle(
        (0.5, -2.5), 3.5, 5, linewidth=1, edgecolor="black", facecolor="none"
    )
    ax.add_patch(left_board)
    ax.add_patch(right_board)

    # Bar region
    bar_region = patches.Rectangle(
        (-0.25, -2.5), 0.5, 5, linewidth=1, edgecolor="gray",
        facecolor="lightyellow", alpha=0.5
    )
    ax.add_patch(bar_region)

    if state.status == FrameStatus.VALID:
        # Draw points and checkers
        for i, count in enumerate(state.points):
            point_name = f"Point_{i + 1}"
            if point_name not in POINT_X:
                continue
            x = POINT_X[point_name]
            is_top = (i + 1) in range(13, 25)
            y_start = 2.5 if is_top else -2.5
            y_end = 1.5 if is_top else -1.5
            plt.plot([x, x], [y_start, y_end], color="black", linewidth=1)

            abs_count = abs(count)
            if abs_count > 0:
                color = "white" if count < 0 else "black"
                _draw_checker_stack(ax, x, y_start, count, color, is_top)

        # Draw bar checkers
        if state.bar_p1 > 0:
            _draw_checker_stack(ax, BAR_X, -2.5, -state.bar_p1, "white", False)
        if state.bar_p2 > 0:
            _draw_checker_stack(ax, BAR_X, 2.5, state.bar_p2, "black", True)

        # Borne off display
        off_x = 4.5
        if state.off_p1 > 0:
            ax.text(off_x, -1.5, f"P1 off: {state.off_p1}",
                    ha="center", va="center", fontsize=10, color="gray",
                    bbox=dict(facecolor="white", edgecolor="gray", alpha=0.7))
        if state.off_p2 > 0:
            ax.text(off_x, 1.5, f"P2 off: {state.off_p2}",
                    ha="center", va="center", fontsize=10, color="gray",
                    bbox=dict(facecolor="black", edgecolor="gray", alpha=0.7),
                    fontdict={"color": "white"})

        # Dice display
        dice_text_parts = []
        if state.dice_values:
            dice_str = "-".join(str(v) for v in state.dice_values)
            dice_text_parts.append(f"Dice: {dice_str}")
        elif state.board_1_dice > 0 or state.board_2_dice > 0:
            dice_text_parts.append(
                f"Dice: L={state.board_1_dice} R={state.board_2_dice}"
            )

        if state.dice_board_half is not None:
            side = "Left" if state.dice_board_half == 0 else "Right"
            dice_text_parts.append(f"({side} board)")

        if dice_text_parts:
            ax.text(0, -3.2, " ".join(dice_text_parts),
                    ha="center", va="center", fontsize=10,
                    bbox=dict(facecolor="lightyellow", edgecolor="orange", alpha=0.8))

        # Cube display
        if state.cube_value is not None:
            cube_label = str(state.cube_value)
            cube_pos_label = (state.cube_position.value
                              if state.cube_position else "?")
            ax.text(-4.5, 0, f"Cube: {cube_label}\n({cube_pos_label})",
                    ha="center", va="center", fontsize=9,
                    bbox=dict(facecolor="lightyellow", edgecolor="brown", alpha=0.8))

    else:
        ax.text(
            x=-1.0,
            y=0,
            s="UNABLE TO PARSE BOARD",
            fontsize=40,
            color="red",
            ha="center",
            va="center",
            zorder=10,
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="black"),
        )

    # Frame index label
    if state.file_index is not None:
        ax.text(-4.5, 2.8, f"Frame: {state.file_index}",
                ha="center", va="center", fontsize=8, color="gray")

    ax.set_xlim(-5, 5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")

    if show:
        plt.show()
        return f, ax
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        plt.close(f)
        buf.seek(0)
        return buf


def join_state_and_frame_visualizations(state_vis, frame_path, output_path):
    state_img = Image.open(state_vis)
    frame_img = Image.open(frame_path)

    new_height = frame_img.height
    state_img = state_img.resize(
        (int(state_img.width * new_height / state_img.height), new_height)
    )
    frame_img = frame_img.resize(
        (int(frame_img.width * new_height / frame_img.height), new_height)
    )

    total_width = state_img.width + frame_img.width
    combined_img = Image.new("RGB", (total_width, new_height))
    combined_img.paste(frame_img, (0, 0))
    combined_img.paste(state_img, (frame_img.width, 0))
    combined_img.save(output_path)


def visualize_single_state(state_path, frame_path, output_path):
    state = BoardState.load(state_path)

    if frame_path is not None and not frame_path.exists():
        raise ValueError("No such frame file exists")

    state_vis = visualize_state(state)

    if frame_path is None:
        with output_path.open("wb") as f:
            f.write(state_vis.read())
    else:
        join_state_and_frame_visualizations(state_vis, frame_path, output_path)


def main():
    args = __parse_args()

    if args.states.is_dir():
        if not args.frames.is_dir():
            raise ValueError("State is directory but frames is not.")

        args.output.mkdir(exist_ok=True)

        for state_path in tqdm(sorted(args.states.iterdir())):
            frame_path = args.frames / state_path.with_suffix(".jpg").name

            frame_index = int(state_path.stem.split("_")[-1])
            output_path = args.output / f"vis_{frame_index:04d}.jpg"

            visualize_single_state(state_path, frame_path, output_path)
    else:
        visualize_single_state(args.states, args.frames, args.output)


if __name__ == "__main__":
    main()
