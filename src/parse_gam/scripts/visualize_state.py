"""Visualize parsed backgammon board states alongside source frames."""

import argparse
import io
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from parse_gam.models import BoardState, FrameStatus


def __parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("states", type=Path)
    args.add_argument("output", type=Path)
    args.add_argument("--frames", type=Path, default=None)
    return args.parse_args()


# ── Board geometry ────────────────────────────────────────────────────────────

POINT_X = {
    **{f"Point_{i}": 3.5 - (i - 1) * 0.5 for i in range(1, 7)},
    **{f"Point_{i}": -1.0 - (i - 7) * 0.5 for i in range(7, 13)},
    **{f"Point_{i}": -3.5 + (i - 13) * 0.5 for i in range(13, 19)},
    **{f"Point_{i}": 1.0 + (i - 19) * 0.5 for i in range(19, 25)},
}

BAR_X = 0.0
CHECKER_R = 0.18
MAX_STACK = 5

# Alternating point triangle colors (odd/even)
_TRI = ["#8B1A1A", "#1C3A5A"]


def _is_top(point_num: int) -> bool:
    return point_num >= 13


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_board_skeleton(ax):
    """Boards, bar, triangles, point labels, home highlights, player legend."""

    # Home board tints: P1 home = 19-24 (top-right), P2 home = 1-6 (bottom-right)
    ax.add_patch(patches.Rectangle((0.75, 1.5), 3.25, 1.0,
                                   facecolor="#ffe8e8", edgecolor="none", zorder=0))
    ax.add_patch(patches.Rectangle((0.75, -2.5), 3.25, 1.0,
                                   facecolor="#e8e8ff", edgecolor="none", zorder=0))

    # Board surfaces
    for x0, w in [(-4.0, 3.5), (0.5, 3.5)]:
        ax.add_patch(patches.Rectangle((x0, -2.5), w, 5.0,
                                       linewidth=1.5, edgecolor="black",
                                       facecolor="#F5DEB3", zorder=1))

    # Bar
    ax.add_patch(patches.Rectangle((-0.26, -2.5), 0.52, 5.0,
                                   linewidth=1, edgecolor="#888",
                                   facecolor="#C8A96E", zorder=1))
    ax.text(BAR_X, 0, "BAR", ha="center", va="center",
            fontsize=7, color="#555", fontweight="bold", zorder=6)

    # Point triangles + number labels
    for name, x in POINT_X.items():
        pnum = int(name.split("_")[1])
        top = _is_top(pnum)
        color = _TRI[(pnum - 1) % 2]

        if top:
            tri = patches.Polygon(
                [[x - 0.22, 2.5], [x + 0.22, 2.5], [x, 1.5]],
                facecolor=color, edgecolor="none", alpha=0.6, zorder=2,
            )
            label_y = 2.73
        else:
            tri = patches.Polygon(
                [[x - 0.22, -2.5], [x + 0.22, -2.5], [x, -1.5]],
                facecolor=color, edgecolor="none", alpha=0.6, zorder=2,
            )
            label_y = -2.73
        ax.add_patch(tri)
        ax.text(x, label_y, str(pnum), ha="center", va="center",
                fontsize=7, fontweight="bold", color="#222", zorder=6)

    # Home board labels
    ax.text(2.25, 2.85, "P1 home", ha="center", va="center",
            fontsize=7, color="#a00", style="italic")
    ax.text(2.25, -2.85, "P2 home", ha="center", va="center",
            fontsize=7, color="#00a", style="italic")

    # Player movement legend (below board)
    ax.text(-2.0, -3.25, "← P1 (white)  moves 1 → 24 →", ha="center", va="center",
            fontsize=8, color="#600")
    ax.text(2.0, 3.25, "← P2 (black)  moves 24 → 1 →", ha="center", va="center",
            fontsize=8, color="#006")


def _draw_stack(ax, x, y_start, count, is_top):
    """Draw checker stack. Sign of count determines color (neg=white/P1, pos=black/P2)."""
    abs_count = abs(count)
    face = "white" if count < 0 else "#1a1a1a"
    label_c = "#000" if count < 0 else "#fff"

    drawn = min(abs_count, MAX_STACK)
    last_y = y_start
    for j in range(drawn):
        y = y_start + (-1 if is_top else 1) * (j * 2 * CHECKER_R + CHECKER_R)
        last_y = y
        ax.add_patch(patches.Circle(
            (x, y), CHECKER_R,
            facecolor=face, edgecolor="black", linewidth=0.8, zorder=4,
        ))

    # Count label on outermost checker (always when > 1)
    if abs_count > 1:
        ax.text(x, last_y, str(abs_count), ha="center", va="center",
                fontsize=7, fontweight="bold", color=label_c, zorder=5)

    # Overflow indicator
    if abs_count > MAX_STACK:
        extra_y = last_y + (-1 if is_top else 1) * 0.28
        ax.text(x, extra_y, f"+{abs_count - MAX_STACK}", ha="center", va="center",
                fontsize=6, color="#c00", zorder=5)


def _draw_dice(ax, values: list[int], cx: float, cy: float):
    """Draw each die value as a small labeled square."""
    size = 0.38
    gap = 0.08
    n = len(values)
    total_w = n * size + (n - 1) * gap
    x0 = cx - total_w / 2

    for i, val in enumerate(values):
        dx = x0 + i * (size + gap) + size / 2
        ax.add_patch(patches.FancyBboxPatch(
            (dx - size / 2, cy - size / 2), size, size,
            boxstyle="round,pad=0.04",
            facecolor="ivory", edgecolor="#333", linewidth=1.5, zorder=10,
        ))
        ax.text(dx, cy, str(val), ha="center", va="center",
                fontsize=13, fontweight="bold", zorder=11)


def _pip_count(state: BoardState) -> tuple[int, int]:
    """Total pip distance remaining for each player."""
    p1 = sum((25 - (i + 1)) * abs(v) for i, v in enumerate(state.points) if v < 0)
    p1 += state.bar_p1 * 25
    p2 = sum((i + 1) * v for i, v in enumerate(state.points) if v > 0)
    p2 += state.bar_p2 * 25
    return p1, p2


# ── Main visualizer ───────────────────────────────────────────────────────────

def visualize_state(state: BoardState, show=False):
    f, ax = plt.subplots(figsize=(16, 8))
    _draw_board_skeleton(ax)

    if state.status == FrameStatus.VALID:
        # ── Checkers on points ──
        for i, count in enumerate(state.points):
            if count == 0:
                continue
            pnum = i + 1
            x = POINT_X[f"Point_{pnum}"]
            top = _is_top(pnum)
            _draw_stack(ax, x, 2.5 if top else -2.5, count, top)

        # ── Bar ──
        if state.bar_p1 > 0:
            _draw_stack(ax, BAR_X, -2.5, -state.bar_p1, is_top=False)
        if state.bar_p2 > 0:
            _draw_stack(ax, BAR_X, 2.5, state.bar_p2, is_top=True)

        # ── Borne off ──
        if state.off_p1 > 0:
            ax.text(4.8, -2.0, f"P1\noff\n{state.off_p1}", ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="black",
                              boxstyle="round", linewidth=1.5))
        if state.off_p2 > 0:
            ax.text(4.8, 2.0, f"P2\noff\n{state.off_p2}", ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white",
                    bbox=dict(facecolor="#1a1a1a", edgecolor="black",
                              boxstyle="round", linewidth=1.5))

        # ── Dice ──
        if state.dice_values:
            side = ("Left" if state.dice_board_half == 0
                    else "Right" if state.dice_board_half == 1
                    else "?")
            _draw_dice(ax, state.dice_values, 0, -3.1)
            ax.text(0, -3.42, f"dice on {side} board", ha="center", va="center",
                    fontsize=7, color="#555")
        elif state.board_1_dice > 0 or state.board_2_dice > 0:
            side = ("Left" if state.dice_board_half == 0
                    else "Right" if state.dice_board_half == 1
                    else "?")
            ax.text(0, -3.1,
                    f"Dice visible — no value  ({side} board)",
                    ha="center", va="center", fontsize=9,
                    bbox=dict(facecolor="lightyellow", edgecolor="orange",
                              boxstyle="round"))

        # ── Pip counts ──
        p1_pips, p2_pips = _pip_count(state)
        ax.text(-4.8, -1.8, f"P1 pips\n{p1_pips}", ha="center", va="center",
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="#888",
                          boxstyle="round", linewidth=1))
        ax.text(-4.8, 1.8, f"P2 pips\n{p2_pips}", ha="center", va="center",
                fontsize=8, color="white",
                bbox=dict(facecolor="#333", edgecolor="#888",
                          boxstyle="round", linewidth=1))

        # ── Checker total sanity check ──
        p1_total = (sum(abs(p) for p in state.points if p < 0)
                    + state.bar_p1 + state.off_p1)
        p2_total = (sum(p for p in state.points if p > 0)
                    + state.bar_p2 + state.off_p2)
        ok = p1_total == 15 and p2_total == 15
        ax.text(-4.8, 0, f"P1: {p1_total}  P2: {p2_total}",
                ha="center", va="center", fontsize=7,
                color="green" if ok else "red",
                bbox=dict(facecolor="white",
                          edgecolor="green" if ok else "red",
                          boxstyle="round", linewidth=1))

        # ── Cube ──
        if state.cube_value is not None:
            pos = state.cube_position.value if state.cube_position else "?"
            ax.text(-4.8, 0.8, f"Cube: {state.cube_value}\n({pos})",
                    ha="center", va="center", fontsize=8,
                    bbox=dict(facecolor="lightyellow", edgecolor="brown",
                              boxstyle="round"))

    else:
        msg = ("HAND / OBSCURED"
               if state.status == FrameStatus.OBSCURED
               else "CANNOT PARSE")
        ax.text(0, 0, msg, ha="center", va="center",
                fontsize=28, color="red", fontweight="bold", zorder=20,
                bbox=dict(facecolor="white", edgecolor="red",
                          alpha=0.9, boxstyle="round"))

    # ── Frame / status badge ──
    badge_color = {
        FrameStatus.VALID: "green",
        FrameStatus.OBSCURED: "orange",
        FrameStatus.UNPARSEABLE: "red",
    }.get(state.status, "gray")
    frame_label = (f"Frame {state.file_index}"
                   if state.file_index is not None else "Frame ?")
    ax.text(4.8, 3.3, f"{frame_label}\n{state.status.value}",
            ha="center", va="center", fontsize=8, color="white",
            bbox=dict(facecolor=badge_color, edgecolor="none",
                      boxstyle="round,pad=0.4"))

    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-3.7, 3.7)
    ax.set_aspect("equal")
    ax.axis("off")

    if show:
        plt.show()
        return f, ax

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(f)
    buf.seek(0)
    return buf


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _join(state_buf, frame_path: Path, output_path: Path):
    state_img = Image.open(state_buf)
    frame_img = Image.open(frame_path)

    h = frame_img.height
    state_img = state_img.resize(
        (int(state_img.width * h / state_img.height), h)
    )
    frame_img = frame_img.resize(
        (int(frame_img.width * h / frame_img.height), h)
    )

    out = Image.new("RGB", (frame_img.width + state_img.width, h))
    out.paste(frame_img, (0, 0))
    out.paste(state_img, (frame_img.width, 0))
    out.save(output_path)


def visualize_single_state(state_path, frame_path, output_path):
    state = BoardState.load(state_path)
    buf = visualize_state(state)
    if frame_path is None or not frame_path.exists():
        with output_path.open("wb") as f:
            f.write(buf.read())
    else:
        _join(buf, frame_path, output_path)


def main():
    args = __parse_args()

    if args.states.is_dir():
        args.output.mkdir(exist_ok=True)
        for state_path in tqdm(sorted(args.states.iterdir())):
            if state_path.suffix != ".json":
                continue
            frame_path = (
                args.frames / state_path.with_suffix(".jpg").name
                if args.frames else None
            )
            frame_index = int(state_path.stem.split("_")[-1])
            visualize_single_state(
                state_path, frame_path,
                args.output / f"vis_{frame_index:04d}.jpg",
            )
    else:
        visualize_single_state(args.states, args.frames, args.output)


if __name__ == "__main__":
    main()
