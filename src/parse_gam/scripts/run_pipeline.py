"""End-to-end pipeline: YOLO detection -> parse -> smooth -> detect moves -> game parse -> visualize -> video."""

import argparse
import subprocess
import sys
from pathlib import Path


STEPS = ["detect", "parse", "smooth", "moves", "game", "visualize", "video"]


def __parse_args():
    parser = argparse.ArgumentParser(
        description="Run the backgammon parsing pipeline.",
    )
    parser.add_argument("model", type=Path, help="Path to YOLO weights")
    parser.add_argument("source", type=Path, help="Path to input video")
    parser.add_argument("name", type=str, help="Output folder name (created in current directory)")
    parser.add_argument(
        "--from-step",
        choices=STEPS,
        default="detect",
        help="Start from this step (skip earlier steps). Default: detect",
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--framerate", type=int, default=4, help="Output video FPS")
    parser.add_argument("--move-window", type=int, default=10)
    parser.add_argument(
        "--die-classifier", type=Path, default=None,
        help="Path to YOLO classify weights for die pip values",
    )
    return parser.parse_args()


def run(cmd: list[str], description: str):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"FAILED: {description} (exit code {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


def find_frames_dir(root: Path) -> Path:
    # YOLO saves frames inside the yolo/ subfolder
    yolo_dir = root / "yolo"
    return yolo_dir


def main():
    args = __parse_args()

    root_dir = Path(args.name)
    yolo_dir = root_dir / "yolo"
    labels_dir = yolo_dir / "labels"
    states_dir = root_dir / "states"
    smooth_dir = root_dir / "smooth_states"
    moves_path = root_dir / "moves.json"
    game_path = root_dir / "game.json"
    vis_dir = root_dir / "vis"
    video_path = root_dir / "visualization.mp4"

    start_idx = STEPS.index(args.from_step)

    # 1. YOLO detection — outputs go into {name}/yolo/
    if start_idx <= STEPS.index("detect"):
        root_dir.mkdir(parents=True, exist_ok=True)
        run(
            [
                "uv", "run", "yolo",
                "predict", "detect",
                f"model={args.model}",
                f"source={args.source}",
                f"project={root_dir}",
                "name=yolo",
                "save_txt=true",
                "save_frames=true",
                "save_conf=true",
                "show_labels=false",
                f"imgsz={args.imgsz}",
            ],
            "Running YOLO detection",
        )

    # 2. Parse predictions to board states
    if start_idx <= STEPS.index("parse"):
        states_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = find_frames_dir(root_dir)
        parse_cmd = [
            "uv", "run", "parse-yolo-predictions",
            str(labels_dir), str(states_dir),
            "--frames", str(frames_dir),
        ]
        if args.die_classifier is not None:
            parse_cmd += ["--die-classifier", str(args.die_classifier)]
        run(parse_cmd, "Parsing YOLO predictions to board states")

    # 3. Smooth states
    if start_idx <= STEPS.index("smooth"):
        run(
            ["uv", "run", "smooth-frame-predictions", str(states_dir), str(smooth_dir)],
            "Smoothing frame predictions",
        )

    # 4. Detect moves
    if start_idx <= STEPS.index("moves"):
        run(
            [
                "uv", "run", "detect-moves",
                str(smooth_dir),
                str(moves_path),
                "--move-window", str(args.move_window),
            ],
            "Detecting moves",
        )

    # 5. Parse game structure (turns, boundaries, results)
    if start_idx <= STEPS.index("game"):
        run(
            [
                "uv", "run", "parse-game",
                str(smooth_dir),
                str(game_path),
            ],
            "Parsing game structure",
        )

    # 6. Visualize
    if start_idx <= STEPS.index("visualize"):
        vis_dir.mkdir(parents=True, exist_ok=True)
        run(
            [
                "uv", "run", "visualize-state",
                str(states_dir), str(vis_dir),
                "--frames", str(frames_dir),
            ],
            "Generating visualizations",
        )

    # 7. Create video
    if start_idx <= STEPS.index("video"):
        run(
            [
                "ffmpeg", "-y",
                "-framerate", str(args.framerate),
                "-i", str(vis_dir / "vis_%04d.jpg"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(video_path),
            ],
            "Creating output video",
        )

    print(f"\nPipeline complete. Output: {root_dir.resolve()}")
    if game_path.exists():
        print(f"Game record: {game_path}")
    if moves_path.exists():
        print(f"Moves: {moves_path}")
    if video_path.exists():
        print(f"Video: {video_path}")


if __name__ == "__main__":
    main()
