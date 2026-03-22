from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


class FrameStatus(str, Enum):
    VALID = "VALID"
    UNPARSEABLE = "UNPARSEABLE"
    OBSCURED = "OBSCURED"


class CubePosition(str, Enum):
    CENTER = "center"
    P1 = "p1"
    P2 = "p2"


# YOLO class indices — matches data/datasets/version2/data.yaml (train5 model)
CLASS_MAPPING = {
    "BOARD": 0,
    "CHECKER_P1": 1,
    "CHECKER_P2": 2,
    "DOUBLING_CUBE": 3,
    "DIE": 4,
    "HAND": 5,
    "POINT": 6,
}

POINT_COLUMNS = [f"Point_{i}" for i in range(1, 25)]

# Standard backgammon starting position.
# Positive = P2, negative = P1.
STARTING_POSITION = [
    -2, 0, 0, 0, 0, 5,    # Points 1-6
    0, 3, 0, 0, 0, -5,    # Points 7-12
    5, 0, 0, 0, -3, 0,    # Points 13-18
    -5, 0, 0, 0, 0, 2,    # Points 19-24
]


@dataclass
class BoardState:
    """Parsed backgammon board state from a single video frame.

    Point values: positive = player 2 (black), negative = player 1 (white).
    """

    points: list[int] = field(default_factory=lambda: [0] * 24)
    status: FrameStatus = FrameStatus.VALID
    file_index: int | None = None

    # Bar
    bar_p1: int = 0
    bar_p2: int = 0

    # Borne off (inferred: 15 - on_points - on_bar)
    off_p1: int = 0
    off_p2: int = 0

    # Dice
    board_1_dice: int = 0
    board_2_dice: int = 0
    dice_values: list[int] = field(default_factory=list)
    dice_board_half: int | None = None  # 0 = left board, 1 = right board

    # Doubling cube
    cube_value: int | None = None
    cube_position: CubePosition | None = None

    def to_dict(self) -> dict:
        d = {f"Point_{i+1}": self.points[i] for i in range(24)}
        d["bar_p1"] = self.bar_p1
        d["bar_p2"] = self.bar_p2
        d["off_p1"] = self.off_p1
        d["off_p2"] = self.off_p2
        d["board_1_dice"] = self.board_1_dice
        d["board_2_dice"] = self.board_2_dice
        d["dice_values"] = self.dice_values
        d["dice_board_half"] = self.dice_board_half
        d["cube_value"] = self.cube_value
        d["cube_position"] = self.cube_position.value if self.cube_position else None
        d["status"] = self.status.value
        d["file_index"] = self.file_index
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BoardState":
        points = [d.get(f"Point_{i}", 0) for i in range(1, 25)]
        cube_pos = d.get("cube_position")
        return cls(
            points=points,
            status=FrameStatus(d.get("status", "UNPARSEABLE")),
            file_index=d.get("file_index"),
            bar_p1=d.get("bar_p1", 0),
            bar_p2=d.get("bar_p2", 0),
            off_p1=d.get("off_p1", 0),
            off_p2=d.get("off_p2", 0),
            board_1_dice=d.get("board_1_dice", 0),
            board_2_dice=d.get("board_2_dice", 0),
            dice_values=d.get("dice_values", []),
            dice_board_half=d.get("dice_board_half"),
            cube_value=d.get("cube_value"),
            cube_position=CubePosition(cube_pos) if cube_pos else None,
        )

    def save(self, path: Path):
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, path: Path) -> "BoardState":
        with path.open() as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def unparseable(cls, file_index: int | None = None) -> "BoardState":
        return cls(status=FrameStatus.UNPARSEABLE, file_index=file_index)

    @classmethod
    def obscured(cls, file_index: int | None = None) -> "BoardState":
        return cls(status=FrameStatus.OBSCURED, file_index=file_index)

    def is_starting_position(self) -> bool:
        return self.points == STARTING_POSITION

    @property
    def has_dice(self) -> bool:
        return self.board_1_dice > 0 or self.board_2_dice > 0

    @property
    def total_visible_p1(self) -> int:
        return sum(abs(p) for p in self.points if p < 0) + self.bar_p1

    @property
    def total_visible_p2(self) -> int:
        return sum(abs(p) for p in self.points if p > 0) + self.bar_p2


@dataclass
class Turn:
    """A single player turn: dice roll + resulting board change."""
    player: int  # 1 or 2 (inferred from dice_board_half)
    dice: list[int]
    state_before: dict
    state_after: dict
    frame_start: int
    frame_end: int
    cube_action: str | None = None  # "double", "take", "drop"

    def to_dict(self) -> dict:
        return {
            "player": self.player,
            "dice": self.dice,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "cube_action": self.cube_action,
        }


@dataclass
class GameRecord:
    """A complete parsed backgammon game."""
    turns: list[Turn] = field(default_factory=list)
    result: str = "unknown"  # "single", "gammon", "backgammon", "drop", "unknown"
    winner: int | None = None  # 1 or 2
    cube_final: int = 1
    start_frame: int = 0
    end_frame: int = 0

    def to_dict(self) -> dict:
        return {
            "turns": [t.to_dict() for t in self.turns],
            "result": self.result,
            "winner": self.winner,
            "cube_final": self.cube_final,
            "points_won": self.points_won,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }

    @property
    def points_won(self) -> int:
        multiplier = {"single": 1, "gammon": 2, "backgammon": 3, "drop": 1}
        return multiplier.get(self.result, 0) * self.cube_final

    def save(self, path: Path):
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)
