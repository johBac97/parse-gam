"""Reconstruct checker moves from before/after board state diffs.

Convention
----------
- Points are numbered 1-24.
- Player 1 (negative checkers) moves in the *increasing* direction (1 → 24).
  Home board: 19-24.  Bears off past point 24.
- Player 2 (positive checkers) moves in the *decreasing* direction (24 → 1).
  Home board: 1-6.  Bears off past point 1.
- BAR (0): special source — checker is on the bar and must enter.
- OFF (25): special destination — checker is borne off.
"""

BAR = 0
OFF = 25


# ── Internals ─────────────────────────────────────────────────────────────────

def _counts(points: list[int], player: int) -> list[int]:
    """Number of `player`'s checkers at each point (index = point - 1)."""
    if player == 1:
        return [abs(p) if p < 0 else 0 for p in points]
    return [p if p > 0 else 0 for p in points]


def _expected_dest(player: int, src: int, die: int) -> int:
    """Where does a checker land from `src` using `die`?"""
    if src == BAR:
        # P1 enters at the die value; P2 enters at 25 - die value.
        return die if player == 1 else (25 - die)
    if player == 1:
        dest = src + die
        return OFF if dest > 24 else dest
    else:
        dest = src - die
        return OFF if dest < 1 else dest


def _nearest_dest(player: int, src: int, dests: list[int]) -> int | None:
    """Best-effort fallback when no die value matches."""
    if not dests:
        return None
    if player == 1:
        forward = [d for d in dests if d == OFF or d > src]
        return min(forward) if forward else dests[0]
    else:
        forward = [d for d in dests if d == OFF or d < src]
        return max(forward) if forward else dests[-1]


def _match(
    player: int,
    sources: list[int],
    dests: list[int],
    dice: list[int],
) -> list[tuple[int, int]]:
    """Greedy (source-order) assignment of dice to (src, dest) pairs."""
    rem_dice = list(dice)
    rem_dests = list(dests)
    moves: list[tuple[int, int]] = []

    for src in sources:
        matched = False
        for di, d in enumerate(rem_dice):
            expected = _expected_dest(player, src, d)
            if expected in rem_dests:
                moves.append((src, expected))
                rem_dests.remove(expected)
                rem_dice.pop(di)
                matched = True
                break

        if not matched:
            # Dice values unavailable or missing — best-effort guess.
            dest = _nearest_dest(player, src, rem_dests)
            if dest is not None:
                moves.append((src, dest))
                rem_dests.remove(dest)

    return moves


# ── Public API ────────────────────────────────────────────────────────────────

def reconstruct_moves(
    player: int,
    before_points: list[int],
    after_points: list[int],
    before_bar: int,
    after_bar: int,
    before_off: int,
    after_off: int,
    dice: list[int],
) -> list[tuple[int, int]]:
    """Infer checker moves from a board state diff.

    Parameters
    ----------
    player       : 1 or 2
    before/after_points : 24-element list (negative = P1, positive = P2)
    before/after_bar    : number of the player's checkers on the bar
    before/after_off    : number of the player's checkers borne off
    dice         : die values rolled this turn (empty → best-effort only)

    Returns list of (from, to) int tuples using BAR=0 and OFF=25.
    """
    bc = _counts(before_points, player)
    ac = _counts(after_points, player)

    # Sources: points where the player lost checkers
    sources: list[int] = []
    for i in range(24):
        lost = bc[i] - ac[i]
        if lost > 0:
            sources.extend([i + 1] * lost)

    # Bar entries must come first (rules require clearing the bar first)
    bar_entries = max(0, before_bar - after_bar)
    if bar_entries > 0:
        sources = [BAR] * bar_entries + sources

    # Destinations: points where the player gained checkers
    dests: list[int] = []
    for i in range(24):
        gained = ac[i] - bc[i]
        if gained > 0:
            dests.extend([i + 1] * gained)

    bore_off = max(0, after_off - before_off)
    if bore_off > 0:
        dests.extend([OFF] * bore_off)

    if not sources:
        return []

    return _match(player, sources, dests, list(dice))


# ── Notation ──────────────────────────────────────────────────────────────────

def move_to_str(move: tuple[int, int]) -> str:
    src, dst = move
    s = "bar" if src == BAR else str(src)
    d = "off" if dst == OFF else str(dst)
    return f"{s}/{d}"


def moves_to_str(moves: list[tuple[int, int]]) -> str:
    return " ".join(move_to_str(m) for m in moves)
