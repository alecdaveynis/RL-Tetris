"""
Pure Tetris game engine: board logic independent of rendering and RL.

Uses a standard 10x20 visible board with a 7-bag randomizer for pieces.
Actions are placement-level: (rotation, column) hard-drops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

BOARD_WIDTH = 10
BOARD_HEIGHT = 20

# Standard guideline scoring for single/double/triple/tetris at level 1.
LINE_SCORES = {1: 100, 2: 300, 3: 500, 4: 800}


class PieceType(IntEnum):
    I = 0
    O = 1
    T = 2
    S = 3
    Z = 4
    J = 5
    L = 6


PIECE_NAMES = ["I", "O", "T", "S", "Z", "J", "L"]

# Relative cell offsets for each piece and each of 4 rotations (SRS-like).
# Coordinates are (row, col) relative to a pivot; positive row is downward.
_SHAPES: dict[PieceType, Tuple[Tuple[Tuple[int, int], ...], ...]] = {
    PieceType.I: (
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 1), (1, 1), (2, 1), (3, 1)),
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 1), (1, 1), (2, 1), (3, 1)),
    ),
    PieceType.O: (
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
    ),
    PieceType.T: (
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 1), (1, 0), (1, 1), (2, 1)),
        ((1, 0), (1, 1), (1, 2), (0, 1)),
        ((0, 0), (1, 0), (2, 0), (1, 1)),
    ),
    PieceType.S: (
        ((0, 1), (0, 2), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (1, 1), (2, 1)),
        ((0, 1), (0, 2), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (1, 1), (2, 1)),
    ),
    PieceType.Z: (
        ((0, 0), (0, 1), (1, 1), (1, 2)),
        ((0, 1), (1, 0), (1, 1), (2, 0)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
        ((0, 1), (1, 0), (1, 1), (2, 0)),
    ),
    PieceType.J: (
        ((0, 0), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 2)),
        ((0, 1), (1, 1), (2, 0), (2, 1)),
    ),
    PieceType.L: (
        ((0, 2), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 0)),
        ((0, 0), (0, 1), (1, 1), (2, 1)),
    ),
}


def get_shape(piece: PieceType, rotation: int) -> Tuple[Tuple[int, int], ...]:
    """Return cell offsets for a piece at the given rotation (0-3)."""
    return _SHAPES[piece][rotation % 4]


def shape_width(cells: Sequence[Tuple[int, int]]) -> int:
    return max(c for _, c in cells) - min(c for _, c in cells) + 1


def shape_min_col(cells: Sequence[Tuple[int, int]]) -> int:
    return min(c for _, c in cells)


@dataclass(frozen=True)
class Placement:
    """A legal hard-drop placement of the current piece."""

    rotation: int
    column: int  # leftmost column of the bounding shape's origin col=0
    piece: PieceType


@dataclass
class StepResult:
    """Outcome of locking one placement."""

    lines_cleared: int
    score_gained: int
    game_over: bool
    pieces_placed: int


@dataclass
class TetrisEngine:
    """
    Stateful Tetris engine with placement-level API.

    Board cells store 0 for empty or (piece_type + 1) for filled cells.
    """

    width: int = BOARD_WIDTH
    height: int = BOARD_HEIGHT
    seed: Optional[int] = None
    board: np.ndarray = field(init=False)
    score: int = field(init=False, default=0)
    lines_cleared_total: int = field(init=False, default=0)
    pieces_placed: int = field(init=False, default=0)
    game_over: bool = field(init=False, default=False)
    current_piece: PieceType = field(init=False)
    next_piece: PieceType = field(init=False)
    _rng: np.random.Generator = field(init=False, repr=False)
    _bag: List[PieceType] = field(init=False, default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.reset(self.seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = seed
        self._rng = np.random.default_rng(self.seed)
        self.board = np.zeros((self.height, self.width), dtype=np.int8)
        self.score = 0
        self.lines_cleared_total = 0
        self.pieces_placed = 0
        self.game_over = False
        self._bag = []
        self.current_piece = self._draw_piece()
        self.next_piece = self._draw_piece()

    def _draw_piece(self) -> PieceType:
        """7-bag randomizer: each bag contains each piece once."""
        if not self._bag:
            bag = list(PieceType)
            self._rng.shuffle(bag)
            self._bag = bag
        return self._bag.pop()

    def copy_board(self) -> np.ndarray:
        return self.board.copy()

    def cells_at(
        self, piece: PieceType, rotation: int, row: int, col: int
    ) -> List[Tuple[int, int]]:
        """Absolute board coordinates for a piece placed at (row, col)."""
        return [(row + dr, col + dc) for dr, dc in get_shape(piece, rotation)]

    def in_bounds(self, cells: Sequence[Tuple[int, int]]) -> bool:
        for r, c in cells:
            if c < 0 or c >= self.width or r >= self.height:
                return False
            # Negative rows are above the visible board (spawn area) — allowed.
        return True

    def collides(self, cells: Sequence[Tuple[int, int]], board: Optional[np.ndarray] = None) -> bool:
        b = self.board if board is None else board
        for r, c in cells:
            if c < 0 or c >= self.width or r >= self.height:
                return True
            if r >= 0 and b[r, c] != 0:
                return True
        return False

    def hard_drop_row(
        self,
        piece: PieceType,
        rotation: int,
        col: int,
        board: Optional[np.ndarray] = None,
    ) -> Optional[int]:
        """
        Return the landing row for a piece at (rotation, col), or None if illegal.

        Starts from above the board and drops until collision.
        """
        b = self.board if board is None else board
        cells0 = self.cells_at(piece, rotation, 0, col)
        # Origin column must keep all cells in horizontal bounds for some row.
        for _, c in cells0:
            if c < 0 or c >= self.width:
                return None

        row = -max(dr for dr, _ in get_shape(piece, rotation))  # start fully above
        # Ensure we start from a non-colliding position above the stack.
        while self.collides(self.cells_at(piece, rotation, row, col), b):
            row -= 1
            if row < -4:
                return None

        while True:
            next_row = row + 1
            if self.collides(self.cells_at(piece, rotation, next_row, col), b):
                # Landing row must place at least one cell on the visible board
                # without colliding; if all cells are above board and blocked, fail.
                final = self.cells_at(piece, rotation, row, col)
                if self.collides(final, b):
                    return None
                return row
            row = next_row
            if row > self.height:
                return None

    def is_valid_placement(
        self,
        piece: PieceType,
        rotation: int,
        col: int,
        board: Optional[np.ndarray] = None,
    ) -> bool:
        landing = self.hard_drop_row(piece, rotation, col, board)
        if landing is None:
            return False
        cells = self.cells_at(piece, rotation, landing, col)
        # Top-out: any cell locked above the visible board means game over on lock,
        # but the placement itself can still be "attempted". For legal enumeration
        # we allow placements that fit entirely on the board after drop.
        return all(r >= 0 for r, _ in cells)

    def enumerate_placements(
        self,
        piece: Optional[PieceType] = None,
        board: Optional[np.ndarray] = None,
    ) -> List[Placement]:
        """Enumerate all legal hard-drop placements for a piece on a board."""
        p = self.current_piece if piece is None else piece
        placements: List[Placement] = []
        seen: set[Tuple[Tuple[int, int], ...]] = set()

        # O piece has identical rotations; I/S/Z have period 2 — still try 0..3
        # and dedupe by resulting locked cell set.
        for rot in range(4):
            shape = get_shape(p, rot)
            min_c = shape_min_col(shape)
            max_c = max(c for _, c in shape)
            # column is the origin col such that cells = (row+dr, col+dc)
            for col in range(-min_c, self.width - max_c):
                if not self.is_valid_placement(p, rot, col, board):
                    continue
                landing = self.hard_drop_row(p, rot, col, board)
                assert landing is not None
                cells = tuple(sorted(self.cells_at(p, rot, landing, col)))
                if cells in seen:
                    continue
                seen.add(cells)
                placements.append(Placement(rotation=rot, column=col, piece=p))
        return placements

    def simulate_placement(
        self,
        placement: Placement,
        board: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, int, int]:
        """
        Apply a placement on a copy of the board.

        Returns (new_board, lines_cleared, score_gained).
        Does not mutate engine state.
        """
        b = (self.board if board is None else board).copy()
        landing = self.hard_drop_row(placement.piece, placement.rotation, placement.column, b)
        if landing is None:
            raise ValueError(f"Illegal placement: {placement}")
        cells = self.cells_at(placement.piece, placement.rotation, landing, placement.column)
        for r, c in cells:
            if r < 0:
                # Would lock above board — treat as top-out board (unchanged cells)
                # Caller should detect game over separately.
                continue
            b[r, c] = int(placement.piece) + 1
        cleared, b = clear_lines(b)
        score_gained = LINE_SCORES.get(cleared, 0)
        return b, cleared, score_gained

    def step(self, placement: Placement) -> StepResult:
        """Lock a placement, clear lines, advance pieces, detect top-out."""
        if self.game_over:
            raise RuntimeError("Cannot step: game is over")

        landing = self.hard_drop_row(placement.piece, placement.rotation, placement.column)
        if landing is None:
            raise ValueError(f"Illegal placement: {placement}")

        cells = self.cells_at(placement.piece, placement.rotation, landing, placement.column)
        # Top-out if any block locks above the visible board or collides at spawn.
        if any(r < 0 for r, _ in cells):
            self.game_over = True
            return StepResult(
                lines_cleared=0,
                score_gained=0,
                game_over=True,
                pieces_placed=self.pieces_placed,
            )

        for r, c in cells:
            self.board[r, c] = int(placement.piece) + 1

        cleared, self.board = clear_lines(self.board)
        score_gained = LINE_SCORES.get(cleared, 0)
        self.score += score_gained
        self.lines_cleared_total += cleared
        self.pieces_placed += 1

        # Advance piece queue
        self.current_piece = self.next_piece
        self.next_piece = self._draw_piece()

        # Game over if no legal placements for the new current piece
        # (stack reached the top / cannot spawn).
        if not self.enumerate_placements():
            self.game_over = True

        return StepResult(
            lines_cleared=cleared,
            score_gained=score_gained,
            game_over=self.game_over,
            pieces_placed=self.pieces_placed,
        )

    def column_heights(self, board: Optional[np.ndarray] = None) -> np.ndarray:
        b = self.board if board is None else board
        heights = np.zeros(self.width, dtype=np.int32)
        for c in range(self.width):
            col = b[:, c]
            filled = np.where(col != 0)[0]
            if len(filled) == 0:
                heights[c] = 0
            else:
                heights[c] = self.height - int(filled[0])
        return heights


def clear_lines(board: np.ndarray) -> Tuple[int, np.ndarray]:
    """Clear full rows; return (num_cleared, new_board)."""
    h, w = board.shape
    keep = [r for r in range(h) if not np.all(board[r] != 0)]
    cleared = h - len(keep)
    if cleared == 0:
        return 0, board
    new_board = np.zeros_like(board)
    new_board[cleared:] = board[keep]
    return cleared, new_board


def rotate_piece(piece: PieceType, rotation: int, times: int = 1) -> int:
    """Return the new rotation index after rotating `times` times clockwise."""
    if piece == PieceType.O:
        return rotation % 4
    return (rotation + times) % 4


def iter_rotations(piece: PieceType) -> Iterator[int]:
    """Yield distinct rotation indices for a piece."""
    if piece == PieceType.O:
        yield 0
    elif piece in (PieceType.I, PieceType.S, PieceType.Z):
        yield 0
        yield 1
    else:
        yield from range(4)
