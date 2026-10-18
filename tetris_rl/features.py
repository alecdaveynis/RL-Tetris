"""
Board and state-action feature extraction for Tetris-RL.

Features are computed from information available at decision time only:
board occupancy, current piece, next (preview) piece, and derived metrics.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from tetris_rl.engine import BOARD_HEIGHT, BOARD_WIDTH, PieceType, Placement, TetrisEngine


def count_holes(board: np.ndarray) -> int:
    """Empty cells with at least one filled cell above them in the same column."""
    h, w = board.shape
    holes = 0
    for c in range(w):
        block_seen = False
        for r in range(h):
            if board[r, c] != 0:
                block_seen = True
            elif block_seen:
                holes += 1
    return holes


def bumpiness(heights: np.ndarray) -> int:
    """Sum of absolute differences between adjacent column heights."""
    return int(np.sum(np.abs(np.diff(heights))))


def aggregate_height(heights: np.ndarray) -> int:
    return int(np.sum(heights))


def max_height(heights: np.ndarray) -> int:
    return int(np.max(heights)) if len(heights) else 0


def board_metrics(board: np.ndarray) -> dict:
    """Compute standard Dellacherie-style board metrics."""
    engine_heights = _column_heights(board)
    return {
        "heights": engine_heights,
        "aggregate_height": aggregate_height(engine_heights),
        "holes": count_holes(board),
        "bumpiness": bumpiness(engine_heights),
        "max_height": max_height(engine_heights),
    }


def _column_heights(board: np.ndarray) -> np.ndarray:
    h, w = board.shape
    heights = np.zeros(w, dtype=np.float32)
    for c in range(w):
        filled = np.where(board[:, c] != 0)[0]
        if len(filled) == 0:
            heights[c] = 0.0
        else:
            heights[c] = float(h - int(filled[0]))
    return heights


def one_hot_piece(piece: PieceType, n: int = 7) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    v[int(piece)] = 1.0
    return v


def observation_vector(
    engine: TetrisEngine,
    board: Optional[np.ndarray] = None,
    lines_just_cleared: int = 0,
) -> np.ndarray:
    """
    Flat observation used by the Gymnasium environment.

    Layout:
      - board occupancy (H*W)
      - current piece one-hot (7)
      - next piece one-hot (7)
      - normalized column heights (W)
      - aggregate_height / (H*W)
      - holes / (H*W)
      - bumpiness / (H*W)
      - max_height / H
      - lines_just_cleared / 4
    """
    b = engine.board if board is None else board
    h, w = b.shape
    metrics = board_metrics(b)
    heights = metrics["heights"] / float(h)

    parts = [
        (b != 0).astype(np.float32).ravel(),
        one_hot_piece(engine.current_piece),
        one_hot_piece(engine.next_piece),
        heights.astype(np.float32),
        np.array(
            [
                metrics["aggregate_height"] / float(h * w),
                metrics["holes"] / float(h * w),
                metrics["bumpiness"] / float(h * w),
                metrics["max_height"] / float(h),
                lines_just_cleared / 4.0,
            ],
            dtype=np.float32,
        ),
    ]
    return np.concatenate(parts)


def observation_size(width: int = BOARD_WIDTH, height: int = BOARD_HEIGHT) -> int:
    return height * width + 7 + 7 + width + 5


def state_action_features(
    engine: TetrisEngine,
    placement: Placement,
    board: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Features for a candidate placement (resulting board + piece info).

    Used by the DQN to score each legal action independently.
    Includes resulting-board metrics and current/next piece identity.
    """
    b = engine.board if board is None else board
    new_board, lines_cleared, _ = engine.simulate_placement(placement, b)
    metrics = board_metrics(new_board)
    h, w = new_board.shape
    heights = metrics["heights"] / float(h)

    # Delta features help the network compare before/after.
    before = board_metrics(b)

    parts = [
        (new_board != 0).astype(np.float32).ravel(),
        one_hot_piece(placement.piece),
        one_hot_piece(engine.next_piece),
        heights.astype(np.float32),
        np.array(
            [
                metrics["aggregate_height"] / float(h * w),
                metrics["holes"] / float(h * w),
                metrics["bumpiness"] / float(h * w),
                metrics["max_height"] / float(h),
                lines_cleared / 4.0,
                (metrics["holes"] - before["holes"]) / float(h * w),
                (metrics["aggregate_height"] - before["aggregate_height"]) / float(h * w),
                (metrics["bumpiness"] - before["bumpiness"]) / float(h * w),
            ],
            dtype=np.float32,
        ),
    ]
    return np.concatenate(parts)


def state_action_feature_size(width: int = BOARD_WIDTH, height: int = BOARD_HEIGHT) -> int:
    return height * width + 7 + 7 + width + 8


def placement_feature_batch(
    engine: TetrisEngine,
    placements: list[Placement],
    board: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Stack state-action features for a list of placements. Shape (N, F)."""
    if not placements:
        return np.zeros((0, state_action_feature_size(engine.width, engine.height)), dtype=np.float32)
    return np.stack([state_action_features(engine, p, board) for p in placements], axis=0)
