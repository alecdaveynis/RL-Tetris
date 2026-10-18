"""
Configurable reward shaping for Tetris-RL training.

Training uses shaped rewards to guide learning. Evaluation reports true
game score and lines cleared — not the shaped training reward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from tetris_rl.engine import Placement, TetrisEngine
from tetris_rl.features import board_metrics


@dataclass
class RewardConfig:
    """Weights for shaped training rewards."""

    line_clear: tuple[float, float, float, float] = (1.0, 3.0, 5.0, 8.0)
    hole_penalty: float = 0.35
    height_penalty: float = 0.01
    bumpiness_penalty: float = 0.01
    game_over_penalty: float = 10.0
    survival_reward: float = 0.01
    # Cap aggregate-height contribution to avoid dominating the signal.
    max_height_for_penalty: float = 15.0


DEFAULT_REWARD_CONFIG = RewardConfig()


def shaped_reward(
    engine: TetrisEngine,
    board_before: np.ndarray,
    board_after: np.ndarray,
    lines_cleared: int,
    game_over: bool,
    config: Optional[RewardConfig] = None,
) -> float:
    """
    Compute shaped reward for a transition.

    Positive for line clears (scaled by combo size), negative for holes/
    height/bumpiness increases and game over. Optional survival bonus.
    """
    cfg = config or DEFAULT_REWARD_CONFIG
    before = board_metrics(board_before)
    after = board_metrics(board_after)

    reward = 0.0

    if 1 <= lines_cleared <= 4:
        reward += cfg.line_clear[lines_cleared - 1]

    hole_delta = after["holes"] - before["holes"]
    reward -= cfg.hole_penalty * max(0, hole_delta)

    # Penalize absolute stack height (encourages keeping the board low).
    height_term = min(after["aggregate_height"], cfg.max_height_for_penalty * engine.width)
    reward -= cfg.height_penalty * height_term

    bump_delta = after["bumpiness"] - before["bumpiness"]
    reward -= cfg.bumpiness_penalty * max(0, bump_delta)

    if game_over:
        reward -= cfg.game_over_penalty
    else:
        reward += cfg.survival_reward

    return float(reward)


def reward_for_placement(
    engine: TetrisEngine,
    placement: Placement,
    config: Optional[RewardConfig] = None,
) -> tuple[float, np.ndarray, int, int, bool]:
    """
    Simulate a placement and return (reward, new_board, lines, score, would_top_out).

    Does not mutate the engine. Top-out is detected when locking would place
    cells above the visible board or leave no legal moves (checked by caller
    after stepping).
    """
    board_before = engine.copy_board()
    landing = engine.hard_drop_row(placement.piece, placement.rotation, placement.column)
    if landing is None:
        raise ValueError(f"Illegal placement: {placement}")
    cells = engine.cells_at(placement.piece, placement.rotation, landing, placement.column)
    top_out = any(r < 0 for r, _ in cells)

    if top_out:
        # No board change; large loss penalty.
        cfg = config or DEFAULT_REWARD_CONFIG
        return -cfg.game_over_penalty, board_before, 0, 0, True

    new_board, lines, score = engine.simulate_placement(placement, board_before)
    # Game-over from no future moves is decided after piece advance; here we
    # only know top-out from locking. Reward uses game_over=False for shaping
    # unless top_out; the environment sets terminal after step.
    reward = shaped_reward(
        engine,
        board_before,
        new_board,
        lines,
        game_over=False,
        config=config,
    )
    return reward, new_board, lines, score, False
