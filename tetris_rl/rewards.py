"""
Configurable reward shaping for Tetris-RL training.

Training uses shaped rewards to guide learning. Evaluation reports true
game score and lines cleared — not the shaped training reward.

Shaping uses *deltas* of board metrics (after − before) so long survival is
not punished merely for having a non-empty stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from tetris_rl.engine import Placement, TetrisEngine
from tetris_rl.features import board_metrics


@dataclass
class RewardConfig:
    """Weights for shaped training rewards (delta-based board terms)."""

    # Line clears must outweigh typical hole/height deltas from a single placement.
    line_clear: tuple[float, float, float, float] = (2.0, 5.0, 8.0, 12.0)
    hole_penalty: float = 0.35
    height_penalty: float = 0.04
    bumpiness_penalty: float = 0.02
    game_over_penalty: float = 10.0
    survival_reward: float = 0.02
    # Legacy field kept for checkpoint/config compatibility; unused by delta shaping.
    max_height_for_penalty: float = 15.0


# Original (buggy) absolute-height shaping — retained for ablation / config A.
@dataclass
class LegacyRewardConfig(RewardConfig):
    """Pre-audit rewards: absolute height penalty each step (dominates return)."""

    line_clear: tuple[float, float, float, float] = (1.0, 3.0, 5.0, 8.0)
    hole_penalty: float = 0.35
    height_penalty: float = 0.01
    bumpiness_penalty: float = 0.01
    game_over_penalty: float = 10.0
    survival_reward: float = 0.01
    max_height_for_penalty: float = 15.0
    use_absolute_height: bool = True
    use_positive_only_deltas: bool = True


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

    Default (improved): delta-based height/holes/bumpiness + line clears.
    LegacyRewardConfig: restores absolute height penalty and one-sided deltas.
    """
    cfg = config or DEFAULT_REWARD_CONFIG
    before = board_metrics(board_before)
    after = board_metrics(board_after)

    reward = 0.0

    if 1 <= lines_cleared <= 4:
        reward += cfg.line_clear[lines_cleared - 1]

    hole_delta = after["holes"] - before["holes"]
    height_delta = after["aggregate_height"] - before["aggregate_height"]
    bump_delta = after["bumpiness"] - before["bumpiness"]

    use_abs_height = getattr(cfg, "use_absolute_height", False)
    use_pos_only = getattr(cfg, "use_positive_only_deltas", False)

    if use_abs_height:
        # Original buggy behavior (config A ablation).
        height_term = min(
            after["aggregate_height"],
            cfg.max_height_for_penalty * engine.width,
        )
        reward -= cfg.height_penalty * height_term
        if use_pos_only:
            reward -= cfg.hole_penalty * max(0, hole_delta)
            reward -= cfg.bumpiness_penalty * max(0, bump_delta)
        else:
            reward -= cfg.hole_penalty * hole_delta
            reward -= cfg.bumpiness_penalty * bump_delta
    else:
        # Improved: signed deltas (filling holes / lowering stack is rewarded).
        reward -= cfg.hole_penalty * hole_delta
        reward -= cfg.height_penalty * height_delta
        reward -= cfg.bumpiness_penalty * bump_delta

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

    Does not mutate the engine.
    """
    board_before = engine.copy_board()
    landing = engine.hard_drop_row(placement.piece, placement.rotation, placement.column)
    if landing is None:
        raise ValueError(f"Illegal placement: {placement}")
    cells = engine.cells_at(placement.piece, placement.rotation, landing, placement.column)
    top_out = any(r < 0 for r, _ in cells)

    if top_out:
        cfg = config or DEFAULT_REWARD_CONFIG
        return -cfg.game_over_penalty, board_before, 0, 0, True

    new_board, lines, score = engine.simulate_placement(placement, board_before)
    reward = shaped_reward(
        engine,
        board_before,
        new_board,
        lines,
        game_over=False,
        config=config,
    )
    return reward, new_board, lines, score, False
