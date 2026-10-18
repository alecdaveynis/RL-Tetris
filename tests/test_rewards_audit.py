"""Tests for audit fixes: delta rewards and terminal shaping."""

from __future__ import annotations

import numpy as np

from tetris_rl.environment import TetrisEnv
from tetris_rl.features import board_metrics
from tetris_rl.rewards import LegacyRewardConfig, RewardConfig, shaped_reward


def test_delta_height_does_not_punish_static_stack():
    """Improved shaping: same board before/after → no height penalty."""
    eng = TetrisEnv(seed=0).engine
    board = np.zeros_like(eng.board)
    board[-3:, :5] = 1
    r = shaped_reward(
        eng,
        board,
        board.copy(),
        lines_cleared=0,
        game_over=False,
        config=RewardConfig(survival_reward=0.0),
    )
    assert abs(r) < 1e-6


def test_lowering_stack_is_rewarded():
    eng = TetrisEnv(seed=0).engine
    before = np.zeros_like(eng.board)
    before[-5:, :] = 1
    after = np.zeros_like(eng.board)
    after[-2:, :] = 1
    r = shaped_reward(
        eng,
        before,
        after,
        lines_cleared=0,
        game_over=False,
        config=RewardConfig(
            height_penalty=0.05,
            hole_penalty=0.0,
            bumpiness_penalty=0.0,
            survival_reward=0.0,
        ),
    )
    assert r > 0


def test_filling_holes_is_rewarded():
    eng = TetrisEnv(seed=0).engine
    before = np.zeros_like(eng.board)
    before[-1, 0] = 0
    before[-2, 0] = 1  # hole at bottom under a block... actually hole needs filled above
    # Proper hole: empty at -1 with filled at -2
    before[:, :] = 0
    before[-2, 0] = 1
    before[-1, 0] = 0  # hole
    after = before.copy()
    after[-1, 0] = 1  # filled hole
    assert board_metrics(before)["holes"] == 1
    assert board_metrics(after)["holes"] == 0
    r = shaped_reward(
        eng,
        before,
        after,
        lines_cleared=0,
        game_over=False,
        config=RewardConfig(
            hole_penalty=0.5,
            height_penalty=0.0,
            bumpiness_penalty=0.0,
            survival_reward=0.0,
        ),
    )
    assert r > 0


def test_legacy_absolute_height_still_available():
    eng = TetrisEnv(seed=0).engine
    board = np.zeros_like(eng.board)
    board[-5:, :] = 1
    r = shaped_reward(
        eng,
        board,
        board.copy(),
        lines_cleared=0,
        game_over=False,
        config=LegacyRewardConfig(survival_reward=0.0),
    )
    assert r < 0  # absolute height penalty


def test_terminal_no_legal_moves_no_survival_bonus():
    """Game-over from no legal moves must not include survival_reward."""
    env = TetrisEnv(seed=0, reward_config=RewardConfig(survival_reward=0.01, game_over_penalty=10.0))
    env.reset(seed=0)
    # Fill board so next placement causes no legal moves after advance if possible.
    # Force near-full board and play until game over; check last reward.
    env.engine.board[2:, :] = 1
    env._refresh_legal()
    last_reward = None
    for _ in range(50):
        if env.engine.game_over or not env.legal_placements():
            break
        _, reward, terminated, _, _ = env.step(0)
        last_reward = float(reward)
        if terminated:
            break
    assert last_reward is not None
    # Terminal reward should be <= -game_over_penalty + small board deltas
    # and must not be survival-only.
    assert last_reward < -1.0
