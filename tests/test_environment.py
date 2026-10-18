"""Tests for Gymnasium environment, observations, and legal actions."""

from __future__ import annotations

import numpy as np
import pytest

from tetris_rl.environment import MAX_ACTIONS, TetrisEnv
from tetris_rl.features import observation_size, state_action_feature_size
from tetris_rl.rewards import RewardConfig, shaped_reward


def test_observation_shape():
    env = TetrisEnv(seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (observation_size(),)
    assert obs.dtype == np.float32
    assert info["n_legal"] > 0
    assert info["action_mask"].shape == (MAX_ACTIONS,)
    assert info["action_mask"].sum() == info["n_legal"]


def test_legal_action_enumeration():
    env = TetrisEnv(seed=1)
    env.reset(seed=1)
    legal = env.legal_placements()
    assert 1 <= len(legal) <= MAX_ACTIONS
    mask = env.action_mask()
    assert mask[: len(legal)].all()
    assert not mask[len(legal) :].any()


def test_step_illegal_action_raises():
    env = TetrisEnv(seed=2)
    env.reset(seed=2)
    n = len(env.legal_placements())
    with pytest.raises(ValueError):
        env.step(n)


def test_episode_runs_to_completion():
    env = TetrisEnv(seed=3)
    obs, info = env.reset(seed=3)
    steps = 0
    done = False
    while not done and steps < 300:
        action = 0  # first legal
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        assert np.isfinite(reward)
    assert steps > 0


def test_reward_line_clear_positive():
    env = TetrisEnv(seed=0)
    board_before = np.zeros_like(env.engine.board)
    board_after = np.zeros_like(env.engine.board)
    r = shaped_reward(env.engine, board_before, board_after, lines_cleared=4, game_over=False)
    assert r > 0


def test_reward_game_over_penalty():
    env = TetrisEnv(seed=0)
    cfg = RewardConfig(game_over_penalty=10.0, survival_reward=0.0)
    board = env.engine.board.copy()
    r = shaped_reward(env.engine, board, board, lines_cleared=0, game_over=True, config=cfg)
    assert r <= -10.0


def test_reward_hole_penalty():
    env = TetrisEnv(seed=0)
    before = np.zeros_like(env.engine.board)
    after = before.copy()
    after[-1, 0] = 1
    after[-2, 0] = 0
    after[-3, 0] = 1  # hole at -2
    # count_holes: empty with filled above
    r = shaped_reward(
        env.engine,
        before,
        after,
        lines_cleared=0,
        game_over=False,
        config=RewardConfig(
            hole_penalty=1.0,
            height_penalty=0.0,
            bumpiness_penalty=0.0,
            survival_reward=0.0,
        ),
    )
    assert r < 0


def test_state_action_feature_size():
    env = TetrisEnv(seed=5)
    env.reset(seed=5)
    from tetris_rl.features import state_action_features

    feats = state_action_features(env.engine, env.legal_placements()[0])
    assert feats.shape == (state_action_feature_size(),)
