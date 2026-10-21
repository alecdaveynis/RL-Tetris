"""Tests for audit fixes: delta rewards and terminal shaping."""

from __future__ import annotations

import numpy as np

from tetris_rl.environment import TetrisEnv
from tetris_rl.features import board_metrics
import pytest

from tetris_rl.rewards import (
    HeuristicAlignedRewardConfig,
    LegacyRewardConfig,
    RewardConfig,
    reward_config_from_name,
    shaped_reward,
)
from tetris_rl.training.train_dqn import build_config, parse_args


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


def test_heuristic_aligned_matches_heuristic_ranking():
    """On a shared board, heuristic-aligned deltas rank placements like HeuristicAgent."""
    from tetris_rl.agents.heuristic_agent import HeuristicAgent

    env = TetrisEnv(seed=3)
    env.reset(seed=3)
    heur = HeuristicAgent(seed=0)
    cfg = HeuristicAlignedRewardConfig(game_over_penalty=0.0, survival_reward=0.0)
    before = env.engine.copy_board()
    n = len(env.legal_placements())
    h_scores = np.array([heur.evaluate_placement(env, i) for i in range(n)])
    r_scores = []
    for i, p in enumerate(env.legal_placements()):
        after, lines, _ = env.engine.simulate_placement(p)
        r_scores.append(
            shaped_reward(env.engine, before, after, lines, game_over=False, config=cfg)
        )
    r_scores = np.array(r_scores)
    # Ranking must match: delta(Φ) = Φ(after) - Φ(before) and Φ(before) is constant.
    assert int(np.argmax(h_scores)) == int(np.argmax(r_scores))
    # Constant offset: heuristic uses absolute Φ(after).
    offset = h_scores - r_scores
    assert np.allclose(offset, offset[0], atol=1e-5)


def test_heuristic_aligned_penalizes_height_more_than_delta_default():
    eng = TetrisEnv(seed=0).engine
    before = np.zeros_like(eng.board)
    after = before.copy()
    after[-4:, 0] = 1  # raise one column by 4
    r_delta = shaped_reward(
        eng, before, after, 0, False,
        RewardConfig(height_penalty=0.04, hole_penalty=0.0, bumpiness_penalty=0.0, survival_reward=0.0),
    )
    r_h = shaped_reward(
        eng, before, after, 0, False,
        HeuristicAlignedRewardConfig(survival_reward=0.0, game_over_penalty=0.0),
    )
    assert r_h < r_delta < 0


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


def test_reward_config_from_name():
    assert type(reward_config_from_name("delta")).__name__ == "RewardConfig"
    assert isinstance(reward_config_from_name("legacy"), LegacyRewardConfig)
    assert isinstance(reward_config_from_name("heuristic_aligned"), HeuristicAlignedRewardConfig)
    assert isinstance(reward_config_from_name("dellacherie"), HeuristicAlignedRewardConfig)
    with pytest.raises(ValueError):
        reward_config_from_name("not_a_mode")


def test_train_cli_reward_mode_override():
    args = parse_args(["--preset", "improved", "--reward-mode", "heuristic_aligned", "--seed", "1"])
    cfg, reward_cfg = build_config(args, device="cpu")
    assert cfg.double_dqn is True
    assert isinstance(reward_cfg, HeuristicAlignedRewardConfig)
    assert reward_cfg.height_penalty == pytest.approx(0.51)

    args_delta = parse_args(["--preset", "improved", "--reward-mode", "delta"])
    _, delta_cfg = build_config(args_delta, device="cpu")
    assert type(delta_cfg) is RewardConfig
    assert delta_cfg.height_penalty == pytest.approx(0.04)


def test_heuristic_aligned_ranking_on_nonempty_board():
    from tetris_rl.agents.heuristic_agent import HeuristicAgent
    from tetris_rl.agents.random_agent import RandomAgent

    env = TetrisEnv(seed=11)
    env.reset(seed=11)
    rand = RandomAgent(seed=11)
    for _ in range(8):
        if env.engine.game_over or not env.legal_placements():
            break
        env.step(rand.select_action(env))
    if env.engine.game_over or len(env.legal_placements()) < 2:
        pytest.skip("episode ended too early for ranking check")

    heur = HeuristicAgent(seed=0)
    cfg = HeuristicAlignedRewardConfig(game_over_penalty=0.0, survival_reward=0.0)
    before = env.engine.copy_board()
    h_scores = np.array([heur.evaluate_placement(env, i) for i in range(len(env.legal_placements()))])
    r_scores = []
    for p in env.legal_placements():
        after, lines, _ = env.engine.simulate_placement(p)
        r_scores.append(
            shaped_reward(env.engine, before, after, lines, game_over=False, config=cfg)
        )
    r_scores = np.array(r_scores)
    assert int(np.argmax(h_scores)) == int(np.argmax(r_scores))
    assert np.allclose(h_scores - r_scores, (h_scores - r_scores)[0], atol=1e-5)
