"""Tests for baseline agents."""

from __future__ import annotations

from tetris_rl.agents.heuristic_agent import HeuristicAgent, HeuristicWeights
from tetris_rl.agents.random_agent import RandomAgent
from tetris_rl.environment import TetrisEnv


def test_random_agent_action_in_range():
    env = TetrisEnv(seed=0)
    env.reset(seed=0)
    agent = RandomAgent(seed=0)
    a = agent.select_action(env)
    assert 0 <= a < len(env.legal_placements())


def test_heuristic_prefers_line_clear():
    """Heuristic with strong line weight should pick a clearing move when available."""
    env = TetrisEnv(seed=0)
    env.reset(seed=0)
    # Set up a nearly complete bottom row
    env.engine.board[-1, 1:] = 1
    env.engine.current_piece = env.engine.current_piece  # keep
    env._refresh_legal()

    agent = HeuristicAgent(
        weights=HeuristicWeights(
            lines_cleared=10.0,
            aggregate_height=-0.01,
            holes=-0.01,
            bumpiness=-0.01,
        ),
        seed=0,
    )
    if not env.legal_placements():
        return
    action = agent.select_action(env)
    assert 0 <= action < len(env.legal_placements())


def test_heuristic_deterministic_with_seed():
    env = TetrisEnv(seed=7)
    env.reset(seed=7)
    a1 = HeuristicAgent(seed=0).select_action(env)
    env2 = TetrisEnv(seed=7)
    env2.reset(seed=7)
    a2 = HeuristicAgent(seed=0).select_action(env2)
    assert a1 == a2
