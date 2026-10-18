"""Tests for DQN forward pass, training update, and save/load."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tetris_rl.agents.dqn_agent import DQNAgent, DQNConfig, QNetwork
from tetris_rl.environment import TetrisEnv
from tetris_rl.features import state_action_feature_size


def test_qnetwork_forward_pass():
    dim = state_action_feature_size()
    net = QNetwork(dim, hidden_sizes=(64, 32))
    x = torch.randn(5, dim)
    out = net(x)
    assert out.shape == (5,)


def test_qnetwork_layer_norm_forward():
    dim = state_action_feature_size()
    net = QNetwork(dim, hidden_sizes=(64, 32), use_layer_norm=True)
    x = torch.randn(5, dim)
    out = net(x)
    assert out.shape == (5,)


def test_double_dqn_train_step():
    cfg = DQNConfig(
        device="cpu",
        seed=7,
        batch_size=8,
        min_buffer_size=8,
        buffer_size=1000,
        double_dqn=True,
        use_layer_norm=True,
    )
    agent = DQNAgent(cfg)
    env = TetrisEnv(seed=7)
    env.reset(seed=7)
    for _ in range(20):
        if env.engine.game_over or not env.legal_placements():
            env.reset()
        action = agent.select_action(env, explore=True)
        sa = agent.features_for_action(env, action)
        _, reward, terminated, truncated, _ = env.step(action)
        agent.observe(
            sa, float(reward), env, terminated=bool(terminated), truncated=bool(truncated)
        )
        if terminated or truncated:
            env.reset()
    loss = agent.train_step()
    assert loss is not None and np.isfinite(loss)


def test_dqn_select_action():
    agent = DQNAgent(DQNConfig(device="cpu", seed=0, epsilon_start=0.0))
    env = TetrisEnv(seed=0)
    env.reset(seed=0)
    action = agent.select_action(env, explore=False)
    assert 0 <= action < len(env.legal_placements())


def test_one_training_update(tmp_path: Path):
    cfg = DQNConfig(
        device="cpu",
        seed=1,
        batch_size=8,
        min_buffer_size=8,
        buffer_size=1000,
        epsilon_start=1.0,
    )
    agent = DQNAgent(cfg)
    env = TetrisEnv(seed=1)
    env.reset(seed=1)

    # Fill buffer with a few transitions
    for _ in range(20):
        if env.engine.game_over or not env.legal_placements():
            env.reset()
        action = agent.select_action(env, explore=True)
        sa = agent.features_for_action(env, action)
        _, reward, terminated, truncated, _ = env.step(action)
        agent.observe(
            sa, float(reward), env, terminated=bool(terminated), truncated=bool(truncated)
        )
        if terminated or truncated:
            env.reset()

    assert len(agent.buffer) >= cfg.batch_size
    loss = agent.train_step()
    assert loss is not None
    assert np.isfinite(loss)


def test_model_save_load(tmp_path: Path):
    cfg = DQNConfig(device="cpu", seed=2, hidden_sizes=(64, 32))
    agent = DQNAgent(cfg)
    path = tmp_path / "test_model.pt"
    agent.epsilon = 0.42
    agent.total_steps = 123
    agent.save(path)

    loaded = DQNAgent.from_checkpoint(path, device="cpu", explore=True)
    assert loaded.epsilon == 0.42
    assert loaded.total_steps == 123

    # Weights match
    for (n1, p1), (n2, p2) in zip(
        agent.online.named_parameters(), loaded.online.named_parameters()
    ):
        assert n1 == n2
        assert torch.allclose(p1, p2)


def test_greedy_no_exploration():
    agent = DQNAgent(DQNConfig(device="cpu", seed=3, epsilon_start=1.0))
    agent.epsilon = 0.0
    env = TetrisEnv(seed=3)
    env.reset(seed=3)
    a1 = agent.select_action(env, explore=True)
    a2 = agent.select_action(env, explore=True)
    # With epsilon=0 both should be greedy argmax — identical
    assert a1 == a2
