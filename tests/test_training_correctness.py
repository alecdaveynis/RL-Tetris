"""Tests for Bellman targets, terminal/truncated, and checkpoint resume."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tetris_rl.agents.dqn_agent import DQNAgent, DQNConfig
from tetris_rl.environment import TetrisEnv
from tetris_rl.features import state_action_feature_size
from tetris_rl.training.replay_buffer import ReplayBuffer, Transition
from tetris_rl.training.train_dqn import VALIDATION_SEEDS, TEST_SEED_BASE, evaluate_greedy


def _fill_agent(agent: DQNAgent, n: int = 30, seed: int = 0) -> None:
    env = TetrisEnv(seed=seed)
    env.reset(seed=seed)
    for _ in range(n):
        if env.engine.game_over or not env.legal_placements():
            env.reset()
        action = agent.select_action(env, explore=True)
        sa = agent.features_for_action(env, action)
        _, reward, terminated, truncated, _ = env.step(action)
        # Force a truncation marker on some steps for buffer diversity.
        hit_cap = env.engine.pieces_placed >= 10 and not terminated
        agent.observe(
            sa,
            float(reward),
            env,
            terminated=bool(terminated),
            truncated=bool(truncated) or hit_cap,
        )
        if terminated or hit_cap:
            env.reset()


def test_validation_seeds_disjoint_from_test_base():
    assert len(VALIDATION_SEEDS) == 50
    assert TEST_SEED_BASE not in VALIDATION_SEEDS
    assert min(VALIDATION_SEEDS) >= 100_000


def test_bellman_terminal_no_bootstrap():
    cfg = DQNConfig(device="cpu", seed=0, gamma=0.9, double_dqn=False, hidden_sizes=(32, 16))
    agent = DQNAgent(cfg)
    feat = np.zeros(state_action_feature_size(), dtype=np.float32)
    batch = [
        Transition(
            state_action=feat,
            reward=-10.0,
            next_state_actions=np.zeros((0, agent.feature_dim), dtype=np.float32),
            terminated=True,
        )
    ]
    targets = agent.bellman_targets(batch, vectorized=True)
    assert torch.allclose(targets, torch.tensor([-10.0]))


def test_bellman_truncated_does_bootstrap():
    cfg = DQNConfig(device="cpu", seed=1, gamma=0.5, double_dqn=False, hidden_sizes=(32, 16))
    agent = DQNAgent(cfg)
    # Make target network produce known values via a single forward — use random
    # next actions and compare vectorized vs loop; bootstrap must be nonzero path.
    feat = np.random.randn(agent.feature_dim).astype(np.float32)
    next_a = np.random.randn(3, agent.feature_dim).astype(np.float32)
    batch = [
        Transition(
            state_action=feat,
            reward=1.0,
            next_state_actions=next_a,
            terminated=False,  # truncation-style: still bootstrap
        )
    ]
    with torch.no_grad():
        q_max = agent.target(torch.as_tensor(next_a)).max().item()
    expected = 1.0 + 0.5 * q_max
    targets_v = agent.bellman_targets(batch, vectorized=True)
    targets_l = agent.bellman_targets(batch, vectorized=False)
    assert abs(float(targets_v[0]) - expected) < 1e-5
    assert torch.allclose(targets_v, targets_l, atol=1e-5)


def test_vectorized_matches_loop_vanilla_and_double():
    for double in (False, True):
        cfg = DQNConfig(
            device="cpu",
            seed=2,
            batch_size=8,
            buffer_size=200,
            min_buffer_size=8,
            double_dqn=double,
            hidden_sizes=(64, 32),
            use_layer_norm=False,
        )
        agent = DQNAgent(cfg)
        _fill_agent(agent, n=40, seed=2)
        batch = agent.buffer.sample(8)
        with torch.no_grad():
            tv = agent.bellman_targets(batch, vectorized=True)
            tl = agent.bellman_targets(batch, vectorized=False)
        assert torch.allclose(tv, tl, atol=1e-5), f"double_dqn={double}"


def test_observe_terminated_vs_truncated():
    cfg = DQNConfig(device="cpu", seed=3, hidden_sizes=(32,), buffer_size=100)
    agent = DQNAgent(cfg)
    env = TetrisEnv(seed=3)
    env.reset(seed=3)
    action = 0
    sa = agent.features_for_action(env, action)
    _, reward, _, _, _ = env.step(action)

    agent.observe(sa, float(reward), env, terminated=True, truncated=False)
    t_term = agent.buffer._storage[0]
    assert t_term is not None and t_term.terminated
    assert t_term.next_state_actions.shape[0] == 0

    env.reset(seed=4)
    sa2 = agent.features_for_action(env, 0)
    _, reward2, _, _, _ = env.step(0)
    agent.observe(sa2, float(reward2), env, terminated=False, truncated=True)
    t_trunc = agent.buffer._storage[1]
    assert t_trunc is not None and not t_trunc.terminated
    assert t_trunc.next_state_actions.shape[0] > 0


def test_replay_buffer_state_dict_roundtrip():
    buf = ReplayBuffer(capacity=50, seed=0)
    f = np.zeros(8, dtype=np.float32)
    for i in range(12):
        buf.add(f + i, float(i), np.zeros((2, 8), dtype=np.float32), terminated=(i % 5 == 0))
    state = buf.state_dict()
    buf2 = ReplayBuffer(capacity=50, seed=99)
    buf2.load_state_dict(state)
    assert len(buf2) == len(buf)
    assert buf2._idx == buf._idx
    # Same RNG stream after restore
    s1 = buf.sample(4)
    s2 = buf2.sample(4)
    for a, b in zip(s1, s2):
        assert np.allclose(a.state_action, b.state_action)
        assert a.terminated == b.terminated
        assert a.reward == b.reward


def test_checkpoint_restores_buffer_rng_and_best_eval(tmp_path: Path):
    cfg = DQNConfig(
        device="cpu",
        seed=5,
        batch_size=4,
        buffer_size=200,
        min_buffer_size=4,
        hidden_sizes=(32, 16),
    )
    agent = DQNAgent(cfg)
    _fill_agent(agent, n=25, seed=5)
    agent.best_eval_score = 123.5
    agent.episodes_trained = 7
    agent.train_step()
    path = tmp_path / "full.pt"
    agent.save(path)

    loaded = DQNAgent.from_checkpoint(path, device="cpu", explore=True)
    assert loaded.best_eval_score == 123.5
    assert loaded.episodes_trained == 7
    assert loaded.total_steps == agent.total_steps
    assert len(loaded.buffer) == len(agent.buffer)
    assert abs(loaded.epsilon - agent.epsilon) < 1e-9

    # Agent RNG restored: next random action index stream matches.
    env = TetrisEnv(seed=99)
    env.reset(seed=99)
    agent.epsilon = 1.0
    loaded.epsilon = 1.0
    a1 = [agent.select_action(env, explore=True) for _ in range(5)]
    env.reset(seed=99)
    a2 = [loaded.select_action(env, explore=True) for _ in range(5)]
    assert a1 == a2


def test_load_survives_mps_mapped_torch_rng(tmp_path: Path):
    """map_location=mps turns CPU ByteTensor RNG into a non-ByteTensor-on-CPU."""
    cfg = DQNConfig(device="cpu", seed=4, hidden_sizes=(32, 16), buffer_size=50)
    agent = DQNAgent(cfg)
    path = tmp_path / "rng.pt"
    agent.save(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # Simulate the failure mode: float tensor / wrong device-like object
    fake = ckpt["torch_rng_state"].float()
    ckpt["torch_rng_state"] = fake
    bad = tmp_path / "bad_rng.pt"
    torch.save(ckpt, bad)
    loaded = DQNAgent.from_checkpoint(bad, device="cpu", explore=True)
    assert loaded.total_steps == agent.total_steps


def test_legacy_checkpoint_without_buffer_loads(tmp_path: Path):
    """Older checkpoints missing buffer/rng keys still load weights."""
    cfg = DQNConfig(device="cpu", seed=6, hidden_sizes=(32, 16))
    agent = DQNAgent(cfg)
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "online": agent.online.state_dict(),
            "target": agent.target.state_dict(),
            "optimizer": agent.optimizer.state_dict(),
            "config": cfg.__dict__,
            "epsilon": 0.2,
            "total_steps": 50,
            "grad_steps": 10,
            "learn_steps": 10,
        },
        path,
    )
    loaded = DQNAgent.from_checkpoint(path, device="cpu", explore=True)
    assert loaded.epsilon == 0.2
    assert loaded.total_steps == 50
    assert len(loaded.buffer) == 0
    assert loaded.best_eval_score == -float("inf")


def test_resume_matches_uninterrupted_train_step(tmp_path: Path):
    """Saving mid-run and resuming yields the same next loss (within tol)."""
    cfg = DQNConfig(
        device="cpu",
        seed=8,
        batch_size=8,
        buffer_size=500,
        min_buffer_size=8,
        hidden_sizes=(64, 32),
        double_dqn=True,
        target_update_freq=10_000,
    )
    continuous = DQNAgent(cfg)
    _fill_agent(continuous, n=40, seed=8)
    # One update before the fork
    continuous.train_step()

    path = tmp_path / "mid.pt"
    continuous.save(path)

    # Continuous continues one more update
    loss_cont = continuous.train_step()

    # Resumed agent performs the same update from the saved state
    resumed = DQNAgent.from_checkpoint(path, device="cpu", explore=True)
    loss_res = resumed.train_step()

    assert loss_cont is not None and loss_res is not None
    assert abs(loss_cont - loss_res) < 1e-5


def test_evaluate_greedy_uses_validation_seed_count():
    agent = DQNAgent(DQNConfig(device="cpu", seed=0, epsilon_start=0.0, hidden_sizes=(32,)))
    metrics = evaluate_greedy(agent, max_pieces=30, n_games=3)
    assert metrics["n_games"] == 3
