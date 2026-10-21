"""Tests for the extended (step-budget) reward experiment tooling."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

import tetris_rl.training.train_dqn as td
from tetris_rl.agents.dqn_agent import DQNAgent, improved_dqn_config
from tetris_rl.agents.heuristic_agent import HeuristicAgent
from tetris_rl.environment import TetrisEnv
from tetris_rl.features import placement_feature_batch
from tetris_rl.rewards import HeuristicAlignedRewardConfig, RewardConfig
from tetris_rl.training import diagnostics as dg
from tetris_rl.training.train_dqn import VALIDATION_SEEDS, evaluate_greedy, parse_args

TINY = [
    "--preset", "improved", "--device", "cpu", "--no-tb",
    "--min-buffer-size", "32", "--batch-size", "32",
    "--epsilon-decay-steps", "300", "--max-pieces", "30",
    "--eval-games", "2",
]


def _tiny_args(tmp: Path, mode: str, seed: int, max_steps: int, eval_every: int):
    return parse_args(
        TINY
        + [
            "--reward-mode", mode, "--seed", str(seed),
            "--max-steps", str(max_steps), "--eval-every-steps", str(eval_every),
            "--checkpoint-dir", str(tmp / "ckpt"), "--log-dir", str(tmp / "logs"),
        ]
    )


def test_validation_seed_constants_agree():
    assert tuple(dg.VALIDATION_SEEDS) == tuple(VALIDATION_SEEDS)


def test_seed_ranges_disjoint():
    train_max = 42 + 43 + 5000  # generous bound on `seed + episode`
    test = set(range(0, 100))
    val = set(VALIDATION_SEEDS)
    diag = set(dg.DIAG_HEURISTIC_SEEDS) | set(dg.DIAG_RANDOM_SEEDS)
    assert not (val & test) and not (val & diag) and not (diag & test)
    assert train_max < min(val | diag)


def test_heuristic_score_decoding_matches_heuristic_agent():
    for seed in range(3):
        env = TetrisEnv(seed=seed)
        env.reset(seed=seed)
        agent = HeuristicAgent(seed=seed)
        for _ in range(12):
            if not env.legal_placements():
                break
            feats = placement_feature_batch(env.engine, env.legal_placements())
            decoded = dg.heuristic_scores_from_features(feats)
            direct = np.array(
                [agent.evaluate_placement(env, i) for i in range(len(feats))]
            )
            assert np.allclose(decoded, direct, atol=1e-6)
            env.step(agent.select_action(env))


def test_spearman_basic():
    assert dg.spearman(np.array([1, 2, 3, 4.0]), np.array([10, 20, 30, 40.0])) == pytest.approx(1.0)
    assert dg.spearman(np.array([1, 2, 3, 4.0]), np.array([4, 3, 2, 1.0])) == pytest.approx(-1.0)
    assert np.isnan(dg.spearman(np.array([1, 1, 1.0]), np.array([1, 2, 3.0])))
    # ties get average ranks
    assert dg.spearman(np.array([1, 1, 2.0]), np.array([1, 1, 2.0])) == pytest.approx(1.0)


def test_diagnostic_states_deterministic_and_nonempty():
    a = dg.build_diagnostic_states()
    b = dg.build_diagnostic_states()
    assert len(a.heuristic) >= 50 and len(a.random) >= 20
    assert len(a.heuristic) == len(b.heuristic) and len(a.random) == len(b.random)
    for x, y in zip(a.random, b.random):
        assert np.array_equal(x.board, y.board)


def test_extended_eval_matches_plain_eval_and_is_read_only():
    cfg = improved_dqn_config(device="cpu", seed=3, hidden_sizes=(32, 16))
    agent = DQNAgent(cfg)
    before = {k: v.clone() for k, v in agent.online.state_dict().items()}
    rng_before = agent.rng.bit_generator.state
    plain = evaluate_greedy(agent, 30, n_games=4)
    ext = dg.evaluate_greedy_extended(
        agent, 30, HeuristicAlignedRewardConfig(), seeds=VALIDATION_SEEDS[:4]
    )
    for k in ("mean_score", "median_score", "mean_lines", "mean_pieces", "top_out_rate"):
        assert plain[k] == pytest.approx(ext[k])
    for k, v in agent.online.state_dict().items():
        assert torch.equal(v, before[k])
    assert agent.rng.bit_generator.state == rng_before
    probe = dg.probe_diagnostic_states(agent, dg.build_diagnostic_states())
    assert agent.rng.bit_generator.state == rng_before
    assert 0.0 <= probe["hs_top1_agreement"] <= 1.0
    assert probe["hs_heuristic_rank"] >= 1.0


def test_epsilon_schedule_is_env_step_based():
    cfg = improved_dqn_config(device="cpu", seed=0, hidden_sizes=(8,))
    assert cfg.epsilon_decay_steps == 15_000 and cfg.epsilon_end == 0.05
    agent = DQNAgent(cfg)
    env = TetrisEnv(seed=0)
    env.reset(seed=0)
    n = 0
    for target in (1, 7500, 15000, 20000):
        while n < target:
            # observe() with a trivial transition; episode boundaries are irrelevant
            agent.observe(
                np.zeros(agent.feature_dim, np.float32), 0.0, env, terminated=False
            )
            n += 1
        expected = max(0.05, 1.0 - 0.95 * min(1.0, n / 15000))
        assert agent.epsilon == pytest.approx(expected)
    assert agent.epsilon == pytest.approx(0.05)


def test_reward_mode_reaches_environment(monkeypatch, tmp_path):
    seen: list[tuple] = []
    real = td.TetrisEnv

    def spy(*a, **k):
        if k.get("reward_config") is not None:
            seen.append(type(k["reward_config"]).__name__)
        return real(*a, **k)

    monkeypatch.setattr(td, "TetrisEnv", spy)
    for mode, cls in (("delta", "RewardConfig"), ("heuristic_aligned", "HeuristicAlignedRewardConfig")):
        seen.clear()
        td.train(_tiny_args(tmp_path / mode, mode, 1, max_steps=40, eval_every=40))
        assert seen and set(seen) == {cls}


def test_arms_differ_only_in_reward():
    a = parse_args(["--reward-mode", "delta", "--seed", "42"])
    b = parse_args(["--reward-mode", "heuristic_aligned", "--seed", "42"])
    ca, ra = td.build_config(a, "cpu")
    cb, rb = td.build_config(b, "cpu")
    assert asdict(ca) == asdict(cb)
    assert type(ra) is RewardConfig and type(rb) is HeuristicAlignedRewardConfig
    assert ra.__dict__ != rb.__dict__
    assert ra.game_over_penalty == rb.game_over_penalty
    pa = DQNAgent(ca)
    pb = DQNAgent(cb)
    assert [p.shape for p in pa.online.parameters()] == [p.shape for p in pb.online.parameters()]
    # identical init for identical seed (fair pairing)
    for x, y in zip(pa.online.parameters(), pb.online.parameters()):
        assert torch.equal(x, y)


def test_step_budget_exact_and_artifacts(tmp_path):
    args = _tiny_args(tmp_path, "heuristic_aligned", 5, max_steps=200, eval_every=50)
    td.train(args)
    summary = json.loads((tmp_path / "logs" / "run_summary.json").read_text())
    assert summary["env_steps"] == 200
    rows = list(csv.DictReader((tmp_path / "logs" / "eval_metrics.csv").open()))
    assert [int(r["env_steps"]) for r in rows] == [50, 100, 150, 200]
    for key in (
        "mean_score", "median_score", "mean_lines", "mean_pieces", "top_out_rate",
        "grad_norm_mean", "train_loss", "train_td_error", "holes_created_per_move",
        "agg_height_delta_per_move", "hs_q_max_mean", "hs_q_max_global",
        "hs_top1_agreement", "hs_spearman", "rs_top1_agreement", "onpolicy_heuristic_agreement",
        "cal_bias_norm",
    ):
        assert key in rows[0]
    ckpts = {p.name for p in (tmp_path / "ckpt").glob("*.pt")}
    assert {"dqn_step50.pt", "dqn_step200.pt", "dqn_best.pt", "dqn_final.pt"} <= ckpts
    # weights-only checkpoints must still load
    DQNAgent.from_checkpoint(tmp_path / "ckpt" / "dqn_step100.pt", device="cpu")


def test_eval_frequency_does_not_change_training(tmp_path):
    """Mid-episode evaluation must not perturb the training trajectory."""
    finals = []
    for tag, every in (("a", 40), ("b", 1000)):
        args = _tiny_args(tmp_path / tag, "delta", 7, max_steps=120, eval_every=every)
        td.train(args)
        ck = torch.load(tmp_path / tag / "ckpt" / "dqn_final.pt", weights_only=False, map_location="cpu")
        finals.append(ck["online"])
    for k in finals[0]:
        assert torch.equal(finals[0][k], finals[1][k]), k


# ---------------------------------------------------------------------------
# Heuristic guidance / demonstrations
# ---------------------------------------------------------------------------
from tetris_rl.training import demos as dm  # noqa: E402


def test_demo_seeds_disjoint():
    d = set(dm.DEMO_TRAIN_SEEDS) | set(dm.DEMO_VAL_SEEDS)
    other = set(VALIDATION_SEEDS) | set(dg.DIAG_HEURISTIC_SEEDS) | set(dg.DIAG_RANDOM_SEEDS) | set(range(0, 100))
    assert not (d & other)
    assert not (set(dm.DEMO_TRAIN_SEEDS) & set(dm.DEMO_VAL_SEEDS))
    assert min(d) > 5000


def _tiny_demos(n_games=2, cap=12):
    raw = dm.collect_demonstrations(range(300_000, 300_000 + n_games), RewardConfig(), max_pieces=cap)
    return dm.DemoSet(raw)


def test_demo_transitions_consistent_with_env():
    ds = _tiny_demos()
    assert len(ds) == 2 * 12
    # replay the first game with the stored actions; features/rewards must match
    env = TetrisEnv(seed=300_000, reward_config=RewardConfig())
    env.reset(seed=300_000)
    for i in range(12):
        t = ds.transition(i)
        feats = placement_feature_batch(env.engine, env.legal_placements())
        a = int(ds.d["t_chosen"][i])
        assert np.allclose(feats[a], t.state_action)
        _, r, term, _, _ = env.step(a)
        assert r == pytest.approx(t.reward) and term == t.terminated
        if not term:
            nxt = placement_feature_batch(env.engine, env.legal_placements())
            assert np.allclose(nxt, t.next_state_actions)
    # chosen action is a heuristic-best action
    for i in range(len(ds)):
        feats = ds.block(int(ds.d["t_state"][i]))
        sc = dg.heuristic_scores_from_features(feats)
        assert int(ds.d["t_chosen"][i]) in set(np.flatnonzero(np.isclose(sc, sc.max())).tolist())


def test_bc_pretrain_learns_and_leaves_counters_alone():
    ds = _tiny_demos(n_games=3, cap=30)
    agent = DQNAgent(improved_dqn_config(device="cpu", seed=0, hidden_sizes=(64, 32)))
    before = demo_top1 = dm.demo_top1(agent, ds, n=90)
    log = dm.bc_pretrain(agent, ds, steps=300, batch_size=32, val_demos=ds, seed=0, log_every=150)
    assert log[-1]["val_top1"] > before + 0.15
    assert agent.total_steps == 0 and agent.grad_steps == 0 and agent.epsilon == 1.0


def test_heuristic_mix_runs_and_off_is_default(tmp_path):
    args = parse_args(
        TINY + ["--reward-mode", "delta", "--seed", "3", "--max-steps", "100", "--eval-every-steps", "50",
                "--heuristic-mix-start", "1.0", "--heuristic-mix-steps", "100",
                "--checkpoint-dir", str(tmp_path / "ck"), "--log-dir", str(tmp_path / "lg")]
    )
    td.train(args)
    rows = list(csv.DictReader((tmp_path / "lg" / "eval_metrics.csv").open()))
    assert float(rows[0]["h_action_frac"]) > 0.5      # p ~ 1 -> 0.5 over first 50 steps
    assert float(rows[1]["h_action_frac"]) < float(rows[0]["h_action_frac"])
    assert float(rows[1]["mix_p"]) == pytest.approx(0.0)
    assert parse_args([]).heuristic_mix_start == 0.0 and parse_args([]).bc_pretrain_steps == 0


def test_prefill_and_bc_pipeline(tmp_path):
    ds_path = tmp_path / "d.npz"
    dm.save_demos(ds_path, _tiny_demos(n_games=2, cap=40).d)
    args = parse_args(
        TINY + ["--reward-mode", "delta", "--seed", "4", "--max-steps", "60", "--eval-every-steps", "30",
                "--demo-path", str(ds_path), "--demo-val-path", str(ds_path), "--demo-prefill",
                "--bc-pretrain-steps", "20", "--eval-at-start",
                "--checkpoint-dir", str(tmp_path / "ck"), "--log-dir", str(tmp_path / "lg")]
    )
    td.train(args)
    rows = list(csv.DictReader((tmp_path / "lg" / "eval_metrics.csv").open()))
    assert [int(r["env_steps"]) for r in rows] == [0, 30, 60]
    info = json.loads((tmp_path / "lg" / "demo_info.json").read_text())
    assert info["prefilled"] == 80
