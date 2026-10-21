"""Tests for imitation-regularised Double DQN (L = L_TD + lambda * L_imit)."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from tetris_rl.agents.dqn_agent import DQNAgent, improved_dqn_config
from tetris_rl.rewards import RewardConfig
from tetris_rl.training import diagnostics as dg
from tetris_rl.training.demos import (
    DEMO_TRAIN_SEEDS,
    DEMO_VAL_SEEDS,
    DemoSet,
    _padded_legal,
    collect_demonstrations,
    listwise_ce,
    save_demos,
)
from tetris_rl.training.imitation import (
    ImitationRegularizer,
    demo_metrics,
    imitation_ce,
    masked_logits,
)
from tetris_rl.training.train_dqn import VALIDATION_SEEDS, parse_args, train

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def demos() -> DemoSet:
    d = collect_demonstrations([7001, 7002, 7003], RewardConfig(), max_pieces=25, tie_seed=0)
    return DemoSet(d)


def _agent(seed: int = 0, **kw) -> DQNAgent:
    cfg = improved_dqn_config(device="cpu", seed=seed, min_buffer_size=16, batch_size=16, **kw)
    return DQNAgent(cfg)


def _fill(agent: DQNAgent, demos: DemoSet) -> None:
    demos.fill_buffer(agent.buffer)


# --------------------------------------------------------------------------- data


def test_demo_labels_are_legal_and_are_heuristic_choices(demos):
    d = demos.d
    for i in range(len(demos)):
        block = demos.block(int(d["t_state"][i]))
        a = int(d["t_chosen"][i])
        assert 0 <= a < block.shape[0]
        scores = dg.heuristic_scores_from_features(block)
        assert a in set(dg._best_set(scores).tolist())


def test_padded_mask_matches_real_legal_counts(demos):
    idx = np.arange(len(demos))[:40]
    feats, mask, chosen = _padded_legal(demos, idx, torch.device("cpu"))
    lens = [demos.block(int(demos.d["t_state"][i])).shape[0] for i in idx]
    assert mask.sum(dim=1).tolist() == lens
    for k, n in enumerate(lens):  # mask is a contiguous prefix of exactly n legal slots
        assert mask[k, :n].all() and not mask[k, n:].any()
        assert not feats[k, n:].any()
    assert mask.gather(1, chosen.unsqueeze(1)).all()


def test_demo_seeds_disjoint_from_eval_and_diag_seeds():
    demo = set(DEMO_TRAIN_SEEDS) | set(DEMO_VAL_SEEDS)
    other = set(VALIDATION_SEEDS) | set(dg.DIAG_HEURISTIC_SEEDS) | set(dg.DIAG_RANDOM_SEEDS) | set(range(100))
    assert not demo & other


def test_experiment_seed_ranges_are_disjoint():
    spec = importlib.util.spec_from_file_location("run_imitreg", ROOT / "scripts" / "run_imitation_experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    protected = (
        set(VALIDATION_SEEDS) | set(dg.DIAG_HEURISTIC_SEEDS) | set(dg.DIAG_RANDOM_SEEDS)
        | set(DEMO_TRAIN_SEEDS) | set(DEMO_VAL_SEEDS) | set(range(100))
    )
    ranges = []
    for seed, base in mod.TRAIN_SEED_BASE.items():
        r = set(range(base, base + 50_000))  # >> any plausible episode count
        assert not r & protected, f"train seeds for {seed} overlap a protected range"
        ranges.append(r)
    assert not ranges[0] & ranges[1], "training seeds overlap between experiment seeds"


# --------------------------------------------------------------------------- loss


def test_imitation_ce_equals_bc_listwise_ce(demos):
    agent = _agent()
    idx = np.arange(32)
    ref, _ = listwise_ce(agent, demos, idx)
    feats, mask, chosen = _padded_legal(demos, idx, agent.device)
    ours, st = imitation_ce(agent.online, feats, mask, chosen)
    assert float(ours) == pytest.approx(float(ref), rel=1e-4, abs=1e-5)
    assert 0.0 <= st["top1"] <= 1.0


def test_imitation_ce_equals_manual_unpadded_cross_entropy(demos):
    agent = _agent(1)
    idx = np.arange(10)
    feats, mask, chosen = _padded_legal(demos, idx, agent.device)
    ours, _ = imitation_ce(agent.online, feats, mask, chosen)
    manual = []
    for i in idx:
        block = torch.as_tensor(demos.block(int(demos.d["t_state"][i])))
        q = agent.online(block)
        manual.append(-torch.log_softmax(q, dim=0)[int(demos.d["t_chosen"][i])])
    assert float(ours) == pytest.approx(float(torch.stack(manual).mean()), rel=1e-5)


def test_padding_slots_cannot_affect_loss_or_gradients(demos):
    agent = _agent(2)
    idx = np.arange(12)
    feats, mask, chosen = _padded_legal(demos, idx, agent.device)

    def run(f):
        agent.online.zero_grad()
        loss, _ = imitation_ce(agent.online, f, mask, chosen)
        loss.backward()
        return float(loss), [p.grad.clone() for p in agent.online.parameters()]

    l1, g1 = run(feats)
    junk = feats.clone()
    junk[~mask] = 50.0 * torch.randn_like(junk[~mask])  # garbage in padded slots
    l2, g2 = run(junk)
    assert l1 == l2
    assert all(torch.equal(a, b) for a, b in zip(g1, g2))


def test_illegal_expert_label_is_rejected(demos):
    agent = _agent()
    feats, mask, chosen = _padded_legal(demos, np.arange(4), agent.device)
    bad = torch.full_like(chosen, feats.shape[1] - 1)
    mask2 = mask.clone()
    mask2[:, -1] = False
    with pytest.raises(ValueError):
        imitation_ce(agent.online, feats, mask2, bad)


def test_masked_logits_pad_value(demos):
    agent = _agent()
    feats, mask, _ = _padded_legal(demos, np.arange(6), agent.device)
    q = masked_logits(agent.online, feats, mask)
    assert (q[~mask] == -1e9).all() and (q[mask] > -1e8).all()


# ------------------------------------------------------------------ gradient flow


def test_gradient_only_through_online_network(demos):
    agent = _agent(3)
    _fill(agent, demos)
    agent.attach_imitation(ImitationRegularizer(demos, 1.0, batch_size=16, seed=1), grad_diag_every=1)
    target_before = {k: v.clone() for k, v in agent.target.state_dict().items()}
    assert all(p.grad is None for p in agent.target.parameters())
    agent.train_step()
    assert all(p.grad is None for p in agent.target.parameters())  # no grads reach target
    assert all(torch.equal(target_before[k], v) for k, v in agent.target.state_dict().items())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.online.parameters())
    st = agent.last_train_stats
    assert st["imit_loss"] > 0 and st["td_loss"] >= 0
    assert st["total_loss"] == pytest.approx(st["td_loss"] + 1.0 * st["imit_loss"], rel=1e-5)
    for k in ("gn_td", "gn_imit_unweighted", "grad_cos_td_imit"):
        assert np.isfinite(st[k])


def test_imitation_gradient_is_nonzero_and_td_gradient_untouched_by_lambda(demos):
    """Separate-loss grad norms (instrumentation) do not depend on lambda."""
    outs = []
    for lam in (0.0, 5.0):
        a = _agent(4)
        _fill(a, demos)
        a.attach_imitation(ImitationRegularizer(demos, lam, batch_size=16, seed=9), grad_diag_every=1)
        a.train_step()
        outs.append(a.last_train_stats)
    assert outs[0]["gn_td"] == pytest.approx(outs[1]["gn_td"], rel=1e-5)
    assert outs[0]["gn_imit_unweighted"] == pytest.approx(outs[1]["gn_imit_unweighted"], rel=1e-5)
    assert outs[1]["gn_imit_weighted"] == pytest.approx(5.0 * outs[1]["gn_imit_unweighted"], rel=1e-5)


def test_lambda_zero_is_identical_to_plain_dqn(demos):
    a, b = _agent(5), _agent(5)
    _fill(a, demos)
    _fill(b, demos)
    b.attach_imitation(ImitationRegularizer(demos, 0.0, batch_size=16, seed=3), grad_diag_every=2)
    for _ in range(6):
        a.train_step()
        b.train_step()
    for pa, pb in zip(a.online.parameters(), b.online.parameters()):
        assert torch.equal(pa, pb)
    assert "imit_loss" in b.last_train_stats and "imit_loss" not in a.last_train_stats


# ---------------------------------------------------- the imitation actually works


def _probe(agent: DQNAgent, demos: DemoSet) -> dict:
    return demo_metrics(agent.online, demos, np.arange(len(demos)))


def test_imitation_updates_raise_relative_expert_score(demos):
    """After imitation updates the expert action has a higher score relative to its alternatives."""
    agent = _agent(6)
    _fill(agent, demos)
    agent.attach_imitation(ImitationRegularizer(demos, 1.0, batch_size=32, seed=2))
    before = _probe(agent, demos)
    for _ in range(60):
        agent.train_step()
    after = _probe(agent, demos)
    assert after["ce"] < before["ce"]
    assert after["expert_prob"] > before["expert_prob"]
    assert after["rel_score"] > before["rel_score"]  # expert Q minus mean legal Q
    assert after["margin"] > before["margin"]        # expert Q minus best alternative
    assert after["top1"] >= before["top1"]


def test_imitation_regularisation_beats_td_only_at_preserving_preferences(demos):
    """Same init & data: lambda>0 ends with a lower expert CE than the lambda=0 control."""
    res = {}
    for lam in (0.0, 1.0):
        a = _agent(7)
        _fill(a, demos)
        a.attach_imitation(ImitationRegularizer(demos, lam, batch_size=32, seed=2))
        for _ in range(60):
            a.train_step()
        res[lam] = _probe(a, demos)
    assert res[1.0]["ce"] < res[0.0]["ce"]
    assert res[1.0]["rel_score"] > res[0.0]["rel_score"]


# ------------------------------------------------------------- checkpoint init


def test_init_from_checkpoint_preserves_weights_and_resets_counters(tmp_path):
    src = _agent(8)
    for p in src.online.parameters():  # make it distinguishable from a fresh init
        p.data.add_(0.123)
    src.target.load_state_dict(src.online.state_dict())
    src.total_steps, src.epsilon, src.best_eval_score = 777, 0.3, 123.0
    src.optimizer.zero_grad()
    sum(p.sum() for p in src.online.parameters()).backward()
    src.optimizer.step()
    ck = tmp_path / "bc.pt"
    src.save(ck, include_buffer=False)
    src_sd = {k: v.clone() for k, v in src.online.state_dict().items()}

    dst = _agent(99, epsilon_start=0.05, epsilon_end=0.05)
    dst.load_init_weights(ck)
    assert all(torch.equal(src_sd[k], v) for k, v in dst.online.state_dict().items())
    assert all(torch.equal(v, dst.target.state_dict()[k]) for k, v in src.target.state_dict().items())
    assert dst.total_steps == 0 and dst.grad_steps == 0 and len(dst.buffer) == 0
    assert dst.epsilon == 0.05
    assert dst.best_eval_score == -float("inf")
    assert len(dst.optimizer.state_dict()["state"]) > 0  # Adam moments carried over
    # identical Q-values on real features
    from tetris_rl.environment import TetrisEnv
    from tetris_rl.features import placement_feature_batch

    env = TetrisEnv(seed=3)
    env.reset(seed=3)
    f = placement_feature_batch(env.engine, env.legal_placements())
    assert np.array_equal(src.q_values(f), dst.q_values(f))


# ------------------------------------------------------------------ CLI / E2E


def test_cli_end_to_end_tiny_run(tmp_path, demos):
    save_demos(tmp_path / "train.npz", demos.d)
    save_demos(tmp_path / "val.npz", demos.d)
    src = _agent(10)
    ck = tmp_path / "bc.pt"
    src.save(ck, include_buffer=False)
    args = parse_args([
        "--preset", "improved", "--device", "cpu", "--no-tb", "--reward-mode", "delta",
        "--seed", "42", "--train-seed-base", "400000",
        "--min-buffer-size", "32", "--batch-size", "32", "--max-pieces", "30", "--eval-games", "2",
        "--epsilon-start", "0.05", "--epsilon-end", "0.05",
        "--max-steps", "60", "--eval-at-steps", "20,60", "--eval-at-start",
        "--init-from", str(ck), "--demo-path", str(tmp_path / "train.npz"),
        "--demo-val-path", str(tmp_path / "val.npz"), "--demo-prefill",
        "--imitation-weight", "0.5", "--imitation-batch-size", "16", "--grad-diag-every", "10",
        "--checkpoint-dir", str(tmp_path / "ckpt"), "--log-dir", str(tmp_path / "logs"),
    ])
    train(args)
    rows = list(csv.DictReader((tmp_path / "logs" / "eval_metrics.csv").open()))
    assert [int(r["env_steps"]) for r in rows] == [0, 20, 60]
    r = rows[-1]
    for k in ("w_td_loss", "w_imit_loss", "w_total_loss", "w_gn_td", "demo_val_top1", "demo_train_ce",
              "demo_val_margin", "hs_top1_agreement", "holes_created_per_move"):
        assert k in r and r[k] != ""
    assert float(r["w_imit_loss"]) > 0
    assert float(r["epsilon"]) == pytest.approx(0.05)
    for s in (20, 60):
        assert (tmp_path / "ckpt" / f"dqn_step{s}.pt").is_file()
    # training episodes used the fresh seed range
    eps = list(csv.DictReader((tmp_path / "logs" / "training_metrics.csv").open()))
    assert eps and int(eps[0]["episode"]) == 1
    import json
    summ = json.loads((tmp_path / "logs" / "run_summary.json").read_text())
    assert summ["rl_env_steps"] == 60 and summ["expert_demo_transitions_available"] == len(demos)
