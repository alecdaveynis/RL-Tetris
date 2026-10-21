#!/usr/bin/env python3
"""Inspect Q-values, TD errors, grads, target lag, and replay stats for a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tetris_rl.agents.dqn_agent import DQNAgent
from tetris_rl.environment import TetrisEnv
from tetris_rl.training.train_dqn import VALIDATION_SEEDS, evaluate_greedy


def param_l2(a: nn.Module, b: nn.Module) -> float:
    tot = 0.0
    n = 0
    for p, q in zip(a.parameters(), b.parameters()):
        tot += float((p.detach() - q.detach()).pow(2).sum().cpu())
        n += p.numel()
    return (tot / max(n, 1)) ** 0.5


def grad_norm(agent: DQNAgent, batch) -> float:
    loss = agent._compute_loss(batch)
    agent.optimizer.zero_grad()
    loss.backward()
    total = 0.0
    for p in agent.online.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().cpu())
    agent.optimizer.zero_grad()
    return total ** 0.5


def replay_stats(agent: DQNAgent, n: int = 512) -> dict:
    buf = agent.buffer
    n = min(n, len(buf))
    batch = buf.sample(n)
    rewards = np.array([t.reward for t in batch], dtype=np.float64)
    term = np.array([t.terminated for t in batch], dtype=np.float64)
    n_next = np.array(
        [0 if t.terminated else t.next_state_actions.shape[0] for t in batch],
        dtype=np.float64,
    )
    return {
        "n": n,
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
        "frac_terminated": float(term.mean()),
        "mean_n_next_actions": float(n_next.mean()),
    }


def q_and_td(agent: DQNAgent, batch) -> dict:
    sa = torch.as_tensor(
        np.stack([t.state_action for t in batch], axis=0),
        dtype=torch.float32,
        device=agent.device,
    )
    agent.online.eval()
    agent.target.eval()
    with torch.no_grad():
        q = agent.online(sa)
        q_tgt_sa = agent.target(sa)
        targets = agent.bellman_targets(batch, vectorized=True)
        td = targets - q
        next_q = agent._next_q_max_vectorized(*agent._pad_next_actions(batch))
    return {
        "q_mean": float(q.mean().cpu()),
        "q_std": float(q.std(unbiased=False).cpu()),
        "q_min": float(q.min().cpu()),
        "q_max": float(q.max().cpu()),
        "q_target_sa_mean": float(q_tgt_sa.mean().cpu()),
        "bootstrap_next_q_mean": float(next_q.mean().cpu()),
        "bootstrap_next_q_max": float(next_q.max().cpu()),
        "target_mean": float(targets.mean().cpu()),
        "td_mean": float(td.mean().cpu()),
        "td_abs_mean": float(td.abs().mean().cpu()),
        "td_std": float(td.std(unbiased=False).cpu()),
        "overestimate_frac": float((q > targets).float().mean().cpu()),
        "huber": float(torch.nn.functional.smooth_l1_loss(q, targets).cpu()),
    }


def greedy_q_spread(agent: DQNAgent, seed: int = 100_000) -> dict:
    """Q spread across legal actions at a validation start state."""
    env = TetrisEnv(seed=seed)
    env.reset(seed=seed)
    from tetris_rl.features import placement_feature_batch

    feats = placement_feature_batch(env.engine, env.legal_placements())
    q = agent.q_values(feats, agent.online)
    return {
        "n_legal": int(len(q)),
        "q_argmax": float(np.max(q)),
        "q_mean_legal": float(np.mean(q)),
        "q_spread": float(np.max(q) - np.min(q)),
        "q_std_legal": float(np.std(q)),
    }


def diagnose(path: str, device: str, val_games: int) -> dict:
    agent = DQNAgent.from_checkpoint(path, device=device, explore=True)
    opt_lrs = [g["lr"] for g in agent.optimizer.param_groups]
    n_opt_state = len(agent.optimizer.state)
    batch = agent.buffer.sample(min(64, len(agent.buffer)))
    out = {
        "path": path,
        "device": str(agent.device),
        "steps": agent.total_steps,
        "episodes_trained": agent.episodes_trained,
        "grad_steps": agent.grad_steps,
        "epsilon": agent.epsilon,
        "best_eval_score": agent.best_eval_score,
        "buffer_size": len(agent.buffer),
        "cfg_lr": agent.config.lr,
        "optimizer_lrs": opt_lrs,
        "optimizer_state_tensors": n_opt_state,
        "target_update_freq": agent.config.target_update_freq,
        "grad_steps_mod_target": agent.grad_steps % agent.config.target_update_freq,
        "online_target_rms_diff": param_l2(agent.online, agent.target),
        "replay": replay_stats(agent),
        "td": q_and_td(agent, batch),
        "grad_norm_unclipped": grad_norm(agent, batch),
        "grad_clip": agent.config.grad_clip,
        "start_state_q": greedy_q_spread(agent),
    }
    if val_games > 0:
        metrics = evaluate_greedy(agent, max_pieces=400, n_games=val_games)
        out["val"] = metrics
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", nargs="+", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--val-games", type=int, default=5)
    p.add_argument("--out", default="results/diagnostics/instability.json")
    args = p.parse_args()
    reports = [diagnose(path, args.device, args.val_games) for path in args.paths]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
