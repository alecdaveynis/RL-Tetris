#!/usr/bin/env python3
"""Phase-1 diagnostics: Q vs returns, reward mix, replay, DQN vs heuristic."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from tetris_rl.agents.dqn_agent import DQNAgent
from tetris_rl.agents.heuristic_agent import HeuristicAgent
from tetris_rl.environment import TetrisEnv
from tetris_rl.features import board_metrics, placement_feature_batch, state_action_features
from tetris_rl.rewards import RewardConfig, shaped_reward
from tetris_rl.training.train_dqn import VALIDATION_SEEDS

FROZEN = Path("models/frozen/dqn_best_val1320.pt")
GAMMA = 0.99
OUT = Path("results/diagnostics/phase1_instability.json")


def greedy_rollout_returns(agent: DQNAgent, seed: int, max_pieces: int = 400) -> dict:
    env = TetrisEnv(seed=seed)
    env.reset(seed=seed)
    placements = env.legal_placements()
    feats = placement_feature_batch(env.engine, placements)
    q = agent.q_values(feats, agent.online)
    a0 = int(np.argmax(q))
    q0 = float(q[a0])
    q_spread = float(np.max(q) - np.min(q))

    rewards = []
    terminated = truncated = False
    info = env._info()
    while not terminated and not truncated:
        if info["n_legal"] == 0:
            break
        if env.engine.pieces_placed >= max_pieces:
            truncated = True
            break
        action = agent.select_action(env, explore=False)
        _, r, terminated, trunc_env, info = env.step(action)
        truncated = bool(trunc_env) or env.engine.pieces_placed >= max_pieces
        rewards.append(float(r))

    g = 0.0
    for r in reversed(rewards):
        g = r + GAMMA * g
    # First-step empirical return is the full discounted sum (from t=0).
    return {
        "q0": q0,
        "q_spread": q_spread,
        "n_legal": len(q),
        "G0": g,
        "n_steps": len(rewards),
        "sum_r": float(sum(rewards)),
        "score": env.engine.score,
        "lines": env.engine.lines_cleared_total,
        "terminated": bool(terminated),
        "q_minus_G": q0 - g,
    }


def reward_components_for_game(agent, seed: int, max_pieces: int = 200) -> dict:
    env = TetrisEnv(seed=seed)
    env.reset(seed=seed)
    cfg = RewardConfig()
    parts = Counter()
    n = 0
    while not env.engine.game_over and env.engine.pieces_placed < max_pieces:
        if not env.legal_placements():
            break
        before = env.engine.copy_board()
        mb = board_metrics(before)
        a = agent.select_action(env, explore=False)
        _, r, term, trunc, info = env.step(a)
        ma = board_metrics(env.engine.board)
        hd = ma["holes"] - mb["holes"]
        he = ma["aggregate_height"] - mb["aggregate_height"]
        bd = ma["bumpiness"] - mb["bumpiness"]
        lc = int(info["lines_cleared"])
        if 1 <= lc <= 4:
            parts["lines"] += cfg.line_clear[lc - 1]
        parts["holes"] += -cfg.hole_penalty * hd
        parts["height"] += -cfg.height_penalty * he
        parts["bump"] += -cfg.bumpiness_penalty * bd
        if term:
            parts["game_over"] += -cfg.game_over_penalty
        else:
            parts["survival"] += cfg.survival_reward
        parts["total"] += float(r)
        n += 1
        if term or trunc:
            break
    return {k: float(v) for k, v in parts.items()} | {"n": n, "score": env.engine.score}


def replay_stats(agent: DQNAgent) -> dict:
    buf = agent.buffer
    n = len(buf)
    # Physical storage: filled slots
    rewards, terms, n_next, holes_feat = [], [], [], []
    # Feature index: occupancy 200, pieces 14, heights 10, then metrics
    # holes is at offset 200+7+7+10+1 = 225 in state-action features? 
    # height*width + 7 + 7 + width + 8; holes is metrics[1]
    occ = 200
    hole_idx = occ + 7 + 7 + 10 + 1
    ages = []
    for i in range(n):
        t = buf._storage[i]
        if t is None:
            continue
        rewards.append(t.reward)
        terms.append(float(t.terminated))
        n_next.append(0 if t.terminated else t.next_state_actions.shape[0])
        if t.state_action.shape[0] > hole_idx:
            holes_feat.append(float(t.state_action[hole_idx]))
        # circular age: newer writes have physical index closer to _idx-1
        dist = (buf._idx - 1 - i) % buf.capacity
        ages.append(dist)
    rewards = np.asarray(rewards)
    return {
        "size": n,
        "capacity": buf.capacity,
        "frac_full": n / buf.capacity,
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "frac_terminated": float(np.mean(terms)),
        "mean_n_next": float(np.mean(n_next)),
        "unique_reward_bins": int(len(np.unique(np.round(rewards, 2)))),
        "sa_std_mean": float(np.std(np.stack([buf._storage[i].state_action for i in range(n) if buf._storage[i] is not None], 0), axis=0).mean()),
    }


def dqn_vs_heuristic(dqn: DQNAgent, n_games: int = 15, max_pieces: int = 80) -> dict:
    heur = HeuristicAgent(seed=0)
    agree = 0
    total = 0
    dqn_more_holes = 0
    heur_more_holes = 0
    dqn_higher = 0
    dqn_missed_clear = 0
    both_clear = 0
    heur_only_clear = 0
    q_rank_of_heur = []

    for g in range(n_games):
        seed = 100_000 + g
        env_d = TetrisEnv(seed=seed)
        env_h = TetrisEnv(seed=seed)
        env_d.reset(seed=seed)
        env_h.reset(seed=seed)
        for _ in range(max_pieces):
            if env_d.engine.game_over or env_h.engine.game_over:
                break
            if not env_d.legal_placements() or not env_h.legal_placements():
                break
            # Compare on the DQN's current board (same if they agreed so far)
            same_board = np.array_equal(env_d.engine.board, env_h.engine.board) and (
                env_d.engine.current_piece == env_h.engine.current_piece
            )
            if same_board:
                legal = env_d.legal_placements()
                a_h = heur.select_action(env_d, explore=False)
                feats = placement_feature_batch(env_d.engine, legal)
                q = dqn.q_values(feats, dqn.online)
                a_d = int(np.argmax(q))
                total += 1
                if a_d == a_h:
                    agree += 1
                # rank of heuristic action among Q
                order = np.argsort(-q)
                rank = int(np.where(order == a_h)[0][0]) + 1
                q_rank_of_heur.append(rank)

                bd, ld, _ = env_d.engine.simulate_placement(legal[a_d])
                bh, lh, _ = env_d.engine.simulate_placement(legal[a_h])
                md, mh = board_metrics(bd), board_metrics(bh)
                if md["holes"] > mh["holes"]:
                    dqn_more_holes += 1
                elif mh["holes"] > md["holes"]:
                    heur_more_holes += 1
                if md["aggregate_height"] > mh["aggregate_height"] + 0.5:
                    dqn_higher += 1
                if lh > 0 and ld == 0:
                    dqn_missed_clear += 1
                    heur_only_clear += 1
                if lh > 0 and ld > 0:
                    both_clear += 1

            # Step each independently from their own state
            if not env_d.engine.game_over and env_d.legal_placements():
                env_d.step(dqn.select_action(env_d, explore=False))
            if not env_h.engine.game_over and env_h.legal_placements():
                env_h.step(heur.select_action(env_h, explore=False))

    return {
        "paired_decisions": total,
        "agreement": (agree / total) if total else 0.0,
        "mean_q_rank_of_heuristic": float(np.mean(q_rank_of_heur)) if q_rank_of_heur else None,
        "median_q_rank_of_heuristic": float(np.median(q_rank_of_heur)) if q_rank_of_heur else None,
        "dqn_more_holes": dqn_more_holes,
        "heur_more_holes": heur_more_holes,
        "dqn_higher_stack": dqn_higher,
        "dqn_missed_available_clear": dqn_missed_clear,
        "both_cleared": both_clear,
        "n_rank_samples": len(q_rank_of_heur),
    }


def main() -> None:
    agent = DQNAgent.from_checkpoint(FROZEN, device="cpu", explore=False)
    agent.config.gamma = GAMMA

    rolls = [greedy_rollout_returns(agent, seed=100_000 + i) for i in range(20)]
    q = np.array([r["q0"] for r in rolls])
    g = np.array([r["G0"] for r in rolls])
    q_vs_g = {
        "n": len(rolls),
        "q_mean": float(q.mean()),
        "G_mean": float(g.mean()),
        "q_minus_G_mean": float((q - g).mean()),
        "q_minus_G_median": float(np.median(q - g)),
        "corr_q_G": float(np.corrcoef(q, g)[0, 1]) if np.std(g) > 1e-6 else None,
        "frac_q_over_G": float(np.mean(q > g)),
        "mean_score": float(np.mean([r["score"] for r in rolls])),
        "mean_steps": float(np.mean([r["n_steps"] for r in rolls])),
        "mean_q_spread": float(np.mean([r["q_spread"] for r in rolls])),
        "examples": rolls[:5],
    }

    heur = HeuristicAgent(seed=0)
    dqn_rew = [reward_components_for_game(agent, 100_000 + i) for i in range(8)]
    heur_rew = [reward_components_for_game(heur, 100_000 + i, max_pieces=120) for i in range(8)]

    def avg_parts(rows):
        keys = set().union(*[r.keys() for r in rows]) - {"n", "score"}
        out = {k: float(np.mean([r.get(k, 0.0) for r in rows])) for k in sorted(keys)}
        out["mean_score"] = float(np.mean([r["score"] for r in rows]))
        out["mean_n"] = float(np.mean([r["n"] for r in rows]))
        return out

    report = {
        "q_vs_empirical_return": q_vs_g,
        "reward_components_dqn_games": avg_parts(dqn_rew),
        "reward_components_heuristic_games": avg_parts(heur_rew),
        "replay": replay_stats(agent),
        "dqn_vs_heuristic": dqn_vs_heuristic(agent),
        "gamma": GAMMA,
        "horizon_note": "placement-level; ~65 pieces at frozen policy; gamma**65=%.4f" % (GAMMA ** 65),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
