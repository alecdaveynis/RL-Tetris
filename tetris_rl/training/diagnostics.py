"""
Gameplay and Q-value diagnostics for the extended reward experiment.

Everything here is read-only with respect to the agent: no RNG is consumed and
no weights are touched, so calling it mid-training does not change the
training trajectory.

Q magnitudes are NOT comparable across reward modes (different reward scales).
We therefore report ranking quality (scale-free) and calibration measures
(Q vs. the empirical discounted return in the arm's own reward units,
normalised by the return's std) separately from raw Q magnitudes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from tetris_rl.agents.heuristic_agent import DEFAULT_WEIGHTS, HeuristicAgent
from tetris_rl.agents.random_agent import RandomAgent
from tetris_rl.engine import BOARD_HEIGHT, BOARD_WIDTH, TetrisEngine
from tetris_rl.environment import TetrisEnv
from tetris_rl.features import board_metrics, placement_feature_batch
from tetris_rl.rewards import RewardConfig

# Seed ranges. Training env seeds are `seed + episode` (< ~1e3); test seeds are
# 0..99; validation is 100000..100049. Diagnostic states use their own ranges.
VALIDATION_SEEDS: tuple[int, ...] = tuple(range(100_000, 100_050))
DIAG_HEURISTIC_SEEDS: tuple[int, ...] = tuple(range(200_000, 200_010))
DIAG_RANDOM_SEEDS: tuple[int, ...] = tuple(range(210_000, 210_010))

_CELLS = float(BOARD_HEIGHT * BOARD_WIDTH)


def heuristic_scores_from_features(feats: np.ndarray) -> np.ndarray:
    """
    Recover the Dellacherie heuristic score of each candidate placement from
    its state-action feature row (exact: metrics are stored as int/200).

    Feature tail layout (see features.state_action_features):
        [agg_h/200, holes/200, bump/200, max_h/20, lines/4, d_holes, d_h, d_bump]
    """
    f = np.asarray(feats, dtype=np.float64)
    agg_h = np.rint(f[:, -8] * _CELLS)
    holes = np.rint(f[:, -7] * _CELLS)
    bump = np.rint(f[:, -6] * _CELLS)
    lines = np.rint(f[:, -4] * 4.0)
    w = DEFAULT_WEIGHTS
    return (
        w.lines_cleared * lines
        + w.aggregate_height * agg_h
        + w.holes * holes
        + w.bumpiness * bump
    )


def _best_set(scores: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isclose(scores, np.max(scores)))


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank); 1 = smallest."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation with tie handling; NaN if either side is constant."""
    if len(a) < 2:
        return float("nan")
    ra, rb = _rankdata(np.asarray(a, dtype=np.float64)), _rankdata(np.asarray(b, dtype=np.float64))
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------------------
# Fixed diagnostic states
# ---------------------------------------------------------------------------


@dataclass
class DiagnosticStates:
    heuristic: list[TetrisEngine]  # states visited by the heuristic agent
    random: list[TetrisEngine]  # states visited by a random agent (messy boards)


def build_diagnostic_states(
    heuristic_every: int = 6,
    heuristic_max_piece: int = 84,
    random_every: int = 2,
) -> DiagnosticStates:
    """
    Deterministically build fixed diagnostic states from play on diagnostic
    seeds (disjoint from train / validation / test seeds).
    """
    heur_states: list[TetrisEngine] = []
    for seed in DIAG_HEURISTIC_SEEDS:
        env = TetrisEnv(seed=seed)
        env.reset(seed=seed)
        agent = HeuristicAgent(seed=seed)
        t = 0
        while not env.engine.game_over and t <= heuristic_max_piece:
            if env._legal and t % heuristic_every == 0:
                heur_states.append(copy.deepcopy(env.engine))
            if not env._legal:
                break
            env.step(agent.select_action(env))
            t += 1

    rand_states: list[TetrisEngine] = []
    for seed in DIAG_RANDOM_SEEDS:
        env = TetrisEnv(seed=seed)
        env.reset(seed=seed)
        agent = RandomAgent(seed=seed)
        t = 0
        while not env.engine.game_over:
            if not env._legal:
                break
            if t % random_every == 0:
                rand_states.append(copy.deepcopy(env.engine))
            env.step(agent.select_action(env))
            t += 1
    return DiagnosticStates(heuristic=heur_states, random=rand_states)


def _state_batches(engines: Sequence[TetrisEngine]):
    for eng in engines:
        placements = eng.enumerate_placements()
        if not placements:
            continue
        yield placement_feature_batch(eng, placements)


def _ranking_stats(agent, engines: Sequence[TetrisEngine]) -> dict:
    q_max, q_mean, q_range, top1, rank_h, rho = [], [], [], [], [], []
    for feats in _state_batches(engines):
        q = agent.q_values(feats, agent.online).astype(np.float64)
        h = heuristic_scores_from_features(feats)
        best_h = _best_set(h)
        a = int(np.argmax(q))
        q_max.append(q.max())
        q_mean.append(q.mean())
        q_range.append(q.max() - q.min())
        top1.append(float(a in set(best_h.tolist())))
        # rank of the best heuristic action under Q (1 = Q also ranks it first);
        # ties in Q resolved optimistically via strict-greater count.
        rank_h.append(1.0 + float(np.sum(q > q[best_h].max())))
        r = spearman(q, h)
        if not np.isnan(r):
            rho.append(r)
    n = len(q_max)
    return {
        "n_states": n,
        "q_max_mean": float(np.mean(q_max)) if n else float("nan"),
        "q_max_global": float(np.max(q_max)) if n else float("nan"),
        "q_mean": float(np.mean(q_mean)) if n else float("nan"),
        "q_action_range": float(np.mean(q_range)) if n else float("nan"),
        "top1_agreement": float(np.mean(top1)) if n else float("nan"),
        "heuristic_rank": float(np.mean(rank_h)) if n else float("nan"),
        "spearman": float(np.mean(rho)) if rho else float("nan"),
    }


def probe_diagnostic_states(agent, states: DiagnosticStates) -> dict:
    """Q magnitude and ranking quality on fixed states (flat dict, prefixed keys)."""
    out: dict = {}
    for name, engines in (("hs", states.heuristic), ("rs", states.random)):
        for k, v in _ranking_stats(agent, engines).items():
            out[f"{name}_{k}"] = v
    return out


# ---------------------------------------------------------------------------
# Extended greedy evaluation
# ---------------------------------------------------------------------------


def evaluate_greedy_extended(
    agent,
    max_pieces: int,
    reward_config: RewardConfig,
    seeds: Optional[Sequence[int]] = None,
    gamma: float = 0.99,
) -> dict:
    """
    Greedy (eps=0) games on fixed validation seeds with per-move diagnostics.

    Gameplay is identical to ``train_dqn.evaluate_greedy`` (argmax over online
    Q of legal placements). Rewards are computed in the *arm's* reward config
    only for Q-vs-return calibration; they never affect play.
    """
    if seeds is None:
        seeds = VALIDATION_SEEDS
    scores, lines, pieces, topped = [], [], [], []
    holes_created, agg_h_delta, max_h_after, agree = [], [], [], []
    q_chosen_all, g_all = [], []
    game_final_holes, game_peak_h = [], []
    n_cal_games = 0

    for seed in seeds:
        env = TetrisEnv(seed=int(seed), reward_config=reward_config)
        _, info = env.reset(seed=int(seed))
        terminated = truncated = False
        g_q: list[float] = []
        g_r: list[float] = []
        peak_h = 0
        while not terminated and not truncated:
            if info["n_legal"] == 0:
                break
            if env.engine.pieces_placed >= max_pieces:
                truncated = True
                break
            placements = env.legal_placements()
            feats = placement_feature_batch(env.engine, placements)
            q = agent.q_values(feats, agent.online)
            a = int(np.argmax(q))
            h = heuristic_scores_from_features(feats)
            agree.append(float(a in set(_best_set(h).tolist())))
            before = board_metrics(env.engine.board)
            _, reward, terminated, trunc_env, info = env.step(a)
            after = board_metrics(env.engine.board)
            truncated = bool(trunc_env) or env.engine.pieces_placed >= max_pieces
            if not terminated:  # a top-out placement leaves no meaningful board delta
                holes_created.append(after["holes"] - before["holes"])
                agg_h_delta.append(after["aggregate_height"] - before["aggregate_height"])
                max_h_after.append(after["max_height"])
                peak_h = max(peak_h, after["max_height"])
            g_q.append(float(q[a]))
            g_r.append(float(reward))
        scores.append(env.engine.score)
        lines.append(env.engine.lines_cleared_total)
        pieces.append(env.engine.pieces_placed)
        topped.append(bool(env.engine.game_over) and not truncated)
        game_final_holes.append(board_metrics(env.engine.board)["holes"])
        game_peak_h.append(peak_h)

        # Discounted return-to-go in the arm's reward units. Only for games that
        # truly ended (a truncated game has an unknown bootstrap tail).
        if terminated:
            n_cal_games += 1
            g = 0.0
            rets = [0.0] * len(g_r)
            for t in range(len(g_r) - 1, -1, -1):
                g = g_r[t] + gamma * g
                rets[t] = g
            q_chosen_all.extend(g_q)
            g_all.extend(rets)

    def _m(x):
        return float(np.mean(x)) if len(x) else float("nan")

    scores_a = np.asarray(scores, dtype=np.float64)
    lines_a = np.asarray(lines, dtype=np.float64)
    pieces_a = np.asarray(pieces, dtype=np.float64)
    out = {
        "mean_score": float(np.mean(scores_a)),
        "median_score": float(np.median(scores_a)),
        "std_score": float(np.std(scores_a, ddof=1)) if len(scores_a) > 1 else 0.0,
        "mean_lines": float(np.mean(lines_a)),
        "median_lines": float(np.median(lines_a)),
        "mean_pieces": float(np.mean(pieces_a)),
        "median_pieces": float(np.median(pieces_a)),
        "top_out_rate": float(np.mean(topped)),
        "n_games": len(scores_a),
        "holes_created_per_move": _m(holes_created),
        "agg_height_delta_per_move": _m(agg_h_delta),
        "mean_max_height": _m(max_h_after),
        "mean_peak_height": _m(game_peak_h),
        "mean_final_holes": _m(game_final_holes),
        "onpolicy_heuristic_agreement": _m(agree),
        "n_cal_games": n_cal_games,
    }
    if len(g_all) > 2:
        qc = np.asarray(q_chosen_all)
        g = np.asarray(g_all)
        g_std = float(np.std(g))
        out.update(
            {
                "cal_q_mean": float(qc.mean()),
                "cal_g_mean": float(g.mean()),
                "cal_g_std": g_std,
                "cal_bias": float((qc - g).mean()),
                "cal_bias_norm": float((qc - g).mean() / g_std) if g_std > 1e-9 else float("nan"),
                "cal_corr": float(np.corrcoef(qc, g)[0, 1])
                if np.std(qc) > 1e-9 and g_std > 1e-9
                else float("nan"),
            }
        )
    else:
        out.update({k: float("nan") for k in (
            "cal_q_mean", "cal_g_mean", "cal_g_std", "cal_bias", "cal_bias_norm", "cal_corr")})
    return out


def evaluate_heuristic_reference(
    max_pieces: int, reward_config: RewardConfig, seeds: Optional[Sequence[int]] = None
) -> dict:
    """Same per-move statistics for the heuristic agent (no Q fields)."""
    if seeds is None:
        seeds = VALIDATION_SEEDS
    scores, lines, pieces, topped = [], [], [], []
    holes_created, agg_h_delta, max_h_after, peaks = [], [], [], []
    for seed in seeds:
        env = TetrisEnv(seed=int(seed), reward_config=reward_config)
        _, info = env.reset(seed=int(seed))
        agent = HeuristicAgent(seed=int(seed))
        terminated = truncated = False
        peak = 0
        while not terminated and not truncated:
            if info["n_legal"] == 0:
                break
            if env.engine.pieces_placed >= max_pieces:
                truncated = True
                break
            before = board_metrics(env.engine.board)
            _, _, terminated, trunc_env, info = env.step(agent.select_action(env))
            after = board_metrics(env.engine.board)
            truncated = bool(trunc_env) or env.engine.pieces_placed >= max_pieces
            if not terminated:
                holes_created.append(after["holes"] - before["holes"])
                agg_h_delta.append(after["aggregate_height"] - before["aggregate_height"])
                max_h_after.append(after["max_height"])
                peak = max(peak, after["max_height"])
        scores.append(env.engine.score)
        lines.append(env.engine.lines_cleared_total)
        pieces.append(env.engine.pieces_placed)
        topped.append(bool(env.engine.game_over) and not truncated)
        peaks.append(peak)
    return {
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "mean_lines": float(np.mean(lines)),
        "mean_pieces": float(np.mean(pieces)),
        "top_out_rate": float(np.mean(topped)),
        "holes_created_per_move": float(np.mean(holes_created)),
        "agg_height_delta_per_move": float(np.mean(agg_h_delta)),
        "mean_max_height": float(np.mean(max_h_after)),
        "mean_peak_height": float(np.mean(peaks)),
        "n_games": len(scores),
    }
