#!/usr/bin/env python3
"""Analyse the extended delta vs heuristic_aligned experiment: tables, charts, JSON."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tetris_rl.agents.dqn_agent import DQNAgent  # noqa: E402
from tetris_rl.rewards import HeuristicAlignedRewardConfig, RewardConfig  # noqa: E402
from tetris_rl.training import diagnostics as dg  # noqa: E402

TAG = "exp_reward_extended"
RES = ROOT / "results" / TAG
MOD = ROOT / "models" / TAG
FROZEN = ROOT / "models" / "frozen" / "dqn_best_val1320.pt"
MODES = ("delta", "heuristic_aligned")
SEEDS = (42, 43)
COL = {"delta": "tab:blue", "heuristic_aligned": "tab:red"}
LS = {42: "-", 43: "--"}
LABEL = {"delta": "delta", "heuristic_aligned": "heuristic-aligned"}
REWARD = {"delta": RewardConfig, "heuristic_aligned": HeuristicAlignedRewardConfig}


def read_eval(mode: str, seed: int) -> list[dict]:
    p = RES / mode / f"seed{seed}" / "eval_metrics.csv"
    rows = []
    for r in csv.DictReader(p.open()):
        rows.append({k: float(v) for k, v in r.items()})
    return rows


def read_train(mode: str, seed: int) -> list[dict]:
    p = RES / mode / f"seed{seed}" / "training_metrics.csv"
    return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(p.open())]


def col(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def per_game(agent, reward_cfg) -> dict:
    scores, lines, pieces = [], [], []
    for s in dg.VALIDATION_SEEDS:
        m = dg.evaluate_greedy_extended(agent, 500, reward_cfg, seeds=[s])
        scores.append(m["mean_score"])
        lines.append(m["mean_lines"])
        pieces.append(m["mean_pieces"])
    return {"scores": scores, "lines": lines, "pieces": pieces}


def boot_ci(diff: np.ndarray, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    summary = json.loads((RES / "run_summary.json").read_text())
    evals = {(m, s): read_eval(m, s) for m in MODES for s in SEEDS}
    trains = {(m, s): read_train(m, s) for m in MODES for s in SEEDS}
    runs_info = {}
    for m in MODES:
        for s in SEEDS:
            runs_info[(m, s)] = json.loads((RES / m / f"seed{s}" / "run_summary.json").read_text())

    # --- re-evaluate best / final checkpoints + references ---------------------
    diag_states = dg.build_diagnostic_states()
    ckpt_eval: dict = {}
    for m in MODES:
        for s in SEEDS:
            rows = evals[(m, s)]
            sc = col(rows, "mean_score")
            best_i = int(np.flatnonzero(sc == sc.max())[-1])  # trainer uses >= (last tie wins)
            for which, path in (("best", MOD / m / f"seed{s}" / "dqn_best.pt"),
                                ("final", MOD / m / f"seed{s}" / "dqn_final.pt")):
                agent = DQNAgent.from_checkpoint(path, device="cpu")
                ext = dg.evaluate_greedy_extended(agent, 500, REWARD[m]())
                probe = dg.probe_diagnostic_states(agent, diag_states)
                games = per_game(agent, REWARD[m]())
                step = int(rows[best_i]["env_steps"]) if which == "best" else int(rows[-1]["env_steps"])
                ckpt_eval[f"{m}/seed{s}/{which}"] = {
                    "env_steps": step, **ext, **probe, "per_game": games,
                    "train_time_logged_score": float(rows[best_i if which == "best" else -1]["mean_score"]),
                }
            print("evaluated", m, s, flush=True)

    frozen_agent = DQNAgent.from_checkpoint(FROZEN, device="cpu")
    frozen = {
        **dg.evaluate_greedy_extended(frozen_agent, 500, RewardConfig()),
        **dg.probe_diagnostic_states(frozen_agent, diag_states),
        "per_game": per_game(frozen_agent, RewardConfig()),
    }
    heur = dg.evaluate_heuristic_reference(500, RewardConfig())

    # --- paired statistics (same 50 val seeds) -----------------------------------
    paired = {}
    for which in ("best", "final"):
        for s in SEEDS:
            a = np.array(ckpt_eval[f"heuristic_aligned/seed{s}/{which}"]["per_game"]["scores"])
            b = np.array(ckpt_eval[f"delta/seed{s}/{which}"]["per_game"]["scores"])
            lo, hi = boot_ci(a - b)
            paired[f"{which}/seed{s}"] = {
                "ha_mean": float(a.mean()), "delta_mean": float(b.mean()),
                "diff": float((a - b).mean()), "ci95_over_games": [lo, hi],
            }

    # --- collapse detection (>40% drop of the val mean from running best, best>=100) -----
    collapses = {}
    for key, rows in evals.items():
        sc, steps = col(rows, "mean_score"), col(rows, "env_steps")
        run_best, ev = -1.0, []
        for i, v in enumerate(sc):
            if run_best >= 100 and v < 0.6 * run_best:
                ev.append({"step": int(steps[i]), "from_best": float(run_best), "to": float(v)})
            run_best = max(run_best, v)
        collapses[f"{key[0]}/seed{key[1]}"] = ev

    # tail-vs-late-window means (last 3 evals) to judge "still improving"
    trend = {}
    for key, rows in evals.items():
        sc, steps = col(rows, "mean_score"), col(rows, "env_steps")
        late = steps >= 15000
        slope = float(np.polyfit(steps[late], sc[late], 1)[0] * 1000) if late.sum() >= 3 else float("nan")
        trend[f"{key[0]}/seed{key[1]}"] = {
            "mean_score_steps>=15000": float(sc[late].mean()),
            "slope_per_1000_steps_after_15000": slope,
            "score_at_15000": float(sc[steps == 15000][0]) if (steps == 15000).any() else None,
            "score_at_20000": float(sc[-1]),
        }

    out = {
        "runs": {f"{m}/seed{s}": runs_info[(m, s)] for m in MODES for s in SEEDS},
        "ckpt_eval": ckpt_eval, "frozen": frozen, "heuristic": heur,
        "paired_ha_minus_delta": paired, "collapses": collapses, "trend": trend,
        "total_wall_s": summary.get("total_wall_s"),
    }
    (RES / "analysis.json").write_text(json.dumps(out, indent=2))

    # ============================ charts ======================================
    def plot_lines(ax, key, ylabel, ref=None, logy=False):
        for m in MODES:
            for s in SEEDS:
                r = evals[(m, s)]
                ax.plot(col(r, "env_steps"), col(r, key), LS[s], color=COL[m], marker="o", ms=4,
                        label=f"{LABEL[m]} s{s}")
        if ref is not None:
            ax.axhline(ref[0], color="gray", ls=":", label=ref[1])
        ax.axvline(15000, color="k", ls=":", alpha=0.3)
        ax.set_xlabel("environment steps")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    plot_lines(axes[0], "mean_score", "val mean score (50 games, eps=0)", (frozen["mean_score"], "frozen DQN (1102)"))
    axes[0].set_title("Validation score vs env steps (heuristic ~21k off-scale)")
    axes[0].legend(fontsize=7)
    plot_lines(axes[1], "median_score", "val median score", (frozen["median_score"], "frozen median"))
    axes[1].set_title("Validation median score")
    plot_lines(axes[2], "mean_lines", "val mean lines", (frozen["mean_lines"], "frozen DQN"))
    axes[2].set_title("Mean lines cleared vs env steps")
    fig.tight_layout()
    fig.savefig(RES / "val_score_lines.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    plot_lines(axes[0], "mean_pieces", "mean pieces survived", (frozen["mean_pieces"], "frozen DQN"))
    axes[0].set_title("Pieces survived")
    plot_lines(axes[1], "top_out_rate", "top-out rate")
    axes[1].set_title("Top-out rate (1.0 = every game ended by game over)")
    plot_lines(axes[2], "epsilon", "epsilon")
    axes[2].set_title("Epsilon vs env steps (identical in all runs)")
    fig.tight_layout()
    fig.savefig(RES / "pieces_topout_epsilon.png", dpi=130)
    plt.close(fig)

    # Q-values: raw magnitudes per arm (own reward scale) + scale-free ranking/calibration
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    for ax, key, ttl in (
        (axes[0, 0], "hs_q_max_mean", "Mean max-Q on fixed heuristic states (RAW, own reward units)"),
        (axes[0, 1], "hs_q_max_global", "Max Q over fixed states (RAW)"),
        (axes[0, 2], "hs_q_action_range", "Mean Q spread across legal actions (RAW)"),
    ):
        plot_lines(ax, key, key)
        ax.set_title(ttl, fontsize=9)
    axes[0, 0].legend(fontsize=7)
    plot_lines(axes[1, 0], "hs_spearman", "Spearman(Q, heuristic score)")
    axes[1, 0].set_title("Q ranking quality (scale-free), heuristic states", fontsize=9)
    plot_lines(axes[1, 1], "hs_top1_agreement", "top-1 agreement with heuristic")
    axes[1, 1].set_title("Action agreement w/ heuristic on fixed states", fontsize=9)
    plot_lines(axes[1, 2], "cal_bias_norm", "(mean Q - G) / std(G)")
    axes[1, 2].set_title("Q calibration vs realised discounted return (scale-normalised)", fontsize=9)
    fig.tight_layout()
    fig.savefig(RES / "q_value_evolution.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    plot_lines(axes[0], "rs_spearman", "Spearman(Q, heuristic score)")
    axes[0].set_title("Q ranking on random-play (messy) fixed states", fontsize=9)
    plot_lines(axes[1], "rs_top1_agreement", "top-1 agreement")
    axes[1].set_title("Heuristic agreement on random-play fixed states", fontsize=9)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RES / "q_ranking_random_states.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    plot_lines(axes[0], "train_loss", "mean Huber loss in window", logy=True)
    axes[0].set_title("Training loss (not comparable across reward scales)", fontsize=9)
    axes[0].legend(fontsize=7)
    plot_lines(axes[1], "train_td_error", "mean |TD error| in window", logy=True)
    axes[1].set_title("TD error", fontsize=9)
    plot_lines(axes[2], "grad_norm_mean", "mean pre-clip grad norm (clip=5)", logy=True)
    axes[2].set_title("Gradient norm", fontsize=9)
    fig.tight_layout()
    fig.savefig(RES / "training_loss_grad.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    plot_lines(axes[0], "holes_created_per_move", "holes created per move",
               (heur["holes_created_per_move"], "heuristic"))
    axes[0].set_title("Holes created per move (val games)", fontsize=9)
    axes[0].legend(fontsize=7)
    plot_lines(axes[1], "agg_height_delta_per_move", "aggregate-height change per move",
               (heur["agg_height_delta_per_move"], "heuristic"))
    axes[1].set_title("Stack-height change per move", fontsize=9)
    plot_lines(axes[2], "mean_max_height", "mean max column height", (heur["mean_max_height"], "heuristic"))
    axes[2].set_title("Max stack height during play", fontsize=9)
    plot_lines(axes[3], "onpolicy_heuristic_agreement", "on-policy heuristic agreement")
    axes[3].set_title("Agreement with heuristic on visited states", fontsize=9)
    fig.tight_layout()
    fig.savefig(RES / "holes_height.png", dpi=130)
    plt.close(fig)

    # seed differences
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    steps = col(evals[("delta", 42)], "env_steps")
    for s in SEEDS:
        d = col(evals[("heuristic_aligned", s)], "mean_score") - col(evals[("delta", s)], "mean_score")
        axes[0].plot(steps, d, LS[s], marker="o", color="k", label=f"HA - delta, seed {s}")
    axes[0].axhline(0, color="gray")
    axes[0].set_title("Paired score difference by seed (same val games)", fontsize=9)
    axes[0].set_xlabel("environment steps")
    axes[0].set_ylabel("val mean score difference")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    x = np.arange(2)
    w = 0.18
    for k, (m, s) in enumerate([(m, s) for m in MODES for s in SEEDS]):
        vals = [ckpt_eval[f"{m}/seed{s}/best"]["mean_score"], ckpt_eval[f"{m}/seed{s}/final"]["mean_score"]]
        axes[1].bar(x + (k - 1.5) * w, vals, w, color=COL[m], alpha=0.5 if s == 43 else 1.0,
                    label=f"{LABEL[m]} s{s}")
    axes[1].axhline(frozen["mean_score"], color="gray", ls=":", label="frozen 1102")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["best ckpt", "final ckpt"])
    axes[1].set_title("Best / final validation score per run", fontsize=9)
    axes[1].legend(fontsize=7)
    for k, (m, s) in enumerate([(m, s) for m in MODES for s in SEEDS]):
        g = np.array(ckpt_eval[f"{m}/seed{s}/final"]["per_game"]["scores"])
        axes[2].boxplot(g, positions=[k], widths=0.5)
    axes[2].set_xticks(range(4))
    axes[2].set_xticklabels([f"{LABEL[m][:5]}\ns{s}" for m in MODES for s in SEEDS])
    axes[2].set_title("Per-game val score, final checkpoints", fontsize=9)
    fig.tight_layout()
    fig.savefig(RES / "seed_differences.png", dpi=130)
    plt.close(fig)

    # train episode curves (smoothed) for context
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for m in MODES:
        for s in SEEDS:
            t = trains[(m, s)]
            st = col(t, "steps")
            for a_i, key in enumerate(("score", "pieces")):
                v = col(t, key)
                k = 25
                sm = np.convolve(v, np.ones(k) / k, mode="valid")
                ax[a_i].plot(st[k - 1:], sm, LS[s], color=COL[m], label=f"{LABEL[m]} s{s}")
    ax[0].set_title("Training-episode score (eps-greedy, 25-ep MA)", fontsize=9)
    ax[1].set_title("Training-episode pieces survived (25-ep MA)", fontsize=9)
    for a in ax:
        a.set_xlabel("environment steps")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RES / "train_episode_curves.png", dpi=130)
    plt.close(fig)

    # ============================ tables (markdown) ============================
    L = []
    L.append("### Best / final validation checkpoints (re-evaluated from the saved files, 50 val seeds, eps=0)\n")
    L.append("| Run | Ckpt | Env step | Mean score | Median | Mean lines | Mean pieces | Top-out | Holes/move | dAggH/move | HS top-1 agree | HS Spearman |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in MODES:
        for s in SEEDS:
            for w in ("best", "final"):
                c = ckpt_eval[f"{m}/seed{s}/{w}"]
                L.append(f"| {LABEL[m]} s{s} | {w} | {c['env_steps']} | {c['mean_score']:.0f} | {c['median_score']:.0f} | "
                         f"{c['mean_lines']:.2f} | {c['mean_pieces']:.1f} | {c['top_out_rate']:.2f} | "
                         f"{c['holes_created_per_move']:.2f} | {c['agg_height_delta_per_move']:.2f} | "
                         f"{c['hs_top1_agreement']:.2f} | {c['hs_spearman']:.2f} |")
    L.append(f"| frozen DQN (ref) | best | 30k+ | {frozen['mean_score']:.0f} | {frozen['median_score']:.0f} | "
             f"{frozen['mean_lines']:.2f} | {frozen['mean_pieces']:.1f} | {frozen['top_out_rate']:.2f} | "
             f"{frozen['holes_created_per_move']:.2f} | {frozen['agg_height_delta_per_move']:.2f} | "
             f"{frozen['hs_top1_agreement']:.2f} | {frozen['hs_spearman']:.2f} |")
    L.append(f"| heuristic (ref) | - | - | {heur['mean_score']:.0f} | {heur['median_score']:.0f} | {heur['mean_lines']:.2f} | "
             f"{heur['mean_pieces']:.1f} | {heur['top_out_rate']:.2f} | {heur['holes_created_per_move']:.2f} | "
             f"{heur['agg_height_delta_per_move']:.2f} | 1.00 | 1.00 |")

    L.append("\n### Mean by reward mode (2 seeds)\n")
    L.append("| Mode | Mean best val | Mean final val | Mean best lines | Mean final lines | Mean final pieces | Mean final HS agree | Mean final HS rho |")
    L.append("|---|---|---|---|---|---|---|---|")
    for m in MODES:
        def avg(w, k):
            return np.mean([ckpt_eval[f"{m}/seed{s}/{w}"][k] for s in SEEDS])
        L.append(f"| {LABEL[m]} | {avg('best','mean_score'):.0f} | {avg('final','mean_score'):.0f} | "
                 f"{avg('best','mean_lines'):.2f} | {avg('final','mean_lines'):.2f} | {avg('final','mean_pieces'):.1f} | "
                 f"{avg('final','hs_top1_agreement'):.2f} | {avg('final','hs_spearman'):.2f} |")

    L.append("\n### Paired HA - delta (same 50 val games per seed pair)\n")
    L.append("| Checkpoint | Seed | HA mean | delta mean | diff | 95% bootstrap CI over games |")
    L.append("|---|---|---|---|---|---|")
    for k, v in paired.items():
        w, s = k.split("/")
        L.append(f"| {w} | {s[4:]} | {v['ha_mean']:.0f} | {v['delta_mean']:.0f} | {v['diff']:+.0f} | "
                 f"[{v['ci95_over_games'][0]:+.0f}, {v['ci95_over_games'][1]:+.0f}] |")

    L.append("\n### Validation score curve (mean of 50 games) by env step\n")
    L.append("| Step | eps | " + " | ".join(f"{LABEL[m]} s{s}" for m in MODES for s in SEEDS) + " |")
    L.append("|---|---|" + "---|" * 4)
    for i, st in enumerate(steps):
        L.append(f"| {int(st)} | {evals[('delta',42)][i]['epsilon']:.3f} | " +
                 " | ".join(f"{evals[(m,s)][i]['mean_score']:.0f}" for m in MODES for s in SEEDS) + " |")

    L.append("\n### Mean lines by env step\n")
    L.append("| Step | " + " | ".join(f"{LABEL[m]} s{s}" for m in MODES for s in SEEDS) + " |")
    L.append("|---|" + "---|" * 4)
    for i, st in enumerate(steps):
        L.append(f"| {int(st)} | " + " | ".join(f"{evals[(m,s)][i]['mean_lines']:.2f}" for m in MODES for s in SEEDS) + " |")

    L.append("\n### Runtime / budget\n")
    L.append("| Run | Env steps | Episodes started | Final eps | Grad steps | Wall (s) | Train (s) | Eval+diag (s) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for m in MODES:
        for s in SEEDS:
            r = runs_info[(m, s)]
            L.append(f"| {LABEL[m]} s{s} | {r['env_steps']} | {r['episodes_started']} | {r['final_epsilon']:.3f} | "
                     f"{r['grad_steps']} | {r['elapsed_s']:.0f} | {r['train_s']:.0f} | {r['eval_s']:.0f} |")

    L.append("\n### Late-training trend and collapses\n")
    L.append("| Run | val @15k | val @20k | mean over evals >=15k | slope /1k steps after 15k | collapse events |")
    L.append("|---|---|---|---|---|---|")
    for k, v in trend.items():
        L.append(f"| {k} | {v['score_at_15000']:.0f} | {v['score_at_20000']:.0f} | {v['mean_score_steps>=15000']:.0f} | "
                 f"{v['slope_per_1000_steps_after_15000']:+.0f} | {collapses[k] or 'none'} |")

    L.append("\n### Q diagnostics at final checkpoint (raw magnitudes are NOT comparable across modes)\n")
    L.append("| Run | Q max-mean (HS) | Q max global | Q action range | cal Q mean | cal G mean | cal (Q-G)/std(G) | corr(Q,G) | HS rho | RS rho | HS top1 | RS top1 | HS Q-rank of heuristic move |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in MODES:
        for s in SEEDS:
            c = ckpt_eval[f"{m}/seed{s}/final"]
            L.append(f"| {LABEL[m]} s{s} | {c['hs_q_max_mean']:.2f} | {c['hs_q_max_global']:.2f} | {c['hs_q_action_range']:.2f} | "
                     f"{c['cal_q_mean']:.2f} | {c['cal_g_mean']:.2f} | {c['cal_bias_norm']:.2f} | {c['cal_corr']:.2f} | "
                     f"{c['hs_spearman']:.2f} | {c['rs_spearman']:.2f} | {c['hs_top1_agreement']:.2f} | "
                     f"{c['rs_top1_agreement']:.2f} | {c['hs_heuristic_rank']:.2f} |")
    c = frozen
    L.append(f"| frozen (delta scale; calib. under delta reward) | {c['hs_q_max_mean']:.2f} | {c['hs_q_max_global']:.2f} | "
             f"{c['hs_q_action_range']:.2f} | {c['cal_q_mean']:.2f} | {c['cal_g_mean']:.2f} | {c['cal_bias_norm']:.2f} | "
             f"{c['cal_corr']:.2f} | {c['hs_spearman']:.2f} | {c['rs_spearman']:.2f} | {c['hs_top1_agreement']:.2f} | "
             f"{c['rs_top1_agreement']:.2f} | {c['hs_heuristic_rank']:.2f} |")
    (RES / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
