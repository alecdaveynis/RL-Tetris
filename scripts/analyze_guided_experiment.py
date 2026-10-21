#!/usr/bin/env python3
"""Analyse the heuristic-guided experiment (A_mix, B_bc) against the delta controls."""

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
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tetris_rl.agents.dqn_agent import DQNAgent  # noqa: E402
from tetris_rl.rewards import RewardConfig  # noqa: E402
from tetris_rl.training import demos as dm  # noqa: E402
from tetris_rl.training import diagnostics as dg  # noqa: E402

TAG = "exp_guided"
RES = ROOT / "results" / TAG
MOD = ROOT / "models" / TAG
CTRL_RES = ROOT / "results" / "exp_reward_extended" / "delta"
CTRL_ANALYSIS = json.loads((ROOT / "results" / "exp_reward_extended" / "analysis.json").read_text())
FROZEN = ROOT / "models" / "frozen" / "dqn_best_val1320.pt"
METHODS = ("B_bc", "A_mix", "control")
SEEDS = (42, 43)
COL = {"B_bc": "tab:green", "A_mix": "tab:orange", "control": "tab:blue"}
LS = {42: "-", 43: "--"}
NAME = {"B_bc": "B: demos+BC", "A_mix": "A: heuristic mix", "control": "control (delta)"}


def read(path: Path):
    return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(path.open())]


def col(rows, k):
    return np.array([r[k] for r in rows], float)


def eval_rows(m, s):
    p = (CTRL_RES / f"seed{s}" if m == "control" else RES / m / f"seed{s}") / "eval_metrics.csv"
    return read(p)


def per_game(agent):
    out = []
    for sd in dg.VALIDATION_SEEDS:
        out.append(dg.evaluate_greedy_extended(agent, 500, RewardConfig(), seeds=[sd])["mean_score"])
    return out


def boot_ci(diff, n=5000):
    rng = np.random.default_rng(0)
    means = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    summary = json.loads((RES / "run_summary.json").read_text())
    ev = {(m, s): eval_rows(m, s) for m in METHODS for s in SEEDS}
    info = {(m, s): json.loads((RES / m / f"seed{s}" / "run_summary.json").read_text())
            for m in ("B_bc", "A_mix") for s in SEEDS}
    val_demos = dm.DemoSet.load(ROOT / "results" / "exp_demos" / "heuristic_demos_val.npz")
    states = dg.build_diagnostic_states()

    # --- best / final checkpoint re-evaluation --------------------------------
    ck: dict = {}
    for m in ("B_bc", "A_mix"):
        for s in SEEDS:
            rows = ev[(m, s)]
            sc = col(rows, "mean_score")
            best_i = int(np.flatnonzero(sc == sc.max())[-1])
            for which, path, step in (
                ("best", MOD / m / f"seed{s}" / "dqn_best.pt", int(rows[best_i]["env_steps"])),
                ("final", MOD / m / f"seed{s}" / "dqn_final.pt", int(rows[-1]["env_steps"])),
            ):
                if which == "final" and best_i == len(rows) - 1:
                    ck[f"{m}/seed{s}/final"] = ck[f"{m}/seed{s}/best"]
                    continue
                a = DQNAgent.from_checkpoint(path, device="cpu")
                ext = dg.evaluate_greedy_extended(a, 500, RewardConfig())
                ck[f"{m}/seed{s}/{which}"] = {
                    "env_steps": step, **ext, **dg.probe_diagnostic_states(a, states),
                    "per_game": per_game(a), "logged": float(rows[best_i if which == "best" else -1]["mean_score"]),
                }
                print("evaluated", m, s, which, ext["mean_score"], flush=True)
    for s in SEEDS:  # controls from the earlier experiment
        for which in ("best", "final"):
            ck[f"control/seed{s}/{which}"] = CTRL_ANALYSIS["ckpt_eval"][f"delta/seed{s}/{which}"]

    # --- imitation retention: Q top-1 on held-out heuristic demos, per step checkpoint ------
    retention = {}
    for m in ("B_bc", "A_mix"):
        for s in SEEDS:
            r = {}
            paths = [(0, MOD / m / f"seed{s}" / "dqn_best.pt")] if m == "B_bc" else []
            paths += [(st, MOD / m / f"seed{s}" / f"dqn_step{st}.pt") for st in range(2500, 20001, 2500)]
            for st, p in paths:
                if st == 0:  # dqn_best may not be step 0 -> use only if best step is 0
                    if int(ev[(m, s)][int(np.flatnonzero(col(ev[(m, s)], 'mean_score') == col(ev[(m, s)], 'mean_score').max())[-1])]["env_steps"]) != 0:
                        continue
                a = DQNAgent.from_checkpoint(p, device="cpu")
                r[st] = dm.demo_top1(a, val_demos, n=2000)
            retention[f"{m}/seed{s}"] = r
    print("retention", retention)

    # --- replay composition at the end of training -------------------------------------------
    buffers = {}
    for m in ("B_bc", "A_mix"):
        for s in SEEDS:
            c = torch.load(MOD / m / f"seed{s}" / "dqn_final.pt", map_location="cpu", weights_only=False)
            st = c["buffer"]["storage"]
            rew = np.array([x["reward"] for x in st if x is not None])
            term = np.array([x["terminated"] for x in st if x is not None])
            n = len(rew)
            n_demo = 25000 if m == "B_bc" else 0
            buffers[f"{m}/seed{s}"] = {
                "size": n, "demo_slots": n_demo, "demo_fraction": n_demo / n,
                "terminal_frac_demo": float(term[:n_demo].mean()) if n_demo else None,
                "terminal_frac_rl": float(term[n_demo:].mean()),
                "mean_reward_demo": float(rew[:n_demo].mean()) if n_demo else None,
                "mean_reward_rl": float(rew[n_demo:].mean()),
                "min_reward_rl": float(rew[n_demo:].min()),
            }
            del c
    print("buffers", buffers)

    # --- paired stats vs control + collapse timing ---------------------------------------------
    paired = {}
    for m in ("B_bc", "A_mix"):
        for s in SEEDS:
            for which in ("best", "final"):
                x = np.array(ck[f"{m}/seed{s}/{which}"]["per_game"])
                y = np.array(ck[f"control/seed{s}/{which}"]["per_game"]["scores"])
                lo, hi = boot_ci(x - y)
                paired[f"{m}/seed{s}/{which}"] = {"mean": float(x.mean()), "control": float(y.mean()),
                                                  "diff": float((x - y).mean()), "ci95": [lo, hi]}
    degr = {}
    for (m, s), rows in ev.items():
        sc, steps = col(rows, "mean_score"), col(rows, "env_steps")
        peak_i = int(np.argmax(sc))
        first_drop = None
        run_best = -1
        for i, v in enumerate(sc):
            if run_best >= 100 and v < 0.6 * run_best and first_drop is None:
                first_drop = {"step": int(steps[i]), "from_best": float(run_best), "to": float(v)}
            run_best = max(run_best, v)
        degr[f"{m}/seed{s}"] = {"peak_step": int(steps[peak_i]), "peak": float(sc[peak_i]),
                                "final": float(sc[-1]), "retained_frac": float(sc[-1] / sc[peak_i]),
                                "first_drop_below_60pct_of_best": first_drop}

    out = {"ckpt": ck, "retention_top1_val_demos": retention, "buffers": buffers, "paired_vs_control": paired,
           "degradation": degr, "runs": {f"{m}/seed{s}": v for (m, s), v in info.items()},
           "total_wall_s": summary.get("total_wall_s"),
           "frozen_val": CTRL_ANALYSIS["frozen"]["mean_score"], "heuristic_val": CTRL_ANALYSIS["heuristic"]["mean_score"]}
    (RES / "analysis.json").write_text(json.dumps(out, indent=2, default=float))

    # ================================ charts =======================================================
    def plot(ax, key, yl, ref=None, logy=False, methods=METHODS):
        for m in methods:
            for s in SEEDS:
                r = ev[(m, s)]
                if key not in r[0]:
                    continue
                ax.plot(col(r, "env_steps"), col(r, key), LS[s], color=COL[m], marker="o", ms=4,
                        label=f"{NAME[m]} s{s}")
        if ref:
            for v, lab, c in ref:
                ax.axhline(v, color=c, ls=":", label=lab)
        ax.axvline(15000, color="k", ls=":", alpha=0.3)
        ax.set_xlabel("environment steps")
        ax.set_ylabel(yl)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)

    refs = [(CTRL_ANALYSIS["frozen"]["mean_score"], "frozen 1102", "gray"),
            (CTRL_ANALYSIS["heuristic"]["mean_score"], "heuristic 21.4k", "k")]
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    plot(ax[0], "mean_score", "val mean score", refs, logy=True)
    ax[0].set_title("Validation score vs env steps (log y)", fontsize=9)
    ax[0].legend(fontsize=6)
    plot(ax[1], "mean_score", "val mean score", refs[:1] + refs[1:])
    ax[1].set_title("Validation score (linear)", fontsize=9)
    plot(ax[2], "mean_lines", "val mean lines")
    ax[2].set_title("Mean lines cleared", fontsize=9)
    fig.tight_layout(); fig.savefig(RES / "val_score_lines.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))
    plot(ax[0], "mean_pieces", "mean pieces survived")
    ax[0].set_title("Pieces survived", fontsize=9)
    ax[0].legend(fontsize=6)
    plot(ax[1], "top_out_rate", "top-out rate")
    ax[1].set_title("Top-out rate", fontsize=9)
    plot(ax[2], "mix_p", "heuristic probability", methods=("A_mix", "B_bc"))
    plot(ax[2], "h_action_frac", "heuristic action fraction", methods=("A_mix",))
    ax[2].set_title("Heuristic share of training actions (A only >0)", fontsize=9)
    plot(ax[3], "epsilon", "epsilon")
    ax[3].set_title("Epsilon (identical)", fontsize=9)
    fig.tight_layout(); fig.savefig(RES / "pieces_topout_mix.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(2, 3, figsize=(18, 8.5))
    plot(ax[0, 0], "hs_q_max_mean", "mean max-Q (heuristic states)")
    ax[0, 0].set_title("Q magnitude (same reward scale across all arms)", fontsize=9)
    ax[0, 0].legend(fontsize=6)
    plot(ax[0, 1], "hs_q_max_global", "max Q over fixed states")
    ax[0, 1].set_title("Max Q", fontsize=9)
    plot(ax[0, 2], "cal_bias_norm", "(mean Q - G)/std(G)")
    ax[0, 2].set_title("Q overestimation vs realised return (normalised)", fontsize=9)
    plot(ax[1, 0], "hs_spearman", "Spearman(Q, heuristic score)")
    ax[1, 0].set_title("Q ranking, heuristic states", fontsize=9)
    plot(ax[1, 1], "hs_top1_agreement", "top-1 agreement")
    ax[1, 1].set_title("Top-1 agreement, heuristic states", fontsize=9)
    plot(ax[1, 2], "onpolicy_heuristic_agreement", "agreement on visited states")
    ax[1, 2].set_title("On-policy agreement with heuristic", fontsize=9)
    fig.tight_layout(); fig.savefig(RES / "q_ranking_calibration.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))
    plot(ax[0], "train_loss", "loss", logy=True)
    ax[0].set_title("Training loss", fontsize=9); ax[0].legend(fontsize=6)
    plot(ax[1], "train_td_error", "|TD|", logy=True); ax[1].set_title("TD error", fontsize=9)
    plot(ax[2], "grad_norm_mean", "pre-clip grad norm (clip=5)", logy=True); ax[2].set_title("Gradient norm", fontsize=9)
    plot(ax[3], "grad_clip_frac", "fraction clipped"); ax[3].set_title("Clip fraction", fontsize=9)
    fig.tight_layout(); fig.savefig(RES / "training_loss_grad.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))
    hr = CTRL_ANALYSIS["heuristic"]
    plot(ax[0], "holes_created_per_move", "holes / move", [(hr["holes_created_per_move"], "heuristic", "k")], logy=True)
    ax[0].set_title("Holes created per move (log)", fontsize=9); ax[0].legend(fontsize=6)
    plot(ax[1], "agg_height_delta_per_move", "agg. height change / move", [(hr["agg_height_delta_per_move"], "heuristic", "k")])
    ax[1].set_title("Stack-height change per move", fontsize=9)
    plot(ax[2], "mean_max_height", "mean max height", [(hr["mean_max_height"], "heuristic", "k")])
    ax[2].set_title("Max stack height", fontsize=9)
    for m in ("B_bc", "A_mix"):
        for s in SEEDS:
            r = retention[f"{m}/seed{s}"]
            ax[3].plot(list(r.keys()), list(r.values()), LS[s], color=COL[m], marker="o", ms=4, label=f"{NAME[m]} s{s}")
    ax[3].set_title("Top-1 agreement with heuristic on held-out DEMO states", fontsize=9)
    ax[3].set_xlabel("environment steps"); ax[3].grid(alpha=0.3); ax[3].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(RES / "holes_height_retention.png", dpi=130); plt.close(fig)

    # ================================ tables ========================================================
    L = ["### Validation score by env step (mean of 50 games, eps=0)\n",
         "| Step | eps | " + " | ".join(f"{NAME[m]} s{s}" for m in METHODS for s in SEEDS) + " |",
         "|---|---|" + "---|" * 6]
    steps_all = sorted({int(x) for rows in ev.values() for x in col(rows, "env_steps")})
    for st in steps_all:
        cells = []
        for m in METHODS:
            for s in SEEDS:
                r = [x for x in ev[(m, s)] if int(x["env_steps"]) == st]
                cells.append(f"{r[0]['mean_score']:.0f}" if r else "n/a")
        eps = [x["epsilon"] for x in ev[("A_mix", 42)] if int(x["env_steps"]) == st]
        L.append(f"| {st} | {eps[0]:.3f} | " + " | ".join(cells) + " |" if eps else f"| {st} | 1.000 | " + " | ".join(cells) + " |")
    L += ["", "### Best / final checkpoints (re-evaluated)\n",
          "| Run | Ckpt | Step | Mean | Median | Lines | Pieces | Top-out | Holes/move | dAggH/move | HS top-1 | HS rho | On-policy agree |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in METHODS:
        for s in SEEDS:
            for w in ("best", "final"):
                c = ck[f"{m}/seed{s}/{w}"]
                L.append(f"| {NAME[m]} s{s} | {w} | {c['env_steps']} | {c['mean_score']:.0f} | {c['median_score']:.0f} | "
                         f"{c['mean_lines']:.1f} | {c['mean_pieces']:.0f} | {c['top_out_rate']:.2f} | "
                         f"{c['holes_created_per_move']:.3f} | {c['agg_height_delta_per_move']:.2f} | "
                         f"{c['hs_top1_agreement']:.2f} | {c['hs_spearman']:.2f} | {c['onpolicy_heuristic_agreement']:.2f} |")
    L += ["", "### Degradation relative to each run's own peak\n",
          "| Run | Peak step | Peak | Final | Final / peak | First eval <60% of running best |", "|---|---|---|---|---|---|"]
    for k, v in degr.items():
        L.append(f"| {k} | {v['peak_step']} | {v['peak']:.0f} | {v['final']:.0f} | {v['retained_frac']:.2f} | {v['first_drop_below_60pct_of_best'] or 'none'} |")
    L += ["", "### Paired vs delta control (same seed, same 50 val games)\n",
          "| Run | Ckpt | Mean | Control | Diff | 95% CI (games) |", "|---|---|---|---|---|---|"]
    for k, v in paired.items():
        a, b, c = k.split("/")
        L.append(f"| {a} {b} | {c} | {v['mean']:.0f} | {v['control']:.0f} | {v['diff']:+.0f} | [{v['ci95'][0]:+.0f}, {v['ci95'][1]:+.0f}] |")
    L += ["", "### Replay composition at end of training\n",
          "| Run | Buffer | Demo share | Terminal frac (demo / RL) | Mean reward (demo / RL) | Min RL reward |", "|---|---|---|---|---|---|"]
    for k, v in buffers.items():
        L.append(f"| {k} | {v['size']} | {v['demo_fraction']:.2f} | {v['terminal_frac_demo']} / {v['terminal_frac_rl']:.4f} | "
                 f"{v['mean_reward_demo']} / {v['mean_reward_rl']:.3f} | {v['min_reward_rl']:.2f} |")
    L += ["", "### Runtime\n", "| Run | Env steps | Episodes | Wall (s) | Train (s) | Eval (s) |", "|---|---|---|---|---|---|"]
    for k, v in info.items():
        r = v
        L.append(f"| {k[0]} s{k[1]} | {r['env_steps']} | {r['episodes_started']} | {r['elapsed_s']:.0f} | {r['train_s']:.0f} | {r['eval_s']:.0f} |")
    (RES / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
